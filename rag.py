"""
rag.py - Step 3 of the CSL448 Study Assistant (RAG pipeline)

Retrieval-Augmented Generation (Unit 5):
  1. retrieve the chunks that best match the question (retriever.py)
  2. if even the best chunk is a weak match, answer "Not found in the notes"
     straight away, without calling Gemini (saves free-tier requests)
  3. otherwise build a prompt that holds the chunks as numbered context
     (prompt engineering) and ask Gemini to answer ONLY from that context
  4. return the answer together with its sources (file + page)

Needs the GEMINI_API_KEY environment variable (the key is never printed).

Try it from the project folder:
  venv\\Scripts\\python.exe rag.py "What is Naive Bayes?"
  venv\\Scripts\\python.exe rag.py "What is Naive Bayes?" --mode bm25 --k 5 --show-prompt
"""

import argparse
import os
import re
import sys
import time
from functools import lru_cache

from google import genai
from google.genai import errors

from retriever import DEFAULT_TOP_K, SEARCH_MODES, get_retriever

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
# Change the model in this ONE place. Backup if this one is unavailable or
# out of quota: "gemma-4-31b-it" (the prompt below has no system instruction,
# so it works with Gemma models too).
GEMINI_MODEL = "gemini-3.5-flash-lite"

NOT_FOUND = "Not found in the notes"

# If the best chunk's cosine similarity is below this, the notes almost
# certainly do not cover the question. Measured on our notes with retriever.py:
# on-topic questions scored about 0.35 - 0.83, off-topic ones 0.10 - 0.25.
SIMILARITY_THRESHOLD = 0.30

# Free tier allows 15 requests per minute. On a rate-limit (429) or
# "model overloaded" (503) error, wait and try again a few times.
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 10


# ----------------------------------------------------------------------------
# 1. Prompt (Unit 5: prompt engineering)
# ----------------------------------------------------------------------------
def build_prompt(question, chunks):
    """
    Put the retrieved chunks into a prompt as numbered, labelled context.

    The rules tell the model to stay inside the context and to reply with the
    exact NOT_FOUND sentence otherwise, which stops it from hallucinating.
    It also has to name the context blocks it used, so the app can show only
    the pages the answer really came from.
    """
    context_blocks = []
    for number, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[{number}] {chunk['source']}, page {chunk['page']}\n{chunk['text']}")
    context = "\n\n".join(context_blocks)

    return f"""You are the CSL448 Study Assistant, a tutor for a course on Computational Linguistics and Natural Language Processing.

Answer the student's question using ONLY the context below. The context comes from the course lecture slides.

Rules:
- Answer ONLY from the given context. Do not use outside knowledge, even if you know the answer.
- If the answer is not in the context, reply exactly: {NOT_FOUND}
- Keep the answer short and clear: a few sentences or bullet points.
- The slides may contain broken formulas or short bullet points; explain them in plain words.
- End your answer with one last line naming the context blocks you used, exactly like this: Sources: [1], [3]

Context:
{context}

Question: {question}

Answer:"""


# ----------------------------------------------------------------------------
# 2. Gemini call
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def get_client():
    """Create the Gemini client once, reading the key from the environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. In PowerShell run:  setx GEMINI_API_KEY \"your-key\"  "
            "then close and reopen the terminal.")
    return genai.Client(api_key=api_key)


def generate_answer(prompt):
    """Send the prompt to Gemini and return the answer text, retrying on rate limits."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = get_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return (response.text or "").strip()
        except errors.APIError as error:
            # 429 = too many requests (rate limit), 503 = model overloaded.
            if error.code in (429, 503) and attempt < MAX_RETRIES:
                print(f"  Gemini busy or rate-limited (error {error.code}). "
                      f"Waiting {RETRY_WAIT_SECONDS} s, then retry {attempt}/{MAX_RETRIES - 1} ...")
                time.sleep(RETRY_WAIT_SECONDS)
            else:
                raise


# ----------------------------------------------------------------------------
# 3. The full RAG pipeline
# ----------------------------------------------------------------------------
# The last line of an answer, e.g. "Sources: [1], [3]" (also "**Sources:** [1] and [2]").
SOURCES_LINE = re.compile(r"\s*\**\s*Sources?\s*\**\s*:\s*\**\s*((?:\[?\d+\]?[\s,]*(?:and\s+)?)+)\**\s*$",
                          re.IGNORECASE)


def split_cited_chunks(answer, chunks):
    """
    Remove the final "Sources: [1], [3]" line from the answer and return
    (answer without that line, the chunks it names). If the model left the
    line out, every retrieved chunk counts as a source.
    """
    match = SOURCES_LINE.search(answer)
    if not match:
        return answer, chunks
    numbers = {int(n) for n in re.findall(r"\d+", match.group(1))}
    cited = [chunk for number, chunk in enumerate(chunks, start=1) if number in numbers]
    return answer[:match.start()].rstrip(), cited or chunks


def unique_sources(chunks):
    """List each (file, page) once, in ranking order."""
    sources = []
    for chunk in chunks:
        source = {"source": chunk["source"], "page": chunk["page"]}
        if source not in sources:
            sources.append(source)
    return sources


def answer_question(question, mode="hybrid", k=DEFAULT_TOP_K, retriever=None):
    """
    Answer a question from the notes.

    Returns a dict:
      answer          - the answer text, or exactly NOT_FOUND
      found           - False when the answer is NOT_FOUND
      reason          - "answered", "low_similarity" (Gemini not called)
                        or "not_in_context" (Gemini said NOT_FOUND)
      sources         - [{"source": file, "page": n}, ...] the answer was taken from
      chunks          - the retrieved chunks (for the Sources panel)
      cited_ids       - ids of the chunks the answer names as its sources
      best_similarity - cosine similarity of the closest chunk
      prompt          - the prompt sent to Gemini (None if not sent)

    retriever lets evaluate.py pass an index built with another chunk size.
    """
    retriever = retriever or get_retriever()
    chunks = retriever.search(question, mode, k)
    best_similarity = max((chunk["similarity"] for chunk in chunks), default=0.0)

    result = {
        "answer": NOT_FOUND,
        "found": False,
        "reason": "low_similarity",
        "sources": [],
        "chunks": chunks,
        "cited_ids": [],
        "best_similarity": best_similarity,
        "prompt": None,
    }

    # Weak match -> do not spend a Gemini request.
    if best_similarity < SIMILARITY_THRESHOLD:
        return result

    prompt = build_prompt(question, chunks)
    answer = generate_answer(prompt)
    result["prompt"] = prompt

    # The model may add a full stop or extra words; treat any of those as NOT_FOUND.
    if not answer or NOT_FOUND.lower() in answer.lower():
        result["reason"] = "not_in_context"
        return result

    answer, cited = split_cited_chunks(answer, chunks)
    result.update(answer=answer, found=True, reason="answered", sources=unique_sources(cited),
                  cited_ids=[chunk["id"] for chunk in cited])
    return result


# ----------------------------------------------------------------------------
# Command-line demo
# ----------------------------------------------------------------------------
def main():
    # Slides contain symbols the Windows console cannot print; show "?" instead of crashing.
    sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Ask the CSL448 Study Assistant a question.")
    parser.add_argument("question", help="the question, in quotes")
    parser.add_argument("--mode", choices=SEARCH_MODES, default="hybrid", help="retriever to use")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="number of chunks to retrieve")
    parser.add_argument("--show-prompt", action="store_true", help="print the prompt sent to Gemini")
    args = parser.parse_args()

    result = answer_question(args.question, args.mode, args.k)

    if args.show_prompt and result["prompt"]:
        print("=" * 30 + " PROMPT " + "=" * 30)
        print(result["prompt"])
        print("=" * 68 + "\n")

    print(f"Question : {args.question}")
    print(f"Retriever: {args.mode}, top {args.k}  |  model: {GEMINI_MODEL}")
    print(f"Best similarity: {result['best_similarity']:.3f} (threshold {SIMILARITY_THRESHOLD})")
    if result["reason"] == "low_similarity":
        print("  -> below threshold, Gemini was NOT called")
    print(f"\nAnswer:\n{result['answer']}")

    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(f"  - {source['source']}, page {source['page']}")


if __name__ == "__main__":
    main()
