"""
rag.py - Step 3 of the CSL448 Study Assistant (RAG pipeline)

Retrieval-Augmented Generation (Unit 5):
  1. retrieve the chunks that best match the question (retriever.py)
  2. if even the best chunk is a weak match, answer "Not found in the notes"
     straight away, without calling Gemini (saves free-tier requests)
  3. otherwise build a prompt that holds the chunks as numbered context
     (prompt engineering) and ask Gemini to answer ONLY from that context
  4. return the answer together with its sources (file + page)

Extras used by the app:
  - follow-up questions: the last turns of the chat go into the prompt, and a
    vague follow-up ("explain that simply") is searched together with the
    previous question
  - streaming: the answer can arrive piece by piece (generate_answer_stream)
  - plain_answer(): Gemini WITHOUT the notes, to show what RAG prevents
  - confidence_level(): High / Medium / Low from the retrieval similarity

Needs the GEMINI_API_KEY environment variable (the key is never printed).

Try it from the project folder:
  venv\\Scripts\\python.exe rag.py "What is ocean acidification?"
  venv\\Scripts\\python.exe rag.py "Why is CNG called a green fuel?" --mode bm25 --k 3 --show-prompt
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
# certainly do not cover the question. Measured on the EVS test questions
# (1000-character chunks, top 5): questions in the notes scored 0.26 - 0.80,
# unrelated ones (NLP, cricket) 0.00 - 0.25. Environmental topics the notes
# don't cover (Kyoto, acid rain) score ~0.47; the prompt rule refuses those.
SIMILARITY_THRESHOLD = 0.20

# Confidence shown with each answer, from the best similarity. Most questions
# that the notes answer well score above 0.5 (see results/answers_detail.csv).
CONFIDENCE_HIGH = 0.50
CONFIDENCE_MEDIUM = 0.35

# A follow-up whose own best match is below this is searched again together
# with the previous question ("explain that more simply" matches nothing alone).
FOLLOW_UP_SIMILARITY = 0.40

# How many earlier question/answer pairs go into the prompt for follow-ups.
HISTORY_TURNS = 2

# Free tier allows 15 requests per minute. On a rate-limit (429) or
# "model overloaded" (503) error, wait and try again a few times.
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 10


# ----------------------------------------------------------------------------
# 1. Prompt (Unit 5: prompt engineering)
# ----------------------------------------------------------------------------
def format_history(history):
    """The last HISTORY_TURNS question/answer pairs as plain text for the prompt."""
    lines = []
    for turn in history[-HISTORY_TURNS:]:
        answer = turn["answer"]
        if len(answer) > 600:                 # keep the prompt short
            answer = answer[:600] + " ..."
        lines.append(f"Student: {turn['question']}\nAssistant: {answer}")
    return "\n\n".join(lines)


def build_prompt(question, chunks, history=None):
    """
    Put the retrieved chunks into a prompt as numbered, labelled context.

    The rules tell the model to stay inside the context and to reply with the
    exact NOT_FOUND sentence otherwise, which stops it from hallucinating.
    It also has to name the context blocks it used, so the app can show only
    the pages the answer really came from. Earlier turns of the chat are
    included only so that follow-ups like "explain that simply" make sense.
    """
    context_blocks = []
    for number, chunk in enumerate(chunks, start=1):
        where = f"page {chunk['page']}" if chunk["page"] else "overview"
        context_blocks.append(f"[{number}] {chunk['source']}, {where}\n{chunk['text']}")
    context = "\n\n".join(context_blocks)

    conversation = ""
    if history:
        conversation = (f"Conversation so far (use it only to understand what the question refers to):\n"
                        f"{format_history(history)}\n\n")

    return f"""You are the CSL448 Study Assistant, a tutor who helps a student revise from their own course notes.

Answer the student's question using ONLY the context below. The context comes from the student's notes.

Rules:
- Answer ONLY from the given context. Do not use outside knowledge, even if you know the answer.
- If the answer is not in the context, reply exactly: {NOT_FOUND}
- Keep the answer short and clear: a few sentences or bullet points.
- Start with the answer itself, not with phrases like "Based on your notes" (the sources are shown separately).
- If the question asks for a difference or comparison, compare point by point: one bullet per aspect,
  saying how each side differs on that aspect.
- If the student asks for more on an earlier answer, give new points from the context instead of
  repeating what was already said; if the context has nothing new, say so in one sentence.
- The notes may contain broken formulas or short bullet points; explain them in plain words.
- End your answer with one last line naming the context blocks you used, exactly like this: Sources: [1], [3]

{conversation}Context:
{context}

Question: {question}

Answer:"""


def build_plain_prompt(question):
    """A prompt with NO notes: what a plain chatbot would answer."""
    return (f"Answer this student's question in a few sentences or bullet points.\n\n"
            f"Question: {question}\n\nAnswer:")


# ----------------------------------------------------------------------------
# 2. Gemini calls
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


def is_retryable(error, attempt):
    """429 = too many requests (rate limit), 503 = model overloaded."""
    return error.code in (429, 503) and attempt < MAX_RETRIES


def wait_before_retry(error, attempt):
    print(f"  Gemini busy or rate-limited (error {error.code}). "
          f"Waiting {RETRY_WAIT_SECONDS} s, then retry {attempt}/{MAX_RETRIES - 1} ...")
    time.sleep(RETRY_WAIT_SECONDS)


def generate_answer(prompt):
    """Send the prompt to Gemini and return the answer text, retrying on rate limits."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = get_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return (response.text or "").strip()
        except errors.APIError as error:
            if not is_retryable(error, attempt):
                raise
            wait_before_retry(error, attempt)


def generate_answer_stream(prompt):
    """
    Like generate_answer, but yields the answer in pieces as Gemini writes it,
    so the app can show it appearing word by word. Retries only before the
    first piece arrives (a half-shown answer can't be retried cleanly).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        started = False
        try:
            for piece in get_client().models.generate_content_stream(model=GEMINI_MODEL, contents=prompt):
                if piece.text:
                    started = True
                    yield piece.text
            return
        except errors.APIError as error:
            if started or not is_retryable(error, attempt):
                raise
            wait_before_retry(error, attempt)


def plain_answer(question):
    """Gemini's answer WITHOUT the notes (for the "compare with plain Gemini" view)."""
    return generate_answer(build_plain_prompt(question))


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


def best_similarity_of(chunks):
    return max((chunk["similarity"] for chunk in chunks), default=0.0)


def retrieve(question, mode, k, retriever, history=None):
    """
    Search the notes for the question. For a follow-up that matches weakly on
    its own, also search with the previous question added and keep whichever
    search matched better.
    """
    chunks = retriever.search(question, mode, k)
    if history and best_similarity_of(chunks) < FOLLOW_UP_SIMILARITY:
        combined = retriever.search(f"{history[-1]['question']} {question}", mode, k)
        if best_similarity_of(combined) > best_similarity_of(chunks):
            chunks = combined
    return chunks


def prepare_answer(question, mode="hybrid", k=DEFAULT_TOP_K, retriever=None, history=None):
    """
    Everything before the Gemini call: retrieve, check the threshold, build
    the prompt. Returns the result dict described in answer_question; its
    "prompt" is None when the answer is already "Not found" (Gemini not needed).
    """
    retriever = retriever or get_retriever()
    chunks = retrieve(question, mode, k, retriever, history)
    best_similarity = best_similarity_of(chunks)

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
    if best_similarity >= SIMILARITY_THRESHOLD:
        result["prompt"] = build_prompt(question, chunks, history)
    return result


def finish_answer(result, answer):
    """Turn Gemini's raw reply into the final result (NOT_FOUND check, cited sources)."""
    answer = (answer or "").strip()

    # The model may add a full stop or extra words; treat any of those as NOT_FOUND.
    if not answer or NOT_FOUND.lower() in answer.lower():
        result["reason"] = "not_in_context"
        return result

    answer, cited = split_cited_chunks(answer, result["chunks"])
    result.update(answer=answer, found=True, reason="answered", sources=unique_sources(cited),
                  cited_ids=[chunk["id"] for chunk in cited])
    return result


def answer_question(question, mode="hybrid", k=DEFAULT_TOP_K, retriever=None, history=None):
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
    history is a list of earlier {"question", "answer"} turns, oldest first.
    """
    result = prepare_answer(question, mode, k, retriever, history)
    if result["prompt"] is None:
        return result
    return finish_answer(result, generate_answer(result["prompt"]))


def confidence_level(result):
    """"High", "Medium" or "Low" for an answer, from its best similarity; None if not found."""
    if not result["found"]:
        return None
    if result["best_similarity"] >= CONFIDENCE_HIGH:
        return "High"
    if result["best_similarity"] >= CONFIDENCE_MEDIUM:
        return "Medium"
    return "Low"


# ----------------------------------------------------------------------------
# Command-line demo
# ----------------------------------------------------------------------------
def main():
    # Notes contain symbols the Windows console cannot print; show "?" instead of crashing.
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
    if result["found"]:
        print(f"Confidence: {confidence_level(result)}")

    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(f"  - {source['source']}, {f'page {source['page']}' if source['page'] else 'overview'}")


if __name__ == "__main__":
    main()
