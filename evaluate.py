"""
evaluate.py - Step 5 of the CSL448 Study Assistant: evaluation (CO5: apply and evaluate)

Three experiments:
  1. Retrieval   - Recall@k and MRR for BM25, Dense and Hybrid.
                   Retrieval only, so it does NOT call Gemini.
  2. Chunk size  - the same retrieval scores for several chunk sizes
                   (Unit 4: context optimization). Does NOT call Gemini.
  3. Answers     - (optional) full RAG answers judged by Gemini against your
                   expected answers. Uses Gemini, about 2 requests per question,
                   with a 4-second pause between requests (free tier: 15/minute).

Test questions live in test_questions.json, a list like this:
  [
    {
      "question": "What is ocean acidification?",
      "expected_source": "L3.pdf",
      "expected_page": 1,
      "expected_answer": "The ongoing decrease in ocean pH caused by the uptake of human-made CO2."
    },
    {
      "question": "What is Lindeman's ten percent law?",
      "expected": [{"source": "L7.pdf", "page": 1}, {"source": "L7-2.pdf", "page": 1}],
      "expected_answer": "Only about 10% of the energy at one trophic level passes to the next."
    },
    {
      "question": "What is word2vec?",
      "expected_source": null,
      "expected_page": null,
      "expected_answer": "Not found in the notes"
    }
  ]
  - expected_page can also be a list, e.g. [1, 2], when the answer spans pages
    (a retrieved chunk from ANY of those pages counts as a hit).
  - "expected" (a list of file + page) is for answers found in more than one file.
  - expected_page null = the notes do not cover it. These questions are skipped
    in the retrieval scores and test that the bot says "Not found in the notes".

Run from the project folder (results go to the results/ folder):
  venv\\Scripts\\python.exe evaluate.py
  venv\\Scripts\\python.exe evaluate.py --chunk-sizes 300 500 800
  venv\\Scripts\\python.exe evaluate.py --answers
  venv\\Scripts\\python.exe evaluate.py --answers --answer-mode dense --answer-k 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw charts straight to PNG files
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from google.genai import errors

import rag
from ingest import NOTES_DIR, build_index
from retriever import DEFAULT_TOP_K, INDEX_DIR, SEARCH_MODES, Retriever

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = PROJECT_DIR / "test_questions.json"
RESULTS_DIR = PROJECT_DIR / "results"

DEFAULT_K_VALUES = [1, 3, 5]
GEMINI_PAUSE_SECONDS = 4                   # 15 requests/minute -> one every 4 s

# Chart style: one fixed colour per retriever, the same in every chart.
RETRIEVER_COLORS = {"bm25": "#2a78d6", "dense": "#eb6834", "hybrid": "#1baf7a"}
RETRIEVER_NAMES = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}
SURFACE, TEXT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


# ----------------------------------------------------------------------------
# Test questions
# ----------------------------------------------------------------------------
def load_questions(path):
    """
    Read the test questions and turn the expected file/page fields into
    q["expected_locations"], a list of (file, page) pairs. The file is None
    when expected_source is null (any file counts). An empty list means the
    question is not in the notes.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path.name} not found. Create it first (see the format at the top of evaluate.py).")

    with open(path, encoding="utf-8") as f:
        questions = json.load(f)

    for number, q in enumerate(questions, start=1):
        if not q.get("question"):
            raise SystemExit(f"Question {number} in {path.name} has no 'question' text.")
        if "expected" in q:
            # The answer is in several files: a list of {"source": ..., "page": ...}.
            q["expected_locations"] = [(item["source"], item["page"]) for item in q["expected"]]
        else:
            page = q.get("expected_page")
            pages = [] if page is None else (page if isinstance(page, list) else [page])
            q["expected_locations"] = [(q.get("expected_source"), p) for p in pages]
        q.setdefault("expected_answer", "")
    return questions


def is_hit(chunk, question):
    """A retrieved chunk is correct if it comes from an expected page of an expected file."""
    return any(page == chunk["page"] and source in (None, chunk["source"])
               for source, page in question["expected_locations"])


def describe(locations):
    """[("L3.pdf", 1), ("L3.pdf", 2)] -> "L3.pdf p.1, L3.pdf p.2" for the result tables."""
    return ", ".join(f"{source} p.{page}" if source else f"p.{page}" for source, page in locations)


# ----------------------------------------------------------------------------
# 1 + 2. Retrieval evaluation (no Gemini)
# ----------------------------------------------------------------------------
def evaluate_retrieval(retriever, questions, k_values):
    """
    For every retriever, search each answerable question once with the largest k
    and find the rank of the first correct chunk.
      Recall@k = share of questions with a correct chunk in the top k
      MRR      = average of 1/rank of the first correct chunk (0 if none)
    Returns (summary rows, per-question rows).
    """
    answerable = [q for q in questions if q["expected_locations"]]
    max_k = max(k_values)
    summary, details = [], []

    for mode in SEARCH_MODES:
        first_hit_ranks = []
        for q in answerable:
            results = retriever.search(q["question"], mode, max_k)
            rank = next((i for i, chunk in enumerate(results, start=1) if is_hit(chunk, q)), None)
            first_hit_ranks.append(rank)
            details.append({
                "retriever": RETRIEVER_NAMES[mode],
                "question": q["question"],
                "expected": describe(q["expected_locations"]),
                "retrieved": describe((chunk["source"], chunk["page"]) for chunk in results),
                "first_hit_rank": rank,
            })

        row = {"retriever": RETRIEVER_NAMES[mode]}
        for k in k_values:
            row[f"Recall@{k}"] = sum(r is not None and r <= k for r in first_hit_ranks) / len(answerable)
        row["MRR"] = sum(1 / r for r in first_hit_ranks if r is not None) / len(answerable)
        summary.append(row)

    return summary, details


def compare_chunk_sizes(questions, chunk_sizes, k_values):
    """Build one index per chunk size (overlap = 20%) and score retrieval on each."""
    rows = []
    for size in chunk_sizes:
        overlap = size // 5
        index_dir = INDEX_DIR / f"chunk_size_{size}"
        print(f"\n--- Chunk size {size} (overlap {overlap}) ---")
        info = build_index(NOTES_DIR, index_dir, size, overlap)
        summary, _ = evaluate_retrieval(Retriever(index_dir), questions, k_values)
        for row in summary:
            rows.append({"chunk_size": size, "chunk_overlap": overlap,
                         "num_chunks": info["num_chunks"], **row})
    return rows


# ----------------------------------------------------------------------------
# 3. Answer evaluation with Gemini as a judge
# ----------------------------------------------------------------------------
_last_gemini_call = 0.0


def wait_for_gemini():
    """Sleep so that Gemini requests are at least GEMINI_PAUSE_SECONDS apart."""
    remaining = GEMINI_PAUSE_SECONDS - (time.time() - _last_gemini_call)
    if remaining > 0:
        time.sleep(remaining)


def mark_gemini_call():
    global _last_gemini_call
    _last_gemini_call = time.time()


def build_judge_prompt(question, expected_answer, answer):
    return f"""You are grading answers from a study assistant for an NLP course.

Question: {question}

Expected answer (from the course notes): {expected_answer}

Assistant's answer: {answer}

Is the assistant's answer correct and consistent with the expected answer?
Extra correct detail is fine. Missing the main point or contradicting it is not.
Reply with exactly one word: CORRECT or INCORRECT."""


def judge_answer(question, expected_answer, answer):
    """Ask Gemini whether the answer matches the expected answer."""
    wait_for_gemini()
    reply = rag.generate_answer(build_judge_prompt(question, expected_answer, answer)).upper()
    mark_gemini_call()
    if "INCORRECT" in reply:
        return False
    return "CORRECT" in reply


def evaluate_answers(retriever, questions, mode, k):
    """
    Run the full RAG pipeline on every question and grade it:
      in the notes     -> CORRECT if Gemini-the-judge agrees with the expected answer
      not in the notes -> CORRECT only if the bot said "Not found in the notes"
    Stops early (keeping the rows so far) if Gemini keeps failing, e.g. daily quota.
    """
    rows = []
    for number, q in enumerate(questions, start=1):
        print(f"  [{number}/{len(questions)}] {q['question']}")
        try:
            wait_for_gemini()
            result = rag.answer_question(q["question"], mode, k, retriever)
            if result["prompt"]:              # Gemini was really called
                mark_gemini_call()

            in_notes = bool(q["expected_locations"])
            if not in_notes:
                correct = not result["found"]
                verdict = "correct refusal" if correct else "answered (hallucination)"
            elif not result["found"]:
                correct, verdict = False, "missed (said not found)"
            else:
                correct = judge_answer(q["question"], q["expected_answer"], result["answer"])
                verdict = "correct" if correct else "incorrect"
        except errors.APIError as error:
            print(f"\n  Gemini error {error.code}; stopping here and saving the results so far.")
            break

        rows.append({
            "question": q["question"],
            "in_notes": in_notes,
            "expected": describe(q["expected_locations"]),
            "retrieved": describe((chunk["source"], chunk["page"]) for chunk in result["chunks"]),
            "best_similarity": round(result["best_similarity"], 3),
            "gemini_called": result["prompt"] is not None,
            "answer": result["answer"],
            "expected_answer": q["expected_answer"],
            "verdict": verdict,
            "correct": correct,
        })
        print(f"      -> {verdict}")
    return rows


def summarize_answers(rows):
    """Accuracy on answerable questions, correct refusals on the others, overall."""
    df = pd.DataFrame(rows)
    in_notes, not_in_notes = df[df["in_notes"]], df[~df["in_notes"]]
    summary = []
    if len(in_notes):
        summary.append({"metric": "Answer accuracy (in the notes)",
                        "value": in_notes["correct"].mean(), "questions": len(in_notes)})
    if len(not_in_notes):
        summary.append({"metric": "Correct refusals (not in the notes)",
                        "value": not_in_notes["correct"].mean(), "questions": len(not_in_notes)})
    summary.append({"metric": "Overall", "value": df["correct"].mean(), "questions": len(df)})
    return summary


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def style_axes(ax, title, ylabel, legend_row=True):
    """Shared look: light surface, faint horizontal grid, quiet axes."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    # Leave room under the title for the legend row when there is one.
    ax.set_title(title, color=TEXT, fontsize=13, loc="left", pad=30 if legend_row else 14)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)


def grouped_bar_chart(groups, values_by_mode, title, xlabel, ylabel, path):
    """Bars grouped along the x-axis, one bar (colour) per retriever, value on each bar."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    style_axes(ax, title, ylabel)
    width = 0.26
    for i, mode in enumerate(SEARCH_MODES):
        positions = [g + (i - 1) * width for g in range(len(groups))]
        bars = ax.bar(positions, values_by_mode[mode], width=width * 0.9,
                      color=RETRIEVER_COLORS[mode], label=RETRIEVER_NAMES[mode])
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in values_by_mode[mode]],
                     padding=3, fontsize=8, color=MUTED)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_xlabel(xlabel, color=MUTED)
    # Legend in one row between the title and the plot, so it never covers a bar.
    ax.legend(frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def retrieval_chart(summary, k_values, path):
    by_name = {row["retriever"]: row for row in summary}
    values = {mode: [by_name[RETRIEVER_NAMES[mode]][f"Recall@{k}"] for k in k_values]
              for mode in SEARCH_MODES}
    grouped_bar_chart([f"Recall@{k}" for k in k_values], values,
                      "Retrieval: share of questions with the right page in the top k",
                      "", "Recall", path)


def chunk_size_chart(rows, k, path):
    sizes = sorted({row["chunk_size"] for row in rows})
    values = {mode: [next(r[f"Recall@{k}"] for r in rows
                          if r["chunk_size"] == size and r["retriever"] == RETRIEVER_NAMES[mode])
                     for size in sizes]
              for mode in SEARCH_MODES}
    grouped_bar_chart([f"{size} chars" for size in sizes], values,
                      f"Chunk size vs Recall@{k}", "Chunk size (overlap = 20%)", f"Recall@{k}", path)


def answer_chart(summary, mode, k, path):
    """One bar per metric; a single series, so no legend."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    style_axes(ax, f"Answer quality ({RETRIEVER_NAMES[mode]} retriever, top-{k}, judged by Gemini)", "",
               legend_row=False)
    labels = [f"{row['metric']}\n({row['questions']} questions)" for row in summary]
    bars = ax.bar(labels, [row["value"] for row in summary], width=0.5,
                  color=RETRIEVER_COLORS[mode])
    ax.bar_label(bars, labels=[f"{row['value']:.0%}" for row in summary],
                 padding=3, fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Evaluate the CSL448 Study Assistant.")
    parser.add_argument("--questions", default=str(QUESTIONS_FILE), help="test questions JSON file")
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES,
                        help="k values for Recall@k (default: 1 3 5)")
    parser.add_argument("--chunk-sizes", type=int, nargs="+",
                        help="also compare these chunk sizes, e.g. 300 500 800")
    parser.add_argument("--answers", action="store_true",
                        help="also judge full answers with Gemini (uses Gemini requests)")
    parser.add_argument("--answer-mode", choices=SEARCH_MODES, default="hybrid")
    parser.add_argument("--answer-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    k_values = sorted(set(args.k_values))
    questions = load_questions(args.questions)
    answerable = sum(1 for q in questions if q["expected_locations"])
    print(f"Loaded {len(questions)} questions ({answerable} in the notes, "
          f"{len(questions) - answerable} not in the notes)")
    if answerable == 0:
        raise SystemExit("Retrieval scores need questions with an expected_page.")

    # ---- 1. Retrieval on the main index ----
    print("\n=== 1. Retrieval evaluation (no Gemini) ===")
    retriever = Retriever()
    summary, details = evaluate_retrieval(retriever, questions, k_values)
    summary_df = pd.DataFrame(summary)
    print(f"Index: chunk size {retriever.info['chunk_size']}, overlap {retriever.info['chunk_overlap']}")
    print(summary_df.to_string(index=False, float_format="{:.2f}".format))
    summary_df.to_csv(results_dir / "retrieval_recall.csv", index=False, float_format="%.3f")
    pd.DataFrame(details).to_csv(results_dir / "retrieval_details.csv", index=False)
    retrieval_chart(summary, k_values, results_dir / "retrieval_recall.png")
    print(f"Saved retrieval_recall.csv, retrieval_details.csv, retrieval_recall.png")

    # ---- 2. Chunk sizes ----
    if args.chunk_sizes:
        print("\n=== 2. Chunk size comparison (no Gemini) ===")
        rows = compare_chunk_sizes(questions, sorted(set(args.chunk_sizes)), k_values)
        chunk_df = pd.DataFrame(rows)
        chart_k = 3 if 3 in k_values else k_values[-1]
        print()
        print(chunk_df.to_string(index=False, float_format="{:.2f}".format))
        chunk_df.to_csv(results_dir / "chunk_size_recall.csv", index=False, float_format="%.3f")
        chunk_size_chart(rows, chart_k, results_dir / "chunk_size_recall.png")
        print("Saved chunk_size_recall.csv, chunk_size_recall.png")

    # ---- 3. Answers with Gemini ----
    if args.answers:
        print(f"\n=== 3. Answer evaluation with Gemini ({args.answer_mode}, top-{args.answer_k}) ===")
        print(f"Up to {2 * len(questions)} Gemini requests, one every {GEMINI_PAUSE_SECONDS} s "
              f"(about {2 * len(questions) * GEMINI_PAUSE_SECONDS // 60 + 1} minutes).")
        rows = evaluate_answers(retriever, questions, args.answer_mode, args.answer_k)
        if rows:
            answer_summary = summarize_answers(rows)
            print()
            print(pd.DataFrame(answer_summary).to_string(index=False, float_format="{:.0%}".format))
            pd.DataFrame(rows).to_csv(results_dir / "answers_detail.csv", index=False)
            pd.DataFrame(answer_summary).to_csv(results_dir / "answers_summary.csv",
                                                index=False, float_format="%.3f")
            answer_chart(answer_summary, args.answer_mode, args.answer_k,
                         results_dir / "answers_summary.png")
            print("Saved answers_detail.csv, answers_summary.csv, answers_summary.png")

    print(f"\nAll results are in {results_dir}")


if __name__ == "__main__":
    main()
