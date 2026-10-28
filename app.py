"""
app.py - Step 4 of the CSL448 Study Assistant: the web app (Streamlit)

Three pages, switched from the bar at the top:
  Ask           - chat with your notes; every answer shows the pages it came from
  Results       - the evaluation charts and tables made by evaluate.py
  How it works  - the RAG pipeline and where each syllabus unit shows up

Colours, font and rounded buttons are set in .streamlit/config.toml;
the finer, Apple-style details are in the CSS below.

Run from the project folder:
  venv\\Scripts\\python.exe -m streamlit run app.py
The page opens in your browser at http://localhost:8501
"""

import html
from pathlib import Path

import pandas as pd
import streamlit as st
from google.genai import errors

from rag import GEMINI_MODEL, SIMILARITY_THRESHOLD, answer_question
from retriever import DEFAULT_TOP_K, Retriever

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

# Search options as students see them -> retriever mode used by rag.py
RETRIEVERS = {"Hybrid": "hybrid", "Meaning": "dense", "Keywords": "bm25"}

# One-tap questions on the empty Ask page. The last one is NOT in the notes,
# so it shows the "Not found in the notes" behaviour.
SUGGESTIONS = [
    "What is ocean acidification?",
    "What is Lindeman's ten percent law?",
    "Who wrote Silent Spring and why?",
    "What is the Kyoto Protocol?",
]

CSS = """
<style>
:root {
  --ink: #1d1d1f;
  --ink-2: #6e6e73;
  --surface: #f5f5f7;
  --hairline: #d2d2d7;
  --accent: #0071e3;
}
html, body { -webkit-font-smoothing: antialiased; }

/* Frosted-glass top bar with the page links */
header[data-testid="stHeader"] {
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
}
[data-testid="stMainBlockContainer"] { padding-top: 4.5rem; padding-bottom: 7rem; }

/* Type */
.stMarkdown p, .stMarkdown li { font-size: 1.0625rem; line-height: 1.55; }
h2.section { font-size: 1.75rem; font-weight: 600; letter-spacing: -0.02em; color: var(--ink);
             margin: 2.5rem 0 0.4rem; }
p.lede { font-size: 1.0625rem; line-height: 1.55; color: var(--ink-2); margin: 0 0 1rem; max-width: 62ch; }

/* Page hero: one big, tight headline */
.hero { text-align: center; margin: 3.5rem auto 2rem; max-width: 36rem; }
.hero h1 { font-size: clamp(2.6rem, 8vw, 4rem); font-weight: 700; letter-spacing: -0.035em;
           line-height: 1.05; color: var(--ink); margin: 0 0 1rem; padding: 0; }
.hero p { font-size: clamp(1.1rem, 2.6vw, 1.35rem); line-height: 1.45; color: var(--ink-2); margin: 0; }
.hero.left { text-align: left; margin: 2rem 0 1.5rem; max-width: none; }

/* Suggestion pills */
.st-key-suggestions button {
  background: var(--surface); border: none; color: var(--ink);
  font-size: 0.95rem; padding: 0.55rem 1.1rem;
}
.st-key-suggestions button:hover { background: #e8e8ed; color: var(--ink); }
.st-key-suggestions button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* Your question: a blue bubble on the right.
   Streamlit wraps the keyed container in a layout wrapper; that wrapper is
   what gets pushed to the right. */
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-user_msg_"]) {
  align-self: flex-end; max-width: 82%; margin: 1.5rem 0 0.5rem;
}
[class*="st-key-user_msg_"] [data-testid="stMarkdownContainer"] {
  background: var(--accent); border-radius: 20px 20px 6px 20px; padding: 0.6rem 1rem;
}
[class*="st-key-user_msg_"] p { color: #ffffff; margin: 0; }

/* The answer: plain text on the page, like a reply in a document */
[class*="st-key-answer_"] { margin-bottom: 0.75rem; }

/* Source chips under an answer */
.sources { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; margin: 0.25rem 0; }
.sources .file { font-size: 0.85rem; color: var(--ink-2); margin-right: 0.15rem; }
.chip { display: inline-block; font-size: 0.8rem; font-weight: 500; color: var(--ink);
        background: var(--surface); border-radius: 980px; padding: 0.2rem 0.65rem; }
.meta { font-size: 0.8rem; color: var(--ink-2); margin-top: 0.6rem; }

/* "Not found in the notes" card */
.notfound { background: var(--surface); border-radius: 18px; padding: 1.1rem 1.25rem; }
.notfound .title { font-size: 1.0625rem; font-weight: 600; color: var(--ink); margin-bottom: 0.3rem; }
.notfound .body { font-size: 0.95rem; line-height: 1.5; color: var(--ink-2); }
.notfound .why { font-size: 0.85rem; line-height: 1.5; color: var(--ink-2); margin-top: 0.5rem; }

/* Slide text inside "Show the slide text" */
.slide { margin: 0.25rem 0 1rem; }
.slide .head { font-size: 0.85rem; font-weight: 600; color: var(--ink); margin-bottom: 0.35rem; }
.slide .head .match { font-weight: 400; color: var(--ink-2); margin-left: 0.4rem; }
.slide .head .used { font-weight: 500; font-size: 0.75rem; color: var(--accent); background: #e8f2fc;
                     border-radius: 980px; padding: 0.1rem 0.5rem; margin-left: 0.5rem; }
.slide .text { white-space: pre-wrap; font-size: 0.9rem; line-height: 1.55; color: #3a3a3c;
               background: var(--surface); border-radius: 14px; padding: 0.8rem 1rem; }

/* Fold-outs: no box, just a quiet line and a blue label */
[data-testid="stExpander"] details { border: none; border-top: 1px solid #e8e8ed; border-radius: 0; }
[data-testid="stExpander"] summary { padding-left: 0; padding-right: 0; color: var(--accent); background: none; }
[data-testid="stExpander"] summary:hover { color: #0077ed; background: none; }
[data-testid="stExpanderDetails"] { padding-left: 0; padding-right: 0; }

/* Charts on the Results page */
[data-testid="stImage"] img { border-radius: 14px; }

/* Chat box pinned to the bottom */
[data-testid="stChatInput"] > div { border-radius: 26px; }

/* Results page: headline numbers */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr)); gap: 1rem; margin: 0.5rem 0 1rem; }
.stat { background: var(--surface); border-radius: 18px; padding: 1.25rem; }
.stat .value { font-size: 2.6rem; font-weight: 700; letter-spacing: -0.03em; color: var(--ink); line-height: 1.1; }
.stat .label { font-size: 0.95rem; line-height: 1.4; color: var(--ink-2); margin-top: 0.35rem; }

/* How it works: the four steps */
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 1rem; }
.step { background: var(--surface); border-radius: 18px; padding: 1.25rem 1.35rem; }
.step .num { font-size: 0.9rem; font-weight: 600; color: var(--accent); }
.step .name { font-size: 1.2rem; font-weight: 600; color: var(--ink); margin: 0.2rem 0 0.4rem; }
.step .text { font-size: 0.95rem; line-height: 1.5; color: var(--ink-2); }

/* Simple tables (syllabus map, index details) */
table.clean { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
table.clean th { text-align: left; font-weight: 600; color: var(--ink); padding: 0.6rem 0.75rem 0.6rem 0;
                 border-bottom: 1px solid var(--hairline); }
table.clean td { vertical-align: top; color: var(--ink-2); line-height: 1.5; padding: 0.6rem 0.75rem 0.6rem 0;
                 border-bottom: 1px solid #e8e8ed; }
table.clean td:first-child { color: var(--ink); font-weight: 500; white-space: nowrap; }
code.inline { display: inline-block; white-space: nowrap; font-size: 0.85em; background: var(--surface);
              border-radius: 6px; padding: 0.1rem 0.35rem; margin: 0 0.25rem 0.25rem 0; }
</style>
"""


@st.cache_resource(show_spinner="Loading your notes ...")
def load_retriever():
    """Load the FAISS index, chunks, BM25 and embedding model once for all users."""
    return Retriever()


def init_state():
    """Default chat and search settings, kept while you switch pages."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("mode", "Hybrid")
    st.session_state.setdefault("top_k", DEFAULT_TOP_K)
    st.session_state.setdefault("show_prompt", False)
    # Streamlit forgets a widget's value on pages where the widget is not
    # drawn; writing the values back keeps them when you return to Ask.
    for key in ("mode", "top_k", "show_prompt"):
        st.session_state[key] = st.session_state[key]


# ----------------------------------------------------------------------------
# Small HTML builders (text from the notes is always escaped)
# ----------------------------------------------------------------------------
def hero(title, text, left=False):
    st.html(f'<div class="hero{" left" if left else ""}"><h1>{title}</h1><p>{text}</p></div>')


def section(title, text=None):
    st.html(f'<h2 class="section">{title}</h2>' + (f'<p class="lede">{text}</p>' if text else ""))


def sources_html(sources):
    """One row per file: the file name, then a chip for each page."""
    pages_by_file = {}
    for source in sources:
        pages_by_file.setdefault(source["source"], []).append(source["page"])
    rows = []
    for file, pages in pages_by_file.items():
        chips = "".join(f'<span class="chip">{f"Page {page}" if page else "Overview"}</span>' for page in pages)
        rows.append(f'<div class="sources"><span class="file">{html.escape(file)}</span>{chips}</div>')
    return "".join(rows)


def slides_html(chunks, cited_ids):
    """The retrieved chunks: page, how closely each matches, and whether the answer used it."""
    parts = []
    for chunk in chunks:
        used = '<span class="used">Used in the answer</span>' if chunk["id"] in cited_ids else ""
        parts.append(
            f'<div class="slide"><div class="head">{html.escape(chunk["source"])}, '
            f'{f"page {chunk['page']}" if chunk["page"] else "overview"}'
            f'<span class="match">{chunk["similarity"]:.0%} match</span>{used}</div>'
            f'<div class="text">{html.escape(chunk["text"])}</div></div>')
    return "".join(parts)


def not_found_html(message):
    if message["reason"] == "low_similarity":
        why = (f"The closest passage is only a {message['best_similarity']:.0%} match "
               f"(the cut-off is {SIMILARITY_THRESHOLD:.0%}), so Gemini wasn't asked.")
    else:
        why = "Gemini read the closest passages and the answer isn't in them."
    return ('<div class="notfound"><div class="title">Not found in the notes</div>'
            '<div class="body">Your notes don\'t cover this, so the assistant won\'t guess. '
            'Try rephrasing, or ask about a topic from your notes.</div>'
            f'<div class="why">{why}</div></div>')


def meta_html(message):
    return (f'<div class="meta">{message["mode"]} search, {message["k"]} passages read, '
            f'best match {message["best_similarity"]:.0%}</div>')


# ----------------------------------------------------------------------------
# Ask page
# ----------------------------------------------------------------------------
def ask_suggestion(question):
    st.session_state.pending_question = question


def new_chat():
    st.session_state.messages = []


def show_answer(index, message):
    """Draw one answer: text (or the not-found card), sources, details."""
    with st.container(key=f"answer_{index}"):
        if message["found"]:
            st.markdown(message["answer"])
            st.html(sources_html(message["sources"]) + meta_html(message))
        else:
            st.html(not_found_html(message) + meta_html(message))

        chunks = message["chunks"]
        if chunks:
            label = "Show the source text" if message["found"] else "Show the closest passages"
            with st.expander(f"{label} ({len(chunks)})"):
                st.html(slides_html(chunks, message.get("cited_ids", [])))
        if st.session_state.show_prompt and message["prompt"]:
            with st.expander("Show the prompt sent to Gemini"):
                st.code(message["prompt"], language=None, wrap_lines=True)


def ask_page():
    retriever = load_retriever()
    messages = st.session_state.messages

    # A new question comes from the chat box or from a suggestion pill.
    question = st.chat_input("Ask about your notes")
    question = question or st.session_state.pop("pending_question", None)
    if question:
        messages.append({"role": "user", "content": question})

    # Toolbar: new chat + search settings, on the right.
    with st.container(horizontal=True, horizontal_alignment="right", key="toolbar"):
        if messages:
            st.button("New chat", icon=":material/edit_square:", type="tertiary", on_click=new_chat)
        with st.popover("Search settings", icon=":material/tune:", type="tertiary"):
            st.segmented_control(
                "Search by", list(RETRIEVERS), key="mode", required=True,
                help="Meaning: sentence embeddings + FAISS. Keywords: BM25. Hybrid: both combined.")
            st.slider("Passages to read", min_value=1, max_value=10, key="top_k",
                      help="How many of the closest passages from your notes Gemini reads before answering.")
            st.toggle("Show the prompt sent to Gemini", key="show_prompt")

    if not messages:
        hero("Ask your notes.",
             "Answers come straight from your course notes, "
             "with the page each one came from.")
        with st.container(horizontal=True, horizontal_alignment="center", key="suggestions"):
            for number, suggestion in enumerate(SUGGESTIONS):
                st.button(suggestion, key=f"suggest_{number}",
                          on_click=ask_suggestion, args=(suggestion,))
        return

    # Chat history; the newest question (if any) is answered below.
    for index, message in enumerate(messages):
        if message["role"] == "user":
            with st.container(key=f"user_msg_{index}", width="content"):
                st.markdown(message["content"])
        else:
            show_answer(index, message)

    if question:
        mode_label = st.session_state.mode
        top_k = st.session_state.top_k
        try:
            with st.spinner("Reading your notes ..."):
                result = answer_question(question, RETRIEVERS[mode_label], top_k, retriever)
        except RuntimeError as error:          # GEMINI_API_KEY missing
            st.error(str(error))
            return
        except errors.APIError as error:       # Gemini failed even after retries
            st.error(f"Gemini didn't answer (error {error.code}). Wait a minute, then ask again.")
            return

        message = {"role": "assistant", "mode": mode_label, "k": top_k, **result}
        messages.append(message)
        show_answer(len(messages) - 1, message)


# ----------------------------------------------------------------------------
# Results page
# ----------------------------------------------------------------------------
def read_result(name):
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def as_percent(df, columns):
    df = df.copy()
    for column in columns:
        df[column] = df[column].map(lambda value: f"{value:.0%}")
    return df


def results_page():
    recall = read_result("retrieval_recall.csv")
    answers = read_result("answers_detail.csv")
    chunk_sizes = read_result("chunk_size_recall.csv")

    if recall is None:
        hero("How well it works.", "No results yet.", left=True)
        st.markdown("Run the evaluation from the project folder, then reload this page:")
        st.code("venv\\Scripts\\python.exe evaluate.py --chunk-sizes 300 500 800 --answers", language=None)
        return

    if answers is not None:
        in_notes = answers[answers["in_notes"]]
        not_in_notes = answers[~answers["in_notes"]]
        hero("How well it works.",
             f"Tested on {len(answers)} questions written from the notes: "
             f"{len(in_notes)} the notes can answer and {len(not_in_notes)} they can't.", left=True)
    else:
        hero("How well it works.", "Tested on questions written from the notes.", left=True)

    # Headline numbers
    hybrid = recall[recall["retriever"] == "Hybrid"].iloc[0]
    k_column = "Recall@3" if "Recall@3" in recall.columns else [c for c in recall.columns if c.startswith("Recall@")][-1]
    stats = [(f"{hybrid[k_column]:.0%}", f"of questions find the right page in the top {k_column.split('@')[1]} (hybrid search)")]
    if answers is not None:
        if len(in_notes):
            stats.append((f"{in_notes['correct'].mean():.0%}", "of answers judged correct by Gemini"))
        if len(not_in_notes):
            stats.append((f"{not_in_notes['correct'].mean():.0%}", "of off-topic questions answered with \"Not found in the notes\""))
    st.html('<div class="stats">' + "".join(
        f'<div class="stat"><div class="value">{value}</div><div class="label">{html.escape(label)}</div></div>'
        for value, label in stats) + "</div>")

    section("Finding the right page",
            "Keywords (BM25) matches words, Meaning (dense embeddings) matches ideas, and Hybrid combines "
            "both. Recall@k is the share of questions where the right page is among the first k results; "
            "MRR rewards finding it near the top.")
    if (RESULTS_DIR / "retrieval_recall.png").exists():
        st.image(str(RESULTS_DIR / "retrieval_recall.png"))
    recall_columns = [c for c in recall.columns if c.startswith("Recall@")]
    table = as_percent(recall, recall_columns).round({"MRR": 2}).rename(columns={"retriever": "Retriever"})
    st.dataframe(table, hide_index=True)

    if chunk_sizes is not None:
        section("Does chunk size matter?",
                "The same questions, searched in indexes that cut the notes into chunks of different "
                "lengths (overlap is 20% of the chunk size).")
        if (RESULTS_DIR / "chunk_size_recall.png").exists():
            st.image(str(RESULTS_DIR / "chunk_size_recall.png"))

    if answers is not None:
        section("Are the answers right?",
                "Each question went through the full app. Gemini compared every answer with the expected "
                "answer; off-topic questions count as correct only when the app says \"Not found in the notes\".")
        if (RESULTS_DIR / "answers_summary.png").exists():
            st.image(str(RESULTS_DIR / "answers_summary.png"))
        table = answers[["question", "verdict", "best_similarity"]].rename(columns={
            "question": "Question", "verdict": "Result", "best_similarity": "Best match"})
        st.dataframe(as_percent(table, ["Best match"]), hide_index=True)


# ----------------------------------------------------------------------------
# How it works page
# ----------------------------------------------------------------------------
STEPS = [
    ("Find", "Your question is matched against every chunk of your notes by keywords (BM25), by meaning "
             "(sentence embeddings in a FAISS index), or both."),
    ("Check", "If even the closest passage is a weak match, the answer is \"Not found in the notes\" "
              "and Gemini is never asked."),
    ("Answer", "The closest passages go into a prompt that tells Gemini to answer only from them, "
               "or to say \"Not found in the notes\"."),
    ("Cite", "The answer is shown with the file and page of each passage used, and you can open "
             "the exact source text to check it."),
]

SYLLABUS = [
    ("Unit 2", "Text preprocessing (lowercase, tokenize, remove stop-words) and BM25 keyword retrieval",
     "retriever.py"),
    ("Unit 3", "Bag of words and term counts (BM25) compared with word embeddings (dense search)",
     "retriever.py, evaluate.py"),
    ("Unit 4", "Transformer sentence embeddings, plus chunk size, overlap and top-k experiments",
     "ingest.py, evaluate.py"),
    ("Unit 5", "RAG, a FAISS vector database, prompt engineering and Gemini", "rag.py, ingest.py"),
    ("CO5", "Apply and evaluate: Recall@k, MRR, chunk sizes and Gemini-judged answers", "evaluate.py"),
]


def how_page():
    hero("How it works.", "Four steps turn your question into an answer you can check against your notes.",
         left=True)
    st.html('<div class="steps">' + "".join(
        f'<div class="step"><div class="num">{number}</div><div class="name">{name}</div>'
        f'<div class="text">{html.escape(text)}</div></div>'
        for number, (name, text) in enumerate(STEPS, start=1)) + "</div>")

    section("Where the syllabus shows up")
    st.html('<table class="clean"><tr><th>Unit</th><th>Topic</th><th>Code</th></tr>' + "".join(
        f'<tr><td>{unit}</td><td>{html.escape(topic)}</td><td>'
        + "".join(f'<code class="inline">{file}</code>' for file in files.split(", "))
        + "</td></tr>"
        for unit, topic, files in SYLLABUS) + "</table>")

    retriever = load_retriever()
    info = retriever.info
    notes = ", ".join(sorted({chunk["source"] for chunk in retriever.chunks}))
    details = [
        ("Notes", notes),
        ("Pages with text", info["num_pages"]),
        ("Chunks", f'{info["num_chunks"]} ({info["chunk_size"]} characters, {info["chunk_overlap"]} overlap)'),
        ("Embeddings", f'{info["embedding_model"]} ({info["vector_dimension"]} dimensions)'),
        ("Vector store", "FAISS IndexFlatIP on normalized vectors (cosine similarity)"),
        ("Language model", GEMINI_MODEL),
        ("Not-found cut-off", f"best match below {SIMILARITY_THRESHOLD:.0%}"),
    ]
    section("About this index")
    st.html('<table class="clean">' + "".join(
        f'<tr><td>{name}</td><td>{html.escape(str(value))}</td></tr>' for name, value in details) + "</table>")


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
st.set_page_config(page_title="CSL448 Study Assistant", page_icon="📚")
st.html(CSS)
init_state()

try:
    load_retriever()
except FileNotFoundError as error:
    hero("Build the index first.", html.escape(str(error)), left=True)
    st.stop()

page = st.navigation([
    st.Page(ask_page, title="Ask", url_path="ask", default=True),
    st.Page(results_page, title="Results", url_path="results"),
    st.Page(how_page, title="How it works", url_path="how-it-works"),
], position="top")
page.run()
