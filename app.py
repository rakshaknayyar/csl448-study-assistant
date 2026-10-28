"""
app.py - Step 4 of the CSL448 Study Assistant: the web app (Streamlit)

Five pages, switched from the bar at the top:
  Ask           - chat with your notes. Answers stream in, follow-up questions
                  work, every answer shows its pages (with page previews) and a
                  confidence level, and it can be compared with plain Gemini
  Lectures      - one card per lecture with TF-IDF key terms, a summary,
                  and shortcuts to ask about it or take a quiz
  Quiz          - multiple-choice questions from one lecture, graded
  Results       - the evaluation numbers and interactive charts (evaluate.py)
  How it works  - the RAG pipeline and where each syllabus unit shows up

Colours, font and rounded buttons are set in .streamlit/config.toml;
the finer, Apple-style details are in the CSS below.

Run from the project folder (or double-click run_app.bat):
  venv\\Scripts\\python.exe -m streamlit run app.py
The page opens in your browser at http://localhost:8501
"""

import html
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from google.genai import errors

import study
from rag import (GEMINI_MODEL, SIMILARITY_THRESHOLD, confidence_level, finish_answer,
                 generate_answer_stream, plain_answer, prepare_answer)
from retriever import DEFAULT_TOP_K, Retriever

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

# Search options as students see them -> retriever mode used by rag.py
RETRIEVERS = {"Hybrid": "hybrid", "Meaning": "dense", "Keywords": "bm25"}

# The same retrievers, as named in results/*.csv, with one fixed colour each.
CHART_NAMES = {"BM25": "Keywords (BM25)", "Dense": "Meaning (Dense)", "Hybrid": "Hybrid"}
CHART_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

# One-tap questions on the empty Ask page. The last one is NOT in the notes,
# so it shows the "Not found in the notes" behaviour.
SUGGESTIONS = [
    "What is ocean acidification?",
    "What is Lindeman's ten percent law?",
    "Who wrote Silent Spring and why?",
    "What is the Kyoto Protocol?",
]

QUIZ_LENGTH = 5

CSS = """
<style>
:root {
  --ink: #1d1d1f;
  --ink-2: #6e6e73;
  --surface: #f5f5f7;
  --surface-2: #e8e8ed;
  --hairline: #d2d2d7;
  --accent: #0071e3;
  --accent-soft: #e8f2fc;
  --good: #248a3d;
  --good-soft: #e9f6ec;
  --bad: #d70015;
  --bad-soft: #fdecee;
  --warn: #b25000;
  --warn-soft: #fff4e5;
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

/* Pill buttons on a gray surface (suggestions, lecture actions) */
.st-key-suggestions button, [class*="st-key-lecture_"] button {
  background: var(--surface); border: none; color: var(--ink); font-size: 0.95rem; padding: 0.55rem 1.1rem;
}
.st-key-suggestions button:hover, [class*="st-key-lecture_"] button:hover { background: var(--surface-2); color: var(--ink); }
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
.col-label { font-size: 0.8rem; font-weight: 600; color: var(--ink-2); margin-bottom: 0.25rem; }
[class*="st-key-plain_"] { background: var(--warn-soft); border-radius: 16px; padding: 0.75rem 1rem; }

/* Source chips, confidence and details under an answer */
.sources { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; margin: 0.25rem 0; }
.sources .file { font-size: 0.85rem; color: var(--ink-2); margin-right: 0.15rem; }
.chip { display: inline-block; font-size: 0.8rem; font-weight: 500; color: var(--ink);
        background: var(--surface); border-radius: 980px; padding: 0.2rem 0.65rem; }
.conf { display: inline-block; font-size: 0.8rem; font-weight: 600; border-radius: 980px; padding: 0.2rem 0.65rem; }
.conf.High { color: var(--good); background: var(--good-soft); }
.conf.Medium { color: var(--warn); background: var(--warn-soft); }
.conf.Low { color: var(--bad); background: var(--bad-soft); }
.meta { font-size: 0.8rem; color: var(--ink-2); margin-top: 0.6rem; }

/* "Not found in the notes" card */
.notfound { background: var(--surface); border-radius: 18px; padding: 1.1rem 1.25rem; }
.notfound .title { font-size: 1.0625rem; font-weight: 600; color: var(--ink); margin-bottom: 0.3rem; }
.notfound .body { font-size: 0.95rem; line-height: 1.5; color: var(--ink-2); }
.notfound .why { font-size: 0.85rem; line-height: 1.5; color: var(--ink-2); margin-top: 0.5rem; }

/* Passages inside "Show the source text" */
.slide .head { font-size: 0.85rem; font-weight: 600; color: var(--ink); margin-bottom: 0.35rem; }
.slide .head .match { font-weight: 400; color: var(--ink-2); margin-left: 0.4rem; }
.slide .head .used { font-weight: 500; font-size: 0.75rem; color: var(--accent); background: var(--accent-soft);
                     border-radius: 980px; padding: 0.1rem 0.5rem; margin-left: 0.5rem; }
.slide .text { white-space: pre-wrap; font-size: 0.9rem; line-height: 1.55; color: #3a3a3c;
               background: var(--surface); border-radius: 14px; padding: 0.8rem 1rem;
               max-height: 16rem; overflow-y: auto; }
[data-testid="stImage"] img { border-radius: 10px; border: 1px solid var(--surface-2); }

/* Fold-outs: no box, just a quiet line and a blue label */
[data-testid="stExpander"] details { border: none; border-top: 1px solid var(--surface-2); border-radius: 0; }
[data-testid="stExpander"] summary { padding-left: 0; padding-right: 0; color: var(--accent); background: none; }
[data-testid="stExpander"] summary:hover { color: #0077ed; background: none; }
[data-testid="stExpanderDetails"] { padding-left: 0; padding-right: 0; }

/* Chat box pinned to the bottom */
[data-testid="stChatInput"] > div { border-radius: 26px; }

/* Lectures page: one gray card per lecture */
[class*="st-key-lecture_"] { background: var(--surface); border-radius: 20px; padding: 1.2rem 1.3rem 1rem; }
[class*="st-key-lecture_"] button { background: #ffffff; font-size: 0.875rem; padding: 0.35rem 0.9rem; }
[class*="st-key-lecture_"] button:hover { background: #ffffff; color: var(--accent); }
.lecture .label { display: inline-block; font-size: 0.75rem; font-weight: 600; color: var(--accent);
                  background: var(--accent-soft); border-radius: 980px; padding: 0.15rem 0.55rem; }
.lecture h3 { font-size: 1.3rem; font-weight: 600; letter-spacing: -0.015em; color: var(--ink);
              margin: 0.5rem 0 0.2rem; padding: 0; line-height: 1.25; }
.lecture .info { font-size: 0.85rem; color: var(--ink-2); margin-bottom: 0.7rem; }
.terms { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.4rem; }
.terms span { font-size: 0.8rem; color: var(--ink); background: #ffffff; border-radius: 980px; padding: 0.2rem 0.6rem; }

/* Quiz page */
[class*="st-key-quizq_"] { background: var(--surface); border-radius: 20px; padding: 1.1rem 1.3rem 0.6rem; margin-bottom: 0.4rem; }
.quiz-q .num { font-size: 0.8rem; font-weight: 600; color: var(--accent); }
.quiz-q .text { font-size: 1.1rem; font-weight: 600; color: var(--ink); line-height: 1.4; margin: 0.2rem 0 0.2rem; }
.feedback { border-radius: 12px; padding: 0.6rem 0.85rem; font-size: 0.92rem; line-height: 1.5; margin: 0.25rem 0 0.6rem; }
.feedback.right { background: var(--good-soft); color: #1d5e2c; }
.feedback.wrong { background: var(--bad-soft); color: #8a0010; }
.feedback b { font-weight: 600; }
.score { font-size: 3.2rem; font-weight: 700; letter-spacing: -0.035em; color: var(--ink); line-height: 1; }
.score-label { font-size: 1rem; color: var(--ink-2); margin: 0.3rem 0 1rem; }

/* Results page: headline numbers */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr)); gap: 1rem; margin: 0.5rem 0 1rem; }
.stat { background: var(--surface); border-radius: 18px; padding: 1.25rem; }
.stat .value { font-size: 2.6rem; font-weight: 700; letter-spacing: -0.03em; color: var(--ink); line-height: 1.1; }
.stat .label { font-size: 0.95rem; line-height: 1.4; color: var(--ink-2); margin-top: 0.35rem; }

/* How it works: the steps */
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
                 border-bottom: 1px solid var(--surface-2); }
table.clean td:first-child { color: var(--ink); font-weight: 500; white-space: nowrap; }
code.inline { display: inline-block; white-space: nowrap; font-size: 0.85em; background: var(--surface);
              border-radius: 6px; padding: 0.1rem 0.35rem; margin: 0 0.25rem 0.25rem 0; }
</style>
"""


@st.cache_resource(show_spinner="Loading your notes ...")
def load_retriever():
    """Load the FAISS index, chunks, BM25 and embedding model once for all users."""
    return Retriever()


@st.cache_data(show_spinner="Reading your lectures ...")
def load_lectures():
    """Every lecture with its pages and TF-IDF key terms (see study.py)."""
    return study.load_lectures()


def init_state():
    """Default chat and search settings, kept while you switch pages."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("mode", "Hybrid")
    st.session_state.setdefault("top_k", DEFAULT_TOP_K)
    st.session_state.setdefault("show_prompt", False)
    st.session_state.setdefault("compare", False)
    # Streamlit forgets a widget's value on pages where the widget is not
    # drawn; writing the values back keeps them when you return to Ask.
    for key in ("mode", "top_k", "show_prompt", "compare"):
        st.session_state[key] = st.session_state[key]


def gemini_error_message(error):
    """A plain-language message for any failure while talking to Gemini."""
    if isinstance(error, RuntimeError):            # GEMINI_API_KEY missing
        return str(error)
    if isinstance(error, errors.APIError):         # still failing after retries
        return f"Gemini didn't answer (error {error.code}). Wait a minute, then try again."
    return "Gemini's reply couldn't be used this time. Try again."


# ----------------------------------------------------------------------------
# Small HTML builders (text from the notes is always escaped)
# ----------------------------------------------------------------------------
def hero(title, text, left=False):
    st.html(f'<div class="hero{" left" if left else ""}"><h1>{title}</h1><p>{text}</p></div>')


def section(title, text=None):
    st.html(f'<h2 class="section">{title}</h2>' + (f'<p class="lede">{text}</p>' if text else ""))


def page_label(page):
    return f"page {page}" if page else "overview"


def sources_html(message):
    """Confidence, then one row per file: the file name and a chip for each page."""
    level = confidence_level(message)
    rows = [f'<div class="sources"><span class="conf {level}">{level} confidence</span></div>'] if level else []
    pages_by_file = {}
    for source in message["sources"]:
        pages_by_file.setdefault(source["source"], []).append(source["page"])
    for file, pages in pages_by_file.items():
        chips = "".join(f'<span class="chip">{page_label(page).capitalize()}</span>' for page in pages)
        rows.append(f'<div class="sources"><span class="file">{html.escape(file)}</span>{chips}</div>')
    return "".join(rows)


def passage_html(chunk, cited_ids):
    """One retrieved passage: where it is from, how closely it matches, and its text."""
    used = '<span class="used">Used in the answer</span>' if chunk["id"] in cited_ids else ""
    return (f'<div class="slide"><div class="head">{html.escape(chunk["source"])}, {page_label(chunk["page"])}'
            f'<span class="match">{chunk["similarity"]:.0%} match</span>{used}</div>'
            f'<div class="text">{html.escape(chunk["text"])}</div></div>')


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


@st.dialog("Page preview", width="large")
def show_page(source, page):
    st.image(str(study.page_image(source, page)), width="stretch")
    st.caption(f"{source}, page {page}")


# ----------------------------------------------------------------------------
# Ask page
# ----------------------------------------------------------------------------
def ask_suggestion(question):
    st.session_state.pending_question = question


def new_chat():
    st.session_state.messages = []


def chat_history(messages):
    """Earlier question/answer pairs, oldest first, for follow-up questions."""
    history = []
    for asked, answered in zip(messages, messages[1:]):
        if asked["role"] == "user" and answered["role"] == "assistant":
            history.append({"question": asked["content"], "answer": answered["answer"]})
    return history


def chat_markdown(messages):
    """The whole chat as Markdown revision notes, for the download button."""
    lines = ["# CSL448 Study Assistant: revision notes", ""]
    for message in messages:
        if message["role"] == "user":
            lines += [f"## {message['content']}", ""]
        else:
            lines += [message["answer"], ""]
            if message["sources"]:
                where = ", ".join(f"{s['source']} {page_label(s['page'])}" for s in message["sources"])
                lines += [f"*Sources: {where}*", ""]
    return "\n".join(lines)


def visible_while_streaming(text):
    """Hide the "Sources: [1]" line and a "Not found" reply while the answer types out."""
    if text.strip().lower().startswith("not found"):
        return ""
    return re.split(r"\n\s*\**\s*Sources?\s*\**\s*:", text)[0]


def show_rag_answer(message):
    if message["found"]:
        st.markdown(message["answer"])
        st.html(sources_html(message) + meta_html(message))
    else:
        st.html(not_found_html(message) + meta_html(message))


def show_answer(index, message):
    """Draw one answer: text (or the not-found card), sources, plain-Gemini comparison, details."""
    with st.container(key=f"answer_{index}"):
        if message.get("plain") is not None:
            notes_column, plain_column = st.columns(2, gap="medium")
            with notes_column:
                st.html('<div class="col-label">From your notes (RAG)</div>')
                show_rag_answer(message)
            with plain_column:
                st.html('<div class="col-label">Plain Gemini, without your notes</div>')
                with st.container(key=f"plain_{index}"):
                    st.markdown(message["plain"])
                st.html('<div class="meta">Not checked against your notes: no pages to verify it, '
                        'and it may include facts your course never taught.</div>')
        else:
            show_rag_answer(message)

        chunks = message["chunks"]
        if chunks:
            label = "Show the source text" if message["found"] else "Show the closest passages"
            with st.expander(f"{label} ({len(chunks)})"):
                for number, chunk in enumerate(chunks):
                    preview, text = st.columns([1, 3], gap="small")
                    with preview:
                        if chunk["page"]:
                            st.image(str(study.page_image(chunk["source"], chunk["page"])), width="stretch")
                            if st.button("Open page", key=f"open_{index}_{number}", type="tertiary",
                                         icon=":material/open_in_full:"):
                                show_page(chunk["source"], chunk["page"])
                    with text:
                        st.html(passage_html(chunk, message.get("cited_ids", [])))
        if st.session_state.show_prompt and message["prompt"]:
            with st.expander("Show the prompt sent to Gemini"):
                st.code(message["prompt"], language=None, wrap_lines=True)


def answer_new_question(question, history, retriever):
    """Retrieve, stream Gemini's answer as it is written, then (optionally) ask plain Gemini."""
    mode_label, top_k = st.session_state.mode, st.session_state.top_k
    result = prepare_answer(question, RETRIEVERS[mode_label], top_k, retriever, history)

    if result["prompt"]:
        typing = st.empty()
        text = ""
        for piece in generate_answer_stream(result["prompt"]):
            text += piece
            typing.markdown(visible_while_streaming(text) + " ▍")
        typing.empty()
        result = finish_answer(result, text)

    plain = None
    if st.session_state.compare:
        with st.spinner("Asking plain Gemini, without your notes ..."):
            plain = plain_answer(question)
    return {"role": "assistant", "mode": mode_label, "k": top_k, "plain": plain, **result}


def ask_page():
    retriever = load_retriever()
    messages = st.session_state.messages

    # A new question comes from the chat box, a suggestion pill or the Lectures page.
    question = st.chat_input("Ask about your notes")
    question = question or st.session_state.pop("pending_question", None)
    history = chat_history(messages)
    if question:
        messages.append({"role": "user", "content": question})

    # Toolbar: new chat, download, search settings, on the right.
    with st.container(horizontal=True, horizontal_alignment="right", key="toolbar"):
        if messages:
            st.button("New chat", icon=":material/edit_square:", type="tertiary", on_click=new_chat)
            st.download_button("Download chat", chat_markdown(messages), file_name="study-assistant-chat.md",
                               mime="text/markdown", icon=":material/download:", type="tertiary",
                               on_click="ignore")
        with st.popover("Search settings", icon=":material/tune:", type="tertiary"):
            st.segmented_control(
                "Search by", list(RETRIEVERS), key="mode", required=True,
                help="Meaning: sentence embeddings + FAISS. Keywords: BM25. Hybrid: both combined.")
            st.slider("Passages to read", min_value=1, max_value=10, key="top_k",
                      help="How many of the closest passages from your notes Gemini reads before answering.")
            st.toggle("Compare with plain Gemini", key="compare",
                      help="Also ask Gemini without your notes, to see what RAG prevents. Uses one more request.")
            st.toggle("Show the prompt sent to Gemini", key="show_prompt")

    if not messages:
        lecture_count = len({chunk["source"] for chunk in retriever.chunks if chunk["page"]})
        hero("Ask your notes.",
             f"Answers come straight from your {lecture_count} lectures, "
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
        try:
            message = answer_new_question(question, history, retriever)
        except (RuntimeError, errors.APIError) as error:
            st.error(gemini_error_message(error))
            return
        messages.append(message)
        show_answer(len(messages) - 1, message)


# ----------------------------------------------------------------------------
# Lectures page
# ----------------------------------------------------------------------------
@st.dialog("Summary", width="large")
def show_summary(lecture):
    st.html(f'<div class="lecture"><span class="label">{lecture["label"]}</span>'
            f'<h3>{html.escape(lecture["title"])}</h3></div>')
    try:
        with st.spinner("Summarizing this lecture ..."):
            summary = study.summarize(lecture)
    except (RuntimeError, errors.APIError) as error:
        st.error(gemini_error_message(error))
        return
    st.markdown(summary)
    st.caption(f"Written by Gemini from {lecture['source']} only, and saved so it opens instantly next time.")


def lectures_page():
    lectures = load_lectures()
    pages = sum(len(lecture["pages"]) for lecture in lectures)
    hero("Your lectures.",
         f"{len(lectures)} lectures, {pages} pages. The key terms on each card are picked with TF-IDF: "
         "words that are frequent in that lecture but rare in the others.", left=True)

    columns = st.columns(2, gap="medium")
    for number, lecture in enumerate(lectures):
        with columns[number % 2]:
            with st.container(key=f"lecture_{number}"):
                terms = "".join(f"<span>{html.escape(term)}</span>" for term in lecture["key_terms"])
                page_count = len(lecture["pages"])
                st.html(f'<div class="lecture"><span class="label">{lecture["label"] or "Notes"}</span>'
                        f'<h3>{html.escape(lecture["title"])}</h3>'
                        f'<div class="info">{page_count} page{"s" if page_count != 1 else ""}, '
                        f'{html.escape(lecture["source"])}</div>'
                        f'<div class="terms">{terms}</div></div>')
                with st.container(horizontal=True, gap="small"):
                    if st.button("Summary", key=f"summary_{number}", icon=":material/notes:"):
                        show_summary(lecture)
                    if st.button("Quiz me", key=f"quiz_{number}", icon=":material/quiz:"):
                        st.session_state.quiz_source = lecture["source"]
                        st.switch_page(PAGES["quiz"])
                    if st.button("Ask", key=f"ask_{number}", icon=":material/chat_bubble:"):
                        st.session_state.pending_question = f"Give me an overview of {lecture['title']}."
                        st.switch_page(PAGES["ask"])


# ----------------------------------------------------------------------------
# Quiz page
# ----------------------------------------------------------------------------
def start_quiz(lecture):
    """Ask Gemini for a new quiz and keep it in the session."""
    with st.spinner(f"Writing a {QUIZ_LENGTH}-question quiz on {lecture['title']} ..."):
        questions = study.make_quiz(lecture, QUIZ_LENGTH)
    st.session_state.quiz = {
        "source": lecture["source"],
        "questions": questions,
        "submitted": False,
        "id": st.session_state.get("quiz_counter", 0) + 1,   # fresh keys for the radio buttons
    }
    st.session_state.quiz_counter = st.session_state.quiz["id"]


def quiz_page():
    lectures = load_lectures()
    by_source = {lecture["source"]: lecture for lecture in lectures}
    hero("Quiz yourself.",
         f"{QUIZ_LENGTH} multiple-choice questions from one lecture, written by Gemini from your notes. "
         "Each answer tells you which page to revise.", left=True)

    sources = list(by_source)
    chosen = st.session_state.get("quiz_source", sources[0])
    with st.container(horizontal=True, vertical_alignment="bottom", key="quiz_controls"):
        source = st.selectbox("Lecture", sources, index=sources.index(chosen) if chosen in sources else 0,
                              format_func=lambda s: f"{by_source[s]['label']} {by_source[s]['title']}".strip())
        new_quiz = st.button("New quiz", type="primary", icon=":material/auto_awesome:")
    st.session_state.quiz_source = source
    lecture = by_source[source]

    if new_quiz:
        try:
            start_quiz(lecture)
        except (RuntimeError, errors.APIError, ValueError) as error:
            st.error(gemini_error_message(error))
            return

    quiz = st.session_state.get("quiz")
    if not quiz or quiz["source"] != source:
        st.html(f'<p class="lede">Press <b>New quiz</b> to get {QUIZ_LENGTH} questions on '
                f'{html.escape(lecture["title"])}.</p>')
        return

    questions = quiz["questions"]
    keys = [f"quiz{quiz['id']}_{n}" for n in range(len(questions))]
    if quiz["submitted"]:
        score = sum(st.session_state.get(key) == q["answer"] for key, q in zip(keys, questions))
        st.html(f'<div class="score">{score} / {len(questions)}</div>'
                f'<div class="score-label">correct. Revise the pages shown under any you missed.</div>')

    for number, (key, q) in enumerate(zip(keys, questions), start=1):
        with st.container(key=f"quizq_{quiz['id']}_{number}"):
            st.html(f'<div class="quiz-q"><div class="num">Question {number}</div>'
                    f'<div class="text">{html.escape(q["question"])}</div></div>')
            st.radio(f"Question {number}", [0, 1, 2, 3], index=None, key=key, label_visibility="collapsed",
                     format_func=lambda option, q=q: q["options"][option], disabled=quiz["submitted"])
            if quiz["submitted"]:
                picked = st.session_state.get(key)
                where = f" (page {q['page']} of {source})" if q.get("page") else ""
                explanation = html.escape(q["explanation"]) + html.escape(where)
                if picked == q["answer"]:
                    st.html(f'<div class="feedback right"><b>Correct.</b> {explanation}</div>')
                else:
                    correct = html.escape(q["options"][q["answer"]])
                    yours = "You skipped this one." if picked is None else \
                        f"You picked {html.escape(q['options'][picked])}."
                    st.html(f'<div class="feedback wrong"><b>Not quite.</b> {yours} The answer is '
                            f'<b>{correct}</b>. {explanation}</div>')

    if not quiz["submitted"]:
        answered = sum(st.session_state.get(key) is not None for key in keys)
        if st.button(f"Check answers ({answered} of {len(questions)} answered)", type="primary",
                     disabled=answered == 0):
            quiz["submitted"] = True
            st.rerun()


# ----------------------------------------------------------------------------
# Results page
# ----------------------------------------------------------------------------
def read_result(name):
    """Read a results CSV. Scores are rounded to whole percents once, here, so the
    charts (rounded in the browser) and tables (rounded in Python) always agree."""
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    scores = [c for c in df.columns if c.startswith("Recall@")]
    df[scores] = df[scores].round(2)
    return df


def as_percent(df, columns):
    df = df.copy()
    for column in columns:
        df[column] = df[column].map(lambda value: f"{value:.0%}")
    return df


def styled(chart):
    """Quiet axes and gridlines in the app's font and colours."""
    return (chart.configure_axis(labelFont="Inter", titleFont="Inter", labelColor="#6e6e73",
                                 titleColor="#6e6e73", gridColor="#e8e8ed", domainColor="#d2d2d7",
                                 tickColor="#d2d2d7", labelFontSize=12, titleFontSize=12)
                 .configure_legend(labelFont="Inter", labelColor="#1d1d1f", labelFontSize=12,
                                   orient="top", title=None, symbolType="square",
                                   labelLimit=0, columnPadding=18)
                 .configure_view(strokeWidth=0))


def grouped_bars(data, group, group_title, value_title):
    """One group of bars per value of `group`, one colour per retriever, % on hover and on each bar."""
    names = list(CHART_NAMES.values())
    base = alt.Chart(data).encode(
        x=alt.X(f"{group}:N", title=group_title, sort=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Retriever:N", sort=names),
        y=alt.Y("Score:Q", title=value_title, scale=alt.Scale(domain=[0, 1.08]),
                axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1])),
        tooltip=[alt.Tooltip("Retriever:N"), alt.Tooltip(f"{group}:N", title=group_title or group),
                 alt.Tooltip("Score:Q", format=".0%")],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        color=alt.Color("Retriever:N", sort=names, scale=alt.Scale(domain=names, range=CHART_COLORS)))
    labels = base.mark_text(dy=-7, fontSize=10, font="Inter", color="#6e6e73").encode(
        text=alt.Text("Score:Q", format=".0%"))
    return styled((bars + labels).properties(height=300))


def results_page():
    recall = read_result("retrieval_recall.csv")
    answers = read_result("answers_detail.csv")
    chunk_sizes = read_result("chunk_size_recall.csv")

    if recall is None:
        hero("How well it works.", "No results yet.", left=True)
        st.markdown("Run the evaluation from the project folder, then reload this page:")
        st.code("venv\\Scripts\\python.exe evaluate.py --chunk-sizes 500 800 1000 1500 --answers", language=None)
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
    recall_columns = [c for c in recall.columns if c.startswith("Recall@")]
    k_column = "Recall@3" if "Recall@3" in recall_columns else recall_columns[-1]
    hybrid = recall[recall["retriever"] == "Hybrid"].iloc[0]
    stats = [(f"{hybrid[k_column]:.0%}",
              f"of questions find the right page in the top {k_column.split('@')[1]} (hybrid search)")]
    if answers is not None:
        if len(in_notes):
            stats.append((f"{in_notes['correct'].mean():.0%}", "of answers judged correct by Gemini"))
        if len(not_in_notes):
            stats.append((f"{not_in_notes['correct'].mean():.0%}",
                          "of off-topic questions answered with \"Not found in the notes\""))
    st.html('<div class="stats">' + "".join(
        f'<div class="stat"><div class="value">{value}</div><div class="label">{html.escape(label)}</div></div>'
        for value, label in stats) + "</div>")

    section("Finding the right page",
            "Keywords (BM25) matches words, Meaning (dense embeddings) matches ideas, and Hybrid combines "
            "both. Recall@k is the share of questions where the right page is among the first k results; "
            "MRR rewards finding it near the top. Hover over a bar for its value.")
    long = recall.melt(id_vars="retriever", value_vars=recall_columns, var_name="Measure", value_name="Score")
    long["Retriever"] = long["retriever"].map(CHART_NAMES)
    st.altair_chart(grouped_bars(long, "Measure", None, "Questions"), width="stretch")
    table = as_percent(recall, recall_columns).round({"MRR": 2})
    table["retriever"] = table["retriever"].map(CHART_NAMES)
    st.dataframe(table.rename(columns={"retriever": "Retriever"}), hide_index=True)

    if chunk_sizes is not None:
        section("Does chunk size matter?",
                f"The same questions, searched in indexes that cut the notes into chunks of different "
                f"lengths (overlap is 20% of the chunk size). Showing {k_column}.")
        sizes = chunk_sizes.assign(Retriever=chunk_sizes["retriever"].map(CHART_NAMES),
                                   Size=chunk_sizes["chunk_size"].map(lambda s: f"{s} characters"),
                                   Score=chunk_sizes[k_column])
        st.altair_chart(grouped_bars(sizes, "Size", "Chunk size", k_column), width="stretch")

    if answers is not None:
        section("Are the answers right?",
                "Each question went through the full app. Gemini compared every answer with the expected "
                "answer; off-topic questions count as correct only when the app says \"Not found in the notes\".")
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
               "or to say \"Not found in the notes\". Earlier questions are included so follow-ups work."),
    ("Cite", "The answer is shown with the file and page of each passage used, a confidence level, "
             "and a preview of the real page to check it."),
]

SYLLABUS = [
    ("Unit 2", "Text preprocessing (lowercase, tokenize, remove stop-words) and BM25 keyword retrieval",
     "retriever.py"),
    ("Unit 3", "Bag of words and term counts (BM25) compared with word embeddings (dense search); "
               "TF-IDF key terms on the Lectures page", "retriever.py, study.py"),
    ("Unit 4", "Transformer sentence embeddings, plus chunk size, overlap and top-k experiments",
     "ingest.py, evaluate.py"),
    ("Unit 5", "RAG, a FAISS vector database, prompt engineering and Gemini; plain Gemini vs RAG "
               "comparison; generated summaries and quizzes", "rag.py, study.py"),
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
    notes = ", ".join(sorted({chunk["source"] for chunk in retriever.chunks if chunk["page"]}))
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

PAGES = {
    "ask": st.Page(ask_page, title="Ask", url_path="ask", default=True),
    "lectures": st.Page(lectures_page, title="Lectures", url_path="lectures"),
    "quiz": st.Page(quiz_page, title="Quiz", url_path="quiz"),
    "results": st.Page(results_page, title="Results", url_path="results"),
    "how": st.Page(how_page, title="How it works", url_path="how-it-works"),
}
st.navigation(list(PAGES.values()), position="top").run()
