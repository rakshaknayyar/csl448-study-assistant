"""
app.py - Step 4 of the CSL448 Study Assistant: the chat web page (Streamlit)

  - chat with history: ask questions about the course notes
  - every answer shows its sources (file + page); the "Sources" panel shows
    the exact chunk text that was given to Gemini
  - sidebar: choose the retriever (BM25 / Dense / Hybrid) and top-k,
    try example questions, and see the index details

Run from the project folder:
  venv\\Scripts\\python.exe -m streamlit run app.py
The page opens in your browser at http://localhost:8501
"""

import streamlit as st
from google.genai import errors

from rag import GEMINI_MODEL, NOT_FOUND, SIMILARITY_THRESHOLD, answer_question
from retriever import DEFAULT_TOP_K, Retriever

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
# Dropdown label -> retriever mode used by rag.py
RETRIEVERS = {
    "Hybrid (BM25 + Dense)": "hybrid",
    "Dense (embeddings + FAISS)": "dense",
    "BM25 (keywords)": "bm25",
}

# One-click questions for the demo. The last one is NOT in the notes,
# so it shows the "Not found in the notes" behaviour.
EXAMPLE_QUESTIONS = [
    "What is Naive Bayes?",
    "How does a decision tree choose the best split?",
    "What are some applications of text classification?",
    "What is word2vec?",
]

st.set_page_config(page_title="CSL448 Study Assistant", page_icon="📚")


@st.cache_resource(show_spinner="Loading the notes index and embedding model ...")
def load_retriever():
    """Load the FAISS index, chunks, BM25 and embedding model once for all users."""
    return Retriever()


# ----------------------------------------------------------------------------
# Showing one assistant answer
# ----------------------------------------------------------------------------
def show_answer(message, show_prompt):
    """Draw an answer, its sources line, and the Sources panel."""
    if message["found"]:
        st.markdown(message["answer"])
        pages = ", ".join(f"{s['source']} p.{s['page']}" for s in message["sources"])
        st.caption(f"📄 Sources: {pages}")
    else:
        st.warning(NOT_FOUND, icon="🔍")
        if message["reason"] == "low_similarity":
            st.caption(f"Best similarity {message['best_similarity']:.2f} is below the threshold "
                       f"{SIMILARITY_THRESHOLD}, so Gemini was not called.")
        else:
            st.caption("Gemini read the retrieved chunks and the answer is not in them.")

    st.caption(f"Retriever: {message['retriever']} · top-{message['k']} · "
               f"best similarity {message['best_similarity']:.2f}")

    # Sources panel: the exact chunks that were retrieved.
    chunks = message["chunks"]
    if chunks:
        title = "Sources" if message["found"] else "Closest chunks (not used for an answer)"
        with st.expander(f"{title} ({len(chunks)})"):
            for number, chunk in enumerate(chunks, start=1):
                st.markdown(f"**[{number}] {chunk['source']}, page {chunk['page']}**  \n"
                            f"score {chunk['score']:.3f} · similarity {chunk['similarity']:.3f}")
                st.code(chunk["text"], language=None, wrap_lines=True)

    # Optional: the full prompt, to show the prompt engineering (Unit 5).
    if show_prompt and message["prompt"]:
        with st.expander("Prompt sent to Gemini"):
            st.code(message["prompt"], language=None, wrap_lines=True)


# ----------------------------------------------------------------------------
# Page layout
# ----------------------------------------------------------------------------
try:
    retriever = load_retriever()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    retriever_label = st.selectbox("Retriever", list(RETRIEVERS),
                                   help="BM25 matches keywords, Dense matches meaning, Hybrid mixes both.")
    top_k = st.slider("Top-k chunks", min_value=1, max_value=10, value=DEFAULT_TOP_K,
                      help="How many chunks are retrieved and given to Gemini as context.")
    show_prompt = st.toggle("Show prompt sent to Gemini")

    st.divider()
    st.subheader("💡 Try an example")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, width="stretch"):
            st.session_state.pending_question = example

    st.divider()
    st.subheader("📊 About the index")
    info = retriever.info
    notes_files = sorted({chunk["source"] for chunk in retriever.chunks})
    st.caption(
        f"**Notes:** {', '.join(notes_files)}  \n"
        f"**Pages:** {info['num_pages']} · **Chunks:** {info['num_chunks']}  \n"
        f"**Chunk size:** {info['chunk_size']} chars, overlap {info['chunk_overlap']}  \n"
        f"**Embeddings:** {info['embedding_model'].split('/')[-1]} ({info['vector_dimension']}-d)  \n"
        f"**Vector store:** FAISS IndexFlatIP (cosine)  \n"
        f"**LLM:** {GEMINI_MODEL}  \n"
        f"**Not-found threshold:** similarity < {SIMILARITY_THRESHOLD}"
    )

    if st.button("🗑️ Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# ---- Main area ---------------------------------------------------------------
st.title("📚 CSL448 Study Assistant")
st.caption("Computational Linguistics & NLP · Answers come only from the course notes, "
           "with the file and page for every answer.")

with st.expander("How it works"):
    st.markdown(
        "1. **Retrieve:** your question is matched against the slide chunks with "
        "BM25 (keywords), dense embeddings + FAISS (meaning), or both (hybrid).\n"
        "2. **Check:** if even the best chunk is a weak match, the answer is "
        f"*{NOT_FOUND}* and Gemini is not called.\n"
        "3. **Generate:** otherwise the top chunks go into a prompt that tells Gemini "
        "to answer **only** from them.\n"
        "4. **Cite:** the answer is shown with the file and page of its sources."
    )

# Replay the chat history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            show_answer(message, show_prompt)

# A new question comes from the chat box or from an example button.
question = st.chat_input("Ask a question about the CSL448 notes ...")
question = question or st.session_state.pop("pending_question", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the notes ..."):
                result = answer_question(question, RETRIEVERS[retriever_label], top_k, retriever)
        except RuntimeError as error:          # GEMINI_API_KEY missing
            st.error(str(error))
            st.stop()
        except errors.APIError as error:       # Gemini failed even after retries
            st.error(f"Gemini error {error.code}: {error}. Please wait a minute and try again.")
            st.stop()

        message = {"role": "assistant", "retriever": retriever_label, "k": top_k, **result}
        show_answer(message, show_prompt)

    st.session_state.messages.append(message)
