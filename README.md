# CSL448 Study Assistant

A Retrieval-Augmented Generation (RAG) chatbot for **CSL448 – Computational Linguistics and Natural Language Processing**.
It answers questions from the course notes (PDFs in `notes/`), shows the **file and page** behind every answer,
and replies **"Not found in the notes"** instead of making something up.

```
question ──► retriever (BM25 / Dense / Hybrid) ──► top-k slide chunks
                                                        │
                       best similarity < 0.30? ── yes ──► "Not found in the notes"  (Gemini not called)
                                                        │ no
                                                        ▼
                         prompt: "answer ONLY from this context" ──► Gemini ──► answer + sources (file, page)
```

## Tech stack

| Part | Tool |
|---|---|
| PDF reading | `pypdf` |
| Chunking | `langchain-text-splitters` (RecursiveCharacterTextSplitter) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free, 384-d) |
| Vector store | FAISS `IndexFlatIP` on normalized vectors (= cosine similarity) |
| Keyword retrieval | `rank-bm25` (BM25Okapi) |
| Answer generation | Gemini `gemini-3.5-flash-lite` via the `google-genai` SDK |
| Web UI | Streamlit |
| Evaluation | pandas + matplotlib |

## How it maps to the syllabus

| Unit | Where to see it |
|---|---|
| **Unit 2:** preprocessing, BM25 | `retriever.py`: `preprocess()` (lowercase, accent removal, tokenization, stop-words) and `bm25_search()` |
| **Unit 3:** Bag of Words vs word embeddings | BM25 (term counts) vs dense embeddings, compared side by side in `retriever.py` and `evaluate.py` |
| **Unit 4:** transformer embeddings, context optimization | `ingest.py` (sentence-transformers); chunk size / overlap / top-k experiments in `evaluate.py` |
| **Unit 5:** RAG, vector database, prompt engineering | `rag.py` (prompt with numbered context, threshold, Gemini); FAISS index from `ingest.py` |
| **CO5:** apply and evaluate | `evaluate.py`: Recall@k, MRR, chunk-size comparison, answer accuracy judged by Gemini |

---

## Setup (Windows, one time)

All commands run in **PowerShell** from the project folder:

```powershell
cd "C:\Users\Rakshak Nayyar\Downloads\csl448-rag"
```

**1. Create the virtual environment** (Python 3.12 recommended, since faiss-cpu and torch install most reliably on it):

```powershell
py -3.12 -m venv venv
```

**2. Install the packages** into the venv (a few minutes; it downloads PyTorch):

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

**3. Set your Gemini API key.** Get one from [Google AI Studio](https://aistudio.google.com/apikey), then run:

```powershell
setx GEMINI_API_KEY "paste-your-key-here"
```

Close PowerShell and open a new window; only new windows see the key.
Never put the key in a `.py` file or commit it.

**4. Add your notes.** Put PDFs in `notes/`. They must have **selectable text**: export slides from
PowerPoint with *File → Save As → PDF*. Scanned or picture-only PDFs are skipped with a warning.

---

## How to run

Run these steps in order. Every command is run from the project folder.

### 1. Build the index (`ingest.py`)

```powershell
venv\Scripts\python.exe ingest.py
```

Reads the PDFs page by page, splits pages into chunks (default **500 characters, 100 overlap**), embeds them and
saves the FAISS index to `index/`. Run it again whenever you change the notes. Other chunk sizes:

```powershell
venv\Scripts\python.exe ingest.py --chunk-size 300 --chunk-overlap 60
```

### 2. Compare the retrievers (`retriever.py`, optional)

```powershell
venv\Scripts\python.exe retriever.py "What is the vector space model?" --k 5
```

Prints the BM25 keywords and the top chunks from BM25, Dense and Hybrid side by side. No Gemini needed.

### 3. Ask one question in the terminal (`rag.py`, optional)

```powershell
venv\Scripts\python.exe rag.py "What are the pros and cons of Naive Bayes?" --mode hybrid --k 3 --show-prompt
```

### 4. Start the chat app (`app.py`)

```powershell
venv\Scripts\python.exe -m streamlit run app.py
```

Opens http://localhost:8501. The first launch takes about 15 seconds while the embedding model loads.
If Streamlit asks for an email the first time, just press **Enter**. Stop it with **Ctrl+C**.

The app has a clean, Apple-style look and three pages in the top bar:

- **Ask:** the chat. Tap a suggested question or type your own. Each answer lists the pages it came from;
  **Show the slide text** opens the retrieved slides and marks the ones the answer used.
  **Search settings** picks the retriever (Hybrid, Meaning = dense, Keywords = BM25), how many slides to read
  (top-k), and can show the prompt sent to Gemini. *"What is word2vec?"* shows the "Not found in the notes" behaviour.
- **Results:** the evaluation numbers, charts and tables from `results/`.
- **How it works:** the four pipeline steps, where each syllabus unit shows up, and details of the index.

Colours, font and button shapes are set in `.streamlit/config.toml`.

### 5. Evaluate (`evaluate.py`)

First write `test_questions.json` (30–40 questions). Use the PDF page number; `null` means the topic is not in the notes:

```json
[
  {
    "question": "What are the pros and cons of Naive Bayes?",
    "expected_source": "text_classification.pdf",
    "expected_page": 25,
    "expected_answer": "Pros: very fast and easy to implement. Cons: seldom the best performance."
  },
  {
    "question": "What is word2vec?",
    "expected_source": null,
    "expected_page": null,
    "expected_answer": "Not found in the notes"
  }
]
```

`expected_page` can be a list such as `[21, 22]` when the answer spans slides. Avoid title-only slides as expected
pages: their content is a picture, so no retriever can find it.

Retrieval only (Recall@1/3/5 and MRR for BM25, Dense, Hybrid; **no Gemini**):

```powershell
venv\Scripts\python.exe evaluate.py
```

Also compare chunk sizes (**no Gemini**; builds extra indexes in `index/chunk_size_N/`):

```powershell
venv\Scripts\python.exe evaluate.py --chunk-sizes 300 500 800
```

Also judge full answers with Gemini (about 2 requests per question, one every 4 seconds):

```powershell
venv\Scripts\python.exe evaluate.py --answers
```

Use `--answer-mode dense --answer-k 5` to judge a different retriever or top-k.

Results are saved in `results/`:

| File | Contents |
|---|---|
| `retrieval_recall.csv` / `.png` | Recall@k and MRR per retriever |
| `retrieval_details.csv` | per question: expected pages, retrieved pages, rank of the first correct one |
| `chunk_size_recall.csv` / `.png` | the same scores for each chunk size |
| `answers_detail.csv` | per question: answer, verdict (correct / incorrect / missed / correct refusal / hallucination) |
| `answers_summary.csv` / `.png` | answer accuracy, correct refusals, overall |

---

## Project structure

```
csl448-rag/
├── notes/                 course PDFs (text-based, exported from PowerPoint)
├── ingest.py              step 1: PDFs -> chunks -> embeddings -> FAISS index
├── retriever.py           step 2: BM25, dense and hybrid search
├── rag.py                 step 3: threshold + prompt + Gemini -> answer with sources
├── app.py                 step 4: Streamlit app (Ask, Results, How it works)
├── .streamlit/config.toml theme for the app (colours, font, rounded buttons)
├── evaluate.py            step 5: Recall@k, MRR, chunk sizes, answer accuracy
├── test_questions.json    your test questions for evaluate.py
├── requirements.txt       Python packages
├── .gitignore             keeps venv/, index/ and .env out of git
├── README.md              this file
├── index/                 (generated by ingest.py, not in git)
│   ├── faiss.index        one normalized vector per chunk
│   ├── chunks.json        chunk text with file name and page number
│   ├── info.json          settings used to build the index
│   └── chunk_size_N/      extra indexes from evaluate.py --chunk-sizes
├── results/               (generated by evaluate.py: CSV tables and PNG charts for the report)
└── venv/                  (virtual environment, not in git)
```

## Settings you can change

| Setting | File | Default |
|---|---|---|
| `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP` | `ingest.py` | 500, 100 |
| `EMBEDDING_MODEL` | `ingest.py` | `sentence-transformers/all-MiniLM-L6-v2` |
| `DEFAULT_TOP_K` | `retriever.py` | 3 |
| `HYBRID_ALPHA` (weight of dense vs BM25) | `retriever.py` | 0.5 |
| `STOP_WORDS` | `retriever.py` | built-in English list |
| `GEMINI_MODEL` | `rag.py` | `gemini-3.5-flash-lite` (backup: `gemma-4-31b-it`) |
| `SIMILARITY_THRESHOLD` | `rag.py` | 0.30 |
| `GEMINI_PAUSE_SECONDS` | `evaluate.py` | 4 |

If you change the embedding model or chunk settings in `ingest.py`, run `ingest.py` again.

## Gemini free-tier limits

`gemini-3.5-flash-lite` allows **15 requests per minute** and **500 per day**.

- **Rate-limit retries:** on a rate-limit (429) or overload (503) error, `rag.py` waits 10 seconds and retries, up to 3 attempts in total.
- **Weak matches:** questions whose best match is below the similarity threshold never call Gemini.
- **Retrieval evaluation:** `evaluate.py` without `--answers` never calls Gemini.
- **Daily limit:** if the limit runs out, switch `GEMINI_MODEL` in `rag.py` to `gemma-4-31b-it`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No PDF files found in ...notes` | Put your PDFs in `notes/`. |
| `WARNING: ... has NO selectable text` | The PDF is pictures of pages. Export it again from PowerPoint with *File → Save As → PDF*. |
| `No index found ... Run ingest.py` | Run `venv\Scripts\python.exe ingest.py` first. |
| `GEMINI_API_KEY is not set` | Run the `setx` command from Setup step 3, then open a **new** PowerShell window. |
| `Gemini error 429` | Rate limit or daily quota. Wait a minute (or until tomorrow), or switch to the backup model. |
| `Warning: You are sending unauthenticated requests to the HF Hub` | Harmless. The embedding model is already downloaded. |
| `python` runs the wrong Python | Always use `venv\Scripts\python.exe ...`, or activate the venv first with `venv\Scripts\activate`. |
