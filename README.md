# CSL448 Study Assistant

A Retrieval-Augmented Generation (RAG) chatbot built for **CSL448 – Computational Linguistics and Natural Language Processing**.
It answers questions from your course notes (PDFs in `notes/`), shows the **file and page** behind every answer,
and replies **"Not found in the notes"** instead of making something up.

The notes in this project are **Environmental Studies (EVS)** lecture notes: 11 PDFs (L1–L11) on the environment,
sustainability, ocean acidification, *Silent Spring*, ecosystems, ecological pyramids and biodiversity. The NLP is in
the code, not the notes: drop any text-based PDFs into `notes/`, run `ingest.py`, and the same pipeline works for them.

```
question ──► retriever (BM25 / Dense / Hybrid) ──► top-k note chunks
                                                        │
                       best similarity < 0.20? ── yes ──► "Not found in the notes"  (Gemini not called)
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
| Web UI | Streamlit (+ Altair for the interactive charts) |
| Page previews | `pypdfium2` (renders PDF pages as pictures) |
| Evaluation | pandas + matplotlib |

## How it maps to the syllabus

| Unit | Where to see it |
|---|---|
| **Unit 2:** preprocessing, BM25 | `retriever.py`: `preprocess()` (lowercase, accent removal, tokenization, stop-words) and `bm25_search()` |
| **Unit 3:** Bag of Words / TF-IDF vs word embeddings | BM25 (term counts) vs dense embeddings, compared in `retriever.py` and `evaluate.py`; TF-IDF key terms for each lecture in `study.py` (Lectures page) |
| **Unit 4:** transformer embeddings, context optimization | `ingest.py` (sentence-transformers); chunk size / overlap / top-k experiments in `evaluate.py` |
| **Unit 5:** RAG, vector database, prompt engineering, LLMs | `rag.py` (prompt with numbered context, threshold, follow-ups, Gemini); "Compare with plain Gemini" shows the hallucinations RAG prevents; summaries and quizzes generated from the notes in `study.py` |
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

**4. Add your notes.** Put PDFs in `notes/`. They must have **selectable text** (you can highlight a
sentence with the mouse). Scanned or picture-only PDFs are skipped with a warning.

---

## How to run

Run these steps in order. Every command is run from the project folder.

### 1. Build the index (`ingest.py`)

```powershell
venv\Scripts\python.exe ingest.py
```

Reads the PDFs page by page, splits pages into chunks (default **1000 characters, 200 overlap**), embeds them and
saves the FAISS index to `index/`. It also adds one **overview** chunk per PDF (title, number of pages and the outline
page, if any) plus one overview listing every file's title, so questions about the notes themselves, such as *"What is the topic of these notes?"*, can be answered. Run it again whenever you change the notes. Other chunk sizes:

```powershell
venv\Scripts\python.exe ingest.py --chunk-size 500 --chunk-overlap 100
```

### 2. Compare the retrievers (`retriever.py`, optional)

```powershell
venv\Scripts\python.exe retriever.py "What is the pyramid of energy?" --k 5
```

Prints the BM25 keywords and the top chunks from BM25, Dense and Hybrid side by side. No Gemini needed.

### 3. Ask one question in the terminal (`rag.py`, optional)

```powershell
venv\Scripts\python.exe rag.py "Why is CNG called a green fuel?" --mode hybrid --k 5 --show-prompt
```

### 4. Start the chat app (`app.py`)

The easy way: **double-click `run_app.bat`** in the project folder. It uses the venv, reads your Gemini key from
your Windows settings (so it works even in a window opened before you ran `setx`), builds the index if it is missing,
and starts the app. Or from PowerShell:

```powershell
.\run_app.bat
```

Opens http://localhost:8501. The first launch takes about 15 seconds while the embedding model loads.
If Streamlit asks for an email the first time, just press **Enter**. Stop it with **Ctrl+C**.

After changing the notes or settings, **restart the app**: a running app keeps the old index in memory.

The app has a clean, Apple-style look and five pages in the top bar:

- **Ask:** the chat. Tap a suggested question or type your own.
  - Answers **type out live** (streaming) and show a **High / Medium / Low confidence** badge and the pages they came from.
  - **Follow-up questions** work ("explain it more simply"): the last two questions and answers go into the prompt,
    and a vague follow-up is searched together with the previous question.
  - **Show the source text** lists the retrieved passages with a **picture of the real PDF page**
    (click *Open page* to enlarge) and marks the passages the answer used.
  - **Search settings**: the retriever (Hybrid, Meaning = dense, Keywords = BM25), how many passages to read (top-k),
    **Compare with plain Gemini** (shows Gemini's answer *without* your notes next to the RAG answer), and the prompt.
  - **Download chat** saves the conversation as Markdown revision notes.
  - *"What is the Kyoto Protocol?"* (not in the notes) shows "Not found in the notes"; turn on the comparison
    to see plain Gemini answer it anyway from general knowledge.
- **Lectures:** a card for every lecture with its **TF-IDF key terms**, a **Summary** (written by Gemini from that
  lecture only, saved so it opens instantly next time), **Quiz me** and **Ask**.
- **Quiz:** Gemini writes 5 multiple-choice questions from one lecture; *Check answers* gives your score and, for each
  question, the explanation and the page to revise.
- **Results:** the evaluation numbers, interactive charts (hover for values) and tables from `results/`.
- **How it works:** the four pipeline steps, where each syllabus unit shows up, and details of the index.

Gemini requests per action: an answer 1 (2 with the comparison on), a summary 1 the first time only, a quiz 1.

Colours, font and button shapes are set in `.streamlit/config.toml`.

### 5. Evaluate (`evaluate.py`)

`test_questions.json` holds 43 EVS questions: 37 answerable from the notes and 6 that are not (3 off-topic, and 3
environmental topics the notes never mention: the Kyoto Protocol, acid rain and the Montreal Protocol). Use the PDF page
number; `null` means the topic is not in the notes:

```json
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
    "question": "What is the Kyoto Protocol?",
    "expected_source": null,
    "expected_page": null,
    "expected_answer": "Not found in the notes"
  }
]
```

`expected_page` can be a list such as `[1, 2]` when the answer spans pages, and `expected` (a list of file + page)
is for answers that appear in more than one file.

Retrieval only (Recall@1/3/5 and MRR for BM25, Dense, Hybrid; **no Gemini**):

```powershell
venv\Scripts\python.exe evaluate.py
```

Also compare chunk sizes (**no Gemini**; builds extra indexes in `index/chunk_size_N/`):

```powershell
venv\Scripts\python.exe evaluate.py --chunk-sizes 500 800 1000 1500
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
├── notes/                 course PDFs with selectable text (EVS notes L1–L11)
├── ingest.py              step 1: PDFs -> chunks -> embeddings -> FAISS index
├── retriever.py           step 2: BM25, dense and hybrid search
├── rag.py                 step 3: threshold + prompt + Gemini -> answer with sources
├── app.py                 step 4: Streamlit app (Ask, Lectures, Quiz, Results, How it works)
├── study.py               lectures, TF-IDF key terms, summaries, quizzes, page pictures
├── run_app.bat            double-click to start the app
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
│   ├── summaries.json     saved lecture summaries (made by the app)
│   ├── page_images/       saved page pictures for previews (made by the app)
│   └── chunk_size_N/      extra indexes from evaluate.py --chunk-sizes
├── results/               (generated by evaluate.py: CSV tables and PNG charts for the report)
└── venv/                  (virtual environment, not in git)
```

## Settings you can change

| Setting | File | Default |
|---|---|---|
| `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP` | `ingest.py` | 1000, 200 |
| `EMBEDDING_MODEL` | `ingest.py` | `sentence-transformers/all-MiniLM-L6-v2` |
| `DEFAULT_TOP_K` | `retriever.py` | 5 |
| `HYBRID_ALPHA` (weight of dense vs BM25) | `retriever.py` | 0.5 |
| `STOP_WORDS` | `retriever.py` | built-in English list |
| `GEMINI_MODEL` | `rag.py` | `gemini-3.5-flash-lite` (backup: `gemma-4-31b-it`) |
| `SIMILARITY_THRESHOLD` | `rag.py` | 0.20 |
| `GEMINI_PAUSE_SECONDS` | `evaluate.py` | 4 |

If you change the embedding model or chunk settings in `ingest.py`, run `ingest.py` again.

## Tuning for the EVS notes (context optimization, Unit 4)

The first EVS run used settings chosen for short PowerPoint slides and got only **84%** of answers right, although the
right *page* was retrieved for 92% of questions. The cause: the EVS PDFs have tables and side boxes, so their text comes
out in a mixed-up order, and on dense pages (about 2,500 characters each) 500-character chunks separated a heading
from its list. Gemini got the right page but the wrong piece of it, and correctly said "Not found in the notes".

Checking whether the missing facts reached Gemini's context (no Gemini needed) for the 6 failed questions:

| Chunk size | Top-3 | Top-5 |
|---|---|---|
| 500 | 0 / 6 | 3 / 6 |
| 800 | 1 / 6 | 4 / 6 |
| **1000** | 2 / 6 | **5 / 6** |
| 1500 | 3 / 6 | 5 / 6 (but 44% more text to read) |

So the defaults became **1000 characters, 200 overlap, top-5**. Bigger chunks also lower similarity scores, so the
"Not found" cut-off moved from 0.30 to **0.20** (a real question, Lindeman's law, scored 0.26).

| Same 43 questions | Before | After |
|---|---|---|
| Answers correct (37 in the notes) | 84% | **97%** |
| Correct refusals (6 not in the notes) | 100% | **100%** |
| Hybrid Recall@3 | 92% | **97%** |

Off-topic questions are stopped two ways: unrelated ones (NLP, the Montreal Protocol) fall below the cut-off and never
reach Gemini, while environmental topics the notes don't cover (the Kyoto Protocol, acid rain) pass the cut-off but are
refused by the prompt rule.

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
| `WARNING: ... has NO selectable text` | The PDF is pictures of pages (a scan). Use a version with selectable text, e.g. export it again as PDF from Word or PowerPoint. |
| `No index found ... Run ingest.py` | Run `venv\Scripts\python.exe ingest.py` first. |
| `GEMINI_API_KEY is not set` | Run the `setx` command from Setup step 3, then open a **new** PowerShell window. |
| `Gemini error 429` | Rate limit or daily quota. Wait a minute (or until tomorrow), or switch to the backup model. |
| `Warning: You are sending unauthenticated requests to the HF Hub` | Harmless. The embedding model is already downloaded. |
| `python` runs the wrong Python | Always use `venv\Scripts\python.exe ...`, or activate the venv first with `venv\Scripts\activate`. |
