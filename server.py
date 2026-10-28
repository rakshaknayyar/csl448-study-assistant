"""
server.py - the web server for the CSL448 Study Assistant (FastAPI)

The React app in web/ is the screen; this file is the bridge to the Python
code that does the work (retriever.py, rag.py, study.py). It offers a small
JSON API and also serves the built React app, so one command runs everything:

  venv\\Scripts\\python.exe server.py        (or double-click run_app.bat)

then open http://localhost:8501 (it opens by itself).

API (all under /api):
  GET  /info                     lecture count, model, defaults, suggestions, index details
  POST /ask                      answer a question; streams events, one JSON object per line:
                                   {"type": "retrieved", "chunks": [...], "best_similarity": ...}
                                   {"type": "token", "text": "..."}        (the answer, as it is written)
                                   {"type": "answer", ...final result...}
                                   {"type": "plain", "text": "..."}        (only with compare on)
                                   {"type": "error", "message": "..."}
  GET  /lectures                 lecture cards (title, pages, TF-IDF key terms)
  POST /lectures/{file}/summary  a revision summary of one lecture
  POST /quiz                     multiple-choice questions from one or more lectures
                                 ({"sources": ["L3.pdf", "L5.pdf"], "questions": 10})
  GET  /page/{file}/{page}       a PNG picture of one PDF page
  GET  /results                  the evaluation tables from results/
"""

import json
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.genai import errors
from pydantic import BaseModel

import study
from rag import (GEMINI_MODEL, SIMILARITY_THRESHOLD, confidence_level, finish_answer,
                 generate_answer_stream, plain_answer, prepare_answer)
from retriever import DEFAULT_TOP_K, SEARCH_MODES, Retriever

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
WEB_DIR = PROJECT_DIR / "web" / "dist"      # the built React app (npm run build)
PORT = 8501

# One-tap questions on the empty Ask screen. The last one is NOT in the notes,
# so it shows the "Not found in the notes" behaviour.
SUGGESTIONS = [
    "What is ocean acidification?",
    "What is Lindeman's ten percent law?",
    "Who wrote Silent Spring and why?",
    "What is the Kyoto Protocol?",
]

QUIZ_LENGTH = 5            # default number of quiz questions
MAX_QUIZ_QUESTIONS = 15

# Loaded once when the server starts (the embedding model takes a few seconds).
state = {}


@asynccontextmanager
async def lifespan(app):
    print("Loading the index, the embedding model and your lectures ...")
    state["retriever"] = Retriever()
    state["lectures"] = {lecture["source"]: lecture for lecture in study.load_lectures()}
    print(f"\n  CSL448 Study Assistant is ready at http://localhost:{PORT}\n"
          "  Keep this window open while you use it. Press Ctrl+C to stop.\n")
    yield


app = FastAPI(title="CSL448 Study Assistant", lifespan=lifespan)


def gemini_error_message(error):
    """A plain-language message for any failure while talking to Gemini."""
    if isinstance(error, RuntimeError):            # GEMINI_API_KEY missing
        return str(error)
    if isinstance(error, errors.APIError):         # still failing after retries
        return f"Gemini didn't answer (error {error.code}). Wait a minute, then try again."
    return "Gemini's reply couldn't be used this time. Try again."


def get_lecture(source):
    lecture = state["lectures"].get(source)
    if lecture is None:
        raise HTTPException(404, f"No lecture called {source}.")
    return lecture


# ----------------------------------------------------------------------------
# Info
# ----------------------------------------------------------------------------
@app.get("/api/info")
def info():
    lectures = list(state["lectures"].values())
    return {
        "lectures": len(lectures),
        "pages": sum(len(lecture["pages"]) for lecture in lectures),
        "model": GEMINI_MODEL,
        "threshold": SIMILARITY_THRESHOLD,
        "default_top_k": DEFAULT_TOP_K,
        "suggestions": SUGGESTIONS,
        "index": state["retriever"].info,
        "notes": [lecture["source"] for lecture in lectures],
    }


# ----------------------------------------------------------------------------
# Ask (streamed)
# ----------------------------------------------------------------------------
class Turn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str
    mode: str = "hybrid"
    k: int = DEFAULT_TOP_K
    history: list[Turn] = []
    compare: bool = False


def ask_events(request):
    """The answer as a stream of JSON lines (see the list at the top of this file)."""
    def line(event):
        return json.dumps(event, ensure_ascii=False) + "\n"

    try:
        history = [turn.model_dump() for turn in request.history]
        result = prepare_answer(request.question, request.mode, request.k, state["retriever"], history)
        yield line({"type": "retrieved", "chunks": result["chunks"],
                    "best_similarity": result["best_similarity"]})

        if result["prompt"]:
            text = ""
            for piece in generate_answer_stream(result["prompt"]):
                text += piece
                yield line({"type": "token", "text": piece})
            result = finish_answer(result, text)

        yield line({"type": "answer", "answer": result["answer"], "found": result["found"],
                    "reason": result["reason"], "sources": result["sources"],
                    "cited_ids": result["cited_ids"], "best_similarity": result["best_similarity"],
                    "confidence": confidence_level(result), "prompt": result["prompt"]})

        if request.compare:
            yield line({"type": "plain", "text": plain_answer(request.question)})
    except (RuntimeError, errors.APIError) as error:
        yield line({"type": "error", "message": gemini_error_message(error)})


@app.post("/api/ask")
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(400, "The question is empty.")
    if request.mode not in SEARCH_MODES:
        raise HTTPException(400, f"mode must be one of {SEARCH_MODES}.")
    request.k = max(1, min(request.k, 10))
    return StreamingResponse(ask_events(request), media_type="application/x-ndjson")


# ----------------------------------------------------------------------------
# Lectures, summaries, quizzes, page pictures
# ----------------------------------------------------------------------------
@app.get("/api/lectures")
def lectures():
    return [{"source": lecture["source"], "label": lecture["label"], "title": lecture["title"],
             "pages": len(lecture["pages"]), "key_terms": lecture["key_terms"]}
            for lecture in state["lectures"].values()]


@app.post("/api/lectures/{source}/summary")
def summary(source: str):
    try:
        return {"summary": study.summarize(get_lecture(source))}
    except (RuntimeError, errors.APIError) as error:
        raise HTTPException(502, gemini_error_message(error))


class QuizRequest(BaseModel):
    sources: list[str]
    questions: int = QUIZ_LENGTH


@app.post("/api/quiz")
def quiz(request: QuizRequest):
    if not request.sources:
        raise HTTPException(400, "Pick at least one lecture.")
    lectures = [get_lecture(source) for source in dict.fromkeys(request.sources)]   # no duplicates
    count = max(1, min(request.questions, MAX_QUIZ_QUESTIONS))
    try:
        return {"questions": study.make_quiz(lectures, count)}
    except (RuntimeError, errors.APIError, ValueError) as error:
        raise HTTPException(502, gemini_error_message(error))


@app.get("/api/page/{source}/{page}")
def page_picture(source: str, page: int):
    lecture = get_lecture(source)             # only files from notes/ can be opened
    if page not in {p["page"] for p in lecture["pages"]}:
        raise HTTPException(404, f"{source} has no page {page}.")
    return FileResponse(study.page_image(source, page), media_type="image/png")


# ----------------------------------------------------------------------------
# Evaluation results
# ----------------------------------------------------------------------------
def read_result(name):
    """A results CSV as a list of rows; scores rounded to whole percents once, here."""
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    scores = [c for c in df.columns if c.startswith("Recall@")]
    df[scores] = df[scores].round(2)
    df = df.astype(object).where(pd.notna(df), None)      # NaN -> null in JSON
    return df.to_dict(orient="records")


@app.get("/api/results")
def results():
    return {"recall": read_result("retrieval_recall.csv"),
            "chunk_sizes": read_result("chunk_size_recall.csv"),
            "answers": read_result("answers_detail.csv")}


# ----------------------------------------------------------------------------
# The React app (built into web/dist with: cd web, npm run build)
# ----------------------------------------------------------------------------
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
else:
    @app.get("/")
    def no_frontend():
        return {"message": "The React app is not built yet. Run: cd web, npm install, npm run build"}


if __name__ == "__main__":
    # Open the browser a moment after the server starts.
    threading.Timer(6, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
