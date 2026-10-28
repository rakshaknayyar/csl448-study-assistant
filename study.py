"""
study.py - study tools for the CSL448 Study Assistant app

  load_lectures() - one entry per PDF in notes/: title, pages, full text and
                    its key terms, picked with TF-IDF (Unit 3)
  summarize()     - a short revision summary of one lecture (Gemini, cached)
  make_quiz()     - multiple-choice questions from one lecture (Gemini)
  page_image()    - a picture of one PDF page, for the source previews

Summaries and page pictures are saved in the index/ folder, so each one is
made only once (this saves Gemini requests and time).
"""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium

from ingest import INDEX_DIR, NOTES_DIR, load_pages
from rag import generate_answer
from retriever import preprocess

SUMMARIES_FILE = INDEX_DIR / "summaries.json"
PAGE_IMAGES_DIR = INDEX_DIR / "page_images"

KEY_TERMS_PER_LECTURE = 6

# Words that are common in lecture notes but say nothing about the topic.
NOTES_STOP_WORDS = {"unit", "page", "pages", "definition", "example", "examples", "introduction",
                    "include", "includes", "including", "known", "called", "like", "per", "via",
                    "level", "levels", "type", "types", "high", "low", "major", "main", "key"}


# ----------------------------------------------------------------------------
# Lectures and TF-IDF key terms (Unit 3)
# ----------------------------------------------------------------------------
def lecture_sort_key(lecture):
    """Sort L1, L2, ..., L7, L7-2, ..., L11 by the number in the file name."""
    numbers = [int(n) for n in re.findall(r"\d+", Path(lecture["source"]).stem)]
    return numbers or [999], lecture["source"]


def nice_title(first_line):
    """'L7 PART-2 ECOLOGICAL PYRAMIDS' -> ('L7', 'Part-2 Ecological Pyramids')."""
    match = re.match(r"^(L\d+)\s+(.*)$", first_line.strip())
    label, title = (match.group(1), match.group(2)) if match else ("", first_line.strip())
    if title.isupper():
        title = title.title()
    return label, title


def terms_of(text):
    """Single words and two-word phrases (after preprocessing) used for TF-IDF."""
    tokens = [t for t in preprocess(text)
              if len(t) > 3 and not t.isdigit() and t not in NOTES_STOP_WORDS]
    bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a != b]
    return tokens + bigrams


def same_term(a, b):
    """True for the same word, ignoring a plural 's' ("forest" and "forests")."""
    return a == b or a + "s" == b or b + "s" == a


def add_key_terms(lectures, top_n=KEY_TERMS_PER_LECTURE):
    """
    TF-IDF over the lectures: a term scores high when it is frequent in one
    lecture (TF) but appears in few other lectures (IDF).
      tf  = count of the term in the lecture / number of terms in the lecture
      idf = log(N / number of lectures containing the term) + 1
    Two-word phrases count double, and a term is skipped if it shares a word
    with one already chosen, so we get "ocean acidification" rather than
    "ocean" plus "acidification". Words must appear at least twice and phrases
    at least three times, which filters out word pairs that only meet because
    the PDF's tables come out in a jumbled order.
    """
    counts = [Counter(terms_of(lecture["text"])) for lecture in lectures]
    document_frequency = Counter(term for count in counts for term in count)
    n = len(lectures)

    for lecture, count in zip(lectures, counts):
        total = sum(count.values()) or 1
        scores = {term: (c / total) * (math.log(n / document_frequency[term]) + 1)
                  for term, c in count.items() if c >= (3 if " " in term else 2)}
        # Phrases count double, so a real phrase outranks the single words inside it.
        ranked = sorted(scores, key=lambda t: scores[t] * (2 if " " in t else 1), reverse=True)
        chosen = []
        for term in ranked:
            words = term.split()
            if len(words) == 2:
                # A phrase replaces single words it contains ("ocean" -> "ocean acidification") ...
                chosen = [other for other in chosen
                          if " " in other or not any(same_term(other, word) for word in words)]
                # ... but is skipped if it overlaps a phrase already chosen.
                phrase_words = [word for other in chosen if " " in other for word in other.split()]
                if any(same_term(word, used) for word in words for used in phrase_words):
                    continue
            elif any(same_term(term, used) for other in chosen for used in other.split()):
                continue                      # a word already chosen, alone or in a phrase
            chosen.append(term)
            if len(chosen) == top_n:
                break
        lecture["key_terms"] = chosen


def load_lectures(notes_dir=NOTES_DIR):
    """
    One dict per PDF: source, label (e.g. "L3"), title, pages (list of
    {"page", "text"}), text (all pages joined) and key_terms.
    """
    # Same reader as ingest.py, so this is exactly the text that was indexed
    # (image-only PDFs are skipped there too).
    pages_by_file = {}
    for page in load_pages(Path(notes_dir)):
        pages_by_file.setdefault(page["source"], []).append({"page": page["page"], "text": page["text"]})

    lectures = []
    for source, pages in pages_by_file.items():
        label, title = nice_title(pages[0]["text"].splitlines()[0])
        lectures.append({
            "source": source,
            "label": label,
            "title": title,
            "pages": pages,
            "text": "\n".join(page["text"] for page in pages),
        })
    lectures.sort(key=lecture_sort_key)
    add_key_terms(lectures)
    return lectures


def lecture_text_with_pages(lecture):
    """The lecture's text with [Page n] markers, so Gemini can cite pages."""
    return "\n\n".join(f"[Page {page['page']}]\n{page['text']}" for page in lecture["pages"])


# ----------------------------------------------------------------------------
# Summaries (Gemini, saved in index/summaries.json)
# ----------------------------------------------------------------------------
def summarize(lecture):
    """A short revision summary of one lecture, made once and then reused."""
    fingerprint = hashlib.md5(lecture["text"].encode("utf-8")).hexdigest()
    cache = json.loads(SUMMARIES_FILE.read_text(encoding="utf-8")) if SUMMARIES_FILE.exists() else {}
    saved = cache.get(lecture["source"])
    if saved and saved["fingerprint"] == fingerprint:
        return saved["summary"]

    prompt = f"""You are helping a student revise. Using ONLY the lecture notes below, write a revision summary:
- 5 to 7 short bullet points covering the most important ideas, definitions and numbers
- then one line starting with "Remember:" giving the single most important takeaway
Do not add anything that is not in the notes.

Lecture notes ({lecture['label']} {lecture['title']}):
{lecture_text_with_pages(lecture)}"""
    summary = generate_answer(prompt)

    cache[lecture["source"]] = {"fingerprint": fingerprint, "summary": summary}
    SUMMARIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARIES_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------
# Quiz (Gemini)
# ----------------------------------------------------------------------------
def make_quiz(lecture, number_of_questions=5):
    """
    Multiple-choice questions from one lecture. Returns a list of
    {"question", "options" (4), "answer" (index 0-3), "explanation", "page"}.
    Raises ValueError if Gemini's reply is not a usable quiz.
    """
    prompt = f"""You are writing a revision quiz for a student, using ONLY the lecture notes below.
Write {number_of_questions} multiple-choice questions that test the most important facts in the notes.
Each question has exactly 4 options and exactly one correct answer. Use only facts stated in the notes,
and make the wrong options believable.

Reply with ONLY a JSON list, no other text, in exactly this format:
[{{"question": "...", "options": ["...", "...", "...", "..."], "answer": 0,
  "explanation": "one short sentence, from the notes, saying why this is correct", "page": 1}}]
"answer" is the position (0 to 3) of the correct option. "page" is the [Page n] the fact comes from.

Lecture notes ({lecture['label']} {lecture['title']}):
{lecture_text_with_pages(lecture)}"""
    reply = generate_answer(prompt)

    # Take the JSON list out of the reply, even if Gemini wrapped it in ```json fences.
    start, end = reply.find("["), reply.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("Gemini did not return a quiz.")
    questions = json.loads(reply[start:end + 1])

    quiz = []
    for item in questions:
        options = item.get("options", [])
        answer = item.get("answer")
        if item.get("question") and len(options) == 4 and isinstance(answer, int) and 0 <= answer < 4:
            quiz.append({"question": item["question"], "options": [str(o) for o in options],
                         "answer": answer, "explanation": item.get("explanation", ""),
                         "page": item.get("page")})
    if not quiz:
        raise ValueError("Gemini's quiz had no usable questions.")
    return quiz


# ----------------------------------------------------------------------------
# Page pictures for source previews
# ----------------------------------------------------------------------------
def page_image(source, page, notes_dir=NOTES_DIR):
    """Path to a PNG picture of one page of a PDF (made once, then reused)."""
    PAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = PAGE_IMAGES_DIR / f"{Path(source).stem}_p{page}.png"
    if not image_path.exists():
        document = pdfium.PdfDocument(Path(notes_dir) / source)
        document[page - 1].render(scale=1.6).to_pil().save(image_path)
    return image_path
