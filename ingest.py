"""
ingest.py - Step 1 of the CSL448 Study Assistant (RAG pipeline)

What this file does:
  1. Reads every PDF in the notes/ folder page by page, keeping the file name
     and page number so every answer can cite its source.
  2. Splits each page's text into overlapping chunks
     (default 500 characters with 100 overlap - Unit 4: context optimization),
     plus one overview chunk per file (title, page count, outline) so that
     questions about the notes themselves can be answered.
  3. Turns each chunk into a sentence embedding with all-MiniLM-L6-v2
     (Unit 4: transformer-based embeddings from HuggingFace).
  4. Stores the vectors in a FAISS index (Unit 5: vector database).

Output files (in the index/ folder):
  faiss.index  - FAISS IndexFlatIP holding one normalized vector per chunk
  chunks.json  - the chunks (id, source, page, text), in the same order as the vectors
  info.json    - the settings used to build the index (model, chunk size, overlap)

How to run (from the project folder):
  venv\\Scripts\\python.exe ingest.py
  venv\\Scripts\\python.exe ingest.py --chunk-size 300 --chunk-overlap 50
"""

import argparse
import json
import unicodedata
from pathlib import Path

import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
NOTES_DIR = PROJECT_DIR / "notes"
INDEX_DIR = PROJECT_DIR / "index"

# Small, fast, free embedding model that runs locally (384-dimensional vectors).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The notes are PowerPoint slides, so each page is short. With 500 characters
# most slides fit in a single chunk; longer slides get split with overlap.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# PowerPoint draws bullets and math symbols with the "Symbol" and "Wingdings"
# fonts. pypdf reads those as private-use codes (U+E000 to U+F8FF) that display
# as empty boxes, so we map the common ones back to real characters.
SYMBOL_FONT_CHARS = {
    "": "•",   # Wingdings bullet
    "": "+",
    "": "-",
    "": "=",
    "": "α",
    "": "β",
    "": "×",
    "": "∏",
    "": "∑",
}


# ----------------------------------------------------------------------------
# 1. Load PDFs
# ----------------------------------------------------------------------------
def clean_text(text):
    """
    Tidy up text extracted from a slide:
      - turn PowerPoint symbol-font codes into real characters
      - drop any other private-use characters (unknown bullet icons)
      - NFKC normalization turns math italic letters like 𝑥 into a plain x,
        so a typed question can match formulas in the slides
      - squash extra spaces and drop blank lines (and lines that are only a bullet)
    """
    for code, char in SYMBOL_FONT_CHARS.items():
        text = text.replace(code, char)
    text = "".join(ch for ch in text if not "" <= ch <= "")
    text = unicodedata.normalize("NFKC", text)

    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line and line != "•")


def load_pages(notes_dir):
    """
    Read every PDF in notes_dir and return a list of pages:
        {"source": "file.pdf", "page": 3, "text": "..."}

    PDFs with no text at all (scanned or picture-only pages) are skipped with
    a warning, because pypdf can only read real, selectable text.
    """
    pdf_paths = sorted(notes_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {notes_dir}. Put your notes there and run again.")

    print(f"Reading {len(pdf_paths)} PDF file(s) from {notes_dir}")
    pages = []

    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            pdf_pages = []     # pages of this PDF that have text
            empty_pages = []   # page numbers with no text

            for page_number, page in enumerate(reader.pages, start=1):
                text = clean_text(page.extract_text() or "")
                if text:
                    pdf_pages.append({"source": pdf_path.name, "page": page_number, "text": text})
                else:
                    empty_pages.append(page_number)
        except Exception as error:
            print(f"  WARNING: could not read {pdf_path.name} ({error}). Skipped.")
            continue

        if not pdf_pages:
            # No page had any text -> almost certainly images of pages.
            print(f"  WARNING: {pdf_path.name} has NO selectable text on any of its "
                  f"{len(reader.pages)} pages.")
            print("           It looks image-only (scanned or exported as pictures). Skipped.")
            print("           Fix: export it again from PowerPoint with File > Save As > PDF.")
            continue

        print(f"  {pdf_path.name}: {len(reader.pages)} pages, {len(pdf_pages)} with text")
        if empty_pages:
            print(f"    note: page(s) {empty_pages} have no text (picture-only slides?) and were skipped.")
        pages.extend(pdf_pages)

    return pages


# ----------------------------------------------------------------------------
# 2. Split pages into chunks
# ----------------------------------------------------------------------------
def chunk_pages(pages, chunk_size, chunk_overlap):
    """
    Split each page into chunks of at most chunk_size characters.

    Each page is split on its own, so a chunk never mixes two slides and its
    page number is always exact. A slide shorter than chunk_size stays as one chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to break at paragraphs, then lines, then words.
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append({
                "id": len(chunks),        # position in the FAISS index
                "source": page["source"],
                "page": page["page"],
                "text": piece,
            })
    return chunks


# First lines that mark a slide as the table of contents.
OUTLINE_TITLES = {"outline", "contents", "table of contents", "agenda", "overview"}


def make_overview_chunks(pages, first_id):
    """
    Build one extra "overview" chunk per PDF with its title, length and outline.

    Slides never say "the topic of these notes is ...", so without this chunk
    questions about the notes themselves ("what is the topic?", "what does
    this cover?") match nothing. The title comes from the first page and the
    outline from the first slide titled Outline / Contents / Agenda.
    """
    pages_by_file = {}
    for page in pages:
        pages_by_file.setdefault(page["source"], []).append(page)

    overviews = []
    for source, file_pages in pages_by_file.items():
        title_page = file_pages[0]
        title = title_page["text"].splitlines()[0]
        outline = next((page for page in file_pages
                        if page["text"].splitlines()[0].strip().lower() in OUTLINE_TITLES), None)

        text = (f"Overview of the notes file {source}.\n"
                f"Title and main topic of these notes: {title}\n"
                f"Number of pages: {file_pages[-1]['page']}")
        if outline:
            topics = "\n".join(outline["text"].splitlines()[1:])
            text += f"\nTopics covered (outline on page {outline['page']}):\n{topics}"

        overviews.append({
            "id": first_id + len(overviews),
            "source": source,
            "page": title_page["page"],     # cite the title page
            "text": text,
        })
    return overviews


# ----------------------------------------------------------------------------
# 3. Embed chunks and build the FAISS index
# ----------------------------------------------------------------------------
def embed_texts(texts, model_name=EMBEDDING_MODEL):
    """
    Turn a list of texts into embedding vectors.

    normalize_embeddings=True makes every vector length 1, so the inner
    product used by IndexFlatIP equals cosine similarity.
    """
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.astype("float32")  # FAISS needs float32


def build_faiss_index(vectors):
    """Create an exact (brute-force) inner-product index and add all vectors."""
    dimension = vectors.shape[1]          # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index


# ----------------------------------------------------------------------------
# 4. Save everything
# ----------------------------------------------------------------------------
def save_index(index, chunks, info, index_dir):
    """Write faiss.index, chunks.json and info.json into index_dir."""
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))

    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    with open(index_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


# ----------------------------------------------------------------------------
# Full pipeline (evaluate.py will reuse this to compare chunk sizes)
# ----------------------------------------------------------------------------
def build_index(notes_dir=NOTES_DIR, index_dir=INDEX_DIR,
                chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Load PDFs -> chunk -> embed -> FAISS -> save. Returns the info dict."""
    if chunk_overlap >= chunk_size:
        raise SystemExit("chunk overlap must be smaller than chunk size.")

    pages = load_pages(Path(notes_dir))
    if not pages:
        raise SystemExit("No text found in any PDF, so no index was built. "
                         "See the warnings above.")

    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    print(f"\nSplit {len(pages)} pages into {len(chunks)} chunks "
          f"(chunk size {chunk_size}, overlap {chunk_overlap})")
    overviews = make_overview_chunks(pages, first_id=len(chunks))
    chunks += overviews
    print(f"Added {len(overviews)} overview chunk(s) with each file's title and outline")

    print(f"Embedding chunks with {EMBEDDING_MODEL} ...")
    vectors = embed_texts([chunk["text"] for chunk in chunks])
    index = build_faiss_index(vectors)

    info = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_pdfs": len({page["source"] for page in pages}),
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "vector_dimension": int(vectors.shape[1]),
    }
    save_index(index, chunks, info, Path(index_dir))
    return info


def main():
    parser = argparse.ArgumentParser(description="Build the FAISS index from the PDF notes.")
    parser.add_argument("--notes-dir", default=str(NOTES_DIR), help="folder with the PDF notes")
    parser.add_argument("--index-dir", default=str(INDEX_DIR), help="folder to save the index in")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    info = build_index(args.notes_dir, args.index_dir, args.chunk_size, args.chunk_overlap)

    # Read the saved chunks back to report their average length.
    with open(Path(args.index_dir) / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    avg_chars = sum(len(chunk["text"]) for chunk in chunks) / len(chunks)

    print("\nDone!")
    print(f"  PDFs indexed      : {info['num_pdfs']}")
    print(f"  Pages with text   : {info['num_pages']}")
    print(f"  Chunks            : {info['num_chunks']} (average {avg_chars:.0f} characters)")
    print(f"  Vector dimension  : {info['vector_dimension']}")
    print(f"  Saved to          : {args.index_dir}")


if __name__ == "__main__":
    main()
