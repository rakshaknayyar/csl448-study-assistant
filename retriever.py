"""
retriever.py - Step 2 of the CSL448 Study Assistant (RAG pipeline)

Three ways to find the chunks that best match a question:
  bm25_search   - keyword retrieval based on term counts (Unit 2: preprocessing + BM25)
  dense_search  - meaning-based retrieval with sentence embeddings and FAISS
                  (Unit 4: transformer embeddings, Unit 5: vector database)
  hybrid_search - mixes both: normalized dense score + normalized BM25 score

Every search returns a list of chunks, best match first:
  {"id", "source", "page", "text", "score", "similarity"}
  score      - the score this retriever ranked by (cosine, BM25 or hybrid)
  similarity - cosine similarity between the question and the chunk.
               It means the same thing for all three retrievers, so rag.py can
               use one threshold to decide "Not found in the notes".

Needs the index from ingest.py. Try it from the project folder:
  venv\\Scripts\\python.exe retriever.py "What is Naive Bayes?"
  venv\\Scripts\\python.exe retriever.py "What is Naive Bayes?" --k 5
"""

import argparse
import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
INDEX_DIR = PROJECT_DIR / "index"

# 5 chunks of about 1000 characters give Gemini enough context without
# flooding it (measured on the EVS notes, see README).
DEFAULT_TOP_K = 5

# Hybrid score = HYBRID_ALPHA * dense + (1 - HYBRID_ALPHA) * BM25
# 0.5 gives both retrievers equal weight.
HYBRID_ALPHA = 0.5

SEARCH_MODES = ["bm25", "dense", "hybrid"]

# Common English words that carry little meaning for keyword search.
# Question words (what, explain, define...) are included because every
# question has them, so they would only add noise to BM25.
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so", "than",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "about", "as",
    "into", "over", "under", "between", "through", "during", "before", "after",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "done", "doing", "have", "has", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "him", "her",
    "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "there", "here", "which", "who", "whom", "whose", "what", "when", "where",
    "why", "how", "all", "any", "both", "each", "some", "such", "no", "nor",
    "not", "only", "own", "same", "too", "very", "just", "also", "more", "most",
    "other", "up", "down", "out", "off", "again", "further", "once",
    "explain", "define", "describe", "meant", "mean", "means", "tell", "give",
    "please", "briefly", "notes", "use", "used", "using",
}


# ----------------------------------------------------------------------------
# Preprocessing for BM25 (Unit 2)
# ----------------------------------------------------------------------------
def preprocess(text):
    """
    Turn text into a list of keyword tokens:
      1. lowercase
      2. remove accents, so "Naïve" in the slides matches a typed "naive"
      3. tokenize: keep runs of letters and digits
      4. remove stop-words
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    tokens = re.findall(r"[a-z0-9]+", text)
    return [token for token in tokens if token not in STOP_WORDS]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def load_embedding_model(model_name):
    """Load the embedding model once and reuse it (loading takes a few seconds)."""
    return SentenceTransformer(model_name)


def top_k_ids(scores, k):
    """Positions of the k highest scores, best first."""
    return np.argsort(-scores, kind="stable")[:k].tolist()


def min_max_normalize(scores):
    """Rescale scores to the range 0..1 so dense and BM25 scores can be added."""
    low, high = scores.min(), scores.max()
    if high - low < 1e-9:              # all scores equal -> no information
        return np.zeros_like(scores, dtype="float32")
    return (scores - low) / (high - low)


# ----------------------------------------------------------------------------
# The retriever
# ----------------------------------------------------------------------------
class Retriever:
    """
    Loads one index folder (made by ingest.py) and searches it.

    evaluate.py can create several Retrievers, one per chunk size,
    by passing different index folders.
    """

    def __init__(self, index_dir=INDEX_DIR):
        index_dir = Path(index_dir)
        if not (index_dir / "faiss.index").exists():
            raise FileNotFoundError(
                f"No index found in {index_dir}. "
                "Build it first with: venv\\Scripts\\python.exe ingest.py")

        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        with open(index_dir / "info.json", encoding="utf-8") as f:
            self.info = json.load(f)

        # Use the same embedding model that built the index.
        self.model = load_embedding_model(self.info["embedding_model"])

        # BM25 keeps its own word statistics, built from the preprocessed chunks.
        self.bm25 = BM25Okapi([preprocess(chunk["text"]) for chunk in self.chunks])

    # ---- building blocks ----------------------------------------------------
    def embed_query(self, query):
        """Embed the question the same way the chunks were embedded (normalized)."""
        vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return vector.astype("float32")

    def all_similarities(self, query):
        """Cosine similarity between the question and every chunk (indexed by chunk id)."""
        scores, ids = self.index.search(self.embed_query(query), self.index.ntotal)
        similarities = np.zeros(self.index.ntotal, dtype="float32")
        similarities[ids[0]] = scores[0]
        return similarities

    def make_results(self, ids, scores, similarities):
        """Turn chunk ids and scores into the result dicts described at the top."""
        results = []
        for chunk_id, score in zip(ids, scores):
            chunk = self.chunks[chunk_id]
            results.append({
                "id": chunk["id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(score),
                "similarity": float(similarities[chunk_id]),
            })
        return results

    # ---- the three retrievers ---------------------------------------------
    def dense_search(self, query, k=DEFAULT_TOP_K):
        """Semantic search: FAISS finds the k chunks with the highest cosine similarity."""
        k = min(k, self.index.ntotal)   # FAISS pads with id -1 if k is too big
        scores, ids = self.index.search(self.embed_query(query), k)
        ids, scores = ids[0].tolist(), scores[0]
        # For dense search the ranking score IS the cosine similarity.
        similarities = dict(zip(ids, scores))
        return self.make_results(ids, scores, similarities)

    def bm25_search(self, query, k=DEFAULT_TOP_K):
        """Keyword search: BM25 scores chunks by matching words (term counts)."""
        scores = self.bm25.get_scores(preprocess(query))
        # Drop chunks with score 0: they share no keyword with the question.
        ids = [i for i in top_k_ids(scores, k) if scores[i] > 0]
        return self.make_results(ids, scores[ids], self.all_similarities(query))

    def hybrid_search(self, query, k=DEFAULT_TOP_K, alpha=HYBRID_ALPHA):
        """Hybrid search: normalize both scores to 0..1 and take a weighted sum."""
        similarities = self.all_similarities(query)
        bm25_scores = self.bm25.get_scores(preprocess(query))
        combined = (alpha * min_max_normalize(similarities)
                    + (1 - alpha) * min_max_normalize(bm25_scores))
        ids = top_k_ids(combined, k)
        return self.make_results(ids, combined[ids], similarities)

    def search(self, query, mode="hybrid", k=DEFAULT_TOP_K):
        """Run the retriever named by mode: "bm25", "dense" or "hybrid"."""
        if mode == "bm25":
            return self.bm25_search(query, k)
        if mode == "dense":
            return self.dense_search(query, k)
        if mode == "hybrid":
            return self.hybrid_search(query, k)
        raise ValueError(f"Unknown mode {mode!r}. Choose one of {SEARCH_MODES}.")


# ----------------------------------------------------------------------------
# Simple functions using the default index/ folder
# ----------------------------------------------------------------------------
_default_retriever = None


def get_retriever():
    """Create the default Retriever the first time it is needed, then reuse it."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever


def dense_search(query, k=DEFAULT_TOP_K):
    return get_retriever().dense_search(query, k)


def bm25_search(query, k=DEFAULT_TOP_K):
    return get_retriever().bm25_search(query, k)


def hybrid_search(query, k=DEFAULT_TOP_K):
    return get_retriever().hybrid_search(query, k)


def search(query, mode="hybrid", k=DEFAULT_TOP_K):
    return get_retriever().search(query, mode, k)


# ----------------------------------------------------------------------------
# Command-line demo: compare the three retrievers on one question
# ----------------------------------------------------------------------------
def main():
    # Slides contain symbols like ∑ that the Windows console cannot print;
    # replace them with "?" instead of crashing.
    sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Compare BM25, dense and hybrid retrieval.")
    parser.add_argument("question", help="the question to search for, in quotes")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="number of chunks to return")
    args = parser.parse_args()

    print(f"Question       : {args.question}")
    print(f"BM25 keywords  : {preprocess(args.question)}")

    for mode in SEARCH_MODES:
        print(f"\n=== {mode.upper()} (top {args.k}) ===")
        results = search(args.question, mode, args.k)
        if not results:
            print("  (no matching chunks)")
        for rank, result in enumerate(results, start=1):
            preview = result["text"].replace("\n", " ")[:70]
            print(f"  {rank}. {result['source']} p.{result['page']:<3} "
                  f"score {result['score']:.3f} | similarity {result['similarity']:.3f} | {preview}")


if __name__ == "__main__":
    main()
