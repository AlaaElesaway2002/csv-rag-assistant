"""
04_vector_representation.py
----------------------------
Stage 4 of the RAG pipeline: turning chunk text into searchable vectors
(TF-IDF, BM25, and dense embeddings).

Moved from rag_engine.py, unchanged, per the refactor mapping:
    build_tfidf, build_bm25, load_embedding_model, embed_chunks

Note on imports: file names in this project start with a digit
(01_documents.py, 02_preprocessing.py, ...), which is required by the
assignment's file-structure spec but is NOT a valid Python identifier, so a
plain `from 02_preprocessing import normalize_lexical_text` would raise a
SyntaxError. We use importlib.import_module("02_preprocessing") instead,
which imports by file name (a string) rather than by identifier and works
fine at runtime. See README.md for details.
"""

import importlib
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

_preprocessing = importlib.import_module("02_preprocessing")
normalize_lexical_text = _preprocessing.normalize_lexical_text

# --------------------------------------------------------------------------- #
# 4. Lexical retrievers (TF-IDF + BM25) -- vector/index construction
# --------------------------------------------------------------------------- #

def build_tfidf(chunks_df: pd.DataFrame):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks_df["search_text"].map(normalize_lexical_text))
    return vectorizer, matrix


def build_bm25(chunks_df: pd.DataFrame) -> BM25Okapi:
    def simple_tokenize(text):
        return re.findall(r"[a-z0-9]+", text.lower())
    tokenized = [simple_tokenize(t) for t in chunks_df["search_text"]]
    return BM25Okapi(tokenized)


# --------------------------------------------------------------------------- #
# 5. Embedding model + embedding construction
# --------------------------------------------------------------------------- #

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2", timeout_seconds: int = 20):
    """Returns (model, error). error is None on success, else a human-readable
    string explaining why embeddings/hybrid are unavailable (e.g. no internet
    access to download the model, or the download timed out).

    Runs the load in a worker thread with a hard timeout so a blocked or slow
    network can't hang the whole app indefinitely.

    Set env RAG_DISABLE_EMBEDDINGS=1 to skip the load entirely (CPU-only /
    CUDA-broken environments where importing sentence_transformers would
    crash the process before the timeout can fire)."""
    import concurrent.futures
    import os

    if os.environ.get("RAG_DISABLE_EMBEDDINGS", "").lower() in ("1", "true", "yes"):
        return None, "Embeddings disabled via RAG_DISABLE_EMBEDDINGS env var."

    def _load():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_load)
    try:
        model = future.result(timeout=timeout_seconds)
        pool.shutdown(wait=False)
        return model, None
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False)
        return None, f"Timed out after {timeout_seconds}s trying to reach Hugging Face to download '{model_name}'."
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        pool.shutdown(wait=False)
        return None, str(exc)


def embed_chunks(model, chunks_df: pd.DataFrame) -> np.ndarray:
    return model.encode(
        chunks_df["search_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=False,
    )
