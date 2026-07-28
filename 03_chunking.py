"""
03_chunking.py
---------------
Stage 3 of the RAG pipeline: documents -> chunks.

Moved from rag_engine.py, unchanged, per the refactor mapping:
    chunk_text, build_chunks
"""

import pandas as pd

# --------------------------------------------------------------------------- #
# 2. Chunking
# --------------------------------------------------------------------------- #

def chunk_text(text: str, chunk_size: int = 60, overlap: int = 15):
    words = text.split()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


_CAR_EXTRA_FIELDS = ["Brand", "Model", "Year", "Fuel_Type", "Transmission", "Mileage", "Doors", "Owner_Count", "Price"]


def build_chunks(documents_df: pd.DataFrame) -> pd.DataFrame:
    """One chunk per document is the overwhelmingly common case here (every
    car-listing row is well under chunk_size words). We short-circuit to that
    directly instead of round-tripping through chunk_text's split/join loop
    for text that was never going to be split anyway. The loop only actually
    runs for the rare long row (e.g. a generic uploaded CSV with a free-text
    column that exceeds chunk_size words)."""
    rows = []
    chunk_id = 0
    chunk_size = 60
    for _, doc in documents_df.iterrows():
        search_text = doc["search_text"]
        if len(search_text.split()) <= chunk_size:
            pieces = [search_text]
        else:
            pieces = chunk_text(search_text, chunk_size=chunk_size)

        for idx, piece in enumerate(pieces):
            row = {
                "chunk_id": chunk_id,
                "document_id": doc["document_id"],
                "chunk_index": idx,
                "title": doc["title"],
                "source_file": doc.get("source_file", ""),
                "schema": doc.get("schema", "generic"),
                "effective_date": doc.get("listing_date", pd.NaT),
                "is_current": doc["is_current"],
                "chunk_text": doc["text"],
                "search_text": piece,
            }
            for field in _CAR_EXTRA_FIELDS:
                row[field] = doc[field] if field in doc.index and pd.notna(doc[field]) else None
            rows.append(row)
            chunk_id += 1
    return pd.DataFrame(rows)
