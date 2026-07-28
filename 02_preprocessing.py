"""
02_preprocessing.py
--------------------
Stage 2 of the RAG pipeline: text cleaning utilities.

Moved from rag_engine.py, unchanged, per the refactor mapping:
    normalize_lexical_text
"""

import re


def normalize_lexical_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
