"""
06_retrieve_context.py
-----------------------
Stage 6 of the RAG pipeline: querying the indexes built in stage 4/5, and
packaging the top results into a context string for the LLM prompt.

Moved from rag_engine.py, unchanged, per the refactor mapping:
    retrieve_top_k_tfidf, retrieve_top_k_bm25, retrieve_top_k_semantic,
    retrieve_top_k_hybrid, build_context_package, and the retrieval metric
    functions used by the Benchmark tab (precision_at_k, recall_at_k,
    hit_rate_at_k, reciprocal_rank, evaluate_retriever, GROUND_TRUTH).

retrieve_top_k_chroma was added later (production-polish pass) so the
Chroma vector store from 05_create_chroma_store.py has a retriever in the
same family/style as the four above, usable both by the live UI and by the
Benchmark tab's evaluation loop.

See 04_vector_representation.py for a note on why importlib is used to pull
in normalize_lexical_text from 02_preprocessing.py instead of a normal
`from 02_preprocessing import ...` statement.
"""

import importlib

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

_preprocessing = importlib.import_module("02_preprocessing")
normalize_lexical_text = _preprocessing.normalize_lexical_text

# --------------------------------------------------------------------------- #
# 3. Retrieval metrics
# --------------------------------------------------------------------------- #

def precision_at_k(retrieved_ids, relevant_ids, k):
    hits = set(retrieved_ids[:k]).intersection(set(relevant_ids))
    return len(hits) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    hits = set(retrieved_ids[:k]).intersection(set(relevant_ids))
    return len(hits) / len(relevant_ids)


def hit_rate_at_k(retrieved_ids, relevant_ids, k):
    hits = set(retrieved_ids[:k]).intersection(set(relevant_ids))
    return int(len(hits) > 0)


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1 / rank
    return 0.0


# --------------------------------------------------------------------------- #
# 4/5/6. Lexical, semantic, and hybrid retrieval
# --------------------------------------------------------------------------- #

def retrieve_top_k_tfidf(query, chunks_df, vectorizer, matrix, k=3):
    qv = vectorizer.transform([normalize_lexical_text(query)])
    scores = cosine_similarity(qv, matrix).flatten()
    ranking = np.argsort(scores)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "TF-IDF"
    return results.reset_index(drop=True)


def retrieve_top_k_bm25(query, chunks_df, bm25, k=3):
    def simple_tokenize(text):
        import re
        return re.findall(r"[a-z0-9]+", text.lower())
    scores = bm25.get_scores(simple_tokenize(query))
    ranking = np.argsort(scores)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "BM25"
    return results.reset_index(drop=True)


def retrieve_top_k_semantic(query, chunks_df, model, chunk_embeddings, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    ranking = np.argsort(scores)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "Embeddings"
    return results.reset_index(drop=True)


def min_max_normalize(scores):
    scores = np.array(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def retrieve_top_k_hybrid(query, chunks_df, vectorizer, matrix, model, chunk_embeddings, alpha=0.6, k=3):
    lexical_scores = cosine_similarity(
        vectorizer.transform([normalize_lexical_text(query)]), matrix
    ).flatten()
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    semantic_scores = cosine_similarity(query_embedding, chunk_embeddings).flatten()

    hybrid_scores = alpha * min_max_normalize(semantic_scores) + (1 - alpha) * min_max_normalize(lexical_scores)
    ranking = np.argsort(hybrid_scores)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["score"] = hybrid_scores[ranking]
    results["retriever"] = "Hybrid"
    return results.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6b. Vector-store retriever (ChromaDB)
# --------------------------------------------------------------------------- #

def retrieve_top_k_chroma(query, chunks_df, collection, model, k=3):
    """Chroma-backed retriever, matching the shape of the other
    retrieve_top_k_* functions above: same (query, chunks_df, ..., k)
    signature, same "score" + "retriever" output columns.

    `collection` is a Chroma collection from
    05_create_chroma_store.create_chroma_collection, already populated via
    add_chunks_to_chroma. It may hold more rows than `chunks_df` (e.g. when
    `chunks_df` is a filtered subset such as the UI's active_chunks_df) --
    results are restricted to chunk_ids present in `chunks_df`, over-fetching
    from Chroma first so a narrow filter doesn't starve the results.
    """
    # Local import (not at module top) so this file doesn't hard-require
    # chromadb just to use the TF-IDF/BM25/semantic/hybrid retrievers above.
    chroma_mod = importlib.import_module("05_create_chroma_store")

    lookup = chunks_df.set_index("chunk_id")
    pool = min(max(k * 5, k), collection.count()) if collection.count() else k
    raw = chroma_mod.query_chroma(collection, query, top_k=pool, model=model)

    ids = [int(cid) for cid in raw["ids"][0]]
    distances = raw["distances"][0]

    rows = []
    for chunk_id, distance in zip(ids, distances):
        if chunk_id not in lookup.index:
            continue
        row = lookup.loc[chunk_id].copy()
        row["score"] = 1 - distance  # cosine distance -> similarity
        rows.append(row)
        if len(rows) >= k:
            break

    if not rows:
        results = chunks_df.iloc[0:0].copy()
        results["score"] = []
    else:
        results = pd.DataFrame(rows).reset_index()
    results["retriever"] = "Chroma"
    return results.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 7. Context building
# --------------------------------------------------------------------------- #

def build_context_package(
    candidates: pd.DataFrame,
    max_context_chunks: int = 3,
    max_chunks_per_document: int = 1,
    word_budget: int = 150,
    prefer_current: bool = True,
):
    candidates = candidates.copy()
    if prefer_current:
        candidates = candidates.sort_values(by=["is_current", "score"], ascending=[False, False]).reset_index(drop=True)

    selected_rows = []
    seen_texts = set()
    seen_docs = {}
    used_words = 0

    for _, row in candidates.iterrows():
        doc_id = row["document_id"]
        seen_docs.setdefault(doc_id, 0)
        if seen_docs[doc_id] >= max_chunks_per_document:
            continue
        if row["chunk_text"] in seen_texts:
            continue
        n_words = len(row["chunk_text"].split())
        if used_words + n_words > word_budget and selected_rows:
            continue

        selected_rows.append(row)
        seen_texts.add(row["chunk_text"])
        seen_docs[doc_id] += 1
        used_words += n_words

        if len(selected_rows) >= max_context_chunks:
            break

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)

    lines = []
    for i, row in selected_df.iterrows():
        status = "CURRENT" if row["is_current"] else "OUTDATED/SUPERSEDED"
        lines.append(f"[Source {i + 1} | {row['title']} | {status}]\n{row['chunk_text']}")
    context_text = "\n\n".join(lines)

    return {
        "context_text": context_text,
        "selected_df": selected_df,
        "used_words": used_words,
        "num_sources": len(selected_df),
    }


# --------------------------------------------------------------------------- #
# 10. Ground-truth query set (verified against the dataset) + evaluation
# --------------------------------------------------------------------------- #

GROUND_TRUTH = {
    "How much would I pay for a 2002 Chevrolet Impala with a diesel engine and a stick shift?": [1282],
    "What's the current asking price for a 2002 Audi Q5 diesel with a manual gearbox?": [5227],
    "Which currently listed car has the lowest odometer reading?": [538],
    "Which currently listed car has been driven the most miles?": [7838],
    "What's the cheapest Audi currently in the lot?": [1283, 7325, 8703],
    "Which vehicles are the most expensive in the current inventory?": [1012, 1100, 7221],
    "Is there a one-owner electric BMW X5 from 2018 or later?": [2553, 3809, 8394, 9331],
    "Looking for a semi-auto hybrid Honda Civic from 2019 onward -- what's available?": [850, 4441],
    "Show me Ford Focus listings from 2020-2023 that have had five previous owners.": [2164, 3904, 4511, 5453, 6338, 7399],
    "I want an eco-friendly two-door Toyota under $8,000 -- what's out there?": [
        1756, 1924, 2097, 2642, 2797, 3321, 4085, 4156, 4746, 4923,
        5021, 5294, 5805, 6174, 6234, 6613, 6781, 7047, 7854,
    ],
    "Any nearly-new, single-owner Kia with under 5,000 miles on it?": [991, 7354, 7555, 9845],
    "Which 3-owner Mercedes with a big petrol engine do you have?": [
        891, 1149, 2065, 4373, 4464, 5808, 6904, 7148, 7621, 8800, 9508, 9751, 9867,
    ],
}


def evaluate_retriever(name, retrieval_fn, ground_truth, k):
    rows = []
    for query, relevant in ground_truth.items():
        res = retrieval_fn(query, k)
        retrieved_ids = res["document_id"].tolist()
        rows.append({
            "retriever": name,
            "query": query,
            f"precision@{k}": precision_at_k(retrieved_ids, relevant, k),
            f"recall@{k}": recall_at_k(retrieved_ids, relevant, k),
            f"hit_rate@{k}": hit_rate_at_k(retrieved_ids, relevant, k),
            "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant),
        })
    return pd.DataFrame(rows)
