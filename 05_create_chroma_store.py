"""
05_create_chroma_store.py
--------------------------
Stage 5 of the RAG pipeline: persistent vector store (ChromaDB).

This file did not exist before -- it is the missing required component
called out in the assignment instructions. It is purely additive: the
existing TF-IDF / BM25 / embeddings retrieval in 04_vector_representation.py
and 06_retrieve_context.py is untouched. Chroma is wired into
streamlit_app.py as one more selectable retrieval method ("Chroma (Vector
DB)") alongside TF-IDF, BM25, Embeddings, and Hybrid.

We reuse the SAME sentence-transformers embedding model that already powers
the "Embeddings"/"Hybrid" retrievers (see 04_vector_representation.py)
instead of letting Chroma download its own default embedding model. That
keeps results consistent across retrieval methods and avoids a second
model download. Concretely: we pass pre-computed vectors to Chroma via
`embeddings=` / `query_embeddings=` rather than `documents=` / `query_texts=`,
so Chroma never has to build embeddings on its own.

Persistent storage location: ./chroma_store (relative to the working
directory the app is launched from).
"""

import importlib

import chromadb

CHROMA_PERSIST_DIR = "./chroma_store"
CHROMA_COLLECTION_NAME = "rag_chunks"


def create_chroma_collection(chunks_df, persist_directory: str = CHROMA_PERSIST_DIR,
                              collection_name: str = CHROMA_COLLECTION_NAME):
    """Create (or open, if it already exists) a persistent Chroma collection.

    `chunks_df` isn't read here -- it's accepted per the assignment's
    required signature so the call site can pass the current chunk table
    without the caller needing to know whether the collection is new. Cosine
    distance is configured explicitly so results are comparable to the
    cosine similarity scores used by the other retrievers.

    Callers that re-use this function across different corpora should pass a
    content-derived `collection_name` (e.g. `f"rag_chunks_{file_sig[:8]}"`)
    so changing the CSV doesn't silently reuse a stale on-disk collection.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_chunks_to_chroma(collection, chunks_df, embeddings):
    """Add (or upsert) chunk rows into the Chroma collection.

    chunks_df: the chunk table from 03_chunking.py (must contain chunk_id,
        document_id, title, chunk_text, search_text, and the car-specific
        columns when present).
    embeddings: numpy array aligned row-for-row with chunks_df, e.g. from
        04_vector_representation.embed_chunks(model, chunks_df).
    """
    if len(chunks_df) == 0:
        return collection

    ids = [str(cid) for cid in chunks_df["chunk_id"].tolist()]
    documents = chunks_df["chunk_text"].astype(str).tolist()

    metadatas = []
    for _, row in chunks_df.iterrows():
        metadatas.append({
            "document_id": int(row["document_id"]),
            "title": str(row.get("title", "")),
            "source_file": str(row.get("source_file", "") or ""),
            "is_current": bool(row.get("is_current", True)),
        })

     # Insert into Chroma in batches to avoid exceeding the maximum batch size.

    embeddings_list = (
        embeddings.tolist()
        if hasattr(embeddings, "tolist")
        else list(embeddings)
    )

    BATCH_SIZE = 5000

    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE

        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings_list[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    return collection


def query_chroma(collection, query, top_k, model=None):
    """Query the Chroma collection and return raw Chroma results
    (dict with ids / documents / metadatas / distances).

    `model` is an already-loaded sentence-transformers model (from
    04_vector_representation.load_embedding_model). If not supplied, one is
    loaded lazily so this function still matches the required
    `query_chroma(collection, query, top_k)` signature when called that way.
    """
    if model is None:
        vector_mod = importlib.import_module("04_vector_representation")
        model, error = vector_mod.load_embedding_model()
        if model is None:
            raise RuntimeError(f"Could not load embedding model for Chroma query: {error}")

    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
    )
    return results
