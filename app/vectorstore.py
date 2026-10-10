"""
Vector store wrapper around ChromaDB.

Embeddings are produced with a local sentence-transformers model
(all-MiniLM-L6-v2), so no external embedding API or key is required —
only the answer-generation step (llm.py) needs an API key.
"""
import os
import uuid
from typing import Dict, List, Optional

# Must be set before chromadb is imported: switches off ChromaDB's anonymous
# usage telemetry (the source of the "Failed to send telemetry event" noise).
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "stacks_chunks"

# anonymized_telemetry=False silences ChromaDB's harmless-but-noisy
# "Failed to send telemetry event" warnings caused by a version mismatch
# between chromadb and its posthog dependency — purely cosmetic, but this
# keeps the server console clean for demos.
_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=chromadb.Settings(anonymized_telemetry=False),
)

_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=_embedder,
    metadata={"hnsw:space": "cosine"},
)


def add_document_chunks(
    doc_id: str,
    doc_name: str,
    chunks: List[str],
    page_numbers: Optional[List[Optional[int]]] = None,
) -> None:
    if not chunks:
        return
    ids = [f"{doc_id}::{i}" for i in range(len(chunks))]
    metadatas = []
    for i in range(len(chunks)):
        meta = {"doc_id": doc_id, "doc_name": doc_name, "chunk_index": i}
        # ChromaDB metadata values can't be None, so the key is only added
        # when a page number actually exists (PDFs); .txt/.md chunks simply
        # won't have this key, and callers treat that as "no page number".
        if page_numbers is not None and page_numbers[i] is not None:
            meta["page_number"] = page_numbers[i]
        metadatas.append(meta)
    _collection.add(ids=ids, documents=chunks, metadatas=metadatas)


def delete_document(doc_id: str) -> int:
    existing = _collection.get(where={"doc_id": doc_id})
    count = len(existing.get("ids", []))
    if count:
        _collection.delete(where={"doc_id": doc_id})
    return count


def query(question: str, top_k: int = 8, doc_id: Optional[str] = None) -> List[Dict]:
    where = {"doc_id": doc_id} if doc_id else None
    result = _collection.query(
        query_texts=[question],
        n_results=top_k,
        where=where,
    )
    if not result["ids"] or not result["ids"][0]:
        return []

    out = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    for i in range(len(ids)):
        # Chroma returns cosine *distance*; convert to a 0-1 similarity score.
        similarity = max(0.0, 1.0 - dists[i])
        out.append(
            {
                "chunk_id": ids[i],
                "text": docs[i],
                "doc_id": metas[i]["doc_id"],
                "doc_name": metas[i]["doc_name"],
                "chunk_index": metas[i]["chunk_index"],
                "page_number": metas[i].get("page_number"),
                "score": similarity,
            }
        )
    return out


def new_doc_id() -> str:
    return uuid.uuid4().hex[:12]
