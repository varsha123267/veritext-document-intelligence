"""
Stacks backend — Document Intelligence + RAG API.

Architecture:
  Upload  ->  parse page-by-page (pypdf/text)  ->  chunk (page-aware)
          ->  embed (sentence-transformers)  ->  store (ChromaDB, persistent)

  Query   ->  expand query with recent conversation  ->  vector search
          ->  optional hybrid re-rank (Gemini)
          ->  grounded, context-aware answer generation with citations (Gemini)

Run with:
    uvicorn app.main:app --reload --port 8000
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from . import ingestion, llm, vectorstore
from .models import (
    DeleteResponse,
    DocumentOut,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    UploadResponse,
)

load_dotenv()

# Where the document registry (names/word counts/summaries/keywords) is
# saved so it survives a server restart. The chunk text + embeddings
# themselves already survive restarts via ChromaDB's own on-disk storage
# (see vectorstore.py) — this file just remembers the matching metadata,
# since that isn't something Chroma stores for us.
REGISTRY_PATH = Path("./documents.json")

app = FastAPI(
    title="Stacks API",
    description="Document Intelligence + Retrieval-Augmented Generation backend.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    # Guarantees every request gets a real HTTP response with CORS headers,
    # even on a bug or an unexpected third-party library error — this is
    # what stops the frontend from ever seeing a raw "Failed to fetch" due
    # to a dropped connection.
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unexpected server error: {exc}"},
    )

def _load_registry() -> Dict[str, DocumentOut]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return {doc_id: DocumentOut(**data) for doc_id, data in raw.items()}
    except Exception:
        # A corrupted or hand-edited registry file shouldn't crash the whole
        # server on startup — just start fresh instead.
        return {}


def _save_registry() -> None:
    try:
        payload = {doc_id: doc.model_dump() for doc_id, doc in _documents.items()}
        REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass  # saving the registry is best-effort; the chunks are safe in Chroma either way


# In-memory registry of document metadata (words/chunk counts/keywords/summary),
# loaded from disk on startup and saved back on every change — this is what
# makes the document list survive a server restart, while the chunk text and
# embeddings themselves always lived in ChromaDB (see vectorstore.py).
_documents: Dict[str, DocumentOut] = _load_registry()


@app.get("/")
def root():
    """A friendly landing response instead of a 404 — the real UI lives in
    frontend/index.html, and the interactive API docs are at /docs."""
    return {
        "app": "Stacks API",
        "status": "running",
        "docs": "/docs",
        "frontend": "open frontend/index.html in your browser",
    }


@app.get("/favicon.ico")
def favicon():
    # Prevents noisy 404 log lines when a browser auto-requests a favicon
    # for this bare API root.
    return Response(status_code=204)


@app.get("/health")
def health():
    return {"status": "ok", "llm_configured": llm.is_available()}


@app.get("/documents", response_model=List[DocumentOut])
def list_documents():
    return list(_documents.values())


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        pages = ingestion.parse_file_pages(file.filename, raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    page_chunks = ingestion.chunk_pages(pages)  # [(page_number_or_None, chunk_text), ...]
    if not page_chunks:
        raise HTTPException(status_code=400, detail="Document appears to be empty.")

    chunks = [c for _, c in page_chunks]
    page_numbers = [p for p, _ in page_chunks]
    full_text = "\n\n".join(chunks)

    doc_id = vectorstore.new_doc_id()
    vectorstore.add_document_chunks(doc_id, file.filename, chunks, page_numbers=page_numbers)

    keywords = ingestion.extract_keywords(chunks)
    summary = llm.summarize(full_text)

    doc = DocumentOut(
        doc_id=doc_id,
        name=file.filename,
        words=len(full_text.split()),
        chunk_count=len(chunks),
        keywords=keywords,
        summary=summary,
    )
    _documents[doc_id] = doc
    _save_registry()
    return UploadResponse(document=doc)


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    if doc_id not in _documents:
        raise HTTPException(status_code=404, detail="Document not found.")
    deleted = vectorstore.delete_document(doc_id)
    del _documents[doc_id]
    _save_registry()
    return DeleteResponse(doc_id=doc_id, deleted_chunks=deleted)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not _documents:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")

    # Conversational retrieval: a short follow-up like "what about the
    # duration?" often doesn't contain enough on its own for good vector
    # search recall, so the most recent prior question is folded into the
    # search query (search only — the original question is still what gets
    # shown to the model and cited, so answers stay precise).
    retrieval_query = req.question
    if req.history:
        retrieval_query = req.history[-1].question + " " + req.question

    pool_size = max(req.top_k, 10) if req.hybrid_rerank else req.top_k
    candidates = vectorstore.query(retrieval_query, top_k=pool_size, doc_id=req.doc_id)
    if not candidates:
        return QueryResponse(
            answer="I couldn't find anything in the indexed passages related to that.",
            sources=[],
            used_llm=False,
        )

    if req.hybrid_rerank:
        candidates = llm.rerank(req.question, candidates, req.top_k)
    else:
        candidates = candidates[: req.top_k]

    sources: List[SourceChunk] = [
        SourceChunk(
            rank=i + 1,
            doc_id=c["doc_id"],
            doc_name=c["doc_name"],
            chunk_index=c["chunk_index"],
            text=c["text"],
            score=round(c["score"], 4),
            page_number=c.get("page_number"),
        )
        for i, c in enumerate(candidates)
    ]

    answer = None
    generation_error = None
    try:
        answer = llm.generate_answer(
            req.question,
            [s.model_dump() for s in sources],
            history=[h.model_dump() for h in req.history] if req.history else None,
        )
    except Exception as e:
        # Never let a Gemini-side failure (bad key, quota, transient network
        # error, etc.) crash the request — fall back to a retrieval-only
        # response instead so the frontend always gets a clean answer.
        generation_error = str(e)

    used_llm = answer is not None
    if not used_llm:
        if generation_error:
            answer = (
                "The retrieved passages are shown below, but generating a "
                f"written answer failed ({generation_error}). Check that "
                "GEMINI_API_KEY in your .env file is correct and has quota "
                "remaining."
            )
        else:
            answer = (
                "No LLM API key is configured on the server, so here are the most "
                "relevant passages retrieved for your question (see sources)."
            )

    return QueryResponse(answer=answer, sources=sources, used_llm=used_llm)
