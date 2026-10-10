"""
Pydantic schemas shared across the API.
"""
from typing import List, Optional
from pydantic import BaseModel


class DocumentOut(BaseModel):
    doc_id: str
    name: str
    words: int
    chunk_count: int
    keywords: List[str] = []
    summary: Optional[str] = None


class UploadResponse(BaseModel):
    document: DocumentOut


class HistoryTurn(BaseModel):
    question: str
    answer: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    doc_id: Optional[str] = None   # restrict retrieval to a single document
    hybrid_rerank: bool = True
    # Recent prior turns in this conversation, oldest first, so follow-up
    # questions ("what about the duration?") can be understood in context.
    # Optional and defaults to empty, so older frontends still work unchanged.
    history: List[HistoryTurn] = []


class SourceChunk(BaseModel):
    rank: int
    doc_id: str
    doc_name: str
    chunk_index: int
    text: str
    score: float
    # Only set for PDFs with page-aware chunking; None for .txt/.md or older
    # documents indexed before this feature existed.
    page_number: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    used_llm: bool


class DeleteResponse(BaseModel):
    doc_id: str
    deleted_chunks: int
