from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    AMBIGUOUS = "AMBIGUOUS"
    LOW = "LOW"


class Source(BaseModel):
    id: str = Field(..., description="Chunk UUID or identifier")
    document_id: Optional[str] = Field(default=None, description="Parent document UUID")
    filename: Optional[str] = Field(default=None, description="Source PDF filename")
    page_number: Optional[int] = Field(default=1, description="1-indexed source PDF page number")
    score: Optional[float] = Field(default=None, description="Fused similarity / RRF score")
    dense_score: Optional[float] = Field(default=None, description="Dense vector similarity score")
    sparse_score: Optional[float] = Field(default=None, description="Sparse lexical BM25/FTS score")
    rerank_score: Optional[float] = Field(default=None, description="Stage 2 Cross-Encoder neural rerank score")
    content: Optional[str] = Field(default=None, description="Text snippet of the chunk")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or search query")
    session_id: Optional[str] = Field(default="default", description="Multi-user session identifier")
    top_k: Optional[int] = Field(default=5, ge=1, le=50, description="Number of candidate chunks to retrieve")
    threshold: Optional[float] = Field(default=0.35, ge=0.0, le=1.0, description="Minimum confidence threshold")
    stream_format: Optional[str] = Field(default=None, description="Streaming protocol: 'sse' or 'text'")


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.HIGH)
    processing_time_ms: int
    retrieval_ms: Optional[int] = None
    generation_ms: Optional[int] = None
    refusal_triggered: bool = False


class ChatHistoryItem(BaseModel):
    id: str
    session_id: str
    user_message: str
    assistant_message: str
    source_documents: Optional[List[Source]] = None
    processing_time_ms: Optional[int] = None
    retrieval_ms: Optional[int] = None
    generation_ms: Optional[int] = None
    created_at: str


class ChatHistoryResponse(BaseModel):
    messages: List[ChatHistoryItem]


class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    session_id: Optional[str] = Field(default=None, description="Filter by session ID if provided")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of fused results to return")
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="Weight for dense vector ranking")
    sparse_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for sparse lexical ranking")
    rrf_k: int = Field(default=60, ge=1, description="RRF smoothing constant")


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    score: float = Field(..., description="Reciprocal Rank Fusion score")
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rerank_score: Optional[float] = Field(default=None, description="Stage 2 Cross-Encoder neural rerank score")
    content: str


class HybridSearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResultItem]
    retrieval_ms: int


class DocumentInfo(BaseModel):
    id: str
    filename: str
    pages: Optional[int] = None
    chunks: Optional[int] = None
    file_size: Optional[int] = None
    session_id: Optional[str] = None
    created_at: Optional[str] = None


class DocumentsResponse(BaseModel):
    documents: List[DocumentInfo]


class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: int
    token_count: int
    content_hash: str


class ApiStatus(BaseModel):
    status: str
    detail: Optional[Any] = None


class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_embeddings: int
    total_sessions: int
    total_messages: int
