from pydantic import BaseModel, Field
from typing import List, Optional, Any


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    top_k: Optional[int] = Field(default=5, ge=1, le=50)
    threshold: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)


class Source(BaseModel):
    id: str
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    processing_time_ms: int


class ChatHistoryItem(BaseModel):
    id: str
    session_id: str
    user_message: str
    assistant_message: str
    source_documents: Optional[List[Source]] = None
    processing_time_ms: Optional[int] = None
    created_at: str


class ChatHistoryResponse(BaseModel):
    messages: List[ChatHistoryItem]


class DocumentInfo(BaseModel):
    id: str
    filename: str
    pages: Optional[int] = None
    chunks: Optional[int] = None
    file_size: Optional[int] = None


class DocumentsResponse(BaseModel):
    documents: List[DocumentInfo]


class ApiStatus(BaseModel):
    status: str
    detail: Optional[Any] = None


class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_embeddings: int
    total_sessions: int
    total_messages: int
