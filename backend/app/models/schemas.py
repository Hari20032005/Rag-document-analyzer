"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response returned immediately after upload."""

    job_id: str
    status: Literal["processing"]


class JobStatusResponse(BaseModel):
    """Background job status payload."""

    job_id: str
    status: Literal["processing", "completed", "failed"]
    progress: float = Field(ge=0.0, le=100.0)
    pages_parsed: int = 0
    total_pages: int = 0
    chunks_created: int = 0
    doc_id: str | None = None
    error: str | None = None


class DocumentInfo(BaseModel):
    """Stored uploaded document metadata."""

    id: str
    title: str
    upload_date: datetime
    page_count: int


class DocumentListResponse(BaseModel):
    """Uploaded documents list payload."""

    documents: list[DocumentInfo]


class DeleteDocumentResponse(BaseModel):
    """Delete document response."""

    doc_id: str
    deleted: bool


class SourceChunk(BaseModel):
    """Retrieved source chunk information for citation display."""

    page: int
    snippet: str
    chunk_id: str
    score: float


class AskRequest(BaseModel):
    """Question-answer endpoint request."""

    doc_id: str
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)
    mode: Literal["explain", "summary", "qa"] = "qa"
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class AskResponse(BaseModel):
    """Question-answer endpoint response."""

    answer: str
    sources: list[SourceChunk]
    prompt_used: str
    query_type: str = "conceptual"
    retrieval_mode: str = "vector"
    retrieval_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    adaptive_boost_used: float = Field(default=1.0, ge=1.0)


class ExplainSectionRequest(BaseModel):
    """Section explain request payload."""

    doc_id: str
    section: Literal["introduction", "methodology", "results", "conclusion", "custom"]
    custom_query: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class ExplainSectionResponse(BaseModel):
    """Section explanation output."""

    section: str
    answer: str
    sources: list[SourceChunk]
    prompt_used: str
    query_type: str = "conceptual"
    retrieval_mode: str = "vector"
    retrieval_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    adaptive_boost_used: float = Field(default=1.0, ge=1.0)


class SummarizeRequest(BaseModel):
    """Summarization request payload."""

    doc_id: str
    length: Literal["short", "medium", "long"] = "medium"
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class SummarizeResponse(BaseModel):
    """Summarization response payload."""

    summary: str
    key_concepts: list[str]
    sources: list[SourceChunk]
    prompt_used: str


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str


# ---------------------------------------------------------------------------
# StructRAG tree schemas
# ---------------------------------------------------------------------------


class TreeNodeSchema(BaseModel):
    """One node in the document section tree (StructRAG dual-index)."""

    node_id: str
    title: str
    start_page: int
    end_page: int
    summary: str
    section_type: str
    children: list["TreeNodeSchema"] = []


# Required so Pydantic resolves the forward reference on the recursive field.
TreeNodeSchema.model_rebuild()


class TreeDocumentResponse(BaseModel):
    """Full section tree for a document."""

    doc_id: str
    node_count: int
    nodes: list[TreeNodeSchema]


class TreeNavigateRequest(BaseModel):
    """Request body for tree navigation."""

    query: str = Field(min_length=3, max_length=2000)


class TreeNavigateResponse(BaseModel):
    """Sections identified as relevant to a query via tree navigation."""

    query: str
    query_type: str
    relevant_sections: list[TreeNodeSchema]
    section_affinity_scores: list[float] = []
