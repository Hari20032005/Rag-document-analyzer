"""Document metadata list and delete endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models.schemas import DeleteDocumentResponse, DocumentInfo, DocumentListResponse
from app.services.jobs import job_repository
from app.services.tree_store import tree_store
from app.services.vectorstore import vectorstore_service

router = APIRouter(tags=["docs"])


@router.get("/docs", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """Return all uploaded documents."""

    records = job_repository.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                id=record.id,
                title=record.title,
                upload_date=record.upload_date,
                page_count=record.page_count,
            )
            for record in records
        ]
    )


@router.delete("/docs/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(doc_id: str) -> DeleteDocumentResponse:
    """Delete document metadata, source PDF, and vectors."""

    record = job_repository.get_document(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    vectorstore_service.delete_doc(doc_id)
    tree_store.delete_tree(doc_id)
    deleted = job_repository.delete_document(doc_id)

    file_path = Path(record.file_path)
    if file_path.exists():
        file_path.unlink(missing_ok=True)

    return DeleteDocumentResponse(doc_id=doc_id, deleted=deleted)
