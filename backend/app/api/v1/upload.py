"""Upload endpoints and background ingestion workflow."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.logger import get_logger
from app.models.schemas import JobStatusResponse, UploadResponse
from app.services.chunker import chunk_pages
from app.services.embeddings import embedding_service
from app.services.jobs import DocumentRecord, job_repository
from app.services.loader import load_pdf
from app.services.vectorstore import vectorstore_service

router = APIRouter(tags=["upload"])
logger = get_logger(__name__)


def process_document_job(job_id: str, file_path: str, title: str | None) -> None:
    """Background processing for parse -> chunk -> embed -> index workflow."""

    try:
        pages = load_pdf(file_path)
        total_pages = len(pages)
        job_repository.update_job(job_id, pages_parsed=total_pages, total_pages=total_pages, progress=30)

        doc_id = str(uuid4())
        doc_title = title.strip() if title else Path(file_path).stem
        chunks = chunk_pages(pages, doc_id=doc_id, doc_title=doc_title)
        job_repository.update_job(job_id, chunks_created=len(chunks), progress=60)

        embeddings = embedding_service.embed_texts([chunk.text for chunk in chunks])
        vectorstore_service.add_chunks(chunks, embeddings)

        job_repository.add_document(
            DocumentRecord(
                id=doc_id,
                title=doc_title,
                upload_date=datetime.now(timezone.utc),
                page_count=total_pages,
                file_path=file_path,
            )
        )

        job_repository.update_job(job_id, status="completed", progress=100, doc_id=doc_id)
        logger.info("Ingestion complete for doc_id=%s chunks=%d", doc_id, len(chunks))
    except Exception as exc:
        logger.exception("Document ingestion failed for job_id=%s", job_id)
        job_repository.update_job(job_id, status="failed", error=str(exc), progress=100)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
) -> UploadResponse:
    """Upload a PDF and enqueue background indexing job."""

    suffix = Path(file.filename or "").suffix.lower()
    if file.content_type not in {"application/pdf", "application/octet-stream"} and suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    settings = get_settings()
    job_id = str(uuid4())
    output_path = Path(settings.uploads_dir) / f"{job_id}.pdf"

    with output_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_repository.create_job(job_id)
    background_tasks.add_task(process_document_job, job_id, str(output_path), title)

    return UploadResponse(job_id=job_id, status="processing")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Get processing status for an ingestion job."""

    record = job_repository.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=record["id"],
        status=record["status"],
        progress=float(record["progress"]),
        pages_parsed=int(record["pages_parsed"]),
        total_pages=int(record["total_pages"]),
        chunks_created=int(record["chunks_created"]),
        doc_id=record["doc_id"],
        error=record["error"],
    )
