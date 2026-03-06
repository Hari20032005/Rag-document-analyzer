"""Question-answering, section explanation, and summarization endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    AskRequest,
    AskResponse,
    ExplainSectionRequest,
    ExplainSectionResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services.jobs import job_repository
from app.services.rag import answer_question, summarize_document

router = APIRouter(tags=["qa"])

SECTION_QUERIES = {
    "introduction": "Explain the introduction section: research motivation, goals, and problem statement.",
    "methodology": "Explain the methodology section: approach, datasets, and experimental design.",
    "results": "Explain the results section: main findings, metrics, and interpretation.",
    "conclusion": "Explain the conclusion section: key takeaway, limitations, and future work.",
}


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Answer user question using retrieved document chunks."""

    if job_repository.get_document(request.doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    rag_result = answer_question(
        doc_id=request.doc_id,
        question=request.question,
        top_k=request.top_k,
        mode=request.mode,
        temperature=request.temperature,
    )
    return AskResponse(answer=rag_result.answer, sources=rag_result.sources, prompt_used=rag_result.prompt_used)


@router.post("/explain-section", response_model=ExplainSectionResponse)
def explain_section(request: ExplainSectionRequest) -> ExplainSectionResponse:
    """Explain a specific section of the document."""

    if job_repository.get_document(request.doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if request.section == "custom":
        if not request.custom_query:
            raise HTTPException(status_code=400, detail="custom_query is required when section is 'custom'")
        question = request.custom_query
    else:
        question = SECTION_QUERIES[request.section]

    rag_result = answer_question(
        doc_id=request.doc_id,
        question=question,
        top_k=request.top_k,
        mode="explain",
        temperature=request.temperature,
    )

    return ExplainSectionResponse(
        section=request.section,
        answer=rag_result.answer,
        sources=rag_result.sources,
        prompt_used=rag_result.prompt_used,
    )


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    """Generate structured summary with key concepts."""

    if job_repository.get_document(request.doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    rag_result, key_concepts = summarize_document(
        doc_id=request.doc_id,
        length=request.length,
        temperature=request.temperature,
    )

    return SummarizeResponse(
        summary=rag_result.answer,
        key_concepts=key_concepts,
        sources=rag_result.sources,
        prompt_used=rag_result.prompt_used,
    )
