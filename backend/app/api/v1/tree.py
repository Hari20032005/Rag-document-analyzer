"""Tree exploration and navigation endpoints (StructRAG dual-index API).

These endpoints expose the hierarchical section tree that is built during
document ingestion, enabling clients to:

  * Browse the full section structure of any indexed document.
  * Perform structure-aware query routing — identify which sections are
    relevant to a question — without triggering full RAG generation.

This is especially useful for interactive document exploration UIs and for
debugging / auditing retrieval behaviour.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    TreeDocumentResponse,
    TreeNavigateRequest,
    TreeNavigateResponse,
    TreeNodeSchema,
)
from app.services.hybrid_retriever import classify_query, navigate_tree
from app.services.jobs import job_repository
from app.services.tree_index import TreeNode
from app.services.tree_store import tree_store

router = APIRouter(tags=["tree"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_to_schema(node: TreeNode) -> TreeNodeSchema:
    return TreeNodeSchema(
        node_id=node.node_id,
        title=node.title,
        start_page=node.start_page,
        end_page=node.end_page,
        summary=node.summary,
        section_type=node.section_type,
        children=[_node_to_schema(c) for c in node.children],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/tree/{doc_id}", response_model=TreeDocumentResponse)
def get_document_tree(doc_id: str) -> TreeDocumentResponse:
    """Return the full hierarchical section tree for a document.

    The tree is built during ingestion and stored in SQLite.  Each node
    exposes its section title, page range, auto-generated summary, and
    canonical section type (e.g. *methodology*, *results*).
    """
    if job_repository.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    nodes = tree_store.get_tree(doc_id)
    if nodes is None:
        raise HTTPException(
            status_code=404,
            detail="Section tree not yet available for this document. "
            "It may still be processing or was indexed before StructRAG was enabled.",
        )

    all_node_count = sum(len(n.all_nodes()) for n in nodes)
    return TreeDocumentResponse(
        doc_id=doc_id,
        node_count=all_node_count,
        nodes=[_node_to_schema(n) for n in nodes],
    )


@router.post("/tree/{doc_id}/navigate", response_model=TreeNavigateResponse)
def navigate_document_tree(doc_id: str, request: TreeNavigateRequest) -> TreeNavigateResponse:
    """Identify which sections of a document are relevant to a query.

    Performs lightweight tree navigation (no vector search, no LLM call)
    and returns the top-3 relevant sections with their metadata.  This
    endpoint is useful for:

    * Previewing which sections will be boosted in a subsequent ``/ask`` call.
    * Building interactive "jump to section" features in document viewers.
    * Debugging or auditing the StructRAG retrieval routing.
    """
    if job_repository.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    nodes = tree_store.get_tree(doc_id)
    if nodes is None:
        raise HTTPException(
            status_code=404,
            detail="Section tree not available for this document.",
        )

    query_type = classify_query(request.query)
    navigated = navigate_tree(request.query, nodes)

    return TreeNavigateResponse(
        query=request.query,
        query_type=query_type,
        relevant_sections=[_node_to_schema(n) for n, _ in navigated],
        section_affinity_scores=[round(s, 3) for _, s in navigated],
    )
