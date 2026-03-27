"""Chunking utilities for extracted PDF text.

StructRAG enhancement: section-aware chunking
-------------------------------------------------
When a section tree is available (built during ingestion by *tree_index.py*)
each chunk is annotated with the ``section_type`` and ``tree_node_id`` of the
tree node whose page range contains the chunk's source page.  This metadata
is stored in ChromaDB and later used by *hybrid_retriever.py* for Coherent-
Narrative Re-ranking (Innovation 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.loader import PageText

if TYPE_CHECKING:
    from app.services.tree_index import TreeNode


@dataclass
class ChunkRecord:
    """Chunk text and metadata stored in vector DB."""

    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    text: str
    snippet: str
    section_type: str = "other"
    tree_node_id: str = ""


DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 800


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_pages(
    pages: list[PageText],
    doc_id: str,
    doc_title: str,
    tree_nodes: "list[TreeNode] | None" = None,
) -> list[ChunkRecord]:
    """Split pages into overlapping chunks preserving page metadata.

    Chunk sizes are character-based and tuned to approximate ~800-token windows
    with ~200-token overlap for RAG retrieval quality.

    StructRAG enhancement
    ~~~~~~~~~~~~~~~~~~~~~
    When *tree_nodes* is provided (post-tree-building stage of ingestion) every
    chunk is annotated with the *section_type* and *tree_node_id* of the tree
    node whose page range covers that chunk's source page.  This enables
    section-proximity-aware re-ranking in the hybrid retriever.
    """
    # Build page → section metadata mapping when a tree is available
    page_to_section_type: dict[int, str] = {}
    page_to_node_id: dict[int, str] = {}
    if tree_nodes:
        for node in tree_nodes:
            for pg in range(node.start_page, node.end_page + 1):
                page_to_section_type[pg] = node.section_type
                page_to_node_id[pg] = node.node_id

    records: list[ChunkRecord] = []
    chunk_index = 0

    for page in pages:
        section_type = page_to_section_type.get(page.page_number, "other")
        tree_node_id = page_to_node_id.get(page.page_number, "")

        splits = _splitter.split_text(page.text)
        for split in splits:
            cleaned = split.strip()
            if not cleaned:
                continue
            snippet = cleaned[:220].replace("\n", " ").strip()
            records.append(
                ChunkRecord(
                    chunk_id=f"{doc_id}-chunk-{chunk_index}",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    page=page.page_number,
                    text=cleaned,
                    snippet=snippet,
                    section_type=section_type,
                    tree_node_id=tree_node_id,
                )
            )
            chunk_index += 1

    return records
