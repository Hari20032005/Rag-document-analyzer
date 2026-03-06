"""Chunking utilities for extracted PDF text."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.loader import PageText


@dataclass
class ChunkRecord:
    """Chunk text and metadata stored in vector DB."""

    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    text: str
    snippet: str


DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 800


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_pages(pages: list[PageText], doc_id: str, doc_title: str) -> list[ChunkRecord]:
    """Split pages into overlapping chunks preserving page metadata.

    Chunk sizes are character-based and tuned to approximate ~800-token windows
    with ~200-token overlap for RAG retrieval quality.
    """

    records: list[ChunkRecord] = []
    chunk_index = 0

    for page in pages:
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
                )
            )
            chunk_index += 1

    return records
