"""Tests for PDF loading, chunking, and upload status behavior."""

from __future__ import annotations

from pathlib import Path

from app.services.chunker import chunk_pages
from app.services.loader import PageText, load_pdf


def test_pdf_loader_extracts_text_from_sample() -> None:
    """Sample PDF should parse into at least one non-empty page."""

    sample_pdf = Path(__file__).resolve().parents[3] / "EXAMPLE_PAPERS" / "sample_paper.pdf"
    pages = load_pdf(str(sample_pdf))

    assert pages
    assert pages[0].page_number >= 1
    assert pages[0].text.strip() != ""


def test_chunking_creates_overlap_and_page_metadata() -> None:
    """Chunker should produce overlapping chunks with page metadata."""

    text = "A" * 4500 + "B" * 1200
    pages = [PageText(page_number=3, text=text)]

    chunks = chunk_pages(pages, doc_id="doc-1", doc_title="Test Title")

    assert len(chunks) >= 2
    assert chunks[0].page == 3
    assert chunks[1].page == 3
    assert chunks[0].doc_id == "doc-1"
    # Overlap check: a trailing segment from chunk0 appears in chunk1.
    assert chunks[0].text[-200:] in chunks[1].text
