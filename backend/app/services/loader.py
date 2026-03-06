"""PDF loading utilities with pypdf primary and unstructured fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PageText:
    """Extracted text content for a single page."""

    page_number: int
    text: str


def _extract_with_pypdf(file_path: str) -> list[PageText]:
    reader = PdfReader(file_path)
    pages: list[PageText] = []
    for idx, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PageText(page_number=idx + 1, text=text))
    return pages


def _extract_with_unstructured(file_path: str) -> list[PageText]:
    try:
        from unstructured.partition.pdf import partition_pdf
    except Exception as exc:  # pragma: no cover - fallback branch only
        raise RuntimeError("unstructured is not available") from exc

    elements = partition_pdf(filename=file_path)
    page_buckets: dict[int, list[str]] = {}
    for element in elements:
        page_number = int(element.metadata.page_number or 1)
        page_buckets.setdefault(page_number, []).append(str(element))

    return [
        PageText(page_number=page, text="\n".join(lines).strip())
        for page, lines in sorted(page_buckets.items())
        if "\n".join(lines).strip()
    ]


def load_pdf(file_path: str) -> list[PageText]:
    """Extract text from a PDF using pypdf, with optional unstructured fallback."""

    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        pages = _extract_with_pypdf(file_path)
        if pages:
            return pages
    except Exception as exc:
        logger.warning("pypdf extraction failed for %s: %s", file_path, exc)

    logger.info("Attempting unstructured fallback for %s", file_path)
    pages = _extract_with_unstructured(file_path)
    if not pages:
        raise ValueError("No readable text found in PDF")
    return pages
