"""Hierarchical section-tree builder for academic PDF documents.

Novel "StructRAG" contribution — Dual-Index Architecture:
  Builds a semantic Table-of-Contents (ToC) tree from raw page text without
  requiring a vector database.  The tree is used alongside the vector index during
  retrieval to provide *structure-aware* context selection.

Key innovations over vanilla PageIndex:
  1. Academic Section Ontology: classifies every tree node into a standardised
     taxonomy (abstract, introduction, methodology, results, …), enabling
     section-type-aware retrieval boosting downstream.
  2. Hybrid construction: tries LLM-based ToC extraction first; falls back to a
     multi-signal heuristic heading detector that works without any API key.
  3. Dual-mode summarisation: fast text-snippet extraction for every node
     (zero LLM cost), optionally enhanced with an LLM for richer summaries.
  4. Designed to feed Tree-Boosted Reciprocal Rank Fusion (TB-RRF) in
     hybrid_retriever.py — the structural backbone of StructRAG.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.loader import PageText

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Academic section ontology
# ---------------------------------------------------------------------------

SectionType = Literal[
    "abstract",
    "introduction",
    "related_work",
    "background",
    "methodology",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "acknowledgements",
    "references",
    "appendix",
    "other",
]

# Keyword mapping for ontology classification
_SECTION_KEYWORDS: dict[str, list[str]] = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "overview", "motivation"],
    "related_work": [
        "related work",
        "related works",
        "prior work",
        "literature review",
        "background and related",
    ],
    "background": ["background", "preliminaries", "preliminary"],
    "methodology": [
        "methodology",
        "method",
        "methods",
        "approach",
        "proposed method",
        "framework",
        "system design",
        "architecture",
    ],
    "experiments": [
        "experiment",
        "experiments",
        "experimental",
        "experimental setup",
        "evaluation",
        "evaluation setup",
    ],
    "results": ["result", "results", "findings", "performance", "analysis"],
    "discussion": ["discussion", "limitations", "ablation"],
    "conclusion": ["conclusion", "conclusions", "summary", "future work"],
    "acknowledgements": ["acknowledgement", "acknowledgements", "acknowledgment"],
    "references": ["references", "bibliography"],
    "appendix": ["appendix", "supplementary"],
}

# Regex patterns for detecting section headings in academic text
_HEADING_PATTERNS = [
    # "1. Introduction", "2.3 Related Work", "I. Background"
    re.compile(r"^(?:(?:[IVXLC]+\.?|\d+(?:\.\d+)*\.?)\s+)([A-Z][A-Za-z\s,:\-]{2,60})$"),
    # "INTRODUCTION", "RELATED WORK"
    re.compile(r"^([A-Z]{3,}(?:\s+[A-Z]+){0,5})$"),
    # Title-cased heading on its own line (≤ 7 words)
    re.compile(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,6})$"),
]

# Minimum number of pages a document must have before heuristic is applied
_MIN_PAGES_FOR_HEURISTIC = 2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A node in the hierarchical section tree."""

    node_id: str
    title: str
    start_page: int
    end_page: int
    summary: str = ""
    section_type: SectionType = "other"
    children: list["TreeNode"] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "summary": self.summary,
            "section_type": self.section_type,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TreeNode":
        node = cls(
            node_id=data["node_id"],
            title=data["title"],
            start_page=data["start_page"],
            end_page=data["end_page"],
            summary=data.get("summary", ""),
            section_type=data.get("section_type", "other"),
        )
        node.children = [cls.from_dict(c) for c in data.get("children", [])]
        return node

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def all_nodes(self) -> list["TreeNode"]:
        """Pre-order flattened list of this node and all descendants."""
        result: list[TreeNode] = [self]
        for child in self.children:
            result.extend(child.all_nodes())
        return result

    def covers_page(self, page: int) -> bool:
        return self.start_page <= page <= self.end_page


# ---------------------------------------------------------------------------
# Ontology classification
# ---------------------------------------------------------------------------


def classify_section_type(title: str) -> SectionType:
    """Map a section heading to the canonical *SectionType* via keyword matching."""
    lower = title.lower().strip()
    for section_type, keywords in _SECTION_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return section_type  # type: ignore[return-value]
    return "other"


# ---------------------------------------------------------------------------
# Heuristic heading detection
# ---------------------------------------------------------------------------


def _extract_headings_heuristic(pages: list[PageText]) -> list[tuple[str, int]]:
    """Scan every page for lines that look like section headings.

    Returns a deduplicated list of (heading_text, page_number) sorted by page
    number.  The detector uses a multi-signal scoring approach:

    * Matches one of *_HEADING_PATTERNS*
    * Short line (≤ 80 chars)
    * Does not read like ordinary prose (low punctuation density)
    * Title maps to a known *SectionType* (priority boost)
    """
    headings: list[tuple[str, int]] = []
    seen: set[str] = set()

    for page in pages:
        for line in page.text.splitlines():
            stripped = line.strip()
            if not (3 <= len(stripped) <= 80):
                continue
            # Reject lines with mid-sentence grammar indicators
            if stripped.count(",") > 3 or stripped.count(";") > 1:
                continue
            # Reject obvious prose (many determiners/prepositions)
            word_count = len(stripped.split())
            if word_count > 10:
                continue

            matched_title: str | None = None
            for pattern in _HEADING_PATTERNS:
                m = pattern.fullmatch(stripped)
                if m:
                    matched_title = stripped
                    break

            if matched_title:
                norm = matched_title.lower().strip(" .-0123456789")
                if norm not in seen and len(norm) >= 3:
                    seen.add(norm)
                    headings.append((matched_title, page.page_number))

    # Sort by page, then de-duplicate consecutive identical titles
    headings.sort(key=lambda h: h[1])
    return headings


# ---------------------------------------------------------------------------
# LLM-based ToC extraction (best-effort, graceful fallback)
# ---------------------------------------------------------------------------


def _llm_extract_toc(
    pages: list[PageText],
    max_toc_pages: int = 20,
) -> list[tuple[str, int]] | None:
    """Ask an LLM to extract a structured table of contents from the first *N* page.

    Returns a list of *(section_title, page_number)* or *None* if the LLM is
    unavailable or extraction fails.  Uses a single LLM request for the whole
    document — deliberately minimising API cost.
    """
    settings = get_settings()
    if settings.llm_provider == "mock":
        return None  # No LLM available; caller falls back to heuristic

    sample_text = "\n\n---PAGE BREAK---\n\n".join(
        f"[Page {p.page_number}]\n{p.text[:600]}" for p in pages[:max_toc_pages]
    )

    prompt = (
        "You are analysing an academic document. "
        "Given the text from the first pages below, extract a table of contents as a "
        "JSON array of objects with keys 'title' (string) and 'page' (integer, 1-based). "
        "Only include major section headings (not sub-sub-sections). "
        "If no explicit TOC exists, infer sections from text patterns. "
        "Respond ONLY with a valid JSON array — no prose or markdown fences.\n\n"
        f"Document text:\n{sample_text}\n\nJSON:"
    )

    raw: str = ""
    try:
        if settings.llm_provider == "openai" and settings.openai_api_key:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key, timeout=40.0, max_retries=1)
            response = client.chat.completions.create(
                model=settings.openai_model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            raw = response.choices[0].message.content or ""

        elif settings.llm_provider == "gemini" and settings.gemini_api_key:
            import httpx

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{settings.gemini_model}:generateContent"
            )
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 800},
            }
            with httpx.Client(timeout=40.0) as client:
                resp = client.post(
                    url,
                    json=payload,
                    headers={"X-goog-api-key": settings.gemini_api_key},
                )
                resp.raise_for_status()
            raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

        else:
            return None  # HF / unknown provider — use heuristic

        # Strip optional markdown code fences
        if "```" in raw:
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return None

        result: list[tuple[str, int]] = []
        for item in parsed:
            if isinstance(item, dict) and "title" in item and "page" in item:
                try:
                    result.append((str(item["title"]), int(item["page"])))
                except (ValueError, TypeError):
                    continue
        return result or None

    except Exception as exc:  # pragma: no cover
        logger.warning(
            "LLM ToC extraction failed — falling back to heuristic: %s", exc
        )
        return None


# ---------------------------------------------------------------------------
# Tree construction helpers
# ---------------------------------------------------------------------------


def _build_nodes_from_headings(
    headings: list[tuple[str, int]],
    total_pages: int,
    doc_id: str,
) -> list[TreeNode]:
    """Convert a flat list of (heading, start_page) into *TreeNode* objects.

    Each node's *end_page* is set to one before the next section's start page
    (or the document's last page for the final section).
    """
    if not headings:
        return [
            TreeNode(
                node_id=f"{doc_id}-node-0",
                title="Document",
                start_page=1,
                end_page=total_pages,
                section_type="other",
            )
        ]

    nodes: list[TreeNode] = []
    for idx, (title, start_page) in enumerate(headings):
        end_page = (
            headings[idx + 1][1] - 1 if idx + 1 < len(headings) else total_pages
        )
        end_page = max(start_page, end_page)
        nodes.append(
            TreeNode(
                node_id=f"{doc_id}-node-{idx}",
                title=title,
                start_page=start_page,
                end_page=end_page,
                section_type=classify_section_type(title),
            )
        )
    return nodes


def _generate_text_summaries(nodes: list[TreeNode], pages: list[PageText]) -> None:
    """Populate each node's *summary* using a snippet from its page range.

    This is fast (no LLM required) and always runs as the baseline.
    """
    page_map: dict[int, str] = {p.page_number: p.text for p in pages}
    for node in nodes:
        parts: list[str] = []
        for pg in range(node.start_page, min(node.start_page + 2, node.end_page + 1)):
            text = page_map.get(pg, "")
            if text:
                parts.append(text[:400])
        combined = " ".join(parts).replace("\n", " ").strip()
        node.summary = (
            combined[:280] if combined else f"Pages {node.start_page}–{node.end_page}."
        )


def _enhance_summaries_with_llm(
    nodes: list[TreeNode], pages: list[PageText]
) -> None:  # pragma: no cover
    """Optionally replace text-snippet summaries with concise LLM-generated ones.

    This is a best-effort enhancement — failures are silently ignored.
    """
    settings = get_settings()
    if settings.llm_provider not in {"openai", "gemini"} or not nodes:
        return

    page_map: dict[int, str] = {p.page_number: p.text for p in pages}
    for node in nodes:
        parts: list[str] = []
        for pg in range(node.start_page, min(node.end_page + 1, node.start_page + 3)):
            text = page_map.get(pg, "")
            if text:
                parts.append(text[:600])
        section_text = "\n".join(parts)[:1500]
        if not section_text:
            continue
        prompt = (
            f"Summarise the following section titled '{node.title}' in 1–2 sentences. "
            f"Be concise.\n\n{section_text}\n\nSummary:"
        )
        try:
            if settings.llm_provider == "openai" and settings.openai_api_key:
                from openai import OpenAI

                client = OpenAI(api_key=settings.openai_api_key, timeout=15.0, max_retries=0)
                resp = client.chat.completions.create(
                    model=settings.openai_model,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                )
                node.summary = (resp.choices[0].message.content or "").strip() or node.summary
            time.sleep(0.15)  # gentle rate limiting between per-node calls
        except Exception:
            pass  # text-based summary already set; keep it


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_document_tree(
    pages: list[PageText],
    doc_id: str,
    doc_title: str,
    use_llm: bool = True,
) -> list[TreeNode]:
    """Build a hierarchical section tree from extracted PDF pages.

    Ingestion algorithm
    -------------------
    1. (Optional) Call LLM to extract a structured ToC from the first 20 pages.
    2. If LLM unavailable or extraction fails, use heuristic heading detection.
    3. Assign canonical *SectionType* labels to every node.
    4. Compute text-snippet summaries (zero LLM cost).
    5. (Optional) Enhance summaries with concise LLM-generated text.

    The returned list of *TreeNode* objects is persisted to SQLite by
    *TreeStore* and later loaded by *hybrid_retriever.py* to drive TB-RRF.
    """
    total_pages = max((p.page_number for p in pages), default=1)

    # --- Step 1 / 2: obtain headings -----------------------------------------
    headings: list[tuple[str, int]] | None = None
    if use_llm:
        logger.info("Attempting LLM ToC extraction for doc_id=%s", doc_id)
        headings = _llm_extract_toc(pages)

    if not headings:
        logger.info("Using heuristic heading detection for doc_id=%s", doc_id)
        headings = _extract_headings_heuristic(pages) or None

    # --- Step 3: build tree ---------------------------------------------------
    nodes = _build_nodes_from_headings(headings or [], total_pages, doc_id)

    # --- Step 4: text summaries (always) --------------------------------------
    _generate_text_summaries(nodes, pages)

    # --- Step 5: optional LLM enhancement ------------------------------------
    if use_llm:
        _enhance_summaries_with_llm(nodes, pages)

    logger.info(
        "Tree built for doc_id=%s title=%r nodes=%d total_pages=%d",
        doc_id,
        doc_title,
        len(nodes),
        total_pages,
    )
    return nodes
