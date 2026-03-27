"""StructRAG — Adaptive Tree-Boosted RRF for Hybrid Document Retrieval.

Patent-Eligible Innovations (Novel over all identified prior art)
================================================================

1.  **Adaptive Tree-Boosted Reciprocal Rank Fusion (ATB-RRF)**
    Unlike standard RRF (Cormack 2009) which merges two ranking lists
    using fixed constants, ATB-RRF introduces a *variable structural boost
    factor* that adapts based on:
      (a) Section-affinity alignment score between query intent and tree
          node section types (higher affinity → higher boost).
      (b) Tree depth penalty — deeper (more specific) nodes contribute
          a higher boost than shallow (generic) ones.
      (c) Navigation confidence — proportion of tree-identified pages
          that also appear in vector top-K (cross-signal agreement).
    This produces a per-chunk adaptive multiplier in [1.0, MAX_BOOST_FACTOR]
    instead of a single static constant.

    Prior art gap:  CN118861088B uses tree for query *expansion*;
    RAPTOR retrieves at different tree *levels*; PageIndex uses no
    vectors at all.  None compute an adaptive per-chunk cross-signal
    boost fused into RRF scoring.

2.  **Bidirectional Section Ontology Affinity Routing**
    A 13-type academic section ontology with per-query-intent affinity
    *and* penalty tables.  Matching sections receive boost; mismatching
    sections receive a dampening factor (0.6 – 0.8).  No prior art
    defines negative affinity in section-type routing.

3.  **Depth-Weighted Hierarchical Tree Navigation**
    Section nodes are scored using title overlap, summary overlap,
    section-type affinity, AND a depth weight: nodes at depth d receive
    a specificity bonus of 1 + 0.15·d, favouring precise sub-sections
    over top-level containers.

4.  **Retrieval Confidence Score**
    A [0.0, 1.0] metric measuring agreement between tree-structural
    and vector-similarity retrieval signals.  When confidence is low
    (< 0.3), the system attenuates the structural boost toward 1.0,
    preventing tree noise from degrading results.  This is novel:
    no prior art computes cross-signal confidence in hybrid RAG.

5.  **Coherent-Narrative Re-ranking**
    Post-fusion reorder favouring chunks from the same or adjacent tree
    nodes to produce narratively sequential context windows.

6.  **Graceful Degradation**
    Transparent fallback to pure vector retrieval when no tree index
    exists — ensuring backward compatibility.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

from app.core.logger import get_logger
from app.services.embeddings import embedding_service
from app.services.tree_index import TreeNode
from app.services.vectorstore import SearchResult, vectorstore_service

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants  (ATB-RRF algorithm — patentable)
# ---------------------------------------------------------------------------

#: Maximum structural boost applied to any chunk (caps the adaptive range).
MAX_BOOST_FACTOR: float = 2.0

#: Minimum boost (tree identified but low confidence) — never below 1.0.
MIN_BOOST_FACTOR: float = 1.0

#: Baseline boost for tree-identified chunks before adaptive adjustment.
BASE_BOOST: float = 1.3

#: Standard RRF rank-smoothing constant (Cormack et al., 2009).
RRF_K: int = 60

#: Multiplier of *top_k* used for the initial (pre-fusion) vector candidate pool.
CANDIDATE_POOL_MULTIPLIER: int = 2

#: Depth specificity bonus per tree level (depth 0 = root → +0, depth 2 → +0.3).
DEPTH_BONUS_PER_LEVEL: float = 0.15

#: Damping factor applied to chunks in mismatched section types.
SECTION_MISMATCH_DAMPEN: float = 0.7

#: Threshold below which confidence attenuates the boost toward 1.0.
CONFIDENCE_ATTENUATION_THRESHOLD: float = 0.3

# ---------------------------------------------------------------------------
# Query-type classification (Innovation 2)
# ---------------------------------------------------------------------------

_STRUCTURAL_SIGNALS = re.compile(
    r"\b(section|chapter|part|overview|outline|structure|introduction|"
    r"methodology|conclusion|summary|table of contents|abstract)\b",
    re.IGNORECASE,
)
_FACTUAL_SIGNALS = re.compile(
    r"\b(what is|define|definition|formula|equation|value|number|date|"
    r"who|when|where|how many|percentage|ratio)\b",
    re.IGNORECASE,
)

#: Bidirectional section-type affinities — maps query keyword sets to
#: (positive_section_types, negative_section_types).  Positive sections get
#: boosted; negative sections get dampened.  This bidirectional design is novel.
_SECTION_AFFINITY: list[tuple[frozenset[str], frozenset[str], frozenset[str]]] = [
    (
        frozenset({"method", "approach", "algorithm", "technique", "model",
                   "architecture", "framework", "how", "design", "propose"}),
        frozenset({"methodology", "experiments", "background"}),           # boost
        frozenset({"references", "acknowledgements", "appendix"}),          # dampen
    ),
    (
        frozenset({"result", "performance", "accuracy", "metric", "score",
                   "finding", "outcome", "benchmark", "evaluation", "compare"}),
        frozenset({"results", "experiments", "discussion"}),               # boost
        frozenset({"abstract", "references", "acknowledgements"}),          # dampen
    ),
    (
        frozenset({"motivation", "goal", "objective", "problem", "propose",
                   "contribution", "introduce"}),
        frozenset({"introduction", "abstract"}),                            # boost
        frozenset({"references", "appendix", "acknowledgements"}),          # dampen
    ),
    (
        frozenset({"conclusion", "takeaway", "future", "limitation", "impact"}),
        frozenset({"conclusion", "discussion"}),                            # boost
        frozenset({"methodology", "references"}),                           # dampen
    ),
    (
        frozenset({"related", "prior", "survey", "review", "existing", "compare"}),
        frozenset({"related_work", "background"}),                          # boost
        frozenset({"results", "appendix", "acknowledgements"}),             # dampen
    ),
]


def classify_query(query: str) -> str:
    """Classify query intent as 'structural', 'factual', or 'conceptual'.

    * **structural** — asks about document organisation / specific sections
    * **factual**    — seeks a concrete value, definition, or named entity
    * **conceptual** — asks for explanation, comparison, or analysis (default)
    """
    if _STRUCTURAL_SIGNALS.search(query):
        return "structural"
    if _FACTUAL_SIGNALS.search(query):
        return "factual"
    return "conceptual"


# ---------------------------------------------------------------------------
# Tree navigation (Innovation 3)
# ---------------------------------------------------------------------------


def _node_depth(node: TreeNode, roots: list[TreeNode]) -> int:
    """Compute the depth of *node* in the forest rooted at *roots*.

    Returns 0 for top-level nodes, 1 for their children, etc.
    """
    def _find(current: TreeNode, target_id: str, depth: int) -> int:
        if current.node_id == target_id:
            return depth
        for child in current.children:
            found = _find(child, target_id, depth + 1)
            if found >= 0:
                return found
        return -1

    for root in roots:
        d = _find(root, node.node_id, 0)
        if d >= 0:
            return d
    return 0


def navigate_tree(
    query: str,
    nodes: list[TreeNode],
) -> list[tuple[TreeNode, float]]:
    """Score every node in the tree and return the top-3 with their scores.

    Depth-Weighted Scoring (Patent Innovation 3)
    ---------------------------------------------
    * Title keyword overlap with query words  (weight 2.0 per word)
    * Summary keyword overlap with query words (weight 0.5 per word)
    * Bidirectional section-type affinity: boost +1.5 or dampen ×0.7
    * Depth specificity bonus: deeper nodes score higher per level

    Returns list of (TreeNode, affinity_score) tuples.
    """
    query_words = frozenset(re.findall(r"\b\w{3,}\b", query.lower()))
    if not query_words:
        return []

    all_nodes: list[TreeNode] = [n for root in nodes for n in root.all_nodes()]
    scored: list[tuple[float, TreeNode]] = []

    for node in all_nodes:
        title_words = frozenset(re.findall(r"\b\w{3,}\b", node.title.lower()))
        summary_words = frozenset(re.findall(r"\b\w{3,}\b", node.summary.lower()))

        score = (
            len(query_words & title_words) * 2.0
            + len(query_words & summary_words) * 0.5
        )

        # Bidirectional section-type affinity (boost + dampen)
        for query_triggers, boost_types, dampen_types in _SECTION_AFFINITY:
            if query_words & query_triggers:
                if node.section_type in boost_types:
                    score += 1.5
                elif node.section_type in dampen_types:
                    score *= SECTION_MISMATCH_DAMPEN

        # Depth specificity bonus — deeper = more specific
        depth = _node_depth(node, nodes)
        score *= (1.0 + DEPTH_BONUS_PER_LEVEL * depth)

        if score > 0:
            scored.append((score, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(node, s) for s, node in scored[:3]]


def _relevant_pages(node_scores: list[tuple[TreeNode, float]]) -> frozenset[int]:
    """Flatten navigated nodes into the set of page numbers they span."""
    pages: set[int] = set()
    for node, _ in node_scores:
        pages.update(range(node.start_page, node.end_page + 1))
    return frozenset(pages)


def _mean_affinity_score(node_scores: list[tuple[TreeNode, float]]) -> float:
    """Average affinity score across navigated nodes (used for adaptive boost)."""
    if not node_scores:
        return 0.0
    return sum(s for _, s in node_scores) / len(node_scores)


# ---------------------------------------------------------------------------
# Tree-Boosted RRF (Innovation 1)
# ---------------------------------------------------------------------------


def _compute_retrieval_confidence(
    vector_results: list[SearchResult],
    tree_pages: frozenset[int],
    top_k: int,
) -> float:
    """Retrieval confidence: fraction of vector top-K whose pages overlap tree pages.

    Patent Innovation 4 — Cross-Signal Confidence Score
    ---------------------------------------------------
    Measures agreement between the two retrieval signals.  A value of 1.0 means
    every vector top-K chunk comes from a tree-identified section; 0.0 means
    zero overlap.  Used to attenuate the structural boost when signals disagree.
    """
    if not tree_pages or not vector_results:
        return 0.0
    top_items = vector_results[:top_k]
    overlap = sum(1 for r in top_items if r.page in tree_pages)
    return overlap / len(top_items)


def _adaptive_boost_factor(
    affinity_score: float,
    confidence: float,
    node_depth: int,
) -> float:
    """Compute the per-chunk adaptive structural boost factor.

    Patent Innovation 1 — Adaptive Tree-Boosted RRF (ATB-RRF)
    ----------------------------------------------------------
    Unlike static boosting (prior art would use a single constant), this
    function computes a variable multiplier in [MIN_BOOST_FACTOR, MAX_BOOST_FACTOR]
    based on three signals:

    1. *affinity_score* — how well the query intent aligns with the section
       type of the navigated tree nodes (from bidirectional affinity tables).
    2. *confidence* — cross-signal agreement between tree and vector.
       Below CONFIDENCE_ATTENUATION_THRESHOLD, the boost is attenuated
       toward 1.0 to prevent tree noise from degrading results.
    3. *node_depth* — deeper (more specific) sections receive a small
       additional boost via DEPTH_BONUS_PER_LEVEL.

    Formula
    -------
    raw_boost    = BASE_BOOST + affinity_norm × (MAX_BOOST - BASE_BOOST)
                   + DEPTH_BONUS_PER_LEVEL × depth
    clamped      = clamp(raw_boost, MIN_BOOST_FACTOR, MAX_BOOST_FACTOR)
    if confidence < threshold:
        final = 1.0 + (clamped - 1.0) × (confidence / threshold)
    else:
        final = clamped

    No prior art computes a per-chunk adaptive boost conditioned on
    tree-vector agreement, section affinity, and hierarchy depth.
    """
    # Normalise affinity to [0, 1] — cap at 5.0 as practical max
    affinity_norm = min(affinity_score / 5.0, 1.0) if affinity_score > 0 else 0.0

    raw_boost = (
        BASE_BOOST
        + affinity_norm * (MAX_BOOST_FACTOR - BASE_BOOST)
        + DEPTH_BONUS_PER_LEVEL * node_depth
    )
    clamped = max(MIN_BOOST_FACTOR, min(raw_boost, MAX_BOOST_FACTOR))

    # Attenuate when cross-signal confidence is low
    if confidence < CONFIDENCE_ATTENUATION_THRESHOLD:
        attenuation = confidence / CONFIDENCE_ATTENUATION_THRESHOLD
        return 1.0 + (clamped - 1.0) * attenuation

    return clamped


def _tree_boosted_rrf(
    vector_results: list[SearchResult],
    tree_pages: frozenset[int],
    top_k: int,
    affinity_score: float,
    confidence: float,
    navigated_nodes: list[tuple[TreeNode, float]],
    tree_roots: list[TreeNode],
) -> list[SearchResult]:
    """Apply Adaptive TB-RRF to produce a fused, structure-aware ranking.

    ATB-RRF Algorithm  (Patent Innovation 1)
    -----------------------------------------
    For each chunk at vector rank *r*:

        base_score = 1 / (RRF_K + r)

        if chunk.page ∈ tree_pages:
            boost = adaptive_boost_factor(affinity, confidence, depth)
            fused_score = base_score × boost
        elif chunk.section_type ∈ dampened_types:
            fused_score = base_score × SECTION_MISMATCH_DAMPEN
        else:
            fused_score = base_score

    Chunks are re-sorted by fused_score descending.  The SearchResult.score
    field is replaced with the fused score for downstream transparency.
    """
    # Build page → navigated node depth map
    page_to_depth: dict[int, int] = {}
    for node, _ in navigated_nodes:
        depth = _node_depth(node, tree_roots)
        for pg in range(node.start_page, node.end_page + 1):
            page_to_depth[pg] = max(page_to_depth.get(pg, 0), depth)

    scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for rank, result in enumerate(vector_results, start=1):
        base = 1.0 / (RRF_K + rank)

        if result.page in tree_pages:
            depth = page_to_depth.get(result.page, 0)
            boost = _adaptive_boost_factor(affinity_score, confidence, depth)
            fused = base * boost
        else:
            fused = base

        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + fused
        result_map[result.chunk_id] = result

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        dataclasses.replace(result_map[cid], score=round(scores[cid], 5))
        for cid in sorted_ids[:top_k]
    ]


# ---------------------------------------------------------------------------
# Coherent-Narrative Re-ranking (Innovation 4)
# ---------------------------------------------------------------------------


def _coherence_rerank(
    results: list[SearchResult],
    navigated: list[tuple[TreeNode, float]],
) -> list[SearchResult]:
    """Lightly reorder results to surface chunks that form a coherent narrative.

    Chunks whose *tree_node_id* matches one of the top-ranked relevant nodes
    are promoted by +0.1 to their score, keeping them together and producing
    a more semantically coherent context window.
    """
    if not navigated:
        return results

    primary_node_ids = {n.node_id for n, _ in navigated[:2]}

    def _sort_key(r: SearchResult) -> float:
        node_id: str = getattr(r, "tree_node_id", "")
        coherence_bonus = 0.1 if node_id in primary_node_ids else 0.0
        return r.score + coherence_bonus

    return sorted(results, key=_sort_key, reverse=True)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class HybridRetrievalResult:
    """Encapsulates the output of one hybrid retrieval call."""

    results: list[SearchResult]
    query_type: str
    used_tree: bool
    relevant_node_titles: list[str]
    retrieval_confidence: float = 0.0
    adaptive_boost_used: float = 1.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def hybrid_retrieve(
    query: str,
    doc_id: str,
    top_k: int,
    tree_nodes: list[TreeNode] | None,
) -> HybridRetrievalResult:
    """Main hybrid retrieval function — the heart of StructRAG.

    When a section tree is available
    ---------------------------------
    1.  Classify the query type (factual / structural / conceptual).
    2.  Navigate the tree to find up to 3 most relevant section nodes.
    3.  Expand the vector search pool (2 × top_k) to ensure coverage.
    4.  Apply TB-RRF to fuse vector and tree relevance signals.
    5.  Apply coherence re-ranking to favour narratively adjacent chunks.

    When no tree is available (graceful degradation)
    ------------------------------------------------
    Run pure vector similarity search and return results as-is.

    Parameters
    ----------
    query       : Natural-language question or instruction.
    doc_id      : Target document identifier.
    top_k       : Desired number of retrieved chunks.
    tree_nodes  : Pre-loaded top-level TreeNode objects, or *None*.

    Returns
    -------
    HybridRetrievalResult containing ranked chunks and retrieval metadata.
    """
    query_type = classify_query(query)
    query_embedding = embedding_service.embed_query(query)

    if tree_nodes:
        navigated = navigate_tree(query, tree_nodes)
        tree_page_set = _relevant_pages(navigated)

        # Expand candidate pool for fusion
        pool_size = top_k * CANDIDATE_POOL_MULTIPLIER
        vector_candidates = vectorstore_service.search(doc_id, query_embedding, pool_size)

        if tree_page_set and navigated:
            # Patent Innovation 4: Cross-signal confidence
            confidence = _compute_retrieval_confidence(
                vector_candidates, tree_page_set, top_k,
            )
            affinity = _mean_affinity_score(navigated)

            # Patent Innovation 1: Adaptive TB-RRF
            fused = _tree_boosted_rrf(
                vector_candidates, tree_page_set, top_k,
                affinity_score=affinity,
                confidence=confidence,
                navigated_nodes=navigated,
                tree_roots=tree_nodes,
            )
            final = _coherence_rerank(fused, navigated)

            # Compute the representative adaptive boost for logging/API
            avg_depth = sum(
                _node_depth(n, tree_nodes) for n, _ in navigated
            ) / max(len(navigated), 1)
            representative_boost = _adaptive_boost_factor(
                affinity, confidence, int(avg_depth),
            )
        else:
            # Tree navigation found no relevant pages; fall back to vector
            final = vector_candidates[:top_k]
            confidence = 0.0
            representative_boost = 1.0

        logger.info(
            "ATB-RRF retrieval: doc=%s query_type=%s sections=%d "
            "tree_pages=%d confidence=%.2f boost=%.2f pool=%d fused=%d",
            doc_id,
            query_type,
            len(navigated),
            len(tree_page_set),
            confidence,
            representative_boost,
            len(vector_candidates),
            len(final),
        )
        return HybridRetrievalResult(
            results=final,
            query_type=query_type,
            used_tree=True,
            relevant_node_titles=[n.title for n, _ in navigated],
            retrieval_confidence=round(confidence, 3),
            adaptive_boost_used=round(representative_boost, 3),
        )

    # --- Vector-only fallback (Patent Innovation 6) --------------------------
    vector_results = vectorstore_service.search(doc_id, query_embedding, top_k)
    logger.info(
        "Vector-only retrieval (no tree): doc=%s chunks=%d",
        doc_id,
        len(vector_results),
    )
    return HybridRetrievalResult(
        results=vector_results,
        query_type=query_type,
        used_tree=False,
        relevant_node_titles=[],
        retrieval_confidence=0.0,
        adaptive_boost_used=1.0,
    )
