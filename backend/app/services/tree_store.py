"""Persistent storage for document section trees (StructRAG dual-index).

Extends the existing *metadata.db* SQLite database with a *document_trees*
table.  Each row stores the full JSON-serialised tree for one document.

Design rationale
----------------
* Co-locating tree data in the same SQLite file as job/document metadata keeps
  the deployment footprint small — no extra database service required.
* A single shared *TreeStore* singleton is imported wherever tree data is
  needed (upload pipeline, retrieval, API layer).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.tree_index import TreeNode

logger = get_logger(__name__)


class TreeStore:
    """SQLite-backed repository for hierarchical section trees."""

    def __init__(self) -> None:
        settings = get_settings()
        self._conn = sqlite3.connect(settings.metadata_db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_table()

    def _init_table(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_trees (
                    doc_id     TEXT PRIMARY KEY,
                    tree_json  TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_tree(self, doc_id: str, nodes: list[TreeNode]) -> None:
        """Persist (or replace) the section tree for *doc_id*."""
        tree_data = [node.to_dict() for node in nodes]
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO document_trees (doc_id, tree_json, created_at)
                VALUES (?, ?, ?)
                """,
                (doc_id, json.dumps(tree_data), now),
            )
            self._conn.commit()
        logger.info("Tree saved for doc_id=%s (%d nodes)", doc_id, len(nodes))

    def delete_tree(self, doc_id: str) -> None:
        """Remove the tree for *doc_id* (called during document deletion)."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM document_trees WHERE doc_id = ?", (doc_id,)
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_tree(self, doc_id: str) -> list[TreeNode] | None:
        """Return the list of top-level *TreeNode* objects, or *None* if absent."""
        with self._lock:
            row = self._conn.execute(
                "SELECT tree_json FROM document_trees WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        if row is None:
            return None
        data: list[dict] = json.loads(row["tree_json"])
        return [TreeNode.from_dict(n) for n in data]

    def tree_exists(self, doc_id: str) -> bool:
        """Return *True* when a tree has been indexed for *doc_id*."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM document_trees WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return row is not None


tree_store = TreeStore()
