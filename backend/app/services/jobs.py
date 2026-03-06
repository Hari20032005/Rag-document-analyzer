"""Background job tracking and SQLite-backed metadata repository."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.config import get_settings


@dataclass
class DocumentRecord:
    """Document metadata record."""

    id: str
    title: str
    upload_date: datetime
    page_count: int
    file_path: str


class JobRepository:
    """Persisted repository for jobs and document metadata."""

    def __init__(self) -> None:
        settings = get_settings()
        self._conn = sqlite3.connect(settings.metadata_db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_tables()

    def _init_tables(self) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    pages_parsed INTEGER NOT NULL,
                    total_pages INTEGER NOT NULL,
                    chunks_created INTEGER NOT NULL,
                    doc_id TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    upload_date TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    file_path TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def create_job(self, job_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (id, status, progress, pages_parsed, total_pages, chunks_created, created_at, updated_at)
                VALUES (?, 'processing', 0, 0, 0, 0, ?, ?)
                """,
                (job_id, now, now),
            )
            self._conn.commit()

    def update_job(self, job_id: str, **updates: Any) -> None:
        if not updates:
            return

        allowed = {
            "status",
            "progress",
            "pages_parsed",
            "total_pages",
            "chunks_created",
            "doc_id",
            "error",
        }
        fields: list[str] = []
        values: list[Any] = []

        for key, value in updates.items():
            if key in allowed:
                fields.append(f"{key} = ?")
                values.append(value)

        if not fields:
            return

        fields.append("updated_at = ?")
        values.append(datetime.now(timezone.utc).isoformat())
        values.append(job_id)

        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?",
                tuple(values),
            )
            self._conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def add_document(self, record: DocumentRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO documents (id, title, upload_date, page_count, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.title,
                    record.upload_date.isoformat(),
                    record.page_count,
                    record.file_path,
                ),
            )
            self._conn.commit()

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        return DocumentRecord(
            id=row["id"],
            title=row["title"],
            upload_date=datetime.fromisoformat(row["upload_date"]),
            page_count=int(row["page_count"]),
            file_path=row["file_path"],
        )

    def list_documents(self) -> list[DocumentRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM documents ORDER BY upload_date DESC"
            ).fetchall()
        return [
            DocumentRecord(
                id=row["id"],
                title=row["title"],
                upload_date=datetime.fromisoformat(row["upload_date"]),
                page_count=int(row["page_count"]),
                file_path=row["file_path"],
            )
            for row in rows
        ]

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            result = self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._conn.commit()
        return result.rowcount > 0


job_repository = JobRepository()
