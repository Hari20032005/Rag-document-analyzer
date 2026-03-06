"""ChromaDB vector store abstraction."""

from __future__ import annotations

from dataclasses import dataclass
import time

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.chunker import ChunkRecord

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Single vector search result."""

    chunk_id: str
    page: int
    snippet: str
    score: float
    text: str
    doc_title: str


class VectorStoreService:
    """Thin wrapper around Chroma collection operations."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client: chromadb.ClientAPI | None = None
        self.collection: Collection | None = None

    def _ensure_collection(self) -> Collection:
        if self.collection is not None:
            return self.collection

        if self.client is None:
            if self.settings.use_external_chroma:
                self.client = chromadb.HttpClient(host=self.settings.chroma_host, port=self.settings.chroma_port)
            else:
                self.client = chromadb.PersistentClient(path=self.settings.chroma_persist_dir)

        retries = 3
        for attempt in range(1, retries + 1):
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.settings.chroma_collection,
                    metadata={"hnsw:space": "cosine"},
                )
                return self.collection
            except Exception:
                if attempt == retries:
                    raise
                logger.warning("Vector store unavailable; retrying connection (%d/%d)", attempt, retries)
                time.sleep(1)
        raise RuntimeError("Unable to initialize vector store collection")

    def add_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        """Insert chunk vectors and metadata."""

        collection = self._ensure_collection()
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "page": chunk.page,
                    "snippet": chunk.snippet,
                }
                for chunk in chunks
            ],
        )

    def search(self, doc_id: str, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        """Search top-k relevant chunks for a document."""

        collection = self._ensure_collection()
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"doc_id": doc_id},
        )

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        matches: list[SearchResult] = []
        for idx, chunk_id in enumerate(ids):
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            similarity = max(0.0, 1.0 - distance)
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            matches.append(
                SearchResult(
                    chunk_id=chunk_id,
                    page=int(metadata.get("page", 0)),
                    snippet=str(metadata.get("snippet", "")),
                    score=similarity,
                    text=str(docs[idx]) if idx < len(docs) else "",
                    doc_title=str(metadata.get("doc_title", "Document")),
                )
            )

        return matches

    def delete_doc(self, doc_id: str) -> None:
        """Delete all vectors for a document."""

        collection = self._ensure_collection()
        collection.delete(where={"doc_id": doc_id})


vectorstore_service = VectorStoreService()
