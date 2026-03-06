"""Tests for embeddings and API integration flow."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services.embeddings import EmbeddingService


client = TestClient(app)


def test_embeddings_pipeline_can_be_mocked(monkeypatch) -> None:
    """Embedding service should return mocked values when patched."""

    service = EmbeddingService()

    def fake_batch(batch: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in batch]

    monkeypatch.setattr(service, "_embed_batch", fake_batch)
    vectors = service.embed_texts(["alpha", "beta"])

    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_upload_status_and_ask_flow(monkeypatch) -> None:
    """Integration smoke test: upload -> status -> ask with mocked LLM output."""

    from app.services import rag

    def fake_generate(system_prompt: str, user_prompt: str, temperature: float, chunks):
        return "Simple explanation.\n- Concept A\n- Concept B\n- Concept C\n\nTechnical summary with citations."

    monkeypatch.setattr(rag, "_generate_answer", fake_generate)

    sample_pdf = Path(__file__).resolve().parents[3] / "EXAMPLE_PAPERS" / "sample_paper.pdf"

    with sample_pdf.open("rb") as pdf_file:
        upload_response = client.post(
            "/api/v1/upload",
            files={"file": ("sample_paper.pdf", pdf_file, "application/pdf")},
            data={"title": "Sample Paper"},
        )

    assert upload_response.status_code == 202
    job_id = upload_response.json()["job_id"]

    status_payload = None
    for _ in range(20):
        status_response = client.get(f"/api/v1/status/{job_id}")
        assert status_response.status_code == 200
        status_payload = status_response.json()
        if status_payload["status"] == "completed":
            break
        time.sleep(0.05)

    assert status_payload is not None
    assert status_payload["status"] == "completed"
    doc_id = status_payload["doc_id"]

    ask_response = client.post(
        "/api/v1/ask",
        json={
            "doc_id": doc_id,
            "question": "What is the paper about?",
            "top_k": 3,
            "mode": "qa",
            "temperature": 0.2,
        },
    )

    assert ask_response.status_code == 200
    payload = ask_response.json()
    assert "Simple explanation" in payload["answer"]
    assert isinstance(payload["sources"], list)
    assert "Context" in payload["prompt_used"]
