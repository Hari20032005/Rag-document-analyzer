"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the RAG academic explainer backend."""

    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG Academic Explainer"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:5173"

    uploads_dir: str = "./data/uploads"
    metadata_db_path: str = "./data/metadata.db"

    chroma_host: str = "chroma"
    chroma_port: int = 8000
    chroma_collection: str = "academic_chunks"
    chroma_persist_dir: str = "./chromadb_data"
    use_external_chroma: bool = False

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: Literal["openai", "hf", "gemini", "mock"] = "mock"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-flash-latest"
    gemini_fallback_models: str = ""
    hf_model_name: str = "google/flan-t5-base"
    llm_request_timeout_seconds: float = 90.0
    llm_request_retries: int = 2
    llm_max_requests_per_minute: int = 5

    default_top_k: int = 5
    max_top_k: int = 10

    rate_limit_per_minute: int = 60

    @field_validator("uploads_dir", "chroma_persist_dir", mode="before")
    @classmethod
    def normalize_paths(cls, value: str) -> str:
        return str(Path(value).expanduser())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    config_path = Path(__file__).resolve()
    # Support running from either project root or backend/ directory.
    env_candidates = [
        Path.cwd() / ".env",
        config_path.parents[3] / ".env",  # repo root
        config_path.parents[2] / ".env",  # backend/
    ]
    env_file = next((path for path in env_candidates if path.exists()), None)

    settings = Settings(_env_file=str(env_file) if env_file else None)
    Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.metadata_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return settings
