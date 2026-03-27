"""FastAPI application entrypoint."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1.docs import router as docs_router
from app.api.v1.qa import router as qa_router
from app.api.v1.tree import router as tree_router
from app.api.v1.upload import router as upload_router
from app.core.config import get_settings
from app.core.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
settings = get_settings()
logger.info("LLM provider configured: %s", settings.llm_provider)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory fixed-window rate limiter for local dev safety."""

    def __init__(self, app: FastAPI, max_requests_per_minute: int) -> None:
        super().__init__(app)
        self.max_requests_per_minute = max_requests_per_minute
        self.window_seconds = 60
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]):
        if request.url.path in {"/health", "/"}:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        bucket = self.requests[client_ip]

        while bucket and now - bucket[0] > self.window_seconds:
            bucket.popleft()

        if len(bucket) >= self.max_requests_per_minute:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

        bucket.append(now)
        return await call_next(request)


app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, max_requests_per_minute=settings.rate_limit_per_minute)

app.include_router(upload_router, prefix=settings.api_v1_prefix)
app.include_router(docs_router, prefix=settings.api_v1_prefix)
app.include_router(qa_router, prefix=settings.api_v1_prefix)
app.include_router(tree_router, prefix=settings.api_v1_prefix)


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint."""

    return {"message": "RAG Academic Explainer API"}


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health endpoint."""

    return {"status": "ok"}
