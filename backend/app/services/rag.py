"""RAG orchestration: retrieval, prompting, safety checks, and LLM generation."""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass

import httpx

from app.core.config import get_settings
from app.core.logger import get_logger
from app.models.schemas import SourceChunk
from app.services.embeddings import embedding_service
from app.services.vectorstore import SearchResult, vectorstore_service

logger = get_logger(__name__)
_llm_window: deque[float] = deque()
_llm_window_lock = threading.Lock()

SYSTEM_PROMPT = (
    "You are an academic tutor for an undergraduate/graduate student. "
    "Use the following retrieved document snippets to answer the user's question. Always:\n"
    "1) Provide a short simple explanation (2-5 sentences).\n"
    "2) Provide 3 bullet key concepts (short).\n"
    "3) Provide a 1-paragraph technical summary.\n"
    "4) Provide explicit citations: for each claim, list [doc_title,page] and the snippet id where the supporting text appears.\n"
    "If the answer is not contained in the retrieved context, say 'I don't know — the document doesn't contain enough information' and avoid hallucination."
)

USER_PROMPT_TEMPLATE = (
    "Context (only use to answer, do not invent facts):\\n{retrieved_chunks_formatted}\\n\\n"
    "User question: {user_question}\\n\\n"
    "Answer using the rules above."
)

DISALLOWED_PATTERNS = [
    r"medical advice",
    r"legal advice",
    r"diagnose",
    r"prescribe",
    r"lawsuit",
    r"criminal liability",
]


@dataclass
class RAGResponse:
    """RAG output payload."""

    answer: str
    sources: list[SourceChunk]
    prompt_used: str


def _is_disallowed_query(question: str) -> bool:
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in DISALLOWED_PATTERNS)


def _format_chunks(chunks: list[SearchResult]) -> str:
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            f"{index}. chunk_id={chunk.chunk_id}; page={chunk.page}; doc_title={chunk.doc_title}; snippet={chunk.snippet}"
        )
    return "\n".join(lines)


def _extract_key_concepts(answer: str) -> list[str]:
    concepts: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            concepts.append(stripped[2:].strip())
    return concepts[:3]


def _mock_generate(prompt: str, chunks: list[SearchResult]) -> str:
    if not chunks:
        return "I don't know — the document doesn't contain enough information."

    citations = ", ".join(f"[{chunk.doc_title},{chunk.page}] ({chunk.chunk_id})" for chunk in chunks[:3])
    return (
        "The document suggests this topic can be explained from the retrieved sections.\n"
        "- Main concept from the introduction and definitions\n"
        "- Methodological detail highlighted in context\n"
        "- Reported outcomes and implications\n\n"
        "Technical summary: The retrieved chunks indicate the paper's objective, method, and findings, "
        f"with support from {citations}."
    )


def _enforce_llm_rate_limit() -> None:
    settings = get_settings()
    max_requests = settings.llm_max_requests_per_minute
    if max_requests <= 0:
        return

    now = time.time()
    with _llm_window_lock:
        while _llm_window and now - _llm_window[0] > 60:
            _llm_window.popleft()

        if len(_llm_window) >= max_requests:
            raise RuntimeError(
                f"LLM rate limit reached ({max_requests}/min). Please wait a minute and retry."
            )

        _llm_window.append(now)


def _call_openai(system_prompt: str, user_prompt: str, temperature: float) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

    client = OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_request_timeout_seconds, max_retries=0)
    retries = max(1, settings.llm_request_retries)
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except RateLimitError as exc:
            if attempt < retries:
                time.sleep(1.5 * attempt)
                continue
            raise RuntimeError("OpenAI rate limit reached. Please retry in about a minute.") from exc
        except APITimeoutError as exc:
            if attempt < retries:
                time.sleep(1.0 * attempt)
                continue
            raise RuntimeError("OpenAI request timed out. Please retry.") from exc
        except APIConnectionError as exc:
            if attempt < retries:
                time.sleep(1.0 * attempt)
                continue
            raise RuntimeError("OpenAI connection failed. Please check network and retry.") from exc
        except APIStatusError as exc:
            if exc.status_code in {500, 502, 503, 504} and attempt < retries:
                time.sleep(1.5 * attempt)
                continue
            raise RuntimeError(f"OpenAI API error {exc.status_code}.") from exc

    raise RuntimeError("OpenAI request failed after retries")


def _call_hf(system_prompt: str, user_prompt: str, temperature: float) -> str:
    settings = get_settings()
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("transformers is not installed") from exc

    prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    generator = pipeline("text2text-generation", model=settings.hf_model_name)
    output = generator(prompt, max_new_tokens=350, temperature=temperature)
    return output[0]["generated_text"]


def _call_gemini(system_prompt: str, user_prompt: str, temperature: float) -> str:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"System instructions:\n{system_prompt}\n\nUser prompt:\n{user_prompt}",
                    }
                ]
            }
        ],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 900},
    }
    headers = {"Content-Type": "application/json", "X-goog-api-key": settings.gemini_api_key}

    timeout = httpx.Timeout(connect=10.0, read=settings.llm_request_timeout_seconds, write=20.0, pool=10.0)
    fallback_models = [item.strip() for item in settings.gemini_fallback_models.split(",") if item.strip()]
    models_to_try = [settings.gemini_model, *fallback_models]
    retryable_statuses = {429, 500, 503}

    last_error: Exception | None = None
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for attempt in range(1, settings.llm_request_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    body = response.json()

                candidates = body.get("candidates", [])
                if not candidates:
                    raise RuntimeError(f"Gemini returned no candidates for model {model}: {body}")

                parts = candidates[0].get("content", {}).get("parts", [])
                texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
                combined = "\n".join(text.strip() for text in texts if text.strip())
                if not combined:
                    raise RuntimeError(f"Gemini returned empty text for model {model}")
                return combined
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as exc:
                last_error = exc
                logger.warning(
                    "Gemini timeout/network error model=%s attempt=%d/%d: %s",
                    model,
                    attempt,
                    settings.llm_request_retries,
                    exc,
                )
                if attempt < settings.llm_request_retries:
                    time.sleep(1.5 * attempt)
                    continue
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                detail = exc.response.text[:300]
                logger.warning(
                    "Gemini HTTP error model=%s status=%d attempt=%d/%d",
                    model,
                    status,
                    attempt,
                    settings.llm_request_retries,
                )
                if status in retryable_statuses and attempt < settings.llm_request_retries:
                    time.sleep(1.5 * attempt)
                    continue
                if status in retryable_statuses:
                    break
                raise RuntimeError(f"Gemini API returned {status}: {detail}") from exc

    raise RuntimeError(f"Gemini unavailable across models {models_to_try}: {last_error}")


def _generate_answer(system_prompt: str, user_prompt: str, temperature: float, chunks: list[SearchResult]) -> str:
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "openai":
        return _call_openai(system_prompt, user_prompt, temperature)
    if provider == "gemini":
        return _call_gemini(system_prompt, user_prompt, temperature)
    if provider == "hf":
        return _call_hf(system_prompt, user_prompt, temperature)
    return _mock_generate(user_prompt, chunks)


def answer_question(doc_id: str, question: str, top_k: int, mode: str, temperature: float = 0.2) -> RAGResponse:
    """Retrieve relevant chunks and generate a grounded answer."""

    if _is_disallowed_query(question):
        disclaimer = (
            "This request appears to involve high-stakes medical or legal guidance. "
            "Please consult a qualified expert before relying on this information."
        )
        return RAGResponse(answer=disclaimer, sources=[], prompt_used="")

    query_start = time.perf_counter()
    query_embedding = embedding_service.embed_query(question)
    results = vectorstore_service.search(doc_id=doc_id, query_embedding=query_embedding, top_k=top_k)
    retrieval_ms = (time.perf_counter() - query_start) * 1000
    logger.info("retrieval completed in %.2f ms for doc_id=%s top_k=%d", retrieval_ms, doc_id, top_k)

    formatted_chunks = _format_chunks(results)
    user_question = f"[{mode}] {question}"
    user_prompt = USER_PROMPT_TEMPLATE.format(
        retrieved_chunks_formatted=formatted_chunks,
        user_question=user_question,
    )

    llm_start = time.perf_counter()
    try:
        _enforce_llm_rate_limit()
        answer = _generate_answer(SYSTEM_PROMPT, user_prompt, temperature, results)
    except Exception as exc:
        logger.exception("LLM provider failed")
        answer = (
            "The language model request failed before completion. "
            "Please retry this question. If it keeps failing, reduce Top K to 3 and try again.\n\n"
            f"Provider error: {exc}"
        )
    llm_ms = (time.perf_counter() - llm_start) * 1000
    logger.info("llm generation completed in %.2f ms provider=%s", llm_ms, get_settings().llm_provider)

    sources = [
        SourceChunk(
            page=result.page,
            snippet=result.snippet,
            chunk_id=result.chunk_id,
            score=result.score,
        )
        for result in results
    ]

    return RAGResponse(answer=answer, sources=sources, prompt_used=user_prompt)


def summarize_document(doc_id: str, length: str, temperature: float = 0.2) -> tuple[RAGResponse, list[str]]:
    """Generate summary and parse key concepts from the answer format."""

    prompt = (
        f"Provide a {length} document summary with the required explanation format. "
        "Focus on central thesis, methods, results, and limitations."
    )
    rag_response = answer_question(doc_id=doc_id, question=prompt, top_k=get_settings().default_top_k, mode="summary", temperature=temperature)
    key_concepts = _extract_key_concepts(rag_response.answer)
    return rag_response, key_concepts
