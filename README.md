# RAG-based Academic Document Explainer System

A production-ready full-stack repository for uploading academic PDFs, indexing them with embeddings in ChromaDB, and asking grounded questions with citations using Retrieval-Augmented Generation (RAG).

## Features
- PDF upload with background ingestion job (`/upload` + `/status/{job_id}`)
- PDF parsing with `pypdf` and `unstructured` fallback
- Recursive chunking with overlap and page-aware metadata
- Embedding generation with `sentence-transformers/all-MiniLM-L6-v2` (hash fallback if model unavailable)
- Chroma vector search with per-document filtering
- RAG Q&A, section explanations, and summarization endpoints
- Source citations with `chunk_id`, `page`, `snippet`, and similarity score
- React + TypeScript + Tailwind UI with upload flow and chat interface
- Docker/Docker Compose for local development
- Backend and frontend test coverage + GitHub Actions CI

## Repository Layout
```text
backend/
  app/
    main.py
    api/v1/{upload.py,docs.py,qa.py}
    services/{loader.py,chunker.py,embeddings.py,vectorstore.py,rag.py,jobs.py}
    models/schemas.py
    core/{config.py,logger.py}
    tests/{test_upload.py,test_ask.py}
  Dockerfile
  requirements.txt
frontend/
  src/
    App.tsx
    main.tsx
    pages/{UploadPage.tsx,ChatPage.tsx,DocsPage.tsx}
    components/{FileUploader.tsx,ChatBox.tsx,MessageBubble.tsx,DocList.tsx,Loader.tsx,SourceHighlight.tsx}
    services/api.ts
    styles/tailwind.css
    tests/{FileUploader.test.tsx,ChatBox.test.tsx}
  Dockerfile
scripts/
  start-local.sh
  init-db.sh
EXAMPLE_PAPERS/sample_paper.pdf
.github/workflows/ci.yml
USAGE.md
```

## Prerequisites
- Docker + Docker Compose
- Python 3.11+
- Node.js 20+

## Environment Variables
Copy `.env.example` to `.env` at repository root:

```bash
cp .env.example .env
```

`./.env.example` includes:

```env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=
GEMINI_MODEL=gemini-flash-latest
GEMINI_FALLBACK_MODELS=gemini-1.5-flash
LLM_PROVIDER=openai
CHROMA_PERSIST_DIR=./chromadb_data
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
FRONTEND_URL=http://localhost:5173
USE_EXTERNAL_CHROMA=true
CHROMA_HOST=chroma
CHROMA_PORT=8000
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
OPENAI_MODEL=gpt-4o-mini
HF_MODEL_NAME=google/flan-t5-base
RATE_LIMIT_PER_MINUTE=60
LLM_REQUEST_TIMEOUT_SECONDS=90
LLM_REQUEST_RETRIES=2
LLM_MAX_REQUESTS_PER_MINUTE=5
```

## Run With Docker Compose

```bash
./scripts/start-local.sh
```

Services:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- Chroma: `http://localhost:8001`
- Redis (optional): `localhost:6379`

Optional one-time DB init:

```bash
./scripts/init-db.sh
```

## Run Backend Locally (without Docker)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run Frontend Locally (without Docker)

```bash
cd frontend
npm install
# Optional when not using Docker (default is already localhost):
# export VITE_PROXY_TARGET=http://localhost:8000
# Optional for slow LLM responses:
# export VITE_LLM_TIMEOUT_MS=120000
npm run dev
```

## API Summary

### 1) Upload PDF
`POST /api/v1/upload`
- `multipart/form-data`: `file` (pdf), `title` (optional)
- returns: `{"job_id": "...", "status": "processing"}`

### 2) Job Status
`GET /api/v1/status/{job_id}`
- returns processing/completed/failed with progress and counts

### 3) List Docs
`GET /api/v1/docs`
- returns all uploaded documents

### 4) Delete Doc
`DELETE /api/v1/docs/{doc_id}`
- deletes metadata + vectors + stored file

### 5) Ask
`POST /api/v1/ask`
- body: `{"doc_id":"...","question":"...","top_k":5,"mode":"qa","temperature":0.2}`
- returns answer, sources, prompt

### 6) Explain Section
`POST /api/v1/explain-section`
- body: `{"doc_id":"...","section":"introduction"}`

### 7) Summarize
`POST /api/v1/summarize`
- body: `{"doc_id":"...","length":"medium"}`

## Example curl Flow
See [`USAGE.md`](./USAGE.md) for a complete sequence (upload -> status poll -> ask).

## Tests

### Backend
```bash
cd backend
pytest -q
```

### Frontend
```bash
cd frontend
npm install
npm run test
```

## CI
GitHub Actions workflow at `.github/workflows/ci.yml` runs:
- backend pytest
- frontend lint + tests + build
- backend/frontend Docker image builds

## Scaling Notes
- Move vector storage to managed Chroma/pgvector for larger corpora
- Store metadata in PostgreSQL for multi-user durability
- Add async workers (Celery/RQ) for heavier ingestion
- Cache embeddings/chunk hashes to reduce repeat compute
- Control LLM cost with top-k limits, response truncation, and summarization fallback

## Security and Reliability Notes
- Secrets are only loaded from environment variables
- Basic in-memory rate limiter enabled
- CORS restricted to configured frontend URL
- Disallowed medical/legal high-stakes prompts return safety disclaimer
- Retrieval and LLM latency are logged for observability
# Rag-document-analyzer
