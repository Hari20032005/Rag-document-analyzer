# StructRAG Study Guide: RAG-Based Academic Document Explainer

Interview prep for the final Accenture round (technical + HR).
Project: **RAG-Based Academic Document Explainer (branded StructRAG)**, a solo project.
Repo state used: last commit `00d8936` (2026-08-13). Guide written 2026-10-09.

---

## How to read this guide

**Labels used on every claim**

| Label | Meaning |
|---|---|
| **FACT** | I read it directly in the repo files. A citation in square brackets shows where, e.g. `[backend/app/services/rag.py: answer_question()]`. |
| **FACT (ran it)** | I ran the repo's real code in a throwaway folder and observed it. The repo itself was not changed. |
| **INFERENCE** | A guess or general knowledge, not written in the repo. Verify before stating it as fact. |
| **NOT IN THE MATERIAL** | The repo does not say. Do not claim it in an interview. |
| **TEACHING EXAMPLE** | Invented numbers used only to explain an idea. Only constants from the code (like k = 60) are real. |
| **[Claim 1]**, **[Claim 2]** | Your two resume bullets (see Part 7.4). |

**How I checked the code.** I read every source file, config file and document. I also ran the repo's 4 backend tests (all passed) and small scripts against the real functions and API. The machine had no `sentence-transformers` or `unstructured` installed, so embeddings ran on the hash fallback. I used those runs only for logic (scores, limits, validation), never to judge answer quality. I did not run the frontend, and I did not open the live website.

**What is missing from the material (so it is NOT IN THE MATERIAL everywhere below)**
1. Anything about the live deployment: the Railway URL, production environment variables, volumes, which LLM provider production uses, "Gemini free tier". Your note that the backend returned "Application not found" on 27 Sep 2026 is your report. I could not check it.
2. Any retrieval evaluation (code, test questions or results). Nothing supports old resume percentages.
3. Facts about outside models: MiniLM's maximum input length, LLM context windows, prices, free-tier limits.
4. Real-paper behaviour. The only PDF is a 1-page, ~1 KB toy sample.
5. Your reasons for design choices. Every "why chosen" is INFERENCE.

---

## Contents

1. The big picture
2. Map of the codebase
3. Concepts from zero
4. End-to-end walkthroughs
5. Data and API reference
6. Design decisions and trade-offs
7. Weaknesses, bugs and honesty check (includes CORRECTIONS TO MY NOTES and the RESUME CLAIM AUDIT)
8. Pitch scripts
9. Interview question bank (40 questions)
10. Code-reading drill
11. Self-test and flash cards
12. Cheat sheet and revision plan

---
# Part 1. The big picture

## 1.1 The problem it solves

> **In simple words**
> Reading a 15-page research paper to find one answer is slow. This app lets you upload the PDF and ask questions in plain English. It finds the few paragraphs that matter and gives them to an AI to write the answer. It shows which pages the paragraphs came from.
> **Analogy:** an open-book exam with a helpful librarian. The librarian finds the right pages first. Then the student answers using only those pages.
> **Java comparison:** think of one method `answer(question)`. It first calls `repository.findRelevant(question)`, then passes the result to `llm.complete(prompt)`.

**FACT:** The project is described as a system for "uploading academic PDFs, indexing them with embeddings in ChromaDB, and asking grounded questions with citations using Retrieval-Augmented Generation (RAG)" [README.md: first paragraph].

**Who uses it.** The AI's instructions say it acts as "an academic tutor for an undergraduate/graduate student" [backend/app/services/rag.py: SYSTEM_PROMPT]. So the intended user is a student reading papers.
- **NOT IN THE MATERIAL:** any real user, user count or feedback. Never say "users liked it".

**What it does, in 4 steps (FACT):**
1. **Upload** a PDF. The server saves it and starts processing it in the background [backend/app/api/v1/upload.py: upload_pdf()].
2. **Index** it. The server reads the text page by page, cuts it into chunks, turns each chunk into numbers, and stores them [upload.py: process_document_job()].
3. **Ask** a question. The server finds the most similar chunks, builds a prompt, and calls an LLM [backend/app/services/rag.py: answer_question()].
4. **Answer with sources.** The reply contains the answer plus a list of source chunks, each with page, snippet, chunk id and score [backend/app/models/schemas.py: AskResponse, SourceChunk].

**Extra backend endpoints (FACT):**
- explain a section: `/explain-section`
- summarize: `/summarize`
- list and delete documents
- view the section tree: `/tree/{doc_id}` and `/tree/{doc_id}/navigate`

There are 9 endpoints under `/api/v1`, plus `/` and `/health` [backend/app/main.py; api/v1/*.py].

**FACT (ran grep):** the website never calls `/explain-section`, `/summarize` or the two `/tree` endpoints. `api.ts` defines functions for the first two, but no page uses them [frontend/src/services/api.ts: explainSection(), summarizeDocument()]. The website only uses upload, status, docs list, docs delete and ask. Do not say "the UI has a section explorer".

## 1.2 Words you will see in this part (one line each, full teaching in Part 3)

| Term | One-line meaning |
|---|---|
| **RAG** (Retrieval-Augmented Generation) | Look up relevant text first, then give it to an LLM with the question, so it answers from that text. |
| **LLM** (Large Language Model) | A program (like GPT or Gemini) that writes text when given a prompt. |
| **Chunk** | A piece of the document text, here up to 3000 characters. |
| **Embedding** | A list of numbers (here 384) that stands for the meaning of a text. Similar meaning gives similar numbers. |
| **Vector database** | A database that stores embeddings and quickly finds the nearest ones to a query. Here: ChromaDB. |
| **REST API** | A server you talk to with HTTP requests such as GET and POST, usually exchanging JSON. |
| **Background job** | Work the server continues after it has already replied to the browser. |
| **CORS** | A browser rule that blocks a web page from calling a different website unless that server allows it. |
| **StructRAG** | The project's brand name for its extra layer. It builds a section tree (Introduction, Methods, and so on) and uses it to adjust ranking. |

## 1.3 Architecture diagram

```
   USER (browser)
        |
        v
+-------------------------------+
| FRONTEND  React + TypeScript  |   hosted on Vercel
| Vite build, Tailwind styles   |   [frontend/vercel.json, README.md "Live url"]
| pages: Home, Upload, Docs,    |
|        Chat                   |
| services/api.ts (axios)       |
+---------------+---------------+
                | HTTPS + JSON   /api/v1/...
                v
+--------------------------------------------------------------+
| BACKEND  FastAPI app  [backend/app/main.py]                  |
| started by gunicorn + 1 uvicorn worker [backend/Dockerfile]  |
|                                                              |
|  middleware:  CORS  ->  per-IP rate limit (60/min default)   |
|  routers:     upload | qa | docs | tree                      |
|                                                              |
|  INGEST (background task)        ASK (request/response)      |
|  loader.py   read PDF pages      rag.py  answer_question()   |
|  tree_index  find headings       hybrid_retriever.py         |
|  chunker     3000/800 chars        embed query               |
|  embeddings  MiniLM, 384 nums      vector search             |
|  vectorstore save to Chroma        tree boost (ATB-RRF)      |
|                                  build prompt -> LLM         |
+-----+------------+----------------+----------------+---------+
      |            |                |                |
      v            v                v                v
 data/uploads/  data/metadata.db  ChromaDB        LLM provider
 original PDFs  SQLite: jobs,     vectors +       openai | gemini
                documents,        metadata        | hf | mock
                document_trees    (local folder
                                  or Chroma server)
```

**Notes on the diagram:**
- **FACT:** The embedding model runs inside the backend process. `sentence_transformers` is imported and loaded in the backend [backend/app/services/embeddings.py: _ensure_model()]. There is no separate embedding service.
- **FACT:** The default storage locations are `./data/uploads`, `./data/metadata.db` and `./chromadb_data` [backend/app/core/config.py: Settings]. With Docker Compose, Chroma runs as its own container (`USE_EXTERNAL_CHROMA=true`) [docker-compose.yml: chroma, backend].
- **FACT:** The `redis` service in `docker-compose.yml` is never used by any code. A search for "redis" finds it only in the compose file. Do not mention Redis as part of your design.
- **INFERENCE:** The backend runs on Railway. The repo has `backend/railway.json` and the Dockerfile reads `${PORT:-8000}`, but no Railway URL or setting is in the material.

## 1.4 Technology table

"Why chosen" is **INFERENCE** in every row, because the repo gives no reasons.

| Technology (version) | Its job in this project | Why probably chosen | One alternative |
|---|---|---|---|
| **FastAPI** 0.115.6 [requirements.txt] | REST routes, input validation, auto API docs, background tasks | Python type hints become validation. Quick to write. | Flask; Spring Boot (Java) |
| **Pydantic** 2.10.5 + pydantic-settings | Request/response models; reads env variables into `Settings` [schemas.py, config.py] | Comes with FastAPI | Plain dataclasses + manual checks |
| **gunicorn + uvicorn worker** [Dockerfile] | Runs the app in production style | Standard FastAPI deployment; `PORT` support for Railway | Plain `uvicorn` |
| **pypdf** 5.1.0 | Extracts text per page [loader.py: _extract_with_pypdf()] | Pure Python, easy | PyMuPDF, pdfplumber |
| **unstructured** 0.16.12 | Fallback parser if pypdf finds no text [loader.py: load_pdf()] | More layout-aware parsing | Run OCR (Tesseract) directly |
| **langchain-text-splitters** 0.3.4 | Only `RecursiveCharacterTextSplitter` is used [chunker.py] | Ready-made splitter | Write your own splitter |
| **sentence-transformers** 3.3.1, model `all-MiniLM-L6-v2` | Text to 384 numbers [config.py: embedding_model_name; PROJECT_DOCUMENTATION.md §2] | Small, free, no API key | OpenAI embeddings API |
| **ChromaDB** 0.5.23 | Stores vectors + metadata; nearest-neighbour search [vectorstore.py] | Easy to start, can run embedded, has metadata filter | pgvector, FAISS, Qdrant |
| **SQLite** (built in) | Tables `jobs`, `documents`, `document_trees` [jobs.py, tree_store.py] | Zero setup, one file | PostgreSQL |
| **OpenAI / Gemini / Hugging Face / mock** | Writes the answer [rag.py: _generate_answer()] | Free choice of vendor; mock for tests | Use only one vendor |
| **tenacity** 9.0.0 | Retries embedding batches 3 times [embeddings.py: _embed_batch()] | Short retry decorator | Hand-written loop |
| **httpx** 0.28.1 | Calls the Gemini REST API [rag.py: _call_gemini()] | Modern HTTP client | requests |
| **React** ^18.3.1 + **TypeScript** ^5.7.2 [package.json] | Builds the UI | Popular; types catch mistakes | Vue, Angular |
| **Vite** ^6.0.5 | Dev server + production build | Fast | webpack |
| **Tailwind CSS** ^3.4.17 | Styling with utility classes | Quick styling | Plain CSS, Bootstrap |
| **Axios** ^1.7.9 | HTTP calls; gives upload-progress events [api.ts: uploadPdf()] | Progress callback built in | `fetch` |
| **React Router** ^6.28.1 | Pages and URLs such as `/chat/:docId` [App.tsx] | Standard for React SPAs | Next.js routing |
| **Vitest** ^2.1.8 + Testing Library | Frontend tests [src/tests/] | Works with Vite | Jest |
| **Docker + Compose** | Run backend, Chroma, frontend (and an unused Redis) together [docker-compose.yml] | Same setup on every machine | Run services by hand |
| **GitHub Actions** | CI: backend tests; frontend lint, test, build; Docker image builds [.github/workflows/ci.yml] | Repo is on GitHub | GitLab CI, Jenkins |
| **Vercel / Railway** | Host the frontend / run the backend [vercel.json, railway.json] | Easy Git-based deploys | Netlify / Render, Fly.io |

> **Version note (CORRECTION, details in Part 7):** `PROJECT_DOCUMENTATION.md` says "React 19" and "Python 3.10". The code says React `^18.3.1` [frontend/package.json] and Python 3.11 [backend/Dockerfile, ci.yml]. The code wins.

## 1.5 What "StructRAG" adds on top of plain RAG

**FACT:** After the PDF text is read, the code also builds a **section tree**. Each node holds a heading, a page range, a short summary and a section type from a list of 13 (abstract, introduction, methodology, results, and so on) [backend/app/services/tree_index.py: build_document_tree(), SectionType].

At question time, the code:
- picks up to 3 sections that look relevant
- multiplies the rank score of chunks on those pages by a boost between 1.0 and 2.0 [backend/app/services/hybrid_retriever.py: hybrid_retrieve(), _tree_boosted_rrf()]

The brand name for this is **ATB-RRF** (Adaptive Tree-Boosted Reciprocal Rank Fusion).

**What this project is NOT (FACT, ran it or read it; evidence in Part 7):**
- It has **no login and no users**. All documents sit in one shared library.
- It does **not send chat history** to the LLM, so each question is independent.
- It has **no retrieval evaluation**. Nothing measures whether StructRAG beats plain vector search.

## 1.6 How to say it in an interview

> **Short description.** "I built a web app where you upload a research paper as a PDF and ask questions about it. The backend splits the paper into chunks, converts them into embeddings, and stores them in a vector database called ChromaDB. When you ask something, it finds the most relevant chunks, gives them to an LLM, and returns the answer with page-level sources. The backend is FastAPI in Python and the frontend is React with TypeScript."

> **Architecture in three sentences.** "The frontend talks to a FastAPI backend over a REST API. Uploading is asynchronous: the server replies right away with a job id and does the parsing, chunking and embedding in the background, and the browser polls for progress. For questions, the server does the vector search, builds a prompt from the retrieved chunks, and calls a pluggable LLM provider, which is OpenAI, Gemini, Hugging Face or a mock for testing."

> **If asked what is special.** "On top of normal vector search I added a section-tree layer, which I called StructRAG. It detects headings, labels each section as, say, methodology or results, and boosts chunks from sections that match the question. I should be honest that I haven't run a formal evaluation to show how much it improves retrieval."

---

# Part 2. Map of the codebase

## 2.1 Folder tree

Line counts are from the files. Tests, `__init__.py` files and the lock file are left out.

```
Rag-document-analyzer/
├── README.md                    setup + API summary (its file list is out of date)
├── USAGE.md                     curl walk-through: upload, poll, ask
├── PROJECT_DOCUMENTATION.md     long design doc; the .docx holds the same text
├── .env.example                 env variable names (placeholders only, no real keys)
├── docker-compose.yml           chroma + backend + frontend (+ redis, unused)
├── .github/workflows/ci.yml     CI: backend / frontend / docker jobs
├── EXAMPLE_PAPERS/sample_paper.pdf   1-page toy PDF used by the tests
├── diagrams/*.mmd, *.png        6 Mermaid diagrams (architecture, ingestion, tree, ATB-RRF...)
├── scripts/                     start-local.sh (docker compose up --build), init-db.sh
│
├── backend/
│   ├── Dockerfile               python:3.11-slim, runs gunicorn + uvicorn worker
│   ├── railway.json             Railway: build from Dockerfile, restart on failure (max 3)
│   ├── requirements.txt         17 pinned packages
│   ├── pytest.ini               pythonpath=. ; testpaths=app/tests
│   └── app/
│       ├── main.py              (82)  FastAPI app, CORS, rate limiter, routers, /health
│       ├── core/
│       │   ├── config.py        (76)  Settings class: every env variable + default
│       │   └── logger.py        (36)  logging setup
│       ├── models/
│       │   └── schemas.py       (171) Pydantic request/response models
│       ├── api/v1/              HTTP layer (thin)
│       │   ├── upload.py        (132) POST /upload, GET /status/{job_id}, ingestion job
│       │   ├── qa.py            (105) /ask, /explain-section, /summarize
│       │   ├── docs.py          (51)  GET /docs, DELETE /docs/{doc_id}
│       │   └── tree.py          (111) GET /tree/{doc_id}, POST /tree/{doc_id}/navigate
│       ├── services/            the real logic
│       │   ├── loader.py        (69)  PDF to pages (pypdf, unstructured fallback)
│       │   ├── chunker.py       (104) pages to chunks (3000/800) + section labels
│       │   ├── embeddings.py    (108) text to 384 numbers; cache; hash fallback
│       │   ├── vectorstore.py   (132) ChromaDB wrapper: add / search / delete
│       │   ├── jobs.py          (174) SQLite: jobs + documents tables
│       │   ├── tree_index.py    (481) section tree builder + 13-type ontology
│       │   ├── tree_store.py    (102) SQLite: document_trees table
│       │   ├── hybrid_retriever.py (540) query type, tree navigation, ATB-RRF
│       │   └── rag.py           (352) prompt, LLM providers, answer_question()
│       └── tests/               test_upload.py (35), test_ask.py (81)
│
└── frontend/
    ├── Dockerfile               node:20-alpine, runs `npm run dev` (dev server)
    ├── vercel.json              Vite build + "send every URL to index.html"
    ├── package.json, vite.config.ts, vitest.config.ts, tsconfig*.json, .eslintrc.cjs
    ├── tailwind.config.cjs, postcss.config.cjs
    └── src/
        ├── main.tsx (13)        mounts React + BrowserRouter
        ├── App.tsx (77)         header, nav, routes: / , /upload, /docs, /chat/:docId
        ├── services/api.ts (121) all HTTP calls (axios) + TypeScript types
        ├── pages/               HomePage (194, marketing), UploadPage (113),
        │                        DocsPage (67), ChatPage (157)
        ├── components/          ChatBox (164), FileUploader (116), MessageBubble (45),
        │                        SourceHighlight (38), DocList (56), PipelineDiagram (97),
        │                        Brand (27), Loader (8)
        ├── styles/tailwind.css  Tailwind layers + button/card classes
        └── tests/               ChatBox.test.tsx, FileUploader.test.tsx, setup.ts
```

**Where to spend study time (FACT, with the arithmetic).** The two StructRAG files are `hybrid_retriever.py` (540 lines) and `tree_index.py` (481 lines). Add them: 540 + 481 = 1,021. The 17 backend source files above total 2,826 lines (excluding tests and `__init__.py`). So 1,021 ÷ 2,826 = 0.361, about 36% of the backend. That is where interviewers will probe the "novel" claims.

## 2.2 Who imports whom (FACT, from the import lines)

```
api/v1/upload.py ──> loader, tree_index, chunker, embeddings, vectorstore, tree_store, jobs
api/v1/qa.py     ──> rag, jobs
api/v1/tree.py   ──> hybrid_retriever (classify_query, navigate_tree), tree_store, jobs
api/v1/docs.py   ──> jobs, tree_store, vectorstore

rag.py              ──> hybrid_retriever, tree_store   (also imports embeddings, vectorstore unused)
hybrid_retriever.py ──> embeddings, vectorstore, tree_index
chunker.py ──> loader (PageText)     vectorstore.py ──> chunker (ChunkRecord)
tree_store.py ──> tree_index (TreeNode)
```

**Layering:** `api/` (HTTP) calls `services/` (logic), which calls storage (SQLite, Chroma, the PDF file, an LLM API). Java comparison: Controller to Service to Repository.

**Singletons.** Four objects are created at import time, at the bottom of their files:
- `job_repository` [jobs.py]
- `tree_store` [tree_store.py]
- `embedding_service` [embeddings.py]
- `vectorstore_service` [vectorstore.py]

They act like Spring singleton beans. For example, `JobRepository()` opens the SQLite file the moment the module is imported. The embedding model itself loads later, on first use [embeddings.py: _ensure_model()].

## 2.3 Reading order (fastest way to understand the project)

| Step | Read | What you learn |
|---|---|---|
| 1 | `core/config.py`, `models/schemas.py` | Every setting, default, and API shape. Both are short. |
| 2 | `main.py` | How the app is assembled: CORS, rate limit, routers. |
| 3 | `api/v1/upload.py`, then `loader.py`, `chunker.py`, `embeddings.py`, `vectorstore.py`, `jobs.py` | The ingestion path, start to end. |
| 4 | `api/v1/qa.py`, then `rag.py` | The question path and the LLM providers. |
| 5 | `hybrid_retriever.py` (read `hybrid_retrieve()` first, then the helpers) | The StructRAG ranking. |
| 6 | `tree_index.py`, `tree_store.py`, `api/v1/tree.py`, `docs.py` | How the section tree is built and stored. |
| 7 | `tests/test_upload.py`, `tests/test_ask.py` | What is and is not tested. |
| 8 | Frontend: `api.ts`, `UploadPage.tsx`, `ChatPage.tsx`, `ChatBox.tsx`, `SourceHighlight.tsx` | How the UI uses the API. |
| 9 | `backend/Dockerfile`, `docker-compose.yml`, `ci.yml`, `vercel.json`, `railway.json` | Run, test and deploy. |
| 10 | `PROJECT_DOCUMENTATION.md` last | Read it as a list of claims to check against the code, not as the truth. |

**FACT:** the README file list is out of date. It omits `hybrid_retriever.py`, `tree_index.py`, `tree_store.py` and `api/v1/tree.py` [README.md: "Repository Layout"].

## 2.4 How to say it in an interview

> "The backend has three layers. The `api` folder holds thin FastAPI routes, the `services` folder holds the real logic, and below that are the stores: SQLite for jobs and metadata, ChromaDB for vectors, and the file system for the PDFs. Ingestion goes loader, chunker, embeddings, vector store, with a section-tree step added for my StructRAG layer. The question path goes through `rag.py`, which calls a hybrid retriever and then an LLM provider. The frontend is a small React app whose only job is to call these endpoints."

---

# Part 3. Concepts from zero

Each concept has an **In simple words** box, how it works, how this project uses it (with file path), and an interview answer.

Two labels on numbers and examples:
- **TEACHING EXAMPLE** means invented numbers used only to explain an idea. Only constants such as k = 60 come from the code.
- **FACT (ran it)** means I observed it by running the repo's real code in a throwaway folder.

## 3.1 What an LLM is: tokens, context window, temperature, prompt, hallucination

> **In simple words**
> An LLM is a very well-read autocomplete. You give it text, and it writes the most likely continuation, one small piece at a time. It sounds sure of itself even when it is wrong.
> **Analogy:** a fast, well-read intern who never says "I'm not sure" unless you tell them to.
> **Java comparison:** a service call `String complete(String prompt, double temperature)`.

**Token.** A token is a small piece of text, often a word or part of a word. Models read and write tokens. Limits and prices are counted in tokens.
- **FACT:** the code never counts tokens. It only sets an output cap: Gemini `maxOutputTokens: 4096` [rag.py: _call_gemini()], Hugging Face `max_new_tokens=350` [rag.py: _call_hf()]. For OpenAI no cap is set.

**Context window.** This is the maximum amount of text (prompt plus answer) the model can handle in one call.
- **NOT IN THE MATERIAL:** the actual window of `gpt-4o-mini` or the Gemini models.
- The project's own estimate is that 3000 characters is about 800 tokens [chunker.py: chunk_pages() docstring]. The most chunks allowed is `top_k` = 10 [schemas.py: AskRequest]. Chunk text alone then costs:
  - top_k = 5: 5 × 800 = 4,000 tokens
  - top_k = 10: 10 × 800 = 8,000 tokens
- The system prompt, question and answer come on top of that.
- **FACT:** no code checks that the prompt fits the model's window.

**Temperature.** This controls randomness. Low (0) is predictable. High (1) is more varied.
- **FACT:** the default is 0.2, allowed range 0.0 to 1.0 [schemas.py: AskRequest.temperature]. The UI slider goes 0 to 1 in steps of 0.1 [ChatBox.tsx].
- **INFERENCE:** a low default suits answering from a document, where you want stable, factual text.

**Prompt.** This is the text you send. Here it has two parts:
- A **system prompt** with rules. It asks for a 2–5 sentence explanation, 3 bullet key concepts, a technical-summary paragraph, and citations as `[doc_title,page]`. If the context lacks the answer, it must say "I don't know — the document doesn't contain enough information" [rag.py: SYSTEM_PROMPT].
- A **user prompt** with the chunks and the question [rag.py: USER_PROMPT_TEMPLATE].
- OpenAI receives them as separate system and user messages. Gemini receives them merged into one text block [rag.py: _call_gemini(), payload].

**Hallucination.** The model states something that is not true or not in the source.

| What the project does against it | What it does NOT do |
|---|---|
| Passes retrieved chunks and says "only use this" | Check that each claim appears in a chunk |
| Tells the model to say "I don't know" | Compute any groundedness score |
| Default temperature 0.2 | Verify that the LLM's cited page numbers are real |
| Returns the retrieved chunks as `sources` | The `sources` are what was retrieved, not proof of what the answer used |

**FACT (ran it):** `USER_PROMPT_TEMPLATE` is written with `\\n`, so the prompt contains the two characters backslash and n, not real line breaks. The chunk text between them does have real newlines. This is a small bug (Part 7).

**How to say it in an interview**
> "An LLM generates text one token at a time from a prompt. It can hallucinate, meaning it writes fluent but unsupported text. In my project I reduce that by giving it retrieved chunks, telling it to say 'I don't know' when the context is missing, and keeping temperature at 0.2. That reduces hallucination but doesn't eliminate it, because I don't verify the answer against the chunks afterwards."

## 3.2 Embeddings and cosine similarity

> **In simple words**
> An embedding turns a piece of text into a list of numbers, like coordinates on a map of meaning. Texts with similar meaning land close together.
> **Analogy:** GPS coordinates, but for meaning. "puppy" and "dog" are neighbours. "tax law" is far away.
> **Java comparison:** `double[] embed(String text)` returns an array of length 384. Similarity is a loop over two arrays.

**Cosine similarity** measures whether two arrows point the same way. It is the dot product divided by the two lengths. If both arrows have length 1, it is just the dot product.

```java
double dot = 0;
for (int i = 0; i < a.length; i++) dot += a[i] * b[i];   // both vectors have length 1
```

**TEACHING EXAMPLE (2 numbers instead of 384):** a = (1, 0), b = (0.8, 0.6), c = (0, 1).
- cos(a, b) = 1×0.8 + 0×0.6 = 0.8. These point nearly the same way, so they are similar.
- cos(a, c) = 1×0 + 0×1 = 0. These are unrelated.
- Real embeddings work the same way, just with 384 numbers.

**How this project does it:**
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` [config.py: embedding_model_name]. The project documentation says 384 dimensions [PROJECT_DOCUMENTATION.md §2].
- **Normalized:** vectors are created with `normalize_embeddings=True` [embeddings.py: _embed_batch()]. Each vector has length 1.
- **Chroma setting:** the collection uses `metadata={"hnsw:space": "cosine"}` [vectorstore.py: _ensure_collection()]. Chroma returns a distance, which is 1 minus the cosine similarity. The code converts it back with `similarity = max(0.0, 1.0 - distance)` [vectorstore.py: search()].
- **Batching:** batches of 32 [embeddings.py: embed_texts()].
- **Cache:** an in-memory dictionary keyed by the SHA-256 hash of the text. It has no size limit and is lost on restart.
- **Retry:** each batch is retried up to 3 times with exponential wait (tenacity) [embeddings.py: _embed_batch()].
- **Lazy loading:** the model loads on first use. The upload page warns "can take ~30–60s while the embedding model warms up" [UploadPage.tsx].
- **Same model for both sides:** documents and queries go through the same `embedding_service`.

**The hash fallback.** If the model fails to import or load, `_hash_embedding()` is used. It takes the 32 bytes of a SHA-256 hash, repeats them across 384 slots, scales to 0–1, and normalizes [embeddings.py: _hash_embedding()].
- **FACT (ran it):** each fallback vector has only 32 distinct values. The cosine between "hello" and "completely different" was 0.71, which is high. So the fallback is deterministic but not semantic.
- **FACT:** failure is logged as a warning only [embeddings.py: _ensure_model()]. The app keeps working and the retrieval becomes close to arbitrary.
- **INFERENCE:** if some documents were indexed under the fallback and others under the real model, queries on the first group would return nonsense. Part 7 covers this.

**How to say it in an interview**
> "An embedding is a vector of 384 numbers that represents the meaning of a text, produced by the MiniLM model. Similar texts get vectors that point in similar directions, and I measure that with cosine similarity. I normalize the vectors, so cosine is just a dot product, and Chroma is configured with cosine distance. I also have a hash-based fallback so the app doesn't crash if the model can't load, but I know it isn't semantic, so it's only a safety net for running the pipeline."

## 3.3 Vector database and nearest-neighbour search (ChromaDB); why SQL LIKE is not enough

> **In simple words**
> A vector database stores the number-lists and answers "give me the k stored items closest to this one". It searches by closeness of meaning, not by matching words.
> **Analogy:** a library where books sit on a giant map by topic. Your question lands on the map and you grab the nearest books.
> **Java comparison:** `List<Chunk> findNearest(double[] query, int k)`. Sorting every vector on every query is too slow at scale, so the database builds an index.

**Nearest-neighbour search** finds the k vectors with the smallest distance to the query. The exact way compares against everything. The approximate way (ANN) uses a prebuilt index and is much faster, with slight inaccuracy.
- **FACT:** the Chroma setting is named `hnsw:space`, which points to an HNSW index.
- **INFERENCE (general knowledge, please verify):** HNSW is a graph-based approximate index.

**How this project uses Chroma** [backend/app/services/vectorstore.py]:
- **One shared collection** `academic_chunks` for all documents [config.py: chroma_collection].
- **Each record holds:**
  - an id: `chunk_id`
  - an embedding
  - the chunk text as `document`
  - metadata: `doc_id`, `doc_title`, `page`, `snippet`, `section_type`, `tree_node_id`
- **Search:** `collection.query(query_embeddings=[q], n_results=top_k, where={"doc_id": doc_id})`. The `where` filter restricts results to one document. This is the "per-document metadata filter".
- **Delete:** `collection.delete(where={"doc_id": doc_id})`.
- **Two modes:** `PersistentClient(path=...)` runs Chroma inside the Python process. `HttpClient(host, port)` is used when `use_external_chroma` is true, as in Docker Compose. Connection setup retries 3 times with a 1-second sleep.
- **FACT (ran it):** asking for more results than exist is fine. A 1-chunk document with top_k = 3 (pool 6) returned 1 result.

**Why SQL `LIKE` is not enough.**

**TEACHING EXAMPLE.** The question is "How does the model avoid overfitting?" The paper says "Dropout and weight decay stop the network from memorizing the training data."
- `WHERE text LIKE '%overfitting%'` finds nothing, because the exact word is missing.
- A vector search can still match, since the meaning is close. This is the typical behaviour of embeddings (INFERENCE; I have not run MiniLM here).

| | SQL `LIKE` / keyword | Vector search |
|---|---|---|
| Matches | exact substrings | similar meaning |
| Synonyms, paraphrase | no | yes, usually |
| Ranking | none (match or not) | by closeness |
| Good for | exact names, IDs, rare terms | natural-language questions |

**Name trap (FACT).** The code has no keyword or BM25 search. "Hybrid" in this repo means vector search plus section-tree boost [hybrid_retriever.py]. It does not mean vector plus keyword. Many interviewers assume the second meaning, so say it explicitly.

**How to say it in an interview**
> "I store each chunk's embedding in ChromaDB with metadata like document id and page. At question time I embed the question and ask Chroma for the nearest chunks, filtered to that one document. A plain SQL LIKE only matches exact text, so a question phrased differently from the paper would find nothing, while vector search matches by meaning. I chose Chroma because it's easy to run embedded or as a server and supports metadata filters, though I haven't benchmarked it against alternatives like pgvector or FAISS."

## 3.4 Chunking and overlap

> **In simple words**
> Cut the paper into index-card-sized pieces so you can find and hand over only the useful cards. Overlap means each card repeats the last few lines of the previous one. A sentence cut at the edge then still appears whole on a neighbouring card.
> **Java comparison:** a sliding window over a string where the step is smaller than the window: `for (i = 0; i < n; i += size - overlap)`.

**Why chunk at all:**
- One vector for a whole paper blurs the meaning.
- The LLM can only read so much, and tokens cost money.
- Small pieces give precise citations.

**Trade-offs:**

| | Smaller chunks | Larger chunks |
|---|---|---|
| Search precision | sharper | blurrier |
| Context per chunk | may lose surrounding meaning | keeps more context |
| Prompt size at the same top_k | smaller | larger |

**How this project does it** [backend/app/services/chunker.py]:
- **Splitter and settings:** `RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=800, separators=["\n\n", "\n", ". ", " ", ""])`. It tries paragraph breaks first, then line breaks, then sentences, then spaces, and finally single characters.
- **Units:** the sizes are **characters**, not tokens. The docstring says they "approximate ~800-token windows with ~200-token overlap".
- **Overlap ratio:** 800 ÷ 3000 = 0.267, about 27%. The step between chunk starts is 3000 − 800 = 2,200 characters.
- **FACT (ran it):** a 6,000-character text gave 3 chunks of 2,999, 2,999 and 1,599 characters. Check: starts at 0, 2,200 and 4,400, and the last chunk is 6,000 − 4,400 = 1,600 characters.
- **One page at a time:** `chunk_pages()` splits each page separately. Chunks never cross a page boundary. A page of 3,000 characters or fewer becomes exactly 1 chunk (FACT, ran it: 2,900 characters gave 1 chunk). Overlap helps only inside long pages.
  - Upside: the `page` metadata is exact.
  - Downside: a sentence that runs across a page break is cut, and overlap does not repair it.
- **Chunk id and snippet:** the id is `{doc_id}-chunk-{n}`, numbered across the whole document. The snippet is the first 220 characters with newlines removed.
- **Section labels:** each chunk also gets the `section_type` and `tree_node_id` of the tree node covering its page.

**Tables and equations.**
- **FACT:** `loader.py` only calls `page.extract_text()`. No code handles tables, equations or figures. OCR for images: NOT IN THE MATERIAL.
- **FACT (ran it):** a 120-row table of 4,579 characters became 2 chunks, split at a row boundary with 800 characters of overlap. Only the first chunk contains the header row, so numbers in the second lose their column names.
- **INFERENCE (general knowledge):** PDF text extraction usually flattens tables into lines of cells and turns equations into broken symbols. Both are poor input for embeddings.

**Why 3000 and 800?**
- **NOT IN THE MATERIAL:** any experiment or measurement behind these numbers.
- **INFERENCE:** they were chosen by hand.
- **INFERENCE, verify on the model card:** MiniLM-class embedding models often cut off long inputs at around 256 word-pieces. That limit is NOT IN THE MATERIAL. If it applies, only the first part of a 3,000-character chunk (about 800 tokens by the repo's own estimate) would shape its embedding, while the LLM still sees the whole chunk.

**How to say it in an interview**
> "I split each page with LangChain's recursive splitter at 3000 characters with 800 overlap, so splits prefer paragraph and sentence boundaries. Chunking per page keeps the page number exact for citations. I picked those sizes by judgement and did not tune them with an experiment, so I'd treat them as a starting point. The weak spots are tables and equations, because I only extract plain text and don't handle those structures specially."

## 3.5 The whole RAG pipeline, end to end

> **In simple words**
> RAG has two phases. Phase 1 (once per document): read it, cut it, convert the pieces to numbers, store them. Phase 2 (per question): convert the question to numbers, fetch the nearest pieces, hand them to the LLM, and show the answer with the sources.
> **Java comparison:** Phase 1 is like `indexer.index(document)`. Phase 2 is `llm.answer(question, searcher.top(question))`.

```
PHASE 1: INGEST (background job)            PHASE 2: ASK (each question)
PDF file                                    question text
  | loader.load_pdf()   pages                 | embedding_service.embed_query()
  | tree_index.build_document_tree()          | vectorstore.search(doc_id, pool=2*top_k)
  | chunker.chunk_pages()  chunks             | hybrid_retriever: tree boost + rerank
  | embedding_service.embed_texts()           | rag._format_chunks() + prompt template
  | vectorstore.add_chunks()                  | rag._generate_answer()  -> LLM provider
  | tree_store.save_tree()                    | AskResponse: answer + sources + metrics
  | jobs.add_document()
```

| Step | File and function |
|---|---|
| Ingest | [upload.py: process_document_job()] |
| Retrieve | [rag.py: answer_question()] calls [hybrid_retriever.py: hybrid_retrieve()] |
| Build prompt | [rag.py: _format_chunks(), USER_PROMPT_TEMPLATE] |
| Generate | [rag.py: _generate_answer()] |
| Cite | [rag.py: answer_question()] builds `SourceChunk` list from the retrieved chunks |

**How "cite" works here (FACT):**
- The LLM is asked to write `[doc_title,page]` citations inside its text.
- Separately, the API returns `sources`: the retrieved chunks with `page`, `snippet`, `chunk_id`, `score`.
- The UI shows only the first 3 sources, even though the label shows the full count [MessageBubble.tsx: `sources.slice(0,3)`].
- These are the retrieved chunks, not a verified list of what the answer actually used.

Part 4 traces each step with code.

**How to say it in an interview**
> "Ingestion happens once per document: parse the PDF, build a section tree, chunk, embed, and store in Chroma. At question time I embed the question, retrieve the nearest chunks for that document, apply my section-tree boost, put the chunks into a prompt, and call the LLM. The response returns the answer plus the source chunks with page numbers. One thing I'd say honestly is that the sources are what I retrieved, not a verified mapping of each sentence to its evidence."

## 3.6 Reciprocal rank fusion and re-ranking (with worked examples)

> **In simple words**
> Two judges rank the same contestants. Their raw scores are on different scales, so you ignore the scores and give points by position, more for higher places. Then you add up each contestant's points.
> **Java comparison:** `score.merge(id, 1.0 / (k + rank), Double::sum)` for every item in every list.

**Formula.** `score(item) = sum over lists of 1 / (k + rank_in_that_list)`. Here k = 60.

**TEACHING EXAMPLE (invented items A, B, C; k = 60).** List 1 (vector) is A, B, C. List 2 (keyword) is B, C, A.

| Item | From list 1 | From list 2 | Total |
|---|---|---|---|
| A | 1/(60+1) = 0.016393 | 1/(60+3) = 0.015873 | 0.032266 |
| B | 1/(60+2) = 0.016129 | 1/(60+1) = 0.016393 | **0.032522** |
| C | 1/(60+3) = 0.015873 | 1/(60+2) = 0.016129 | 0.032002 |

Final order is B, A, C. B wins because it is high in both lists.

**What k does.** With k = 60, ranks 1 and 2 score 1/61 = 0.016393 and 1/62 = 0.016129, a ratio of 1.016. The top rank barely beats the second. With k = 1 they would be 1/2 and 1/3, a ratio of 1.5, so the top spot would dominate.

**Why k = 60 here.** The code says only "Standard RRF rank-smoothing constant (Cormack et al., 2009)" [hybrid_retriever.py: RRF_K]. Any tuning experiment: NOT IN THE MATERIAL.

**Re-ranking** means taking a candidate list and reordering it with a second, often better, rule.

**What THIS project does** [hybrid_retriever.py: hybrid_retrieve(), _tree_boosted_rrf()]. The honest summary is **ATB-RRF is a rank-based score with a multiplier, not a fusion of two ranked lists.**

1. `classify_query()` labels the query factual, structural or conceptual. **FACT:** the label is only reported in the response. It does not change ranking.
2. `navigate_tree()` scores every tree node: +2.0 per query word in the title, +0.5 per query word in the summary, then +1.5 if the node's section type matches a keyword-triggered group, or ×0.7 if it is in that group's "dampen" list. It keeps the top 3.
3. The pages of those nodes form the tree page set.
4. The vector search fetches a pool of `2 × top_k` chunks.
5. **Confidence** = the fraction of the vector top-k whose page is in the tree page set.
6. **Boost** (`_adaptive_boost_factor()`):
   - `raw = 1.3 + min(affinity/5, 1) × (2.0 − 1.3) + 0.15 × depth`, clamped to [1.0, 2.0]
   - if confidence < 0.3, pull the boost toward 1: `1 + (boost − 1) × confidence/0.3`
7. **Fuse:** `score = 1/(60 + vector_rank)`, multiplied by the boost if the chunk's page is in the tree page set. Take the best `top_k`. There is no second ranked list.
8. **Coherence re-rank** (`_coherence_rerank()`): add 0.1 to chunks whose tree node is one of the top 2 navigated nodes, then sort.

**Worked example, step by step (FACT (ran it); synthetic 4-section tree, query "How does the proposed method work?", top_k = 3).** The tree was Abstract p1, Methodology p4–6, Introduction p7–9, Results p10–12.
- **Navigate:** only Methodology scored. The score is 1.5, because "how" and "method" trigger the methodology group and its section type matches. There is no title overlap since "method" is not the whole word "methodology". Tree pages = {4, 5, 6}.
- **Confidence:** the vector top-3 are c1 (page 9), c2 (page 4), c3 (page 4). Two of three are on tree pages, so 2/3 = 0.667.
- **Boost:** affinity 1.5 gives norm 1.5/5 = 0.3. Raw = 1.3 + 0.3 × 0.7 + 0.15 × 0 = 1.51. Confidence 0.667 ≥ 0.3, so no attenuation. Boost = **1.51**.

| Chunk | Vector rank | Page | On tree page? | Base 1/(60+r) | Boost | Fused |
|---|---|---|---|---|---|---|
| c1 | 1 | 9 | no | 1/61 = 0.016393 | none | 0.016393 |
| c2 | 2 | 4 | yes | 1/62 = 0.016129 | ×1.51 | **0.024355** |
| c3 | 3 | 4 | yes | 1/63 = 0.015873 | ×1.51 | **0.023968** |
| c4 | 4 | 10 | no | 1/64 = 0.015625 | none | 0.015625 |
| c5 | 5 | 5 | yes | 1/65 = 0.015385 | ×1.51 | **0.023231** |
| c6 | 6 | 1 | no | 1/66 = 0.015152 | none | 0.015152 |

Top 3 are c2, c3, c5. The rank-1 vector hit (c1) was dropped, and a rank-5 chunk was promoted into the answer. That is the whole point of the boost. The coherence step changed nothing here, because all three chunks belong to the one primary node.

**Two facts to know before an interviewer finds them:**
- **Scale problem (FACT, ran it).** Fused scores are at most 2.0/61 = 0.0328. The coherence bonus is 0.1, which is 0.1 ÷ 0.0328 = 3.05 times the largest possible score. A chunk with fused 0.015 inside a primary node outranked a chunk with 0.032 outside it. So coherence re-rank effectively means "chunks from the top 2 sections first".
- **Depth never counts.** `_build_nodes_from_headings()` creates only a flat list, and no code fills `children` (FACT, grep), so depth is always 0 and the depth bonus never applies. Part 7 covers this.

**How to say it in an interview**
> "Reciprocal rank fusion combines ranked lists by giving each item 1 over k plus its rank from each list and adding them up, with k set to 60. In my project there's really one ranked list, the vector results, so I use the same rank-based score and multiply it by a boost between 1 and 2 for chunks that sit on pages of the sections the question seems to target. So I'd describe it as a structure-aware re-ranking, not a full two-list fusion. I took k equals 60 as the usual default and didn't tune it, and I haven't measured whether the boost improves retrieval."

## 3.7 FastAPI: routes, Pydantic, background tasks, status codes, 202 plus polling

> **In simple words**
> FastAPI turns Python functions into web endpoints. It checks incoming data against type rules automatically. It can also run slow work after it has already replied.
> **Java comparison:** `@RestController` and `@PostMapping`. Pydantic models are DTOs with Bean Validation. `BackgroundTasks` is like handing a `Runnable` to an `ExecutorService` after the HTTP response is sent.

**Route and model.** Example: `@router.post("/ask", response_model=AskResponse)` on `def ask_question(request: AskRequest)` [qa.py]. `AskRequest` declares its limits: `question` is 3–4,000 characters, `top_k` is 1–10, `temperature` is 0–1, `mode` is one of "explain", "summary", "qa" [schemas.py]. A bad request gets an automatic **422** (FACT, ran it).

**The upload endpoint, explained** [upload.py: upload_pdf()]:

```python
@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                     title: str | None = Form(default=None)) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if file.content_type not in {"application/pdf", "application/octet-stream"} and suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    job_id = str(uuid4())
    output_path = Path(settings.uploads_dir) / f"{job_id}.pdf"
    with output_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    job_repository.create_job(job_id)
    background_tasks.add_task(process_document_job, job_id, str(output_path), title)
    return UploadResponse(job_id=job_id, status="processing")
```

- **Decorator:** a POST route that answers **202**, not 200.
- **Signature:** FastAPI injects a `BackgroundTasks` helper. The file arrives as multipart form data, and `title` is an optional form field.
- **Validation:** reject a file only if **both** the content type and the extension look wrong. **FACT (ran it):** a file named `virus.exe` with content type `application/pdf` was accepted (202), and so was `data.bin` as `application/octet-stream`. The check is weak.
- **Save:** make a job id (a UUID), save the file as `<job_id>.pdf`, streaming the copy.
- **Job row:** insert a "processing" row into the `jobs` table.
- **Schedule:** `add_task` runs the heavy work after the response goes out.
- **Reply:** return at once with `{job_id, status: "processing"}`.

**Status codes used (FACT):**

| Code | When |
|---|---|
| 200 | normal success |
| 202 | upload accepted, work continues in the background |
| 400 | wrong file type; `custom_query` missing for `section=custom` |
| 404 | unknown job, document, or tree |
| 422 | request body breaks the Pydantic rules (automatic) |
| 429 | per-IP rate limit hit [main.py: RateLimitMiddleware] |

**FACT (ran it):** when the LLM call fails (including the 5-per-minute LLM limiter), the API still returns **200** with an apology text inside `answer`. The error is hidden in the body.

**Why 202 plus polling.**
- **202 Accepted** means "I got it and will process it; it's not finished". 200 would imply the document is ready to query.
- Ingestion has several stages (parse, tree, chunk, embed, store). The UI says the first upload can take 30–60 s. The app's Axios timeout for normal calls is 30 s [api.ts], and a held-open request is fragile.
- So: reply fast with a `job_id`, then the client polls `GET /status/{job_id}`. The UI polls every 1,000 ms until `completed` or `failed` [UploadPage.tsx]. Progress jumps through 25, 50, 60, 80 and 100 [upload.py: process_document_job()].
- **Alternatives (INFERENCE):** WebSockets or server-sent events for push updates, or a real job queue (Celery/RQ). The README lists Celery/RQ as a scaling idea [README.md: Scaling Notes].

**Limits of this design (FACT from the code):**
- `BackgroundTasks` runs inside the web process. A restart kills a running job. The job row stays `processing` forever, because no startup code repairs old jobs.
- Job status lives in SQLite, so it survives restarts as data, but it is never corrected.

**How to say it in an interview**
> "The upload endpoint saves the file, creates a job record, schedules ingestion as a FastAPI background task, and immediately returns 202 with a job id. I used 202 because the request is accepted but not finished, and parsing and embedding can take a while. The frontend polls a status endpoint every second and shows progress. The limitation is that background tasks run inside the web process, so a restart mid-job leaves it stuck. For production I'd move that to a proper queue like Celery or RQ."

## 3.8 Provider abstraction (swappable LLM backends) and the mock provider

> **In simple words**
> The app talks to "an LLM" through one function. A setting decides which real service stands behind it: OpenAI, Gemini, Hugging Face, or a fake one.
> **Analogy:** a universal power adapter. The plug in the wall stays the same.
> **Java comparison:** the Strategy pattern, `interface LlmClient` with four implementations. In this project it is a simpler `if`-chain over functions, not a real interface.

**How it works** [rag.py: _generate_answer()]:

```python
if provider == "openai":  return _call_openai(system_prompt, user_prompt, temperature)
if provider == "gemini":  return _call_gemini(system_prompt, user_prompt, temperature)
if provider == "hf":      return _call_hf(system_prompt, user_prompt, temperature)
return _mock_generate(user_prompt, chunks)
```

- The choice comes from `llm_provider: Literal["openai","hf","gemini","mock"] = "mock"` [config.py]. The `Literal` type means any other value fails at startup.
- **Adding a provider** means writing one function plus one `if`.
- **OpenAI** [_call_openai()]: retries on rate-limit, timeout, connection error and 5xx, with sleeps of 1.0–1.5 × attempt seconds.
- **Gemini** [_call_gemini()]: calls the REST API with `httpx`. It tries the configured model, then fallback models, then built-in "latest" aliases. It retries 429/500/503 and moves to the next model on other HTTP errors such as 404. This came from commit `4d0d4fb`, whose message says to make 404s try the next model.
- **Hugging Face** [_call_hf()]: builds a `pipeline(...)` on every call, which is slow (FACT, read).
- **Global LLM rate limit:** 5 calls per minute for all users together (`llm_max_requests_per_minute = 5`) [config.py]. **FACT (ran it):** it applies even to the mock provider. The 6th call within a minute returns 200 with a failure message.

**The mock provider** [rag.py: _mock_generate()]:
- With no chunks, it returns "I don't know — the document doesn't contain enough information."
- Otherwise it returns a **fixed template** (three generic bullets plus a technical-summary sentence) citing the first 3 chunks. **FACT (ran it):** it ignores the question's content.

**Why a mock helps:**
- No API key, cost or network in CI and local runs.
- Fast and deterministic.
- The rest of the pipeline (upload, chunking, embedding, retrieval, response shape) runs for real.
- The default `LLM_PROVIDER` is `mock`, so CI needs no secret. CI has no `.env` file.

**Honest limits:**
- It can't test answer quality, since it never reads the question.
- **FACT:** the tests replace `rag._generate_answer` using `monkeypatch` [test_ask.py], so the mock branch itself isn't exercised by any test.
- **FACT:** `.env.example` and the README set `LLM_PROVIDER=openai`. Copying that file turns the default off, which is easy to forget.
- With the mock provider, tree building skips the LLM and uses heading heuristics [upload.py: `use_llm=(provider != "mock")`].

**How to say it in an interview**
> "I wrote the LLM call behind one function that dispatches on a setting, so I can switch between OpenAI, Gemini, Hugging Face or a mock without touching the retrieval code. The mock returns canned text, which lets CI and local development run the whole pipeline with no API key or cost. It doesn't test answer quality, and my tests actually patch the generate function anyway. If I did it again I'd define a proper interface class for providers so each one is easier to test in isolation."

## 3.9 RAG evaluation: recall@k, MRR, groundedness (not in this repo)

> **In simple words**
> Evaluation answers "is it actually good?" with numbers. For RAG you check two things: did search find the right text, and did the answer stick to that text?
> **Java comparison:** unit tests for search quality, with a fixed set of questions and expected results.

**FACT:** the repo has no evaluation code, no test questions, and no result numbers. I searched for recall, MRR, groundedness, benchmark and evaluation. The only matches were keyword lists in the section classifier. The `retrieval_confidence` field is **not** an evaluation metric. It measures how many of the vector top-k lie on pages of the tree-chosen sections [hybrid_retriever.py: _compute_retrieval_confidence()].

**Metrics (TEACHING EXAMPLE, invented results for 3 questions):** the correct chunk was found at rank 1 for Q1, rank 3 for Q2, and not in the top 5 for Q3.
- **Recall@k** = share of questions whose correct chunk appears in the top k. Recall@5 = 2 ÷ 3 = 0.667.
- **MRR** (mean reciprocal rank) = average of 1/rank of the first correct result, counting 0 if absent. MRR = (1/1 + 1/3 + 0) ÷ 3 = 1.3333 ÷ 3 = 0.444.
- **Groundedness** = how much of the answer is supported by the retrieved chunks. It is judged by a human or by another LLM ("LLM-as-judge"). A simple version is the fraction of answer claims that the chunks support.

**How you would build it here (INFERENCE):**
1. Write 30–50 questions over a few papers and note which page holds the answer.
2. Run `hybrid_retrieve()` with and without the tree (`tree_nodes=None`) and compare recall@5 and MRR.
3. Score a sample of LLM answers for groundedness.
4. Tune chunk size and the boost constants using these numbers.

**How to say it in an interview**
> "I haven't built a retrieval evaluation yet, and I'd rather say that than invent numbers. The approach I'd take is a small gold set of questions with the correct page for each, then compare recall at 5 and MRR for plain vector search versus my section-boosted ranking. For answer quality I'd check groundedness, meaning whether each claim is supported by the retrieved chunks. That's the first thing I'd add, because right now I can't prove the StructRAG layer helps."

## 3.10 Docker, Docker Compose, environment variables, CORS

> **In simple words**
> Docker packs the app and everything it needs into a box (an image) that runs the same on any machine. Compose starts several boxes together. Environment variables are settings you inject from outside, so no passwords live in the code. CORS is a browser safety rule that decides which websites may call your API.
> **Java comparison:** an image is like a fat JAR plus its JVM. Compose is like a small script that launches several services. Env variables are `System.getenv()`.

**Backend Dockerfile, line by line** [backend/Dockerfile]:
- `FROM python:3.11-slim`: start from a small Python 3.11 image.
- `ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1`: no `.pyc` files; logs appear immediately.
- `apt-get install build-essential curl`: compiler tools for packages that need them.
- `COPY requirements.txt` then `pip install`, and only then `COPY app`: dependency layers are cached when only code changes.
- `CMD ... gunicorn -k uvicorn.workers.UvicornWorker app.main:app -b 0.0.0.0:${PORT:-8000} --workers 1 --timeout 180`: gunicorn manages one uvicorn worker. It uses the platform's `PORT` or 8000, and allows 180 s per request.
- **FACT (history):** commit `c4bb7e0` changed this from 2 workers and a 120 s timeout to 1 worker and 180 s. The reason is NOT IN THE MATERIAL.
- **FACT:** nothing downloads the embedding model during the build. **INFERENCE:** it is fetched at first use.

**docker-compose.yml:**

| Service | What it does |
|---|---|
| `chroma` | image `chromadb/chroma:0.5.5`, host port 8001, data in `./chromadb_data` |
| `backend` | builds `./backend`, reads `.env`, sets `USE_EXTERNAL_CHROMA=true`, `CHROMA_HOST=chroma`, mounts `./backend/data` and `./chromadb_data`, port 8000 |
| `frontend` | builds `./frontend`, runs the Vite dev server, proxies `/api` to `http://backend:8000` |
| `redis` | **declared but unused**; no code mentions it |

- **FACT:** the Chroma server image is 0.5.5, but the Python client is pinned to `chromadb==0.5.23` [requirements.txt]. **INFERENCE:** client and server are normally kept on matching versions, so this is a risk.
- **INFERENCE (general Docker knowledge):** `depends_on` only orders start-up; it does not wait until Chroma is ready. **FACT:** the retry loop in `_ensure_collection()` makes 3 tries, 1 s apart. **INFERENCE:** that is probably why it exists.

**Environment variables.** `Settings` (pydantic-settings) fills fields from env vars or a `.env` file. `get_settings()` looks for `.env` in the current folder, the repo root, then `backend/` [config.py]. **FACT (ran it):** setting `UPLOADS_DIR`, `METADATA_DB_PATH` and `CHROMA_PERSIST_DIR` redirected storage as expected, so upper-case env names map to the lower-case fields. `.gitignore` excludes `.env`. **FACT (scanned):** I found no real secret in the files or git history, only the placeholder `OPENAI_API_KEY=sk-...` and an empty Gemini key in `.env.example`.

**CORS.** Browsers block a page on site A from calling an API on site B unless B sends permission headers.
- **FACT:** `CORSMiddleware` allows the origins listed in `FRONTEND_URL`, comma-separated since commit `3f4109f`. It also sets `allow_credentials=True`, methods GET/POST/DELETE/OPTIONS, and all headers [main.py].
- **Dev:** the Vite proxy forwards `/api` to the backend, so the browser sees one origin and CORS is not needed [vite.config.ts].
- **Production:** the frontend (Vercel) and backend (Railway) are on different origins, so `FRONTEND_URL` on the backend must contain the Vercel URL. Its value in production is NOT IN THE MATERIAL.

**How to say it in an interview**
> "The backend ships as a Docker image that starts gunicorn with a uvicorn worker. Locally, Docker Compose runs the backend, a Chroma server and the frontend together. All settings come from environment variables through a Pydantic Settings class, so no secrets are in the code. CORS is configured from a FRONTEND_URL variable, because in production the Vercel frontend and the backend are on different origins. One thing I'd clean up is the unused Redis service in the compose file."

## 3.11 GitHub Actions CI pipeline

> **In simple words**
> Every time you push code, GitHub starts a clean machine that installs your project, runs the tests and checks the build. If something breaks, you find out before merging.
> **Java comparison:** a Jenkins or Maven build that runs on each push.

**What the pipeline does** [.github/workflows/ci.yml]:
- **Triggers:** push to `main`, and every pull request.
- **Job `backend`:** Python 3.11, `pip install -r backend/requirements.txt`, then `pytest -q` in `backend/`.
- **Job `frontend`:** Node 20, `npm install`, `npm run lint`, `npm run test`, `npm run build`.
- **Job `docker`:** runs only after the first two pass, and builds both Docker images (`docker build`). It does not push them or deploy.

**Why the backend tests need no API key.**
- The default provider is `mock`.
- CI has no `.env`.
- The ask test replaces `_generate_answer` [test_ask.py].

**What it does not do (FACT):**
- No deploy step. Vercel and Railway deployment lives outside this workflow (how is NOT IN THE MATERIAL).
- No dependency caching.
- It uses `npm install`, not `npm ci`.
- There is no test for the StructRAG modules.
- Whether CI is currently green: NOT IN THE MATERIAL, since there is no run history.
- **FACT (ran it):** the 4 backend tests passed on my machine. There the embedding model wasn't installed, so they ran on the hash fallback. I did not run the frontend lint, tests or build.

**Test inventory (FACT):**

| File | Tests |
|---|---|
| `test_ask.py` | mocked embedding batch; upload → status → ask flow with a patched LLM |
| `test_upload.py` | loader on the 1-page sample; chunk overlap and page metadata |
| `ChatBox.test.tsx` | typing and pressing Send calls `onSend` |
| `FileUploader.test.tsx` | choosing a PDF calls the upload API and emits the job id |

**How to say it in an interview**
> "I have a GitHub Actions workflow with three jobs. The backend job installs dependencies and runs pytest, the frontend job runs lint, tests and a production build, and a Docker job builds both images once those pass. Tests don't need an API key because the default LLM provider is a mock. The gap is that the workflow doesn't deploy, and the tests are thin: nothing yet covers the section tree or the ranking code."

## 3.12 React + TypeScript basics used here; deployment on Vercel and Railway

> **In simple words**
> React builds a page from small reusable pieces called components. Each is a function that returns what to draw. TypeScript adds types to JavaScript, so mistakes are caught before running. Vercel hosts the finished website. Railway runs the backend container.
> **Java comparison:** a component is like a class whose only method returns the UI. Props are constructor arguments. State is a field that, when changed, redraws the screen. A TypeScript `interface` is like a Java DTO.

**Basics with project examples:**

| Idea | Meaning | Where used |
|---|---|---|
| Component | a function returning JSX (HTML-like markup) | `export default function UploadPage()` |
| Props | typed inputs, declared in an `interface Props` | `ChatBox` receives `messages`, `topK`, `onSend` |
| `useState` | `const [x, setX] = useState(init)`; calling `setX` re-renders | `status`, `jobId`, `messages` |
| `useEffect` | runs side effects after render, returns a cleanup | polling in `UploadPage` |
| React Router | URL to component mapping; `useParams()` reads `:docId` | `/chat/:docId` in `App.tsx` |
| Axios | HTTP client | all calls in `services/api.ts` |

**Polling code, line by line** [UploadPage.tsx]:

```tsx
useEffect(() => {
  if (!jobId) return;                                   // nothing to poll yet
  const poll = setInterval(async () => {                // run every 1000 ms
    try {
      const payload = await getJobStatus(jobId);        // GET /api/v1/status/{jobId}
      setStatus(payload);                               // new state, screen redraws
      if (payload.status === 'completed' || payload.status === 'failed') clearInterval(poll);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to fetch status');
      clearInterval(poll);                              // any error stops polling
    }
  }, 1000);
  return () => clearInterval(poll);                     // cleanup, like a close() in finally
}, [jobId]);                                            // re-run only when jobId changes
```

- **FACT:** TypeScript types in `api.ts` mirror the backend models, but they are hand-copied. The TS `AskResponse` omits `query_type`, `retrieval_mode`, `retrieval_confidence` and `adaptive_boost_used`.
- **FACT (read):** the chat page's "Retrieval confidence" badge is computed in the browser as the average of `source.score`: above 0.75 is High, above 0.45 is Medium, otherwise Low [ChatPage.tsx]. It ignores the backend's `retrieval_confidence`. In tree mode the scores are about 0.02, so the badge shows Low and each source shows about 2%. Part 7 covers this.

**Vercel (frontend).** `vercel.json` says: framework Vite, build `npm run build`, output `dist`, and rewrite every URL to `/index.html`. That rewrite lets a refresh on `/chat/<id>` load the React app instead of a 404. This is the standard single-page-app (SPA) setup.

**Railway (backend).** `railway.json` says: build with the Dockerfile; on failure restart, at most 3 times. The container reads `PORT` from the platform.

**How the two connect (INFERENCE).**
- The frontend sends calls to `import.meta.env.VITE_API_BASE_URL`, which defaults to an empty string [api.ts]. In production it must be set at build time to the backend URL.
- Without it, `/api/...` would go to Vercel itself and match the catch-all rewrite.
- The real value is NOT IN THE MATERIAL.

**Storage on Railway (INFERENCE).** SQLite, uploads and Chroma data default to folders inside the container [config.py]. Nothing in the material attaches a volume, so a redeploy would probably erase them. This is relevant to your note that the backend returned "Application not found"; Part 7 covers it.

**How to say it in an interview**
> "The frontend is a React app in TypeScript built with Vite. It has pages for upload, document list and chat, and one API module that wraps the backend calls with Axios. For upload I poll the status endpoint every second inside a useEffect and clean up the timer when the job finishes. The frontend deploys to Vercel with a rewrite so deep links work, and the backend deploys from its Dockerfile to Railway. The backend URL is passed to the frontend through a build-time environment variable."

## 3.13 Other concepts the code uses (short table)

| Concept | One-line meaning | Where |
|---|---|---|
| **SQLite + a lock** | File database; a Python `threading.Lock` stops two threads using one connection at once | [jobs.py, tree_store.py] |
| **Retry with backoff** | Try again after waiting longer each time | [embeddings.py: tenacity]; [rag.py: provider loops] |
| **Rate limiting** | Refuse requests above N per minute | per IP, 60/min, in memory [main.py]; for all LLM calls together, 5/min [rag.py] |
| **Section tree and ontology** | A list of headings with page ranges, each tagged with one of 13 types | [tree_index.py: TreeNode, classify_section_type()] |
| **Heading heuristics** | Regex patterns plus length and punctuation checks to spot headings | [tree_index.py: _extract_headings_heuristic()] |
| **Graceful fallback** | If a step fails, use a simpler step | pypdf→unstructured [loader.py]; real embeddings→hash; tree→vector-only |
| **Safety filter** | Refuse questions containing words like "diagnose" or "lawsuit" | [rag.py: DISALLOWED_PATTERNS]; **FACT (ran it):** "How do the authors diagnose model errors?" is blocked |
| **In-memory state** | Data held in process memory, lost on restart, not shared between workers | embedding cache, rate-limit windows |

---

# Part 4. End-to-end walkthroughs

Each walkthrough follows one user action through the real code. Every step names the file and function.

## 4.0 What happens when the server starts

> **In simple words**
> Before any request, Python imports the files. Some files create their objects right then (database connections, settings). Knowing this order explains many "why does it already exist?" questions.

1. `gunicorn ... app.main:app` imports `app/main.py` [Dockerfile CMD].
2. `main.py` calls `configure_logging()` [core/logger.py] and `get_settings()` [core/config.py]. `get_settings()` looks for a `.env` file (current folder, repo root, `backend/`), builds `Settings`, and **creates three folders** if missing: uploads, the SQLite file's folder, and the Chroma folder. It is cached with `@lru_cache`, so it runs once.
3. `main.py` imports the four routers. That imports the services, which creates the singletons:
   - `job_repository = JobRepository()` opens `metadata.db` and runs `CREATE TABLE IF NOT EXISTS` for `jobs` and `documents` [jobs.py].
   - `tree_store = TreeStore()` creates `document_trees` [tree_store.py].
   - `embedding_service = EmbeddingService()` stores settings but does **not** load the model yet [embeddings.py].
   - `vectorstore_service = VectorStoreService()` does **not** connect to Chroma yet [vectorstore.py].
4. `embeddings.py` also sets `TOKENIZERS_PARALLELISM=false` and calls `multiprocessing.set_start_method("spawn", force=True)` at import. This came from commit `aff68da` ("Fix broken pipe errors"). It is a process-wide side effect.
5. `main.py` registers middleware, then routers under `/api/v1`, then `/` and `/health`.

**Middleware order matters (FACT (ran it)).** The code adds `CORSMiddleware` first and `RateLimitMiddleware` second. The last one added is the outermost [main.py]. The runtime order is `['RateLimitMiddleware', 'CORSMiddleware']`. So a request is rate-checked before CORS handles it.

## 4.1 Flow A: uploading a PDF (click to "completed")

```
Browser                         FastAPI                              Background thread
-------                         -------                              -----------------
drop PDF
FileUploader.handleFile()
api.uploadPdf()  --POST /api/v1/upload (multipart)-->
                                RateLimitMiddleware, CORS
                                upload_pdf(): check type, save file,
                                create_job(), add_task(...)
              <--202 {job_id, status:"processing"}--
UploadPage sets jobId                                        process_document_job(job_id,...)
useEffect: setInterval 1000ms                                  1 load_pdf        -> 25%
  GET /api/v1/status/{job_id} ---> get_job_status()            2 build tree      -> 50%
  <-- {status, progress, ...}                                  3 chunk_pages     -> 60%
  (repeat until completed/failed)                              4 embed_texts     -> 80%
                                                               5 Chroma + tree + document row
  completed + doc_id --> button "Open chat"                      -> status completed, 100%
```

**Step-by-step with names:**

| # | Where | What happens |
|---|---|---|
| 1 | `FileUploader.tsx: handleFile()` | Checks the file name ends with `.pdf` (else shows "Please upload a PDF file."). Sets `uploading`. |
| 2 | `api.ts: uploadPdf()` | Builds `FormData` with `file` and, only if non-empty, `title`. Posts to `/api/v1/upload` and reports upload progress. |
| 3 | `main.py: RateLimitMiddleware.dispatch()` | Counts requests per client IP in the last 60 s. `/health` and `/` are exempt. Over 60 returns 429. |
| 4 | `upload.py: upload_pdf()` | Checks type, saves `<uploads_dir>/<job_id>.pdf`, inserts a `processing` job row, queues the background task, returns 202. |
| 5 | `UploadPage.tsx` | `onUploaded(job_id)` stores the id; the polling effect starts (see 3.12). |
| 6 | `upload.py: process_document_job()` | The five-stage job below. |

**The ingestion job, with line-by-line notes** [upload.py: process_document_job()]:

```python
pages = load_pdf(file_path)                                   # (1)
total_pages = len(pages)
job_repository.update_job(job_id, pages_parsed=total_pages, total_pages=total_pages, progress=25)

doc_id = str(uuid4())                                         # (2)
doc_title = title.strip() if title else Path(file_path).stem
tree_nodes = build_document_tree(pages=pages, doc_id=doc_id, doc_title=doc_title,
                                 use_llm=(settings.llm_provider != "mock"))
job_repository.update_job(job_id, progress=50)

chunks = chunk_pages(pages, doc_id=doc_id, doc_title=doc_title, tree_nodes=tree_nodes)   # (3)
job_repository.update_job(job_id, chunks_created=len(chunks), progress=60)

embeddings = embedding_service.embed_texts([chunk.text for chunk in chunks])             # (4)
job_repository.update_job(job_id, progress=80)

vectorstore_service.add_chunks(chunks, embeddings)            # (5)
tree_store.save_tree(doc_id, tree_nodes)
job_repository.add_document(DocumentRecord(id=doc_id, title=doc_title, ...))
job_repository.update_job(job_id, status="completed", progress=100, doc_id=doc_id)
# any exception:  update_job(job_id, status="failed", error=str(exc), progress=100)
```

- **(1) `load_pdf()`** [loader.py]:
  - It first checks the file exists. It tries `pypdf` and keeps only pages with non-empty text, numbered `index + 1`.
  - If pypdf raises an error or returns no text, it tries `unstructured`.
  - If that also returns nothing, it raises `ValueError("No readable text found in PDF")`.
  - **FACT (read):** `total_pages` here is the count of pages that had text, not the PDF's real page count. A PDF with blank pages shows a smaller `page_count`.
- **(2) `doc_id` is made here**, inside the job, not at upload. So the upload response contains only a `job_id`. The `doc_id` appears in the status response when the job completes.
  - **FACT (ran it):** with no `title`, the title becomes `Path(file_path).stem`, which is the `job_id` UUID, not the original file name. The documents list then shows a UUID. A bug (Part 7).
- **(2) `build_document_tree()`** — see Flow G below.
- **(3) `chunk_pages()`** — see 3.4. Each chunk gets `section_type` and `tree_node_id` from the page it sits on.
- **(4) `embed_texts()`** [embeddings.py]:
  - It hashes each text with SHA-256 and reuses cached vectors.
  - It sends uncached texts to the model in batches of 32.
  - The first call loads the model (slow); on failure it switches to the hash fallback for the rest of the process.
- **(5) Order of writes:** vectors first, then the tree, then the `documents` row. The document appears in lists only after the last write, so a question can't hit a half-built document. But there is **no rollback**: if `save_tree` or `add_document` fails, the vectors stay in Chroma (INFERENCE: orphaned vectors).
- **Failure path:** the exception text goes into `jobs.error` and is shown on the upload page [UploadPage.tsx]. That leaks internal messages to the user.

## 4.2 Flow B: checking status

1. `UploadPage.tsx` effect calls `api.ts: getJobStatus(jobId)` every 1,000 ms.
2. `GET /api/v1/status/{job_id}` runs `get_job_status()` [upload.py], which reads the row with `job_repository.get_job()` and returns a `JobStatusResponse`: `status`, `progress`, `pages_parsed`, `total_pages`, `chunks_created`, `doc_id`, `error`.
3. Unknown id gives 404 "Job not found".
4. The page shows a progress bar, pages, chunks and status. At `completed` with a `doc_id`, it shows the "Open chat for this document" button, which navigates to `/chat/<doc_id>`.

**Two cautions (details in Part 7):**
- The effect stops polling on any thrown error. **FACT (ran it, simulated clock):** with perfectly regular 1-second polling after an upload, the 60th poll is refused with 429 by the per-IP limiter. A cross-origin browser then sees a CORS-style network error, because the 429 comes from the outermost middleware and carries no CORS header (**FACT (ran it)**). So a long job can leave the page stuck on an error while the job still finishes.
- After a restart during ingestion, the job row stays `processing` forever.

## 4.3 Flow C: asking a question (click Send to rendered answer)

```
ChatPage.handleSend()
  -> api.askQuestion()  POST /api/v1/ask {doc_id, question, top_k, mode, temperature}
     -> qa.ask_question()           404 if the document row is missing
        -> rag.answer_question()
           1 safety regex check
           2 tree_store.get_tree(doc_id)
           3 hybrid_retriever.hybrid_retrieve()
                classify_query -> embed_query -> navigate_tree -> vectorstore.search(pool)
                -> confidence -> boost -> _tree_boosted_rrf -> _coherence_rerank
           4 _format_chunks + USER_PROMPT_TEMPLATE
           5 _enforce_llm_rate_limit + _generate_answer (LLM provider)
           6 build sources list + RAGResponse
     <- AskResponse JSON
  -> MessageBubble + SourceHighlight
```

| # | Where | What happens |
|---|---|---|
| 1 | `ChatPage.tsx: handleSend()` | Adds the user message to local state. Sends **only** `doc_id`, `question`, `top_k`, `mode`, `temperature`. No earlier messages are sent (FACT). |
| 2 | `api.ts: askQuestion()` | POST with a 120 s timeout (or `VITE_LLM_TIMEOUT_MS`). |
| 3 | `qa.py: ask_question()` | 404 "Document not found" if no `documents` row. Otherwise calls `answer_question()`. |
| 4 | `rag.py: answer_question()` | The orchestration below. |
| 5 | `ChatPage.tsx` | Appends an assistant message (`answer` + `sources`). Computes the "Retrieval confidence" badge from the average source score. |

**`answer_question()` with notes** [rag.py]:

```python
if _is_disallowed_query(question):                     # regex: "medical advice", "diagnose", "lawsuit"...
    return RAGResponse(answer=disclaimer, sources=[], prompt_used="")

tree_nodes = tree_store.get_tree(doc_id)               # JSON from SQLite -> TreeNode list (None if absent)
hybrid_result = hybrid_retrieve(query=question, doc_id=doc_id, top_k=top_k, tree_nodes=tree_nodes)
results = hybrid_result.results

formatted_chunks = _format_chunks(results)             # "1. chunk_id=...; page=...; doc_title=...\n<text>"
user_question = f"[{mode}] {question}"                 # mode is only a text prefix
user_prompt = USER_PROMPT_TEMPLATE.format(retrieved_chunks_formatted=formatted_chunks,
                                          user_question=user_question)
try:
    _enforce_llm_rate_limit()                          # 5 per minute, all users together
    answer = _generate_answer(SYSTEM_PROMPT, user_prompt, temperature, results)
except Exception as exc:                               # any failure becomes normal text
    answer = "The language model request failed ... Provider error: " + str(exc)

sources = [SourceChunk(page=r.page, snippet=r.snippet, chunk_id=r.chunk_id, score=r.score) for r in results]
return RAGResponse(answer=answer, sources=sources, prompt_used=user_prompt,
                   query_type=hybrid_result.query_type,
                   retrieval_mode="hybrid" if hybrid_result.used_tree else "vector", ...)
```

- **Safety filter:** a blocked question returns HTTP 200 with a disclaimer and no sources. **FACT (ran it):** "How do the authors diagnose model errors?" is blocked, because the word "diagnose" matches. That is a false positive on a normal research question.
- **`mode`** changes only the `[qa]` / `[explain]` / `[summary]` tag in front of the question. The system prompt is the same for all modes (FACT, read).
- **Error handling:** a failed LLM call is caught and returned as a normal 200 answer. The error text shows up in the chat bubble.
- **`prompt_used`** returns the whole prompt to the browser, including all retrieved chunk text. It is useful for debugging, but it makes responses large.

**`hybrid_retrieve()`, tree branch** [hybrid_retriever.py], simplified:

```python
query_type = classify_query(query)                         # reported only
query_embedding = embedding_service.embed_query(query)
if tree_nodes:
    navigated = navigate_tree(query, tree_nodes)           # top-3 sections + scores
    tree_page_set = _relevant_pages(navigated)             # union of their pages
    vector_candidates = vectorstore_service.search(doc_id, query_embedding, top_k * 2)
    if tree_page_set and navigated:
        confidence = _compute_retrieval_confidence(vector_candidates, tree_page_set, top_k)
        affinity = _mean_affinity_score(navigated)
        fused = _tree_boosted_rrf(vector_candidates, tree_page_set, top_k, affinity, confidence, navigated, tree_nodes)
        final = _coherence_rerank(fused, navigated)
    else:
        final = vector_candidates[:top_k]; confidence = 0.0     # nothing matched: plain vector order
    return HybridRetrievalResult(results=final, used_tree=True, ...)
# no tree: plain vector search, used_tree=False
```

- **FACT (ran it):** `used_tree=True` whenever a tree exists, even if navigation found nothing and the code fell back to vector order. Then `retrieval_mode` says `"hybrid"` although nothing was boosted.
- **FACT (ran it):** the `score` field means different things per branch. In the boosted branch it is the fused rank score (about 0.015–0.033). In the vector-only branch it is cosine similarity (0 to 1).
- **A tree almost always exists.** For a new upload, `build_document_tree()` always returns at least one node, so the "no tree" path only occurs for documents indexed before StructRAG was added (see the message in `tree.py`).

**The vector search** [vectorstore.py: search()]:

```python
result = collection.query(query_embeddings=[query_embedding], n_results=top_k, where={"doc_id": doc_id})
...
distance = float(distances[idx]) if idx < len(distances) else 1.0
similarity = max(0.0, 1.0 - distance)
```

- Chroma returns ids, distances, documents and metadatas as lists of lists (one list per query). The code takes `[0]` for the single query.
- Each hit becomes a `SearchResult` carrying the chunk text and metadata, including `section_type` and `tree_node_id`.

**What the browser shows** [MessageBubble.tsx, SourceHighlight.tsx, ChatPage.tsx]:
- The answer text, then "Sources · N" with the **first 3** sources only.
- Each source shows `Page n`, a bar and `round(score × 100)%`.
- **FACT:** in tree mode the largest possible fused score is 2.0/61 = 0.0328, so the percentage is at most 3%. The badge thresholds (above 0.75 High, above 0.45 Medium) can never be reached, so it shows "Low" (Part 7).

## 4.4 Flow D: "explain section" and "summarize" (backend only)

- `POST /api/v1/explain-section` [qa.py: explain_section()]:
  - It maps `introduction`, `methodology`, `results`, `conclusion` to fixed English questions in `SECTION_QUERIES`. For `custom` it requires `custom_query`, else 400.
  - It then calls `answer_question(..., mode="explain")`.
  - It does **not** filter by section. It only asks a differently worded question, so the normal tree boost applies.
- `POST /api/v1/summarize` [qa.py: summarize()]:
  - It calls `summarize_document()` [rag.py]. That builds one question, "Provide a {length} document summary ... Focus on central thesis, methods, results, and limitations.", and calls `answer_question()` with `top_k = default_top_k` (5) and `mode="summary"`.
  - `key_concepts` are the first 3 lines of the answer that start with `- ` or `* `.
- **FACT (ran it):** `classify_query` labels that summary prompt "structural", because it contains the word "summary".
- **Honest limit:** a "document summary" is built from only the top 5 retrieved chunks, not the whole paper.
- **FACT:** the website never calls these two endpoints.

## 4.5 Flow E: deleting a document

1. `DocList.tsx` Delete button calls `DocsPage.tsx`, which calls `api.ts: deleteDocument()` (`DELETE /api/v1/docs/{id}`), then reloads the list.
2. `docs.py: delete_document()` returns 404 if the id is unknown. Otherwise it runs, in order:
   - `vectorstore_service.delete_doc(doc_id)`: removes the Chroma vectors
   - `tree_store.delete_tree(doc_id)`
   - `job_repository.delete_document(doc_id)`
   - deletes `<uploads_dir>/<job_id>.pdf`
3. **FACT (ran it):** after delete, `/ask` and `/tree/{id}` return 404 and the upload file is gone, but the old `jobs` row remains (status still `completed`).
4. The four steps are not atomic. A failure part-way leaves a partial state.

## 4.6 Flow F: the tree endpoints (backend only)

- `GET /api/v1/tree/{doc_id}` [tree.py: get_document_tree()]: 404 if the document is unknown, or if no tree is stored. Otherwise it returns `node_count` and the nested nodes via `_node_to_schema()`.
- `POST /api/v1/tree/{doc_id}/navigate` [tree.py: navigate_document_tree()]: body `{"query": ...}` (3–2,000 characters). It runs `classify_query()` and `navigate_tree()` only, with no vector search and no LLM. It returns the top 3 sections and their scores, rounded to 3 decimals.
- **FACT:** the UI does not use either endpoint. They are for debugging and demos with curl or the API docs page.

## 4.7 Flow G: how the section tree is built (the StructRAG ingestion step)

> **In simple words**
> The code scans the text for lines that look like headings, then draws page ranges between them: "Introduction runs from page 1 to 3, Methods from 4 to 8". It then labels each with one of 13 section types by looking for keywords in the heading.

`build_document_tree(pages, doc_id, doc_title, use_llm)` [tree_index.py]:

1. **Try an LLM table of contents** (only if `use_llm`, which means the provider is not `mock`). `_llm_extract_toc()` sends the first 20 pages, 600 characters each, and asks for a JSON list of `{title, page}`. Only OpenAI and Gemini are handled. Any failure returns `None` with a warning, and the code moves on.
   - **FACT (read):** the Gemini path uses only `settings.gemini_model`, with no fallback models. A stale model name silently disables this step.
   - **FACT (read):** for papers over 20 pages, sections after page 20 are never shown to the LLM. LLM results are not sorted.
2. **Otherwise use heuristics** (`_extract_headings_heuristic()`). For every line it requires:
   - 3 to 80 characters
   - at most 3 commas and 1 semicolon
   - at most 10 words
   - a match to one of three patterns: a numbered heading ("1. Introduction", "2.3 Related Work", roman numerals), an ALL-CAPS line, or a short Title-Case line of up to 7 words
   - not a repeat of an earlier heading (compared lowercase, stripped of digits, dots and dashes)
3. **Build nodes** (`_build_nodes_from_headings()`): node *i* starts on its heading's page and ends on the page before the next heading (at least its own start page). With **no headings**, it returns one node titled "Document" covering pages 1 to the last page, type `other`.
4. **Classify each title** (`classify_section_type()`): lowercase the title and return the first ontology type (in dictionary order) whose keywords appear as a substring. No match gives `other`.
5. **Summaries** (`_generate_text_summaries()`): the first 280 characters of the text on the node's first two pages. Optionally replaced by short OpenAI summaries (`_enhance_summaries_with_llm()`); for Gemini this function only sleeps.

**FACT (ran it): what the heuristic did on synthetic text** (page 1: paper title, author name, Abstract, "1 Introduction"; page 2: "2 Related Work", "3 Methodology"; page 3: "3.1 Model Architecture", "4 Results", "Table 1"; page 4: "5 Conclusion", "REFERENCES"):

| Detected "heading" | Pages | Type | Correct? |
|---|---|---|---|
| Attention Is All You Need | 1–1 | other | paper title, not a section |
| John Smith | 1–1 | other | author name, not a section |
| Abstract | 1–1 | abstract | yes |
| 1 Introduction | 1–1 | introduction | yes |
| 2 Related Work | 2–2 | related_work | yes |
| 3 Methodology | 2–2 | methodology | yes |
| 3.1 Model Architecture | 3–3 | methodology | sub-section, but stored as a top-level node |
| 4 Results | 3–3 | results | yes |
| 5 Conclusion | 4–4 | conclusion | yes |
| REFERENCES | 4–4 | references | yes |

(`Table 1` was correctly rejected because it matches none of the three patterns: a lone digit is not a capitalised word. Real papers will behave differently; this was a short synthetic example.)

**What this means, plainly (FACT):**
- The tree is **flat**. `children` is never filled by any code, so the "hierarchical tree", "depth bonus" and the sub-sections in `diagrams/3_section_tree.mmd` are not produced by the code.
- Title and author lines can become fake sections.
- The chunker maps each *page* to a node. When several headings share a page, later nodes overwrite earlier ones in that page-to-node map [chunker.py: chunk_pages()], so a chunk's `section_type` is only page-level accurate.
- **With no headings**, the whole paper is one `other` node. Navigation then only matches if a query word hits the node's summary text or the word "document" (**FACT (ran it)**, on a synthetic one-node tree: "Summarize the document" matched; "What problem does this paper solve?" did not). When nothing matches, ranking is plain vector order and the label still says `hybrid`.
- **With the mock provider**, no LLM is called, so only heuristics run, and the summaries are plain text slices.

## 4.8 How to say it in an interview

> **Upload to answer, in one breath.** "The browser uploads the PDF; the server saves it, creates a job row and returns 202 with a job id. A background task reads the pages, builds a section tree, chunks each page, embeds the chunks and stores vectors in Chroma and the tree in SQLite, updating progress as it goes, while the browser polls the status endpoint. When I ask a question, the server embeds it, searches Chroma for the nearest chunks of that document, boosts those that fall on pages of the sections that seem relevant, formats them into a prompt and calls the LLM. The answer comes back with the retrieved chunks as sources, each with page, snippet and score."

> **If asked what happens when it fails.** "If any ingestion stage throws, the job is marked failed with the error message and the UI shows it. If the LLM call fails, the endpoint still returns 200 with an error message in the answer text, which I know isn't ideal. I'd change that to a proper 502 or 503 and keep the retrieved sources in the response."

---

# Part 5. Data and API reference

> **In simple words**
> This part is the "dictionary" of the project: every table, every stored record, every endpoint, every setting. You do not memorise it. You use it to answer "what exactly is stored?" and "what exactly does this endpoint take and return?" quickly.
> **Java comparison:** SQLite tables are like JPA entities, the Pydantic models are like request/response DTOs, and the endpoint table is like the list of `@RequestMapping` methods.

## 5.1 Where data lives (FACT)

| Data | Store | Default location | Lost on container restart? |
|---|---|---|---|
| Original PDFs | file system | `./data/uploads/<job_id>.pdf` | INFERENCE: yes, unless a volume is attached (none is configured for Railway in the material) |
| Jobs, documents, section trees | SQLite file | `./data/metadata.db` | same |
| Chunk vectors + metadata | ChromaDB | `./chromadb_data` (embedded) or a Chroma server | same (Compose mounts `./chromadb_data` from the host) |
| Embedding cache | Python dictionary in memory | n/a | always |
| Rate-limit counters | Python deques in memory | n/a | always |

Paths come from `Settings` [backend/app/core/config.py]. In Docker Compose the paths are `/app/data/...` and are mounted from the host folders [docker-compose.yml].

## 5.2 SQLite tables (FACT)

All three live in the same file. `JobRepository` creates `jobs` and `documents` [jobs.py: _init_tables()]. `TreeStore` creates `document_trees` [tree_store.py: _init_table()].

**`jobs`** (one row per upload attempt)

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT, PRIMARY KEY | the job id (UUID); also the PDF file name |
| `status` | TEXT NOT NULL | `processing`, `completed` or `failed` |
| `progress` | REAL NOT NULL | 0 to 100; steps 0, 25, 50, 60, 80, 100 |
| `pages_parsed` | INTEGER NOT NULL | count of pages that had text |
| `total_pages` | INTEGER NOT NULL | same value as `pages_parsed` in the current code |
| `chunks_created` | INTEGER NOT NULL | number of chunks |
| `doc_id` | TEXT, nullable | filled when completed |
| `error` | TEXT, nullable | `str(exception)` when failed |
| `created_at`, `updated_at` | TEXT NOT NULL | ISO-8601 UTC strings |

**`documents`** (one row per finished document)

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT, PRIMARY KEY | the `doc_id` (UUID) |
| `title` | TEXT NOT NULL | the given title, or the job id when none was given |
| `upload_date` | TEXT NOT NULL | ISO-8601 string; list is sorted by it, newest first |
| `page_count` | INTEGER NOT NULL | pages with text |
| `file_path` | TEXT NOT NULL | where the PDF is saved |

**`document_trees`** (one row per document)

| Column | Type | Notes |
|---|---|---|
| `doc_id` | TEXT, PRIMARY KEY | same value as `documents.id` |
| `tree_json` | TEXT NOT NULL | the whole tree as JSON (list of nodes with nested `children`) |
| `created_at` | TEXT NOT NULL | ISO-8601 string |

**Relations and indexes (FACT):**
- There are **no foreign keys**. The links are by convention only: `jobs.doc_id` and `document_trees.doc_id` point to `documents.id`, and the `doc_id` in Chroma metadata points to it too.
- The only indexes are the ones SQLite creates for the three primary keys. `ORDER BY upload_date` is not indexed, which is fine for small lists.
- Deleting a document removes its rows in `documents` and `document_trees`, and its Chroma vectors, but **not** its `jobs` row (FACT, ran it).
- **FACT (read):** `JobRepository` and `TreeStore` each open their own connection to the same file, each with its own lock. **INFERENCE:** under heavy parallel writes they could contend for the file lock.

## 5.3 The Chroma record (FACT)

Collection name `academic_chunks`, created with `metadata={"hnsw:space": "cosine"}` [vectorstore.py].

| Part | Content |
|---|---|
| `id` | `{doc_id}-chunk-{n}` (n counts across the whole document) |
| `embedding` | 384 floats (normalized) |
| `document` | the full chunk text (up to 3,000 characters) |
| `metadata.doc_id` | document id; used by the `where` filter |
| `metadata.doc_title` | title text |
| `metadata.page` | integer page number |
| `metadata.snippet` | first 220 characters, newlines removed |
| `metadata.section_type` | one of the 13 types, taken from the page's tree node |
| `metadata.tree_node_id` | the node id, e.g. `<doc_id>-node-3`, or `""` |

## 5.4 In-memory objects used between functions (FACT)

| Object | Defined in | Fields |
|---|---|---|
| `PageText` | loader.py | `page_number`, `text` |
| `ChunkRecord` | chunker.py | `chunk_id`, `doc_id`, `doc_title`, `page`, `text`, `snippet`, `section_type="other"`, `tree_node_id=""` |
| `TreeNode` | tree_index.py | `node_id`, `title`, `start_page`, `end_page`, `summary`, `section_type`, `children` (+ `to_dict`, `from_dict`, `all_nodes`, `covers_page`) |
| `DocumentRecord` | jobs.py | `id`, `title`, `upload_date`, `page_count`, `file_path` |
| `SearchResult` | vectorstore.py | `chunk_id`, `page`, `snippet`, `score`, `text`, `doc_title`, `section_type`, `tree_node_id` |
| `HybridRetrievalResult` | hybrid_retriever.py | `results`, `query_type`, `used_tree`, `relevant_node_titles`, `retrieval_confidence`, `adaptive_boost_used` |
| `RAGResponse` | rag.py | `answer`, `sources`, `prompt_used`, `query_type`, `retrieval_mode`, `retrieval_confidence`, `adaptive_boost_used` |

Java comparison: Python `@dataclass` is like a Java `record` with a mutable option.

## 5.5 API models (Pydantic, in `models/schemas.py`) (FACT)

| Model | Fields (type, rule) |
|---|---|
| `UploadResponse` | `job_id: str`, `status: "processing"` |
| `JobStatusResponse` | `job_id`, `status: processing/completed/failed`, `progress: 0..100`, `pages_parsed`, `total_pages`, `chunks_created`, `doc_id?`, `error?` |
| `DocumentInfo` | `id`, `title`, `upload_date: datetime`, `page_count` |
| `DocumentListResponse` | `documents: [DocumentInfo]` |
| `DeleteDocumentResponse` | `doc_id`, `deleted: bool` |
| `SourceChunk` | `page: int`, `snippet: str`, `chunk_id: str`, `score: float` |
| **`AskRequest`** | `doc_id`, `question` (3–4000 chars), `top_k` (1–10, default 5), `mode` (`explain`/`summary`/`qa`, default `qa`), `temperature` (0–1, default 0.2) |
| **`AskResponse`** | `answer`, `sources: [SourceChunk]`, `prompt_used`, `query_type="conceptual"`, `retrieval_mode="vector"`, `retrieval_confidence` (0–1), `adaptive_boost_used` (≥ 1.0) |
| `ExplainSectionRequest` | `doc_id`, `section` (`introduction`/`methodology`/`results`/`conclusion`/`custom`), `custom_query?`, `top_k`, `temperature` |
| `ExplainSectionResponse` | `section`, `answer`, `sources`, `prompt_used`, + the four StructRAG fields |
| `SummarizeRequest` | `doc_id`, `length` (`short`/`medium`/`long`), `temperature` |
| `SummarizeResponse` | `summary`, `key_concepts: [str]`, `sources`, `prompt_used` |
| `TreeNodeSchema` | `node_id`, `title`, `start_page`, `end_page`, `summary`, `section_type`, `children: [TreeNodeSchema]` (recursive; `model_rebuild()` is needed so Pydantic resolves the self-reference) |
| `TreeDocumentResponse` | `doc_id`, `node_count`, `nodes` |
| `TreeNavigateRequest` | `query` (3–2000 chars) |
| `TreeNavigateResponse` | `query`, `query_type`, `relevant_sections`, `section_affinity_scores` |
| `ErrorResponse` | `detail` (**defined but never used** anywhere) |

There are 17 models, not "15+" as the project documentation says loosely. `sources` in the response is always a list of retrieved chunks.

## 5.6 Endpoint reference

All paths below start with `/api/v1` except the last two rows. **Who may call: anyone.** There is no login, token or user concept anywhere in the code (FACT, read). The only gates are:
1. the per-IP limit of 60 requests per minute (everything except `/` and `/health`);
2. for browsers, the CORS allow-list from `FRONTEND_URL`. CORS does not stop `curl` or other servers;
3. for the LLM-calling endpoints, a global limit of 5 LLM calls per minute for all users together.

| Method | Path | Input | Success output | Errors | Called by the UI? |
|---|---|---|---|---|---|
| POST | `/upload` | multipart: `file` (required), `title` (optional) | **202** `UploadResponse` | 400 wrong type; 422 no file | yes, `FileUploader` |
| GET | `/status/{job_id}` | path id | 200 `JobStatusResponse` | 404 | yes, `UploadPage` (every 1 s) |
| GET | `/docs` | none | 200 `DocumentListResponse` | none | yes, `DocsPage`, `ChatPage` |
| DELETE | `/docs/{doc_id}` | path id | 200 `DeleteDocumentResponse` | 404 | yes, `DocsPage` |
| POST | `/ask` | `AskRequest` | 200 `AskResponse` | 404 unknown doc; 422 bad body | yes, `ChatPage` |
| POST | `/explain-section` | `ExplainSectionRequest` | 200 `ExplainSectionResponse` | 400 `custom` without query; 404; 422 | **no** |
| POST | `/summarize` | `SummarizeRequest` | 200 `SummarizeResponse` | 404; 422 | **no** |
| GET | `/tree/{doc_id}` | path id | 200 `TreeDocumentResponse` | 404 unknown doc or no tree | **no** |
| POST | `/tree/{doc_id}/navigate` | `TreeNavigateRequest` | 200 `TreeNavigateResponse` | 404; 422 | **no** |
| GET | `/` (no prefix) | none | `{"message": "RAG Academic Explainer API"}` | none | no |
| GET | `/health` (no prefix) | none | `{"status": "ok"}` | none | no (exempt from the rate limit) |

Extra notes:
- `GET /api/v1/docs` returns **all** documents to everybody, and `DELETE` lets anybody delete any document (FACT).
- **FACT (ran it):** LLM or rate-limit failures inside `/ask`, `/explain-section` and `/summarize` still return **200** with an error message in the text.
- **INFERENCE (general FastAPI behaviour):** FastAPI also serves its own interactive docs at `/docs` and `/openapi.json`. That is a different path from `/api/v1/docs`.
- **FACT (ran it):** an `OPTIONS` preflight for an allowed origin returns 200 with the CORS header. For an origin not in `FRONTEND_URL` it returns 400.

## 5.7 Settings (environment variables) [core/config.py] (FACT)

| Variable (upper-case) | Default in code | Used by | Note |
|---|---|---|---|
| `APP_NAME` | RAG Academic Explainer | FastAPI title | |
| `ENVIRONMENT` | development | nothing | defined, never read |
| `API_V1_PREFIX` | /api/v1 | router prefix | |
| `BACKEND_HOST`, `BACKEND_PORT` | 0.0.0.0, 8000 | nothing | defined, never read (gunicorn uses `PORT`) |
| `FRONTEND_URL` | http://localhost:5173 | CORS origins, comma-separated | |
| `UPLOADS_DIR` | ./data/uploads | saved PDFs | |
| `METADATA_DB_PATH` | ./data/metadata.db | SQLite | |
| `CHROMA_HOST`, `CHROMA_PORT` | chroma, 8000 | external Chroma | |
| `CHROMA_COLLECTION` | academic_chunks | collection name | |
| `CHROMA_PERSIST_DIR` | ./chromadb_data | embedded Chroma | |
| `USE_EXTERNAL_CHROMA` | false | embedded vs server mode | Compose sets true |
| `EMBEDDING_MODEL_NAME` | sentence-transformers/all-MiniLM-L6-v2 | embeddings | |
| `LLM_PROVIDER` | **mock** | provider switch | `.env.example` says `openai` |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | none, gpt-4o-mini | OpenAI | |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | none, gemini-flash-latest | Gemini | |
| `GEMINI_FALLBACK_MODELS` | empty | extra Gemini models, comma-separated | `.env.example` says `gemini-1.5-flash` |
| `HF_MODEL_NAME` | google/flan-t5-base | Hugging Face | |
| `LLM_REQUEST_TIMEOUT_SECONDS` | 90 | LLM read timeout | |
| `LLM_REQUEST_RETRIES` | 2 | attempts per model | |
| `LLM_MAX_REQUESTS_PER_MINUTE` | 5 | global LLM limiter (0 or less disables) | |
| `DEFAULT_TOP_K` | 5 | used by `summarize_document()` | |
| `MAX_TOP_K` | 10 | nothing | the limit of 10 is hard-coded in `AskRequest` |
| `RATE_LIMIT_PER_MINUTE` | 60 | per-IP limiter | |

Frontend variables: `VITE_API_BASE_URL` (default empty), `VITE_LLM_TIMEOUT_MS` (default 120000) [api.ts]; `VITE_PROXY_TARGET` for the dev proxy [vite.config.ts].

## 5.8 Frontend routes and API functions (FACT)

| URL | Page | Calls |
|---|---|---|
| `/` | `HomePage` (marketing, innovation cards) | none |
| `/upload` | `UploadPage` + `FileUploader` | `uploadPdf()`, `getJobStatus()` |
| `/docs` | `DocsPage` + `DocList` | `listDocuments()`, `deleteDocument()` |
| `/chat/:docId` | `ChatPage` + `ChatBox` | `listDocuments()`, `askQuestion()` |

`api.ts` also exports `explainSection()` and `summarizeDocument()`, which no page uses.

## 5.9 How to say it in an interview

> "The data model is deliberately small. SQLite holds three tables: jobs for upload progress, documents for the finished files, and document trees that store each section tree as JSON. ChromaDB holds one record per chunk with the vector, the text and metadata like document id, page and section type. The API has nine endpoints under a versioned prefix, and Pydantic models define and validate every request and response. I should say clearly that there is no authentication, so it's a single shared library, not a multi-user product."

---

# Part 6. Design decisions and trade-offs

> **In simple words**
> A design decision is a choice where another choice was also possible. Interviewers do not want "I used X because it is popular". They want "I chose X over Y because of Z, and I would switch to Y if W happens".
> **How to read the tables:** the "Chosen" column is **FACT** (it is in the code). The "Why sensible" and "When the alternative is better" columns are **INFERENCE**, because the repo contains no written reasons. The commit messages are one-liners.

## 6.1 Application structure and the upload flow

| Decision | Chosen (FACT) | Why sensible (INFERENCE) | Main alternative | Alternative is better when |
|---|---|---|---|---|
| Backend framework | FastAPI + Pydantic [requirements.txt] | Type hints give validation and API docs for free; fast to build | Spring Boot (Java); Flask | The team is Java-only, or you need a large existing Java ecosystem |
| Frontend/backend split | React SPA on Vercel, API on Railway [vercel.json, railway.json] | Static hosting is cheap and simple; the API can be scaled separately | Server-rendered pages from the same app | You want one deployment and no CORS |
| Slow work after upload | `BackgroundTasks` + 202 + status polling [upload.py] | No extra infrastructure; easy to understand | Celery/RQ queue with a worker; WebSockets or server-sent events for push progress | Jobs must survive restarts or run on many machines; users should not wait on polling |
| Job state storage | SQLite `jobs` table [jobs.py] | Survives a restart as data; zero setup | In-memory dict; Redis; PostgreSQL | Many workers write at once (use PostgreSQL or Redis) |
| Metadata and trees | SQLite; tree saved as one JSON text column [tree_store.py] | One file; no extra service; the tree is always read whole | PostgreSQL with a `jsonb` column or a nodes table | Many users, or you need to query inside trees |
| Workers | 1 gunicorn worker [Dockerfile]; commit `c4bb7e0` changed 2 to 1 | Counters and caches live in memory, embedded Chroma and SQLite are single-process friendly, the embedding model uses RAM | Several workers behind a load balancer | You have shared state (Redis, a Chroma server, PostgreSQL) and need throughput |

**Honest note on the worker count.** The reason for 2 → 1 is NOT IN THE MATERIAL. The explanation above is a guess. If asked, say "I reduced it to one worker, and I believe it was about memory and in-process state, but I did not write that down".

## 6.2 Parsing, chunking and embedding

| Decision | Chosen (FACT) | Why sensible (INFERENCE) | Main alternative | Alternative is better when |
|---|---|---|---|---|
| PDF text | `pypdf` first, `unstructured` only if pypdf finds nothing [loader.py] | pypdf is light and fast; the fallback adds a safety net | PyMuPDF/pdfplumber; OCR (Tesseract) | Papers have complex layouts or scans; you need tables |
| Chunking unit | Recursive splitter, 3000 characters, 800 overlap, **per page** [chunker.py] | Splits at paragraphs and sentences first; per page gives exact page numbers | Token-based or semantic chunking; chunking across pages | You must keep sentences that cross pages, or the embedding model has a short input limit |
| Chunk size values | 3000 / 800, no measurement behind them (NOT IN THE MATERIAL) | Big enough to hold a whole idea | Tune on a gold question set | Always, once you have an evaluation set |
| Embedding model | Local `all-MiniLM-L6-v2`, 384 numbers, normalized [embeddings.py] | Free, offline, no API key, fast on CPU | A hosted embedding API; a bigger or domain-tuned model | Retrieval quality matters more than cost, or papers are in a niche field |
| Embedding fallback | Deterministic SHA-256 "hash embedding" when the model fails [embeddings.py] | Keeps the pipeline and tests running anywhere | Fail loudly with an error | Always better for production. The fallback hides a failure and gives meaningless retrieval |
| One Chroma collection | All documents in `academic_chunks`, filtered by `doc_id` [vectorstore.py] | One collection to manage; a metadata filter isolates a document | One collection per document | Hundreds of large documents, where per-collection indexes search faster and deletion is trivial |
| Chroma mode | Embedded `PersistentClient` by default, `HttpClient` when `USE_EXTERNAL_CHROMA` [vectorstore.py] | Same code runs locally and with Compose | Always use a server | Several app instances share one index |

## 6.3 Retrieval (StructRAG)

| Decision | Chosen (FACT) | Why sensible (INFERENCE) | Main alternative | Alternative is better when |
|---|---|---|---|---|
| Ranking base | `1 / (60 + vector_rank)` [hybrid_retriever.py: RRF_K] | Rank-based scores avoid mixing scales; k = 60 is the usual default | Use the raw cosine score directly | Scores are well calibrated and you want finer control |
| Using document structure | Boost chunks on pages of up to 3 sections picked by keyword overlap and section-type triggers, multiplier 1.0–2.0 [hybrid_retriever.py] | Academic questions often target a section (methods, results) | Plain vector search; vector + BM25 keyword fusion; RAPTOR-style summary tree | You have no evidence the boost helps (true today); keyword fusion helps for exact terms |
| Pool before ranking | Fetch `2 × top_k`, keep `top_k` [CANDIDATE_POOL_MULTIPLIER] | Gives the boost something to promote from | Fetch more (e.g. 5×) | The relevant chunk often sits far below rank `2 × top_k` |
| Confidence attenuation | Shrink the boost when few vector hits lie on the chosen sections [_adaptive_boost_factor()] | Avoids trusting a wrong section guess | Fixed boost | You have measured that a fixed boost works as well |
| Coherence re-rank | +0.1 for chunks of the top 2 sections [_coherence_rerank()] | Keeps related chunks together | No re-rank | Since 0.1 is about 3× the largest fused score, it acts as a hard "primary sections first" rule. A smaller bonus, or none, may be better |
| Query intent label | `classify_query()` with regexes, reported but unused in ranking | Prepared for later use; useful for debugging | Remove it, or really use it to change the boost | Use it only when it changes behaviour |

**Why the tree helps in theory (INFERENCE):** an academic paper has predictable parts. A question like "what accuracy did they get" is probably answered in Results, not in References. A page-level boost nudges the ranking that way.

**Why it may not help in practice (FACT + INFERENCE):**
- The tree is flat and the headings come from regex rules, so false "sections" (title, author names) can exist.
- A bad section guess can push the wrong chunks up. The confidence rule only softens this.
- Nothing has been measured.

## 6.4 Tree building

| Decision | Chosen (FACT) | Why sensible (INFERENCE) | Main alternative | Alternative is better when |
|---|---|---|---|---|
| Heading source | LLM-extracted table of contents if a real provider is set, else regex heuristics [tree_index.py] | Works with no API key; the LLM can find headings the regexes miss | PDF bookmarks/outline; font-size analysis with a layout parser | The PDF has an outline or clear font hierarchy, which is far more reliable than regexes |
| Section labels | 13 types, matched by keyword substrings in the heading [classify_section_type()] | Cheap and predictable | An LLM or a small classifier | Headings are unusual ("Our Take", "Datasets") and substring rules give `other` |
| Tree shape | Flat list; the `children` field exists but is never filled | Simple | A real nested tree (numbered headings "3.1" under "3") | You want depth-based scoring, which currently never takes effect |
| Per-node summary | First 280 characters of the first two pages (optionally an LLM summary for OpenAI) | Free and fast | LLM summaries for every node | Quality of title/summary matching matters more than cost |

## 6.5 LLM layer, safety and errors

| Decision | Chosen (FACT) | Why sensible (INFERENCE) | Main alternative | Alternative is better when |
|---|---|---|---|---|
| Provider design | One dispatch function over four providers; default `mock` [rag.py, config.py] | Switch vendors by config; CI needs no key | A provider interface class plus a registry | You add more providers, or want to unit-test each in isolation |
| Prompt | Fixed system prompt (explanation, 3 bullets, summary, citations, "I don't know" rule); chunks and question in the user prompt [rag.py] | Forces a consistent, grounded format | Separate prompts per mode; structured JSON output | You want to parse citations reliably, or `mode` should really change the behaviour (today it only adds a `[mode]` tag) |
| Conversation memory | None: each question stands alone [AskRequest has no history field] | Simpler and cheaper | Send the last few turns, or rewrite follow-ups into standalone questions | Users ask follow-ups such as "what about its limitations?" |
| Safety | Regex words such as "diagnose", "lawsuit" return a disclaimer [rag.py] | Cheap guard for high-stakes advice | A moderation model or a classifier | Always, since word lists wrongly block research questions like "how do the authors diagnose model errors?" |
| Errors | LLM failures are caught and returned as normal text with HTTP 200 [rag.py: answer_question()] | The chat UI shows a message instead of a crash | Return 502/503 with a structured error | A client needs to detect failure and retry automatically |
| Rate limiting | In-memory: 60 requests/min per IP and 5 LLM calls/min in total [main.py, rag.py] | Protects a free-tier LLM key from a burst | A gateway/Redis-backed limiter; per-user limits | Several workers or instances, or real users (one global LLM limit means one busy user blocks everyone) |

## 6.6 Delivery, tests and deployment

| Decision | Chosen (FACT) | Why sensible (INFERENCE) | Main alternative | Alternative is better when |
|---|---|---|---|---|
| Containers | Backend Dockerfile; Compose with Chroma + backend + frontend dev server [docker-compose.yml] | Same environment on every machine | Run services by hand | Docker is not available |
| CI | GitHub Actions: pytest; lint, test, build; Docker image builds [ci.yml] | Catches breakage early | Add deploy, caching, coverage, security scans | The project grows beyond a solo repo |
| Tests | 4 backend + 2 frontend tests; LLM patched out; 1-page sample PDF | Fast and key-free | Add unit tests for tree building and ranking; a retrieval evaluation set | You want to claim anything about quality |
| Hosting | Vercel + Railway | Quick, low-ops | A VM or a cloud container service with a persistent volume | You need durable storage for SQLite and Chroma (the repo shows none for Railway) |

## 6.7 The three decisions to revisit first (my ranking, INFERENCE)

1. **Measure retrieval.** Build a small gold set and compare plain vector search with the boosted ranking. This turns StructRAG from a claim into a result, or shows it should be dropped.
2. **Make tree and chunks honest.** Build a real nested tree, or stop calling it hierarchical; fix the 0.1 coherence scale; use the backend's `retrieval_confidence` in the UI instead of the raw score.
3. **Make state durable.** Persistent storage (a volume or a managed database/vector store) and a job queue, so a restart doesn't erase documents or strand jobs.

## 6.8 How to say it in an interview

> "Most of my choices favoured simplicity for a solo project: SQLite and an embedded Chroma so there was nothing else to run, a local MiniLM model so no API key was needed for embeddings, and FastAPI background tasks with polling instead of a queue. Each of those has a clear point where I'd switch: PostgreSQL or Redis when there are many workers, a queue like Celery when jobs must survive restarts, and a hosted or larger embedding model if retrieval quality becomes the bottleneck. The one choice I can't defend with data yet is the section-tree boost, because I never ran an evaluation, so that's the first thing I'd measure."

---

# Part 7. Weaknesses, bugs and honesty check

> **In simple words**
> Every project has gaps. In an interview, a gap you name first is a strength ("I know where my project is weak"). A gap the interviewer finds is a problem ("you didn't know"). This part lists your gaps, the proof for each, how to fix it, and how to say it honestly.
> **Rule for you:** never say something this part marks as unsupported, even if it appears in your own README or UI.

## 7.1 Your suspected list: confirmed or refuted

### S1. "Multi-turn" may be false: each question may be independent. **CONFIRMED.**

**Evidence (FACT):**
- `AskRequest` has exactly these fields: `doc_id`, `question`, `top_k`, `mode`, `temperature`. There is no history field [schemas.py: AskRequest].
- `answer_question(doc_id, question, top_k, mode, temperature)` takes no history [rag.py].
- `ChatPage.handleSend()` sends only the current question [ChatPage.tsx]. The `messages` list is used only to draw the screen.
- A search for "history", "conversation" and "multi-turn" in the code finds nothing.

**What is true:** the chat screen keeps a visible transcript, and you can ask many questions in a row. Each one is answered from scratch. A follow-up like "and what about its limitations?" has no referent for "its".

**Fix:** add `history` to the request (last N messages), then either put it in the prompt or have the LLM rewrite the follow-up into a standalone question before retrieval. Files: `schemas.py`, `qa.py`, `rag.py`, `api.ts`, `ChatPage.tsx`, tests.

**Honest way to say it:** "My chat interface lets you ask several questions in a row, but each question is answered independently. I don't pass conversation history to the model yet, so I wouldn't call it multi-turn in the strict sense. Adding history, or rewriting follow-ups into standalone queries, is my next step."

### S2. No retrieval-evaluation code. **CONFIRMED.**

**Evidence (FACT):** a search for recall, MRR, groundedness, benchmark and evaluation finds only keyword lists in the section classifier. There are no question sets, scripts, notebooks or result files. The only tests are 4 backend and 2 frontend tests [Part 3.11]. Nothing supports old resume percentages. Never quote them.

**Related overclaims (FACT):** `PROJECT_DOCUMENTATION.md` says "Patent-Pending Innovation", "no prior art does this", and "completely novel". The UI says "100% Cited, grounded answers" [HomePage.tsx: stats]. No evaluation or filing is in the material for any of these.

**Fix:** build a gold set of 30–50 questions with the correct page for each. Compare recall@5 and MRR for plain vector search against the boosted ranking. Files: a new `backend/eval/` folder, plus a test.

**Honest way to say it:** "I haven't evaluated retrieval quantitatively. The section-aware ranking is an idea I implemented and tried by hand, not something I've proven. The first thing I'd do is build a small question set and compare recall at 5 with and without the boost."

### S3. Chunk size 3000 and overlap 800 chosen by inspection, not measured. **NOT MEASURED in the repo (the "by inspection" part is NOT IN THE MATERIAL).**

**Evidence:** no script, log or note compares chunk sizes. The code comment says the sizes are "tuned to approximate ~800-token windows with ~200-token overlap" [chunker.py]. That describes a unit conversion (characters to tokens), not a tuning experiment. Only you know how you picked them.

**Extra concern (INFERENCE, verify on the model card):** small embedding models often cut long input at a few hundred tokens. If MiniLM does, a 3000-character chunk is partly invisible to its embedding. The limit itself is NOT IN THE MATERIAL.

**Honest way to say it:** "I picked 3000 characters with 800 overlap as starting values, aiming for roughly 800-token chunks, and I didn't tune them. I'd test a few sizes on a question set and also check the embedding model's input limit."

### S4. The live Railway backend returned "Application not found" on 27 Sep 2026. **NOT IN THE MATERIAL (your report); consistent with the repo's gaps.**

**What the material shows (FACT):**
- The README has one live URL: the Vercel frontend.
- Railway appears only as `backend/railway.json` and a commit message.
- No backend URL, production env vars or volume is in the repo.
- The frontend sends API calls to `VITE_API_BASE_URL`, which defaults to empty [api.ts].

**What follows (INFERENCE):**
- If the Railway app is gone or stopped, the Vercel pages still load, but upload, status and ask fail.
- With default Railway storage, SQLite, uploads and Chroma data sit in the container (INFERENCE) and would not survive a redeploy.

**Before the interview:** check the live URL yourself, and don't promise a live demo. Prepare a local fallback: `docker compose up --build` with a filled `.env`. Mock mode runs the pipeline but gives canned answers (FACT, ran it). A real key gives real answers. Also record a short screen video as a backup.

**Honest way to say it:** "The frontend is deployed on Vercel. My backend deployment on Railway is currently not running, so I'd show you the project running locally with Docker. Hosted backends on free plans don't keep disk storage, so I'd move to a persistent volume or a managed database."

### S5. Where is job status stored, and what happens after a restart or with more than one worker? **ANSWERED.**

**Evidence (FACT):** job status lives in the SQLite `jobs` table in `metadata.db` [jobs.py]. The background task runs inside the web process [upload.py]. The Dockerfile runs **one** worker [Dockerfile].

| Situation | What happens |
|---|---|
| Server restarts while a job runs | The task dies. The `jobs` row stays `processing` forever (no startup repair code). The saved PDF stays in the uploads folder. Vectors are written only at stage 5, so earlier stages leave none. |
| Server restarts and the disk is kept | Finished documents, trees and jobs are still there. |
| Server restarts and the disk is lost (INFERENCE for Railway) | Everything is gone; the UI would list nothing. |
| More than one worker (INFERENCE) | Any worker can read job status from the shared SQLite file. But each worker has its own connections and locks, its own embedding cache and model copy in RAM, and its own rate-limit counters, so limits multiply by the worker count. Embedded Chroma used by several processes is not safe, which is why Compose uses a Chroma server. |

**Honest way to say it:** "Job status is stored in SQLite, so it survives a restart as data, but the background task itself doesn't. A job interrupted by a restart stays in processing forever. For real scale I'd use a queue like Celery with Redis and a worker process, and PostgreSQL for state."

### S6. What happens when a PDF has no clear headings, and what does the mock provider return? **ANSWERED.**

**No headings (FACT (ran it)):** `build_document_tree()` returns a single node titled "Document", pages 1 to the last, type `other`. At question time:
- If a query word hits that node's summary text or the word "document", the whole paper counts as relevant. Every chunk then gets the same boost, so the order stays the vector order, but the score numbers become fused scores.
- If nothing matches, `navigate_tree()` returns nothing and the code uses plain vector order, but the response still says `retrieval_mode: "hybrid"`.

So with no headings the section layer does nothing useful, and nothing is reported as a failure.

**Mock provider (FACT (ran it)):** with no chunks, it returns "I don't know — the document doesn't contain enough information." Otherwise it returns a fixed text: three generic bullets plus a sentence naming the first 3 chunks, such as `[Title,3] (chunk-id)`. It ignores what the question says.

**Honest way to say it:** "If a paper has no detectable headings, the tree is a single 'Document' node, so the structural layer has nothing to boost, and the system effectively behaves like plain vector search. The mock provider returns a fixed template, so it's only for exercising the pipeline, not for judging answers."

## 7.2 Ranked findings (most serious first)

**How I ranked:** a finding ranks higher if it makes a resume or UI claim false, misleads a user, or loses data. "Honest way" is the sentence to use.

| # | Sev. | Finding | Evidence | Fix | Honest way to say it |
|---|---|---|---|---|---|
| 1 | HIGH | **UI shows wrong relevance numbers and a permanent "Low" confidence in tree mode.** Each source shows `round(score × 100)%`, and the badge averages scores with thresholds 0.75 and 0.45. The scores in tree mode are fused rank scores, at most 2.0/61 = 0.0328. | [SourceHighlight.tsx], [ChatPage.tsx: confidence], [hybrid_retriever.py: _tree_boosted_rrf()]; FACT (ran it): score 0.02246 → "2%" | Return a separate cosine `similarity` per source, and show the backend's `retrieval_confidence` | "The percentage in the UI currently shows my fused rank score, so it looks tiny. That's a display bug I know about." |
| 2 | HIGH | **"Multi-turn" is not implemented** [Claim 2] | S1 above | Add history | See S1 |
| 3 | HIGH | **Novelty and quality claims have no evidence** (patent-pending, no prior art, 100% grounded, "Confidence Gate", "narrative order") | [PROJECT_DOCUMENTATION.md], [HomePage.tsx], [PipelineDiagram.tsx]; no evaluation (S2) | Remove or soften; run an evaluation | "StructRAG is an experimental ranking idea. It isn't validated or patented." |
| 4 | HIGH | **StructRAG is weaker than described**: (a) tree is flat, so the depth bonus never applies; (b) `classify_query` is reported only and doesn't change ranking; (c) ATB-RRF has one ranked list, not two; (d) the documented "mismatched-section ×0.7" for chunks is not in the fusion code; (e) the +0.1 coherence bonus is 3.05 × the largest fused score, so it overrides the ranking; (f) the "hybrid" label shows even when nothing was boosted | [tree_index.py: _build_nodes_from_headings()], [hybrid_retriever.py]; FACT (ran it) for (e) and (f) | Build a nested tree or drop the claim; use a smaller bonus; label modes honestly | "The structural layer is simple: a flat list of detected sections and a rank boost. Some of what my documents describe, like depth weighting, isn't active yet." |
| 5 | HIGH | **Durability and deployment.** Demo backend reported down (S4). Default storage is container-local. CI doesn't deploy. | S4; [config.py], [ci.yml] | Persistent volume or managed DB; deploy step; health check | See S4 |
| 6 | HIGH | **No authentication, one shared library.** Anyone can list or delete any document. The 5-per-minute LLM limit is global, so one user can block everyone. | [docs.py], [rag.py: _enforce_llm_rate_limit()] | Add users and per-user ownership of documents | "It's a single-user demo, not a multi-user product." |
| 7 | HIGH | **Silent hash-embedding fallback.** If the model can't load, retrieval becomes meaningless while everything reports success. | [embeddings.py]; FACT (ran it): cosine of unrelated strings 0.71, 32 distinct values | Fail loudly in production; store the embedding model name with each document | "The fallback exists so the pipeline can run without the model, but it isn't semantic, so in production it should fail loudly." |
| 8 | HIGH | **Fragile background jobs.** Restart leaves `processing` forever. No timeout, no cleanup, no rollback; vectors can be orphaned if a later write fails. | [upload.py], [jobs.py] | Queue + startup repair; transaction-like cleanup | See S5 |
| 9 | MED | **Default title is the job UUID** when the user gives no title (the original file name is lost) | [upload.py]; FACT (ran it) | Use `file.filename` as the default | "A small bug: untitled uploads show an ID." |
| 10 | MED | **Weak upload validation.** A file is rejected only if both the content type and the extension look wrong. No size limit. | [upload.py]; FACT (ran it): `virus.exe` with PDF content type got 202 | Check the `%PDF` header, add a size cap | "Validation is minimal. I'd check the file signature and size." |
| 11 | MED | **Errors hidden in HTTP 200.** LLM failures return 200 with an error text; ingestion errors and provider errors are shown to users as raw text. | [rag.py: answer_question()], [upload.py]; FACT (ran it) | Return 502/503 and a safe message | "I turn LLM failures into chat messages, but a proper error status would be better." |
| 12 | MED | **The 5-per-minute LLM limiter also blocks the mock and is shared by all endpoints** | FACT (ran it): 6th ask in a minute failed | Make it per user and skip it for the mock | "It protects a free-tier key, but it is global." |
| 13 | MED | **Polling vs. the per-IP limit.** One poll per second plus the upload reaches 60 requests in a minute; the 60th poll is refused. A 429 carries no CORS header (the limiter wraps CORS), so a cross-origin browser sees a network error. The UI stops polling on any error. | FACT (ran it): simulated clock, poll #60 → 429; FACT (ran it): 429 had no `access-control-allow-origin` | Poll every 2–3 s with backoff; exempt `/status`; put CORS outside the limiter; keep polling on 429 | "Polling every second sits right at my rate limit, so a long job can leave the page showing an error even though ingestion continues." |
| 14 | MED | **Prompt glue contains literal `\n` text** instead of line breaks | FACT (ran it) [rag.py: USER_PROMPT_TEMPLATE] | Use single backslash `\n` | "A typo in the prompt template. It works, but isn't what I intended." |
| 15 | MED | **Safety filter blocks normal research questions** ("diagnose", "lawsuit") | FACT (ran it) | Use a classifier; or check intent, not words | "A keyword guard is crude." |
| 16 | MED | **Heading detection is weak.** Paper title and author names became "sections". Sub-sections are stored as top-level nodes. Chunk section labels are page-level (later nodes overwrite earlier ones on shared pages). The LLM table of contents sees only the first 20 pages × 600 characters. | FACT (ran it): synthetic test; [chunker.py], [tree_index.py] | Use PDF outlines or font sizes; nest by numbering | "Heading detection is regex-based, so it makes mistakes on titles and author lines." |
| 17 | MED | **Scanned PDFs fail in practice (INFERENCE).** pypdf finds no text, so the `unstructured` fallback runs, but the pinned package has no extras and the Dockerfile installs no OCR tools (Tesseract/Poppler). The job would most likely end `failed`. | [loader.py], [requirements.txt], [Dockerfile] | Detect scans and run OCR (Tesseract/ocrmypdf) | "Scanned PDFs aren't supported yet. I'd add an OCR step." |
| 18 | MED | **Tables and equations aren't handled.** Plain text extraction; a table can be split across chunks and lose its header. | [loader.py]; FACT (ran it): 120-row table → 2 chunks | Table-aware parser; keep header rows | "I extract plain text, so tables and formulas are weak spots." |
| 19 | MED | **Embedding input limit may truncate 3000-char chunks** | INFERENCE, limit NOT IN THE MATERIAL | Check the model card; use ~model-limit chunks | "I need to verify the model's input length against my chunk size." |
| 20 | MED | **Tests are thin.** No tests for the tree, ranking, providers or error paths. The only PDF is 1 page. TestClient finishes background tasks before returning, so the polling loop is not truly tested. | [tests/]; FACT (ran it): job already completed at the first poll | Add unit tests for `tree_index` and `hybrid_retriever`; a multi-page fixture | "My tests cover the main flow but not the StructRAG logic." |
| 21 | MED | **Compose pins Chroma server 0.5.5 while the client is 0.5.23** | [docker-compose.yml], [requirements.txt] | Match versions | "I'd align the versions." |
| 22 | MED | **"Summary" uses only 5 chunks**, so it can't summarize a whole paper. `mode` only adds a text tag. `explain-section` doesn't filter by section. | [rag.py: summarize_document()] | Map-reduce summary; real modes | "Summary is based on the top few chunks." |
| 23 | LOW | **UI shows only 3 sources** even when 10 were retrieved; "sources" are retrieved chunks, not verified citations | [MessageBubble.tsx] | Show all; verify citations | "Sources are the evidence retrieved, not a claim-by-claim proof." |
| 24 | LOW | **Dead or unused parts:** Redis service, `ErrorResponse`, `MAX_TOP_K`, `BACKEND_HOST/PORT`, `ENVIRONMENT`, `tree_exists()`, `explainSection()`/`summarizeDocument()` in the UI, the whole `langchain` package (only the splitter is used), `pytest` in production requirements | grep results | Remove | "A few leftovers I'd clean up." |
| 25 | LOW | **Docs drift** (see 7.3) | | Update docs | "My docs lag behind the code in places." |
| 26 | LOW | **`page_count` counts pages with text**, not all pages | [upload.py] | Use the PDF's real page count | |
| 27 | LOW | **Hugging Face path rebuilds its pipeline on every call** | [rag.py: _call_hf()] | Cache it | |
| 28 | LOW | **Embedding cache is unbounded** and in memory | [embeddings.py] | LRU cache or disk | |

## 7.3 CORRECTIONS TO MY NOTES

Where your notes and the material differ, the material wins.

1. **Endpoint paths.** Every API route has the prefix `/api/v1`: `/api/v1/upload`, `/api/v1/status/{job_id}`, `/api/v1/ask`, `/api/v1/explain-section`, `/api/v1/summarize`, `/api/v1/tree/{doc_id}`, `/api/v1/tree/{doc_id}/navigate`. Only `/` and `/health` have none [main.py].
2. **Citations and score.** The `score` is a cosine similarity only on the vector-only path. On the boosted path it is the fused rank score, at most 0.0328. Also, the "citations" are the retrieved chunks, not a verified list of what the answer used.
3. **"Query intent classification … adaptive boost from intent-to-section match".** The intent label (`factual`/`conceptual`/`structural`) is only returned in the response. The boost comes from keyword trigger groups matched against section types, averaged over the top-3 sections and capped at 5 [hybrid_retriever.py].
4. **Tree depth.** The boost includes depth, but the tree is flat, so depth is always 0.
5. **"ATB-RRF = reciprocal rank fusion".** There is one ranked list (vector). The tree gives a page set and a multiplier. It is not a fusion of two ranked lists.
6. **"Fallback to plain vector search when there is no tree".** The code does this, but new uploads always get a tree (at least one "Document" node), so that path is rarely used. The code also falls back to vector order when navigation finds nothing, but still reports `hybrid`.
7. **"Default is mock".** True in the code. But `.env.example` and the README set `LLM_PROVIDER=openai`, so copying that file changes it.
8. **pypdf with unstructured fallback.** True. The fallback runs only if pypdf raises or finds no text. It is unlikely to rescue scanned PDFs (INFERENCE, Part 7.2 #17).
9. **"Deployed as Vercel + Railway + Gemini free tier".** Only the Vercel URL (README) and the two config files are in the material. The Railway URL, the live provider, "Gemini" in production and "free tier" are NOT IN THE MATERIAL.
10. **"Chunking … page number kept".** True. Also note chunks are cut per page and sized in characters, not tokens.
11. **"top_k default 5, max 10".** True, but the maximum is hard-coded in the schema; the `MAX_TOP_K` setting is unused.
12. **"Docker and Compose".** True. Compose also starts an unused Redis, and the Chroma server version (0.5.5) differs from the client (0.5.23).
13. **"mode" parameter.** It exists but only adds a `[mode]` tag in the prompt.
14. **Tests.** 4 backend + 2 frontend tests exist, but none cover the StructRAG modules.

**Project documents that disagree with the code** (so don't repeat them):

| Document says | Code says |
|---|---|
| "React 19" [PROJECT_DOCUMENTATION.md §2, §10] | `react ^18.3.1` [package.json] |
| "Python 3.10" [§2, architecture diagram] | Python 3.11 [Dockerfile, ci.yml] |
| "builds parent-child hierarchy" [§3] | flat list, `children` never filled |
| Standard RRF uses "a static constant (e.g., 1.6×)" [§5] | Standard RRF has no such boost. It sums `1/(k+rank)` over lists. |
| Query type "behavior: tree navigation emphasized…" [§6] | The type is only reported |
| "falls back to unstructured for scanned PDFs" [§3] | Fallback runs only when pypdf finds no text; OCR tools absent (INFERENCE) |
| `CHROMA_HOST` default `localhost`; `GEMINI_FALLBACK_MODELS` default `gemini-1.5-flash` [§12] | Code defaults: `chroma`; empty string |
| "Redis: optional caching" [§10] | Not used by any code |
| "~3,600 lines of code" [§11] | My count: backend 2,826 + backend tests 117 + frontend 1,436 = 4,379 lines |
| "Status: Patent-Pending" [title block] | NOT IN THE MATERIAL |
| README: "production-ready" | No auth, no durability, thin tests |
| README file list [Repository Layout] | Omits the four StructRAG files |

## 7.4 RESUME CLAIM AUDIT

Your two bullets are called [Claim 1] and [Claim 2].

**[Claim 1]:** "Built and deployed a full-stack Retrieval-Augmented Generation (RAG) app that answers questions over academic papers using text embeddings and vector search."

**[Claim 2]:** "Implemented a REST API backend for document chunking, indexing, retrieval, and multi-turn LLM-based Q&A."

| Claim part | Verdict | Evidence |
|---|---|---|
| [Claim 1] "Built a full-stack RAG app" | **SUPPORTED** | FastAPI backend + React frontend + ChromaDB + LLM providers; 9 endpoints [Parts 1, 5] |
| [Claim 1] "answers questions over academic papers" | **SUPPORTED** (works end to end in a test run); answer quality **NOT IN THE MATERIAL** | Upload → status → ask test passes with a patched LLM [test_ask.py]; no quality evaluation; only a 1-page sample PDF |
| [Claim 1] "using text embeddings and vector search" | **SUPPORTED** | MiniLM embeddings, Chroma cosine search [embeddings.py, vectorstore.py] |
| [Claim 1] "deployed" | **PARTLY** | `vercel.json`, `railway.json`, README live URL for the frontend. Backend deployment evidence is NOT IN THE MATERIAL; your note says it was down on 27 Sep 2026 |
| [Claim 2] "REST API backend" | **SUPPORTED** | Part 5.6 |
| [Claim 2] "document chunking, indexing" | **SUPPORTED** | [chunker.py], [upload.py], [vectorstore.py] |
| [Claim 2] "retrieval" | **SUPPORTED** | Vector search + section boost [hybrid_retriever.py]; effectiveness unmeasured |
| [Claim 2] "LLM-based Q&A" | **SUPPORTED** | OpenAI/Gemini/HF code paths [rag.py]; default provider is the mock |
| [Claim 2] "multi-turn" | **NOT IN THE MATERIAL, and the code contradicts it** | No history field anywhere (S1) |
| Old resume percentage improvements | **NOT IN THE MATERIAL** | No evaluation code or results |
| "StructRAG improves retrieval" | **NOT IN THE MATERIAL** | Nothing measures it |
| "Patent-pending" | **NOT IN THE MATERIAL** | Only a "Draft Patent Claim" paragraph in a document |

**Overall: [Claim 1] PARTLY, [Claim 2] PARTLY.**

**Safer wording you can use on your resume (these match the repo):**
- "Built a full-stack RAG application (FastAPI, ChromaDB, sentence-transformers, React/TypeScript) that answers questions over uploaded academic PDFs, returning answers with page-level source snippets; ingestion runs as an asynchronous background job with progress polling."
- "Implemented REST endpoints for PDF ingestion (page-aware chunking, embeddings, vector indexing), retrieval with an experimental section-aware re-ranking layer, and LLM Q&A with swappable providers (OpenAI, Gemini, Hugging Face, mock); added pytest/Vitest tests, Docker, and GitHub Actions CI."

## 7.5 How to say it in an interview

> **If asked "what are the weaknesses?"** "Three things. First, I haven't evaluated retrieval, so I can't prove my section-aware ranking helps. Second, my chat isn't truly multi-turn, because each question is answered independently. Third, the system isn't production-ready: there's no authentication, background jobs live inside the web process, and my hosted backend isn't currently running. I know the fix for each: a gold question set, history in the request, and a queue with persistent storage."

---

# Part 8. Pitch scripts

> **In simple words**
> A pitch is the answer to "tell me about your project". You need three sizes, because the interviewer's time changes. Speak them aloud, not just read them. Every sentence below is backed by the repo. Square brackets `[ ]` are places where you say whichever version is true on the day.
> **Pace:** about 150 spoken words per minute (my rule of thumb, INFERENCE). The 30-second script is about 90 words, the 2-minute about 300, the 5-minute about 720 (I counted them). At that pace the 5-minute script takes about 4.8 minutes, and the section labels in 8.3 add up to that.

## 8.1 The 30-second pitch

> "I built StructRAG, a web app for asking questions about research papers. You upload a PDF; the backend splits it into chunks, turns them into embeddings and stores them in a vector database, ChromaDB. When you ask a question, it retrieves the most relevant chunks, gives them to an LLM, and returns an answer with page-level sources. It's FastAPI and Python on the backend, React with TypeScript on the front, with Docker and GitHub Actions CI. I also added an experimental section-aware ranking layer, which I haven't formally evaluated yet."

## 8.2 The 2-minute pitch

> "The problem I wanted to solve is that finding an answer inside a long research paper is slow. So I built a RAG application, which stands for Retrieval-Augmented Generation. You upload a paper as a PDF, and then you can ask questions about it.
>
> Here's how it works. Uploading is asynchronous: the FastAPI server saves the file, creates a job record and immediately returns a 202 with a job id. A background task then reads the PDF page by page, splits each page into chunks of about three thousand characters with some overlap, converts every chunk into a 384-number embedding with the MiniLM sentence-transformer model, and stores the vectors in ChromaDB along with the page number. The browser polls a status endpoint, so the user sees progress.
>
> When someone asks a question, I embed the question, search Chroma for the nearest chunks of that document, put them into a prompt and call an LLM. The LLM layer is pluggable: OpenAI, Gemini, Hugging Face, or a mock provider, so my tests and CI run without any API key. The response includes the answer plus the source chunks with page numbers and scores.
>
> The part I'm most interested in is what I called StructRAG. During ingestion I detect headings and build a section tree, and at query time I boost the ranking of chunks that fall in the sections the question seems to be about, for example Methodology for a how-question.
>
> To be honest about limits: I haven't run an evaluation to prove that boost helps, the chat sends each question independently without conversation history, and there's no authentication, so it's a demo rather than a production system. [The frontend is on Vercel; say whether the backend is currently running.] If I continued, I'd build an evaluation set first."

## 8.3 The 5-minute pitch

> **1. The problem and the idea (about 30 seconds).**
> "Research papers are long and dense. A student who wants to know 'what dataset did they use?' has to skim fifteen pages. My project is a Retrieval-Augmented Generation system for exactly that. The idea behind RAG is simple: don't ask the language model to remember the paper. First retrieve the few passages that matter, then ask the model to answer only from them, and show where they came from.
>
> **2. Architecture (about 25 seconds).**
> "The backend is FastAPI. The frontend is a React and TypeScript single-page app built with Vite and Tailwind. State is in three places: SQLite for jobs, documents and section trees; ChromaDB for the vectors and their metadata; and the file system for the uploaded PDFs. In development, Docker Compose runs the backend, a Chroma server and the frontend together.
>
> **3. The two flows (about 70 seconds).**
> "On upload, the server saves the PDF and returns 202 with a job id straight away, because parsing and embedding can take a while. A background task does five stages: parse the pages with pypdf, build a section tree, chunk each page at three thousand characters with eight hundred overlap, embed the chunks in batches, and write everything to the stores. Progress is written to the jobs table, and the browser polls it every second.
>
> "On a question, the server embeds the query and asks Chroma for the nearest chunks, filtered to that one document. Then my ranking layer runs. I fetch twice as many candidates as needed, give each a reciprocal-rank score of one over sixty plus its rank, and multiply it by a boost between one and two if its page belongs to one of the sections that the question points to. The top chunks go into a prompt that tells the model to answer from the context only and to say 'I don't know' otherwise. The answer comes back with the chunks as sources.
>
> **3b. How the StructRAG layer works (about 60 seconds).**
> "During ingestion, I look for headings in the text. If an LLM is configured I ask it for a table of contents; otherwise, I use regular-expression rules for numbered headings, all-caps lines and short title-case lines. Each heading becomes a section with a page range, and I label it with one of thirteen types such as introduction, methodology or results. At question time, I score the sections against the question using word overlap and a small keyword table, so a how-question favours methodology and a results-question favours results. I take the top three sections and boost the chunks on their pages. I also compute a confidence value, the share of the top vector hits that land on those pages, and when it's low I shrink the boost. My hypothesis is that this helps on questions aimed at one part of a paper, but that is exactly what I still need to test.
>
> **4. Engineering problems I solved (about 40 seconds).**
> "A few real fixes from the commit history. I hit broken-pipe errors from forked processes inside uvicorn workers and fixed them by switching the multiprocessing start method to spawn and turning off tokenizer parallelism. I improved answer quality by sending the full chunk text in the prompt instead of only a short snippet, and raised the Gemini output limit so answers weren't cut off. When a Gemini model name I configured was retired and returned a 404, I added model fallback so the call tries the next model. And I made the CORS setting accept several origins so the Vercel frontend can call the backend.
>
> **5. Honest limitations and next steps (about 50 seconds).**
> "Let me be upfront about what's missing. First, I haven't measured retrieval quality. The section-aware boost is an idea I implemented, and the right next step is a small gold question set and a comparison of recall at five with and without the boost. Second, the tree is flat and built with regex heuristics, so title and author lines can be mistaken for headings. Third, the chat is not multi-turn: each question is independent. Fourth, ingestion runs as a background task inside the web process, so a restart mid-job leaves it stuck, and there's no authentication. [Say the truth about the hosted backend today.] For a production version I'd add a job queue, persistent storage, auth and the evaluation.
>
> **6. Close (about 15 seconds).**
> "What I learned is how many moving parts sit between 'I have embeddings' and 'users can trust the answer': chunking, ranking, prompt design, error handling, and above all evaluation. I'm happy to go deep on any part."

## 8.4 WHAT I BUILT (fill the blanks)

Fill the blanks with true, specific details. Interviewers like precise personal stories.

| Area | What exists (FACT) | Where | My own detail (fill in) |
|---|---|---|---|
| Problem | Q&A over uploaded academic PDFs | README.md | Why I picked this project: ___ |
| Backend | FastAPI app, 9 endpoints under `/api/v1`, Pydantic models, CORS, rate limiter | `backend/app/` | The part I wrote first: ___ |
| Ingestion | pypdf parse, section tree, 3000/800 chunking, MiniLM embeddings, Chroma + SQLite writes, progress 0–100 | `upload.py`, `services/` | The hardest bug in ingestion: ___ |
| Retrieval | Vector search + section boost (k = 60, boost 1.0–2.0, pool 2×top_k) | `hybrid_retriever.py` | Why I built it: ___ |
| Section tree | Regex/LLM headings, 13 section types, stored as JSON in SQLite | `tree_index.py`, `tree_store.py` | What I'd change: ___ |
| LLM layer | OpenAI, Gemini (model fallback + retries), Hugging Face, mock; default mock | `rag.py` | Which provider I used for demos: ___ |
| Frontend | React 18 + TypeScript + Vite + Tailwind; upload, documents list, chat | `frontend/src/` | What I designed myself: ___ |
| Quality | 4 backend + 2 frontend tests | `tests/`, `src/tests/` | What I'd test next: ___ |
| DevOps | Dockerfile, Compose, GitHub Actions (3 jobs), Vercel and Railway config | root, `.github/` | Where I deployed and what it costs: ___ |
| Evaluation | none | n/a | What I plan to measure: ___ |
| Timeline | 9 commits, 2026-03-06 to 2026-08-13 (first commit; StructRAG on 03-27; deploy/UI on 07-07; live URL on 08-13) | `git log` | Hours spent / how I split my time: ___ |
| Papers I tested | NOT IN THE MATERIAL (the repo has one 1-page sample) | n/a | Papers I really tried: ___ |
| Live status today | NOT IN THE MATERIAL | n/a | Frontend: ___ Backend: ___ |

**Numbers you may quote (all from the repo):** 9 endpoints, 13 section types, 384-dimension embeddings, chunk 3000 / overlap 800 characters, k = 60, boost range 1.0 to 2.0, candidate pool 2 × top_k, top_k 1–10 (default 5), default temperature 0.2, 4 LLM providers, 3 SQLite tables, 4 backend + 2 frontend tests, 3 CI jobs, 9 commits.

**Strong true points you can lead with:**
- Asynchronous ingestion with a 202 response, a jobs table and progress polling.
- Page-aware chunking, so every source has a page number.
- A pluggable LLM layer with retries and Gemini model fallback; a mock provider so CI needs no key.
- Pydantic validation, typed settings from environment variables, Docker, Compose and CI.
- Real debugging stories from the commit history (Part 8.3, section 4).

## 8.5 Things I must NOT claim

| Do not say | Because (evidence) |
|---|---|
| "Multi-turn", "conversation memory", "it remembers earlier questions" | No history is sent [Part 7, S1] |
| Any percentage, accuracy, recall or "X% better" number | No evaluation exists [S2] |
| "StructRAG is proven better", "novel", "patented", "patent-pending", "no prior art" | Nothing measured; no filing in the material |
| "Production-ready", "scalable", "secure", "multi-user" | No auth, in-process jobs, thin tests, storage not durable |
| "It is live and working" | Backend status unknown; your 27 Sep note says it was down |
| "Answers are always grounded", "no hallucinations", "100% cited" | Nothing checks answers against sources |
| "Hierarchical tree" or "depth-weighted scoring works" | Tree is flat, so depth is always 0 |
| "Query intent drives retrieval" | The intent label is reported only |
| "It fuses two ranked lists" or "hybrid keyword + vector search" | One ranked list; no keyword search exists |
| "It handles scanned PDFs, tables, equations, figures" | Plain text extraction only; OCR missing |
| "3000/800 are tuned/optimal" | Not measured |
| "I use Redis / caching" | Redis is unused; only an in-memory embedding cache exists |
| "Tests cover the retrieval" or "CI deploys automatically" | Tests don't cover StructRAG; CI has no deploy |
| "The UI lets you explore sections or summarize" | The UI calls only upload, status, list, delete, ask |
| "The hash fallback is a real embedding" | It isn't semantic |
| "I used Gemini free tier in production" | NOT IN THE MATERIAL; say it only if true |
| "The mock gives real answers" | It returns a fixed template |
| "Sources prove each statement" | They are the retrieved chunks, not verified evidence |
| Any user count or feedback | NOT IN THE MATERIAL |

## 8.6 How to say it in an interview

> **If they ask "is this your own work?"** "Yes, it's a solo project and all the commits are mine. I can walk through any file, and I can also tell you which parts I consider unfinished."

> **If they ask for the one thing to remember.** "It's a working end-to-end RAG pipeline with an asynchronous ingestion design and swappable LLM providers, plus an experimental ranking idea that I know I still have to evaluate."

---

# Part 9. Interview question bank (40 questions)

> **In simple words**
> Cover the answer, say yours aloud, then compare. The model answers are short and spoken-style (3–6 sentences) so you can say them without reading. Each has the most likely next question, because interviewers rarely stop at the first answer.
> **Every answer is consistent with the repo.** Where the honest answer is "I haven't done that", the answer says so and shows what you would do. The 9 questions from your must-ask list are marked **[MUST-ASK]**.

Tiers: **Basic** (Q1–Q10), **Follow-up** (Q11–Q20), **Deep** (Q21–Q30), **Tricky / trap** (Q31–Q40).

---

## Tier 1: Basic

**Q1. Tell me about your project.**
*Model answer:* "It's a web app called StructRAG for asking questions about research papers. You upload a PDF, the backend splits it into chunks, converts them into embeddings and stores them in ChromaDB. When you ask a question, it finds the most relevant chunks, passes them to an LLM, and returns the answer with page-level sources. The backend is FastAPI and the frontend is React with TypeScript. I also added an experimental section-aware ranking layer that I haven't evaluated yet."
*Next follow-up:* "What was the hardest part?"

**Q2. What is RAG, and why use it instead of asking an LLM directly?**
*Model answer:* "RAG means Retrieval-Augmented Generation: first retrieve relevant text, then ask the LLM to answer using that text. A plain LLM has never seen my PDF and can't read it unless I paste it in. RAG finds only the useful passages, so the prompt stays small, and it lets me show which pages the answer came from. It reduces made-up answers, but it doesn't remove them."
*Next follow-up:* "Why not paste the whole PDF into the prompt?" (Answer: long papers may exceed the model's context window and cost more; I don't know the exact windows of my models, since that's not in my repo; and too much text can distract the model.)

**Q3. What is an embedding?**
*Model answer:* "An embedding is a list of numbers that represents the meaning of a piece of text. I use the MiniLM sentence-transformer, which gives 384 numbers per chunk. Texts with similar meaning get vectors that point in similar directions. That lets me search by meaning instead of by exact words."
*Next follow-up:* "How do you compare two embeddings?" (Cosine similarity; my vectors are normalized, so it's a dot product.)

**Q4. Which technologies did you use, and why FastAPI?**
*Model answer:* "Backend: FastAPI, Pydantic, ChromaDB, SQLite, sentence-transformers and pypdf. Frontend: React, TypeScript, Vite and Tailwind. I chose Python because the embedding and vector libraries are there, and FastAPI because type hints give me request validation, background tasks and automatic API docs. Coming from Java, I think of it as Spring Boot with Pydantic models as DTOs."
*Next follow-up:* "FastAPI versus Flask or Spring Boot?" (Flask has less built-in validation; Spring Boot would be natural in a Java shop but the ML libraries are in Python.)

**Q5. What is chunking and why do you need it?**
*Model answer:* "Chunking means cutting the document into smaller pieces. One vector for a whole paper would blur everything together, and the LLM can only take so much text. Small pieces also give precise page citations. I use LangChain's recursive splitter, 3000 characters with 800 overlap, applied page by page."
*Next follow-up:* "What happens to a sentence at a chunk boundary?" (Overlap keeps it whole in one neighbouring chunk, but only inside a page, because I chunk page by page.)

**Q6. [MUST-ASK] Walk me through the path from uploading a PDF to getting a cited answer.**
*Model answer:* "The browser posts the PDF to `/api/v1/upload`. The server saves it, creates a job row and returns 202 with a job id. A background task parses the pages with pypdf, builds a section tree, chunks each page, embeds the chunks, and writes the vectors to Chroma and the tree to SQLite, while the browser polls the status endpoint. When I ask a question, the server embeds it, searches Chroma for the nearest chunks of that document, boosts chunks from the sections the question points to, builds a prompt and calls the LLM. The response has the answer plus the retrieved chunks with page, snippet and score, and the UI shows them under the answer."
*Next follow-up:* "Where could that fail?" (Parsing a scanned PDF, the embedding model not loading, a restart mid-job, the LLM rate limit, or the LLM returning an unsupported claim.)

**Q7. What endpoints does your REST API have?**
*Model answer:* "Under `/api/v1`: `POST /upload`, `GET /status/{job_id}`, `GET /docs`, `DELETE /docs/{doc_id}`, `POST /ask`, `POST /explain-section`, `POST /summarize`, `GET /tree/{doc_id}` and `POST /tree/{doc_id}/navigate`. That's nine, plus `/` and `/health`. Requests and responses are Pydantic models, so invalid input gets a 422 automatically. My web UI uses only upload, status, list, delete and ask. The others are available through the API."
*Next follow-up:* "Why version the API with `/v1`?" (So breaking changes can go under `/v2` without breaking existing clients.)

**Q8. How do you use Docker?**
*Model answer:* "The backend has a Dockerfile based on python 3.11 slim that starts gunicorn with a uvicorn worker. Docker Compose starts the backend, a Chroma server and the frontend dev server together for local development. The same Dockerfile is what the hosting platform builds from. I also have an unused Redis service in the compose file that I should remove."
*Next follow-up:* "Why gunicorn with uvicorn?" (Gunicorn manages the process; the uvicorn worker runs the async FastAPI app. I use one worker.)

**Q9. What does your CI pipeline do?**
*Model answer:* "GitHub Actions runs on pushes to main and on pull requests. The backend job installs dependencies and runs pytest. The frontend job runs lint, tests and a production build. A Docker job builds both images once those pass. It doesn't deploy, and the tests don't yet cover the section-tree or ranking code."
*Next follow-up:* "Does CI need an API key?" (No. The default LLM provider is a mock, and the ask test patches the generate function.)

**Q10. How do you handle configuration and secrets?**
*Model answer:* "All settings come from environment variables through a Pydantic Settings class, with defaults in code. Keys such as the OpenAI and Gemini keys are never in the code. `.env` is git-ignored, and `.env.example` only has placeholders. I checked the repo and history and found no real secret. One caution is that `.env.example` sets the provider to openai while the code default is mock."
*Next follow-up:* "How does the frontend know the backend URL?" (Through `VITE_API_BASE_URL` at build time; in development the Vite proxy forwards `/api` to the backend.)

---

## Tier 2: Follow-up

**Q11. [MUST-ASK] Why does upload return 202 and not 200?**
*Model answer:* "Because the work isn't finished when I reply. 202 Accepted means 'I've taken your request and I'm processing it'. Parsing and embedding a paper can take a while, and the UI even warns that the first upload can take 30 to 60 seconds, so holding the request open would risk timeouts. Instead I return a job id immediately and the client polls a status endpoint that reports progress. The limitation is that the task runs inside the web process, so a restart mid-job leaves the job stuck; a queue would fix that."
*Next follow-up:* "What would you use instead of polling?" (Server-sent events or WebSockets for push updates, with a real queue like Celery behind it.)

**Q12. [MUST-ASK] Why ChromaDB? Why MiniLM? Why chunk size 3000 and overlap 800?**
*Model answer:* "ChromaDB because it runs embedded with no extra server, supports a metadata filter for per-document search, and switches to a server mode for Docker; I didn't benchmark alternatives like pgvector or FAISS. MiniLM because it's small, free, runs on CPU, and needs no API key; I didn't compare it with larger or hosted models. For 3000 and 800, my code comment says I aimed for roughly 800-token chunks, but I picked those values by judgement and didn't tune them with an experiment. I'd treat all three as defaults and test them on a question set, and I'd also check MiniLM's input limit against my chunk size."
*Next follow-up:* "How would you choose chunk size properly?" (Build a gold set; compare recall at 5 and MRR for several sizes; also look at answer quality.)

**Q13. [MUST-ASK] Why do you have a mock provider?**
*Model answer:* "So the whole pipeline can run without an API key, a network or any cost, which matters for CI and local development. When the provider is `mock`, upload, chunking, embedding, retrieval and the response shape all run for real, and only the final text is a fixed template. It also lets the tree builder skip LLM calls. The limit is that it ignores the question, so it can't test answer quality, and my tests actually patch the generate function anyway."
*Next follow-up:* "Doesn't a mock hide real bugs?" (Yes, for provider-specific code like retries and model fallback. I'd add a small opt-in integration test with a real provider.)

**Q14. [MUST-ASK] What does "multi-turn" mean in your project?**
*Model answer:* "Honestly, in my project it means the chat screen lets you ask several questions in a row and shows the transcript. It does not mean the model remembers earlier turns: each question goes to the backend alone, with no history. So a follow-up like 'what about its limitations?' won't work, because 'its' has no referent. I should have described it as a multi-question chat. To make it truly multi-turn, I'd send the last few messages and rewrite follow-ups into standalone queries before retrieval."
*Next follow-up:* "How would you stop the history growing without limit?" (Keep only the last N turns, or summarize older ones, within a token budget.)

**Q15. [MUST-ASK] Does RAG stop hallucination?**
*Model answer:* "No, it reduces it but doesn't stop it. RAG gives the model real text to answer from, but the model can still misread, mix passages, or add things that aren't there. In my project the prompt says to answer only from the context and to say 'I don't know' otherwise, and the default temperature is low. But I don't check the answer against the chunks, so I can't promise it's grounded. Retrieval can also miss the right passage, and then the model has nothing correct to use."
*Next follow-up:* "How would you measure hallucination?" (Groundedness: check each claim in the answer against the retrieved chunks, by human review or an LLM judge, on a test set.)

**Q16. How do your citations work? Are they reliable?**
*Model answer:* "Two things happen. I ask the LLM to write `[title, page]` in its answer, and separately the API returns the retrieved chunks as `sources`, each with page, snippet, chunk id and score. So the sources show what evidence was given to the model, not a verified link from each sentence to its evidence. The UI shows only the first three. Also, in my boosted mode the score is a fused rank score, so it isn't a similarity percentage."
*Next follow-up:* "How would you verify citations?" (Ask the model for chunk ids per claim, then check that each cited chunk really supports the sentence.)

**Q17. What is temperature and what do you use?**
*Model answer:* "Temperature controls randomness: low values give more predictable text, high values give more varied text. My default is 0.2, and the user can move a slider from 0 to 1. I chose it low because answering from a document should be stable and factual. Low temperature makes output more consistent, but it doesn't guarantee correctness."
*Next follow-up:* "Does temperature zero make it fully deterministic?" (Not guaranteed; it's more deterministic, but I wouldn't promise identical output.)

**Q18. How does cosine similarity work, and why normalize vectors?**
*Model answer:* "Cosine similarity compares the direction of two vectors: the dot product divided by the product of their lengths. I normalize every embedding to length 1, so cosine becomes just the dot product. Chroma is configured with cosine distance, which is one minus the similarity, and I convert it back to a similarity. A value near 1 means very similar meaning."
*Next follow-up:* "Why cosine and not Euclidean distance?" (For unit-length vectors both give the same ranking, since squared Euclidean distance equals 2 minus 2 times cosine.)

**Q19. How do you handle LLM errors and rate limits?**
*Model answer:* "For OpenAI I retry on rate-limit, timeout, connection and 5xx errors with a short backoff. For Gemini I try the configured model, then fallback models, and move on when a model returns 404. I also have a global limiter of five LLM calls per minute to protect a free-tier key. When something still fails, I catch it and return the error text inside the answer with status 200. That's convenient for the chat UI, but a proper 502 or 429 would be better."
*Next follow-up:* "Why is the limiter global?" (It was meant to protect one shared API key. With real users I'd make it per user.)

**Q20. How does the frontend track ingestion progress, and why not WebSockets?**
*Model answer:* "After the 202, the React page polls `GET /status/{job_id}` once a second inside a `useEffect` and clears the timer when the job completes or fails. The backend writes progress at fixed steps: 25, 50, 60, 80 and 100. Polling is simple and needs no extra infrastructure. The downside is request volume: one poll per second sits right at my 60-requests-per-minute rate limit, and on any error the UI stops polling."
*Next follow-up:* "How would you fix that?" (Poll every 2–3 seconds, exempt the status route from the limiter, and keep polling on a 429.)

---

## Tier 3: Deep

**Q21. [MUST-ASK] What is reciprocal rank fusion, what did you change, and why k = 60?**
*Model answer:* "RRF merges several ranked lists by giving each item `1 / (k + rank)` from each list and adding the scores, so items that rank well in multiple lists rise. I use k = 60, the common default from the original paper; my code comment cites it, and I didn't tune it. What I changed is that I only have one ranked list, the vector search, so I use the same rank score and multiply it by a boost between 1 and 2 for chunks on pages of the sections the question seems to target. The boost depends on how well the question matches the section types and on a confidence value, so it's not fixed. I'd call it a structure-aware re-ranking rather than a true fusion, and I haven't proven it improves retrieval."
*Next follow-up:* "Why not do a real RRF with BM25?" (That would fuse two real lists, vector and keyword, and would help with exact terms like names and formulas. It's a good next step.)

**Q22. Explain your ATB-RRF end to end with an example.**
*Model answer:* "Take the question 'How does the proposed method work?' with top-k 3. I fetch six candidates by vector search and navigate the section tree. Say only Methodology, pages 4 to 6, matches. Two of the top three vector hits are on those pages, so confidence is 2/3, and the boost works out to 1.51. A chunk at vector rank 2 on page 4 scores 1.51 over 62, about 0.0244, which beats the rank-1 chunk on another page at 1 over 61, about 0.0164. So the final three are the boosted ones, and the rank-1 chunk drops out. Then a small bonus re-ranks chunks of the top sections first."
*Next follow-up:* "What if the section guess is wrong?" (The confidence rule shrinks the boost when the vector hits don't agree, but a wrong guess with partial agreement can still push bad chunks up. I haven't measured that.)

**Q23. [MUST-ASK] How do you evaluate your RAG system?**
*Model answer:* "Right now I don't have a quantitative evaluation, and I'd rather say that than make up numbers. I do log retrieval and LLM latency, and I tested the flow by hand and with a few tests. The evaluation I'd build is a gold set of 30 to 50 questions over several papers, each with the correct page. Then I'd compare recall at 5 and MRR for plain vector search versus my boosted ranking, and score answers for groundedness. That would also tell me whether the section layer is worth keeping."
*Next follow-up:* "How would you avoid bias in the question set?" (Write questions without looking at the chunks, use several papers and question types, and have someone else label the answers.)

**Q24. How is the section tree built, and what happens if a PDF has no headings?**
*Model answer:* "If a real LLM provider is configured, I first ask it for a table of contents from the first 20 pages. Otherwise, or if that fails, I use regular-expression rules for numbered headings, all-caps lines and short title-case lines. Each heading becomes a node with a page range and one of 13 section types chosen by keyword matching. If no headings are found, the whole paper becomes one 'Document' node, so the structural layer has nothing useful to boost and retrieval is effectively plain vector order. The tree is flat today, and title or author lines can be mistaken for headings."
*Next follow-up:* "How would you make it hierarchical?" (Use heading numbering such as 3.1 under 3, or PDF outlines and font sizes, and fill the `children` field.)

**Q25. [MUST-ASK] What happens with a scanned PDF? How would you scale to many users and many documents?**
*Model answer:* "A scanned PDF has no text layer, so pypdf returns nothing and my code tries the unstructured fallback. I haven't installed OCR tools, so I expect the job to end as failed with 'no readable text'. I'd add detection and an OCR step with Tesseract. To scale, I'd stop doing ingestion inside the web process: use a queue like Celery with Redis and separate workers, PostgreSQL for metadata, a shared vector database such as a Chroma server, pgvector or Qdrant, and object storage like S3 for PDFs. I'd also add authentication with per-user documents, per-user rate limits, and caching or batching for embeddings."
*Next follow-up:* "What would break first?" (The single worker with in-process jobs and CPU embedding, then local disk storage, then the global LLM limit.)

**Q26. What happens if the server restarts during ingestion, or if you run more than one worker?**
*Model answer:* "Job status is in SQLite, so rows survive a restart, but the background task is lost. The job stays 'processing' forever because nothing repairs old jobs at startup, and the saved PDF stays on disk. With more than one worker, any worker can read status from the shared file, but each has its own embedding cache, model copy, and rate-limit counters, and embedded Chroma isn't safe across processes. That's why I run one worker. A queue plus startup recovery would solve most of it."
*Next follow-up:* "How would you make ingestion retryable?" (Store a stage checkpoint, make each stage idempotent, and clean up partial vectors if a later stage fails.)

**Q27. What are the trade-offs in chunking, and how do tables and equations behave?**
*Model answer:* "Smaller chunks give sharper search but less context; bigger chunks keep context but blur the match and cost more prompt tokens. I split per page, so page citations are exact, but a sentence that crosses a page break is cut. I only extract plain text, so tables become lines of cells that may be split across chunks and lose their header row, and equations come out as broken symbols. I tested a synthetic 120-row table and it split into two chunks at a row boundary. A table-aware parser would be the fix."
*Next follow-up:* "How would you handle tables specifically?" (Detect them, keep each table as one chunk with its header and caption, or convert it to Markdown.)

**Q28. How would you add true multi-turn conversation?**
*Model answer:* "I'd add a `history` list to the request with the last few user and assistant messages. The simplest change is to include that history in the prompt. Better, I'd first ask the LLM to rewrite the latest question into a standalone query using the history, then run retrieval on the rewritten query, so 'what about its limitations?' becomes a full question. I'd cap history by token budget. The files to change are `schemas.py`, `qa.py`, `rag.py`, `api.ts` and `ChatPage.tsx`, plus tests."
*Next follow-up:* "How would you test it?" (Scripted two-turn conversations with an expected rewritten query, using the mock provider for the pipeline and a stubbed rewriter.)

**Q29. What is `retrieval_confidence` and how reliable is it?**
*Model answer:* "It is the fraction of the top vector hits whose page falls inside the pages of the sections my tree navigation picked. So it measures agreement between two heuristics, not correctness. If the tree is a single 'Document' node, it is trivially 1.0. It is also used to shrink the boost when it's below 0.3. It isn't a calibrated probability, and my UI doesn't even use it; the UI computes its own badge from the average source score, which is misleading in boosted mode."
*Next follow-up:* "How would you calibrate a real confidence?" (Compare predicted confidence with correctness on a labelled test set, then calibrate or threshold; or add an 'answer not found' check.)

**Q30. What are the security risks of your project?**
*Model answer:* "There's no authentication, so anyone who can reach the API can list and delete any document. Upload validation is weak: no size limit, and a wrongly named file with a PDF content type is accepted. Error messages from the LLM and from ingestion are shown to users. Also the retrieved text goes straight into the prompt, so a malicious PDF could contain instructions aimed at the model, which is prompt injection; my app has no tools to abuse, which limits the damage, but the answer could be manipulated. I'd add auth, per-user ownership, a file signature and size check, and prompt delimiters plus output filtering."
*Next follow-up:* "Does CORS protect your API?" (Only browsers. It doesn't stop curl or other servers, so it isn't authentication.)

---

## Tier 4: Tricky / trap

**Q31. Your README says "production-ready". Is it?**
*Model answer:* "No, and I'd change that wording. It works end to end, and has Docker and CI. But there's no authentication, state lives in local SQLite and Chroma folders, ingestion runs in the web process and can strand a job on restart, the tests don't cover the ranking logic, and I have no evaluation. It's a solid demo, not a production service. For production I'd fix those in order: persistence, queue, auth, tests, evaluation."
*Next follow-up:* "Which would you do first?" (Evaluation, because without it I can't tell if the retrieval design is any good; then persistence.)

**Q32. Your documents call ATB-RRF "patent-pending" and "novel". Is it?**
*Model answer:* "I can't support that. I haven't filed a patent, and I haven't done a formal prior-art search. The method itself is a rank-based score multiplied by a boost chosen from section guesses. I haven't shown it works better, so I'd call it an experimental idea, not a novel or protected method. The 'patent-pending' wording in the documentation is wrong and I should remove it."
*Next follow-up:* "Then why is it in the document?" (It was written as a pitch. I now treat only what I can show in code or measurements as a claim.)

**Q33. The UI shows "2%" for my best source. Is retrieval that bad?**
*Model answer:* "No, that's a display bug. In boosted mode the score I return is a fused rank score, `1/(60+rank)` times the boost, which can't exceed about 0.033. The UI multiplies it by 100 and shows a percentage, so it looks like 2 or 3 percent. The confidence badge has the same problem, because its thresholds assume cosine similarity from 0 to 1, so it shows 'Low'. The fix is to return the cosine similarity as a separate field and show the backend's own confidence value."
*Next follow-up:* "Why didn't you catch it?" (I had no test or evaluation touching that path. It's exactly the kind of bug that ranking tests would catch.)

**Q34. Is your "hybrid" retrieval BM25 plus vector search?**
*Model answer:* "No. 'Hybrid' in my code means vector search combined with a section-tree boost. I don't have any keyword or BM25 search. A true hybrid with BM25 would help for exact terms such as author names, dataset names and formula symbols, and it would give real two-list reciprocal rank fusion. I'd like to add that."
*Next follow-up:* "Would BM25 help on academic papers?" (Likely for exact terms; but I'd measure on a gold set before claiming it.)

**Q35. Show me the depth-weighted scoring in action.**
*Model answer:* "The code has it: navigation multiplies a node's score by `1 + 0.15 × depth`, and the boost formula adds `0.15 × depth`. But my tree builder creates only a flat list, and the `children` field is never filled, so depth is always zero and that part never takes effect. The nested tree in my architecture diagram is what I intended, not what the code produces today. To make it real, I'd nest headings by their numbering, so 3.1 sits under 3."
*Next follow-up:* "So is the 'hierarchical tree' claim false?" (For the current code, yes. I'd call it a flat section list until nesting is built.)

**Q36. Is the live demo working right now?**
*Model answer:* "[Say the truth after you check.] The frontend is on Vercel. My Railway backend has been reported as not running, so I wouldn't count on it. I can run the whole thing locally with Docker Compose. With a real API key it gives real answers; without one, it uses the mock provider, which returns fixed text, but the upload, chunking, embedding and retrieval all run for real. Free hosting plans often lose their disk on redeploy, so the proper fix is a persistent volume or managed storage."
*Next follow-up:* "Can you show me now?" (Yes, locally, if you prepared it. Prepare it the day before and keep a screen recording as backup.)

**Q37. How much did StructRAG improve accuracy over plain RAG?**
*Model answer:* "I don't have a number, because I haven't measured it. I'd rather tell you that than guess. The boost is a hypothesis that questions aimed at a section, like 'what were the results', should favour chunks from that section. To test it, I'd build a gold set and compare recall at 5 and MRR with and without the boost, which is easy in my code because the tree can be switched off by passing no tree nodes."
*Next follow-up:* "Then how do you know it isn't making things worse?" (I don't yet. The confidence rule is meant to limit damage, but that is also untested.)

**Q38. Which part of your code are you least proud of?**
*Model answer:* "Three things. First, the silent hash-embedding fallback: it keeps the app running if the model fails to load, but it gives meaningless retrieval without any error. Second, returning LLM failures as normal 200 responses with the error text in the answer. Third, my ranking layer: some of what my documents describe, like the depth weighting, isn't actually active, and I haven't evaluated it. I know how to fix each of them."
*Next follow-up:* "How would you fix the fallback?" (Fail loudly in production, and store the embedding model name with each document so a mismatch can be detected.)

**Q39. If the embedding model fails to load, what happens, and isn't silent fallback dangerous?**
*Model answer:* "The service logs a warning and switches to a deterministic SHA-256 hash embedding for the rest of the process. The pipeline keeps working, so uploads succeed and questions get answered. But hash vectors aren't semantic, so retrieval is almost arbitrary; in a test, two unrelated strings had a cosine of 0.71. And if some documents were indexed under the fallback and others under the real model, queries on the first group would return junk. So yes, it's dangerous in production. It should raise an error, or at least flag the document."
*Next follow-up:* "How would you detect a mixed index?" (Save the model name and version in each chunk's metadata and compare it at query time.)

**Q40. A user asks six questions in a minute and the sixth returns an error message with status 200. Is that a bug?**
*Model answer:* "It's by design but poorly designed. I have a global limit of five LLM calls per minute, set to protect a free-tier API key. The sixth call raises an error, which my code catches and returns as the answer text with status 200. It applies to everyone together, and it even applies when the provider is the mock. The proper design is a per-user limiter returning 429 with a Retry-After header."
*Next follow-up:* "How would you rate limit properly?" (A token bucket per user or API key, stored in Redis so it works across workers.)

---

# Part 10. Code-reading drill

> **In simple words**
> Interviewers often paste a few lines of your code and ask "what does this do, and what is wrong with it?". Here are 10 real excerpts from your repo. For each: read the code, answer aloud, then check the answer. After that come 5 "how would you change X" tasks that name the files.
> **How to use it:** cover everything under "Answer", write or say your answer, then compare. Spend 2 minutes per excerpt.

---

## 10.1 Ten excerpts

### Excerpt 1: the hash embedding [backend/app/services/embeddings.py: _hash_embedding()]

```python
def _hash_embedding(text: str, dimension: int = 384) -> list[float]:
    """Deterministic fallback embedding used when model init fails."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [(digest[i % len(digest)] / 255.0) for i in range(dimension)]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]
```

**Your turn:** What does this return? Is it a useful embedding? Why `or 1.0`?

**Answer:**
- It hashes the text with SHA-256, which gives 32 bytes. `digest[i % 32]` repeats those 32 bytes across 384 slots, and dividing by 255 scales each to 0–1. Then it divides by the vector's length, so the vector has length 1, like a real normalized embedding.
- The same text always gives the same vector (deterministic), and the size is 384, so it fits Chroma. But it has no meaning. **FACT (ran it):** the vector has only 32 distinct values, and two unrelated strings had a cosine of 0.71. All values are non-negative, so every pair looks fairly similar.
- `or 1.0` avoids dividing by zero if the norm were 0. With SHA-256 bytes it practically never is; it is a safety net.
- Takeaway: it keeps the pipeline running, but retrieval with it is close to arbitrary. It should not be silent in production.

### Excerpt 2: converting Chroma distance to similarity [backend/app/services/vectorstore.py: search()]

```python
result = collection.query(
    query_embeddings=[query_embedding],
    n_results=top_k,
    where={"doc_id": doc_id},
)
...
for idx, chunk_id in enumerate(ids):
    distance = float(distances[idx]) if idx < len(distances) else 1.0
    similarity = max(0.0, 1.0 - distance)
```

**Your turn:** What does `where` do? What is `distance` here? Why `max(0.0, ...)`?

**Answer:**
- `where={"doc_id": doc_id}` filters the search to chunks whose metadata `doc_id` matches, so one document's questions never return another document's chunks.
- The collection was created with `hnsw:space = cosine`, so Chroma's distance is 1 minus cosine similarity. The code turns it back into a similarity: `1 - distance`.
- `max(0.0, ...)` stops a negative similarity (distance above 1). The default of `1.0` when a distance is missing makes similarity 0.
- Why `[0]` isn't shown: the omitted lines take `result[...][0]`, because Chroma returns one list per query embedding.
- Note: this score is used directly only when the tree boost is not applied. Otherwise it is replaced by a fused rank score (Excerpt 8).

### Excerpt 3: mapping pages to tree nodes [backend/app/services/chunker.py: chunk_pages()]

```python
page_to_section_type: dict[int, str] = {}
page_to_node_id: dict[int, str] = {}
if tree_nodes:
    for node in tree_nodes:
        for pg in range(node.start_page, node.end_page + 1):
            page_to_section_type[pg] = node.section_type
            page_to_node_id[pg] = node.node_id
```

**Your turn:** What does this build? What happens if two headings are on the same page?

**Answer:**
- It builds two lookup dictionaries from page number to the section type and node id. Later, each chunk looks up its page to get `section_type` and `tree_node_id` for Chroma metadata.
- If two nodes cover the same page, the loop writes the same dictionary key twice and the **later node wins**. **FACT (ran it):** in a synthetic page 1 with paper title, author, Abstract and Introduction, four nodes covered page 1, so every chunk on that page would be labelled with the last one (introduction).
- So a chunk's section label is only accurate at page level, not at chunk level. Fix: map by character offsets, or split pages at headings before chunking.

### Excerpt 4: dynamic SQL in `update_job` [backend/app/services/jobs.py: update_job()]

```python
allowed = {"status", "progress", "pages_parsed", "total_pages", "chunks_created", "doc_id", "error"}
fields: list[str] = []
values: list[Any] = []

for key, value in updates.items():
    if key in allowed:
        fields.append(f"{key} = ?")
        values.append(value)
...
with self._lock:
    self._conn.execute(
        f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?",
        tuple(values),
    )
    self._conn.commit()
```

**Your turn:** The SQL string is built with an f-string. Is that SQL injection? What is `self._lock` for?

**Answer:**
- It is safe here. Only the **column names** come from the f-string, and they pass through the `allowed` whitelist, so a caller can't inject a name. All **values** use `?` placeholders and are bound by SQLite, so values can't change the SQL.
- Placeholders can't be used for column names, which is why the whitelist exists.
- `self._lock` is a `threading.Lock`. The SQLite connection is shared across threads (`check_same_thread=False`), and FastAPI runs sync endpoints and background tasks in a thread pool. The lock makes one thread use the connection at a time.
- Java parallel: like a `synchronized` block around a shared `Connection`.
- Caution: `TreeStore` opens a second connection to the same file with its own lock (INFERENCE: some contention under heavy writes).

### Excerpt 5: the LLM rate limiter [backend/app/services/rag.py: _enforce_llm_rate_limit()]

```python
now = time.time()
with _llm_window_lock:
    while _llm_window and now - _llm_window[0] > 60:
        _llm_window.popleft()

    if len(_llm_window) >= max_requests:
        raise RuntimeError(
            f"LLM rate limit reached ({max_requests}/min). Please wait a minute and retry."
        )

    _llm_window.append(now)
```

**Your turn:** What algorithm is this? Who is limited? What happens to the user on the 6th call in a minute?

**Answer:**
- It is a **sliding-window log**: a deque of timestamps. It drops entries older than 60 seconds, and if 5 or more remain, it raises an error. Otherwise it records the call.
- It limits **all users together**: one module-level deque in one process. It is not per user, and not shared across workers or restarts.
- **FACT (ran it):** the 6th call raises `RuntimeError`, which `answer_question()` catches and returns as the answer text with **HTTP 200** ("The language model request failed ... Provider error: ..."). It applies to the mock provider too.
- Better design: a per-user token bucket in Redis, returning 429 with `Retry-After`.

### Excerpt 6: the per-IP rate limiter [backend/app/main.py: RateLimitMiddleware.dispatch()]

```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory fixed-window rate limiter for local dev safety."""
    ...
    async def dispatch(self, request: Request, call_next):
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
```

**Your turn:** The docstring says "fixed-window". Is it? Name three weaknesses.

**Answer:**
- It is **not** a fixed window. It keeps a timestamp per request and drops those older than 60 s, so it is a **sliding-window log**. The docstring is wrong.
- Weaknesses:
  1. **Proxy IP (INFERENCE):** it uses `request.client.host` and ignores `X-Forwarded-For`. Behind a hosting proxy, many users may look like one IP and share one bucket.
  2. **In memory, per process:** not shared across workers, lost on restart. Keys for IPs that never return are never removed, so memory grows slowly (INFERENCE).
  3. **429 has no CORS header (FACT, ran it):** the limiter is the outermost middleware, so its response never passes through CORS. A cross-origin browser sees a network error, not a 429.
- Also: preflight `OPTIONS` requests count toward the limit, and the UI's one-per-second status polling reaches 60 per minute (the 60th poll is refused in a simulation).

### Excerpt 7: the adaptive boost [backend/app/services/hybrid_retriever.py: _adaptive_boost_factor()]

```python
affinity_norm = min(affinity_score / 5.0, 1.0) if affinity_score > 0 else 0.0

raw_boost = (
    BASE_BOOST
    + affinity_norm * (MAX_BOOST_FACTOR - BASE_BOOST)
    + DEPTH_BONUS_PER_LEVEL * node_depth
)
clamped = max(MIN_BOOST_FACTOR, min(raw_boost, MAX_BOOST_FACTOR))

if confidence < CONFIDENCE_ATTENUATION_THRESHOLD:
    attenuation = confidence / CONFIDENCE_ATTENUATION_THRESHOLD
    return 1.0 + (clamped - 1.0) * attenuation

return clamped
```

Constants: `BASE_BOOST = 1.3`, `MAX_BOOST_FACTOR = 2.0`, `MIN_BOOST_FACTOR = 1.0`, `DEPTH_BONUS_PER_LEVEL = 0.15`, `CONFIDENCE_ATTENUATION_THRESHOLD = 0.3`.

**Your turn:** Compute the result for `affinity_score = 3.0`, `confidence = 0.15`, `node_depth = 0`. Then say what happens at confidence 0.

**Answer (step by step):**
1. `affinity_norm = min(3.0 / 5.0, 1.0) = 0.6`.
2. `raw_boost = 1.3 + 0.6 × (2.0 − 1.3) + 0.15 × 0 = 1.3 + 0.42 = 1.72`.
3. `clamped = max(1.0, min(1.72, 2.0)) = 1.72`.
4. `confidence 0.15 < 0.3`, so `attenuation = 0.15 / 0.3 = 0.5`.
5. Return `1.0 + (1.72 − 1.0) × 0.5 = 1.0 + 0.36 = 1.36`. (**FACT (ran it):** the function returned 1.36.)
- At confidence 0, attenuation is 0 and the boost is exactly 1.0, so nothing is promoted. That also means that when none of the vector top-k lie on tree pages, a relevant chunk just below the top-k can never be rescued, which limits the point of the larger candidate pool.
- At affinity 5 or more, the boost reaches 2.0. Depth never adds anything because the tree is flat.

### Excerpt 8: the boosted rank score [backend/app/services/hybrid_retriever.py: _tree_boosted_rrf()]

```python
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
```

**Your turn:** What is `RRF_K`? Why `scores.get(...) + fused` if each chunk appears once? What does the returned `score` mean now?

**Answer:**
- `RRF_K = 60`. `base = 1 / (60 + rank)`, so rank 1 gives 0.016393 and rank 10 gives 0.014286.
- `scores.get(id, 0.0) + fused` is the accumulate step of a multi-list RRF. Here there is only **one** list, so each chunk id occurs once and the line just stores `fused`. It is a leftover from the textbook form, and it shows that this is not a true fusion.
- `dataclasses.replace(..., score=...)` returns a copy of the `SearchResult` with `score` changed. So the `score` the API sends is now the fused rank score (at most 2.0/61 = 0.0328), **not** cosine similarity. That is what makes the UI show "2%".
- The final sort is by score, highest first, keeping `top_k`.

### Excerpt 9: the UI confidence badge [frontend/src/pages/ChatPage.tsx]

```tsx
const confidence = useMemo(() => {
  const latest = [...messages].reverse().find((message) => message.role === 'assistant' && message.sources?.length);
  if (!latest?.sources?.length) {
    return null;
  }
  const avg = latest.sources.reduce((acc, source) => acc + source.score, 0) / latest.sources.length;
  if (avg > 0.75) return 'High';
  if (avg > 0.45) return 'Medium';
  return 'Low';
}, [messages]);
```

**Your turn:** What does this compute? What is wrong when the backend used the tree boost? What does the label in the UI claim?

**Answer:**
- `useMemo` recomputes only when `messages` changes. It copies and reverses the list to find the **latest assistant message that has sources**, averages their `score`, and maps the average to High, Medium or Low.
- The thresholds assume `score` is a cosine similarity from 0 to 1. In boosted mode `score` is at most 0.0328, so the average can never pass 0.45. The badge shows **Low** for essentially every boosted answer.
- The UI text under it says "Cross-signal agreement between the section-tree and vector retrieval", which is what the backend's `retrieval_confidence` measures. But this code ignores that field (the TypeScript `AskResponse` type doesn't even include it).
- Fix: add `retrieval_confidence` to the TS type and show it, and return a separate cosine `similarity` for each source.

### Excerpt 10: the upload type check [backend/app/api/v1/upload.py: upload_pdf()]

```python
suffix = Path(file.filename or "").suffix.lower()
if file.content_type not in {"application/pdf", "application/octet-stream"} and suffix != ".pdf":
    raise HTTPException(status_code=400, detail="Only PDF files are supported")
```

**Your turn:** Which of these are accepted? (a) `paper.pdf` with type `text/plain`; (b) `notes.txt` with type `text/plain`; (c) `virus.exe` with type `application/pdf`; (d) `data.bin` with `application/octet-stream`.

**Answer:**
- The error is raised only when the content type is **not** PDF/octet-stream **and** the suffix is **not** `.pdf`. So a file is accepted if **either** looks right.
- (a) accepted (suffix is `.pdf`). (b) rejected. (c) accepted (content type). (d) accepted. **FACT (ran it):** (b) → 400; (c) and (d) → 202.
- Accepted non-PDFs fail later in the background job, and the job ends `failed` with an error message. There is also no size limit.
- Better: require both checks (use `or` for rejection), check the file starts with `%PDF`, and cap the size.

---

## 10.2 Five "how would you add or change X" tasks

For each: the idea, then the files that change.

### Task 1. Make the chat truly multi-turn

- **Idea:** send the last N messages; rewrite the latest question into a standalone query using them; then retrieve.
- **Files:**
  - `backend/app/models/schemas.py`: add `history: list[HistoryMessage]` to `AskRequest` (limit its length).
  - `backend/app/api/v1/qa.py`: pass `history` through.
  - `backend/app/services/rag.py`: a `rewrite_question(history, question)` step before `hybrid_retrieve()`, and put recent turns into the prompt within a size limit.
  - `frontend/src/services/api.ts`: add `history` to the `AskRequest` type.
  - `frontend/src/pages/ChatPage.tsx`: send the last few `messages` in `handleSend()`.
  - `backend/app/tests/test_ask.py`: a two-turn test with the LLM patched.

### Task 2. Add a retrieval evaluation (recall@k and MRR)

- **Idea:** a gold file of questions with the correct page, run for the plain vector path and the boosted path, and print the metrics.
- **Files:**
  - New `backend/eval/gold_questions.json`: question, paper, correct pages.
  - New `backend/eval/run_eval.py`: ingest the papers, call `hybrid_retrieve(query, doc_id, top_k, tree_nodes)` once with the tree and once with `tree_nodes=None`, compute recall@5 and MRR.
  - New `backend/app/tests/test_eval_metrics.py`: unit tests for the metric functions.
  - `README.md`: how to run it. Optionally an extra job in `.github/workflows/ci.yml`.
  - Because the code can switch the tree off by passing `None`, no change to retrieval is needed.

### Task 3. Fix the score display and confidence badge

- **Idea:** keep cosine similarity separate from the fused ranking score.
- **Files:**
  - `backend/app/services/hybrid_retriever.py`: keep the original cosine value in a new field (for example `similarity`) when `dataclasses.replace()` changes `score`.
  - `backend/app/models/schemas.py`: add `similarity` to `SourceChunk`.
  - `backend/app/services/rag.py`: copy it into the `SourceChunk` list.
  - `frontend/src/services/api.ts`: add `similarity` to `SourceChunk` and the four StructRAG fields to `AskResponse`.
  - `frontend/src/components/SourceHighlight.tsx`: show the similarity.
  - `frontend/src/pages/ChatPage.tsx`: show `retrieval_confidence` from the response instead of averaging scores.
  - Tests: a backend test that the similarity is in 0–1, and a Vitest test for the badge.

### Task 4. Recover stuck jobs and avoid orphaned data

- **Idea:** on startup, mark old `processing` jobs as failed; on ingestion failure, remove partial vectors.
- **Files:**
  - `backend/app/services/jobs.py`: add `fail_stale_jobs()` that sets `status='failed'` and an explanatory `error` for rows still `processing`.
  - `backend/app/main.py`: call it at startup.
  - `backend/app/api/v1/upload.py`: in the `except` block, call `vectorstore_service.delete_doc(doc_id)` and `tree_store.delete_tree(doc_id)` if they were already written, and delete the PDF.
  - `backend/railway.json` / platform settings: attach a persistent volume for `data/` and `chromadb_data`.
  - Later step: replace `BackgroundTasks` with a Celery or RQ worker.

### Task 5. Support scanned PDFs with OCR

- **Idea:** detect pages without text and run OCR.
- **Files:**
  - `backend/app/services/loader.py`: when pypdf yields empty pages, run an OCR function (for example Tesseract through `ocrmypdf` or `pytesseract` on rendered page images) and return `PageText` objects.
  - `backend/requirements.txt`: add the OCR Python packages.
  - `backend/Dockerfile`: `apt-get install` the Tesseract and PDF rendering system packages.
  - `backend/app/tests/test_upload.py`: add a small scanned-PDF fixture and a test.
  - A clear job error when OCR also finds nothing.

**Bonus ideas if asked for more:**
- A real nested tree: fill `children` in `tree_index.py: _build_nodes_from_headings()` by heading numbering, and map chunks to the deepest node in `chunker.py`.
- A true two-list fusion: add a keyword (BM25) search next to `vectorstore.py` and feed both ranked lists into the same `1/(k+rank)` sum.

---

# Part 11. Self-test and flash cards

> **In simple words**
> Test yourself without looking at the answers. Questions 1–15 are multiple choice. Questions 16–25 are short answer, some with small calculations. Answers come after all the questions, each with a one-line reason. Then 30 flash cards. Score yourself: 22 or more correct means you are ready for the project round.

## 11.1 Self-test: questions

**Multiple choice**

**1.** What does the HTTP status **202** mean on `POST /api/v1/upload`?
A) The document is fully indexed  B) The request was accepted and is still being processed  C) A new resource was created and is ready  D) The file type is wrong

**2.** What are the default chunk settings in the code?
A) 800 / 200 characters  B) 3000 / 800 characters  C) 3000 / 800 tokens  D) 1000 / 100 characters

**3.** How many numbers are in each MiniLM embedding in this project?
A) 128  B) 256  C) 384  D) 768

**4.** Which statement about conversation memory is true for this project?
A) The last 5 messages are sent to the LLM  B) The backend stores chat history in SQLite  C) Each question is sent alone, with no history  D) History is sent only in "explain" mode

**5.** What is the default value of `llm_provider` in the code?
A) openai  B) gemini  C) hf  D) mock

**6.** Where is ingestion job status stored?
A) A Python dictionary in memory  B) The SQLite `jobs` table  C) Redis  D) ChromaDB metadata

**7.** What does `where={"doc_id": doc_id}` do in the Chroma query?
A) Deletes the document  B) Restricts the search to chunks of one document  C) Sorts by document id  D) Chooses the embedding model

**8.** In this repo, what does "hybrid" retrieval mean?
A) Vector search + BM25 keyword search  B) Vector search + a section-tree boost  C) Two LLMs voting  D) Chroma + SQLite full-text search

**9.** How many section types are in the ontology (including `other`)?
A) 5  B) 8  C) 13  D) 20

**10.** Which middleware is the outermost (runs first) in `main.py`?
A) CORSMiddleware  B) RateLimitMiddleware  C) Both run in parallel  D) Neither: routers run first

**11.** Does the repo contain retrieval-evaluation code (recall@k, MRR)?
A) Yes, in `backend/eval`  B) Yes, in the tests  C) Only for the frontend  D) No

**12.** If you upload a PDF without giving a title, what is the document's title?
A) The original file name  B) "Untitled"  C) The first heading in the PDF  D) The job id (a UUID)

**13.** What is the purpose of `"rewrites": [{"source": "/(.*)", "destination": "/index.html"}]` in `vercel.json`?
A) Proxy `/api` to the backend  B) Make deep links such as `/chat/<id>` load the React app  C) Force HTTPS  D) Cache images

**14.** What does CORS protect against?
A) A browser page on one site calling an API on another site without permission  B) SQL injection  C) Any HTTP client calling your API  D) Large uploads

**15.** Which of these does the repo have a test for?
A) The section-tree builder  B) The ATB-RRF ranking  C) The chunk overlap and page metadata  D) The Gemini model fallback

**Short answer (with calculations)**

**16.** With k = 60, an item is ranked 2nd in list 1 and 1st in list 2. What is its RRF score?

**17.** Using `_adaptive_boost_factor()`, what is the boost for `affinity_score = 5.0`, `confidence = 1.0`, `node_depth = 0`?

**18.** Same function, `affinity_score = 3.0`, `confidence = 0.1`, `node_depth = 0`. What is the boost?

**19.** Using the code comment's ratio (3000 characters ≈ 800 tokens), about how many tokens do the retrieved chunks take when `top_k = 7`?

**20.** Why doesn't the 800-character overlap help when a sentence crosses a page break?

**21.** Give two reasons why the UI can show about 2% for the best source.

**22.** What happens to a job in the `processing` state if the server restarts?

**23.** List the five ingestion stages with their progress percentages.

**24.** What reason does the repo give for running gunicorn with `--workers 1`?

**25.** What does the mock provider return, and which test relies on it?

---

## 11.2 Answers (each with a one-line reason)

**1. B.** 202 Accepted means the work is queued, not finished [upload.py: status_code=202].

**2. B.** `DEFAULT_CHUNK_SIZE = 3000`, `DEFAULT_CHUNK_OVERLAP = 800`, counted in characters; the comment says this approximates 800 / 200 tokens [chunker.py].

**3. C.** MiniLM gives 384 dimensions; the hash fallback also uses `dimension=384` [embeddings.py; PROJECT_DOCUMENTATION.md §2].

**4. C.** `AskRequest` has no history field, and `ChatPage` sends only the new question [schemas.py, ChatPage.tsx].

**5. D.** `llm_provider: Literal[...] = "mock"` [config.py]. (`.env.example` says openai, but the code default is mock.)

**6. B.** `JobRepository` writes the `jobs` table in `metadata.db` [jobs.py].

**7. B.** The `where` filter matches the `doc_id` metadata, so searches never mix documents [vectorstore.py: search()].

**8. B.** The retriever combines vector search with a section-tree boost; there is no BM25 or keyword search anywhere [hybrid_retriever.py].

**9. C.** 13 types including `other` [tree_index.py: SectionType].

**10. B.** `RateLimitMiddleware` is added after CORS, and the last one added is outermost; I confirmed the order `['RateLimitMiddleware', 'CORSMiddleware']` (FACT, ran it) [main.py].

**11. D.** I searched for recall, MRR, groundedness and evaluation; nothing exists (Part 7, S2).

**12. D.** The file is saved as `<job_id>.pdf` and the default title is `Path(file_path).stem`, which is the job id (FACT, ran it) [upload.py].

**13. B.** It sends every URL to `index.html`, so refreshing a client-side route works in a single-page app [vercel.json].

**14. A.** CORS is a browser rule; `curl` and other servers ignore it, so it isn't authentication [main.py].

**15. C.** `test_chunking_creates_overlap_and_page_metadata` exists; the tree, ranking and provider fallback have no tests [test_upload.py].

**16. 0.032522.** 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.032522.

**17. 2.0.** `affinity_norm = min(5/5, 1) = 1`; raw = 1.3 + 1 × 0.7 = 2.0; confidence 1.0 ≥ 0.3, so no attenuation.

**18. 1.24.** `affinity_norm = 3/5 = 0.6`; raw = 1.3 + 0.6 × 0.7 = 1.72; confidence 0.1 < 0.3, so attenuation = 0.1/0.3 = 0.3333; result = 1 + (1.72 − 1) × 0.3333 = 1 + 0.24 = 1.24.

**19. About 5,600 tokens.** 7 × 800 = 5,600, using the comment's estimate; the system prompt, question and answer are extra.

**20. Chunks are made page by page.** Each page is split separately, so overlap exists only between chunks inside the same page [chunker.py: chunk_pages()].

**21. Any two of:**
- In boosted mode, `score` is a fused rank score (at most 2.0/61 = 0.0328), not a cosine similarity.
- `SourceHighlight` shows `round(score × 100)%`.
- The badge thresholds (0.75 and 0.45) assume cosine similarity, so it shows "Low".
- The UI ignores the backend's `retrieval_confidence`.

**22. It stays `processing` forever.** The background task dies with the process and no startup code repairs old rows (Part 7, S5).

**23. Parse (25%), build tree (50%), chunk (60%), embed (80%), store vectors, tree and document row, then completed (100%)** [upload.py: process_document_job()].

**24. None.** The reason is NOT IN THE MATERIAL; commit `c4bb7e0` only changed 2 workers to 1 and the timeout from 120 s to 180 s.

**25. A fixed text template citing the first 3 chunks (or "I don't know" if there are none).** The mock ignores the question. **No test** uses it directly: `test_ask.py` patches `_generate_answer`. The mock only runs when `LLM_PROVIDER` is `mock`.

**Score:** ___ / 25.

---

## 11.3 Thirty flash cards

| # | Term | One-line meaning |
|---|---|---|
| 1 | **RAG** | Retrieve relevant text first, then let the LLM answer from it |
| 2 | **Embedding** | A list of 384 numbers standing for the meaning of a text |
| 3 | **Cosine similarity** | Direction match of two vectors; with unit vectors it is the dot product |
| 4 | **Chunk** | A piece of page text, up to 3000 characters |
| 5 | **Overlap** | 800 characters repeated between neighbouring chunks (within a page only) |
| 6 | **Vector database** | Stores embeddings and finds the nearest ones fast |
| 7 | **ChromaDB** | The vector store used here; embedded or server mode; metadata filter |
| 8 | **HNSW (`hnsw:space`)** | The graph-based approximate index that Chroma's config key refers to (general knowledge) |
| 9 | **top_k** | Number of chunks sent to the LLM; 1 to 10, default 5 |
| 10 | **Candidate pool** | `2 × top_k` vector hits fetched before ranking |
| 11 | **RRF** | Score = sum of `1/(k + rank)` over ranked lists |
| 12 | **k = 60** | The usual RRF smoothing constant; not tuned here |
| 13 | **ATB-RRF** | My rank score `1/(60+rank)` times a boost of 1.0–2.0 for chunks on chosen sections |
| 14 | **Boost formula** | `1.3 + min(affinity/5,1) × 0.7 + 0.15 × depth`, clamped to 1–2, shrunk if confidence < 0.3 |
| 15 | **Retrieval confidence** | Share of the vector top-k lying on the tree's chosen pages |
| 16 | **Coherence re-rank** | +0.1 for chunks of the top 2 sections (acts as "primary sections first") |
| 17 | **Section tree** | List of headings with page ranges, summary and section type (flat in this code) |
| 18 | **Ontology** | The 13 section types: abstract, introduction, related_work, background, methodology, experiments, results, discussion, conclusion, acknowledgements, references, appendix, other |
| 19 | **Query type** | factual / structural / conceptual, by regex; reported only |
| 20 | **Hash embedding** | SHA-256-based fake vector used when the model fails; not semantic |
| 21 | **Provider dispatch** | `_generate_answer()` picks openai / gemini / hf / mock from settings |
| 22 | **Mock provider** | Returns a fixed template; lets CI run without a key |
| 23 | **Temperature** | Randomness of LLM output; default 0.2, range 0–1 |
| 24 | **Hallucination** | Fluent text that isn't supported by the source |
| 25 | **202 Accepted** | Request taken, work not finished; used for upload |
| 26 | **BackgroundTasks** | FastAPI runs a function after the response, inside the same process |
| 27 | **Polling** | Client asks `/status/{job_id}` every second until done |
| 28 | **CORS** | Browser rule controlling which sites may call the API; set from `FRONTEND_URL` |
| 29 | **Recall@k / MRR** | Share of questions with the right chunk in the top k / average of 1/rank of the first right chunk |
| 30 | **Groundedness** | How much of an answer is supported by the retrieved chunks |

---

# Part 12. Cheat sheet and revision plan

## 12.1 One-page cheat sheet

**Project:** RAG-Based Academic Document Explainer, branded **StructRAG**. Solo project; 9 commits from 2026-03-06 to 2026-08-13.

**Stack:** FastAPI + Pydantic (Python 3.11) · ChromaDB · SQLite · sentence-transformers `all-MiniLM-L6-v2` · pypdf (+ unstructured fallback) · LangChain text splitter · OpenAI / Gemini / Hugging Face / mock · React 18 + TypeScript + Vite + Tailwind + Axios · Docker + Compose · GitHub Actions · Vercel (frontend) + Railway (backend config).

**Key numbers (all from the repo)**

| Item | Value |
|---|---|
| Chunk size / overlap | 3000 / 800 **characters** (comment: about 800 / 200 tokens) |
| Embedding size | 384 numbers, normalized |
| Embedding batch / retries | 32 / 3 tries |
| `top_k` | 1–10, default 5 |
| Vector candidate pool | `2 × top_k` |
| RRF constant | k = 60 |
| Boost range | 1.0 to 2.0; base 1.3; depth bonus 0.15/level (never active); confidence threshold 0.3 |
| Sections navigated | top 3 (top 2 get the +0.1 coherence bonus) |
| Section types | 13 (including `other`) |
| Default temperature | 0.2 (range 0–1) |
| Per-IP rate limit | 60 requests/min (not `/` or `/health`) |
| Global LLM rate limit | 5 calls/min |
| LLM retries / timeout | 2 attempts / 90 s |
| Gunicorn | 1 worker, timeout 180 s |
| Progress steps | 25 → 50 → 60 → 80 → 100 |
| Polling | every 1000 ms |
| Endpoints | 9 under `/api/v1` + `/` + `/health` |
| SQLite tables | 3: `jobs`, `documents`, `document_trees` |
| Tests | 4 backend + 2 frontend; CI has 3 jobs |

**Flow 1: Ingestion (background job)**
`upload_pdf()` → 202 + job_id → `process_document_job()`: `load_pdf` → `build_document_tree` → `chunk_pages` → `embed_texts` → `add_chunks` + `save_tree` + `add_document` → completed.

**Flow 2: Ask**
`ChatPage.handleSend` → `POST /api/v1/ask` → `answer_question()`: safety regex → `get_tree` → `hybrid_retrieve` (classify, embed, navigate, vector search, confidence, boost, coherence) → `_format_chunks` → `_generate_answer` → `AskResponse` (answer, sources, prompt_used, query_type, retrieval_mode, retrieval_confidence, adaptive_boost_used).

**Formulas**
- Base score: `1 / (60 + rank)`; fused = base × boost if the chunk's page is in the chosen sections.
- Boost: `1.3 + min(affinity/5, 1) × 0.7 + 0.15 × depth`, clamped to 1–2; if confidence < 0.3: `1 + (boost − 1) × confidence/0.3`.
- Confidence = (vector top-k chunks on chosen pages) ÷ (top-k).

**The five points to remember**
1. **What it is:** a working end-to-end RAG pipeline: asynchronous ingestion (202 + job id + polling), page-aware chunking, MiniLM embeddings in Chroma, a pluggable LLM layer with a mock for CI, and cited sources.
2. **What StructRAG really is:** a section tree plus a rank boost on one vector list. It is an experiment: k = 60 and the boost constants are untuned, the tree is flat, depth is unused, the intent label isn't used, and nothing is measured.
3. **What it isn't:** not multi-turn (no history sent), no authentication, no evaluation, not production-ready, scanned PDFs and tables unsupported, hosted backend status uncertain.
4. **Known bugs you can name first:** UI shows about 2% and "Low" in boosted mode; untitled uploads get the job id as title; weak upload validation; 429 has no CORS header; silent hash-embedding fallback; errors returned as HTTP 200.
5. **How to answer anything:** *what I built → why I chose it → the trade-off → what I would improve.* If you don't know or didn't measure: say so, then say how you would find out.

**Never say:** multi-turn · any percentage improvement · patent-pending / novel / no prior art · production-ready · "100% grounded" · hierarchical / depth-weighted works · hybrid means BM25 · handles scanned PDFs · live demo works (unless you checked today). Full list: Part 8.5.

## 12.2 HR round: stories from the repo

The material has no HR content, so these only give you true facts to build on. Fill the blanks (___) with your own details.

| Question | True facts you can use | Your own part (fill in) |
|---|---|---|
| **A technical challenge you solved** | Broken-pipe errors from forked processes in the uvicorn workers; fixed with the `spawn` start method and `TOKENIZERS_PARALLELISM=false` [embeddings.py, commit `aff68da`] | How I found the cause: ___ |
| **A time something broke after a change in the outside world** | A configured Gemini model returned 404 after it was retired; I added model fallback and "-latest" aliases [rag.py, commit `4d0d4fb`] | How I noticed: ___ |
| **Improving quality** | Switched the prompt from short snippets to the full chunk text, and raised Gemini's output limit from 900 to 4096 tokens [commit `aff68da`] | How I judged "better": ___ |
| **Deployment problem** | CORS had to accept several origins; the Docker command had to use the platform's `PORT`; 2 workers became 1 [commits `3f4109f`, `c4bb7e0`] | What went wrong first: ___ |
| **Learning new tools fast** | FastAPI, ChromaDB, sentence-transformers, React/TypeScript, Docker and CI were new to me (your note: strong in Java, new to most of these) | How I learned: ___ |
| **A weakness / a mistake** | I overstated parts of my documentation (patent-pending, hierarchical tree) before checking the code; I now verify claims against the code | What I changed afterwards: ___ |
| **Working alone** | Solo project, all commits mine | How I planned the work: ___ |
| **What would you do differently?** | Build the evaluation set first; keep the tree nested; add history; use a queue and persistent storage | ___ |

A short **STAR** shape for any story: *Situation* (one sentence), *Task* (what you had to achieve), *Action* (what **you** did, in 2–3 steps), *Result* (what changed, with only true numbers).

## 12.3 Revision plans

The time estimates are my suggestions, not from the repo.

### 1-day plan (about 8 hours of work)

| Block | Time | Do this |
|---|---|---|
| 1 | 45 min | Read the cheat sheet (12.1) and Part 1. Say the 30-second pitch aloud 3 times. |
| 2 | 90 min | Part 3: 3.2 to 3.8. For each, read "How to say it" and say it aloud without looking. |
| 3 | 60 min | Part 4: Flow A (upload) and Flow C (ask). Open the files and find each function name. |
| 4 | 60 min | Part 7: S1–S6 and the top 10 findings. Write your own one-sentence honest answer for each. |
| 5 | 45 min | Part 8: practise the 2-minute pitch twice, then the "do not claim" list. |
| 6 | 90 min | Part 9: all 9 [MUST-ASK] questions first, then Q1–Q10 and Q31–Q40. Answer aloud. |
| 7 | 30 min | Part 11 self-test; review every wrong answer. |
| 8 | 30 min | Check the live Vercel URL and the backend. Start the project locally with `docker compose up --build`. Prepare a backup screen recording. |

### 3-day plan

| Day | Focus | Tasks |
|---|---|---|
| **Day 1** | Understand | Parts 1, 2, 3 (read all), Part 5 tables. Draw the architecture diagram from memory. |
| **Day 2** | Trace the code | Part 4 flows with the real files open; Part 6 decisions; Part 10 code drill (all 10 excerpts, then Tasks 1–5 verbally). |
| **Day 3** | Perform | Part 7 (honesty and resume wording), Part 8 pitches (record yourself), Part 9 (all 40 questions aloud), Part 11 self-test, demo rehearsal, final check of the live site. |

### 7-day plan

| Day | Focus | Tasks |
|---|---|---|
| **Day 1** | Run it | Run the project locally (Compose or manual). Upload a real paper with the mock provider, then with a real API key if you have one. Read Parts 1–2. Write what you observe: does the UI show 2%? Is the title a UUID? |
| **Day 2** | Concepts | Part 3 (3.1–3.7). Do the cosine and RRF examples by hand. |
| **Day 3** | Concepts + data | Part 3 (3.8–3.13) and Part 5. Skim the real files `config.py`, `schemas.py`, `jobs.py`. |
| **Day 4** | Code | Part 4 with the code open; Part 10 excerpts. Run the 4 backend tests (`pytest -q` in `backend`) and read them. |
| **Day 5** | Honesty and a fix | Part 7. Optionally fix one or two small bugs so you can say "I found and fixed this": the default title (use the file name), the UI confidence badge, or the prompt `\n`. Re-run tests. Decide your final resume wording (Part 7.4). |
| **Day 6** | Speak | Part 8 pitches; Part 9 with a friend asking the questions; record yourself and cut filler words. |
| **Day 7** | Polish | Part 11 self-test; Part 12 cheat sheet; verify the live site; rehearse the demo; sleep early. |

### Final checklist for the morning of the interview

- [ ] I can say what the project does in 30 seconds without notes.
- [ ] I can draw the ingestion and ask flows and name the files.
- [ ] I know the 12 key numbers in the table above.
- [ ] I know the 5 "never say" items that matter most: multi-turn, percentages, patent, production-ready, 100% grounded.
- [ ] I know what I will say about the hosted backend (checked today).
- [ ] I have a local demo or a recording ready.
- [ ] I can name three weaknesses and one fix for each, before being asked.

---

*End of the study guide.*
