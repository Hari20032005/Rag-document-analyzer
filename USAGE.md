# Minimal Usage

## 1) Upload a PDF

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@EXAMPLE_PAPERS/sample_paper.pdf" \
  -F "title=Sample Academic Paper"
```

Example response:

```json
{"job_id":"2d9ea9f2-4ace-4f05-8cf8-2dea1488df1c","status":"processing"}
```

## 2) Poll job status

```bash
curl "http://localhost:8000/api/v1/status/2d9ea9f2-4ace-4f05-8cf8-2dea1488df1c"
```

Wait until:

```json
{"status":"completed","doc_id":"<DOCUMENT_ID>","progress":100.0,...}
```

## 3) Ask a question

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "<DOCUMENT_ID>",
    "question": "What is the main methodology?",
    "top_k": 5,
    "mode": "qa",
    "temperature": 0.2
  }'
```

## 4) Get section explanation

```bash
curl -X POST "http://localhost:8000/api/v1/explain-section" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "<DOCUMENT_ID>",
    "section": "methodology"
  }'
```

## 5) Get summary

```bash
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "<DOCUMENT_ID>",
    "length": "medium"
  }'
```
