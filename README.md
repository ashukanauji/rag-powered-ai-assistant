# Production-Ready AI Knowledge Assistant (RAG)

A full-stack Generative AI application that supports document upload and grounded Q&A using Retrieval-Augmented Generation (RAG).

## Project Overview

This app lets users upload PDF/TXT documents, indexes them, and answers questions by retrieving relevant chunks and injecting them into an LLM prompt.

### Why RAG?
- Keeps responses grounded in user-provided data (lower hallucination risk).
- Faster iteration than fine-tuning when documents change frequently.
- Enables explainability through source attribution.

### Why Hybrid Retrieval?
- **Dense retrieval (SentenceTransformers + ChromaDB)** captures semantic similarity.
- **BM25 retrieval** captures exact keyword matches.
- Combining both improves recall for technical and long-tail queries.

## Features

- FastAPI backend with `/upload`, `/query`, `/health`
- Full RAG pipeline (ingest → chunk → embed → store → retrieve → generate)
- Hybrid retrieval (dense + BM25)
- Persistent vector storage (ChromaDB)
- Groq LLM integration (Llama 3 family)
- Streaming responses via SSE
- Redis caching for repeated queries
- API key authentication (`x-api-key`)
- Evaluation script with similarity scoring
- Next.js frontend with chat UI, upload flow, source display
- Dockerized deployment setup

## Architecture (Text Diagram)

```text
[Next.js Frontend]
   | HTTP
   v
[FastAPI API Layer]
   |- Auth (API key)
   |- Cache (Redis)
   |- Ingestion (PDF/TXT parsing + chunking)
   |- Retrieval Engine
       |- Dense: SentenceTransformer -> ChromaDB (persistent)
       |- Sparse: BM25 over chunk corpus
   |- Prompt Builder (context injection)
   |- LLM Client (Groq Llama 3)
   v
[Answer + Sources]
```

## Project Structure

```text
backend/
  app/
    main.py
    rag.py
    ingestion.py
    database.py
    auth.py
    cache.py
  evaluate.py
  data/
frontend/
  app/
  components/
docker/
  docker-compose.yml
requirements.txt
Dockerfile
README.md
.env.example
```

## Setup Instructions

### 1) Prerequisites
- Python 3.11+
- Node.js 20+
- Redis (optional but recommended)
- Groq API key

### 2) Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.app.main:app --reload --port 8000
```

### 3) Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## API Endpoints

### GET `/health`
Health check.

### POST `/upload`
Upload PDF/TXT and index chunks.
- Multipart field: `file`
- Optional form field: `chunk_size`
- Header: `x-api-key` (if `API_KEY` env var is set)

### POST `/query`
Ask a question.

```json
{
  "question": "What are the key findings?",
  "top_k": 5,
  "stream": false
}
```

If `stream=true`, returns `text/event-stream` token chunks.

## Evaluation

Create a JSON dataset:

```json
[
  {"query": "What is X?", "expected_answer": "..."}
]
```

Run:

```bash
python backend/evaluate.py --dataset evaluation.json --top-k 5
```

Outputs average similarity score using RapidFuzz token-set ratio.

## Docker

### Backend-only image

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 --env-file .env rag-assistant
```

### Full stack (backend + frontend + redis)

```bash
cd docker
docker compose --env-file ../.env up --build
```

## Deployment Notes (Render / AWS)

- Set required env vars:
  - `GROQ_API_KEY`
  - `GROQ_MODEL` (optional)
  - `API_KEY` (recommended for production)
  - `REDIS_URL`
  - `CORS_ALLOW_ORIGINS`
- Persist `backend/data` for ChromaDB durability.
- Scale considerations:
  - Move BM25 + metadata to managed search backend (OpenSearch/Elastic) at large scale.
  - Use background workers for ingestion.
  - Add rate limiting and request tracing.
  - Use JWT/OIDC auth and per-tenant isolation.

## Tradeoffs & Scalability Considerations

- ChromaDB local persistence is simple and developer-friendly, but less suitable than managed vector DBs for very high throughput.
- In-memory BM25 rebuild is easy to maintain; for massive corpora, switch to an incremental search index.
- Redis caching reduces repeated LLM calls but must be tuned for staleness and tenant isolation.

## License

MIT
