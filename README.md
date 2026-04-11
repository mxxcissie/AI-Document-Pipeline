# AI Document Pipeline (RAG System)

A production-style Retrieval-Augmented Generation (RAG) backend system for grounded question answering over custom documents. Built with FastAPI, FAISS, and LLMs (Ollama + Gemini), this system focuses on reliability, modular design, and performance visibility.

## Overview

This project implements an end-to-end RAG pipeline:

- Ingests and preprocesses documents
- Converts text into embeddings
- Stores embeddings in a vector database (FAISS)
- Retrieves relevant context for a query
- Generates grounded responses using an LLM

Unlike simple LLM wrappers, this system reduces hallucination by constraining responses to retrieved document context.

## Live Demo
https://ai-document-pipeline.onrender.com

## Architecture

```text
User Query
   ↓
FastAPI Endpoint (/api/rag-query)
   ↓
Embedding (query → vector)
   ↓
Vector Search (FAISS)
   ↓
Relevance Filtering
   ↓
Context Construction
   ↓
LLM (Ollama / Gemini)
   ↓
Grounded Answer + Sources + Metrics
```

### Document pipeline:

```text
Documents → Load → Clean → Chunk → Embed → Store → Retrieve
```

## Features

- Semantic search using vector embeddings
- Document ingestion and text chunking pipeline
- Retrieval-Augmented Generation (RAG)
- Grounded answers with source attribution
- Multi-provider LLM support (local + cloud)
- Containerized backend (Docker)
- Cloud deployment (Render)
- Built-in latency and retrieval quality metrics

## API Endpoints

- Health Check
`GET /health`

- RAG Query
`POST /api/rag-query`

Request:
```json
{
  "question": "What is RAG?",
  "top_k": 3
}
```

Response:
```json
{
  "question": "What is RAG?",
  "answer": "...",
  "sources": [...],
  "metrics": {
    "retrieval_time_ms": 20.5,
    "generation_time_ms": 600.2,
    "total_time_ms": 620.7,
    "retrieved_chunks": 3,
    "relevant_chunks": 2
  }
}
```

- Preview Document Chunks
`GET /api/documents/chunks`

## Tech Stack

- Backend: FastAPI
- Vector DB: FAISS
- Embeddings: sentence-transformers
- LLM: Ollama (local), Gemini (cloud)
- Infra: Docker, Render
- Language: Python
- Server: Uvicorn
- Config: python-dotenv

## Project Structure

```bash
project_root/
├── app/
│   ├── api/                # FastAPI routes (endpoints)
│   │   └── routes.py
│   ├── core/               # Configuration management
│   │   └── config.py
│   ├── models/             # Pydantic schemas (request/response)
│   │   └── schemas.py
│   ├── pipeline/           # RAG data pipeline
│   │   ├── document_loader.py
│   │   ├── text_chunker.py
│   │   ├── embedder.py
│   │   └── retriever.py
│   ├── services/           # Business logic + LLM integration
│   │   ├── llm_service.py
│   │   ├── ollama_service.py
│   │   ├── gemini_service.py
│   │   ├── llm_factory.py
│   │   └── rag_service.py
│   └── main.py             # FastAPI application entrypoint
├── vectorstore/            # Vector database (FAISS)
│   ├── build_index.py
│   ├── embedding_service.py
│   └── faiss_store.py
├── data/                   # Sample documents for indexing
│   └── sample_docs/
├── scripts/                # Utility & evaluation scripts
│   ├── evaluate_rag.py
│   └── inspect_chunks.py
├── tests/                  # Unit tests
│   └── test_pipeline.py
├── .env.example            # Environment variable template
├── docker-compose.yml      # Multi-container setup
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── pytest.ini              # Pytest configuration
└── README.md
```

### Architecture Overview

The system follows a Retrieval-Augmented Generation (RAG) pipeline:

- Documents are loaded and split into chunks
- Chunks are embedded using sentence-transformers
- Embeddings are stored in a FAISS vector index
- User queries are embedded and matched against stored vectors
- Relevant context is retrieved and passed to an LLM (Ollama or Gemini)
- The LLM generates a grounded response based on retrieved context

## How to Run

### Option 1 — Docker (Recommended)
```bash
docker compose up --build
```

Open:
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

### Option 2 — Local Development
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Configuration

Create a .env file:

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b

For cloud deployment:

LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash

## Deployment

Deployed on Render:

https://ai-document-pipeline.onrender.com

Supports environment-based switching between local and cloud LLM providers.

## Evaluation & Metrics

The system is evaluated using curated queries to validate:

- Correct responses for in-scope questions
- Safe fallback for out-of-scope queries
- Grounded answers using retrieved context

### Runtime Metrics

Each request includes:

- Retrieval latency
- Generation latency
- Total response time
- Retrieved vs relevant chunk count

### Evaluation Script

```bash
python scripts/evaluate_rag.py
```

Tests:

- In-scope queries
- Out-of-scope handling
- Response consistency

## Design Highlights

- Separation of concerns
    - API layer vs pipeline vs LLM services
- Provider-agnostic LLM architecture
    - Easily extendable to new providers
- Config-driven system
    - Behavior controlled via environment variables
- Extensible pipeline
    - Ready for additional data sources and scaling

## Scaling Considerations

- Stateless FastAPI service supports horizontal scaling
- FAISS can be replaced with managed vector DBs (e.g., Pinecone, Weaviate)
- LLM abstraction enables distributed or remote inference
- Suitable for integration into larger microservice architectures

## Future Improvements

- Add Redis caching layer for query optimization
- Add automated testing and CI/CD pipeline
- Support PDF and large-scale document ingestion
- Add frontend interface

## Summary

This project demonstrates:

- End-to-end backend system design
- Integration of LLMs into production-style services
- Retrieval-based AI architecture (RAG)
- Performance measurement and evaluation
- Cloud deployment and containerization

## What This Project Demonstrates

- Backend API design with FastAPI
- Service-oriented architecture
- Data pipeline design (ingestion → retrieval → generation)
- Applied AI system design
- Production-oriented engineering practices