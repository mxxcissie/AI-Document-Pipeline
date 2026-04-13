# AI Document Pipeline (RAG System)

A production-style Retrieval-Augmented Generation (RAG) backend system for grounded question answering over custom documents. Built with FastAPI, FAISS, and LLMs (Ollama + Gemini), this system emphasizes modular architecture, reliability, and measurable performance through built-in metrics and caching.

## Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline for grounded question answering over custom documents.

The system:

- Ingests and preprocesses documents  
- Splits text into manageable chunks  
- Converts text into embeddings  
- Stores embeddings in a FAISS vector index  
- Retrieves relevant context for a query  
- Generates grounded responses using an LLM (Ollama or Gemini)  

Unlike simple LLM wrappers, this system reduces hallucination by enforcing strict context usage and generating responses only from retrieved document content.

The architecture separates retrieval and generation, enabling modular design, improved explainability, and scalable system evolution. It also incorporates optional caching and performance benchmarking to optimize repeated query latency and provide visibility into system efficiency.

This design mirrors production AI systems by separating retrieval from generation, enabling scalable, reliable, and cost-efficient LLM applications.

## Key Results

- Reduced repeated query latency from ~2 s to ~2–3 ms using Redis caching  
- Achieved ~750×–900× speedup for identical queries  
- Built a modular RAG architecture supporting both local and cloud LLM providers  

These results highlight the impact of caching on reducing latency and improving system throughput for repeated queries.

## Live Demo (Render Deployment)

Try the interactive API via Swagger UI:
- https://ai-document-pipeline.onrender.com/docs  

*Note: Uses Gemini in the cloud environment for low-memory deployment.*

## Architecture

```text
User Query
   ↓
FastAPI Endpoint (/api/rag-query)
   ↓
Query Embedding (text → vector)
   ↓
Vector Search (FAISS)
   ↓
Relevance Filtering
   ↓
Context Construction
   ↓
LLM (Ollama / Gemini)
   ↓
(Optional) Redis Cache
   ↓
Grounded Answer + Sources + Metrics
```

### Architecture Overview

The system follows a Retrieval-Augmented Generation (RAG) pipeline:

- Documents are loaded and split into chunks
- Chunks are embedded using a lightweight TF-IDF pipeline for deployment-friendly retrieval
- Embeddings are stored in a FAISS vector index
- User queries are embedded and matched against stored vectors
- Relevant context is retrieved and passed to an LLM (Ollama or Gemini)
- The LLM generates a grounded response based on retrieved context

This architecture separates retrieval and generation to ensure responses are grounded, traceable, and resistant to hallucination.

It enables scalable, explainable, and production-ready AI systems by decoupling data retrieval from model inference.

An optional Redis caching layer is applied at the RAG response level to eliminate redundant retrieval and LLM generation, improving latency for repeated queries without affecting retrieval correctness.

### Ingestion & Retrieval Pipeline

```text
Documents
   ↓
Load → Clean → Chunk
   ↓
Embedding Generation
   ↓
Vector Store (FAISS)
   ↓
Retrieval at Query Time
```

## Features

- Vector-based document retrieval with FAISS  
- End-to-end document ingestion and text chunking pipeline  
- Retrieval-Augmented Generation (RAG) for grounded responses  
- Source attribution for explainability and traceability  
- Multi-provider LLM support (Ollama for local, Gemini for cloud)  
- Containerized backend with Docker  
- Cloud deployment on Render  
- Built-in performance metrics (latency and retrieval quality)  
- Optional Redis caching for repeated queries  
- Benchmarking for cache performance and latency optimization  

## API Endpoints

### Health Check

`GET /api/health`

Response:
```json
{
  "status": "ok"
}
```

### Basic LLM Query

`POST /api/query`

Request:
```json
{
  "question": "Explain what RAG is."
}
```

Response:
```json
{
  "question": "Explain what RAG is.",
  "answer": "..."
}
```

### Document Search (Vector Retrieval)

`POST /api/search`

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
  "query": "What is RAG?",
  "results": [
    {
      "source": "doc.txt",
      "chunk_id": 0,
      "text": "...",
      "score": 0.42
    }
  ]
}
```

### RAG Query (Retrieval-Augmented Generation)

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
  "sources": [
    {
      "source": "doc.txt",
      "chunk_id": 0,
      "text": "...",
      "score": 0.42
    }
  ],
  "metrics": {
    "retrieval_time_ms": 20.5,
    "generation_time_ms": 600.2,
    "total_time_ms": 620.7,
    "retrieved_chunks": 3,
    "relevant_chunks": 2,
    "cache": "hit"
  }
}
```

### Preview Document Chunks

`GET /api/documents/chunks`

Response:
```json
{
  "total_chunks": 10,
  "chunks": [
    {
      "source": "doc.txt",
      "chunk_id": 0,
      "text": "..."
    }
  ]
}
```

## Tech Stack

- **Backend:** FastAPI  
- **Vector Database:** FAISS  
- **Embeddings:** TF-IDF (lightweight and deployment-friendly)  
- **LLM Providers:** Ollama (local), Gemini (cloud)  
- **Caching:** Redis (optional)  
- **Infrastructure:** Docker, Render  
- **Language:** Python  
- **Server:** Uvicorn  
- **Configuration:** python-dotenv  

## Project Structure

```bash
project_root/
├── app/
│   ├── api/                # FastAPI routes (endpoints)
│   │   └── routes.py
│   ├── core/               # Configuration and environment setup
│   │   └── config.py
│   ├── models/             # Pydantic schemas (request/response)
│   │   └── schemas.py
│   ├── pipeline/           # RAG pipeline (ingestion → retrieval)
│   │   ├── document_loader.py
│   │   ├── text_chunker.py
│   │   ├── embedder.py
│   │   └── retriever.py
│   ├── services/           # Business logic and LLM integration
│   │   ├── cache.py
│   │   ├── llm_service.py
│   │   ├── ollama_service.py
│   │   ├── gemini_service.py
│   │   ├── llm_factory.py
│   │   └── rag_service.py
│   └── main.py             # FastAPI application entrypoint
│
├── vectorstore/            # Vector database layer (FAISS)
│   ├── build_index.py
│   ├── embedding_service.py
│   └── faiss_store.py
│
├── data/                   # Sample documents for indexing
│   └── sample_docs/
│
├── scripts/                # Evaluation and debugging tools
│   ├── benchmark_rag.py
│   ├── evaluate_rag.py
│   └── inspect_chunks.py
│
├── tests/                  # Unit tests
│   ├── test_pipeline.py
│   └── test_rag_service.py
│
├── .env.example            # Environment variable template
├── docker-compose.yml      # Local container orchestration
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── pytest.ini              # Pytest configuration
└── README.md
```

## How to Run

### Option 1 — Docker

```bash
docker compose up --build
```

Open:
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/api/health

### Option 2 — Local Development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Quick Test

After starting the server, verify the system is working:

You can also explore the API via Swagger UI:

- http://127.0.0.1:8000/docs

### Health Check

```bash
curl http://127.0.0.1:8000/api/health
```

Expected:
```json
{"status": "ok"}
```

### RAG Query

```bash
curl -X POST "http://127.0.0.1:8000/api/rag-query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is RAG?","top_k":3}'
```

### Verify Cache Behavior

Run the same query twice:

- First request → slower (cache miss)
- Second request → faster (cache hit)

Response includes:
```json
{
  "cache": "hit"
}
```

## Run Tests

```bash
pytest -v
```

## Configuration

### Create a `.env` file for local development

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

### For cloud deployment

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
```

Gemini is used in cloud environments to reduce memory usage by offloading LLM inference to an external API, enabling deployment on low-memory instances.

### Optional Redis (Caching)

```env
REDIS_URL=redis://127.0.0.1:6379/0
RAG_CACHE_TTL_SECONDS=300
```

Caching is optional. If `REDIS_URL` is not provided, the system will automatically fall back to direct RAG execution without caching.

## Deployment

Deployed on Render:

- https://ai-document-pipeline.onrender.com/docs

Supports environment-based switching between local (Ollama) and cloud (Gemini) LLM providers.

Supports optional Redis caching in both local and cloud environments with graceful fallback when unavailable.

## Evaluation & Metrics

The system is evaluated using curated queries to validate:

- Correct responses for in-scope questions  
- Safe fallback behavior for out-of-scope queries  
- Grounded answers based strictly on retrieved document context  

### Runtime Metrics

Each API response includes:

- Retrieval latency (vector search time)  
- Generation latency (LLM response time)  
- Total response time  
- Retrieved vs. relevant chunk count  
- Cache status (hit/miss) for repeated queries  

These metrics provide visibility into system performance, retrieval quality, and caching effectiveness.

### Evaluation Script

Run the evaluation script:

```bash
python scripts/evaluate_rag.py
```

The script validates:

- In-scope query accuracy
- Out-of-scope fallback behavior
- Response consistency and grounding

## Caching & Performance

To improve response efficiency for repeated queries, the system integrates an optional Redis caching layer for RAG responses.

### Cache Behavior

- First request (cache miss): performs full retrieval and LLM generation  
- Subsequent identical requests (cache hit): returns cached response instantly  

### Benchmark Results

Measured using `scripts/benchmark_rag.py` with Redis cache cleared before each run:

```bash
python scripts/benchmark_rag.py
```

- Cache miss latency: ~1.7–2.2 s
- Average cache hit latency: ~2.4–2.8 ms
- Cache hit range: ~1.1–6.3 ms
- Approximate speedup: **~750×–900× (seconds → milliseconds)**

These results show that caching eliminates redundant retrieval and LLM generation, reducing response latency from seconds to milliseconds for identical queries and improving overall system throughput.

## Design Highlights

- Separation of concerns  
  - Clear boundaries between API layer, data pipeline, and LLM services  
- Provider-agnostic LLM architecture  
  - Easily extendable to additional providers beyond Ollama and Gemini  
- Config-driven system  
  - Runtime behavior controlled via environment variables  
- Extensible pipeline  
  - Designed to support new data sources and scalable ingestion workflows  
- Lazy loading for embeddings and vector index  
  - Avoids heavy initialization at startup and enables deployment on low-memory environments  
- Observability through built-in metrics  
  - Enables monitoring of retrieval quality and system performance  
- Modular pipeline design  
  - Decoupled embedder and retriever components allow flexible experimentation and easy replacement of retrieval strategies  
- Graceful degradation for caching layer  
  - System continues to function correctly when Redis is unavailable  

## Scaling Considerations

- Stateless FastAPI service enables horizontal scaling  
- FAISS can be replaced with managed vector databases (e.g., Pinecone, Weaviate)  
- LLM abstraction supports distributed or remote inference  
- Redis caching reduces repeated LLM load and improves throughput for repeated queries  
- Architecture is suitable for integration into microservice-based systems  

## Future Improvements

- Add distributed caching and cache invalidation strategies  
- Implement async/background processing for large-scale ingestion  
- Integrate managed vector databases (e.g., Pinecone, Weaviate)  
- Add CI/CD pipeline for automated testing and deployment  
- Build a frontend interface for interactive querying  

## Summary

This project demonstrates:

- End-to-end backend system design  
- Integration of LLMs into production-style services  
- Retrieval-Augmented Generation (RAG) architecture  
- Performance measurement and optimization (metrics + caching)  
- Cloud deployment and containerization  

## What This Project Demonstrates

- Backend API design with FastAPI  
- Service-oriented and modular architecture  
- Data pipeline design (ingestion → retrieval → generation)  
- Applied AI system design (RAG + LLM integration)  
- Production-oriented engineering practices (Docker, metrics, caching)  