# AI Document Pipeline (RAG System)

A production-style Retrieval-Augmented Generation (RAG) backend system for grounded question answering over custom documents. Built with FastAPI, FAISS, and LLMs (Ollama + Gemini), this system emphasizes modular design, performance optimization, and reliability through metrics and caching.

## Key Results

- Identified repeated LLM calls as the primary latency bottleneck (~1.6–2.6 s per query)
- Introduced Redis caching at the RAG response layer
- Reduced repeated-query latency to ~3.7–6.0 ms (~300×–670× speedup)
- Built a modular RAG architecture supporting both local and cloud LLM providers  

These results highlight the impact of caching on reducing latency and improving system throughput for repeated queries.

## System Summary

FastAPI-based RAG backend that retrieves document context using TF-IDF vectorization and FAISS similarity search, then generates grounded responses via LLMs (Ollama/Gemini), with optional Redis caching for low-latency repeated queries.

## Request Flow

1. Client sends a query to `/api/rag-query`
2. The query is vectorized using TF-IDF
3. FAISS retrieves the top-k relevant document chunks
4. Retrieved chunks are filtered and assembled into context
5. The LLM generates a grounded response using the retrieved context
6. The response is optionally cached in Redis
7. The API returns the answer, sources, and performance metrics

## Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline for grounded question answering over custom documents.

The system:

- Ingests and preprocesses documents  
- Splits text into manageable chunks  
- Converts text into TF-IDF vectors  
- Stores vectors in a FAISS index for similarity search  
- Retrieves relevant context for a query  
- Generates grounded responses using an LLM (Ollama or Gemini)  

Unlike simple LLM wrappers, this system reduces hallucination by enforcing strict context usage and generating responses only from retrieved document content.

The architecture separates retrieval and generation, enabling modular design, improved explainability, and scalable system evolution. It also incorporates optional caching and performance benchmarking to optimize repeated query latency and provide visibility into system efficiency.

This design mirrors production AI systems by separating retrieval from generation, enabling scalable, reliable, and cost-efficient LLM applications.

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

### Architecture Diagram

```text
[Client]
   ↓
[FastAPI API Layer]
   ↓
[Retriever Service]
   ↓
[Vectorization (TF-IDF)]
   ↓
[FAISS Index]
   ↓
[LLM Service (Ollama / Gemini)]
   ↓
[Redis Cache (optional)]
   ↓
[Response + Sources + Metrics]
```

### Architecture Overview

The system follows a Retrieval-Augmented Generation (RAG) pipeline:

- Documents are loaded and split into chunks
- Chunks are converted into TF-IDF vectors for lightweight, deployment-friendly retrieval
- Vectors are stored in a FAISS index for similarity search
- User queries are vectorized and matched against stored vectors
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

## Reliability Features

- Graceful fallback when Redis is unavailable
- Safe fallback response for out-of-scope queries
- Error handling for LLM failures
- Input validation and structured API responses

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
      "chunk_id": "doc.txt_0",
      "chunk_index": 0,
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
      "chunk_id": "doc.txt_0",
      "chunk_index": 0,
      "text": "...",
      "score": 0.42
    }
  ],
  "metrics": {
    "retrieval_time_ms": 20.5,
    "generation_time_ms": 600.2,
    "total_time_ms": 620.7,
    "retrieved_chunks": 3,
    "relevant_chunks": 3,
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
      "chunk_id": "doc.txt_0",
      "chunk_index": 0,
      "text": "..."
    }
  ]
}
```

## Tech Stack

- **Backend:** FastAPI  
- **Vector Index:** FAISS    
- **Vectorization:** TF-IDF (lightweight and deployment-friendly)  
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
├── tests/                  # Unit tests with mocked RAG/LLM dependencies
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
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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
python -m pytest -v
```

The test suite uses mocks for RAG and LLM interactions, so it does not require a live Ollama, Gemini, or Redis instance.

## Configuration

### For local development

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

### For Docker

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
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

Local development:

```env
REDIS_URL=redis://127.0.0.1:6379/0
RAG_CACHE_TTL_SECONDS=300
```

Docker:

```env
REDIS_URL=redis://redis:6379/0
RAG_CACHE_TTL_SECONDS=300
```

Caching is optional. If `REDIS_URL` is not provided, or Redis is unavailable, the system will automatically fall back to direct RAG execution without caching.

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

Measured using `scripts/benchmark_rag.py`:

```bash
python scripts/benchmark_rag.py
```

The script attempts to clear the benchmark cache key before running:

- First via REDIS_URL if Redis is directly reachable
- Otherwise via the local Docker Redis container (ai-redis)

If Redis is unavailable, the script still runs and reports the observed cache status of the first request and follow-up requests.

Local cold-cache benchmark results with Redis available:

- First cold-cache request latency: ~1.6–2.6 s
- Average cached follow-up latency: ~3.7–6.0 ms
- Cached follow-up range: ~2.3–11.8 ms
- Approximate speedup: **~300×–670× (seconds → milliseconds)**

These results show that caching eliminates redundant retrieval and LLM generation, reducing response latency from seconds to milliseconds for identical queries and improving overall system throughput.

## Design Trade-offs

- TF-IDF vs dense embeddings  
  Chose TF-IDF for lower memory usage, simpler deployment, and lightweight retrieval.

- Local LLM (Ollama) vs cloud LLM (Gemini)  
  Uses Ollama for local development and Gemini for cloud-friendly deployment on lower-memory infrastructure.

- Redis caching at response level  
  Reduces repeated-query latency substantially, but introduces cache lifecycle and invalidation considerations.

- FAISS vs managed vector database  
  Uses FAISS for simplicity and local control, while keeping the architecture flexible enough to migrate to Pinecone or Weaviate later.

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

- Stateless FastAPI service supports horizontal scaling via replicas  
- FAISS can be replaced with distributed or managed vector databases such as Pinecone or Weaviate  
- Retrieval and generation can be separated into independent services  
- Async or background ingestion can support larger document processing workloads  
- Distributed caching and cache invalidation strategies can be added as traffic grows  
- LLM abstraction supports remote or distributed inference backends  

## Limitations

- TF-IDF retrieval may underperform on semantic queries compared to dense embeddings
- FAISS is single-node and not distributed
- Cache invalidation is not implemented for dynamic document updates

## Future Improvements

- Add distributed caching and cache invalidation strategies  
- Implement async/background processing for large-scale ingestion  
- Integrate managed vector databases (e.g., Pinecone, Weaviate)  
- Add CI/CD pipeline for automated testing and deployment  
- Build a frontend interface for interactive querying  

## What This Project Demonstrates

- End-to-end backend system design
- Backend API design with FastAPI
- Service-oriented, modular architecture
- Data pipeline design (ingestion → retrieval → generation)
- Retrieval-Augmented Generation (RAG) system design
- Integration of LLMs into production-style services
- Performance measurement and optimization through metrics and caching
- Production-oriented engineering practices (Docker, cloud deployment, caching)