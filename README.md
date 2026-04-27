# AI Document Pipeline (RAG System)

> Production-style RAG backend that reduced query latency from ~2 s to ~2–3 ms (~447×–684×) using Redis caching to eliminate redundant LLM calls.

A Retrieval-Augmented Generation (RAG) backend for grounded question answering over custom documents, built with FastAPI, FAISS, and multi-provider LLMs (Ollama + Gemini). Focused on performance, scalability, and production-style system design.

## Key Results

- Identified LLM inference as the primary latency bottleneck (~1.4–2.0 s)
- Introduced Redis caching at the RAG response layer
- Reduced repeated-query latency to ~2.6–3.8 ms (~447×–684× speedup)
- Eliminated redundant LLM calls, improving throughput and reducing inference load
- Uses a stateless FastAPI API layer compatible with horizontal scaling
- Designed a modular RAG architecture supporting local and cloud LLMs
- Designed for production trade-offs (latency vs retrieval quality, local vs cloud inference)

## Why This Project

Built to simulate real-world LLM systems where latency, cost, and scalability are critical.

- Eliminates LLM bottlenecks through caching
- Demonstrates stateless, horizontally scalable API design
- Explores trade-offs between retrieval quality, latency, and deployment constraints

## Live Demo (Render Deployment)

Try the interactive API via Swagger UI:

- https://ai-document-pipeline.onrender.com/docs  

*Note: Uses Gemini for cloud inference to reduce memory usage and enable deployment on low-resource instances.*

## Request Flow

1. Client sends a query to `/api/rag-query`
2. Cache lookup (Redis)
   - Cache hit → return the cached response immediately
   - Cache miss → continue through retrieval and generation
3. The query is vectorized using the configured retrieval provider (TF-IDF by default for low memory and fast rebuilds)
4. FAISS retrieves the top-k relevant document chunks
5. Retrieved chunks are filtered and assembled into context
6. The LLM generates a grounded response using the retrieved context
7. The response is optionally stored in Redis for repeated queries
8. The API returns the answer, sources, and performance metrics

## Architecture

```text
User Query
   ↓
FastAPI Endpoint (/api/rag-query)
   ↓
Vectorization (provider-configurable)
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
[Vectorization Layer]
   ↓
[FAISS Index]
   ↓
[Saved Index + Metadata]
   ↓
[LLM Service (Ollama / Gemini)]
   ↓
[Redis Cache (optional)]
   ↓
[Response + Sources + Metrics]
```

### Design Notes

The system follows a Retrieval-Augmented Generation (RAG) pipeline:

- Documents are loaded and split into chunks
- Chunks are converted into vectors using the configured retrieval provider
- Vectors are stored in a FAISS index for similarity search
- User queries are vectorized using the same provider and matched against stored vectors
- Relevant context is retrieved and passed to an LLM (Ollama or Gemini)
- The LLM generates a grounded response based on retrieved context

This architecture separates retrieval and generation to ensure responses are grounded, traceable, and resistant to hallucination.

It enables scalable, explainable, and production-ready AI systems by decoupling data retrieval from model inference.

An optional Redis caching layer is applied at the RAG response level to eliminate redundant retrieval and LLM generation, improving latency for repeated queries without affecting retrieval correctness.

The vector index, document metadata, and an index manifest can be persisted to disk, allowing the API to reuse a previously built FAISS index and automatically rebuild it when source documents or retrieval settings change.

### Ingestion & Retrieval Pipeline

```text
Documents
   ↓
Load → Clean → Chunk
   ↓
Vectorization Layer
   ↓
Vector Store (FAISS)
   ↓
Retrieval at Query Time
```

## Features

### Core Capabilities

- End-to-end RAG pipeline (ingestion -> retrieval -> generation)
- FAISS-based vector similarity search with persisted index loading and stale-index detection
- Source attribution for explainability

### Performance & Scalability

- Stateless FastAPI APIs for horizontal scaling
- Redis caching for repeated queries (~447×–684× speedup)
- Built-in performance metrics (latency, retrieval quality)

### Flexibility

- Multi-provider LLM support (Ollama, Gemini)
- Configurable retrieval providers (TF-IDF default, optional local dense embeddings)

### Infrastructure

- Dockerized deployment on Render
- Environment-based configuration

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

### Direct LLM Query (Non-RAG)

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
  "total_chunks": 18,
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
- **Index Persistence:** On-disk FAISS index + metadata + manifest  
- **Vectorization:** Configurable retrieval provider (`tfidf` by default, optional `local_dense`)  
- **LLM Providers:** Ollama (local), Gemini (cloud)  
- **Caching:** Redis (optional)  
- **Infrastructure:** Docker, Render  
- **Language:** Python  
- **Server:** Uvicorn  
- **Configuration:** python-dotenv  

## Project Structure

```text
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
│   ├── indexes/            # Persisted FAISS indexes + metadata + manifests
│   └── sample_docs/
│
├── scripts/                # Evaluation and debugging tools
│   ├── build_index.py
│   ├── benchmark_rag.py
│   ├── compare_retrieval_providers.py
│   ├── evaluate_rag.py
│   ├── inspect_chunks.py
│   └── retrieval_eval_cases.json
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

If you are using the default Docker configuration with Ollama, make sure Ollama is running on the host machine:

```bash
ollama serve
ollama pull qwen2.5:3b
```

The API container connects to Ollama through `http://host.docker.internal:11434`.

Open:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/api/health

### Option 2 — Local Development

Configure `.env.local` first if you want to use `local_dense`; otherwise the default retrieval provider is `tfidf`.

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/build_index.py
```

Rebuild the persisted index any time your sample documents or retrieval provider changes:

```bash
python scripts/build_index.py
```

The API also uses the saved manifest to detect stale indexes and automatically rebuild them when the source documents or retrieval configuration no longer match the persisted index.

If you want caching enabled during local development, start Redis first:

```bash
docker compose up -d redis
```

You can also use a locally installed Redis server instead.

If you are using Ollama locally, start it before launching the API:

```bash
ollama serve
```

Make sure the configured model is available, for example:

```bash
ollama pull qwen2.5:3b
```

```bash
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
EMBEDDING_PROVIDER=local_dense
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

The first `local_dense` run may download the MiniLM model unless it is already cached locally or you provide `LOCAL_EMBEDDING_MODEL_PATH`.

For fully offline local dense rebuilds, point to a downloaded model directory and enable offline-only mode:

```env
LOCAL_EMBEDDING_MODEL_PATH=/absolute/path/to/all-MiniLM-L6-v2
LOCAL_EMBEDDING_OFFLINE_ONLY=true
```

### For Docker

```env
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=tfidf
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5:3b
```

### For cloud deployment

```env
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=tfidf
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
```

Gemini is used in cloud environments to reduce memory usage by offloading LLM inference to an external API, enabling deployment on low-memory instances.

### Optional Redis (Caching)

#### Local development:

```env
REDIS_URL=redis://127.0.0.1:6379/0
RAG_CACHE_TTL_SECONDS=300
```

#### Docker:

```env
REDIS_URL=redis://redis:6379/0
RAG_CACHE_TTL_SECONDS=300
```

Caching is optional. If `REDIS_URL` is not provided, or Redis is unavailable, the system will automatically fall back to direct RAG execution without caching.

You can run Redis either as:

- a local Redis process (`redis-server`)
- a Docker container with `docker compose up -d redis`

Because the Docker Redis service publishes port `6379`, the local app can use the same `.env.local` Redis URL:

```env
REDIS_URL=redis://127.0.0.1:6379/0
```

## Deployment

Deployed on Render:

- https://ai-document-pipeline.onrender.com/docs

Supports environment-based switching between local (Ollama) and cloud (Gemini) LLM providers, as well as configurable retrieval providers.

Supports optional Redis caching in both local and cloud environments with graceful fallback when unavailable.

## Evaluation & Metrics

The system is evaluated using curated test queries to validate retrieval correctness, grounding, and fallback behavior:

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

This demonstrates that caching eliminates redundant LLM calls, turning a latency-bound workflow into a fast lookup path for repeated queries.

### Evaluation Script

Run the evaluation script:

```bash
python scripts/evaluate_rag.py
```

The script validates:

- In-scope query accuracy
- Out-of-scope fallback behavior
- Response consistency and grounding

### Retrieval Provider Comparison

Compare `tfidf` and `local_dense` retrieval behavior side by side:

```bash
python scripts/compare_retrieval_providers.py
```

The script reports:

- Index build time per provider
- Query latency per provider
- Retrieval hit@1 on the labeled evaluation query set
- Automatic skip output when `local_dense` dependencies are unavailable

If you want `local_dense` rebuilds to work without internet access, download the model locally and set `LOCAL_EMBEDDING_MODEL_PATH` with `LOCAL_EMBEDDING_OFFLINE_ONLY=true`.

Current evaluation set:

- `6` sample documents
- `12` labeled retrieval queries across RAG, vector search, FastAPI, Redis, Docker, and FAISS topics

Recent local comparison on the expanded sample corpus:

- `tfidf`: ~13.3–14.3 ms index build, ~0.21–0.22 ms average query latency, `10/12` hit@1
- `local_dense`: ~3.26–3.50 s index build, ~36–39 ms average query latency, `12/12` hit@1

Trade-off summary:

- `tfidf` is the better default for low-memory cloud deployment and fast rebuilds
- `local_dense` is better suited to local experimentation when retrieval quality matters more than startup cost and more variable query latency
- the expanded benchmark surfaces the real quality gap: `local_dense` retrieved the correct top result on all 12 labeled questions, while `tfidf` missed 2 semantically broader queries

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

A practical local benchmark setup is:

- local FastAPI app
- Docker Redis
- local Ollama

The script attempts to clear the benchmark cache key before running:

- First via REDIS_URL if Redis is directly reachable
- Otherwise via the local Docker Redis container (ai-redis)

If Redis is unavailable, the script still runs and reports the observed cache status of the first request and follow-up requests.

This introduces a clear cold-path (full RAG execution) vs warm-path (cached response) optimization strategy.

Local cold-cache benchmark results with Redis available:

- First cold-cache request latency: ~1.4–2.0 s
- Average cached follow-up latency: ~2.6–3.8 ms
- Cached follow-up range: ~1.6–5.2 ms
- Approximate speedup: **~447×–684× (seconds → milliseconds)**

These results show that caching eliminates redundant retrieval and LLM generation, reducing response latency from seconds to milliseconds for identical queries and improving overall system throughput.

## Design Trade-offs

- TF-IDF vs dense embeddings  
  Uses TF-IDF by default for lower memory usage and simpler deployment, while supporting optional local dense embeddings for higher-quality retrieval in less constrained environments.

- Local LLM (Ollama) vs cloud LLM (Gemini)  
  Uses Ollama for local development and Gemini for cloud-friendly deployment on lower-memory infrastructure.

- Redis caching at response level  
  Reduces repeated-query latency substantially, but introduces cache lifecycle and invalidation considerations.

- FAISS vs managed vector database  
  Uses FAISS for simplicity and local control, while keeping the architecture flexible enough to migrate to Pinecone or Weaviate later.

## Design Highlights

- Separation of concerns  
  Clear boundaries between API layer, data pipeline, and LLM services

- Provider-agnostic LLM architecture  
  Easily extendable to additional providers beyond Ollama and Gemini  

- Config-driven system  
  Runtime behavior controlled via environment variables

- Extensible pipeline  
  Designed to support new data sources and scalable ingestion workflows 

- Lazy loading for vectorizer/model and vector index  
  Avoids heavy initialization at startup and enables deployment on low-memory environments  

- Observability through built-in metrics  
  Enables monitoring of retrieval quality and system performance  

- Modular pipeline design  
  Decoupled embedder and retriever components allow flexible experimentation and easy replacement of retrieval strategies  

- Graceful degradation for caching layer  
  System continues to function correctly when Redis is unavailable  

## Scaling Considerations

- Stateless FastAPI service supports horizontal scaling via replicas  
- FAISS can be replaced with distributed or managed vector databases such as Pinecone or Weaviate  
- Retrieval and generation can be separated into independent services  
- Async or background ingestion can support larger document processing workloads  
- Distributed caching and cache invalidation strategies can be added as traffic grows  
- LLM abstraction supports remote or distributed inference backends  

## Limitations

- TF-IDF retrieval may underperform on semantic queries compared to dense embeddings (e.g., sentence transformers) when the lightweight default provider is used
- FAISS is single-node and not distributed
- Cache invalidation is not implemented for dynamic document updates

## Future Improvements

- Add distributed caching and cache invalidation strategies  
- Implement async/background processing for large-scale ingestion  
- Integrate managed vector databases (e.g., Pinecone, Weaviate)  
- Add CI/CD pipeline for automated testing and deployment  
- Build a frontend interface for interactive querying  

## What This Project Demonstrates

- Backend and distributed system design for AI applications
- Production-style RAG system architecture
- Performance optimization through caching and latency reduction
- API design and service modularization
- LLM integration across local and cloud providers
- Trade-off analysis for retrieval methods and deployment constraints
- Observability and metrics-driven engineering