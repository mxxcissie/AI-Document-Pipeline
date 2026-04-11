# AI Document Pipeline (RAG System)

A production-style Retrieval-Augmented Generation (RAG) backend system for grounded question answering over custom documents. Built with FastAPI, FAISS, and LLMs (Ollama + Gemini), this system emphasizes reliability, modular design, and performance visibility.

## Overview

This project implements an end-to-end RAG pipeline:

- Ingests and preprocesses documents
- Converts text into embeddings
- Stores embeddings in a vector database (FAISS)
- Retrieves relevant context for a query
- Generates grounded responses using an LLM

Unlike simple LLM wrappers, this system reduces hallucination by grounding responses in retrieved document context and enforcing strict context usage.

## Live Demo (Render Deployment)

Try the interactive API via Swagger UI:
- https://ai-document-pipeline.onrender.com/docs

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
Grounded Answer + Sources + Metrics
```

### Document Pipeline:

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
- Built-in performance metrics (latency + retrieval quality)

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
    "relevant_chunks": 2
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
- **Embeddings:** TF-IDF (lightweight, deployment-friendly) 
- **LLM Providers:** Ollama (local), Gemini (cloud)  
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
│   ├── evaluate_rag.py
│   └── inspect_chunks.py
│
├── tests/                  # Unit tests
│   └── test_pipeline.py
│
├── .env.example            # Environment variable template
├── docker-compose.yml      # Local container orchestration
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── pytest.ini              # Pytest configuration
└── README.md
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

## Configuration

Create a `.env` file for local development:
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

For cloud deployment:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
```

Gemini is used in cloud environments to reduce memory usage by offloading LLM inference to an external API, enabling deployment on low-memory instances.

## Deployment

Deployed on Render:

🔗 https://ai-document-pipeline.onrender.com

Supports environment-based switching between local (Ollama) and cloud (Gemini) LLM providers.

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

These metrics provide visibility into both system performance and retrieval quality.

### Evaluation Script

Run the evaluation script:

```bash
python scripts/evaluate_rag.py
```

The script validates:

- In-scope query accuracy
- Out-of-scope fallback behavior
- Response consistency and grounding

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
  - Separate embedder and retriever components enable flexible experimentation and easy replacement of retrieval strategies
  
## Scaling Considerations

- Stateless FastAPI service enables horizontal scaling
- FAISS can be replaced with managed vector databases (e.g., Pinecone, Weaviate)
- LLM abstraction supports distributed or remote inference
- Architecture is suitable for integration into microservice-based systems

## Future Improvements

- Add Redis caching layer for query optimization
- Implement CI/CD pipeline for automated testing and deployment
- Support PDF and large-scale document ingestion
- Add a frontend interface for interactive querying

## Summary

This project demonstrates:

- End-to-end backend system design
- Integration of LLMs into production-style services
- Retrieval-Augmented Generation (RAG) architecture
- Performance measurement and evaluation
- Cloud deployment and containerization

## What This Project Demonstrates

- Backend API design with FastAPI
- Service-oriented architecture
- Data pipeline design (ingestion → retrieval → generation)
- Applied AI system design
- Production-oriented engineering practices