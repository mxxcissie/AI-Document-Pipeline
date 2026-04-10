# AI Document Pipeline

A backend RAG-based document QA system built with FastAPI, local LLM inference, semantic chunk retrieval, and relevance-filtered FAISS search.

This project demonstrates backend system design for integrating LLM-based services using a clean, extensible architecture that supports both local and cloud-based models.

## Overview

This project builds an end-to-end pipeline for:

- document ingestion
- text preprocessing and chunking
- LLM-based query answering
- extensible model integration

It is designed to evolve into a Retrieval-Augmented Generation (RAG) system.

## Architecture

```text
Client Request
      ↓
FastAPI API Layer
      ↓
LLM Service Layer (abstraction)
      ↓
LLM Provider (Ollama / future cloud API)
      ↓
Response
```

### Document pipeline:

```text
Documents → Load → Clean → Chunk → (Next: Embedding → Retrieval)
```

## Tech Stack

- Backend: FastAPI
- Language: Python
- LLM Runtime: Ollama
- API Server: Uvicorn
- Config Management: python-dotenv

## Project Structure

```text
app/
├── main.py
├── api/
│   └── routes.py
├── core/
│   └── config.py
├── services/
│   ├── llm_service.py
│   ├── ollama_service.py
│   └── llm_factory.py
├── pipeline/
│   ├── document_loader.py
│   └── text_chunker.py
data/
└── sample_docs/
```

## Development Progress

### Backend Setup

- Initialized FastAPI backend
- Created /api/health and /api/query endpoints
- Set up project structure and routing
- Added Swagger API documentation

### LLM Integration

- Integrated LLM via Ollama local API
- Implemented LLM service abstraction layer
- Added provider-based architecture (llm_factory)
- Connected /api/query to real model responses
- Configured environment-based model selection

### Document Pipeline (Ingestion + Chunking)

- Implemented document loader for .txt files
- Built text chunking with overlap
- Created preprocessing pipeline for documents
- Added /api/documents/chunks endpoint for preview

### Embedding

- Added embedding generation using sentence-transformers
- Integrated FAISS vector database for similarity search
- Built semantic search pipeline for document retrieval
- Added `/api/search` endpoint for querying relevant document chunks

### RAG

- Combined semantic retrieval and LLM generation into a RAG pipeline
- Built a context assembly step using top retrieved document chunks
- Added `/api/rag-query` endpoint for grounded question answering
- Returned retrieved sources alongside generated responses

## Improved retrieval quality

- Added structured request schemas for search and RAG endpoints
- Returned retrieval distance scores from FAISS results
- Added relevance thresholding to reduce low-quality retrieval matches
- Improved RAG behavior for unsupported questions by returning empty sources when no relevant context is found
- Refactored vector store initialization for cleaner startup logic

## Configuration

Create a .env file:

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b


## How to Run

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Start Ollama
```bash
ollama run qwen2.5:3b
```
(Exit after verifying it works — Ollama runs in background.)

3. Start backend
```bash
uvicorn app.main:app --reload
```

4. Open API docs
http://127.0.0.1:8000/docs

## API Endpoints

Health Check
`GET /api/health`

Query LLM
`POST /api/query`
```json
{
  "question": "What is a vector database?"
}
```

Preview Document Chunks
`GET /api/documents/chunks`

## Design Highlights

- Separation of concerns
    - API layer vs LLM vs pipeline
- Provider-agnostic LLM architecture
    - supports future integration with OpenAI or Google
- Config-driven design
    - model and provider controlled via .env
- Extensible pipeline
    - ready for embeddings and vector search

## Next Steps

- Add embeddings for document chunks
- Integrate vector database (FAISS / Chroma)
- Implement semantic retrieval (RAG)
- Add caching and performance optimization
- Deploy backend service

## What This Project Demonstrates

- Backend API design using FastAPI
- Service-oriented architecture
- Integration of LLM systems into production-style backend
- Data pipeline design (ingestion → preprocessing → chunking)
- Extensibility for modern AI systems