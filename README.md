
# AI Document Pipeline

A backend project for semantic document retrieval and question answering using FastAPI, embeddings, vector search, and LLM integration.

## Scope

- FastAPI backend setup
- Health check endpoint
- Placeholder query endpoint
- Project structure initialization

## Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoints

- GET /health
- POST /query

## Docs

Swagger UI: http://127.0.0.1:8000/docs