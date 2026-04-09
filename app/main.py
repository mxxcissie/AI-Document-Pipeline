from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="AI Document Pipeline",
    description="Backend service for semantic document search and question answering.",
    version="0.1.0"
)

app.include_router(router)