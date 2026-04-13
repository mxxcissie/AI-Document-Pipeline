import logging

from fastapi import FastAPI

from app.api.routes import router


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="AI Document Pipeline",
    version="1.0.0",
    description="RAG-based document question answering system",
)


app.include_router(router, prefix="/api")


@app.on_event("startup")
def startup_event():
    logger.info("AI Document Pipeline API started successfully.")