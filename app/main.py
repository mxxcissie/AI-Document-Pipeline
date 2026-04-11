from fastapi import FastAPI
from app.api.routes import router


app = FastAPI(
    title="AI Document Pipeline",
    version="1.0.0",
    description="RAG-based document question answering system",
)


app.include_router(router, prefix="/api")


@app.on_event("startup")
def startup_event():
    print("[STARTUP] AI Document Pipeline API started successfully.")