from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="AI Document Pipeline",
    version="0.1.0"
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
def startup_event():
    print("AI Document Pipeline API started successfully.")