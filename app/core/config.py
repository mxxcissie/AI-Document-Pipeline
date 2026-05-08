import os
from pathlib import Path
from dotenv import load_dotenv


env_file = ".env.local"

if os.getenv("RUN_ENV") == "docker":
    env_file = ".env.docker"

load_dotenv(env_file)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "sample_docs"
INDEX_DIR = BASE_DIR / "data" / "indexes"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "tfidf")
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
LOCAL_EMBEDDING_MODEL_PATH = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "").strip()
LOCAL_EMBEDDING_OFFLINE_ONLY = os.getenv("LOCAL_EMBEDDING_OFFLINE_ONLY", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
EMBEDDING_API_PROVIDER = os.getenv("EMBEDDING_API_PROVIDER", "")
EMBEDDING_API_MODEL = os.getenv("EMBEDDING_API_MODEL", "")
WEB_LOOKUP_ENABLED = os.getenv("WEB_LOOKUP_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WEB_LOOKUP_URL = os.getenv("WEB_LOOKUP_URL", "https://api.duckduckgo.com/")


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


REDIS_URL = os.getenv("REDIS_URL")  # no default
RAG_CACHE_TTL_SECONDS = int(os.getenv("RAG_CACHE_TTL_SECONDS", "300"))


if LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required when using Gemini provider")

elif LLM_PROVIDER == "ollama":
    if not OLLAMA_BASE_URL:
        raise ValueError("OLLAMA_BASE_URL is required for Ollama provider")


print(f"[CONFIG] Provider: {LLM_PROVIDER}")
print(f"[CONFIG] Env file: {env_file}")
print(f"[CONFIG] Embedding provider: {EMBEDDING_PROVIDER}")
print(f"[CONFIG] External lookup enabled: {WEB_LOOKUP_ENABLED}")

if EMBEDDING_PROVIDER == "local_dense":
    source = LOCAL_EMBEDDING_MODEL_PATH or LOCAL_EMBEDDING_MODEL
    print(f"[CONFIG] Local embedding source: {source}")
    print(f"[CONFIG] Local embedding offline-only: {LOCAL_EMBEDDING_OFFLINE_ONLY}")

if REDIS_URL:
    print(f"[CONFIG] Redis URL: {REDIS_URL}")
    print(f"[CONFIG] RAG cache TTL: {RAG_CACHE_TTL_SECONDS}s")
else:
    print("[CONFIG] Redis disabled (no REDIS_URL provided)")
