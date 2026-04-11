import os
from dotenv import load_dotenv


env_file = ".env.local"

if os.getenv("RUN_ENV") == "docker":
    env_file = ".env.docker"

load_dotenv(env_file)


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


if LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required when using Gemini provider")

elif LLM_PROVIDER == "ollama":
    if not OLLAMA_BASE_URL:
        raise ValueError("OLLAMA_BASE_URL is required for Ollama provider")


print(f"[CONFIG] Provider: {LLM_PROVIDER}")