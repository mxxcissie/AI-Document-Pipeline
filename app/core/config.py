import os
from dotenv import load_dotenv

env_file = ".env.local"
if os.getenv("RUN_ENV") == "docker":
    env_file = ".env.docker"

load_dotenv(env_file)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")