from app.core.config import LLM_PROVIDER
from app.services.ollama_service import OllamaService


def get_llm_service():
    if LLM_PROVIDER == "ollama":
        return OllamaService()

    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")