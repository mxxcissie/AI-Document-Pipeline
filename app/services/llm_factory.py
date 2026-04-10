from app.core.config import LLM_PROVIDER
from app.services.ollama_service import OllamaService
from app.services.gemini_service import GeminiService


def get_llm_service():
    if LLM_PROVIDER == "ollama":
        return OllamaService()
    if LLM_PROVIDER == "gemini":
        return GeminiService()

    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")