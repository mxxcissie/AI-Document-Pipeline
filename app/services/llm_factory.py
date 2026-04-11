from app.core.config import LLM_PROVIDER
from app.services.ollama_service import OllamaService
from app.services.gemini_service import GeminiService


def get_llm_service():
    provider = (LLM_PROVIDER or "").lower()

    services = {
        "ollama": OllamaService,
        "gemini": GeminiService,
    }

    if provider not in services:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

    return services[provider]()