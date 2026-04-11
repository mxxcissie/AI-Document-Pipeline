import requests

from app.services.llm_service import LLMService
from app.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL


class OllamaService(LLMService):
    def generate(self, prompt: str) -> str:
        url = f"{OLLAMA_BASE_URL}/api/generate"

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            data = response.json()
            output = data.get("response", "").strip()

            if not output:
                raise RuntimeError("Empty response from Ollama")

            return output

        except requests.Timeout:
            raise RuntimeError("Ollama request timed out")

        except requests.RequestException:
            raise RuntimeError("Ollama request failed")