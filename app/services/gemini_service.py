import time

from google import genai

from app.core.config import GEMINI_API_KEY, GEMINI_MODEL
from app.services.llm_service import LLMService


class GeminiService(LLMService):
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def generate(self, prompt: str) -> str:
        last_error = None

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )

                output = getattr(response, "text", None)
                if not output or not output.strip():
                    raise RuntimeError("Empty response from Gemini")

                return output.strip()

            except Exception as e:
                last_error = e

                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise RuntimeError("Gemini request failed after retries") from e

        raise RuntimeError(f"Gemini request failed: {last_error}")