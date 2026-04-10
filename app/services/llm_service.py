class LLMService:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("LLMService must implement generate()")