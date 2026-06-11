from .client import LLMClient


class DeepSeekLLM:
    def __init__(self, client: LLMClient):
        self.client = client

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
    ) -> str:
        return self.client.chat(messages, temperature=temperature)
