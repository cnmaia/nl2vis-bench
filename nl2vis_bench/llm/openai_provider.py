"""OpenAI LLM provider."""
from openai import OpenAI


class OpenAIProvider:
    """Provider for OpenAI API (GPT-4o, etc.)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def complete(self, prompt: str) -> str:
        """Send prompt to OpenAI and return response."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content

    def model_name(self) -> str:
        """Return model identifier."""
        return f"openai/{self.model}"
