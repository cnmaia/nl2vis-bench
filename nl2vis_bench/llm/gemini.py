"""Google Gemini LLM provider."""
import google.generativeai as genai


class GeminiProvider:
    """Provider for Google Gemini API."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(model)

    def complete(self, prompt: str) -> str:
        """Send prompt to Gemini and return response."""
        response = self._client.generate_content(prompt)
        return response.text

    def model_name(self) -> str:
        """Return model identifier."""
        return f"gemini/{self.model}"
