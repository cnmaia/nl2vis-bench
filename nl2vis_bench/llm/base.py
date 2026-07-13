"""Base protocol for LLM providers."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def complete(self, prompt: str) -> str:
        """Send prompt and return completion."""
        ...

    def model_name(self) -> str:
        """Return the model identifier."""
        ...
