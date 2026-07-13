"""Registry for file handlers."""
from .base import FileHandler


class HandlerRegistry:
    """Registry for file handlers with extension-based lookup."""

    def __init__(self):
        self._handlers: dict[str, FileHandler] = {}

    def register(self, handler: FileHandler) -> None:
        """Register a handler for its supported extensions."""
        for ext in handler.supported_extensions():
            self._handlers[ext.lower().lstrip(".")] = handler

    def get_handler(self, file_path: str) -> FileHandler | None:
        """Get handler for a file path based on extension."""
        ext = file_path.rsplit(".", 1)[-1].lower()
        return self._handlers.get(ext)

    def list_supported(self) -> list[str]:
        """List all supported extensions."""
        return list(self._handlers.keys())

    def has_handler(self, extension: str) -> bool:
        """Check if a handler exists for an extension."""
        return extension.lower().lstrip(".") in self._handlers
