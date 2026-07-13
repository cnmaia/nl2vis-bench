"""Base protocol for file handlers."""
from typing import Protocol, runtime_checkable
import pandas as pd
from nl2vis_bench.models import DatasetSchema


@runtime_checkable
class FileHandler(Protocol):
    """Protocol for file handlers that index and load data files."""

    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions (without dot)."""
        ...

    def can_handle(self, file_path: str) -> bool:
        """Check if this handler can process the given file."""
        ...

    def extract_schema(self, file_path: str) -> DatasetSchema:
        """Extract schema/metadata from a file."""
        ...

    def load(self, file_path: str) -> pd.DataFrame:
        """Load file contents as DataFrame."""
        ...
