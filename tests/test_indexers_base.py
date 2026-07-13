"""Tests for indexer base and registry."""
import pytest
import pandas as pd
from nl2vis_bench.indexers.base import FileHandler
from nl2vis_bench.indexers.registry import HandlerRegistry
from nl2vis_bench.models import DatasetSchema


class MockHandler:
    """Mock handler for testing."""

    def supported_extensions(self) -> list[str]:
        return ["mock", "test"]

    def can_handle(self, file_path: str) -> bool:
        return file_path.endswith((".mock", ".test"))

    def extract_schema(self, file_path: str) -> DatasetSchema:
        return DatasetSchema(name="mock", file_type="mock", columns=[])

    def load(self, file_path: str) -> pd.DataFrame:
        return pd.DataFrame({"col": [1, 2, 3]})


def test_registry_register_handler():
    registry = HandlerRegistry()
    handler = MockHandler()
    registry.register(handler)

    assert "mock" in registry.list_supported()
    assert "test" in registry.list_supported()


def test_registry_get_handler():
    registry = HandlerRegistry()
    handler = MockHandler()
    registry.register(handler)

    retrieved = registry.get_handler("data.mock")
    assert retrieved is not None
    assert retrieved.supported_extensions() == ["mock", "test"]


def test_registry_get_handler_not_found():
    registry = HandlerRegistry()
    result = registry.get_handler("data.unknown")
    assert result is None


def test_registry_list_supported():
    registry = HandlerRegistry()
    handler = MockHandler()
    registry.register(handler)

    supported = registry.list_supported()
    assert len(supported) == 2
