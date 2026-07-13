"""Tests for CSV handler."""
import pytest
from pathlib import Path
from nl2vis_bench.indexers.csv import CSVHandler


@pytest.fixture
def csv_handler():
    return CSVHandler()


@pytest.fixture
def sample_csv():
    return Path(__file__).parent / "fixtures" / "sample.csv"


def test_supported_extensions(csv_handler):
    exts = csv_handler.supported_extensions()
    assert "csv" in exts
    assert "txt" in exts


def test_can_handle(csv_handler, sample_csv):
    assert csv_handler.can_handle(str(sample_csv))
    assert not csv_handler.can_handle("data.xlsx")


def test_extract_schema(csv_handler, sample_csv):
    schema = csv_handler.extract_schema(str(sample_csv))

    assert schema.file_type == "csv"
    assert len(schema.columns) == 3

    col_names = [c.name for c in schema.columns]
    assert "date" in col_names
    assert "value" in col_names


def test_load(csv_handler, sample_csv):
    df = csv_handler.load(str(sample_csv))

    assert len(df) == 10
    assert "value" in df.columns
