"""Tests for XLSX handler."""
import pytest
from pathlib import Path
from nl2vis_bench.indexers.xlsx import XLSXHandler


@pytest.fixture
def xlsx_handler():
    return XLSXHandler()


@pytest.fixture
def sample_xlsx():
    return Path(__file__).parent / "fixtures" / "sample.xlsx"


def test_supported_extensions(xlsx_handler):
    exts = xlsx_handler.supported_extensions()
    assert "xlsx" in exts
    assert "xls" in exts


def test_can_handle(xlsx_handler, sample_xlsx):
    assert xlsx_handler.can_handle(str(sample_xlsx))
    assert not xlsx_handler.can_handle("data.csv")


def test_extract_schema(xlsx_handler, sample_xlsx):
    schema = xlsx_handler.extract_schema(str(sample_xlsx))

    assert schema.file_type == "xlsx"
    assert len(schema.columns) == 4

    col_names = [c.name for c in schema.columns]
    assert "TIMESTAMP" in col_names
    assert "temperature" in col_names
    assert "humidity" in col_names


def test_extract_schema_column_stats(xlsx_handler, sample_xlsx):
    schema = xlsx_handler.extract_schema(str(sample_xlsx))

    temp_col = next(c for c in schema.columns if c.name == "temperature")
    assert temp_col.dtype == "float"
    assert temp_col.min_value == pytest.approx(24.8, 0.1)
    assert temp_col.max_value == pytest.approx(27.3, 0.1)
    assert temp_col.null_count == 0


def test_load(xlsx_handler, sample_xlsx):
    df = xlsx_handler.load(str(sample_xlsx))

    assert len(df) == 10
    assert "temperature" in df.columns
    assert "humidity" in df.columns
