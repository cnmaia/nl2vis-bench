"""Tests for Campbell Scientific CR6 handler."""
import pytest
from pathlib import Path
from nl2vis_bench.indexers.campbell_cr6 import CampbellCR6Handler


@pytest.fixture
def cr6_handler():
    return CampbellCR6Handler()


@pytest.fixture
def sample_dat():
    return Path(__file__).parent / "fixtures" / "sample.dat"


def test_supported_extensions(cr6_handler):
    exts = cr6_handler.supported_extensions()
    assert "dat" in exts


def test_can_handle(cr6_handler, sample_dat):
    assert cr6_handler.can_handle(str(sample_dat))
    assert not cr6_handler.can_handle("data.csv")


def test_extract_schema(cr6_handler, sample_dat):
    schema = cr6_handler.extract_schema(str(sample_dat))

    assert schema.file_type == "dat"
    assert len(schema.columns) == 5

    col_names = [c.name for c in schema.columns]
    assert "TIMESTAMP" in col_names
    assert "BattV_Avg" in col_names
    assert "AirTC_Avg" in col_names


def test_extract_schema_stats(cr6_handler, sample_dat):
    schema = cr6_handler.extract_schema(str(sample_dat))

    air_temp = next(c for c in schema.columns if c.name == "AirTC_Avg")
    assert air_temp.dtype == "float"
    assert air_temp.min_value == pytest.approx(24.5, 0.1)
    assert air_temp.max_value == pytest.approx(25.6, 0.1)


def test_load(cr6_handler, sample_dat):
    df = cr6_handler.load(str(sample_dat))

    assert len(df) == 5
    assert "AirTC_Avg" in df.columns
    assert "TIMESTAMP" in df.columns
