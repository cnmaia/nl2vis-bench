"""Tests for data models."""
import pytest
from nl2vis_bench.models import ColumnInfo, DatasetSchema, QueryResult, VizSpec, DataProfile


def test_column_info_creation():
    col = ColumnInfo(
        name="temperature",
        dtype="float",
        min_value=18.5,
        max_value=38.2,
        mean_value=27.1,
        null_count=0,
        sample_values=["25.3", "26.1", "27.0"]
    )
    assert col.name == "temperature"
    assert col.dtype == "float"
    assert col.min_value == 18.5


def test_dataset_schema_creation():
    col = ColumnInfo(name="temp", dtype="float")
    schema = DatasetSchema(
        name="test_dataset",
        file_type="xlsx",
        columns=[col],
        file_count=3
    )
    assert schema.name == "test_dataset"
    assert len(schema.columns) == 1


def test_query_result_creation():
    result = QueryResult(
        query="df.mean()",
        columns_used=["temp"],
        explanation="Calculate mean",
        llm_provider="gemini",
        latency_ms=100.0
    )
    assert result.query == "df.mean()"
    assert result.latency_ms == 100.0


def test_viz_spec_creation():
    spec = VizSpec(type="line", x="date", y="temp")
    assert spec.type == "line"
    assert spec.y == "temp"


def test_viz_spec_histogram_no_y():
    spec = VizSpec(type="histogram", x="temp")
    assert spec.y is None


def test_data_profile_creation():
    profile = DataProfile(
        has_datetime_column=True,
        has_numeric_columns=True,
        numeric_column_count=2,
        row_count=100,
        datetime_column="date",
        numeric_columns=["temp", "humidity"]
    )
    assert profile.has_datetime_column is True
    assert profile.numeric_column_count == 2
