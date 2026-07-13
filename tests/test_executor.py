"""Tests for Query Executor."""
import pytest
import pandas as pd
import numpy as np
from nl2vis_bench.execution import QueryExecutor
from nl2vis_bench.models import ExecutionResult, DataProfile


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
        'temperature': [25.0, 26.5, 27.0, 26.0, 25.5],
        'humidity': [60.0, 65.0, 70.0, 68.0, 62.0],
        'location': ['A', 'A', 'B', 'B', 'A'],
    })


def test_executor_simple_query(sample_df):
    executor = QueryExecutor()
    result = executor.execute(sample_df, "df['temperature'].mean()")

    assert result.success
    assert result.data is not None
    assert result.profile is not None
    assert result.execution_time_ms > 0


def test_executor_returns_dataframe(sample_df):
    executor = QueryExecutor()
    result = executor.execute(sample_df, "df[['temperature', 'humidity']]")

    assert result.success
    assert isinstance(result.data, pd.DataFrame)
    assert 'temperature' in result.data.columns
    assert 'humidity' in result.data.columns
    assert len(result.data) == 5


def test_executor_extracts_profile(sample_df):
    executor = QueryExecutor()
    result = executor.execute(sample_df, "df")

    assert result.success
    profile = result.profile

    assert profile.has_datetime_column
    assert profile.datetime_column == 'timestamp'
    assert profile.has_numeric_columns
    assert profile.numeric_column_count == 2
    assert 'temperature' in profile.numeric_columns
    assert 'humidity' in profile.numeric_columns
    assert 'location' in profile.categorical_columns
    assert profile.row_count == 5


def test_executor_handles_groupby(sample_df):
    executor = QueryExecutor()
    result = executor.execute(sample_df, "df.groupby('location')['temperature'].mean()")

    assert result.success
    assert result.data is not None
    assert len(result.data) == 2  # Two locations: A and B


def test_executor_handles_invalid_query(sample_df):
    executor = QueryExecutor()
    result = executor.execute(sample_df, "df['nonexistent']")

    assert not result.success
    assert result.error_message is not None
    assert 'nonexistent' in result.error_message.lower() or 'key' in result.error_message.lower()


def test_executor_sandbox_blocks_builtins(sample_df):
    executor = QueryExecutor()
    result = executor.execute(sample_df, "open('/etc/passwd')")

    assert not result.success
    assert result.error_message is not None


def test_executor_converts_scalar_to_dataframe(sample_df):
    executor = QueryExecutor()
    result = executor.execute(sample_df, "df['temperature'].max()")

    assert result.success
    assert isinstance(result.data, pd.DataFrame)
    assert 'result' in result.data.columns


def test_executor_converts_series_to_dataframe(sample_df):
    executor = QueryExecutor()
    result = executor.execute(sample_df, "df['temperature']")

    assert result.success
    assert isinstance(result.data, pd.DataFrame)


def test_executor_handles_period_index():
    """Verify groupby with to_period produces valid datetime output."""
    df = pd.DataFrame({
        'time': pd.date_range('2002-01-01', periods=100, freq='D'),
        'value': range(100)
    })

    executor = QueryExecutor()
    # This query creates PeriodIndex
    result = executor.execute(df, "df.groupby(df['time'].dt.to_period('M'))['value'].mean()")

    assert result.success, f"Execution failed: {result.error_message}"
    assert result.profile is not None
    # Verify the index was converted to datetime
    assert result.profile.has_datetime_column, "Should detect datetime after Period conversion"
