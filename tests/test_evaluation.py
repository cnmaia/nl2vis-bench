"""Tests for Evaluation module."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock
import pandas as pd

from nl2vis_bench.evaluation import EvaluationRunner
from nl2vis_bench.models import DatasetSchema, ColumnInfo


@pytest.fixture
def sample_schema():
    return DatasetSchema(
        name="test_data",
        file_type="csv",
        columns=[
            ColumnInfo(name="timestamp", dtype="datetime"),
            ColumnInfo(name="temperature", dtype="float", min_value=20.0, max_value=35.0),
            ColumnInfo(name="humidity", dtype="float", min_value=50.0, max_value=90.0),
            ColumnInfo(name="location", dtype="string"),
        ]
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
        'temperature': [25.0, 26.5, 27.0, 26.0, 25.5, 24.0, 23.5, 25.0, 26.0, 27.0],
        'humidity': [60.0, 65.0, 70.0, 68.0, 62.0, 58.0, 55.0, 60.0, 65.0, 70.0],
        'location': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'A', 'A'],
    })


@pytest.fixture
def mini_questions_file(tmp_path):
    """Create a minimal questions file for testing."""
    content = """
version: "1.0"
questions:
  - id: test_001
    question: "What is the average temperature?"
    expected_query_contains: ["mean", "temperature"]
    expected_columns: ["temperature"]
    expected_viz: "histogram"
    category: "aggregation"

  - id: test_002
    question: "Show temperature over time"
    expected_query_contains: ["timestamp", "temperature"]
    expected_columns: ["timestamp", "temperature"]
    expected_viz: "line"
    category: "temporal"
"""
    file_path = tmp_path / "test_questions.yaml"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def mock_llm():
    mock = Mock()
    mock.model_name.return_value = "mock/test-model"
    return mock


def test_evaluation_runner_loads_questions(mock_llm, mini_questions_file):
    runner = EvaluationRunner(
        llm_provider=mock_llm,
        questions_path=mini_questions_file
    )
    assert len(runner.questions) == 2


def test_evaluation_runner_evaluates_questions(mock_llm, mini_questions_file, sample_schema, sample_df):
    # Configure mock to return valid translation
    mock_llm.complete.return_value = '{"query": "df[\'temperature\'].mean()", "columns_used": ["temperature"], "explanation": "Calculate mean temperature"}'

    runner = EvaluationRunner(
        llm_provider=mock_llm,
        questions_path=mini_questions_file
    )

    result = runner.run(sample_schema, sample_df, max_questions=1)

    assert result.total_questions == 1
    assert len(result.question_results) == 1
    assert result.question_results[0].question_id == "test_001"


def test_evaluation_result_metrics(mock_llm, mini_questions_file, sample_schema, sample_df):
    # Configure mock
    mock_llm.complete.return_value = '{"query": "df[\'temperature\'].mean()", "columns_used": ["temperature"], "explanation": "Calculate mean"}'

    runner = EvaluationRunner(
        llm_provider=mock_llm,
        questions_path=mini_questions_file
    )

    result = runner.run(sample_schema, sample_df, max_questions=1)

    # Check metrics are calculated
    assert result.translation_accuracy >= 0
    assert result.execution_accuracy >= 0
    assert result.viz_accuracy >= 0
    assert result.column_accuracy >= 0


def test_evaluation_handles_failed_translation(mock_llm, mini_questions_file, sample_schema, sample_df):
    # Configure mock to return invalid response
    mock_llm.complete.return_value = 'invalid json'

    runner = EvaluationRunner(
        llm_provider=mock_llm,
        questions_path=mini_questions_file
    )

    result = runner.run(sample_schema, sample_df, max_questions=1)

    assert result.total_questions == 1
    # Should handle error gracefully
    assert result.question_results[0].error is not None
