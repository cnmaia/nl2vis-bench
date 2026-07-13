"""Tests for NL to Query translator."""
import pytest
from unittest.mock import Mock
from nl2vis_bench.models import ColumnInfo, DatasetSchema, QueryResult
from nl2vis_bench.translation.translator import NLTranslator
from nl2vis_bench.translation.prompts import build_translation_prompt
from nl2vis_bench.translation.validator import QueryValidator


@pytest.fixture
def sample_schema():
    return DatasetSchema(
        name="test_data",
        file_type="xlsx",
        columns=[
            ColumnInfo(name="TIMESTAMP", dtype="datetime"),
            ColumnInfo(name="temperature", dtype="float", min_value=20.0, max_value=35.0),
            ColumnInfo(name="humidity", dtype="float", min_value=50.0, max_value=90.0),
        ]
    )


def test_build_prompt_includes_schema(sample_schema):
    prompt = build_translation_prompt("What is the average temperature?", sample_schema)

    assert "temperature" in prompt
    assert "humidity" in prompt
    assert "TIMESTAMP" in prompt
    assert "What is the average temperature?" in prompt


def test_validator_accepts_valid_columns(sample_schema):
    validator = QueryValidator(sample_schema)
    result = validator.validate_columns(["temperature", "humidity"])
    assert result.is_valid


def test_validator_rejects_invalid_columns(sample_schema):
    validator = QueryValidator(sample_schema)
    result = validator.validate_columns(["nonexistent"])
    assert not result.is_valid
    assert "nonexistent" in result.error


def test_translator_parse_response():
    translator = NLTranslator(llm_provider=Mock())

    response = '{"query": "df.mean()", "columns_used": ["temp"], "explanation": "Calculate mean"}'
    result = translator._parse_response(response)

    assert result["query"] == "df.mean()"
    assert result["columns_used"] == ["temp"]


def test_translator_translate(sample_schema):
    mock_llm = Mock()
    mock_llm.complete.return_value = '''{"query": "df['temperature'].mean()", "columns_used": ["temperature"], "explanation": "Average temperature"}'''
    mock_llm.model_name.return_value = "test/model"

    translator = NLTranslator(llm_provider=mock_llm)
    result = translator.translate("What is the average temperature?", sample_schema)

    assert isinstance(result, QueryResult)
    assert "temperature" in result.query
    assert result.columns_used == ["temperature"]


def test_format_columns_info_minimal():
    """Test minimal column formatting (no descriptions)."""
    from nl2vis_bench.translation.prompts import format_columns_info_minimal
    from nl2vis_bench.models import ColumnInfo

    columns = [
        ColumnInfo(
            name="temperature",
            dtype="float",
            description="Air temperature in Celsius",
            min_value=20.0,
            max_value=35.0,
            mean_value=27.5,
        ),
    ]

    result = format_columns_info_minimal(columns)

    # Should NOT include description, stats, or samples
    assert "temperature" in result
    assert "float" in result
    assert "Celsius" not in result
    assert "20.0" not in result
    assert "Range" not in result


def test_translator_translate_without_enrichment(sample_schema):
    """Test translation without semantic enrichment."""
    mock_llm = Mock()
    mock_llm.complete.return_value = '{"query": "df[\'temperature\'].mean()", "columns_used": ["temperature"], "explanation": "Average"}'
    mock_llm.model_name.return_value = "test/model"

    translator = NLTranslator(llm_provider=mock_llm)
    result = translator.translate(
        "What is the average temperature?",
        sample_schema,
        use_enrichment=False
    )

    # Verify the prompt sent to LLM does NOT contain enrichment
    call_args = mock_llm.complete.call_args[0][0]
    assert "temperature" in call_args  # column name present
    assert "Range:" not in call_args   # no stats
