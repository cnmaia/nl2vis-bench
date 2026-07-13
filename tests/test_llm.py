"""Tests for LLM providers."""
import pytest
from unittest.mock import Mock, patch
from nl2vis_bench.llm.base import LLMProvider
from nl2vis_bench.llm.gemini import GeminiProvider
from nl2vis_bench.llm.openai_provider import OpenAIProvider


def test_llm_provider_protocol():
    """Test that GeminiProvider implements LLMProvider protocol."""
    provider = GeminiProvider(api_key="test-key")
    assert hasattr(provider, 'complete')
    assert hasattr(provider, 'model_name')


def test_gemini_model_name():
    provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
    assert provider.model_name() == "gemini/gemini-2.5-flash"


@patch('nl2vis_bench.llm.gemini.genai')
def test_gemini_complete(mock_genai):
    """Test Gemini completion with mocked API."""
    mock_model = Mock()
    mock_response = Mock()
    mock_response.text = '{"query": "df.mean()", "columns_used": ["col1"]}'
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    provider = GeminiProvider(api_key="test-key")
    result = provider.complete("Test prompt")

    assert '{"query"' in result
    mock_model.generate_content.assert_called_once()


def test_openai_model_name():
    provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
    assert provider.model_name() == "openai/gpt-4o"


@patch('nl2vis_bench.llm.openai_provider.OpenAI')
def test_openai_complete(mock_openai_class):
    """Test OpenAI completion with mocked API."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"query": "df.sum()"}'))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    provider = OpenAIProvider(api_key="test-key")
    result = provider.complete("Test prompt")

    assert '{"query"' in result
