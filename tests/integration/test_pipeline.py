"""Integration tests for the full NL2Vis pipeline."""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock
import pandas as pd

from nl2vis_bench.catalog import CatalogRepository
from nl2vis_bench.indexers import HandlerRegistry, XLSXHandler, CSVHandler
from nl2vis_bench.translation import NLTranslator
from nl2vis_bench.execution import QueryExecutor
from nl2vis_bench.visualization import VizSelector, PluginRegistry
from nl2vis_bench.models import DatasetSchema


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a temporary CSV file with sample data."""
    content = """timestamp,temperature,humidity,location
2024-01-01 00:00:00,25.0,60.0,A
2024-01-01 01:00:00,25.5,62.0,A
2024-01-01 02:00:00,26.0,65.0,B
2024-01-01 03:00:00,26.5,68.0,B
2024-01-01 04:00:00,27.0,70.0,A
2024-01-01 05:00:00,26.0,67.0,A
2024-01-01 06:00:00,25.0,63.0,B
2024-01-01 07:00:00,24.5,60.0,B
2024-01-01 08:00:00,25.0,58.0,A
2024-01-01 09:00:00,26.0,55.0,A
"""
    file_path = tmp_path / "sample_data.csv"
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def temp_catalog(tmp_path):
    """Create a temporary catalog database."""
    db_path = tmp_path / "catalog.db"
    return CatalogRepository(str(db_path))


@pytest.fixture
def handler_registry():
    """Create handler registry with all handlers."""
    registry = HandlerRegistry()
    registry.register(XLSXHandler())
    registry.register(CSVHandler())
    return registry


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    mock = Mock()
    mock.model_name.return_value = "mock/test-model"
    return mock


class TestIndexingPipeline:
    """Tests for the indexing pipeline."""

    def test_index_csv_file(self, sample_csv_file, handler_registry, temp_catalog):
        """Test indexing a CSV file end-to-end."""
        # Get handler
        handler = handler_registry.get_handler(sample_csv_file)
        assert handler is not None

        # Extract schema
        schema = handler.extract_schema(sample_csv_file)
        assert schema.name == Path(sample_csv_file).stem
        assert len(schema.columns) == 4
        assert schema.file_type == "csv"

        # Save to catalog
        schema.file_paths = [sample_csv_file]
        temp_catalog.save_dataset(schema)

        # Retrieve from catalog
        retrieved = temp_catalog.get_dataset(schema.name)
        assert retrieved is not None
        assert len(retrieved.columns) == 4
        assert retrieved.file_paths == [sample_csv_file]


class TestTranslationPipeline:
    """Tests for the translation pipeline."""

    def test_translate_mean_query(self, sample_csv_file, handler_registry, mock_llm):
        """Test translating a simple mean query."""
        # Setup
        handler = handler_registry.get_handler(sample_csv_file)
        schema = handler.extract_schema(sample_csv_file)

        # Configure mock response
        mock_llm.complete.return_value = '''{"query": "df['temperature'].mean()", "columns_used": ["temperature"], "explanation": "Calculate average temperature"}'''

        # Translate
        translator = NLTranslator(llm_provider=mock_llm)
        result = translator.translate("What is the average temperature?", schema)

        assert result.query == "df['temperature'].mean()"
        assert result.columns_used == ["temperature"]
        assert result.llm_provider == "mock/test-model"


class TestExecutionPipeline:
    """Tests for the execution pipeline."""

    def test_execute_mean_query(self, sample_csv_file, handler_registry):
        """Test executing a pandas query."""
        # Load data
        handler = handler_registry.get_handler(sample_csv_file)
        df = handler.load(sample_csv_file)

        # Execute query
        executor = QueryExecutor()
        result = executor.execute(df, "df['temperature'].mean()")

        assert result.success
        assert result.data is not None
        # Mean of [25, 25.5, 26, 26.5, 27, 26, 25, 24.5, 25, 26] = 25.65
        assert abs(result.data['result'].iloc[0] - 25.65) < 0.01

    def test_execute_groupby_query(self, sample_csv_file, handler_registry):
        """Test executing a groupby query."""
        handler = handler_registry.get_handler(sample_csv_file)
        df = handler.load(sample_csv_file)

        executor = QueryExecutor()
        result = executor.execute(df, "df.groupby('location')['temperature'].mean()")

        assert result.success
        assert result.profile.row_count == 2  # Two locations


class TestVisualizationPipeline:
    """Tests for the visualization pipeline."""

    def test_select_line_chart_for_timeseries(self, sample_csv_file, handler_registry):
        """Test selecting line chart for time series data."""
        # Load and process data
        handler = handler_registry.get_handler(sample_csv_file)
        df = handler.load(sample_csv_file)

        # Parse timestamp as datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        executor = QueryExecutor()
        result = executor.execute(df, "df[['timestamp', 'temperature']]")

        # Select visualization
        selector = VizSelector()
        viz_spec = selector.select(result.profile, "Show temperature over time")

        assert viz_spec.type == "line"
        assert viz_spec.selected_by == "heuristic"

    def test_render_line_chart(self, sample_csv_file, handler_registry, tmp_path):
        """Test rendering a line chart."""
        # Load and process data
        handler = handler_registry.get_handler(sample_csv_file)
        df = handler.load(sample_csv_file)

        executor = QueryExecutor()
        result = executor.execute(df, "df[['timestamp', 'temperature']]")

        # Select and render
        selector = VizSelector()
        viz_spec = selector.select(result.profile, "Show temperature over time")

        registry = PluginRegistry.default()
        plugin = registry.get(viz_spec.type)

        output_path = str(tmp_path / "chart.png")
        render_result = plugin.render(result.data, viz_spec, output_path)

        assert render_result.success
        assert os.path.exists(output_path)
        assert render_result.data_points_rendered == 10


class TestEndToEndPipeline:
    """Full end-to-end tests."""

    def test_full_pipeline_mean_query(self, sample_csv_file, handler_registry, temp_catalog, mock_llm, tmp_path):
        """Test the complete pipeline from indexing to visualization."""
        # Step 1: Index
        handler = handler_registry.get_handler(sample_csv_file)
        schema = handler.extract_schema(sample_csv_file)
        schema.file_paths = [sample_csv_file]
        temp_catalog.save_dataset(schema)

        # Step 2: Retrieve schema
        retrieved_schema = temp_catalog.get_dataset(schema.name)
        assert retrieved_schema is not None

        # Step 3: Translate (mock LLM)
        mock_llm.complete.return_value = '''{"query": "df[['timestamp', 'temperature']]", "columns_used": ["timestamp", "temperature"], "explanation": "Get temperature over time"}'''
        translator = NLTranslator(llm_provider=mock_llm)
        translation = translator.translate("Show temperature over time", retrieved_schema)

        # Step 4: Execute
        df = handler.load(sample_csv_file)
        executor = QueryExecutor()
        exec_result = executor.execute(df, translation.query)
        assert exec_result.success

        # Step 5: Select visualization
        selector = VizSelector()
        viz_spec = selector.select(exec_result.profile, "Show temperature over time")

        # Step 6: Render
        registry = PluginRegistry.default()
        plugin = registry.get(viz_spec.type)

        output_path = str(tmp_path / "result.png")
        render_result = plugin.render(exec_result.data, viz_spec, output_path)

        assert render_result.success
        assert os.path.exists(output_path)

    def test_full_pipeline_histogram(self, sample_csv_file, handler_registry, mock_llm, tmp_path):
        """Test pipeline for histogram visualization."""
        # Setup
        handler = handler_registry.get_handler(sample_csv_file)
        schema = handler.extract_schema(sample_csv_file)
        df = handler.load(sample_csv_file)

        # Translate
        mock_llm.complete.return_value = '''{"query": "df['temperature']", "columns_used": ["temperature"], "explanation": "Get temperature distribution"}'''
        translator = NLTranslator(llm_provider=mock_llm)
        translation = translator.translate("Show the distribution of temperature", schema)

        # Execute
        executor = QueryExecutor()
        exec_result = executor.execute(df, translation.query)
        assert exec_result.success

        # Select (should be histogram due to keywords)
        selector = VizSelector()
        viz_spec = selector.select(exec_result.profile, "Show the distribution of temperature")
        assert viz_spec.type == "histogram"

        # Render
        registry = PluginRegistry.default()
        plugin = registry.get(viz_spec.type)

        output_path = str(tmp_path / "histogram.png")
        render_result = plugin.render(exec_result.data, viz_spec, output_path)

        assert render_result.success
        assert os.path.exists(output_path)
