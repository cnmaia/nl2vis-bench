"""Tests for Visualization Plugins."""
import pytest
import pandas as pd
import tempfile
import os
from nl2vis_bench.visualization.plugins import (
    PluginRegistry,
    LineChartPlugin,
    HistogramPlugin,
    ScatterPlugin,
)
from nl2vis_bench.models import VizSpec


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=20, freq='h'),
        'temperature': [25 + i * 0.5 for i in range(20)],
        'humidity': [60 + i * 0.3 for i in range(20)],
    })


@pytest.fixture
def default_registry():
    return PluginRegistry.default()


def test_registry_has_all_types(default_registry):
    types = default_registry.supported_types()
    assert "line" in types
    assert "histogram" in types
    assert "scatter" in types


def test_registry_get_plugin(default_registry):
    line_plugin = default_registry.get("line")
    assert line_plugin is not None
    assert line_plugin.name == "line_chart"


def test_line_chart_render(sample_df):
    plugin = LineChartPlugin()
    spec = VizSpec(type="line", x="timestamp", y="temperature")

    result = plugin.render(sample_df, spec)

    assert result.success
    assert result.format == "png"
    assert len(result.content) > 0
    assert result.data_points_rendered == 20
    assert result.plugin_used == "line_chart"


def test_line_chart_render_to_file(sample_df):
    plugin = LineChartPlugin()
    spec = VizSpec(type="line", x="timestamp", y="temperature", title="Test Chart")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "chart.png")
        result = plugin.render(sample_df, spec, output_path=output_path)

        assert result.success
        assert os.path.exists(output_path)
        assert result.file_path == output_path


def test_histogram_render(sample_df):
    plugin = HistogramPlugin()
    spec = VizSpec(type="histogram", x="temperature")

    result = plugin.render(sample_df, spec)

    assert result.success
    assert result.format == "png"
    assert len(result.content) > 0
    assert result.plugin_used == "histogram"


def test_scatter_render(sample_df):
    plugin = ScatterPlugin()
    spec = VizSpec(type="scatter", x="temperature", y="humidity")

    result = plugin.render(sample_df, spec)

    assert result.success
    assert result.format == "png"
    assert len(result.content) > 0
    assert result.data_points_rendered == 20
    assert result.plugin_used == "scatter"


def test_plugin_handles_invalid_column(sample_df):
    plugin = LineChartPlugin()
    spec = VizSpec(type="line", x="nonexistent", y="temperature")

    result = plugin.render(sample_df, spec)

    assert not result.success
    assert result.error_message is not None


def test_registry_default_creates_fresh_instance():
    registry1 = PluginRegistry.default()
    registry2 = PluginRegistry.default()

    assert registry1 is not registry2
    assert registry1.supported_types() == registry2.supported_types()
