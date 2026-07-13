"""Visualization module for chart selection and rendering."""
from .selector import VizSelector
from .plugins import PluginRegistry, LineChartPlugin, HistogramPlugin, ScatterPlugin

__all__ = [
    "VizSelector",
    "PluginRegistry",
    "LineChartPlugin",
    "HistogramPlugin",
    "ScatterPlugin",
]
