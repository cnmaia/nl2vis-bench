"""Visualization plugins for rendering charts."""
from .base import VizPlugin
from .registry import PluginRegistry
from .line import LineChartPlugin
from .histogram import HistogramPlugin
from .scatter import ScatterPlugin

__all__ = [
    "VizPlugin",
    "PluginRegistry",
    "LineChartPlugin",
    "HistogramPlugin",
    "ScatterPlugin",
]
