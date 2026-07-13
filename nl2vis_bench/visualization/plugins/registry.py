"""Registry for visualization plugins."""
from .base import VizPlugin


class PluginRegistry:
    """Registry for visualization plugins."""

    def __init__(self):
        self._plugins: dict[str, VizPlugin] = {}

    def register(self, plugin: VizPlugin) -> None:
        """Register a plugin for its supported types."""
        for viz_type in plugin.supported_types:
            self._plugins[viz_type] = plugin

    def get(self, viz_type: str) -> VizPlugin | None:
        """Get plugin for a visualization type."""
        return self._plugins.get(viz_type)

    def supported_types(self) -> list[str]:
        """List all supported visualization types."""
        return list(self._plugins.keys())

    @classmethod
    def default(cls) -> "PluginRegistry":
        """Create registry with all default plugins."""
        from .line import LineChartPlugin
        from .histogram import HistogramPlugin
        from .scatter import ScatterPlugin

        registry = cls()
        registry.register(LineChartPlugin())
        registry.register(HistogramPlugin())
        registry.register(ScatterPlugin())
        return registry
