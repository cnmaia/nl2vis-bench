"""Base protocol for visualization plugins."""
from typing import Protocol
import pandas as pd
from nl2vis_bench.models import VizSpec, RenderResult


class VizPlugin(Protocol):
    """Protocol for visualization plugins."""

    @property
    def name(self) -> str:
        """Plugin name."""
        ...

    @property
    def supported_types(self) -> list[str]:
        """List of visualization types this plugin supports."""
        ...

    def render(
        self,
        df: pd.DataFrame,
        spec: VizSpec,
        output_path: str | None = None,
    ) -> RenderResult:
        """Render the visualization."""
        ...
