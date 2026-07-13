"""Scatter plot plugin."""
import time
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from nl2vis_bench.models import VizSpec, RenderResult


class ScatterPlugin:
    """Plugin for rendering scatter plots."""

    @property
    def name(self) -> str:
        return "scatter"

    @property
    def supported_types(self) -> list[str]:
        return ["scatter"]

    def render(
        self,
        df: pd.DataFrame,
        spec: VizSpec,
        output_path: str | None = None,
    ) -> RenderResult:
        """Render a scatter plot."""
        start_time = time.time()

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            x_data = df[spec.x]
            y_data = df[spec.y] if spec.y else df.iloc[:, 1]

            ax.scatter(x_data, y_data, alpha=0.6, edgecolors='black', linewidths=0.5)

            ax.set_xlabel(spec.x)
            ax.set_ylabel(spec.y or "Value")

            if spec.title:
                ax.set_title(spec.title)
            else:
                ax.set_title(f"{spec.x} vs {spec.y}")

            plt.tight_layout()

            # Save to buffer or file
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                content = output_path
                fmt = output_path.split('.')[-1]
            else:
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                content = buffer.getvalue()
                fmt = "png"

            plt.close(fig)

            render_time = (time.time() - start_time) * 1000

            return RenderResult(
                success=True,
                format=fmt,
                content=content,
                file_path=output_path,
                render_time_ms=render_time,
                plugin_used=self.name,
                data_points_rendered=len(df),
            )

        except Exception as e:
            plt.close('all')
            render_time = (time.time() - start_time) * 1000
            return RenderResult(
                success=False,
                render_time_ms=render_time,
                error_message=str(e),
                plugin_used=self.name,
            )
