"""Histogram plugin."""
import time
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from nl2vis_bench.models import VizSpec, RenderResult


class HistogramPlugin:
    """Plugin for rendering histograms."""

    @property
    def name(self) -> str:
        return "histogram"

    @property
    def supported_types(self) -> list[str]:
        return ["histogram"]

    def render(
        self,
        df: pd.DataFrame,
        spec: VizSpec,
        output_path: str | None = None,
    ) -> RenderResult:
        """Render a histogram."""
        start_time = time.time()

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            data = df[spec.x]

            # Calculate appropriate number of bins
            n_bins = min(50, max(10, len(data) // 10))

            ax.hist(data, bins=n_bins, edgecolor='black', alpha=0.7)

            ax.set_xlabel(spec.x)
            ax.set_ylabel("Frequency")

            if spec.title:
                ax.set_title(spec.title)
            else:
                ax.set_title(f"Distribution of {spec.x}")

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
                data_points_rendered=len(data),
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
