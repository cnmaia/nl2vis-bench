"""Line chart plugin."""
import time
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from nl2vis_bench.models import VizSpec, RenderResult


class LineChartPlugin:
    """Plugin for rendering line charts."""

    @property
    def name(self) -> str:
        return "line_chart"

    @property
    def supported_types(self) -> list[str]:
        return ["line"]

    def render(
        self,
        df: pd.DataFrame,
        spec: VizSpec,
        output_path: str | None = None,
    ) -> RenderResult:
        """Render a line chart."""
        start_time = time.time()

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            x_data = df[spec.x]
            y_data = df[spec.y] if spec.y else df.iloc[:, 0]

            ax.plot(x_data, y_data, marker='o', markersize=3, linewidth=1, color='#2563eb')

            # Use user-friendly labels if available, otherwise column names
            ax.set_xlabel(spec.x_label or spec.x, fontsize=11)
            ax.set_ylabel(spec.y_label or spec.y or "Value", fontsize=11)

            # Title and subtitle for context
            if spec.title:
                if spec.subtitle:
                    ax.set_title(f"{spec.title}\n{spec.subtitle}", fontsize=12, pad=10)
                else:
                    ax.set_title(spec.title, fontsize=12)

            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
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
