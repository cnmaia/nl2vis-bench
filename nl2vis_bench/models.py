"""Data models for scientific data catalogs-Vis."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ColumnInfo:
    """Metadata about a column in a dataset."""
    name: str
    dtype: str  # float, int, datetime, string
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    null_count: int = 0
    unique_count: int | None = None
    sample_values: list[str] = field(default_factory=list)
    description: str | None = None


@dataclass
class DatasetSchema:
    """Schema of an indexed dataset."""
    name: str
    file_type: str  # xlsx, csv, dat, nc
    columns: list[ColumnInfo]
    file_count: int = 1
    file_paths: list[str] = field(default_factory=list)
    handler_used: str = ""
    description: str | None = None
    category: str | None = None
    sample_file_path: str | None = None  # MinIO path to sample JSON
    sample_data: str | None = None  # Loaded sample data (JSON string)


@dataclass
class QueryResult:
    """Result of NL to Query translation."""
    query: str
    columns_used: list[str]
    explanation: str
    llm_provider: str
    latency_ms: float
    raw_response: str = ""
    retry_count: int = 0


@dataclass
class VizSpec:
    """Specification for visualization."""
    type: Literal["line", "histogram", "scatter"]
    x: str
    y: str | None = None
    title: str | None = None
    subtitle: str | None = None  # Brief explanation for end users
    x_label: str | None = None   # User-friendly label for x-axis
    y_label: str | None = None   # User-friendly label for y-axis
    selected_by: Literal["heuristic", "llm"] = "heuristic"
    reasoning: str | None = None


@dataclass
class DataProfile:
    """Profile of query result data."""
    has_datetime_column: bool
    has_numeric_columns: bool
    numeric_column_count: int
    row_count: int
    datetime_column: str | None = None
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Result of query execution."""
    success: bool
    data: "pd.DataFrame | None" = None
    profile: DataProfile | None = None
    execution_time_ms: float = 0.0
    error_message: str | None = None


@dataclass
class RenderResult:
    """Result of visualization rendering."""
    success: bool
    format: Literal["png", "html", "svg"] = "png"
    content: bytes | str = b""
    file_path: str | None = None
    render_time_ms: float = 0.0
    error_message: str | None = None
    plugin_used: str = ""
    data_points_rendered: int = 0
