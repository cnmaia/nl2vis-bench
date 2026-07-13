"""CSV file handler."""
import pandas as pd
from pathlib import Path
from nl2vis_bench.models import ColumnInfo, DatasetSchema


class CSVHandler:
    """Handler for CSV files (.csv, .txt)."""

    def supported_extensions(self) -> list[str]:
        return ["csv", "txt"]

    def can_handle(self, file_path: str) -> bool:
        ext = Path(file_path).suffix.lower().lstrip(".")
        return ext in self.supported_extensions()

    def extract_schema(self, file_path: str) -> DatasetSchema:
        """Extract schema from CSV file."""
        df = pd.read_csv(file_path)
        columns = self._extract_columns(df)

        return DatasetSchema(
            name=Path(file_path).stem,
            file_type="csv",
            columns=columns,
            file_count=1,
            file_paths=[file_path],
            handler_used="CSVHandler",
        )

    def load(self, file_path: str) -> pd.DataFrame:
        """Load CSV file as DataFrame."""
        return pd.read_csv(file_path)

    def _extract_columns(self, df: pd.DataFrame) -> list[ColumnInfo]:
        """Extract column metadata from DataFrame."""
        columns = []

        for col_name in df.columns:
            col_data = df[col_name]
            dtype = self._infer_dtype(col_data)

            col_info = ColumnInfo(
                name=str(col_name),
                dtype=dtype,
                null_count=int(col_data.isna().sum()),
                unique_count=int(col_data.nunique()),
            )

            if dtype in ("float", "int"):
                col_info.min_value = float(col_data.min())
                col_info.max_value = float(col_data.max())
                col_info.mean_value = float(col_data.mean())

            samples = col_data.dropna().head(5).astype(str).tolist()
            col_info.sample_values = samples

            columns.append(col_info)

        return columns

    def _infer_dtype(self, series: pd.Series) -> str:
        """Infer simplified dtype from pandas series."""
        pd_dtype = str(series.dtype)

        if "datetime" in pd_dtype:
            return "datetime"
        elif "int" in pd_dtype:
            return "int"
        elif "float" in pd_dtype:
            return "float"
        elif pd_dtype == "object":
            try:
                pd.to_numeric(series.dropna().head(10))
                return "float"
            except (ValueError, TypeError):
                return "string"
        else:
            return "string"
