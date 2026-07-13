"""Campbell Scientific CR6 data logger file handler."""
import pandas as pd
from pathlib import Path
from nl2vis_bench.models import ColumnInfo, DatasetSchema


class CampbellCR6Handler:
    """Handler for Campbell Scientific CR6 data logger files (.dat).

    CR6 .dat files have a specific structure:
    - Line 1: Logger metadata (ignored)
    - Line 2: Column names (header)
    - Line 3: Units
    - Line 4: Aggregation types (Avg, Smp, etc.)
    - Line 5+: Data
    """

    def supported_extensions(self) -> list[str]:
        return ["dat"]

    def can_handle(self, file_path: str) -> bool:
        ext = Path(file_path).suffix.lower().lstrip(".")
        return ext in self.supported_extensions()

    def extract_schema(self, file_path: str) -> DatasetSchema:
        """Extract schema from CR6 .dat file."""
        df = self.load(file_path)
        columns = self._extract_columns(df)

        return DatasetSchema(
            name=Path(file_path).stem,
            file_type="dat",
            columns=columns,
            file_count=1,
            file_paths=[file_path],
            handler_used="CampbellCR6Handler",
        )

    def load(self, file_path: str) -> pd.DataFrame:
        """Load CR6 .dat file as DataFrame.

        Skips the first 4 rows (metadata, headers, units, aggregation)
        and uses the second line as column names.
        """
        # Read header row (line 2, index 1)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Parse header from line 2
        header_line = lines[1].strip()
        headers = [h.strip('"') for h in header_line.split(',')]

        # Read data, skipping first 4 rows
        df = pd.read_csv(
            file_path,
            skiprows=4,
            names=headers,
            parse_dates=['TIMESTAMP'] if 'TIMESTAMP' in headers else None,
        )

        return df

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
        else:
            return "string"
