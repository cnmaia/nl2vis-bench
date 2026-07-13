"""Safe query execution with sandboxing."""
import time
import pandas as pd
import numpy as np
from nl2vis_bench.models import DataProfile, ExecutionResult


class QueryExecutor:
    """Executes pandas queries in a safe sandbox."""

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds

    def execute(self, df: pd.DataFrame, query: str) -> ExecutionResult:
        """Execute a pandas query safely and return result with profile."""
        start_time = time.time()

        try:
            # Create restricted namespace
            namespace = {
                'df': df,
                'pd': pd,
                'np': np,
            }

            # Execute query in sandbox (no builtins)
            result = eval(query, {"__builtins__": {}}, namespace)

            # Convert result to DataFrame if needed
            result_df = self._to_dataframe(result)

            # Extract profile from result
            profile = self._extract_profile(result_df)

            execution_time = (time.time() - start_time) * 1000

            return ExecutionResult(
                success=True,
                data=result_df,
                profile=profile,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e),
            )

    def _to_dataframe(self, result) -> pd.DataFrame:
        """Convert various result types to DataFrame."""
        if isinstance(result, pd.DataFrame):
            result_df = result
        elif isinstance(result, pd.Series):
            result_df = result.to_frame()
        elif isinstance(result, (int, float, np.number)):
            result_df = pd.DataFrame({'result': [result]})
        elif isinstance(result, dict):
            result_df = pd.DataFrame([result])
        else:
            result_df = pd.DataFrame({'result': [result]})

        # Handle Period index - convert to timestamp and reset
        if isinstance(result_df.index, pd.PeriodIndex):
            result_df.index = result_df.index.to_timestamp()
            result_df = result_df.reset_index()

        # Handle Period columns - convert to timestamp
        for col in result_df.columns:
            if isinstance(result_df[col].dtype, pd.PeriodDtype):
                result_df[col] = result_df[col].dt.to_timestamp()

        return result_df

    def _extract_profile(self, df: pd.DataFrame) -> DataProfile:
        """Extract data profile from DataFrame."""
        datetime_column = None
        datetime_columns = []
        numeric_columns = []
        categorical_columns = []

        for col in df.columns:
            dtype = df[col].dtype

            # Expanded datetime detection - include Period dtype
            is_datetime = (
                pd.api.types.is_datetime64_any_dtype(dtype) or
                isinstance(dtype, pd.PeriodDtype)
            )

            if is_datetime:
                datetime_columns.append(col)
                if datetime_column is None:
                    datetime_column = col
            elif pd.api.types.is_numeric_dtype(dtype):
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)

        return DataProfile(
            has_datetime_column=len(datetime_columns) > 0,
            has_numeric_columns=len(numeric_columns) > 0,
            numeric_column_count=len(numeric_columns),
            row_count=len(df),
            datetime_column=datetime_column,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )
