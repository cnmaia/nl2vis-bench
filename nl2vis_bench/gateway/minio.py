"""MinIO object storage gateway for reading sample data."""

import io
import json
import tempfile
from typing import Any

import pandas as pd
from minio import Minio


class MinIOGateway:
    """Read-only gateway for MinIO operations."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        self._client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._bucket = bucket

    def get_sample_data(self, sample_file_path: str) -> pd.DataFrame:
        """Download sample JSON from MinIO and return as DataFrame.

        Automatically detects and converts datetime columns.
        """
        response = self._client.get_object(self._bucket, sample_file_path)
        try:
            data = json.loads(response.read().decode("utf-8"))
            df = pd.DataFrame(data.get("rows", []))

            # Convert datetime columns (detect by name and content)
            datetime_column_names = {"time", "date", "datetime", "timestamp", "created_at", "updated_at"}
            for col in df.columns:
                # Check if column name suggests datetime
                if col.lower() in datetime_column_names or "time" in col.lower() or "date" in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except (ValueError, TypeError):
                        pass  # Not a valid datetime, keep as-is
                # Also try to detect datetime strings in object columns
                elif df[col].dtype == "object" and len(df) > 0:
                    sample_val = str(df[col].iloc[0])
                    # Check for ISO format patterns
                    if "T" in sample_val or (len(sample_val) >= 10 and sample_val[4:5] == "-"):
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except (ValueError, TypeError):
                            pass

            return df
        finally:
            response.close()
            response.release_conn()

    def get_sample_json(self, sample_file_path: str) -> dict[str, Any]:
        """Download sample JSON from MinIO and return raw dict."""
        response = self._client.get_object(self._bucket, sample_file_path)
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()

    def download_file(self, storage_path: str, extension: str) -> str:
        """Download file to temporary location and return path."""
        temp_file = tempfile.NamedTemporaryFile(
            suffix=f".{extension}",
            delete=False,
        )
        temp_path = temp_file.name
        temp_file.close()
        self._client.fget_object(self._bucket, storage_path, temp_path)
        return temp_path

    def file_exists(self, path: str) -> bool:
        """Check if a file exists in MinIO."""
        try:
            self._client.stat_object(self._bucket, path)
            return True
        except Exception:
            return False
