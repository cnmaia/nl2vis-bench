"""Tests for validator CLI."""

import pytest
import tempfile
from pathlib import Path
from nl2vis_bench.validator.cli import load_schema_from_csv


class TestLoadSchemaFromCSV:
    """Tests for load_schema_from_csv function."""

    def test_load_schema_with_default_name(self):
        """Test loading schema with default dataset name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("column_name,dtype,description\n")
            f.write("time,datetime,Timestamp\n")
            f.write("temperature,float,Temperature in C\n")
            csv_path = f.name

        try:
            schema = load_schema_from_csv(csv_path)
            assert schema.name == "unknown"
            assert len(schema.columns) == 2
            assert schema.columns[0].name == "time"
            assert schema.columns[1].name == "temperature"
        finally:
            Path(csv_path).unlink()

    def test_load_schema_with_custom_name(self):
        """Test loading schema with custom dataset name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("column_name,dtype\n")
            f.write("col1,float\n")
            csv_path = f.name

        try:
            schema = load_schema_from_csv(csv_path, dataset_name="my-dataset")
            assert schema.name == "my-dataset"
        finally:
            Path(csv_path).unlink()

    def test_load_schema_missing_column_name(self):
        """Test that missing 'column_name' column raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,dtype\n")
            f.write("col1,float\n")
            csv_path = f.name

        try:
            with pytest.raises(ValueError, match="CSV must have 'column_name' column"):
                load_schema_from_csv(csv_path)
        finally:
            Path(csv_path).unlink()

    def test_load_schema_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_schema_from_csv("/nonexistent/file.csv")
