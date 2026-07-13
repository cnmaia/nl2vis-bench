"""Tests for catalog repository."""
import pytest
import tempfile
import os
from pathlib import Path
from nl2vis_bench.catalog.repository import CatalogRepository
from nl2vis_bench.models import ColumnInfo, DatasetSchema


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


def test_create_repository(temp_db):
    repo = CatalogRepository(temp_db)
    assert repo is not None
    assert Path(temp_db).exists()


def test_save_and_get_dataset(temp_db):
    repo = CatalogRepository(temp_db)

    col = ColumnInfo(
        name="temperature",
        dtype="float",
        min_value=18.5,
        max_value=38.2,
        mean_value=27.1
    )
    schema = DatasetSchema(
        name="test_dataset",
        file_type="xlsx",
        columns=[col],
        file_count=3,
        file_paths=["/data/file1.xlsx", "/data/file2.xlsx"]
    )

    repo.save_dataset(schema)

    retrieved = repo.get_dataset("test_dataset")
    assert retrieved is not None
    assert retrieved.name == "test_dataset"
    assert len(retrieved.columns) == 1
    assert retrieved.columns[0].name == "temperature"


def test_list_datasets(temp_db):
    repo = CatalogRepository(temp_db)

    schema1 = DatasetSchema(name="ds1", file_type="xlsx", columns=[])
    schema2 = DatasetSchema(name="ds2", file_type="csv", columns=[])

    repo.save_dataset(schema1)
    repo.save_dataset(schema2)

    datasets = repo.list_datasets()
    assert len(datasets) == 2
    assert "ds1" in [d.name for d in datasets]


def test_get_nonexistent_dataset(temp_db):
    repo = CatalogRepository(temp_db)
    result = repo.get_dataset("nonexistent")
    assert result is None


def test_delete_dataset(temp_db):
    repo = CatalogRepository(temp_db)
    schema = DatasetSchema(name="to_delete", file_type="xlsx", columns=[])
    repo.save_dataset(schema)

    repo.delete_dataset("to_delete")

    assert repo.get_dataset("to_delete") is None
