"""SQLite repository for dataset catalog."""
import json
import sqlite3
from pathlib import Path
from nl2vis_bench.models import ColumnInfo, DatasetSchema


class CatalogRepository:
    """Repository for storing and retrieving dataset schemas."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    file_type TEXT NOT NULL,
                    file_count INTEGER DEFAULT 1,
                    file_paths TEXT,
                    handler_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS columns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    dtype TEXT NOT NULL,
                    min_value REAL,
                    max_value REAL,
                    mean_value REAL,
                    null_count INTEGER DEFAULT 0,
                    unique_count INTEGER,
                    sample_values TEXT
                )
            """)
            conn.commit()

    def save_dataset(self, schema: DatasetSchema) -> int:
        """Save a dataset schema to the catalog."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO datasets
                (name, file_type, file_count, file_paths, handler_used)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    schema.name,
                    schema.file_type,
                    schema.file_count,
                    json.dumps(schema.file_paths),
                    schema.handler_used,
                ),
            )
            dataset_id = cursor.lastrowid

            # Delete old columns if replacing
            conn.execute("DELETE FROM columns WHERE dataset_id = ?", (dataset_id,))

            # Insert columns
            for col in schema.columns:
                conn.execute(
                    """
                    INSERT INTO columns
                    (dataset_id, name, dtype, min_value, max_value, mean_value,
                     null_count, unique_count, sample_values)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_id,
                        col.name,
                        col.dtype,
                        col.min_value,
                        col.max_value,
                        col.mean_value,
                        col.null_count,
                        col.unique_count,
                        json.dumps(col.sample_values),
                    ),
                )
            conn.commit()
            return dataset_id

    def get_dataset(self, name: str) -> DatasetSchema | None:
        """Retrieve a dataset schema by name."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM datasets WHERE name = ?", (name,)
            ).fetchone()

            if row is None:
                return None

            columns = conn.execute(
                "SELECT * FROM columns WHERE dataset_id = ?", (row["id"],)
            ).fetchall()

            return DatasetSchema(
                name=row["name"],
                file_type=row["file_type"],
                file_count=row["file_count"],
                file_paths=json.loads(row["file_paths"] or "[]"),
                handler_used=row["handler_used"] or "",
                columns=[
                    ColumnInfo(
                        name=col["name"],
                        dtype=col["dtype"],
                        min_value=col["min_value"],
                        max_value=col["max_value"],
                        mean_value=col["mean_value"],
                        null_count=col["null_count"] or 0,
                        unique_count=col["unique_count"],
                        sample_values=json.loads(col["sample_values"] or "[]"),
                    )
                    for col in columns
                ],
            )

    def list_datasets(self) -> list[DatasetSchema]:
        """List all datasets in the catalog."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT name FROM datasets").fetchall()
            return [self.get_dataset(row["name"]) for row in rows]

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset from the catalog."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM datasets WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0
