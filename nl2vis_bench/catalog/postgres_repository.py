"""PostgreSQL repository for reading metadata from gatekeeper DB."""
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import BigInteger, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, scoped_session, sessionmaker

from nl2vis_bench.models import ColumnInfo, DatasetSchema

Base = declarative_base()


class DataFileDB(Base):
    """Read-only model for data_files table."""
    __tablename__ = "data_files"
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    name = Column(String(1024), nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    extension = Column(String(512), nullable=True)
    storage_file_name = Column(String(1024), nullable=True)
    storage_path = Column(String(2048), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    version_id = Column(PGUUID(as_uuid=True), nullable=True)


class FileMetadataDB(Base):
    __tablename__ = "file_metadata"
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    data_file_id = Column(PGUUID(as_uuid=True), unique=True, nullable=False)
    row_count = Column(BigInteger, nullable=False)
    sample_file_path = Column(String(2048), nullable=True)
    extracted_at = Column(DateTime(timezone=True), nullable=False)
    extractor_version = Column(String(50), nullable=False)
    llm_provider = Column(String(100), nullable=True)
    columns = relationship("ColumnMetadataDB", lazy="subquery", backref="file_metadata")


class ColumnMetadataDB(Base):
    __tablename__ = "column_metadata"
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    file_metadata_id = Column(PGUUID(as_uuid=True), ForeignKey("file_metadata.id"))
    name = Column(String(512), nullable=False)
    dtype = Column(String(50), nullable=False)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    mean_value = Column(Float, nullable=True)
    std_value = Column(Float, nullable=True)
    null_count = Column(BigInteger, default=0)
    sample_values = Column(JSONB, nullable=True)
    description = Column(Text, nullable=True)
    position = Column(Integer, nullable=False)


@dataclass
class FileInfo:
    """Extended file information including storage paths."""
    data_file_id: UUID
    name: str
    extension: str | None
    storage_path: str | None
    storage_file_name: str | None
    sample_file_path: str | None
    row_count: int
    column_count: int
    extracted_at: str


class PostgresCatalogRepository:
    """Read-only repository for metadata from gatekeeper database."""

    def __init__(self, db_url: str):
        self._engine = create_engine(db_url)
        self._session_factory = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
        )

    def get_dataset_schema(self, data_file_id: UUID) -> DatasetSchema | None:
        """Get schema for a specific file, including file name from data_files."""
        session = self._session_factory()
        try:
            metadata = (
                session.query(FileMetadataDB)
                .filter(FileMetadataDB.data_file_id == data_file_id)
                .first()
            )
            if metadata is None:
                return None

            # Fetch file name from data_files table
            data_file = session.query(DataFileDB).filter(DataFileDB.id == data_file_id).first()
            file_name = data_file.name if data_file else str(data_file_id)

            return DatasetSchema(
                name=file_name,
                file_type=data_file.extension if data_file else "extracted",
                columns=[
                    ColumnInfo(
                        name=col.name,
                        dtype=col.dtype,
                        min_value=col.min_value,
                        max_value=col.max_value,
                        mean_value=col.mean_value,
                        null_count=col.null_count or 0,
                        sample_values=col.sample_values or [],
                        description=col.description,
                    )
                    for col in sorted(metadata.columns, key=lambda c: c.position)
                ],
                sample_file_path=metadata.sample_file_path,
            )
        finally:
            session.close()

    def get_file_info(self, data_file_id: UUID) -> FileInfo | None:
        """Get extended file information including storage paths."""
        session = self._session_factory()
        try:
            metadata = (
                session.query(FileMetadataDB)
                .filter(FileMetadataDB.data_file_id == data_file_id)
                .first()
            )
            if metadata is None:
                return None

            data_file = session.query(DataFileDB).filter(DataFileDB.id == data_file_id).first()
            if data_file is None:
                return None

            return FileInfo(
                data_file_id=data_file_id,
                name=data_file.name,
                extension=data_file.extension,
                storage_path=data_file.storage_path,
                storage_file_name=data_file.storage_file_name,
                sample_file_path=metadata.sample_file_path,
                row_count=metadata.row_count,
                column_count=len(metadata.columns),
                extracted_at=str(metadata.extracted_at),
            )
        finally:
            session.close()

    def list_files_with_metadata(self) -> list[FileInfo]:
        """List all files that have extracted metadata."""
        session = self._session_factory()
        try:
            results = (
                session.query(FileMetadataDB, DataFileDB)
                .join(DataFileDB, FileMetadataDB.data_file_id == DataFileDB.id)
                .all()
            )
            return [
                FileInfo(
                    data_file_id=m.data_file_id,
                    name=f.name,
                    extension=f.extension,
                    storage_path=f.storage_path,
                    storage_file_name=f.storage_file_name,
                    sample_file_path=m.sample_file_path,
                    row_count=m.row_count,
                    column_count=len(m.columns),
                    extracted_at=str(m.extracted_at),
                )
                for m, f in results
            ]
        finally:
            session.close()
