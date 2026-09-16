"""Adapts the synthetic corpus (already generated into data/db.sqlite by
data/generate_synthetic_data.py) to the same DataSource interface as
FilesystemDataSource — so both are provably interchangeable, not just described as
such. Not used by generate_synthetic_data.py itself (that script builds richer
FileRecord rows directly, with real topic_cluster ground truth the interface's
"uncategorized" default would throw away); this exists so ingestion tooling and tests
can treat "synthetic" and "real filesystem" as two instances of one interface.
"""

from __future__ import annotations

from sqlmodel import select

from app.datasources.base import DataSource, RawFile
from app.db import get_session
from app.models import FileRecord


class SyntheticDataSource(DataSource):
    def list_files(self) -> list[RawFile]:
        with get_session() as session:
            records = session.exec(select(FileRecord)).all()

        return [
            RawFile(
                filename=r.filename,
                path=r.path,
                file_type=r.file_type,
                size_bytes=r.size_bytes,
                created_at=r.created_at,
                modified_at=r.modified_at,
                extracted_text=r.extracted_text,
                topic_cluster=r.topic_cluster,
            )
            for r in records
        ]
