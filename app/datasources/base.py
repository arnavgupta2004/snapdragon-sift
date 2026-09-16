"""The DataSource interface — anything that can enumerate files and extract their text
content. Retrieval, personalization, and routing code never talks to a DataSource
directly; they only ever query FileRecord rows in the DB. An ingestion step
(data/ingest_datasource.py) populates those rows from whichever DataSource is
configured. That's what makes swapping the synthetic corpus for a real filesystem
crawler (or, in principle, a Google Drive connector implementing this same interface)
a zero-code-change operation for the rest of the system — see
app/datasources/filesystem_source.py for the concrete real-data implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RawFile:
    filename: str
    path: str
    file_type: str
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    extracted_text: str
    topic_cluster: str = "uncategorized"  # real data has no ground-truth topic label


class DataSource(ABC):
    @abstractmethod
    def list_files(self) -> list[RawFile]:
        raise NotImplementedError
