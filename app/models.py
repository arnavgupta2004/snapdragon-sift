"""Shared SQLModel schema for the metadata store.

Used both by the synthetic data generator (data/generate_synthetic_data.py) and the
runtime retrieval/personalization code (app/retrieval/metadata_search.py,
app/personalization/*), so there is exactly one definition of the schema.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class FileRecord(SQLModel, table=True):
    __tablename__ = "files"

    id: int | None = Field(default=None, primary_key=True)
    filename: str
    path: str
    file_type: str = Field(index=True)
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    topic_cluster: str = Field(index=True)
    extracted_text: str = ""


class UserRecord(SQLModel, table=True):
    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    name: str
    persona_key: str = Field(index=True)


class AccessEvent(SQLModel, table=True):
    __tablename__ = "access_log"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    file_id: int = Field(foreign_key="files.id", index=True)
    action: str  # "open" | "edit" | "search"
    timestamp: datetime = Field(index=True)
    session_id: str = Field(index=True)


class FeedbackEvent(SQLModel, table=True):
    """Implicit/explicit feedback written back by the UI (thumbs up/down, "I opened this").
    Consumed by personalization/retrain.py in the extended-scope phase."""

    __tablename__ = "feedback_log"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    file_id: int = Field(foreign_key="files.id", index=True)
    query: str
    signal: str  # "thumbs_up" | "thumbs_down" | "opened"
    timestamp: datetime = Field(index=True)
