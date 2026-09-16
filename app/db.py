"""Shared SQLite engine/session access for the metadata store."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from sqlmodel import Session, create_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "db.sqlite"


@lru_cache(maxsize=8)
def get_engine(db_path: Path = DEFAULT_DB_PATH):
    return create_engine(f"sqlite:///{db_path}")


def get_session(db_path: Path = DEFAULT_DB_PATH) -> Session:
    return Session(get_engine(db_path))
