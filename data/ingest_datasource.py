#!/usr/bin/env python3
"""Ingests a DataSource's files into the metadata DB (the same `files` table the
synthetic corpus populates) and rebuilds the semantic search index. This is what lets
the professor demo the system against real files live: everything downstream
(retrieval, personalization, routing) only ever reads FileRecord rows and has no idea
whether a given row came from the synthetic generator or a real directory crawl.

    python data/ingest_datasource.py --root /path/to/real/directory --max-files 300

Always targets app.db.DEFAULT_DB_PATH (the same DB every other script/API/UI in this
repo reads) — there is exactly one metadata DB and one Chroma vector store in this
architecture, deliberately: the semantic index (app.retrieval.semantic_search) is a
process-wide singleton with no per-database scoping, so ingesting into a different
SQLite file would silently leave the vector index out of sync with it. If you need an
isolated test DB, test the DataSource/insertion logic directly (see
tests/test_ingest_datasource.py) rather than pointing this script elsewhere.

By default this ADDS to whatever's already in the DB (synthetic + real side by side,
richer demo material). Pass --clear-existing to replace the corpus entirely with just
the crawled files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session, delete  # noqa: E402

from app.datasources.base import RawFile  # noqa: E402
from app.datasources.filesystem_source import FilesystemDataSource  # noqa: E402
from app.db import get_session  # noqa: E402
from app.models import FileRecord  # noqa: E402


def ingest_files(raw_files: list[RawFile], session: Session, clear_existing: bool = False) -> dict[str, int]:
    """Pure insertion logic, no semantic reindexing — kept separate from main() so
    tests can exercise it against an isolated in-memory session."""
    if clear_existing:
        session.exec(delete(FileRecord))
        session.commit()

    by_type: dict[str, int] = {}
    for rf in raw_files:
        session.add(
            FileRecord(
                filename=rf.filename,
                path=rf.path,
                file_type=rf.file_type,
                size_bytes=rf.size_bytes,
                created_at=rf.created_at,
                modified_at=rf.modified_at,
                topic_cluster=rf.topic_cluster,
                extracted_text=rf.extracted_text,
            )
        )
        by_type[rf.file_type] = by_type.get(rf.file_type, 0) + 1
    session.commit()
    return by_type


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="local directory to crawl")
    parser.add_argument("--max-files", type=int, default=300)
    parser.add_argument(
        "--clear-existing", action="store_true",
        help="delete existing FileRecord rows first (default: add alongside them)",
    )
    args = parser.parse_args()

    source = FilesystemDataSource(root=args.root, max_files=args.max_files)
    raw_files = source.list_files()
    print(f"[ingest_datasource] crawled {len(raw_files)} files from {source.root}")

    with get_session() as session:
        by_type = ingest_files(raw_files, session, clear_existing=args.clear_existing)

    print(f"[ingest_datasource] inserted {len(raw_files)} files, by type: {dict(sorted(by_type.items()))}")

    print("[ingest_datasource] rebuilding semantic search index (this may take a moment)...")
    from app.retrieval.semantic_search import build_index

    n = build_index(force=True)
    print(f"[ingest_datasource] semantic index now covers {n} files total")


if __name__ == "__main__":
    main()
