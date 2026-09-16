"""Structured metadata filtering — file type, topic, filename substring, date range.

Also zero-LLM, zero-embedding: the second building block of the fast route (alongside
filename_search) for queries like "show me my pptx files from last week".

Usage:
    python -m app.retrieval.metadata_search --file-type pptx --since-days 7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlmodel import select

from app.db import get_session
from app.models import FileRecord
from app.retrieval.base import ScoredFile


@dataclass
class MetadataFilters:
    file_types: list[str] | None = None
    topic_cluster: str | None = None
    filename_contains: str | None = None
    modified_after: datetime | None = None
    modified_before: datetime | None = None


def search(filters: MetadataFilters, limit: int = 50) -> list[ScoredFile]:
    """Boolean-filters the corpus, then ranks matches by recency (more recently
    modified = higher score) since there's no relevance signal to rank on otherwise."""
    with get_session() as session:
        stmt = select(FileRecord)
        if filters.file_types:
            stmt = stmt.where(FileRecord.file_type.in_(filters.file_types))
        if filters.topic_cluster:
            stmt = stmt.where(FileRecord.topic_cluster == filters.topic_cluster)
        if filters.filename_contains:
            stmt = stmt.where(FileRecord.filename.contains(filters.filename_contains))
        if filters.modified_after:
            stmt = stmt.where(FileRecord.modified_at >= filters.modified_after)
        if filters.modified_before:
            stmt = stmt.where(FileRecord.modified_at <= filters.modified_before)

        matched = session.exec(stmt).all()

    if not matched:
        return []

    matched.sort(key=lambda f: f.modified_at, reverse=True)
    matched = matched[:limit]

    newest = matched[0].modified_at
    oldest = matched[-1].modified_at
    span = (newest - oldest).total_seconds() or 1.0

    results = []
    for f in matched:
        recency_frac = (f.modified_at - oldest).total_seconds() / span
        score = 0.5 + 0.5 * recency_frac  # all matches score >= 0.5, ranked by recency
        results.append(
            ScoredFile(
                file_id=f.id,
                score=score,
                source="metadata",
                explanation="matched metadata filters, ranked by recency",
            )
        )
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description="Metadata filter search over the corpus DB.")
    parser.add_argument("--file-type", action="append", dest="file_types")
    parser.add_argument("--topic")
    parser.add_argument("--contains")
    parser.add_argument("--since-days", type=int)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    filters = MetadataFilters(
        file_types=args.file_types,
        topic_cluster=args.topic,
        filename_contains=args.contains,
        modified_after=(datetime.now() - timedelta(days=args.since_days)) if args.since_days else None,
    )

    with get_session() as session:
        id_to_name = {f.id: f.filename for f in session.exec(select(FileRecord)).all()}

    for r in search(filters, limit=args.limit):
        print(f"{r.score:.3f}  {id_to_name.get(r.file_id, '?')}")


if __name__ == "__main__":
    _main()
