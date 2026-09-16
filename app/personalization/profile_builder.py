"""Builds a per-user behavioral profile from the access log: frequency, recency,
file-type affinity, and topic affinity.

Topic affinity deliberately does NOT use the corpus's ground-truth `topic_cluster`
label — a real deployment wouldn't have hand-labeled topics for arbitrary files. It
instead clusters the semantic embeddings already computed for search (KMeans, same
count as the synthetic corpus's 10 topics, purely for eval comparability) and tracks
per-user engagement against those *discovered* clusters. eval/ can compare discovered
clusters against the ground-truth labels to report clustering quality, which the
weighted-sum vs. ground-truth-label alternative could never support.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

from sqlmodel import select

from app.config import N_CLUSTERS, RECENCY_HALF_LIFE_DAYS
from app.db import get_session
from app.models import AccessEvent, FileRecord


@dataclass
class UserProfile:
    user_id: int
    file_frequency: dict[int, float] = field(default_factory=dict)   # file_id -> [0,1]
    type_affinity: dict[str, float] = field(default_factory=dict)    # file_type -> [0,1]
    cluster_affinity: dict[int, float] = field(default_factory=dict) # cluster_id -> [0,1]
    file_recency: dict[int, float] = field(default_factory=dict)     # file_id -> [0,1]
    preferred_file_types: list[str] = field(default_factory=list)    # most -> least preferred


@lru_cache(maxsize=1)
def discover_topic_clusters(n_clusters: int = N_CLUSTERS) -> dict[int, int]:
    """KMeans over the semantic embeddings already indexed for search. Returns
    file_id -> discovered cluster id. Cached process-wide since embeddings don't
    change without a corpus regeneration."""
    from sklearn.cluster import KMeans

    from app.retrieval.semantic_search import _get_collection, build_index

    build_index(force=False)
    collection = _get_collection()
    data = collection.get(include=["embeddings"])
    ids = [int(i) for i in data["ids"]]
    embeddings = data["embeddings"]

    if len(ids) < n_clusters:
        return {file_id: 0 for file_id in ids}

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, algorithm="lloyd")
    labels = km.fit_predict(embeddings)
    return dict(zip(ids, (int(label) for label in labels)))


def _normalize(counter: Counter) -> dict:
    if not counter:
        return {}
    max_v = max(counter.values())
    if max_v == 0:
        return {k: 0.0 for k in counter}
    return {k: v / max_v for k, v in counter.items()}


def build_user_profile(
    user_id: int, now: datetime | None = None, half_life_days: float = RECENCY_HALF_LIFE_DAYS
) -> UserProfile:
    now = now or datetime.now()

    with get_session() as session:
        events = session.exec(select(AccessEvent).where(AccessEvent.user_id == user_id)).all()
        files = session.exec(select(FileRecord)).all()

    file_type_by_id = {f.id: f.file_type for f in files if f.id is not None}

    if not events:
        return UserProfile(user_id=user_id)

    file_counts: Counter = Counter(e.file_id for e in events)
    type_counts: Counter = Counter(file_type_by_id.get(e.file_id, "unknown") for e in events)

    file_to_cluster = discover_topic_clusters()
    cluster_counts: Counter = Counter(
        file_to_cluster[e.file_id] for e in events if e.file_id in file_to_cluster
    )

    last_access: dict[int, datetime] = {}
    for e in events:
        if e.file_id not in last_access or e.timestamp > last_access[e.file_id]:
            last_access[e.file_id] = e.timestamp

    file_recency = {
        file_id: 0.5 ** (max(0.0, (now - ts).total_seconds() / 86400.0) / half_life_days)
        for file_id, ts in last_access.items()
    }

    type_affinity = _normalize(type_counts)
    preferred_types = [t for t, _ in sorted(type_affinity.items(), key=lambda x: x[1], reverse=True)]

    return UserProfile(
        user_id=user_id,
        file_frequency=_normalize(file_counts),
        type_affinity=type_affinity,
        cluster_affinity=_normalize(cluster_counts),
        file_recency=file_recency,
        preferred_file_types=preferred_types,
    )
