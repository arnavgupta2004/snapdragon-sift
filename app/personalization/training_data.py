"""Builds learning-to-rank training data for the LightGBM personalizer.

Two sources, both producing graded-relevance groups for lambdarank:

  1. The access log. An access event has no associated search query, so we treat the
     accessed file's topic as an implicit query context and construct a group: the
     accessed file (label=2, "opened"), a few same-topic files that would plausibly
     have been shown but weren't opened (label=1), and a few different-topic files
     (label=0, clear negatives). This gives the model real signal on what this user's
     profile features look like for files they actually engage with vs. don't, without
     needing a real query log. The `relevance_signal` feature for these rows is each
     candidate's semantic-embedding cosine similarity to the topic's centroid — a
     continuous stand-in for "how relevant would a real retriever have scored this",
     chosen over a binary same-topic/not flag because a near-binary training
     distribution taught the model to ignore this feature almost entirely (see
     app/personalization/features.py for the full explanation).

  2. app.models.FeedbackEvent — real feedback written back through the UI/API
     (thumbs_up/thumbs_down/opened), grouped by the query it was given for. This is
     what makes retrain.py's "the model improves as feedback accumulates" story real:
     these rows get heavier weight in the label (thumbs_down is an explicit, confident
     negative; thumbs_up/opened an explicit, confident positive) and grow over time as
     the feedback loop runs.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
from sqlmodel import select

from app.config import (
    ACCESS_LOG_ROW_WEIGHT,
    FEEDBACK_ROW_WEIGHT,
    MAX_EVENTS_PER_USER,
    N_NEGATIVES_OTHER_TOPIC,
    N_NEGATIVES_SAME_TOPIC,
)
from app.db import get_session
from app.models import AccessEvent, FeedbackEvent, FileRecord, UserRecord
from app.personalization.features import featurize
from app.personalization.profile_builder import build_user_profile, discover_topic_clusters
from app.personalization.temporal_patterns import detect_recurring_patterns


def _get_all_embeddings() -> dict[int, np.ndarray]:
    """All corpus file embeddings, keyed by file_id — reuses the same index built for
    semantic search rather than maintaining a second copy."""
    from app.retrieval.semantic_search import _get_collection, build_index

    build_index(force=False)
    data = _get_collection().get(include=["embeddings"])
    return {int(i): np.array(e) for i, e in zip(data["ids"], data["embeddings"])}


def _topic_centroids(files_by_topic: dict[str, list[FileRecord]], embeddings: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    centroids = {}
    for topic, files in files_by_topic.items():
        vecs = [embeddings[f.id] for f in files if f.id in embeddings]
        if vecs:
            centroids[topic] = np.mean(vecs, axis=0)
    return centroids


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return max(0.0, float(np.dot(a, b) / denom))


def build_training_data(
    include_feedback: bool = True, seed: int = 13
) -> tuple[list[list[float]], list[int], list[int], list[float]]:
    """Returns (X, y, groups, weights) — flat feature rows, flat labels, group sizes
    (each group's rows are contiguous in X/y), and per-row sample weights, ready for
    lightgbm.Dataset(group=groups, weight=weights)."""
    rng = random.Random(seed)

    with get_session() as session:
        files = session.exec(select(FileRecord)).all()
        users = session.exec(select(UserRecord)).all()
        access_events = session.exec(select(AccessEvent)).all()
        feedback_events = session.exec(select(FeedbackEvent)).all() if include_feedback else []

    files_by_id = {f.id: f for f in files}
    files_by_topic: dict[str, list[FileRecord]] = defaultdict(list)
    for f in files:
        files_by_topic[f.topic_cluster].append(f)

    file_to_cluster = discover_topic_clusters()
    embeddings = _get_all_embeddings()
    topic_centroids = _topic_centroids(files_by_topic, embeddings)

    X: list[list[float]] = []
    y: list[int] = []
    groups: list[int] = []
    weights: list[float] = []

    for user in users:
        user_events = [e for e in access_events if e.user_id == user.id]
        if not user_events:
            continue

        profile = build_user_profile(user.id)
        recurring_file_ids = {
            fid for p in detect_recurring_patterns(user.id) for fid in p.file_ids
        }

        sampled_events = rng.sample(user_events, k=min(MAX_EVENTS_PER_USER, len(user_events)))
        for event in sampled_events:
            positive_file = files_by_id.get(event.file_id)
            if positive_file is None:
                continue
            topic = positive_file.topic_cluster

            same_topic_pool = [f for f in files_by_topic[topic] if f.id != positive_file.id]
            other_topic_pool = [f for f in files if f.topic_cluster != topic]
            same_topic_negs = rng.sample(same_topic_pool, k=min(N_NEGATIVES_SAME_TOPIC, len(same_topic_pool)))
            other_topic_negs = rng.sample(other_topic_pool, k=min(N_NEGATIVES_OTHER_TOPIC, len(other_topic_pool)))

            centroid = topic_centroids.get(topic)

            def _rel(f: FileRecord) -> float:
                if centroid is None or f.id not in embeddings:
                    return 0.0
                return _cosine_sim(embeddings[f.id], centroid)

            group = [(positive_file, 2, _rel(positive_file))]
            group += [(f, 1, _rel(f)) for f in same_topic_negs]
            group += [(f, 0, _rel(f)) for f in other_topic_negs]
            rng.shuffle(group)

            _append_group(X, y, groups, weights, group, profile, file_to_cluster, recurring_file_ids, ACCESS_LOG_ROW_WEIGHT)

        if include_feedback:
            user_feedback = [e for e in feedback_events if e.user_id == user.id]
            by_query: dict[str, list[FeedbackEvent]] = defaultdict(list)
            for e in user_feedback:
                by_query[e.query].append(e)

            for events_for_query in by_query.values():
                group = []
                for e in events_for_query:
                    f = files_by_id.get(e.file_id)
                    if f is None:
                        continue
                    label = 2 if e.signal in ("thumbs_up", "opened") else 0
                    group.append((f, label, 1.0))  # it was shown for this query
                if group:
                    _append_group(X, y, groups, weights, group, profile, file_to_cluster, recurring_file_ids, FEEDBACK_ROW_WEIGHT)

    return X, y, groups, weights


def _append_group(X, y, groups, weights, group, profile, file_to_cluster, recurring_file_ids, row_weight: float) -> None:
    size = 0
    for f, label, relevance_signal in group:
        X.append(
            featurize(
                relevance_signal=relevance_signal,
                file_id=f.id,
                profile=profile,
                file_type=f.file_type,
                cluster_id=file_to_cluster.get(f.id),
                is_recurring_now=f.id in recurring_file_ids,
            )
        )
        y.append(label)
        weights.append(row_weight)
        size += 1
    groups.append(size)
