"""Shared feature vector definition for the learned (LightGBM) personalizer —
imported by both training_data.py (builds synthetic training examples) and
learned_ranker.py (scores real candidates at inference time), so the two can never
drift out of sync with each other.

`relevance_signal` is deliberately the same feature slot at train and inference time
despite being computed differently:
  - at training time (app/personalization/training_data.py): cosine similarity between
    the candidate's semantic embedding and the centroid embedding of the (synthetic)
    query-context topic — continuous, not binary
  - at inference time (learned_ranker.py): the real base retrieval/rerank score for
    the actual query — also continuous, roughly the same [0, 1]-ish shape
Both represent "how relevant does this candidate look for the query independent of who
is asking" — this is a standard bootstrap for training an LTR model without a real
labeled query log. An earlier version used a binary same-topic/not-same-topic signal
at training time; LightGBM's feature importance showed it learned to almost ignore
that feature (~11 vs. ~80 for recency) and the resulting model underperformed the
hand-tuned baseline (see eval/learned_ranker_comparison.py) — a binary training
distribution doesn't teach split thresholds that generalize to continuous inference-
time scores. The continuous cosine-similarity version fixes that.
"""

from __future__ import annotations

from app.personalization.profile_builder import UserProfile

FEATURE_NAMES = [
    "relevance_signal",
    "frequency",
    "recency",
    "type_affinity",
    "cluster_affinity",
    "is_recurring_now",
]


def featurize(
    relevance_signal: float,
    file_id: int,
    profile: UserProfile,
    file_type: str | None,
    cluster_id: int | None,
    is_recurring_now: bool,
) -> list[float]:
    return [
        relevance_signal,
        profile.file_frequency.get(file_id, 0.0),
        profile.file_recency.get(file_id, 0.0),
        profile.type_affinity.get(file_type, 0.0) if file_type else 0.0,
        profile.cluster_affinity.get(cluster_id, 0.0) if cluster_id is not None else 0.0,
        1.0 if is_recurring_now else 0.0,
    ]
