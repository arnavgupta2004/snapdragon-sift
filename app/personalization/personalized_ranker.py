"""Interpretable weighted-sum personalization re-ranker — the baseline implementation
of BasePersonalizer. Weights are exposed as a dataclass specifically so eval/ can sweep
them and so the extended-scope LightGBM ranker (learned_ranker.py) has something
concrete to be A/B'd against on the same metric.

final_score = alpha * base_retrieval_score + (1 - alpha) * personalization_score
personalization_score = weighted blend of frequency, recency, file-type affinity,
                         discovered-topic-cluster affinity, and current-context boost
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlmodel import select

from app.db import get_session
from app.models import FileRecord
from app.personalization.base import BasePersonalizer
from app.personalization.profile_builder import build_user_profile, discover_topic_clusters
from app.personalization.temporal_patterns import current_context_boost
from app.retrieval.base import ScoredFile


@dataclass
class PersonalizationWeights:
    alpha: float = 0.6  # base retrieval score vs personalization score
    w_frequency: float = 0.30
    w_recency: float = 0.30
    w_type_affinity: float = 0.15
    w_cluster_affinity: float = 0.15
    w_context: float = 0.10


class WeightedSumPersonalizer(BasePersonalizer):
    def __init__(self, weights: PersonalizationWeights | None = None):
        self.weights = weights or PersonalizationWeights()

    def rank(
        self, user_id: int, candidates: list[ScoredFile], now: datetime | None = None
    ) -> list[ScoredFile]:
        if not candidates:
            return []
        now = now or datetime.now()
        w = self.weights

        profile = build_user_profile(user_id, now=now)
        context_boost = current_context_boost(user_id, now=now)
        file_to_cluster = discover_topic_clusters()

        with get_session() as session:
            file_ids = [c.file_id for c in candidates]
            records = session.exec(select(FileRecord).where(FileRecord.id.in_(file_ids))).all()
        type_by_id = {r.id: r.file_type for r in records}

        personal_weight_sum = (
            w.w_frequency + w.w_recency + w.w_type_affinity + w.w_cluster_affinity + w.w_context
        ) or 1.0

        results = []
        for c in candidates:
            freq = profile.file_frequency.get(c.file_id, 0.0)
            recency = profile.file_recency.get(c.file_id, 0.0)
            ftype = type_by_id.get(c.file_id)
            type_aff = profile.type_affinity.get(ftype, 0.0) if ftype else 0.0
            cluster_id = file_to_cluster.get(c.file_id)
            cluster_aff = profile.cluster_affinity.get(cluster_id, 0.0) if cluster_id is not None else 0.0
            ctx = context_boost.get(c.file_id, 0.0)

            personalization_score = (
                w.w_frequency * freq
                + w.w_recency * recency
                + w.w_type_affinity * type_aff
                + w.w_cluster_affinity * cluster_aff
                + w.w_context * ctx
            ) / personal_weight_sum

            final_score = w.alpha * c.score + (1 - w.alpha) * personalization_score
            results.append(
                ScoredFile(
                    file_id=c.file_id,
                    score=final_score,
                    source="personalized",
                    explanation=(
                        f"base={c.score:.3f} + personalization={personalization_score:.3f} "
                        f"(freq={freq:.2f}, recency={recency:.2f}, type={type_aff:.2f}, "
                        f"cluster={cluster_aff:.2f}, context={ctx:.2f})"
                    ),
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results
