"""LightGBM learning-to-rank personalizer — the production-default implementation of
BasePersonalizer, meant to be A/B'd against WeightedSumPersonalizer (see
eval/learned_ranker_comparison.py) using the exact same personalization-lift
methodology from eval/personalization_lift.py.

Falls back to returning candidates unchanged (no reordering) if no trained model file
exists yet, so the rest of the system never breaks on a missing model — train one with
`python -m app.personalization.retrain`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlmodel import select

from app.db import get_session
from app.models import FileRecord
from app.personalization.base import BasePersonalizer
from app.personalization.features import featurize
from app.personalization.profile_builder import build_user_profile, discover_topic_clusters
from app.personalization.temporal_patterns import detect_recurring_patterns
from app.retrieval.base import ScoredFile

MODEL_PATH = Path(__file__).resolve().parent / "models" / "lgbm_ranker.txt"


class LearnedPersonalizer(BasePersonalizer):
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._model = None
        if model_path.exists():
            import lightgbm as lgb

            self._model = lgb.Booster(model_file=str(model_path))

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def rank(
        self, user_id: int, candidates: list[ScoredFile], now: datetime | None = None
    ) -> list[ScoredFile]:
        if not candidates:
            return []
        if self._model is None:
            return list(candidates)

        now = now or datetime.now()
        profile = build_user_profile(user_id, now=now)
        recurring_file_ids = {
            fid for p in detect_recurring_patterns(user_id) for fid in p.file_ids
        }
        file_to_cluster = discover_topic_clusters()

        with get_session() as session:
            ids = [c.file_id for c in candidates]
            records = session.exec(select(FileRecord).where(FileRecord.id.in_(ids))).all()
        type_by_id = {r.id: r.file_type for r in records}

        import numpy as np

        features = np.array(
            [
                featurize(
                    relevance_signal=c.score,
                    file_id=c.file_id,
                    profile=profile,
                    file_type=type_by_id.get(c.file_id),
                    cluster_id=file_to_cluster.get(c.file_id),
                    is_recurring_now=c.file_id in recurring_file_ids,
                )
                for c in candidates
            ]
        )
        raw_scores = self._model.predict(features)

        lo, hi = float(min(raw_scores)), float(max(raw_scores))
        span = (hi - lo) or 1.0
        results = [
            ScoredFile(
                file_id=c.file_id,
                score=(float(s) - lo) / span,
                source="learned_personalized",
                explanation=f"LightGBM LTR score {float(s):.3f} (raw, min-max normalized)",
            )
            for c, s in zip(candidates, raw_scores)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results
