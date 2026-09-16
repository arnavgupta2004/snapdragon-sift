"""Common interface for personalization scorers.

Two implementations share this interface: WeightedSumPersonalizer (interpretable
baseline, personalized_ranker.py) and, in the extended-scope phase, a LightGBM
learning-to-rank model (learned_ranker.py). The eval harness A/B's them by swapping
which one it calls — nothing else in the pipeline needs to know which is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from app.retrieval.base import ScoredFile


class BasePersonalizer(ABC):
    @abstractmethod
    def rank(
        self, user_id: int, candidates: list[ScoredFile], now: datetime | None = None
    ) -> list[ScoredFile]:
        """Re-scores `candidates` (already-retrieved results) using the user's
        behavioral profile. Must return the same file_ids, re-scored and re-sorted."""
        raise NotImplementedError
