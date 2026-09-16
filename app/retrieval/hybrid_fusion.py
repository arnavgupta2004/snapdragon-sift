"""Reciprocal Rank Fusion, implemented from scratch (no library) — combines ranked
lists from heterogeneous retrievers (keyword, semantic, ...) into one ranking without
needing their raw scores to be on comparable scales, which is exactly the problem with
naively summing BM25 scores and cosine similarities.

RRF score for a document d given a set of ranked lists R:
    score(d) = sum over lists r in R containing d of  1 / (k + rank_r(d))
where rank_r(d) is d's 1-indexed rank in list r, and k is a small constant (60 is the
standard default from the original RRF paper) that dampens the influence of very high
ranks so one list's #1 doesn't dominate irrecoverably.
"""

from __future__ import annotations

from app.config import RRF_K
from app.retrieval.base import ScoredFile


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredFile]], k: int = RRF_K, limit: int | None = None
) -> list[ScoredFile]:
    scores: dict[int, float] = {}
    contributing_sources: dict[int, list[str]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            scores[item.file_id] = scores.get(item.file_id, 0.0) + 1.0 / (k + rank)
            contributing_sources.setdefault(item.file_id, []).append(item.source)

    if not scores:
        return []

    max_score = max(scores.values())
    fused = [
        ScoredFile(
            file_id=file_id,
            score=score / max_score,  # normalize to [0, 1]
            source="hybrid",
            explanation=f"RRF fusion of {contributing_sources[file_id]}",
        )
        for file_id, score in scores.items()
    ]
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused[:limit] if limit else fused
