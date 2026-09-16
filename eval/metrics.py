"""Standard IR metrics, binary relevance. Used by every script in eval/."""

from __future__ import annotations

import math


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id in relevant_ids)
    return hits / len(top)


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for doc_id in top if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    top = retrieved_ids[:k]
    dcg = sum(
        1.0 / math.log2(rank + 2) for rank, doc_id in enumerate(top) if doc_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    for rank, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (rank + 1)
    return 0.0
