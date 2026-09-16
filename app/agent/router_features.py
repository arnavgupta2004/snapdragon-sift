"""Feature vector for the learned router — cheap, rule-based signals only (the same
ones app/agent/router.py's rule-based tiers already compute), so scoring a query with
the learned model costs nothing beyond what entity extraction already pays for.
"""

from __future__ import annotations

from app.agent.query_understanding import QueryIntent

ROUTER_FEATURE_NAMES = [
    "word_count",
    "residual_content_word_count",
    "has_exact_filename",
    "has_filter_signal",
    "vague_marker_hit",
]


def router_featurize(intent: QueryIntent) -> list[float]:
    has_filter_signal = 1.0 if (intent.filters.file_types or intent.filters.modified_after) else 0.0
    return [
        float(intent.word_count),
        float(intent.residual_content_word_count),
        1.0 if intent.exact_filename else 0.0,
        has_filter_signal,
        1.0 if intent.vague_marker_hit else 0.0,
    ]
