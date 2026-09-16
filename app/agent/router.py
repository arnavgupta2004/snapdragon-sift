"""Three-tier complexity router — the centerpiece of Objective 3.

Combines cheap rule-based signals (already extracted by query_understanding.py) with a
classification step used ONLY when those rules land in a genuinely ambiguous middle
ground. That classification step is itself two-tiered: try the learned router first
(app/agent/learned_router.py — a cheap classifier trained to approximate LLM judgment,
zero LLM calls), and only call the real LLM when the learned router is unavailable or
its confidence is below LOW_CONFIDENCE_THRESHOLD. eval/latency_comparison.py and
eval/router_agreement.py report what fraction of queries ever reach each tier of this
fallback chain.

Tiers:
  fast     — exact filename match, or a simple type+date filter with no ambiguous
             language. Filename fuzzy match + metadata filter only.
  standard — moderately specific query. Metadata + keyword + semantic, RRF-fused,
             personalization re-rank. No reranker, no LLM.
  deep     — vague/conversational/long query. Full hybrid retrieval, cross-encoder
             rerank, personalization, LLM reasoning pass for explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.agent.query_understanding import QueryIntent
from app.config import DEEP_MIN_WORDS, FAST_MAX_WORDS_FOR_FILTER_ONLY, LOW_CONFIDENCE_THRESHOLD


@dataclass
class RouteDecision:
    tier: str  # "fast" | "standard" | "deep"
    rationale: str
    used_llm_fallback: bool = False
    used_learned_router: bool = False


@lru_cache(maxsize=1)
def _get_learned_router():
    from app.agent.learned_router import LearnedRouter

    return LearnedRouter()


def _llm_classify(intent: QueryIntent) -> RouteDecision | None:
    from app.llm_client import get_client

    client = get_client()
    if not client.is_available:
        return None

    try:
        prompt = (
            "Classify this file-search query's complexity into exactly one word: "
            "fast, standard, or deep.\n"
            "- fast: an exact filename, or a simple file-type/date filter with no "
            "ambiguity (e.g. 'show me my pptx files from last week').\n"
            "- standard: a moderately specific topical search (e.g. 'find my recent "
            "notes about transformers').\n"
            "- deep: vague, conversational, or ambiguous, needing real reasoning to "
            "resolve (e.g. 'find that thing I was working on with my advisor a few "
            "weeks ago').\n"
            f"Query: \"{intent.raw_query}\"\nRespond with only the single word."
        )
        result = client.complete(prompt, max_tokens=10, temperature=0.0)
        tier = result.text.strip().lower()
        if tier in ("fast", "standard", "deep"):
            return RouteDecision(
                tier=tier,
                rationale=f"LLM fallback classification (rule-based signals were ambiguous): {tier}",
                used_llm_fallback=True,
            )
    except Exception:
        pass
    return None


def route(intent: QueryIntent) -> RouteDecision:
    if intent.exact_filename:
        return RouteDecision(
            tier="fast",
            rationale=f"query contains an exact filename pattern ('{intent.exact_filename}')",
        )

    has_filter_signal = bool(intent.filters.file_types or intent.filters.modified_after)
    if (
        not intent.vague_marker_hit
        and has_filter_signal
        and intent.word_count <= FAST_MAX_WORDS_FOR_FILTER_ONLY
        and intent.residual_content_word_count <= 1
    ):
        return RouteDecision(
            tier="fast",
            rationale="query is *just* a file-type/date filter — no topical content left "
            "after stripping filter terms, so no keyword/semantic search is needed",
        )

    if intent.vague_marker_hit or intent.word_count >= DEEP_MIN_WORDS:
        reason = "vague/conversational phrasing" if intent.vague_marker_hit else f"long, underspecified query ({intent.word_count} words)"
        return RouteDecision(tier="deep", rationale=f"{reason}, needs full hybrid+rerank+LLM reasoning")

    # Borderline: not an exact filename, not a clean filter query, not obviously vague
    # or long either. Try the learned router first (no LLM call); fall back to the
    # real LLM only if it's unavailable or unconfident; fall back further to a
    # rule-based default if neither is available.
    learned = _get_learned_router()
    if learned.is_available:
        learned_decision = learned.classify(intent)
        if learned_decision and learned_decision.confidence >= LOW_CONFIDENCE_THRESHOLD:
            return RouteDecision(
                tier=learned_decision.tier,
                rationale=(
                    f"learned router classification (confidence={learned_decision.confidence:.2f}), "
                    "approximates LLM judgment without calling it"
                ),
                used_learned_router=True,
            )

    llm_decision = _llm_classify(intent)
    if llm_decision:
        return llm_decision

    return RouteDecision(
        tier="standard",
        rationale="moderately specific topical query; defaulted to standard (no learned router or LLM available)",
    )
