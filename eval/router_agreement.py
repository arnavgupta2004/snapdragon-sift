#!/usr/bin/env python3
"""Reports how well the rule-based router and the learned router agree with genuine
LLM routing judgment (eval/router_labels.json, from eval/build_router_labels.py —
requires GEMINI_API_KEY, no synthetic substitute; see that script's docstring).

    python eval/router_agreement.py

Two numbers matter here:
  - rule_based_vs_llm agreement: how often the fast rule-based checks (exact filename,
    filter-only, vague/long) already match what the LLM would have said, *before* any
    classifier runs — this is the "we avoid an LLM call X% of the time" number from
    eval/latency_comparison.py, cross-checked against real LLM judgment instead of
    just assumed correct.
  - learned_vs_llm agreement: the learned router's accuracy specifically on the
    queries that reach it (the borderline ones neither rule matched) — this is what
    spec section 11 asks for directly.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.agent.learned_router import LearnedRouter  # noqa: E402
from app.agent.query_understanding import extract_entities  # noqa: E402
from app.agent.router import DEEP_MIN_WORDS, FAST_MAX_WORDS_FOR_FILTER_ONLY  # noqa: E402
from eval.run_benchmark import RESULTS_DIR  # noqa: E402

LABELS_PATH = REPO_ROOT / "eval" / "router_labels.json"


def _rule_based_only_decision(intent) -> str | None:
    """Re-implements just the zero-cost rule checks from app.agent.router.route(),
    returning None for queries that would reach the classifier fallback chain — kept
    separate from route() itself so this script can measure the rules in isolation."""
    if intent.exact_filename:
        return "fast"
    has_filter_signal = bool(intent.filters.file_types or intent.filters.modified_after)
    if (
        not intent.vague_marker_hit
        and has_filter_signal
        and intent.word_count <= FAST_MAX_WORDS_FOR_FILTER_ONLY
        and intent.residual_content_word_count <= 1
    ):
        return "fast"
    if intent.vague_marker_hit or intent.word_count >= DEEP_MIN_WORDS:
        return "deep"
    return None


def compute_agreement(labels: list[dict], learned: LearnedRouter) -> dict:
    rule_matches, rule_total = 0, 0
    rule_covered = 0  # queries the rules decided without needing any classifier
    learned_matches, learned_total = 0, 0
    confusion: Counter = Counter()

    for item in labels:
        intent = extract_entities(item["query"])
        llm_tier = item["llm_tier"]

        rule_tier = _rule_based_only_decision(intent)
        rule_total += 1
        if rule_tier is not None:
            rule_covered += 1
            if rule_tier == llm_tier:
                rule_matches += 1
            confusion[(llm_tier, rule_tier)] += 1
        elif learned.is_available:
            decision = learned.classify(intent)
            if decision is not None:
                learned_total += 1
                if decision.tier == llm_tier:
                    learned_matches += 1
                confusion[(llm_tier, decision.tier)] += 1

    return {
        "n_labeled_queries": len(labels),
        "rule_based": {
            "n_covered_without_any_classifier": rule_covered,
            "coverage_fraction": rule_covered / len(labels),
            "agreement_with_llm_on_covered": (rule_matches / rule_covered) if rule_covered else None,
        },
        "learned_router": {
            "available": learned.is_available,
            "n_evaluated_on_borderline": learned_total,
            "agreement_with_llm": (learned_matches / learned_total) if learned_total else None,
        },
    }


def main() -> None:
    if not LABELS_PATH.exists():
        raise SystemExit(
            f"{LABELS_PATH} not found. Run `python eval/build_router_labels.py` first "
            "(requires GEMINI_API_KEY — there is no synthetic substitute for real LLM "
            "labels here, since that's exactly what this script measures agreement "
            "against)."
        )

    labels = json.loads(LABELS_PATH.read_text())
    if not labels:
        raise SystemExit(f"{LABELS_PATH} is empty.")

    result = compute_agreement(labels, LearnedRouter())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "router_agreement_summary.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
