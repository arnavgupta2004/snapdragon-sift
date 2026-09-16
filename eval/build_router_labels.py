#!/usr/bin/env python3
"""Generates ground-truth routing-tier labels by genuinely calling the LLM classifier
(app.agent.router._llm_classify) on every query in the eval set.

This is the one script in this repo that has no rule-based substitute: the entire
point of app/agent/learned_router.py is to measure how well a cheap classifier can
approximate *real* LLM judgment, so faking these labels would make that measurement
meaningless. Requires GEMINI_API_KEY — exits with a clear error if it's not set,
rather than producing fabricated labels.

    python eval/build_router_labels.py

Writes eval/router_labels.json, consumed by app/agent/train_router.py and
eval/router_agreement.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.agent.query_understanding import extract_entities  # noqa: E402
from app.agent.router import _llm_classify  # noqa: E402
from app.llm_client import get_client  # noqa: E402
from eval.run_benchmark import load_eval_set  # noqa: E402

OUT_PATH = REPO_ROOT / "eval" / "router_labels.json"


def main() -> None:
    client = get_client()
    if not client.is_available:
        raise SystemExit(
            "GEMINI_API_KEY is not set. This script generates ground-truth routing "
            "labels by actually calling the LLM (app.agent.router._llm_classify) — "
            "there is no rule-based substitute for that, since the whole point of "
            "the learned router is to measure agreement with genuine LLM judgment. "
            "Set GEMINI_API_KEY in .env and re-run."
        )

    eval_set = load_eval_set()
    labels = []
    for q in eval_set:
        intent = extract_entities(q["query"])
        decision = _llm_classify(intent)
        if decision is None:
            print(f"warning: LLM classification failed for {q['id']!r}, skipping")
            continue
        labels.append({"id": q["id"], "query": q["query"], "difficulty": q["difficulty"], "llm_tier": decision.tier})
        print(f"  {q['id']:24s} -> {decision.tier}")

    OUT_PATH.write_text(json.dumps(labels, indent=2))
    print(f"\nwrote {len(labels)} LLM-labeled queries to {OUT_PATH}")


if __name__ == "__main__":
    main()
