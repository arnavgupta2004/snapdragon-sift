#!/usr/bin/env python3
"""Adaptive routing vs. an 'always run the full deep pipeline' baseline: mean/p95
latency and LLM-call counts, per difficulty tier and overall, plus the retrieval-
quality delta between the two (this is the honesty check — the whole point of
Objective 3 is large latency savings with a *small* quality cost, not a free lunch).

The 'always full' baseline isn't a fake slow path: it invokes the exact same LangGraph
node functions as the real deep route (app/agent/graph.py), just with the router's
decision forced to "deep" instead of actually classified — so timings are real.

    python eval/latency_comparison.py

Run this after eval/run_benchmark.py (or at least after the corpus/models are warm) —
first-call cold-start (embedding + cross-encoder model load) is excluded by a warm-up
pass, since that one-time cost isn't what "per-query routing savings" is measuring.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import select  # noqa: E402

from app.agent.graph import (  # noqa: E402
    GraphState,
    node_deep_retrieve,
    node_enrich_query,
    node_explain_llm,
    node_extract_entities,
    node_finalize,
    node_personalize,
)
from app.agent.router import RouteDecision  # noqa: E402
from app.db import get_session  # noqa: E402
from app.models import UserRecord  # noqa: E402
from app.agent.graph import run_query  # noqa: E402
from app.tracing import RoutingTrace  # noqa: E402
from eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from eval.run_benchmark import EVAL_SET_PATH, RESULTS_DIR, load_eval_set  # noqa: E402

K_PRECISION, K_RECALL, K_NDCG = 5, 5, 10


def run_always_full_pipeline(query: str, user_id: int, now: datetime | None = None) -> dict:
    """Forces every query through the same pipeline the real deep route uses —
    routing is disabled, not faked, so this is a genuine 'run everything' baseline."""
    now = now or datetime.now()
    trace = RoutingTrace()
    state: GraphState = {"query": query, "user_id": user_id, "now": now, "trace": trace}

    state.update(node_extract_entities(state))

    decision = RouteDecision(
        tier="deep", rationale="always-full-pipeline baseline: routing disabled"
    )
    trace.tier, trace.rationale = decision.tier, decision.rationale
    state["decision"] = decision

    state.update(node_enrich_query(state))
    state.update(node_deep_retrieve(state))
    state.update(node_personalize(state))
    state.update(node_explain_llm(state))
    state.update(node_finalize(state))

    return {"query": query, "results": state["final_results"], "routing_trace": trace.to_dict()}


def _quality(retrieved_ids: list[int], relevant: set[int]) -> dict[str, float]:
    return {
        "precision_at_5": precision_at_k(retrieved_ids, relevant, K_PRECISION),
        "recall_at_5": recall_at_k(retrieved_ids, relevant, K_RECALL),
        "ndcg_at_10": ndcg_at_k(retrieved_ids, relevant, K_NDCG),
        "mrr": mrr(retrieved_ids, relevant),
    }


def _warm_up(user_id: int) -> None:
    """One throwaway deep-pipeline call to pay the model-load cold-start cost before
    timing starts — see module docstring."""
    run_always_full_pipeline("warm up query about nothing in particular", user_id)


def run_comparison(eval_set: list[dict] | None = None, user_ids: list[int] | None = None) -> list[dict]:
    eval_set = eval_set if eval_set is not None else load_eval_set()
    if user_ids is None:
        with get_session() as session:
            user_ids = [u.id for u in session.exec(select(UserRecord)).all()]

    _warm_up(user_ids[0])

    rows = []
    for q in eval_set:
        relevant = set(q["ground_truth_file_ids"])
        for uid in user_ids:
            t0 = time.perf_counter()
            adaptive = run_query(q["query"], uid)
            adaptive_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            full = run_always_full_pipeline(q["query"], uid)
            full_ms = (time.perf_counter() - t0) * 1000

            adaptive_ids = [r["file_id"] for r in adaptive["results"]]
            full_ids = [r["file_id"] for r in full["results"]]

            rows.append(
                {
                    "id": q["id"],
                    "difficulty": q["difficulty"],
                    "user_id": uid,
                    "adaptive_tier": adaptive["routing_trace"]["tier"],
                    "adaptive_ms": adaptive_ms,
                    "adaptive_llm_calls": adaptive["routing_trace"]["llm_call_count"],
                    "full_ms": full_ms,
                    "full_llm_calls": full["routing_trace"]["llm_call_count"],
                    **{f"adaptive_{k}": v for k, v in _quality(adaptive_ids, relevant).items()},
                    **{f"full_{k}": v for k, v in _quality(full_ids, relevant).items()},
                }
            )
    return rows


def _p95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100)[94]


def summarize(rows: list[dict]) -> dict:
    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_difficulty[r["difficulty"]].append(r)
    by_difficulty["overall"] = rows

    summary = {}
    for diff, rs in by_difficulty.items():
        adaptive_latencies = [r["adaptive_ms"] for r in rs]
        full_latencies = [r["full_ms"] for r in rs]
        summary[diff] = {
            "n": len(rs),
            "adaptive_mean_ms": statistics.mean(adaptive_latencies),
            "adaptive_p95_ms": _p95(adaptive_latencies),
            "adaptive_mean_llm_calls": statistics.mean(r["adaptive_llm_calls"] for r in rs),
            "full_mean_ms": statistics.mean(full_latencies),
            "full_p95_ms": _p95(full_latencies),
            "full_mean_llm_calls": statistics.mean(r["full_llm_calls"] for r in rs),
            "speedup_x": statistics.mean(full_latencies) / statistics.mean(adaptive_latencies)
            if statistics.mean(adaptive_latencies) > 0 else float("inf"),
            "adaptive_ndcg_at_10": statistics.mean(r["adaptive_ndcg_at_10"] for r in rs),
            "full_ndcg_at_10": statistics.mean(r["full_ndcg_at_10"] for r in rs),
            "quality_delta_ndcg_at_10": statistics.mean(r["adaptive_ndcg_at_10"] for r in rs)
            - statistics.mean(r["full_ndcg_at_10"] for r in rs),
        }
    return summary


def plot_latency(summary: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diffs = [d for d in ("easy", "medium", "hard", "overall") if d in summary]
    x = range(len(diffs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([xi - width / 2 for xi in x], [summary[d]["adaptive_mean_ms"] for d in diffs], width, label="adaptive routing")
    ax.bar([xi + width / 2 for xi in x], [summary[d]["full_mean_ms"] for d in diffs], width, label="always-full-pipeline")
    ax.set_xticks(list(x))
    ax.set_xticklabels(diffs)
    ax.set_ylabel("mean latency (ms, log scale)")
    ax.set_yscale("log")
    ax.set_title("Adaptive routing vs. always-full-pipeline: mean latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not EVAL_SET_PATH.exists():
        raise SystemExit("eval/eval_set.json not found. Run eval/build_eval_set.py first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[latency_comparison] warming up models, then timing adaptive vs. always-full-pipeline...")
    rows = run_comparison()
    summary = summarize(rows)

    csv_path = RESULTS_DIR / "latency_comparison_per_query.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = RESULTS_DIR / "latency_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plot_path = RESULTS_DIR / "latency_comparison.png"
    plot_latency(summary, plot_path)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {csv_path}\nwrote {summary_path}\nwrote {plot_path}")


if __name__ == "__main__":
    main()
