#!/usr/bin/env python3
"""Runs the full labeled eval set through the actual production system (the LangGraph
agent, routing + personalization included) and reports Precision@5, Recall@5, NDCG@10,
and MRR, broken down by difficulty. Every query is run once per simulated user and
averaged, since ground truth is query-relevance (not user-specific) but the system's
output is personalized per user.

    python eval/run_benchmark.py

Writes eval/results/retrieval_quality_per_query.csv,
eval/results/retrieval_quality_summary.json, and eval/results/retrieval_quality.png.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import select  # noqa: E402

from app.agent.graph import run_query  # noqa: E402
from app.db import get_session  # noqa: E402
from app.models import UserRecord  # noqa: E402
from eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402

EVAL_SET_PATH = REPO_ROOT / "eval" / "eval_set.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

K_PRECISION = 5
K_RECALL = 5
K_NDCG = 10

METRIC_NAMES = ["precision_at_5", "recall_at_5", "ndcg_at_10", "mrr"]


def load_eval_set() -> list[dict]:
    if not EVAL_SET_PATH.exists():
        raise SystemExit("eval/eval_set.json not found. Run eval/build_eval_set.py first.")
    return json.loads(EVAL_SET_PATH.read_text())


def _all_user_ids() -> list[int]:
    with get_session() as session:
        return [u.id for u in session.exec(select(UserRecord)).all()]


def run_benchmark(eval_set: list[dict] | None = None, user_ids: list[int] | None = None) -> list[dict]:
    eval_set = eval_set if eval_set is not None else load_eval_set()
    user_ids = user_ids if user_ids is not None else _all_user_ids()

    rows = []
    for q in eval_set:
        relevant = set(q["ground_truth_file_ids"])
        per_metric: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}

        for uid in user_ids:
            out = run_query(q["query"], uid)
            retrieved_ids = [r["file_id"] for r in out["results"]]
            per_metric["precision_at_5"].append(precision_at_k(retrieved_ids, relevant, K_PRECISION))
            per_metric["recall_at_5"].append(recall_at_k(retrieved_ids, relevant, K_RECALL))
            per_metric["ndcg_at_10"].append(ndcg_at_k(retrieved_ids, relevant, K_NDCG))
            per_metric["mrr"].append(mrr(retrieved_ids, relevant))

        rows.append(
            {
                "id": q["id"],
                "difficulty": q["difficulty"],
                "query": q["query"],
                **{m: mean(vals) for m, vals in per_metric.items()},
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_difficulty[r["difficulty"]].append(r)

    summary = {}
    for diff, rs in by_difficulty.items():
        summary[diff] = {m: mean(r[m] for r in rs) for m in METRIC_NAMES}
        summary[diff]["n"] = len(rs)

    summary["overall"] = {m: mean(r[m] for r in rows) for m in METRIC_NAMES}
    summary["overall"]["n"] = len(rows)
    return summary


def plot_summary(summary: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diffs = [d for d in ("easy", "medium", "hard", "overall") if d in summary]
    x = range(len(diffs))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(METRIC_NAMES):
        vals = [summary[d][m] for d in diffs]
        ax.bar([xi + i * width for xi in x], vals, width, label=m)

    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(diffs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Retrieval quality by difficulty (full system: routed + personalized)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[run_benchmark] running eval set through the full agent graph (all users)...")
    rows = run_benchmark()

    per_query_path = RESULTS_DIR / "retrieval_quality_per_query.csv"
    with open(per_query_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "difficulty", "query", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_path = RESULTS_DIR / "retrieval_quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plot_path = RESULTS_DIR / "retrieval_quality.png"
    plot_summary(summary, plot_path)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {per_query_path}\nwrote {summary_path}\nwrote {plot_path}")


if __name__ == "__main__":
    main()
