#!/usr/bin/env python3
"""Four-way baseline comparison — the single most persuasive artifact in the whole
project: Precision@5 / NDCG@10 / MRR / mean latency for four systems, so the value of
routing and personalization is measured against real alternatives, not asserted.

  1. naive_keyword     - BM25 over filenames+content only. No agent, no personalization.
  2. naive_semantic     - embed query, top-k cosine similarity only. No routing, no
                          personalization, no reranking. ("what a lazy RAG wrapper looks like")
  3. always_full_pipeline - the real deep-route pipeline with routing disabled (every
                          query pays for hybrid retrieval + reranker + personalization)
  4. full_system        - this project: routed + personalized (LangGraph agent)

    python eval/baseline_comparison.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import select  # noqa: E402

from app.agent.graph import run_query  # noqa: E402
from app.db import get_session  # noqa: E402
from app.models import UserRecord  # noqa: E402
from app.retrieval import keyword_search, semantic_search  # noqa: E402
from eval.latency_comparison import run_always_full_pipeline  # noqa: E402
from eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from eval.run_benchmark import EVAL_SET_PATH, RESULTS_DIR, load_eval_set  # noqa: E402

K_PRECISION, K_RECALL, K_NDCG = 5, 5, 10
BASELINES = ["naive_keyword", "naive_semantic", "always_full_pipeline", "full_system"]


def _quality(retrieved_ids: list[int], relevant: set[int]) -> dict[str, float]:
    return {
        "precision_at_5": precision_at_k(retrieved_ids, relevant, K_PRECISION),
        "recall_at_5": recall_at_k(retrieved_ids, relevant, K_RECALL),
        "ndcg_at_10": ndcg_at_k(retrieved_ids, relevant, K_NDCG),
        "mrr": mrr(retrieved_ids, relevant),
    }


def _timed(fn, *args) -> tuple[list[int], float]:
    t0 = time.perf_counter()
    result = fn(*args)
    ms = (time.perf_counter() - t0) * 1000
    return result, ms


def _naive_keyword_ids(query: str) -> list[int]:
    return [r.file_id for r in keyword_search.search(query, limit=10)]


def _naive_semantic_ids(query: str) -> list[int]:
    return [r.file_id for r in semantic_search.search(query, limit=10)]


def _always_full_ids(query: str, user_id: int) -> list[int]:
    return [r["file_id"] for r in run_always_full_pipeline(query, user_id)["results"]]


def _full_system_ids(query: str, user_id: int) -> list[int]:
    return [r["file_id"] for r in run_query(query, user_id)["results"]]


def run_comparison(eval_set: list[dict] | None = None, user_ids: list[int] | None = None) -> list[dict]:
    eval_set = eval_set if eval_set is not None else load_eval_set()
    if user_ids is None:
        with get_session() as session:
            user_ids = [u.id for u in session.exec(select(UserRecord)).all()]

    # warm up models once before timing
    run_always_full_pipeline("warm up query", user_ids[0])

    rows = []
    for q in eval_set:
        relevant = set(q["ground_truth_file_ids"])

        kw_ids, kw_ms = _timed(_naive_keyword_ids, q["query"])
        sem_ids, sem_ms = _timed(_naive_semantic_ids, q["query"])

        full_pipeline_metrics = {m: [] for m in ("precision_at_5", "recall_at_5", "ndcg_at_10", "mrr")}
        full_pipeline_ms = []
        full_system_metrics = {m: [] for m in ("precision_at_5", "recall_at_5", "ndcg_at_10", "mrr")}
        full_system_ms = []

        for uid in user_ids:
            ids, ms = _timed(_always_full_ids, q["query"], uid)
            for k, v in _quality(ids, relevant).items():
                full_pipeline_metrics[k].append(v)
            full_pipeline_ms.append(ms)

            ids, ms = _timed(_full_system_ids, q["query"], uid)
            for k, v in _quality(ids, relevant).items():
                full_system_metrics[k].append(v)
            full_system_ms.append(ms)

        row = {"id": q["id"], "difficulty": q["difficulty"], "query": q["query"]}
        for k, v in _quality(kw_ids, relevant).items():
            row[f"naive_keyword_{k}"] = v
        row["naive_keyword_ms"] = kw_ms
        for k, v in _quality(sem_ids, relevant).items():
            row[f"naive_semantic_{k}"] = v
        row["naive_semantic_ms"] = sem_ms
        for k, v in full_pipeline_metrics.items():
            row[f"always_full_pipeline_{k}"] = mean(v)
        row["always_full_pipeline_ms"] = mean(full_pipeline_ms)
        for k, v in full_system_metrics.items():
            row[f"full_system_{k}"] = mean(v)
        row["full_system_ms"] = mean(full_system_ms)

        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict[str, dict]:
    summary = {}
    for baseline in BASELINES:
        summary[baseline] = {
            "precision_at_5": mean(r[f"{baseline}_precision_at_5"] for r in rows),
            "recall_at_5": mean(r[f"{baseline}_recall_at_5"] for r in rows),
            "ndcg_at_10": mean(r[f"{baseline}_ndcg_at_10"] for r in rows),
            "mrr": mean(r[f"{baseline}_mrr"] for r in rows),
            "mean_latency_ms": mean(r[f"{baseline}_ms"] for r in rows),
        }
    return summary


def plot_comparison(summary: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    metrics_names = ["precision_at_5", "recall_at_5", "ndcg_at_10", "mrr"]
    x = range(len(BASELINES))
    width = 0.2
    for i, m in enumerate(metrics_names):
        vals = [summary[b][m] for b in BASELINES]
        ax1.bar([xi + i * width for xi in x], vals, width, label=m)
    ax1.set_xticks([xi + 1.5 * width for xi in x])
    ax1.set_xticklabels(BASELINES, rotation=20, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Retrieval quality")
    ax1.legend()

    latencies = [summary[b]["mean_latency_ms"] for b in BASELINES]
    ax2.bar(BASELINES, latencies, color="tab:red")
    ax2.set_yscale("log")
    ax2.set_ylabel("mean latency (ms, log scale)")
    ax2.set_xticks(range(len(BASELINES)))
    ax2.set_xticklabels(BASELINES, rotation=20, ha="right")
    ax2.set_title("Latency")

    fig.suptitle("Four-way baseline comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not EVAL_SET_PATH.exists():
        raise SystemExit("eval/eval_set.json not found. Run eval/build_eval_set.py first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[baseline_comparison] running all four systems across the eval set...")
    rows = run_comparison()
    summary = summarize(rows)

    csv_path = RESULTS_DIR / "baseline_comparison_per_query.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = RESULTS_DIR / "baseline_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plot_path = RESULTS_DIR / "baseline_comparison.png"
    plot_comparison(summary, plot_path)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {csv_path}\nwrote {summary_path}\nwrote {plot_path}")


if __name__ == "__main__":
    main()
