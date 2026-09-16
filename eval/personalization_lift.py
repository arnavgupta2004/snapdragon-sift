#!/usr/bin/env python3
"""Personalization lift: NDCG@10 with vs. without the personalization re-ranker, per
simulated user persona.

This is a genuinely different question from eval/ablation_study.py's personalization
row. Ablation measures personalization against *query* relevance (the labeled eval
set's ground truth), where it can legitimately look neutral-to-negative — the ground
truth there has no notion of "this particular user". This script instead asks: for
queries where a user has real history to draw on, does personalization surface files
*that user has actually engaged with* more often than generic retrieval alone? That's
the value proposition personalization is actually supposed to deliver, so it needs its
own ground truth: each user's own access log, intersected with each query's topically
relevant files.

    python eval/personalization_lift.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import select  # noqa: E402

from app.agent.query_understanding import extract_entities  # noqa: E402
from app.db import get_session  # noqa: E402
from app.models import AccessEvent, UserRecord  # noqa: E402
from app.personalization.personalized_ranker import WeightedSumPersonalizer  # noqa: E402
from eval.ablation_study import _ids, stage_plus_reranker  # noqa: E402
from eval.metrics import ndcg_at_k  # noqa: E402
from eval.run_benchmark import EVAL_SET_PATH, RESULTS_DIR, load_eval_set  # noqa: E402

K_NDCG = 10


def _user_accessed_file_ids(user_id: int) -> set[int]:
    with get_session() as session:
        events = session.exec(select(AccessEvent).where(AccessEvent.user_id == user_id)).all()
    return {e.file_id for e in events}


def personally_relevant_queries(user_id: int, eval_set: list[dict]) -> list[dict]:
    """Eval-set queries whose ground truth overlaps with files this user has actually
    accessed before — the personal ground truth is that overlap, not the full query
    ground truth (which includes files the user has never touched and has no basis to
    prefer)."""
    accessed = _user_accessed_file_ids(user_id)
    out = []
    for q in eval_set:
        overlap = set(q["ground_truth_file_ids"]) & accessed
        if overlap:
            out.append({**q, "personal_relevant_ids": overlap})
    return out


def run_lift_for_user(user_id: int, eval_set: list[dict]) -> list[dict]:
    queries = personally_relevant_queries(user_id, eval_set)
    rows = []
    for q in queries:
        intent = extract_entities(q["query"])
        relevant = q["personal_relevant_ids"]

        candidates = stage_plus_reranker(intent)  # retrieval + rerank, no personalization
        base_ids = _ids(candidates)
        personalized_ids = _ids(WeightedSumPersonalizer().rank(user_id, candidates))

        rows.append(
            {
                "user_id": user_id,
                "query_id": q["id"],
                "query": q["query"],
                "n_personal_relevant": len(relevant),
                "ndcg_without_personalization": ndcg_at_k(base_ids, relevant, K_NDCG),
                "ndcg_with_personalization": ndcg_at_k(personalized_ids, relevant, K_NDCG),
            }
        )
    return rows


def main() -> None:
    if not EVAL_SET_PATH.exists():
        raise SystemExit("eval/eval_set.json not found. Run eval/build_eval_set.py first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    eval_set = load_eval_set()
    with get_session() as session:
        users = session.exec(select(UserRecord)).all()

    all_rows = []
    per_persona_summary = {}
    for user in users:
        rows = run_lift_for_user(user.id, eval_set)
        all_rows.extend(rows)
        if rows:
            per_persona_summary[user.persona_key] = {
                "n_queries": len(rows),
                "ndcg_without_personalization": mean(r["ndcg_without_personalization"] for r in rows),
                "ndcg_with_personalization": mean(r["ndcg_with_personalization"] for r in rows),
                "lift": mean(r["ndcg_with_personalization"] - r["ndcg_without_personalization"] for r in rows),
            }
        else:
            per_persona_summary[user.persona_key] = {"n_queries": 0}

    if all_rows:
        per_persona_summary["overall"] = {
            "n_queries": len(all_rows),
            "ndcg_without_personalization": mean(r["ndcg_without_personalization"] for r in all_rows),
            "ndcg_with_personalization": mean(r["ndcg_with_personalization"] for r in all_rows),
            "lift": mean(r["ndcg_with_personalization"] - r["ndcg_without_personalization"] for r in all_rows),
        }

    summary_path = RESULTS_DIR / "personalization_lift_summary.json"
    summary_path.write_text(json.dumps(per_persona_summary, indent=2))

    import csv

    csv_path = RESULTS_DIR / "personalization_lift_per_query.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    plot_path = RESULTS_DIR / "personalization_lift.png"
    _plot(per_persona_summary, plot_path)

    print(json.dumps(per_persona_summary, indent=2))
    print(f"\nwrote {summary_path}\nwrote {csv_path}\nwrote {plot_path}")


def _plot(summary: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    personas = [k for k, v in summary.items() if v.get("n_queries", 0) > 0]
    x = range(len(personas))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([xi - width / 2 for xi in x], [summary[p]["ndcg_without_personalization"] for p in personas], width, label="without personalization")
    ax.bar([xi + width / 2 for xi in x], [summary[p]["ndcg_with_personalization"] for p in personas], width, label="with personalization")
    ax.set_xticks(list(x))
    ax.set_xticklabels(personas, rotation=15, ha="right")
    ax.set_ylabel("NDCG@10 against user's own access history")
    ax.set_title("Personalization lift per persona")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
