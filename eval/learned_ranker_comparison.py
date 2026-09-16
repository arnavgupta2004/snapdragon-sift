#!/usr/bin/env python3
"""A/B: WeightedSumPersonalizer (hand-tuned baseline) vs. LearnedPersonalizer
(LightGBM), using the exact same personal-ground-truth methodology as
eval/personalization_lift.py — directly follows up on that script's finding of a
-0.016 NDCG@10 lift for the hand-tuned baseline.

Retrains a clean access-log-only model before comparing (include_feedback=False):
eval/feedback_loop_demo.py's simulated feedback is constructed FROM the eval ground
truth (that's the whole point of that demo — it shows the loop works when users give
real signal), so a model trained on it would leak ground truth into this comparison.
This script deliberately avoids that leak. Note this means running this script after
feedback_loop_demo.py overwrites the feedback-enriched model on disk with a clean one
— re-run feedback_loop_demo.py afterward if you want that state back.

    python eval/learned_ranker_comparison.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import select  # noqa: E402

from app.agent.query_understanding import extract_entities  # noqa: E402
from app.db import get_session  # noqa: E402
from app.models import UserRecord  # noqa: E402
from app.personalization.learned_ranker import LearnedPersonalizer  # noqa: E402
from app.personalization.personalized_ranker import WeightedSumPersonalizer  # noqa: E402
from app.personalization.retrain import train_and_save  # noqa: E402
from eval.ablation_study import _ids, stage_plus_reranker  # noqa: E402
from eval.metrics import ndcg_at_k  # noqa: E402
from eval.personalization_lift import personally_relevant_queries  # noqa: E402
from eval.run_benchmark import EVAL_SET_PATH, RESULTS_DIR, load_eval_set  # noqa: E402

K_NDCG = 10


def run_comparison(eval_set: list[dict] | None = None) -> list[dict]:
    eval_set = eval_set if eval_set is not None else load_eval_set()
    with get_session() as session:
        users = session.exec(select(UserRecord)).all()

    weighted_ranker = WeightedSumPersonalizer()
    learned_ranker = LearnedPersonalizer()

    rows = []
    for user in users:
        queries = personally_relevant_queries(user.id, eval_set)
        for q in queries:
            intent = extract_entities(q["query"])
            candidates = stage_plus_reranker(intent)
            relevant = q["personal_relevant_ids"]

            weighted_ids = _ids(weighted_ranker.rank(user.id, candidates))
            learned_ids = _ids(learned_ranker.rank(user.id, candidates))

            rows.append(
                {
                    "user_id": user.id,
                    "persona": user.persona_key,
                    "query_id": q["id"],
                    "ndcg_weighted_sum": ndcg_at_k(weighted_ids, relevant, K_NDCG),
                    "ndcg_learned": ndcg_at_k(learned_ids, relevant, K_NDCG),
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    by_persona: dict[str, list[dict]] = {}
    for r in rows:
        by_persona.setdefault(r["persona"], []).append(r)

    summary = {}
    for persona, rs in by_persona.items():
        summary[persona] = {
            "n_queries": len(rs),
            "ndcg_weighted_sum": mean(r["ndcg_weighted_sum"] for r in rs),
            "ndcg_learned": mean(r["ndcg_learned"] for r in rs),
            "lift_learned_vs_weighted_sum": mean(r["ndcg_learned"] - r["ndcg_weighted_sum"] for r in rs),
        }
    summary["overall"] = {
        "n_queries": len(rows),
        "ndcg_weighted_sum": mean(r["ndcg_weighted_sum"] for r in rows),
        "ndcg_learned": mean(r["ndcg_learned"] for r in rows),
        "lift_learned_vs_weighted_sum": mean(r["ndcg_learned"] - r["ndcg_weighted_sum"] for r in rows),
    }
    return summary


def plot_comparison(summary: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    personas = list(summary.keys())
    x = range(len(personas))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([xi - width / 2 for xi in x], [summary[p]["ndcg_weighted_sum"] for p in personas], width, label="weighted-sum (hand-tuned)")
    ax.bar([xi + width / 2 for xi in x], [summary[p]["ndcg_learned"] for p in personas], width, label="LightGBM (learned)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(personas, rotation=15, ha="right")
    ax.set_ylabel("NDCG@10 (vs. personal access history)")
    ax.set_title("Learned ranker vs. hand-tuned baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not EVAL_SET_PATH.exists():
        raise SystemExit("eval/eval_set.json not found. Run eval/build_eval_set.py first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[learned_ranker_comparison] retraining a clean access-log-only model (no feedback leakage)...")
    train_and_save(include_feedback=False)

    print("[learned_ranker_comparison] comparing WeightedSumPersonalizer vs LearnedPersonalizer...")
    rows = run_comparison()
    summary = summarize(rows)

    csv_path = RESULTS_DIR / "learned_ranker_comparison_per_query.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = RESULTS_DIR / "learned_ranker_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plot_path = RESULTS_DIR / "learned_ranker_comparison.png"
    plot_comparison(summary, plot_path)

    print(json.dumps(summary, indent=2))

    prior_lift_path = RESULTS_DIR / "personalization_lift_summary.json"
    if prior_lift_path.exists():
        prior = json.loads(prior_lift_path.read_text())
        prior_lift = prior.get("overall", {}).get("lift")
        if prior_lift is not None:
            print(
                f"\n[learned_ranker_comparison] Phase 9 finding: weighted-sum-vs-nothing lift "
                f"was {prior_lift:+.4f} NDCG@10. This run: learned-vs-weighted-sum lift is "
                f"{summary['overall']['lift_learned_vs_weighted_sum']:+.4f} NDCG@10."
            )

    print(f"\nwrote {csv_path}\nwrote {summary_path}\nwrote {plot_path}")


if __name__ == "__main__":
    main()
