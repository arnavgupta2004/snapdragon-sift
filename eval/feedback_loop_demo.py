#!/usr/bin/env python3
"""Demonstrates the feedback loop actually closing: simulate several rounds of user
feedback, retrain the LightGBM ranker after each round, and show ranking quality
(NDCG@10 against the user's own access history — same ground truth methodology as
eval/personalization_lift.py) trending upward as feedback accumulates.

Simulated feedback per round is a plausible model of real usage, not hand-picked to
flatter the result: for each query, whatever the *current* model ranks in the top 5,
thumbs-up the ones that are in the user's personal ground truth and thumbs-down the
top 2 that aren't. This is what a real user roughly does — reward good picks, downvote
bad ones near the top — and it's exactly the FeedbackEvent flow POST /api/feedback and
the dev UI's thumbs buttons write.

Clears app.models.FeedbackEvent before starting, for a reproducible demo run.

    python eval/feedback_loop_demo.py [--persona priya_grad_student] [--rounds 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import delete, select  # noqa: E402

from app.agent.query_understanding import extract_entities  # noqa: E402
from app.db import get_session  # noqa: E402
from app.models import FeedbackEvent, UserRecord  # noqa: E402
from app.personalization.feedback import record_feedback  # noqa: E402
from app.personalization.learned_ranker import LearnedPersonalizer  # noqa: E402
from app.personalization.retrain import train_and_save  # noqa: E402
from eval.ablation_study import _ids, stage_plus_reranker  # noqa: E402
from eval.metrics import ndcg_at_k  # noqa: E402
from eval.personalization_lift import personally_relevant_queries  # noqa: E402
from eval.run_benchmark import RESULTS_DIR, load_eval_set  # noqa: E402

K_NDCG = 10
TOP_K_FOR_FEEDBACK = 5
N_THUMBS_DOWN = 2


def _reset_feedback() -> None:
    with get_session() as session:
        session.exec(delete(FeedbackEvent))
        session.commit()


def _measure_ndcg(user_id: int, queries: list[dict], ranker: LearnedPersonalizer) -> float:
    scores = []
    for q in queries:
        intent = extract_entities(q["query"])
        candidates = stage_plus_reranker(intent)
        ranked_ids = _ids(ranker.rank(user_id, candidates)) if ranker.is_available else _ids(candidates)
        scores.append(ndcg_at_k(ranked_ids, q["personal_relevant_ids"], K_NDCG))
    return mean(scores) if scores else 0.0


def _simulate_feedback_round(user_id: int, queries: list[dict], ranker: LearnedPersonalizer) -> int:
    n_events = 0
    for q in queries:
        intent = extract_entities(q["query"])
        candidates = stage_plus_reranker(intent)
        ranked = ranker.rank(user_id, candidates) if ranker.is_available else candidates
        top = ranked[:TOP_K_FOR_FEEDBACK]

        thumbs_down_budget = N_THUMBS_DOWN
        for r in top:
            if r.file_id in q["personal_relevant_ids"]:
                record_feedback(user_id, r.file_id, q["query"], "thumbs_up")
                n_events += 1
            elif thumbs_down_budget > 0:
                record_feedback(user_id, r.file_id, q["query"], "thumbs_down")
                thumbs_down_budget -= 1
                n_events += 1
    return n_events


def run_demo(persona_key: str, rounds: int) -> dict:
    with get_session() as session:
        user = session.exec(select(UserRecord).where(UserRecord.persona_key == persona_key)).first()
    if user is None:
        raise SystemExit(f"no user with persona_key={persona_key!r}")

    eval_set = load_eval_set()
    queries = personally_relevant_queries(user.id, eval_set)
    if not queries:
        raise SystemExit(f"no personally-relevant queries found for {persona_key} — try a different persona")

    print(f"[feedback_loop_demo] persona={persona_key} user_id={user.id} n_queries={len(queries)}")

    _reset_feedback()

    # round 0: train on access log only (no feedback yet), measure baseline
    print("[feedback_loop_demo] round 0: training on access log only (no feedback)...")
    train_and_save(include_feedback=False)
    history = [{"round": 0, "ndcg_at_10": _measure_ndcg(user.id, queries, LearnedPersonalizer()), "n_feedback_events": 0}]
    print(f"  NDCG@10 = {history[0]['ndcg_at_10']:.4f}")

    total_events = 0
    for round_idx in range(1, rounds + 1):
        ranker = LearnedPersonalizer()
        n_new = _simulate_feedback_round(user.id, queries, ranker)
        total_events += n_new
        print(f"[feedback_loop_demo] round {round_idx}: recorded {n_new} feedback events (total={total_events}), retraining...")
        train_and_save(include_feedback=True)
        ndcg = _measure_ndcg(user.id, queries, LearnedPersonalizer())
        history.append({"round": round_idx, "ndcg_at_10": ndcg, "n_feedback_events": total_events})
        print(f"  NDCG@10 = {ndcg:.4f}")

    return {"persona": persona_key, "user_id": user.id, "n_queries": len(queries), "history": history}


def plot_history(result: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [h["round"] for h in result["history"]]
    ndcgs = [h["ndcg_at_10"] for h in result["history"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rounds, ndcgs, marker="o", linewidth=2)
    ax.set_xlabel("feedback round")
    ax.set_ylabel("NDCG@10 (vs. personal access history)")
    ax.set_title(f"Feedback loop: ranking quality over rounds ({result['persona']})")
    ax.set_xticks(rounds)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", default="priya_grad_student")
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = run_demo(args.persona, args.rounds)

    summary_path = RESULTS_DIR / "feedback_loop_demo_summary.json"
    summary_path.write_text(json.dumps(result, indent=2))

    plot_path = RESULTS_DIR / "feedback_loop_demo.png"
    plot_history(result, plot_path)

    first, last = result["history"][0]["ndcg_at_10"], result["history"][-1]["ndcg_at_10"]
    print(f"\n[feedback_loop_demo] NDCG@10 went {first:.4f} -> {last:.4f} over {args.rounds} rounds")
    print(f"wrote {summary_path}\nwrote {plot_path}")


if __name__ == "__main__":
    main()
