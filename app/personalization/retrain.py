#!/usr/bin/env python3
"""Trains (or retrains) the LightGBM learning-to-rank personalization model on the
access log plus any accumulated feedback (app.models.FeedbackEvent). This is the
script the feedback loop calls after each round of feedback to refit the ranker —
see eval/feedback_loop_demo.py for a demonstration that ranking quality actually
improves as feedback accumulates.

    python -m app.personalization.retrain [--no-feedback]

Splits by *group* (not by row) into train/validation so no group's candidates leak
across the split, and reports validation NDCG@5/@10 so training quality is visible.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.personalization.features import FEATURE_NAMES  # noqa: E402
from app.personalization.learned_ranker import MODEL_PATH  # noqa: E402
from app.personalization.training_data import build_training_data  # noqa: E402


def _split_by_group(X, y, groups, weights, val_frac: float = 0.2, seed: int = 13):
    rng = random.Random(seed)
    n_groups = len(groups)
    order = list(range(n_groups))
    rng.shuffle(order)
    n_val = max(1, int(n_groups * val_frac))
    val_group_idx = set(order[:n_val])

    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)

    X_train, y_train, g_train, w_train = [], [], [], []
    X_val, y_val, g_val, w_val = [], [], [], []
    for i, g in enumerate(groups):
        start, end = offsets[i], offsets[i + 1]
        if i in val_group_idx:
            X_val.extend(X[start:end])
            y_val.extend(y[start:end])
            w_val.extend(weights[start:end])
            g_val.append(g)
        else:
            X_train.extend(X[start:end])
            y_train.extend(y[start:end])
            w_train.extend(weights[start:end])
            g_train.append(g)
    return (X_train, y_train, g_train, w_train), (X_val, y_val, g_val, w_val)


def train_and_save(include_feedback: bool = True, model_path: Path = MODEL_PATH) -> dict:
    import lightgbm as lgb
    import numpy as np

    X, y, groups, weights = build_training_data(include_feedback=include_feedback)
    if len(groups) < 5:
        raise SystemExit(
            f"Only {len(groups)} training groups available — need data/generate_synthetic_data.py "
            "to have been run (and access log populated) before training."
        )

    (X_train, y_train, g_train, w_train), (X_val, y_val, g_val, w_val) = _split_by_group(X, y, groups, weights)
    X_train, y_train, w_train = np.array(X_train), np.array(y_train), np.array(w_train)
    X_val, y_val, w_val = np.array(X_val), np.array(y_val), np.array(w_val)

    train_set = lgb.Dataset(X_train, label=y_train, group=g_train, weight=w_train, feature_name=FEATURE_NAMES)
    val_set = lgb.Dataset(X_val, label=y_val, group=g_val, weight=w_val, feature_name=FEATURE_NAMES, reference=train_set)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 7,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 13,
        "verbose": -1,
    }

    # Fixed boosting-round count rather than early stopping: with only ~150-300 rows
    # total, early stopping against a small re-shuffled validation split introduces
    # retrain-to-retrain variance (a different best_iteration each call) that isn't
    # related to the actual feedback signal — it drowned out the effect we're trying
    # to observe in eval/feedback_loop_demo.py. A smaller, fixed, regularized model
    # (fewer leaves, feature/bagging subsampling) trades a bit of peak accuracy for
    # much more run-to-run stability, which is what makes round-over-round comparisons
    # meaningful here.
    model = lgb.train(
        params,
        train_set,
        num_boost_round=60,
        valid_sets=[val_set],
        callbacks=[lgb.log_evaluation(0)],
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    metrics = {
        "n_train_groups": len(g_train),
        "n_val_groups": len(g_val),
        "n_train_rows": len(X_train),
        "num_boost_round": model.current_iteration(),
        "val_ndcg": model.best_score.get("valid_0", {}),
        "feature_importance": dict(zip(FEATURE_NAMES, [int(v) for v in model.feature_importance()])),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-feedback", dest="include_feedback", action="store_false", default=True)
    args = parser.parse_args()

    print(f"[retrain] building training data (include_feedback={args.include_feedback})...")
    metrics = train_and_save(include_feedback=args.include_feedback)

    print(f"[retrain] trained on {metrics['n_train_rows']} rows across {metrics['n_train_groups']} groups")
    print(f"[retrain] boosting rounds: {metrics['num_boost_round']}")
    print(f"[retrain] validation NDCG: {metrics['val_ndcg']}")
    print(f"[retrain] feature importance: {metrics['feature_importance']}")
    print(f"[retrain] saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
