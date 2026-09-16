#!/usr/bin/env python3
"""Trains the learned router classifier on real LLM-labeled data.

    python -m app.agent.train_router

Requires eval/router_labels.json (generate with eval/build_router_labels.py, which
needs GEMINI_API_KEY). There is no synthetic-label fallback here — see that script's
docstring for why.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.agent.learned_router import MODEL_PATH  # noqa: E402
from app.agent.query_understanding import extract_entities  # noqa: E402
from app.agent.router_features import ROUTER_FEATURE_NAMES, router_featurize  # noqa: E402

LABELS_PATH = REPO_ROOT / "eval" / "router_labels.json"


def train_and_save(labels_path: Path = LABELS_PATH, model_path: Path = MODEL_PATH) -> dict:
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    if not labels_path.exists():
        raise SystemExit(
            f"{labels_path} not found. Run `python eval/build_router_labels.py` first "
            "(requires GEMINI_API_KEY — see that script's docstring)."
        )

    labels = json.loads(labels_path.read_text())
    if len(labels) < 10:
        raise SystemExit(f"only {len(labels)} labeled examples in {labels_path} — need more to train on.")

    X = np.array([router_featurize(extract_entities(item["query"])) for item in labels])
    y = np.array([item["llm_tier"] for item in labels])

    classes_present = len(set(y))
    stratify = y if classes_present > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=13, stratify=stratify
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_accuracy": clf.score(X_train, y_train),
        "test_accuracy": clf.score(X_test, y_test),
        "classes": list(clf.classes_),
        "feature_names": ROUTER_FEATURE_NAMES,
    }


def main() -> None:
    metrics = train_and_save()
    print(f"trained on {metrics['n_train']} examples, held out {metrics['n_test']}")
    print(f"train accuracy: {metrics['train_accuracy']:.3f}")
    print(f"test accuracy:  {metrics['test_accuracy']:.3f}")
    print(f"saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
