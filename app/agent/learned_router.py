"""Learned router: a cheap classifier that approximates the LLM's routing-tier
judgment on borderline queries without calling it — trained on real LLM labels
(eval/router_labels.json, produced by eval/build_router_labels.py, which genuinely
calls the LLM; there is no rule-based substitute for that, since the whole point is to
measure agreement with real LLM judgment).

Used as router.py's first resort for borderline queries (see LOW_CONFIDENCE_THRESHOLD
there): if this model is available and confident, its decision is used with zero LLM
calls; otherwise the real LLM classification (_llm_classify) is the fallback — exactly
per spec: "production default with the LLM as fallback for low-confidence cases."

Falls back to unavailable (classify() returns None) gracefully if no model is trained
yet, same pattern as LearnedPersonalizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.agent.query_understanding import QueryIntent
from app.agent.router_features import router_featurize

MODEL_PATH = Path(__file__).resolve().parent / "models" / "router_classifier.joblib"


@dataclass
class LearnedRouteDecision:
    tier: str
    confidence: float


class LearnedRouter:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._model = None
        if model_path.exists():
            import joblib

            self._model = joblib.load(model_path)

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def classify(self, intent: QueryIntent) -> LearnedRouteDecision | None:
        if self._model is None:
            return None
        import numpy as np

        features = np.array([router_featurize(intent)])
        proba = self._model.predict_proba(features)[0]
        idx = int(np.argmax(proba))
        return LearnedRouteDecision(tier=self._model.classes_[idx], confidence=float(proba[idx]))
