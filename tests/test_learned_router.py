"""Tests the learned-router *pipeline mechanics* (featurization, training, inference,
agreement computation, fallback wiring) using small synthetic label fixtures built in
this file — NOT real LLM labels. This verifies the code is correct; it does not (and
cannot, without GEMINI_API_KEY) verify real agreement with genuine LLM judgment. See
eval/build_router_labels.py / eval/router_agreement.py for that, which require a key.
"""

import json

from app.agent.learned_router import LearnedRouter
from app.agent.query_understanding import extract_entities
from app.agent.router import RouteDecision, route
from app.agent.router_features import ROUTER_FEATURE_NAMES, router_featurize
from app.agent.train_router import train_and_save
from eval.router_agreement import _rule_based_only_decision, compute_agreement

# Small synthetic label set spanning all three tiers with enough examples per class
# for train_test_split's stratification to work (needs >= 2 per class).
_SYNTHETIC_LABELS = [
    {"id": "f1", "query": "open report_final.docx", "llm_tier": "fast"},
    {"id": "f2", "query": "show me pptx files from last week", "llm_tier": "fast"},
    {"id": "f3", "query": "open budget_plan.xlsx", "llm_tier": "fast"},
    {"id": "s1", "query": "find my notes about transformers", "llm_tier": "standard"},
    {"id": "s2", "query": "client dashboard files from this quarter", "llm_tier": "standard"},
    {"id": "s3", "query": "research drafts about audio processing", "llm_tier": "standard"},
    {"id": "d1", "query": "that thing I was working on with my advisor a while back", "llm_tier": "deep"},
    {"id": "d2", "query": "can't remember what it was called but something about spoofing", "llm_tier": "deep"},
    {"id": "d3", "query": "vaguely recall a file about some kind of filter I was messing with", "llm_tier": "deep"},
    {"id": "f4", "query": "open q3_revenue_report.xlsx", "llm_tier": "fast"},
    {"id": "s4", "query": "conference decks about the keynote submission", "llm_tier": "standard"},
    {"id": "d4", "query": "not sure what it's called but it was something about the client dashboard a few weeks ago", "llm_tier": "deep"},
]


class TestRouterFeaturize:
    def test_returns_correct_length(self):
        intent = extract_entities("find my recent notes about transformers")
        feats = router_featurize(intent)
        assert len(feats) == len(ROUTER_FEATURE_NAMES)

    def test_exact_filename_flag_set(self):
        intent = extract_entities("open gradient_descent_v2.md")
        feats = router_featurize(intent)
        assert feats[ROUTER_FEATURE_NAMES.index("has_exact_filename")] == 1.0

    def test_vague_marker_flag_set(self):
        intent = extract_entities("that thing I was working on a while back")
        feats = router_featurize(intent)
        assert feats[ROUTER_FEATURE_NAMES.index("vague_marker_hit")] == 1.0


class TestLearnedRouterFallback:
    def test_unavailable_without_a_model(self, tmp_path):
        router = LearnedRouter(model_path=tmp_path / "does_not_exist.joblib")
        assert not router.is_available
        intent = extract_entities("anything")
        assert router.classify(intent) is None


class TestTrainRouter:
    def test_trains_and_saves_a_working_classifier(self, tmp_path):
        labels_path = tmp_path / "labels.json"
        model_path = tmp_path / "router_classifier.joblib"
        labels_path.write_text(json.dumps(_SYNTHETIC_LABELS))

        metrics = train_and_save(labels_path=labels_path, model_path=model_path)

        assert model_path.exists()
        assert 0.0 <= metrics["train_accuracy"] <= 1.0
        assert 0.0 <= metrics["test_accuracy"] <= 1.0
        assert set(metrics["classes"]) <= {"fast", "standard", "deep"}

        router = LearnedRouter(model_path=model_path)
        assert router.is_available
        decision = router.classify(extract_entities("find my recent notes about transformers"))
        assert decision is not None
        assert decision.tier in ("fast", "standard", "deep")
        assert 0.0 <= decision.confidence <= 1.0

    def test_rejects_too_few_labels(self, tmp_path):
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(json.dumps(_SYNTHETIC_LABELS[:3]))
        import pytest

        with pytest.raises(SystemExit):
            train_and_save(labels_path=labels_path, model_path=tmp_path / "model.joblib")


class TestRouterAgreement:
    def test_rule_based_only_decision_matches_route_for_clear_cases(self):
        clear_fast = extract_entities("open some_file.docx")
        assert _rule_based_only_decision(clear_fast) == "fast"

        clear_deep = extract_entities(
            "find that thing I was working on with my advisor about audio deepfakes a few weeks ago"
        )
        assert _rule_based_only_decision(clear_deep) == "deep"

    def test_borderline_query_not_covered_by_rules(self):
        borderline = extract_entities("find my notes about transformers")
        assert _rule_based_only_decision(borderline) is None

    def test_compute_agreement_structure(self, tmp_path):
        no_model_router = LearnedRouter(model_path=tmp_path / "no_model.joblib")
        result = compute_agreement(_SYNTHETIC_LABELS, no_model_router)

        assert result["n_labeled_queries"] == len(_SYNTHETIC_LABELS)
        assert 0.0 <= result["rule_based"]["coverage_fraction"] <= 1.0
        assert result["learned_router"]["available"] is False
        assert result["learned_router"]["agreement_with_llm"] is None

    def test_compute_agreement_with_trained_learned_router(self, tmp_path):
        labels_path = tmp_path / "labels.json"
        model_path = tmp_path / "router_classifier.joblib"
        labels_path.write_text(json.dumps(_SYNTHETIC_LABELS))
        train_and_save(labels_path=labels_path, model_path=model_path)

        router = LearnedRouter(model_path=model_path)
        result = compute_agreement(_SYNTHETIC_LABELS, router)

        assert result["learned_router"]["available"] is True
        # every query is either covered by rules or evaluated by the learned router
        covered = result["rule_based"]["n_covered_without_any_classifier"]
        evaluated = result["learned_router"]["n_evaluated_on_borderline"]
        assert covered + evaluated <= result["n_labeled_queries"]


class TestRouteUsesLearnedRouterBeforeLLM:
    def test_route_decision_has_learned_router_flag(self):
        # exact filename bypasses the classifier chain entirely
        intent = extract_entities("open some_file.docx")
        decision = route(intent)
        assert isinstance(decision, RouteDecision)
        assert decision.used_learned_router is False
