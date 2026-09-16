from datetime import datetime

from app.personalization.features import FEATURE_NAMES, featurize
from app.personalization.learned_ranker import LearnedPersonalizer
from app.personalization.profile_builder import build_user_profile
from app.personalization.training_data import build_training_data
from app.retrieval import keyword_search, semantic_search
from app.retrieval.base import ScoredFile
from app.retrieval.hybrid_fusion import reciprocal_rank_fusion


class TestFeaturize:
    def test_returns_correct_length_and_bounded_values(self):
        profile = build_user_profile(1)
        feats = featurize(
            relevance_signal=0.8, file_id=1, profile=profile,
            file_type="pdf", cluster_id=0, is_recurring_now=True,
        )
        assert len(feats) == len(FEATURE_NAMES)
        assert all(0.0 <= v <= 1.0 for v in feats)

    def test_unknown_file_defaults_to_zero_signals(self):
        profile = build_user_profile(1)
        feats = featurize(
            relevance_signal=0.5, file_id=999999, profile=profile,
            file_type="pdf", cluster_id=None, is_recurring_now=False,
        )
        # frequency/recency for a never-accessed file should be 0
        assert feats[FEATURE_NAMES.index("frequency")] == 0.0
        assert feats[FEATURE_NAMES.index("recency")] == 0.0


class TestBuildTrainingData:
    def test_produces_nonempty_groups_without_feedback(self):
        X, y, groups, weights = build_training_data(include_feedback=False)
        assert X and y and groups
        assert len(X) == len(y) == sum(groups)

    def test_every_group_has_a_positive_label(self):
        X, y, groups, weights = build_training_data(include_feedback=False)
        offset = 0
        for g in groups:
            group_labels = y[offset : offset + g]
            assert max(group_labels) >= 1, "expected at least one non-zero-relevance label per group"
            offset += g

    def test_feature_rows_match_declared_width(self):
        X, y, groups, weights = build_training_data(include_feedback=False)
        assert all(len(row) == len(FEATURE_NAMES) for row in X)

    def test_weights_align_with_rows_and_are_positive(self):
        X, y, groups, weights = build_training_data(include_feedback=False)
        assert len(weights) == len(X)
        assert all(w > 0 for w in weights)


class TestLearnedPersonalizer:
    def test_falls_back_to_unchanged_candidates_without_a_model(self, tmp_path):
        ranker = LearnedPersonalizer(model_path=tmp_path / "does_not_exist.txt")
        assert not ranker.is_available

        candidates = [ScoredFile(file_id=1, score=0.9, source="test"), ScoredFile(file_id=2, score=0.5, source="test")]
        result = ranker.rank(1, candidates)
        assert [r.file_id for r in result] == [1, 2]

    def test_empty_candidates_returns_empty(self):
        ranker = LearnedPersonalizer()
        assert ranker.rank(1, []) == []

    def test_trained_model_reranks_real_candidates(self):
        ranker = LearnedPersonalizer()
        if not ranker.is_available:
            import pytest

            pytest.skip("no trained model at app/personalization/models/lgbm_ranker.txt — run retrain.py first")

        query = "weekly status report"
        fused = reciprocal_rank_fusion(
            [keyword_search.search(query, limit=15), semantic_search.search(query, limit=15)], limit=10
        )
        result = ranker.rank(1, fused, now=datetime.now())

        assert {r.file_id for r in result} == {c.file_id for c in fused}
        assert all(0.0 <= r.score <= 1.0 for r in result)
        assert all(r.source == "learned_personalized" for r in result)
