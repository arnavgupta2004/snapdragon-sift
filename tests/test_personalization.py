from datetime import datetime, timedelta

from sqlmodel import select

from app.db import get_session
from app.models import UserRecord
from app.personalization.personalized_ranker import WeightedSumPersonalizer
from app.personalization.profile_builder import build_user_profile, discover_topic_clusters
from app.personalization.temporal_patterns import current_context_boost, detect_recurring_patterns
from app.retrieval import keyword_search, semantic_search
from app.retrieval.hybrid_fusion import reciprocal_rank_fusion

PERSONA_RECURRING_WEEKDAY = {
    "priya_grad_student": 0,  # Monday
    "david_analyst": 0,       # Monday
    "maria_freelancer": 4,    # Friday
}


def _all_users() -> list[UserRecord]:
    with get_session() as session:
        return session.exec(select(UserRecord)).all()


class TestProfileBuilder:
    def test_profile_built_for_each_user_is_nonempty(self):
        for user in _all_users():
            profile = build_user_profile(user.id)
            assert profile.file_frequency, f"expected access history for {user.name}"
            assert profile.type_affinity
            assert profile.preferred_file_types

    def test_scores_bounded_zero_to_one(self):
        user = _all_users()[0]
        profile = build_user_profile(user.id)
        assert all(0.0 <= v <= 1.0 for v in profile.file_frequency.values())
        assert all(0.0 <= v <= 1.0 for v in profile.type_affinity.values())
        assert all(0.0 <= v <= 1.0 for v in profile.file_recency.values())

    def test_discover_topic_clusters_covers_corpus(self):
        clusters = discover_topic_clusters()
        assert len(clusters) > 300  # full corpus, minus nothing
        assert len(set(clusters.values())) <= 10


class TestTemporalPatterns:
    def test_each_persona_recurring_pattern_detected_on_expected_weekday(self):
        for user in _all_users():
            patterns = detect_recurring_patterns(user.id)
            assert patterns, f"expected a recurring pattern for {user.name}"
            expected_weekday = PERSONA_RECURRING_WEEKDAY[user.persona_key]
            weekdays_found = {p.weekday for p in patterns}
            assert expected_weekday in weekdays_found

    def test_context_boost_empty_on_non_matching_day(self):
        user = _all_users()[0]
        # a time far from any detected pattern's weekday/hour
        off_pattern_time = datetime.now().replace(hour=3) + timedelta(days=(2 - datetime.now().weekday()) % 7)
        patterns = detect_recurring_patterns(user.id)
        pattern_hours = {(p.weekday, p.hour) for p in patterns}
        assert (off_pattern_time.weekday(), off_pattern_time.hour) not in pattern_hours
        boost = current_context_boost(user.id, now=off_pattern_time)
        assert boost == {}

    def test_context_boost_nonempty_on_matching_day_and_hour(self):
        user = _all_users()[0]
        patterns = detect_recurring_patterns(user.id)
        assert patterns
        top = patterns[0]
        # construct a "now" on the same weekday/hour as the detected pattern
        now = datetime.now()
        days_ahead = (top.weekday - now.weekday()) % 7
        matching_time = (now + timedelta(days=days_ahead)).replace(hour=top.hour)
        boost = current_context_boost(user.id, now=matching_time)
        assert set(boost.keys()) & set(top.file_ids)


class TestPersonalizedRanker:
    def test_reranking_preserves_candidate_set(self):
        query = "weekly status report"
        fused = reciprocal_rank_fusion(
            [keyword_search.search(query, limit=15), semantic_search.search(query, limit=15)], limit=10
        )
        user = _all_users()[0]
        ranked = WeightedSumPersonalizer().rank(user.id, fused, now=datetime.now())
        assert {r.file_id for r in ranked} == {c.file_id for c in fused}

    def test_different_users_get_different_rankings_for_same_query(self):
        query = "weekly status report"
        fused = reciprocal_rank_fusion(
            [keyword_search.search(query, limit=15), semantic_search.search(query, limit=15)], limit=10
        )
        users = _all_users()
        ranker = WeightedSumPersonalizer()
        now = datetime.now()
        rankings = [tuple(r.file_id for r in ranker.rank(u.id, fused, now=now)) for u in users]
        assert len(set(rankings)) > 1, "expected personalization to differentiate at least two users"

    def test_empty_candidates_returns_empty(self):
        user = _all_users()[0]
        assert WeightedSumPersonalizer().rank(user.id, []) == []
