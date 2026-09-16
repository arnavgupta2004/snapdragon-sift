from app.retrieval import keyword_search, semantic_search
from app.retrieval.base import ScoredFile
from app.retrieval.hybrid_fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank


class TestReciprocalRankFusion:
    def test_empty_input_returns_empty(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_list_preserves_order(self):
        lst = [ScoredFile(file_id=i, score=1.0, source="test") for i in [5, 3, 9]]
        fused = reciprocal_rank_fusion([lst])
        assert [r.file_id for r in fused] == [5, 3, 9]

    def test_agreement_across_lists_boosts_rank(self):
        # file 1 is #1 in list A and #3 in list B; file 2 is #1 in list B only.
        list_a = [ScoredFile(file_id=1, score=1.0, source="a"), ScoredFile(file_id=8, score=0.5, source="a")]
        list_b = [
            ScoredFile(file_id=2, score=1.0, source="b"),
            ScoredFile(file_id=9, score=0.7, source="b"),
            ScoredFile(file_id=1, score=0.6, source="b"),
        ]
        fused = reciprocal_rank_fusion([list_a, list_b])
        # file 1 appears near the top of both lists -> should outrank file 2, which is
        # top of only one list.
        ranked_ids = [r.file_id for r in fused]
        assert ranked_ids.index(1) < ranked_ids.index(2)

    def test_scores_normalized_to_unit_interval(self):
        list_a = [ScoredFile(file_id=i, score=1.0, source="a") for i in range(5)]
        fused = reciprocal_rank_fusion([list_a])
        assert fused[0].score == 1.0
        assert all(0.0 <= r.score <= 1.0 for r in fused)

    def test_real_retrievers_fuse_without_error(self):
        query = "quarterly budget report numbers"
        kw = keyword_search.search(query, limit=10)
        sem = semantic_search.search(query, limit=10)
        fused = reciprocal_rank_fusion([kw, sem], limit=10)
        assert fused
        assert all(r.source == "hybrid" for r in fused)


class TestReranker:
    def test_empty_candidates_returns_empty(self):
        assert rerank("anything", []) == []

    def test_reranks_real_candidates_and_preserves_ids(self):
        query = "find that thing I was working on about audio deepfakes"
        sem = semantic_search.search(query, limit=8)
        candidate_ids = {c.file_id for c in sem}

        reranked = rerank(query, sem, top_k=5)

        assert len(reranked) == 5
        assert all(r.file_id in candidate_ids for r in reranked)
        assert all(r.source == "reranker" for r in reranked)

    def test_scores_normalized_and_descending(self):
        query = "budget forecast"
        candidates = semantic_search.search(query, limit=8)
        reranked = rerank(query, candidates)
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_image_sourced_candidates_bypass_cross_encoder(self):
        """A candidate whose only evidence is a CLIP visual match must keep its
        original score/identity rather than being re-scored by the text-only
        cross-encoder, which never sees the actual pixels — see reranker.py's
        module docstring for the eval finding this fixed."""
        image_candidate = ScoredFile(
            file_id=1, score=0.91, source="hybrid",
            explanation="RRF fusion of ['image_semantic']",
        )
        text_candidate = ScoredFile(
            file_id=2, score=0.5, source="hybrid",
            explanation="RRF fusion of ['keyword', 'semantic']",
        )
        reranked = rerank("some query", [image_candidate, text_candidate])

        by_id = {r.file_id: r for r in reranked}
        assert by_id[1].score == 0.91  # untouched
        assert by_id[1].source == "hybrid"
        assert by_id[2].source == "reranker"  # actually went through the cross-encoder

    def test_all_image_sourced_candidates_never_calls_cross_encoder(self, monkeypatch):
        def _fail(*args, **kwargs):
            raise AssertionError("cross-encoder should not be invoked for pure-image candidate sets")

        monkeypatch.setattr("app.retrieval.reranker._get_cross_encoder", _fail)

        candidates = [
            ScoredFile(file_id=1, score=0.8, source="hybrid", explanation="RRF fusion of ['image_semantic']"),
            ScoredFile(file_id=2, score=0.6, source="hybrid", explanation="RRF fusion of ['image_semantic']"),
        ]
        reranked = rerank("a photo with a blue triangle", candidates)
        assert [r.file_id for r in reranked] == [1, 2]
