from sqlmodel import select

from app.agent.graph import run_query
from app.agent.query_understanding import extract_entities
from app.agent.router import route
from app.db import get_session
from app.models import FileRecord, UserRecord


def _first_user_id() -> int:
    with get_session() as session:
        return session.exec(select(UserRecord)).first().id


class TestQueryUnderstanding:
    def test_exact_filename_detected(self):
        intent = extract_entities("open gradient_descent_v2.md please")
        assert intent.exact_filename == "gradient_descent_v2.md"

    def test_filter_only_query_has_low_residual_content(self):
        intent = extract_entities("show me my pptx files from last week")
        assert intent.filters.file_types == ["pptx"]
        assert intent.filters.modified_after is not None
        assert intent.residual_content_word_count <= 1

    def test_topical_query_with_filter_word_keeps_residual_content(self):
        # regression test: "recent" alone must not zero out the topical content
        intent = extract_entities("find my recent notes about transformers")
        assert intent.filters.modified_after is not None  # "recent" still detected
        assert intent.residual_content_word_count >= 1  # but "transformers" survives

    def test_vague_marker_detected(self):
        intent = extract_entities("that thing I was working on a while back")
        assert intent.vague_marker_hit


class TestRouter:
    def test_exact_filename_routes_fast(self):
        intent = extract_entities("open gradient_descent_v2.md")
        decision = route(intent)
        assert decision.tier == "fast"

    def test_filter_only_query_routes_fast(self):
        intent = extract_entities("show me my pptx files from last week")
        decision = route(intent)
        assert decision.tier == "fast"

    def test_topical_query_with_filter_word_does_not_route_fast(self):
        intent = extract_entities("find my recent notes about transformers")
        decision = route(intent)
        assert decision.tier != "fast"

    def test_vague_long_query_routes_deep(self):
        intent = extract_entities(
            "find that thing I was working on with my advisor about audio deepfakes a few weeks ago"
        )
        decision = route(intent)
        assert decision.tier == "deep"


class TestGraphEndToEnd:
    def test_fast_route_skips_expensive_stages(self):
        out = run_query("show me my pptx files from last week", _first_user_id())
        trace = out["routing_trace"]
        assert trace["tier"] == "fast"
        skipped = {s["name"] for s in trace["stages"] if s["skipped"]}
        assert {"keyword_search", "semantic_search", "reranker", "llm_explanation"} <= skipped
        assert trace["llm_call_count"] == 0

    def test_standard_route_runs_hybrid_but_not_reranker(self):
        out = run_query("find my recent notes about transformers", _first_user_id())
        trace = out["routing_trace"]
        assert trace["tier"] == "standard"
        ran = {s["name"] for s in trace["stages"] if not s["skipped"]}
        assert {"keyword_search", "semantic_search", "hybrid_fusion"} <= ran
        skipped = {s["name"] for s in trace["stages"] if s["skipped"]}
        assert "reranker" in skipped

    def test_deep_route_runs_full_pipeline(self):
        out = run_query(
            "find that thing I was working on with my advisor about audio deepfakes a few weeks ago",
            _first_user_id(),
        )
        trace = out["routing_trace"]
        assert trace["tier"] == "deep"
        ran = {s["name"] for s in trace["stages"] if not s["skipped"]}
        assert {"keyword_search", "semantic_search", "hybrid_fusion", "reranker"} <= ran
        assert out["results"], "deep route should surface results for a well-formed query"

    def test_results_have_required_fields(self):
        out = run_query("show me my pptx files", _first_user_id())
        for r in out["results"]:
            assert {"file_id", "filename", "score", "explanation"} <= r.keys()
            assert 0.0 <= r["score"] <= 1.0  # convex blend of two [0,1] scores stays in [0,1]

    def test_exact_filename_query_ranks_named_file_first_regardless_of_user(self):
        # Regression test: personalization used to be able to bump the exact-named
        # file out of first place for a user with unrelated access history, and a
        # filename's own extension (e.g. ".pptx") used to spuriously trigger a
        # metadata file-type filter that flooded the ranking with unrelated files.
        with get_session() as session:
            target = session.exec(select(FileRecord)).first()
            all_user_ids = [u.id for u in session.exec(select(UserRecord)).all()]

        query = f"open {target.filename}"
        for uid in all_user_ids:
            out = run_query(query, uid)
            assert out["routing_trace"]["tier"] == "fast"
            assert out["results"], f"expected results for {query!r}"
            assert out["results"][0]["file_id"] == target.id, (
                f"user {uid}: expected {target.filename} (id={target.id}) ranked first, "
                f"got {out['results'][0]['filename']} (id={out['results'][0]['file_id']})"
            )

        skipped = {s["name"] for s in out["routing_trace"]["stages"] if s["skipped"]}
        assert "metadata_search" in skipped
        assert "personalization" in skipped
