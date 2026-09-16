"""Smoke tests for the ablation and latency-comparison harnesses, on a tiny subset of
the eval set and a single user — the full runs (all 49 queries x 3 users) are exercised
by actually running the scripts (see eval/results/), not by the test suite, since that
would make `pytest` slow for no additional coverage."""

from eval.ablation_study import STAGE_ORDER, run_ablation, summarize as summarize_ablation
from eval.build_eval_set import main as build_eval_set_main  # noqa: F401 (ensures importable)
from eval.latency_comparison import run_always_full_pipeline, run_comparison, summarize as summarize_latency
from eval.run_benchmark import load_eval_set

_SMALL_SET = load_eval_set()[:3]


class TestAblation:
    def test_all_stages_produce_rows_for_small_set(self):
        rows = run_ablation(eval_set=_SMALL_SET, user_ids=[1])
        assert set(rows.keys()) == set(STAGE_ORDER)
        for stage, stage_rows in rows.items():
            assert len(stage_rows) == len(_SMALL_SET), stage

    def test_summary_has_all_metrics_bounded(self):
        rows = run_ablation(eval_set=_SMALL_SET, user_ids=[1])
        summary = summarize_ablation(rows)
        for stage_summary in summary.values():
            for v in stage_summary.values():
                assert 0.0 <= v <= 1.0


class TestLatencyComparison:
    def test_full_pipeline_always_routes_deep(self):
        out = run_always_full_pipeline("open some_exact_file.py", user_id=1)
        assert out["routing_trace"]["tier"] == "deep"

    def test_full_pipeline_runs_reranker_even_for_filename_query(self):
        # the real router would fast-route this; the always-full baseline must not
        out = run_always_full_pipeline("open some_exact_file.py", user_id=1)
        ran = {s["name"] for s in out["routing_trace"]["stages"] if not s["skipped"]}
        assert "reranker" in ran
        assert "semantic_search" in ran

    def test_comparison_runs_on_small_set(self):
        rows = run_comparison(eval_set=_SMALL_SET, user_ids=[1])
        assert len(rows) == len(_SMALL_SET)
        for r in rows:
            assert r["adaptive_ms"] >= 0
            assert r["full_ms"] >= 0

    def test_summary_reports_speedup(self):
        rows = run_comparison(eval_set=_SMALL_SET, user_ids=[1])
        summary = summarize_latency(rows)
        assert "overall" in summary
        assert summary["overall"]["speedup_x"] > 0
