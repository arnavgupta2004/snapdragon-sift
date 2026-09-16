from eval.baseline_comparison import BASELINES, run_comparison, summarize
from eval.run_benchmark import load_eval_set

_SMALL_SET = load_eval_set()[:3]


class TestBaselineComparison:
    def test_runs_all_four_baselines_on_small_set(self):
        rows = run_comparison(eval_set=_SMALL_SET, user_ids=[1])
        assert len(rows) == len(_SMALL_SET)
        for r in rows:
            for baseline in BASELINES:
                assert f"{baseline}_precision_at_5" in r
                assert f"{baseline}_ms" in r

    def test_summary_bounded_and_complete(self):
        rows = run_comparison(eval_set=_SMALL_SET, user_ids=[1])
        summary = summarize(rows)
        assert set(summary.keys()) == set(BASELINES)
        for stats in summary.values():
            assert 0.0 <= stats["precision_at_5"] <= 1.0
            assert stats["mean_latency_ms"] >= 0
