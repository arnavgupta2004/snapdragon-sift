from eval.personalization_lift import personally_relevant_queries, run_lift_for_user
from eval.run_benchmark import load_eval_set

_EVAL_SET = load_eval_set()


class TestPersonalizationLift:
    def test_personally_relevant_queries_are_a_subset_with_real_overlap(self):
        queries = personally_relevant_queries(1, _EVAL_SET)
        assert len(queries) <= len(_EVAL_SET)
        for q in queries:
            assert q["personal_relevant_ids"]
            assert q["personal_relevant_ids"] <= set(q["ground_truth_file_ids"])

    def test_lift_rows_have_both_ndcg_values_bounded(self):
        rows = run_lift_for_user(1, _EVAL_SET[:10])
        for r in rows:
            assert 0.0 <= r["ndcg_without_personalization"] <= 1.0
            assert 0.0 <= r["ndcg_with_personalization"] <= 1.0
