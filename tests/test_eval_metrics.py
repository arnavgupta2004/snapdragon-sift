import json
from pathlib import Path

from sqlmodel import select

from app.db import get_session
from app.models import FileRecord
from eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = REPO_ROOT / "eval" / "eval_set.json"


class TestPrecisionRecall:
    def test_perfect_precision_and_recall(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0
        assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_no_overlap_is_zero(self):
        assert precision_at_k([1, 2, 3], {9, 8}, k=3) == 0.0
        assert recall_at_k([1, 2, 3], {9, 8}, k=3) == 0.0

    def test_partial_hit(self):
        assert precision_at_k([1, 9, 8], {1, 2}, k=3) == 1 / 3
        assert recall_at_k([1, 9, 8], {1, 2}, k=3) == 1 / 2

    def test_empty_retrieved_precision_is_zero(self):
        assert precision_at_k([], {1}, k=5) == 0.0

    def test_empty_relevant_recall_is_zero(self):
        assert recall_at_k([1, 2], set(), k=5) == 0.0

    def test_k_truncates_retrieved_list(self):
        # relevant item is at rank 4, outside k=3
        assert precision_at_k([9, 8, 7, 1], {1}, k=3) == 0.0
        assert recall_at_k([9, 8, 7, 1], {1}, k=3) == 0.0


class TestNDCG:
    def test_relevant_at_rank_one_scores_higher_than_rank_two(self):
        first = ndcg_at_k([1, 9], {1}, k=5)
        second = ndcg_at_k([9, 1], {1}, k=5)
        assert first > second
        assert first == 1.0  # single relevant item at ideal rank -> perfect NDCG

    def test_no_relevant_items_in_result_is_zero(self):
        assert ndcg_at_k([9, 8, 7], {1}, k=5) == 0.0

    def test_no_relevant_items_at_all_is_zero(self):
        assert ndcg_at_k([1, 2, 3], set(), k=5) == 0.0


class TestMRR:
    def test_hit_at_rank_one(self):
        assert mrr([1, 2, 3], {1}) == 1.0

    def test_hit_at_rank_three(self):
        assert mrr([9, 8, 1], {1}) == 1 / 3

    def test_no_hit_is_zero(self):
        assert mrr([9, 8, 7], {1}) == 0.0


class TestEvalSetIntegrity:
    def test_eval_set_exists_and_well_formed(self):
        assert EVAL_SET_PATH.exists(), "run eval/build_eval_set.py first"
        data = json.loads(EVAL_SET_PATH.read_text())
        assert 40 <= len(data) <= 60
        difficulties = {q["difficulty"] for q in data}
        assert difficulties == {"easy", "medium", "hard"}
        for q in data:
            assert q["query"].strip()
            assert q["ground_truth_file_ids"]

    def test_ground_truth_ids_exist_in_corpus(self):
        data = json.loads(EVAL_SET_PATH.read_text())
        with get_session() as session:
            valid_ids = {f.id for f in session.exec(select(FileRecord)).all()}
        for q in data:
            for file_id in q["ground_truth_file_ids"]:
                assert file_id in valid_ids, f"{q['id']}: ground truth file {file_id} not in corpus"

    def test_easy_queries_have_single_ground_truth_file(self):
        data = json.loads(EVAL_SET_PATH.read_text())
        for q in data:
            if q["difficulty"] == "easy":
                assert len(q["ground_truth_file_ids"]) == 1
