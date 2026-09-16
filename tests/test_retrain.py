from app.personalization.retrain import _split_by_group, train_and_save


class TestSplitByGroup:
    def test_split_preserves_all_rows(self):
        X = list(range(10))
        y = [0, 1, 0, 2, 1, 0, 1, 2, 0, 1]
        groups = [3, 4, 3]  # sums to 10
        weights = [1.0] * 10

        (X_train, y_train, g_train, w_train), (X_val, y_val, g_val, w_val) = _split_by_group(
            X, y, groups, weights, val_frac=0.34
        )

        assert len(X_train) + len(X_val) == len(X)
        assert sum(g_train) == len(X_train)
        assert sum(g_val) == len(X_val)
        assert set(X_train) | set(X_val) == set(X)

    def test_split_is_deterministic_for_fixed_seed(self):
        X = list(range(9))
        y = [0] * 9
        groups = [3, 3, 3]
        weights = [1.0] * 9

        result_a = _split_by_group(X, y, groups, weights, seed=7)
        result_b = _split_by_group(X, y, groups, weights, seed=7)
        assert result_a == result_b


class TestTrainAndSave:
    def test_trains_and_saves_a_model(self, tmp_path):
        model_path = tmp_path / "test_model.txt"
        metrics = train_and_save(include_feedback=False, model_path=model_path)

        assert model_path.exists()
        assert metrics["n_train_rows"] > 0
        assert "feature_importance" in metrics
        assert set(metrics["feature_importance"].keys()) == {
            "relevance_signal", "frequency", "recency", "type_affinity", "cluster_affinity", "is_recurring_now",
        }
