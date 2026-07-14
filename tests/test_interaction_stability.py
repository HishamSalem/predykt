"""InteractionTester / InteractionVoter: guard, seed loop, voting, schema."""
import numpy as np
import pytest
from lightgbm import LGBMClassifier
from predykt import InteractionTester, InteractionVoter

LGBM_PARAMS = {"n_estimators": 15, "max_depth": 3, "verbosity": -1}


class TestNumericGuard:
    def test_rejects_object_columns_loudly(self, binary_data):
        df, y, _ = binary_data
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_seeds=2)
        with pytest.raises(ValueError, match="fully numeric"):
            it.test_pairs(df, y, [("num1", "num2")])

    def test_rejects_category_dtype(self, binary_data):
        df, y, _ = binary_data
        df2 = df[["num1", "cat2"]]
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_seeds=2)
        with pytest.raises(ValueError, match="fully numeric"):
            it.test_pairs(df2, y, [("num1", "cat2")])


class TestInteractionTester:
    def test_seed_loop_and_schema(self, numeric_data):
        X, y = numeric_data
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_seeds=3, n_jobs=1)
        results = it.test_pairs(X, y, [("x0", "x1")])
        assert len(results) == 1
        r = results[0]
        assert np.isfinite(r.instability_score)
        assert r.n_seeds == 3
        df = it.results_to_dataframe(results)
        assert {"Feature_i", "Feature_j", "Instability_Score",
                "Robust"} <= set(df.columns)

    def test_fit_params_full_length_sample_weight(self, numeric_data):
        X, y = numeric_data
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_seeds=2, n_jobs=1,
                               fit_params={"sample_weight":
                                           np.where(y == 1, 2.0, 1.0)})
        results = it.test_pairs(X, y, [("x0", "x1")])
        assert np.isfinite(results[0].instability_score)

    def test_get_top_n(self, numeric_data):
        X, y = numeric_data
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_seeds=2)
        top = it.get_top_n_interactions(X, y, n=2)
        assert len(top) == 2
        assert all(isinstance(p, tuple) and len(p) == 2 for p in top)


class TestInteractionVoter:
    def test_vote_and_summary(self, numeric_data):
        X, y = numeric_data
        voter = InteractionVoter(
            algorithm_configs={
                "lgbm": {"model_class": LGBMClassifier, "params": LGBM_PARAMS},
            },
            n_seeds=2, n_jobs=1,
        )
        votes = voter.vote(X, y, [("x0", "x1"), ("x0", "x2")])
        assert len(votes) == 2
        for v in votes:
            assert 0 <= v.n_votes <= 1 and v.n_algorithms == 1
        summary = voter.summary(votes)
        assert len(summary) == 2
