"""InteractionTester / InteractionVoter: guard, additive null, voting, schema."""
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from predykt import InteractionTester, InteractionVoter

LGBM_PARAMS = {"n_estimators": 15, "max_depth": 3, "verbosity": -1}

# Small but discriminating: separates the true pair from the null pair in ~2s.
XGB_FAST = {"n_estimators": 60, "max_depth": 3, "subsample": 1.0,
            "eval_metric": "logloss", "verbosity": 0}

# The reference DGP from the acceptance criteria.
XGB_REFERENCE = {"n_estimators": 150, "max_depth": 4, "subsample": 1.0,
                 "eval_metric": "logloss", "verbosity": 0}

TRUE_PAIR = ("x0", "x1")
NULL_PAIR = ("x2", "x3")


def _interaction_dgp(n, seed=0):
    """logit = 2.5*x0*x1 + 1.0*x2 + 1.0*x3.

    (x0, x1) genuinely interact. (x2, x3) do NOT — both carry main effects of
    the same magnitude but no interaction, which is exactly the case a
    stability-only screen cannot tell apart from a real one.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)),
                     columns=["x0", "x1", "x2", "x3"])
    logit = 2.5 * X["x0"] * X["x1"] + 1.0 * X["x2"] + 1.0 * X["x3"]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    return X, y


@pytest.fixture(scope="module")
def interaction_dgp():
    return _interaction_dgp(600)


@pytest.fixture(scope="module")
def fast_results(interaction_dgp):
    """One test_pairs run shared by the assertions below (each run costs ~2s)."""
    from xgboost import XGBClassifier
    X, y = interaction_dgp
    tester = InteractionTester(
        model_class=XGBClassifier, base_params=XGB_FAST,
        n_bootstrap=8, n_null=60, alpha=0.05, n_jobs=1, random_state=0,
    )
    results = tester.test_pairs(X, y, [TRUE_PAIR, NULL_PAIR])
    return tester, {(r.feature_i, r.feature_j): r for r in results}


class TestNumericGuard:
    def test_rejects_object_columns_loudly(self, binary_data):
        df, y, _ = binary_data
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_bootstrap=2,
                               n_null=2, alpha=0.5)
        with pytest.raises(ValueError, match="fully numeric"):
            it.test_pairs(df, y, [("num1", "num2")])

    def test_rejects_category_dtype(self, binary_data):
        df, y, _ = binary_data
        df2 = df[["num1", "cat2"]]
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_bootstrap=2,
                               n_null=2, alpha=0.5)
        with pytest.raises(ValueError, match="fully numeric"):
            it.test_pairs(df2, y, [("num1", "cat2")])


class TestAdditiveNull:
    """The core of the 0.2.0 change: `robust` must discriminate.

    Under the old instability_score these assertions were unsatisfiable — with
    a deterministic learner every seed produced a bit-identical fit, so the
    score was exactly 0.0 and `robust` was True for every pair including pure
    noise.
    """

    def test_null_pair_not_robust_true_pair_robust(self, fast_results):
        _, by_pair = fast_results
        true_r, null_r = by_pair[TRUE_PAIR], by_pair[NULL_PAIR]

        assert true_r.robust, (
            f"true interaction should be significant: p={true_r.p_value:.4f}, "
            f"observed={true_r.mean_abs_interaction:.4f}, "
            f"null mean={true_r.null_mean:.4f}"
        )
        assert not null_r.robust, (
            f"null pair must NOT be flagged robust: p={null_r.p_value:.4f}, "
            f"observed={null_r.mean_abs_interaction:.4f}, "
            f"null mean={null_r.null_mean:.4f}"
        )
        assert true_r.p_value <= 0.05
        assert null_r.p_value > 0.05

    def test_observed_statistic_is_non_negative(self, fast_results):
        _, by_pair = fast_results
        for r in by_pair.values():
            assert r.mean_abs_interaction >= 0
            assert np.all(r.null_distribution >= 0)

    def test_bootstrap_ci_is_not_a_decision_rule(self, fast_results):
        """`ci_low > 0` would flag pure noise as robust — the exact trap the
        redesign exists to avoid. Pinned as a test so nobody reintroduces it."""
        _, by_pair = fast_results
        for pair, r in by_pair.items():
            assert r.ci_low > 0, (
                f"{pair}: mean|phi| is strictly positive so its bootstrap CI "
                "can never contain zero — which is why ci_low>0 must not be "
                "used as the robustness criterion"
            )
        # The null pair has ci_low > 0 yet is correctly NOT robust.
        assert by_pair[NULL_PAIR].ci_low > 0
        assert not by_pair[NULL_PAIR].robust

    def test_p_value_respects_resolution_floor(self, fast_results):
        _, by_pair = fast_results
        for r in by_pair.values():
            assert r.p_value >= 1.0 / (r.n_null + 1) - 1e-12
            assert r.p_value <= 1.0

    def test_warns_when_alpha_below_resolution_floor(self):
        with pytest.warns(UserWarning, match="smallest attainable p-value"):
            InteractionTester(model_class=LGBMClassifier,
                              base_params=LGBM_PARAMS,
                              n_null=10, alpha=0.01)

    def test_schema_and_dataframe(self, fast_results):
        tester, by_pair = fast_results
        results = list(by_pair.values())
        r = results[0]
        assert np.isfinite(r.p_value)
        assert np.isfinite(r.mean_abs_interaction)
        assert r.n_bootstrap == 8 and r.n_null == 60
        assert not hasattr(r, "instability_score")
        assert not hasattr(r, "mean_interaction")

        df = tester.results_to_dataframe(results)
        assert {"Feature_i", "Feature_j", "Mean_Abs_Interaction", "P_Value",
                "CI_Low", "CI_High", "Robust"} <= set(df.columns)
        assert "Instability_Score" not in df.columns
        # BH correction now has a real p-value to correct
        assert {"P_Value_Adjusted", "Robust_Adjusted"} <= set(df.columns)

    @pytest.mark.slow
    def test_reference_dgp_separation(self):
        """Full acceptance config: n=1500, XGB n_estimators=150 max_depth=4."""
        from xgboost import XGBClassifier
        X, y = _interaction_dgp(1500)
        tester = InteractionTester(
            model_class=XGBClassifier, base_params=XGB_REFERENCE,
            n_bootstrap=20, n_null=100, alpha=0.05, n_jobs=1, random_state=0,
        )
        res = {(r.feature_i, r.feature_j): r
               for r in tester.test_pairs(X, y, [TRUE_PAIR, NULL_PAIR])}
        assert res[TRUE_PAIR].p_value <= 0.05
        assert res[NULL_PAIR].p_value > 0.05


class TestNullSurrogate:
    def test_depth_param_resolution(self):
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from predykt.interaction_stability import _resolve_depth_param

        # XGBClassifier hides max_depth behind **kwargs; inspect alone misses it
        assert _resolve_depth_param(XGBClassifier, {}) == "max_depth"
        assert _resolve_depth_param(RandomForestClassifier, {}) == "max_depth"
        # An explicitly-set key wins, so no duplicate alias is passed
        assert _resolve_depth_param(RandomForestClassifier,
                                    {"max_depth": 5}) == "max_depth"

    def test_no_depth_param_raises_clearly(self, interaction_dgp):
        from sklearn.linear_model import LogisticRegression
        X, y = interaction_dgp
        it = InteractionTester(model_class=LogisticRegression,
                               base_params={}, n_bootstrap=1, n_null=2,
                               alpha=0.5)
        with pytest.raises(ValueError, match="no recognised depth parameter"):
            it._build_null_surrogate()

    def test_injected_surrogate_is_used(self, interaction_dgp):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        X, y = interaction_dgp
        surrogate = LogisticRegression(max_iter=200)
        it = InteractionTester(
            model_class=GradientBoostingClassifier,
            base_params={"n_estimators": 10, "max_depth": 2},
            n_bootstrap=1, n_null=3, alpha=0.5,
            null_surrogate=surrogate, random_state=0,
        )
        assert it._build_null_surrogate() is surrogate
        p_add = it._additive_probabilities(X, y)
        assert p_add.shape == (len(y),)
        assert np.all((p_add > 0) & (p_add < 1))


class TestDeprecations:
    def test_n_seeds_ctor_alias_warns_and_maps(self):
        with pytest.warns(DeprecationWarning, match="n_seeds is deprecated"):
            it = InteractionTester(model_class=LGBMClassifier,
                                   base_params=LGBM_PARAMS, n_seeds=7,
                                   n_null=60)
        assert it.n_bootstrap == 7

    def test_n_seeds_property_warns(self):
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_bootstrap=4,
                               n_null=60)
        with pytest.warns(DeprecationWarning):
            assert it.n_seeds == 4

    def test_seeds_argument_warns(self, interaction_dgp):
        X, y = interaction_dgp
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_bootstrap=6,
                               n_null=3, alpha=0.5, n_jobs=1)
        with pytest.warns(DeprecationWarning, match="`seeds` argument"):
            res = it.test_pairs(X, y, [TRUE_PAIR], seeds=np.arange(2))
        assert res[0].n_bootstrap == 2


class TestInteractionTester:
    def test_fit_params_full_length_sample_weight(self, interaction_dgp):
        X, y = interaction_dgp
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_bootstrap=3,
                               n_null=3, alpha=0.5, n_jobs=1,
                               fit_params={"sample_weight":
                                           np.where(y == 1, 2.0, 1.0)})
        results = it.test_pairs(X, y, [TRUE_PAIR])
        assert np.isfinite(results[0].p_value)

    def test_sample_weight_is_resampled_with_the_bootstrap_draw(self):
        """An unpermuted weight vector would pair each resampled row with
        another row's weight."""
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_bootstrap=1,
                               n_null=3, alpha=0.5,
                               fit_params={"sample_weight": np.arange(10.0)})
        idx = np.array([3, 3, 7, 0, 1, 9, 9, 2, 5, 4])
        fp = it._resampled_fit_params(idx)
        assert np.array_equal(fp["sample_weight"], idx.astype(float))

    def test_get_top_n(self, interaction_dgp):
        X, y = interaction_dgp
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_bootstrap=2,
                               n_null=2, alpha=0.5)
        top = it.get_top_n_interactions(X, y, n=2)
        assert len(top) == 2
        assert all(isinstance(p, tuple) and len(p) == 2 for p in top)
        # Deliberately NOT asserting that TRUE_PAIR ranks first. With this
        # weak learner (15 stumps of depth 3) the screen actually ranks the
        # null pair (x2, x3) top and the true pair last — the documented
        # false-positive behaviour of a cheap single-fit pre-filter, and the
        # reason test_pairs exists.


class TestInteractionVoter:
    def test_vote_and_summary(self, interaction_dgp):
        X, y = interaction_dgp
        voter = InteractionVoter(
            algorithm_configs={
                "lgbm": {"model_class": LGBMClassifier, "params": LGBM_PARAMS},
            },
            n_bootstrap=3, n_null=20, n_jobs=1,
        )
        votes = voter.vote(X, y, [TRUE_PAIR, NULL_PAIR])
        assert len(votes) == 2
        for v in votes:
            assert 0 <= v.n_votes <= 1 and v.n_algorithms == 1
        summary = voter.summary(votes)
        assert len(summary) == 2
        # Vote semantics must key off the new robust flag
        assert "lgbm_p_value" in summary.columns
        assert "lgbm_robust" in summary.columns
        assert "lgbm_instability" not in summary.columns
        for v in votes:
            expected = sum(r.robust for r in v.algorithm_results.values())
            assert v.n_votes == expected
            assert v.unanimous == (v.n_votes == v.n_algorithms)
