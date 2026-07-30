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
                "CI_Low", "CI_High", "OOF_Interaction_AUC",
                "Robust"} <= set(df.columns)
        assert "Instability_Score" not in df.columns
        assert "Per_Interaction_AUC" not in df.columns
        # BH correction now has a real p-value to correct
        assert {"P_Value_Adjusted", "Robust_Adjusted"} <= set(df.columns)


class TestOOFInteractionAUC:
    """In-sample scoring of a model's own SHAP values is circular: it gave
    AUC 0.72 for a pair with no interaction, where the honest answer is ~0.50."""

    def test_null_pair_auc_near_half(self, fast_results):
        _, by_pair = fast_results
        auc = by_pair[NULL_PAIR].oof_interaction_auc
        assert 0.45 <= auc <= 0.55, (
            f"null pair OOF AUC should be ~0.50, got {auc:.4f}"
        )

    def test_true_pair_auc_discriminates(self, fast_results):
        _, by_pair = fast_results
        auc = by_pair[TRUE_PAIR].oof_interaction_auc
        assert auc >= 0.70, f"true pair OOF AUC should be >= 0.70, got {auc:.4f}"

    def test_no_duplicate_auc_field(self, fast_results):
        _, by_pair = fast_results
        r = by_pair[TRUE_PAIR]
        assert not hasattr(r, "per_interaction_auc")
        assert not hasattr(r, "mean_auc")
        assert hasattr(r, "oof_interaction_auc")

    def test_auc_distribution_is_per_fold(self, fast_results):
        tester, by_pair = fast_results
        r = by_pair[TRUE_PAIR]
        assert r.auc_distribution.size == tester.n_folds
        assert np.isfinite(r.std_auc)

    def test_no_direction_invariant_floor_in_executable_code(self):
        """max(auc, 1-auc) floored the metric at 0.5 by construction, so it
        could never report "no discrimination".

        Scans executable code only — string literals and comments are stripped,
        so the docstrings that explain why the floor was removed do not trip it.
        """
        import io
        import tokenize

        from predykt import interaction_stability

        with open(interaction_stability.__file__, encoding="utf-8") as fh:
            source = fh.read()
        code = "".join(
            tok.string
            for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type not in (tokenize.STRING, tokenize.COMMENT)
        )
        collapsed = code.replace(" ", "")
        assert "max(auc,1-auc)" not in collapsed
        assert "max(1-auc,auc)" not in collapsed

    def test_auc_can_fall_below_half(self):
        """Without the floor, a pure-noise pair is free to land on either
        side of 0.5 rather than being pushed above it."""
        rng = np.random.default_rng(3)
        X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
        y = rng.binomial(1, 0.5, size=300)  # pure noise: y independent of X
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_bootstrap=1,
                               n_null=3, alpha=0.5, n_folds=3, random_state=1)
        pair_indices = [(0, 1)]
        obs = it._pair_metrics(X, y, 0, pair_indices)
        signs = {k: obs[k]["sign"] for k in pair_indices}
        oof = it._oof_interaction_auc(X, y, pair_indices, signs, 0)
        auc = oof[(0, 1)]["oof_auc"]
        assert 0.0 <= auc <= 1.0
        assert abs(auc - 0.5) < 0.15, f"pure noise should be near 0.5, got {auc}"

    def test_sign_fixed_once_not_per_fold(self, interaction_dgp):
        """One sign, chosen outside the fold loop. Choosing it per fold
        reintroduces the selection bias cross-fitting removes."""
        X, y = interaction_dgp
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_bootstrap=1,
                               n_null=3, alpha=0.5, n_folds=3, random_state=0)
        pair_indices = [(2, 3)]
        obs = it._pair_metrics(X, y, 0, pair_indices)
        sign = obs[(2, 3)]["sign"]
        assert sign in (1.0, -1.0)

        # Flipping the single fixed sign must flip the OOF AUC about 0.5 —
        # proof that one orientation is applied throughout rather than the
        # better of two being picked per fold.
        a = it._oof_interaction_auc(X, y, pair_indices, {(2, 3): sign}, 0)
        b = it._oof_interaction_auc(X, y, pair_indices, {(2, 3): -sign}, 0)
        assert a[(2, 3)]["oof_auc"] == pytest.approx(
            1.0 - b[(2, 3)]["oof_auc"], abs=1e-9)

class TestReferenceDGPAcceptance:
    """The published acceptance criteria, at full size. ~24s, hence `slow`."""

    @pytest.mark.slow
    def test_reference_dgp_separation(self):
        """n=1500, 4 features, logit = 2.5*x0*x1 + 1.0*x2 + 1.0*x3,
        XGBClassifier n_estimators=150 max_depth=4 subsample=1.0."""
        from xgboost import XGBClassifier
        X, y = _interaction_dgp(1500)
        tester = InteractionTester(
            model_class=XGBClassifier, base_params=XGB_REFERENCE,
            n_bootstrap=20, n_null=100, alpha=0.05, n_folds=5, n_jobs=1,
            random_state=0,
        )
        res = {(r.feature_i, r.feature_j): r
               for r in tester.test_pairs(X, y, [TRUE_PAIR, NULL_PAIR])}

        # Additive null separates the pairs
        assert res[TRUE_PAIR].p_value <= 0.05
        assert res[NULL_PAIR].p_value > 0.05
        # Cross-fitted AUC is honest about the null pair
        assert 0.45 <= res[NULL_PAIR].oof_interaction_auc <= 0.55
        assert res[TRUE_PAIR].oof_interaction_auc >= 0.70


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
        fp = it._subset_fit_params(idx, 10)
        assert np.array_equal(fp["sample_weight"], idx.astype(float))

    def test_sample_weight_is_subset_for_cv_folds(self):
        """A CV fold is shorter than the full data, so an unsubset weight
        vector does not merely misalign — LightGBM rejects it outright."""
        it = InteractionTester(model_class=LGBMClassifier,
                               base_params=LGBM_PARAMS, n_bootstrap=1,
                               n_null=3, alpha=0.5,
                               fit_params={"sample_weight": np.arange(10.0)})
        fold = np.array([0, 2, 4, 6, 8])
        fp = it._subset_fit_params(fold, 10)
        assert len(fp["sample_weight"]) == len(fold)
        assert np.array_equal(fp["sample_weight"], fold.astype(float))

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
        assert "lgbm_oof_auc" in summary.columns
        assert "lgbm_instability" not in summary.columns
        for v in votes:
            expected = sum(r.robust for r in v.algorithm_results.values())
            assert v.n_votes == expected
            assert v.unanimous == (v.n_votes == v.n_algorithms)
