"""ResidualRepresentationTester: modes, guards, power, categorical adapters."""
import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from predykt import (
    CatBoostAdapter,
    OLSEstimator,
    PandasCategoricalAdapter,
    ResidualRepresentationTester,
)

PAIR = [("num1", "num2")]
REQUIRED_COLS = {"pair", "representation", "criterion", "beta",
                 "statistic", "pvalue", "pvalue_bh", "rejected"}


def _mode_a(model, df, y, reps, **kw):
    t = ResidualRepresentationTester(model=model, n_folds=3,
                                     criterion=[OLSEstimator()], **kw)
    t.fit(PAIR, df, y, reps)
    return t.results_to_dataframe()


class TestModeB:
    def test_precomputed_residuals_schema(self, binary_data):
        df, y, reps = binary_data
        rng = np.random.default_rng(0)
        resid = 0.4 * reps["num1_x_num2"].to_numpy() + rng.normal(
            scale=0.3, size=len(y))
        t = ResidualRepresentationTester(n_folds=3)
        t.fit(PAIR, df, y, {PAIR[0]: reps}, Y_resid=resid)
        res = t.results_to_dataframe()
        assert REQUIRED_COLS <= set(res.columns)
        assert bool(res["rejected"].iloc[0])  # dependence is injected directly

    def test_requires_model_or_residuals(self, binary_data):
        df, y, reps = binary_data
        with pytest.raises(ValueError):
            ResidualRepresentationTester(n_folds=3).fit(PAIR, df, y, reps)

    def test_length_mismatch_raises(self, binary_data):
        df, y, reps = binary_data
        t = ResidualRepresentationTester(n_folds=3)
        with pytest.raises(ValueError, match="align 1:1"):
            t.fit(PAIR, df, y, {PAIR[0]: reps.iloc[:100]},
                  Y_resid=np.random.default_rng(0).normal(size=len(y)))

    def test_index_order_mismatch_warns(self, binary_data, caplog):
        df, y, reps = binary_data
        shuffled = reps.sample(frac=1.0, random_state=1)  # same rows, new order
        t = ResidualRepresentationTester(n_folds=3)
        with caplog.at_level(logging.WARNING, logger="predykt.residual_test"):
            t.fit(PAIR, df, y, {PAIR[0]: shuffled},
                  Y_resid=np.random.default_rng(0).normal(size=len(y)))
        assert any("verify row alignment" in m for m in caplog.messages)


class TestModeA:
    def test_power_depth1_base_model_must_miss_interaction(self, binary_data):
        """An additive (depth-1) model cannot represent num1*num2, so the
        residual test is required to reject H0. This is the core scientific
        claim of the framework."""
        df, y, reps = binary_data
        res = _mode_a(GradientBoostingClassifier(n_estimators=80, max_depth=1),
                      df[["num1", "num2"]], y, reps)
        assert bool(res["rejected"].iloc[0]), \
            f"expected rejection, p={res['pvalue'].iloc[0]:.3g}"

    def test_backward_compat_bare_estimator(self, binary_data):
        df, y, reps = binary_data
        res = _mode_a(GradientBoostingClassifier(n_estimators=40, max_depth=3),
                      df[["num1", "num2"]], y, reps)
        assert len(res) == 1 and np.isfinite(res["pvalue"].iloc[0])

    def test_fit_params_forwarded_fold_invariant(self, binary_data):
        from lightgbm import LGBMClassifier
        df, y, reps = binary_data
        df_cat = df.copy()
        for c in ["cat1", "cat2"]:
            df_cat[c] = df_cat[c].astype("category")
        res = _mode_a(LGBMClassifier(n_estimators=40, verbosity=-1),
                      df_cat, y, reps,
                      fit_params={"categorical_feature": ["cat1", "cat2"]})
        assert np.isfinite(res["pvalue"].iloc[0])


class TestDeprecatedFwlAlias:
    """predykt.fwl is kept as a shim for one release. The name was wrong:
    FWL residualizes both sides of the regression, this residualizes only the
    outcome."""

    def test_shim_reexports_and_warns(self):
        import importlib
        import sys

        sys.modules.pop("predykt.fwl", None)
        with pytest.warns(DeprecationWarning, match="predykt.fwl is deprecated"):
            fwl = importlib.import_module("predykt.fwl")
        # The star-import must actually re-export something usable, which
        # requires residual_test to define __all__.
        assert fwl.__all__ == ["ResidualRepresentationTester"]
        assert fwl.ResidualRepresentationTester is ResidualRepresentationTester


class TestCategoricalAdapters:
    """The failure modes that motivated the adapter layer: each of these
    crashed on bare model.fit before the patch."""

    def test_catboost_mode_a(self, binary_data):
        # CI installs .[test] so this runs there; without the guard the suite
        # looks broken on any local checkout that lacks catboost.
        pytest.importorskip("catboost")
        from catboost import CatBoostClassifier
        df, y, reps = binary_data
        res = _mode_a(CatBoostAdapter(
            CatBoostClassifier(iterations=30, depth=3, verbose=0),
            cat_cols=["cat1", "cat2"]), df, y, reps)
        assert np.isfinite(res["pvalue"].iloc[0])

    def test_xgboost_enable_categorical_mode_a(self, binary_data):
        from xgboost import XGBClassifier
        df, y, reps = binary_data
        res = _mode_a(PandasCategoricalAdapter(
            XGBClassifier(n_estimators=30, max_depth=3, tree_method="hist",
                          enable_categorical=True, verbosity=0),
            cat_cols=["cat1", "cat2"]), df, y, reps)
        assert np.isfinite(res["pvalue"].iloc[0])

    def test_lightgbm_mode_a(self, binary_data):
        from lightgbm import LGBMClassifier
        df, y, reps = binary_data
        res = _mode_a(PandasCategoricalAdapter(
            LGBMClassifier(n_estimators=30, verbosity=-1),
            cat_cols=["cat1", "cat2"]), df, y, reps)
        assert np.isfinite(res["pvalue"].iloc[0])


# =============================================================================
# refute() and winning_representations(): README-advertised, previously untested
# =============================================================================

def _fit_tester(binary_data, **kw):
    df, y, _ = binary_data
    X = df[["num1", "num2"]]
    reps = pd.DataFrame({
        "true_product": X["num1"] * X["num2"],   # the term actually in the DGP
        "noise":        np.random.default_rng(7).normal(size=len(X)),
    }, index=X.index)
    t = ResidualRepresentationTester(
        model=GradientBoostingClassifier(n_estimators=40, random_state=0),
        criterion=[OLSEstimator()], n_folds=3, random_state=0,
        **{"alpha": 0.05, **kw})
    t.fit(feature_pairs=[("num1", "num2")], X=X, y=y, representations=reps)
    return t


class TestRefute:

    def test_populates_refutation_columns(self, binary_data):
        t = _fit_tester(binary_data)
        before = t.results_to_dataframe()
        assert before["empirical_pvalue"].isna().all(), "should be unset before refute()"

        t.refute(n_permutations=40, n_bootstrap=20)
        after = t.results_to_dataframe()

        assert after["empirical_pvalue"].notna().all()
        assert after["stability_score"].notna().all()
        assert after["empirical_pvalue"].between(0, 1).all()
        assert after["stability_score"].between(0, 1).all()

    def test_discriminates_real_signal_from_noise(self, binary_data):
        """The DGP term should survive refutation; pure noise should not."""
        t = _fit_tester(binary_data)
        t.refute(n_permutations=60, n_bootstrap=25)
        r = t.results_to_dataframe().set_index("representation")

        assert r.loc["true_product", "empirical_pvalue"] < 0.05
        assert r.loc["true_product", "stability_score"] >= 0.8
        assert r.loc["true_product", "robust"]

        assert r.loc["noise", "empirical_pvalue"] > 0.05
        assert not r.loc["noise", "robust"]

    def test_robust_requires_rejected(self, binary_data):
        """robust is a conjunction: it can never be True where rejected is False."""
        t = _fit_tester(binary_data)
        t.refute(n_permutations=30, n_bootstrap=15)
        r = t.results_to_dataframe()
        assert not (r["robust"] & ~r["rejected"]).any()

    def test_before_fit_raises(self):
        t = ResidualRepresentationTester(
            model=GradientBoostingClassifier(n_estimators=5, random_state=0))
        with pytest.raises(RuntimeError, match="fit"):
            t.refute(n_permutations=5, n_bootstrap=5)


class TestWinningRepresentations:

    def test_picks_the_true_term(self, binary_data):
        t = _fit_tester(binary_data)
        win = t.winning_representations()

        assert set(win) == {("num1", "num2")}
        w = win[("num1", "num2")]
        assert w["winner"] is True
        assert w["representation"] == "true_product"
        assert w["stage2_result"] is not None
        assert w["pvalue_bh"] < 0.05

    def test_largest_absolute_statistic_wins(self, binary_data):
        """Documented rule: among survivors, max |statistic| wins."""
        t = _fit_tester(binary_data)
        df = t.results_to_dataframe()
        survivors = df[df["rejected"]]
        expected = survivors.loc[survivors["statistic"].abs().idxmax(), "representation"]
        assert t.winning_representations()[("num1", "num2")]["representation"] == expected

    def test_no_survivor_yields_empty_slot(self, binary_data):
        """An alpha nothing can clear must give an empty slot, not a stale one.

        The winner flag is decided inside fit(), so alpha has to be set there;
        lowering it afterwards leaves the flag untouched.
        """
        t = _fit_tester(binary_data, alpha=1e-12)
        assert not any(r["rejected"] for r in t.results_)

        w = t.winning_representations()[("num1", "num2")]
        assert w["winner"] is False
        assert w["representation"] is None
        assert w["stage2_result"] is None
        assert w["rejected"] is False

    def test_before_fit_raises(self):
        t = ResidualRepresentationTester(
            model=GradientBoostingClassifier(n_estimators=5, random_state=0))
        with pytest.raises(RuntimeError, match="fit"):
            t.winning_representations()
