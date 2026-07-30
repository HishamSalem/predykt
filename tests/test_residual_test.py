"""ResidualRepresentationTester: modes, guards, power, categorical adapters."""
import logging
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from predykt import (ResidualRepresentationTester, OLSEstimator,
                     CatBoostAdapter, PandasCategoricalAdapter)

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
