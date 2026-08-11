"""Unit tests for the ModelAdapter protocol (no boosting-library fits)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from predykt import (
    CatBoostAdapter,
    ModelAdapter,
    PandasCategoricalAdapter,
    SklearnAdapter,
    resolve_adapter,
)


@pytest.fixture
def mixed_df():
    return pd.DataFrame({
        "num": [1.0, 2.0, np.nan],
        "cat": ["a", None, "b"],
        "already_cat": pd.Categorical(["x", "y", None]),
    })


class TestCatBoostAdapter:
    def test_prepare_fills_nan_and_stringifies(self, mixed_df):
        ad = CatBoostAdapter(LogisticRegression(), cat_cols=["cat"],
                             na_token="MISSING")
        out = ad.prepare(mixed_df)
        assert out["cat"].tolist() == ["a", "MISSING", "b"]
        # pandas <3 gives object dtype; pandas >=3 gives str dtype -- both are
        # string-valued, which is what CatBoost requires.
        assert pd.api.types.is_string_dtype(out["cat"]) or out["cat"].dtype == object
        # numeric NaN untouched (CatBoost handles numeric NaN natively)
        assert np.isnan(out["num"].iloc[2])

    def test_prepare_catches_stray_category_dtype(self, mixed_df):
        ad = CatBoostAdapter(LogisticRegression(), cat_cols=["cat"])
        out = ad.prepare(mixed_df)
        assert out["already_cat"].tolist() == ["x", "y", "missing"]
        assert set(ad._effective_cats(mixed_df)) == {"cat", "already_cat"}

    def test_prepare_does_not_mutate_input(self, mixed_df):
        before = mixed_df.copy(deep=True)
        CatBoostAdapter(LogisticRegression(), cat_cols=["cat"]).prepare(mixed_df)
        pd.testing.assert_frame_equal(mixed_df, before)

    def test_rejects_cat_features_in_fit_params(self):
        with pytest.raises(ValueError, match="cat_cols"):
            CatBoostAdapter(LogisticRegression(), cat_cols=["cat"],
                            fit_params={"cat_features": ["cat"]})


class TestPandasCategoricalAdapter:
    def test_prepare_casts_and_ignores_missing_cols(self, mixed_df):
        ad = PandasCategoricalAdapter(LogisticRegression(),
                                      cat_cols=["cat", "not_there"])
        out = ad.prepare(mixed_df)
        assert str(out["cat"].dtype) == "category"
        assert "not_there" not in out.columns


class TestCloneAndResolve:
    def test_clone_returns_fresh_instances(self):
        for ad in [SklearnAdapter(LogisticRegression(), {"sample_weight": None}),
                   PandasCategoricalAdapter(LogisticRegression(), ["c"]),
                   CatBoostAdapter(LogisticRegression(), ["c"], "NA")]:
            c = ad.clone()
            assert type(c) is type(ad)
            assert c is not ad and c.model is not ad.model
            assert c.fit_params == ad.fit_params

    def test_resolve_wraps_bare_estimator(self):
        ad = resolve_adapter(LogisticRegression(), {"sample_weight": None})
        assert isinstance(ad, SklearnAdapter)

    def test_resolve_passes_adapter_through(self):
        ad = SklearnAdapter(LogisticRegression())
        assert resolve_adapter(ad) is ad

    def test_resolve_rejects_adapter_plus_fit_params(self):
        with pytest.raises(ValueError, match="inside the adapter"):
            resolve_adapter(SklearnAdapter(LogisticRegression()), {"a": 1})

    def test_sklearn_adapter_end_to_end(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
        y = (X["a"] > 0).astype(int)
        ad = SklearnAdapter(LogisticRegression()).fit(X, y)
        proba = ad.predict_proba(X)
        assert proba.shape == (200, 2)
        assert isinstance(ad, ModelAdapter)
