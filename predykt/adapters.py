"""
Model Adapters
==============
Uniform fit/predict interface over heterogeneous gradient-boosting libraries,
so every internal fit site in predykt (cross-fitting folds, seed loops) can
handle native categorical features without user-side wrapper code.

Why this exists
---------------
- CatBoost requires ``cat_features=`` as a *fit argument* (it cannot be
  conveyed through ``sklearn.base.clone`` or constructor params) and rejects
  NaN inside categorical columns.
- LightGBM and XGBoost (``enable_categorical=True``) require pandas
  ``category`` dtypes to be present at fit AND predict time.
- Fit/predict dtype asymmetry is the classic silent-failure source, so an
  adapter owns one ``prepare()`` applied identically on both paths.

Responsibilities of an adapter:
1. ``prepare(X)``       -- dtype/NaN normalisation (fit and predict).
2. ``fit(X, y)``        -- library-correct fit call, including fit kwargs.
3. ``predict_proba(X)`` -- probability predictions on prepared data.
4. ``clone()``          -- fresh, unfitted copy for cross-fitting loops.

Usage
-----
    from predykt import ResidualRepresentationTester, CatBoostAdapter

    adapter = CatBoostAdapter(
        CatBoostClassifier(iterations=200, depth=5, verbose=0),
        cat_cols=["state", "segment"],
    )
    tester = ResidualRepresentationTester(model=adapter, n_folds=5)
    tester.fit(feature_pairs, X, y, representations)   # Mode A now works
"""

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import clone as _sk_clone


class ModelAdapter(ABC):
    """
    Abstract base for model adapters.

    Subclasses must set ``self.model`` and ``self.fit_params`` in their
    constructor and implement ``clone()``. Override ``prepare()`` (and
    ``fit()`` if the library needs fit-time arguments derived from X, as
    CatBoost does).
    """

    model = None
    fit_params: dict = {}

    @abstractmethod
    def clone(self) -> "ModelAdapter":
        """Return a fresh, unfitted copy of this adapter."""

    def prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Dtype/NaN normalisation. Applied at fit AND predict time."""
        return X

    def fit(self, X, y) -> "ModelAdapter":
        self.model.fit(self.prepare(X), y, **self.fit_params)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.prepare(X))


class SklearnAdapter(ModelAdapter):
    """
    Default pass-through adapter for plain sklearn-compatible estimators.
    Preserves the historical predykt behaviour: no data preparation,
    ``clone`` via ``sklearn.base.clone``.

    Parameters
    ----------
    model : sklearn-compatible classifier with predict_proba
    fit_params : dict, optional
        Extra keyword arguments forwarded to ``model.fit()``
        (e.g. ``{"categorical_feature": [...]}`` for LightGBM,
        ``{"sample_weight": w}`` for any estimator).
    """

    def __init__(self, model, fit_params: dict | None = None):
        self.model = model
        self.fit_params = dict(fit_params or {})

    def clone(self) -> "SklearnAdapter":
        return SklearnAdapter(_sk_clone(self.model), self.fit_params)


class PandasCategoricalAdapter(ModelAdapter):
    """
    Adapter for models that consume pandas ``category`` dtype natively:
    LightGBM, and XGBoost constructed with ``enable_categorical=True``.

    Casts ``cat_cols`` to ``category`` dtype on a copy of X at BOTH fit and
    predict time, so fold slicing / reindexing / joins upstream cannot break
    the dtype contract.

    Parameters
    ----------
    model : LGBMClassifier or XGBClassifier(enable_categorical=True)
    cat_cols : list of str
        Columns to cast to ``category``. Missing columns are ignored.
    fit_params : dict, optional
        Extra keyword arguments forwarded to ``model.fit()``.
    """

    def __init__(self, model, cat_cols: list[str],
                 fit_params: dict | None = None):
        self.model = model
        self.cat_cols = list(cat_cols)
        self.fit_params = dict(fit_params or {})

    def prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cat_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X

    def clone(self) -> "PandasCategoricalAdapter":
        return PandasCategoricalAdapter(
            _sk_clone(self.model), self.cat_cols, self.fit_params
        )


class CatBoostAdapter(ModelAdapter):
    """
    Adapter for CatBoost: NaN-safe string cast for categorical columns plus
    ``cat_features=`` passed at fit time.

    ``prepare()`` converts declared categorical columns -- and any stray
    ``category``-dtype columns not declared (which would otherwise crash
    CatBoost) -- to string with NaN replaced by ``na_token``. The same
    normalisation runs on the predict path.

    Parameters
    ----------
    model : catboost.CatBoostClassifier
    cat_cols : list of str
        Categorical columns. Missing columns are ignored.
    na_token : str, default="missing"
        Replacement for NaN inside categorical columns (CatBoost rejects
        NaN in cat features).
    fit_params : dict, optional
        Extra keyword arguments forwarded to ``model.fit()`` in addition
        to ``cat_features`` (which this adapter supplies).
    """

    def __init__(self, model, cat_cols: list[str], na_token: str = "missing",
                 fit_params: dict | None = None):
        self.model = model
        self.cat_cols = list(cat_cols)
        self.na_token = na_token
        self.fit_params = dict(fit_params or {})
        if "cat_features" in self.fit_params:
            raise ValueError(
                "Pass categorical columns via cat_cols=, not "
                "fit_params['cat_features']; the adapter supplies "
                "cat_features itself."
            )

    def _effective_cats(self, X: pd.DataFrame) -> list[str]:
        declared = [c for c in self.cat_cols if c in X.columns]
        stray = [
            c for c in X.select_dtypes(include="category").columns
            if c not in declared
        ]
        return declared + stray

    def prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self._effective_cats(X):
            X[col] = (
                X[col].astype(object).fillna(self.na_token).astype(str)
            )
        return X

    def fit(self, X, y) -> "CatBoostAdapter":
        cats = self._effective_cats(X)
        self.model.fit(self.prepare(X), y, cat_features=cats,
                       **self.fit_params)
        return self

    def clone(self) -> "CatBoostAdapter":
        return CatBoostAdapter(
            _sk_clone(self.model), self.cat_cols, self.na_token,
            self.fit_params
        )


def resolve_adapter(model, fit_params: dict | None = None) -> ModelAdapter:
    """
    Normalise a user-supplied model into a ModelAdapter.

    - ModelAdapter instance -> returned as-is (fit_params must then be None;
      put fit kwargs inside the adapter instead).
    - Bare estimator -> wrapped in SklearnAdapter(model, fit_params).
    """
    if isinstance(model, ModelAdapter):
        if fit_params:
            raise ValueError(
                "fit_params= was given alongside a ModelAdapter; pass fit "
                "kwargs inside the adapter instead."
            )
        return model
    return SklearnAdapter(model, fit_params)
