"""
SHAP Interaction Testing against an Additive Null
=================================================

Tests whether a feature pair's SHAP interaction is larger than what an
additive data-generating process would produce, then validates across
algorithm families.

Core idea:
    - Measure the interaction magnitude mean|Φ_ij| over rows.
    - Compare it against a null reference distribution simulated from an
      ADDITIVE surrogate fitted to the same data (p_value, `robust`).
    - Quantify its precision with a row bootstrap (ci_low, ci_high).
    - Confirm it appears across multiple algorithms (InteractionVoter).

The null follows the design in Friedman, J.H. & Popescu, B.E. (2008),
"Predictive Learning via Rule Ensembles", Annals of Applied Statistics
2(3):916-954, §8, where the H-statistic's reference distribution is obtained by
simulating outcomes from a fitted model stripped of the interaction being
tested. Here: fit depth-1 (additive by construction) stumps to (X, y), draw
y* ~ Binomial(1, p_additive), refit the real model class on (X, y*), and
recompute mean|Φ_ij|. Repeating gives the distribution of interaction magnitude
attributable to noise and to the estimator's own bias, against which the
observed magnitude is scored.

CALIBRATION LIMIT — read this before quoting the p-value:
    The surrogate is an *approximation* to the additive null, not the true
    null. Depth-1 stumps can only represent additive structure, which is the
    property that matters, but they need not recover the true additive
    component of the DGP. The p-value is therefore calibrated only insofar as
    the surrogate approximates the additive part of the data. Treat it as a
    principled screen with a real null, not as an exact test.

What changed in 0.2.0:
    Earlier versions reported an `instability_score`: the proportion of seeds
    on which the signed mean interaction flipped direction. It had no power at
    all under a deterministic learner — with XGBoost at subsample=1.0 every
    seed produces a bit-identical fit, so the score was exactly 0 for every
    pair including pure noise, and `robust` was True for everything. The
    statistic was also signed, and SHAP interaction values are roughly
    sign-symmetric across rows, so the signed mean discarded ~95% of the
    magnitude it was meant to measure. Both the statistic and the decision
    rule are replaced; see CHANGELOG.
"""

import inspect
import multiprocessing
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# Depth parameter names in rough order of preference. A depth of 1 makes any
# tree ensemble additive by construction, which is what the null requires.
_DEPTH_PARAM_NAMES = ("max_depth", "depth")


def _resolve_depth_param(model_class, base_params: dict) -> Optional[str]:
    """
    Name of the depth hyperparameter for model_class, or None if it has none.

    Checks base_params first so an explicitly-set depth key is the one that
    gets overridden — setting a second alias alongside it (CatBoost accepts
    both ``depth`` and ``max_depth``) would be rejected as a duplicate.

    Signature inspection alone is not sufficient: XGBClassifier.__init__
    declares only three explicit parameters and absorbs the rest through
    **kwargs, so ``max_depth`` is invisible to inspect. When a class accepts
    **kwargs and declares no depth parameter explicitly, "max_depth" is assumed
    (it is by far the most common name, and XGBoost's).
    """
    for name in _DEPTH_PARAM_NAMES:
        if name in base_params:
            return name
    try:
        params = inspect.signature(model_class.__init__).parameters
    except (TypeError, ValueError):
        return None
    for name in _DEPTH_PARAM_NAMES:
        if name in params:
            return name
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    return "max_depth" if accepts_kwargs else None


def _shap_interaction_values(model, X: pd.DataFrame, use_gpu: bool = False):
    """SHAP interaction values as an (n, p, p) array for the positive class."""
    if use_gpu:
        try:
            explainer = shap.explainers.GPUTree(
                model, X, feature_perturbation="tree_path_dependent",
            )
            interactions = explainer(X, interactions=True)
        except Exception:
            warnings.warn("GPU explainer failed, falling back to CPU.")
            interactions = shap.TreeExplainer(model).shap_interaction_values(X)
    else:
        interactions = shap.TreeExplainer(model).shap_interaction_values(X)

    # Binary classifiers may return one array per class
    if isinstance(interactions, list):
        interactions = interactions[1]
    return interactions


# =============================================================================
# RESULT CONTAINERS
# =============================================================================

@dataclass
class InteractionResult:
    """Result for a single feature pair from a single algorithm.

    Attributes
    ----------
    mean_abs_interaction : float
        Observed mean|Φ_ij| over rows, from a single fit on the real data.
        Non-negative by construction.
    std_interaction : float
        Standard deviation of mean|Φ_ij| across bootstrap replicates.
    ci_low, ci_high : float
        2.5 / 97.5 percentiles of the bootstrap distribution of
        mean|Φ_ij|.

        DESCRIPTIVE ONLY — NOT A DECISION RULE. This is a precision interval
        on a magnitude, and mean|Φ_ij| is strictly positive for any fitted
        tree ensemble, so the interval can never contain zero. "ci_low > 0" is
        a tautology that flags pure noise as robust; it is not a test against
        a null. Use `p_value` for that.
    p_value : float
        (#{null >= observed} + 1) / (n_null + 1) against the additive null.
        Bounded below by 1/(n_null + 1).
    null_mean : float
        Mean of the null distribution, for context on the observed value.
    robust : bool
        True if p_value < alpha. This is the decision rule.
    """
    feature_i: str
    feature_j: str
    algorithm: str
    mean_abs_interaction: float
    std_interaction: float
    ci_low: float
    ci_high: float
    p_value: float
    null_mean: float
    per_interaction_auc: float
    mean_auc: float
    std_auc: float
    n_bootstrap: int
    n_null: int
    robust: bool
    interaction_distribution: np.ndarray = field(repr=False)
    null_distribution: np.ndarray = field(repr=False)
    auc_distribution: np.ndarray = field(repr=False)


@dataclass
class VoteResult:
    """Cross-algorithm vote result for a single feature pair.

    A pair receives a vote from an algorithm if its interaction magnitude is
    significant against that algorithm's additive null (p_value < alpha).
    """
    feature_i: str
    feature_j: str
    n_votes: int
    n_algorithms: int
    vote_ratio: float
    algorithm_results: Dict[str, InteractionResult]
    unanimous: bool
    mean_auc_across_algorithms: float

    def __repr__(self):
        status = "UNANIMOUS" if self.unanimous else f"{self.n_votes}/{self.n_algorithms}"
        return (
            f"VoteResult({self.feature_i} x {self.feature_j}: "
            f"{status}, mean_auc={self.mean_auc_across_algorithms:.4f})"
        )


# =============================================================================
# CORE: SINGLE ALGORITHM INTERACTION TESTER
# =============================================================================

class InteractionTester:
    """
    Test whether a feature pair's SHAP interaction magnitude exceeds what an
    additive data-generating process would produce.

    This is a hypothesis test against a simulated null, not a stability screen.
    It answers "is this interaction magnitude larger than an additive DGP would
    yield?" — subject to the calibration limit documented on the module.

    Parameters
    ----------
    model_class : class
        Unfitted model class (e.g., XGBClassifier).
    base_params : dict
        Frozen hyperparameters. Must NOT include the random seed param.
    seed_param : str
        Name of the random seed parameter for this model class.
    n_bootstrap : int, default=100
        Row-bootstrap replicates used for the descriptive interval on
        mean|Φ_ij|. Does not affect `robust`.
    n_null : int, default=100
        Replicates drawn from the additive null. The smallest attainable
        p-value is 1/(n_null + 1), so n_null must be large enough to resolve
        `alpha`; a warning is emitted when it is not.
    alpha : float, default=0.05
        Significance level for the null p-value. `robust` is p_value < alpha.
    null_surrogate : estimator or None, default=None
        Additive surrogate used to generate the null. When None, one is built
        from model_class with its depth parameter set to 1. Pass an explicit
        unfitted estimator when model_class has no depth parameter (a clear
        error is raised in that case if this is left None), or when depth-1
        stumps are a poor additive fit for the data.
    n_folds : int, default=5
        Reserved for cross-fitted interaction scoring.
    use_gpu : bool
        Whether to use GPU-accelerated SHAP explainer.
    n_jobs : int
        Parallel jobs across bootstrap and null replicates. -1 for all cores.
    random_state : int or None, default=0
        Seeds the bootstrap resampling and the null outcome draws.
    fit_params : dict, optional
        Extra keyword arguments forwarded to ``model.fit()`` on every fit
        (e.g. ``{"sample_weight": w}``).
    n_seeds : int, optional
        DEPRECATED alias for n_bootstrap. Emits DeprecationWarning.

    Notes
    -----
    COST: one fit on the real data, plus n_bootstrap + n_null fits, plus one
    surrogate fit — each followed by a full SHAP interaction pass, which is the
    dominant term. At the defaults that is 202 fits per call. All pairs are
    scored from each pass, so testing more pairs is nearly free; reduce
    n_bootstrap first if this is too slow, since it does not affect `robust`.
    """

    def __init__(
        self,
        model_class,
        base_params: dict,
        seed_param: str = "random_state",
        n_bootstrap: int = 100,
        n_null: int = 100,
        alpha: float = 0.05,
        null_surrogate=None,
        n_folds: int = 5,
        use_gpu: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = 0,
        fit_params: Optional[dict] = None,
        n_seeds: Optional[int] = None,
    ):
        if n_seeds is not None:
            warnings.warn(
                "n_seeds is deprecated and will be removed in v0.4.0; use "
                "n_bootstrap instead. The procedure no longer refits across "
                "seeds — it bootstraps rows for a descriptive interval and "
                "simulates an additive null for the p-value.",
                DeprecationWarning,
                stacklevel=2,
            )
            n_bootstrap = n_seeds

        self.model_class = model_class
        self.base_params = base_params
        self.seed_param = seed_param
        self.n_bootstrap = n_bootstrap
        self.n_null = n_null
        self.alpha = alpha
        self.null_surrogate = null_surrogate
        self.n_folds = n_folds
        self.use_gpu = use_gpu
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        self.random_state = random_state
        self.fit_params = dict(fit_params or {})

        # Validate seed_param not in base_params
        if seed_param in base_params:
            raise ValueError(
                f"'{seed_param}' should not be in base_params. "
                f"It will be set automatically per seed."
            )

        min_p = 1.0 / (n_null + 1)
        if alpha < min_p:
            warnings.warn(
                f"alpha={alpha} is below the smallest attainable p-value "
                f"1/(n_null+1)={min_p:.4g}, so robust can never be True. "
                f"Raise n_null to at least {int(np.ceil(1 / alpha)) - 1}.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def n_seeds(self) -> int:
        """DEPRECATED alias for n_bootstrap."""
        warnings.warn(
            "n_seeds is deprecated and will be removed in v0.4.0; "
            "use n_bootstrap instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.n_bootstrap

    @staticmethod
    def _validate_numeric_X(X: pd.DataFrame) -> None:
        bad = {c: X[c].dtype for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])}
        if bad:
            details = ", ".join(f"{c}: {dt}" for c, dt in bad.items())
            raise ValueError(
                "InteractionTester requires X to be fully numeric (int, float, "
                f"or bool). Non-numeric columns found: {details}"
            )

    # =========================================================================
    # BUILDING BLOCKS
    # =========================================================================

    def _fit_model(self, X: pd.DataFrame, y: np.ndarray, seed: int,
                   fit_params: Optional[dict] = None):
        params = {**self.base_params, self.seed_param: seed}
        model = self.model_class(**params)
        model.fit(X, y, **(self.fit_params if fit_params is None else fit_params))
        return model

    def _resampled_fit_params(self, idx: np.ndarray) -> dict:
        """
        fit_params with row-aligned entries reindexed to a bootstrap draw.

        Only ``sample_weight`` is handled: it is the one row-aligned fit
        parameter common to every supported library, and leaving it unpermuted
        would silently pair each resampled row with another row's weight. Any
        other row-aligned entry is passed through unchanged and warned about,
        because fit_params is an open dict and there is no general way to tell
        a row-aligned array from a scalar hyperparameter.
        """
        fp = dict(self.fit_params)
        n = len(idx)
        if "sample_weight" in fp and fp["sample_weight"] is not None:
            fp["sample_weight"] = np.asarray(fp["sample_weight"])[idx]
        for key, val in fp.items():
            if key == "sample_weight":
                continue
            if isinstance(val, (np.ndarray, pd.Series, list)) and len(val) == n:
                warnings.warn(
                    f"fit_params['{key}'] looks row-aligned (length {n}) but is "
                    "not resampled with the bootstrap draw; only sample_weight "
                    "is. Results for this replicate may pair rows with the "
                    "wrong values.",
                    UserWarning,
                    stacklevel=2,
                )
        return fp

    def _pair_metrics(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        seed: int,
        pair_indices: List[Tuple[int, int]],
        fit_params: Optional[dict] = None,
    ) -> Dict:
        """
        Fit, compute SHAP interactions, and extract per-pair metrics.

        The interaction statistic is mean|Φ_ij|, NOT the signed mean. SHAP
        interaction values are roughly sign-symmetric across rows, so the
        signed mean collapses towards zero and destroys the very signal it is
        meant to measure — measured at ~3-6% of the magnitude on a DGP with a
        known interaction.
        """
        model = self._fit_model(X, y, seed, fit_params)
        interactions = _shap_interaction_values(model, X, self.use_gpu)

        pair_results = {}
        for idx_i, idx_j in pair_indices:
            interaction_vals = interactions[:, idx_i, idx_j]
            mean_abs = float(np.mean(np.abs(interaction_vals)))

            try:
                auc = roc_auc_score(y, interaction_vals)
                auc = max(auc, 1 - auc)  # direction-invariant
            except ValueError:
                auc = 0.5

            pair_results[(idx_i, idx_j)] = {
                "mean_abs_interaction": mean_abs,
                "auc": auc,
            }

        return pair_results

    def _build_null_surrogate(self):
        """
        Unfitted additive surrogate used to generate the null outcomes.

        Depth-1 boosted stumps are additive by construction: a tree with one
        split is a function of a single feature, so any ensemble of them is a
        sum of univariate terms and can carry no interaction at all.
        """
        if self.null_surrogate is not None:
            return self.null_surrogate

        depth_param = _resolve_depth_param(self.model_class, self.base_params)
        if depth_param is None:
            raise ValueError(
                f"{self.model_class.__name__} has no recognised depth parameter "
                f"(looked for {', '.join(_DEPTH_PARAM_NAMES)}), so an additive "
                "surrogate cannot be built automatically. Pass one explicitly "
                "via null_surrogate=..., e.g. a depth-1 gradient boosting "
                "classifier or a logistic regression on the raw features."
            )

        params = {**self.base_params, depth_param: 1,
                  self.seed_param: self.random_state or 0}
        return self.model_class(**params)

    def _additive_probabilities(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """p_add: fitted event probabilities under the additive surrogate."""
        surrogate = self._build_null_surrogate()
        surrogate.fit(X, y, **self.fit_params)
        p_add = np.asarray(surrogate.predict_proba(X))[:, 1]
        return np.clip(p_add, 1e-6, 1 - 1e-6)

    def _bootstrap_replicate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        pair_indices: List[Tuple[int, int]],
        seed: int,
    ) -> Dict:
        """One row-bootstrap replicate: resample rows, refit, rescore."""
        rng = np.random.default_rng(seed)
        n = len(y)
        idx = rng.choice(n, n, replace=True)
        X_b = X.iloc[idx].reset_index(drop=True)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:  # degenerate draw
            return None
        return self._pair_metrics(X_b, y_b, seed, pair_indices,
                                  fit_params=self._resampled_fit_params(idx))

    def _null_replicate(
        self,
        X: pd.DataFrame,
        p_add: np.ndarray,
        pair_indices: List[Tuple[int, int]],
        seed: int,
    ) -> Dict:
        """One draw from the additive null: y* ~ Binomial(1, p_add), refit."""
        rng = np.random.default_rng(seed)
        y_star = rng.binomial(1, p_add).astype(int)
        if len(np.unique(y_star)) < 2:  # degenerate draw
            return None
        return self._pair_metrics(X, y_star, seed, pair_indices)

    def _run_replicates(self, fn, arg_list, desc):
        """Run replicates honouring n_jobs; drop degenerate (None) draws."""
        if self.n_jobs > 1:
            out = Parallel(n_jobs=self.n_jobs)(
                delayed(fn)(*args) for args in tqdm(arg_list, desc=desc)
            )
        else:
            out = [fn(*args) for args in tqdm(arg_list, desc=desc)]
        return [o for o in out if o is not None]

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def test_pairs(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_pairs: List[Tuple[str, str]],
        seeds: Optional[np.ndarray] = None,
    ) -> List[InteractionResult]:
        """
        Test multiple feature pairs against the additive null.

        All pairs are scored from each model pass, so testing more pairs costs
        almost nothing beyond the SHAP extraction.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Must be fully numeric.
        y : array-like
            Binary target.
        feature_pairs : list of (str, str)
            Feature pairs to test.
        seeds : array-like, optional
            DEPRECATED. The procedure no longer refits across seeds. If
            provided, only its length is used, as n_bootstrap for this call.

        Returns
        -------
        List of InteractionResult, one per pair. `robust` is p_value < alpha
        against the additive null; ci_low / ci_high are descriptive only.
        """
        self._validate_numeric_X(X)

        n_bootstrap = self.n_bootstrap
        if seeds is not None:
            warnings.warn(
                "The `seeds` argument is deprecated and will be removed in "
                "v0.4.0; the procedure no longer refits across seeds. Using "
                "len(seeds) as n_bootstrap for this call.",
                DeprecationWarning,
                stacklevel=2,
            )
            n_bootstrap = len(seeds)

        y = np.asarray(y).ravel()
        columns = X.columns.tolist()

        pair_indices = []
        for feat_i, feat_j in feature_pairs:
            pair_indices.append((columns.index(feat_i), columns.index(feat_j)))

        base_seed = 0 if self.random_state is None else int(self.random_state)

        # --- 1. observed statistic, from a single fit on the real data -------
        observed = self._pair_metrics(X, y, base_seed, pair_indices)

        # --- 2. bootstrap over rows: descriptive interval only ---------------
        boot_results = self._run_replicates(
            self._bootstrap_replicate,
            [(X, y, pair_indices, base_seed + 1_000 + r)
             for r in range(n_bootstrap)],
            desc=f"{self.model_class.__name__} bootstrap",
        )

        # --- 3. additive null: this is what makes `robust` mean anything -----
        p_add = self._additive_probabilities(X, y)
        null_results = self._run_replicates(
            self._null_replicate,
            [(X, p_add, pair_indices, base_seed + 500_000 + r)
             for r in range(self.n_null)],
            desc=f"{self.model_class.__name__} null",
        )
        if not null_results:
            raise RuntimeError(
                "Every additive-null draw was degenerate (single-class y*). "
                "The additive surrogate's fitted probabilities are likely "
                "saturated at 0 or 1; pass a better null_surrogate."
            )

        # --- 4. assemble ----------------------------------------------------
        results = []
        for p_idx, (feat_i, feat_j) in enumerate(feature_pairs):
            key = pair_indices[p_idx]

            observed_stat = observed[key]["mean_abs_interaction"]
            boot_dist = np.array(
                [r[key]["mean_abs_interaction"] for r in boot_results]
            )
            null_dist = np.array(
                [r[key]["mean_abs_interaction"] for r in null_results]
            )
            auc_dist = np.array([observed[key]["auc"]]
                                + [r[key]["auc"] for r in boot_results])

            # +1 conservative correction for finite replicates
            p_value = float(
                (np.sum(null_dist >= observed_stat) + 1) / (len(null_dist) + 1)
            )

            if boot_dist.size:
                ci_low, ci_high = np.percentile(boot_dist, [2.5, 97.5])
                std_interaction = float(np.std(boot_dist))
            else:
                ci_low = ci_high = float("nan")
                std_interaction = float("nan")

            results.append(InteractionResult(
                feature_i=feat_i,
                feature_j=feat_j,
                algorithm=self.model_class.__name__,
                mean_abs_interaction=float(observed_stat),
                std_interaction=std_interaction,
                ci_low=float(ci_low),
                ci_high=float(ci_high),
                p_value=p_value,
                null_mean=float(np.mean(null_dist)),
                per_interaction_auc=float(np.mean(auc_dist)),
                mean_auc=float(np.mean(auc_dist)),
                std_auc=float(np.std(auc_dist)),
                n_bootstrap=len(boot_dist),
                n_null=len(null_dist),
                robust=p_value < self.alpha,
                interaction_distribution=boot_dist,
                null_distribution=null_dist,
                auc_distribution=auc_dist,
            ))

        return results

    def get_top_n_interactions(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n: int = 10,
        seed: int = 42,
    ) -> List[Tuple[str, str]]:
        """
        Quick screening: single-fit SHAP interactions to identify
        candidate pairs for full testing against the null.

        NOTE: This uses a single fit for speed. Pairs selected here may include
        false positives that the full test_pairs run will filter out. This is
        intentional: it's a cheap pre-filter, not a final result.
        """
        self._validate_numeric_X(X)
        model = self._fit_model(X, y, seed)
        interactions = _shap_interaction_values(model, X, use_gpu=False)

        columns = X.columns.tolist()
        pair_scores = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                mean_abs = float(np.abs(interactions[:, i, j]).mean())
                pair_scores.append((columns[i], columns[j], mean_abs))

        pair_scores.sort(key=lambda x: x[2], reverse=True)
        return [(p[0], p[1]) for p in pair_scores[:n]]

    def results_to_dataframe(
        self,
        results: List[InteractionResult],
        correction_method: Optional[str] = "fdr_bh",
    ) -> pd.DataFrame:
        """
        Convert results to DataFrame with optional multiple testing correction.

        The correction is applied to `P_Value`, which is a genuine permutation-
        style p-value against the additive null. Before 0.2.0 the same call
        corrected `instability_score`, a quantity that was not a p-value at
        all, so the adjustment had no inferential meaning.
        """
        df = pd.DataFrame([
            {
                "Feature_i": r.feature_i,
                "Feature_j": r.feature_j,
                "Algorithm": r.algorithm,
                "Mean_Abs_Interaction": r.mean_abs_interaction,
                "Std_Interaction": r.std_interaction,
                "CI_Low": r.ci_low,
                "CI_High": r.ci_high,
                "P_Value": r.p_value,
                "Null_Mean": r.null_mean,
                "Per_Interaction_AUC": r.per_interaction_auc,
                "Std_AUC": r.std_auc,
                "Robust": r.robust,
                "N_Bootstrap": r.n_bootstrap,
                "N_Null": r.n_null,
            }
            for r in results
        ])

        if correction_method and len(df) > 1:
            reject, adj_p, _, _ = multipletests(
                df["P_Value"], alpha=self.alpha, method=correction_method
            )
            df["P_Value_Adjusted"] = adj_p
            df["Robust_Adjusted"] = reject

        return df

    # =========================================================================
    # PLOTTING
    # =========================================================================

    def plot_interaction_distribution(
        self,
        result: InteractionResult,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """Plot the null distribution against the observed statistic."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

        # Left: additive null vs observed — the actual decision
        ax = axes[0]
        sns.histplot(result.null_distribution, kde=True, ax=ax,
                     color="grey", label="Additive null")
        ax.axvline(
            result.mean_abs_interaction, color="g", linestyle="-",
            label=f"Observed: {result.mean_abs_interaction:.6f}", alpha=0.9,
        )
        ax.axvline(
            float(np.percentile(result.null_distribution, 95)),
            color="r", linestyle="--", label="Null 95th pct", alpha=0.7,
        )
        ax.set_title(
            f"Observed vs Additive Null: {result.feature_i} x {result.feature_j}\n"
            f"{result.algorithm} | p={result.p_value:.4f} | "
            f"n_null={result.n_null}"
        )
        ax.set_xlabel("mean |SHAP interaction value|")
        ax.set_ylabel("Frequency")
        ax.legend()

        # Right: bootstrap precision interval — descriptive only
        ax = axes[1]
        sns.histplot(result.interaction_distribution, kde=True, ax=ax)
        ax.axvline(result.ci_low, color="r", linestyle="--",
                   label=f"2.5 pct: {result.ci_low:.6f}", alpha=0.7)
        ax.axvline(result.ci_high, color="r", linestyle="--",
                   label=f"97.5 pct: {result.ci_high:.6f}", alpha=0.7)
        ax.set_title(
            f"Bootstrap precision (DESCRIPTIVE, not a test)\n"
            f"{result.feature_i} x {result.feature_j} | "
            f"n_bootstrap={result.n_bootstrap}"
        )
        ax.set_xlabel("mean |SHAP interaction value|")
        ax.set_ylabel("Frequency")
        ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_convergence(
        self,
        result: InteractionResult,
        figsize: Tuple[int, int] = (12, 5),
    ):
        """
        Plot running mean and std of the statistic across bootstrap replicates.
        Useful for determining whether n_bootstrap is sufficient.
        """
        import matplotlib.pyplot as plt

        dist = result.interaction_distribution
        running_mean = np.cumsum(dist) / np.arange(1, len(dist) + 1)
        running_std = np.array([
            np.std(dist[:i + 1]) for i in range(len(dist))
        ])

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].plot(running_mean)
        axes[0].axhline(result.mean_abs_interaction, color="r", linestyle="--",
                        alpha=0.5, label="Observed (full data)")
        axes[0].set_title(f"Convergence: {result.feature_i} x {result.feature_j}")
        axes[0].set_xlabel("Number of Bootstrap Replicates")
        axes[0].set_ylabel("Running Mean of mean|interaction|")
        axes[0].legend()

        axes[1].plot(running_std)
        axes[1].set_title("Running Std")
        axes[1].set_xlabel("Number of Bootstrap Replicates")
        axes[1].set_ylabel("Std of mean|interaction|")

        plt.tight_layout()
        plt.show()

    def plot_top_interactions(
        self,
        results_df: pd.DataFrame,
        top_n: int = 10,
        color: str = "#1f77b4",
        figsize: Tuple[int, int] = (12, 8),
    ):
        """Plot top feature interactions by per-interaction AUC."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        plot_df = results_df.nlargest(top_n, "Per_Interaction_AUC").copy()
        plot_df["Feature Pair"] = plot_df["Feature_i"] + " x " + plot_df["Feature_j"]

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            x="Per_Interaction_AUC", y="Feature Pair",
            data=plot_df, color=color, ax=ax,
        )
        ax.axvline(0.5, color="r", linestyle="--", alpha=0.5, label="No discrimination")
        ax.set_title(f"Top {top_n} Interactions by Per-Interaction AUC")
        ax.set_xlabel("Mean Per-Interaction AUC")
        ax.legend()
        plt.tight_layout()
        plt.show()


# =============================================================================
# CROSS-ALGORITHM VOTER
# =============================================================================

class InteractionVoter:
    """
    Run interaction testing across multiple algorithms and vote.

    An interaction earns a vote from an algorithm when its magnitude is
    significant against that algorithm's additive null (p_value < alpha).
    Cross-algorithm voting then identifies interactions that are properties of
    the data, not artifacts of a specific algorithm's splitting strategy.

    Parameters
    ----------
    algorithm_configs : dict
        Mapping of name -> dict with:
            "model_class": unfitted model class
            "params": frozen hyperparameters (no seed param)
            "seed_param": name of random seed parameter
            "null_surrogate": optional additive surrogate for that algorithm
    n_bootstrap : int
        Bootstrap replicates per algorithm (descriptive interval).
    n_null : int
        Additive-null replicates per algorithm (drives the p-value).
    alpha : float
        Significance level for the null p-value.
    use_gpu : bool
        GPU acceleration for SHAP.
    n_jobs : int
        Parallel jobs per algorithm.
    random_state : int or None
        Seeds bootstrap resampling and null draws.
    n_seeds : int, optional
        DEPRECATED alias for n_bootstrap.
    """

    def __init__(
        self,
        algorithm_configs: Dict[str, Dict[str, Any]],
        n_bootstrap: int = 100,
        n_null: int = 100,
        alpha: float = 0.05,
        use_gpu: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = 0,
        n_seeds: Optional[int] = None,
    ):
        if n_seeds is not None:
            warnings.warn(
                "n_seeds is deprecated and will be removed in v0.4.0; "
                "use n_bootstrap instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            n_bootstrap = n_seeds

        self.algorithm_configs = algorithm_configs
        self.n_bootstrap = n_bootstrap
        self.n_null = n_null
        self.alpha = alpha

        self.testers = {}
        for name, config in algorithm_configs.items():
            self.testers[name] = InteractionTester(
                model_class=config["model_class"],
                base_params=config["params"],
                seed_param=config.get("seed_param", "random_state"),
                n_bootstrap=n_bootstrap,
                n_null=n_null,
                alpha=alpha,
                null_surrogate=config.get("null_surrogate"),
                use_gpu=use_gpu,
                n_jobs=n_jobs,
                random_state=random_state,
            )

    def vote(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_pairs: List[Tuple[str, str]],
        seeds: Optional[np.ndarray] = None,
    ) -> List[VoteResult]:
        """
        Test each feature pair across all algorithms and tally votes.

        A pair receives a vote from an algorithm if it is robust
        (p_value < alpha against that algorithm's additive null).
        """
        all_results = {}
        for algo_name, tester in self.testers.items():
            print(f"\n{'='*60}")
            print(f"  {algo_name} ({tester.model_class.__name__})")
            print(f"{'='*60}")
            results = tester.test_pairs(X, y, feature_pairs, seeds)
            for r in results:
                all_results[(algo_name, r.feature_i, r.feature_j)] = r

        # Tally votes per pair
        vote_results = []
        n_algorithms = len(self.testers)

        for feat_i, feat_j in feature_pairs:
            algo_results = {}
            votes = 0
            aucs = []

            for algo_name in self.testers:
                r = all_results[(algo_name, feat_i, feat_j)]
                algo_results[algo_name] = r
                if r.robust:
                    votes += 1
                aucs.append(r.per_interaction_auc)

            vote_results.append(VoteResult(
                feature_i=feat_i,
                feature_j=feat_j,
                n_votes=votes,
                n_algorithms=n_algorithms,
                vote_ratio=votes / n_algorithms,
                algorithm_results=algo_results,
                unanimous=(votes == n_algorithms),
                mean_auc_across_algorithms=float(np.mean(aucs)),
            ))

        return sorted(
            vote_results,
            key=lambda v: (-v.n_votes, -v.mean_auc_across_algorithms),
        )

    def summary(self, vote_results: List[VoteResult]) -> pd.DataFrame:
        """Summary DataFrame from vote results."""
        rows = []
        for vr in vote_results:
            row = {
                "Feature_i": vr.feature_i,
                "Feature_j": vr.feature_j,
                "Votes": vr.n_votes,
                "Total_Algorithms": vr.n_algorithms,
                "Vote_Ratio": vr.vote_ratio,
                "Unanimous": vr.unanimous,
                "Mean_AUC": vr.mean_auc_across_algorithms,
            }
            for algo_name, r in vr.algorithm_results.items():
                row[f"{algo_name}_p_value"] = r.p_value
                row[f"{algo_name}_mean_abs_interaction"] = r.mean_abs_interaction
                row[f"{algo_name}_auc"] = r.per_interaction_auc
                row[f"{algo_name}_robust"] = r.robust
            rows.append(row)

        return pd.DataFrame(rows)

    def plot_vote_heatmap(
        self,
        vote_results: List[VoteResult],
        figsize: Tuple[int, int] = (14, 8),
    ):
        """Heatmap: algorithms (columns) x feature pairs (rows), colored by AUC."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        algo_names = list(self.testers.keys())
        pair_labels = [f"{vr.feature_i} x {vr.feature_j}" for vr in vote_results]

        auc_matrix = np.zeros((len(vote_results), len(algo_names)))
        robust_matrix = np.zeros((len(vote_results), len(algo_names)), dtype=bool)

        for i, vr in enumerate(vote_results):
            for j, algo in enumerate(algo_names):
                r = vr.algorithm_results[algo]
                auc_matrix[i, j] = r.per_interaction_auc
                robust_matrix[i, j] = r.robust

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            auc_matrix,
            xticklabels=algo_names,
            yticklabels=pair_labels,
            annot=True, fmt=".3f",
            cmap="RdYlGn", center=0.5,
            vmin=0.45, vmax=0.7,
            ax=ax,
        )

        # Mark robust cells
        for i in range(robust_matrix.shape[0]):
            for j in range(robust_matrix.shape[1]):
                if robust_matrix[i, j]:
                    ax.text(
                        j + 0.5, i + 0.85, "*",
                        ha="center", va="center",
                        fontsize=14, fontweight="bold", color="black",
                    )

        ax.set_title("Per-Interaction AUC by Algorithm (* = robust vs additive null)")
        plt.tight_layout()
        plt.show()


# =============================================================================
# USAGE
# =============================================================================

# if __name__ == "__main__":
#     from xgboost import XGBClassifier
#     from lightgbm import LGBMClassifier
#     from catboost import CatBoostClassifier
#     from sklearn.ensemble import RandomForestClassifier

#     # ---- Frozen hyperparameters per algorithm ----
#     configs = {
#         "rf": {
#             "model_class": RandomForestClassifier,
#             "params": {"n_estimators": 200, "max_depth": 5, "n_jobs": -1},
#             "seed_param": "random_state",
#         },
#         "xgb": {
#             "model_class": XGBClassifier,
#             "params": {
#                 "n_estimators": 200, "max_depth": 5,
#                 "use_label_encoder": False, "eval_metric": "logloss",
#                 "verbosity": 0,
#             },
#             "seed_param": "random_state",
#         },
#         "lgbm": {
#             "model_class": LGBMClassifier,
#             "params": {
#                 "n_estimators": 200, "max_depth": 5, "verbose": -1,
#             },
#             "seed_param": "random_state",
#         },
#         "catboost": {
#             "model_class": CatBoostClassifier,
#             "params": {
#                 "iterations": 200, "depth": 5, "verbose": 0,
#             },
#             "seed_param": "random_seed",
#         },
#     }

    # ---- Single algorithm quick test ----
    # tester = InteractionTester(
    #     model_class=XGBClassifier,
    #     base_params=configs["xgb"]["params"],
    #     n_bootstrap=100,
    #     n_null=100,
    #     n_jobs=4,
    # )
    # top_pairs = tester.get_top_n_interactions(X, y, n=10)
    # results = tester.test_pairs(X, y, top_pairs)
    # df = tester.results_to_dataframe(results)
    # tester.plot_interaction_distribution(results[0])
    # tester.plot_convergence(results[0])

    # ---- Full cross-algorithm vote ----
    # voter = InteractionVoter(configs, n_bootstrap=100, n_null=100, alpha=0.05, n_jobs=4)
    # vote_results = voter.vote(X, y, top_pairs)
    # print(voter.summary(vote_results))
    # voter.plot_vote_heatmap(vote_results)
