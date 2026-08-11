"""
Stage-2 estimators for ResidualRepresentationTester.

In the residual framework the stage-2 question is:

    Does Tₖ explain the base-model residuals Ỹ?

This is operationalised as regressing Ỹ on Tₖ and testing H₀: β₁ = 0.
The abstract base class standardises the interface so ResidualRepresentationTester
is estimator-agnostic. Users can subclass to use any test they like.

Default: OLSEstimator with HC3 heteroskedasticity-robust standard errors.
HC3 is mandatory (not optional) for binary classification residuals because
Var(Ỹᵢ) = p̂ᵢ(1 − p̂ᵢ), observation-specific, never constant.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import statsmodels.api as sm

# =============================================================================
# KERNEL HELPERS (used by HSICEstimator)
# =============================================================================

def _rbf_kernel(x: np.ndarray, bandwidth: float | None) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    dists_sq = (x[:, None] - x[None, :]) ** 2
    if bandwidth is None:
        nonzero_sq = dists_sq[dists_sq > 0]
        bandwidth = float(np.sqrt(np.median(nonzero_sq))) if nonzero_sq.size else 1.0
    return np.exp(-dists_sq / (2.0 * bandwidth ** 2))


# n above which HSIC's O(n²) memory and time are worth warning about.
# Module-level so tests can lower it without allocating a 5000x5000 Gram matrix.
_HSIC_LARGE_N_WARN = 5000


def _center(M: np.ndarray) -> np.ndarray:
    """
    Double-center a Gram matrix: HMH with H = I − 11ᵀ/n.

    Computed by mean subtraction rather than as two matrix products. Expanding
    the definition gives an exactly equivalent O(n²) form:

        HMH = M − (1/n)11ᵀM − (1/n)M11ᵀ + (1/n²)11ᵀM11ᵀ
            = M − colmeans − rowmeans + grandmean

    Forming H explicitly and evaluating ``H @ M @ H`` would cost two n×n
    matrix products, i.e. O(n³) — which would dominate everything else in
    HSICEstimator.fit even when centering is hoisted out of the permutation
    loop. Same values, one order cheaper.
    """
    return (
        M
        - M.mean(axis=0, keepdims=True)
        - M.mean(axis=1, keepdims=True)
        + M.mean()
    )


def _hsic_statistic(K: np.ndarray, L: np.ndarray) -> float:
    """
    Biased HSIC estimator tr(KHLH)/(n−1)² (Gretton et al. 2005).

    Two exact simplifications keep this O(n²) rather than O(n³):

    * Only one matrix is centered. H is idempotent (HH = H) and trace is
      invariant under cyclic permutation, so
      tr((HKH)(HLH)) = tr(HKH·L), i.e. centering both is redundant.
    * The trace of the product is read off as an elementwise sum,
      ``np.sum(Kc * L)``, since both matrices are symmetric. This avoids
      materialising an n×n matmul just to sum its diagonal.

    Neither step is an approximation.
    """
    return _hsic_from_centered(_center(K), L)


def _hsic_from_centered(Kc: np.ndarray, L: np.ndarray) -> float:
    """
    HSIC from an already-centered Kc and an uncentered L.

    Lets the permutation loop in HSICEstimator.fit hoist the centering out,
    paying it once instead of once per permutation. Exact, for two reasons:

    * H is idempotent, so tr(Kc·(HLH)) = tr(Kc·L) — L never needs centering
      as long as Kc is centered.
    * Permutation matrices commute with centering:
      PᵀHP = PᵀP − (1/n)Pᵀ11ᵀP = I − (1/n)11ᵀ = H, hence
      Pᵀ(HLH)P = H(PᵀLP)H. So permuting L and then centering gives the same
      statistic as centering and then permuting.
    """
    n = Kc.shape[0]
    return float(np.sum(Kc * L) / (n - 1) ** 2)


# =============================================================================
# RESULT CONTAINER
# =============================================================================

@dataclass
class Stage2Result:
    """
    Unified result from a stage-2 residual regression.

    Attributes
    ----------
    beta : float
        Estimated coefficient on Tₖ in Ỹ ~ β₀ + β₁·Tₖ.
    t_stat : float
        t-statistic for β₁ (signed).
    pvalue : float
        Two-sided p-value for H₀: β₁ = 0.
    significant : bool
        True if pvalue < alpha.
    method : str
        Estimator identifier, e.g. "ols_hc3".
    model_result : Any
        Full underlying model result object (e.g. statsmodels RegressionResults).
        Call .summary() on this for the complete diagnostic output.
    """
    beta: float
    t_stat: float
    pvalue: float
    significant: bool
    method: str
    model_result: Any = field(repr=False)


# =============================================================================
# ABSTRACT BASE
# =============================================================================

class Stage2Estimator(ABC):
    """
    Abstract base for stage-2 estimators in the residual framework.

    Subclass this to plug any regression or dependence test into
    ResidualRepresentationTester. The only requirement is that fit() returns
    a Stage2Result.

    Example: custom Spearman rank criterion:

        class SpearmanEstimator(Stage2Estimator):
            def __init__(self, alpha=0.05):
                self.alpha = alpha
            def fit(self, T_k, Y_resid):
                from scipy.stats import spearmanr
                rho, pval = spearmanr(T_k, Y_resid)
                return Stage2Result(
                    beta=rho, t_stat=rho, pvalue=pval,
                    significant=pval < self.alpha,
                    method='spearman', model_result=None,
                )
    """

    @abstractmethod
    def fit(self, T_k: np.ndarray, Y_resid: np.ndarray) -> Stage2Result:
        """
        Fit the stage-2 model.

        Parameters
        ----------
        T_k : np.ndarray, shape (n,)
            Candidate representation, engineered by the user. Passed as-is;
            no transforms are applied here.
        Y_resid : np.ndarray, shape (n,)
            OOF residuals Ỹ = y − p̂ from the base model.

        Returns
        -------
        Stage2Result
        """


# =============================================================================
# OLS WITH ROBUST STANDARD ERRORS (default)
# =============================================================================

class OLSEstimator(Stage2Estimator):
    """
    Stage-2 OLS with heteroskedasticity-robust standard errors.

    Fits Ỹ = β₀ + β₁·Tₖ + ε using statsmodels OLS, then tests H₀: β₁ = 0
    using the robust t-statistic.

    Parameters
    ----------
    cov_type : str, default="HC3"
        Covariance estimator passed to statsmodels .fit(cov_type=...).
        "HC3" (MacKinnon & White 1985) is the recommended choice for
        binary classification residuals; it is the standard small-sample
        heteroskedasticity-robust estimator.
        Other valid options: "HC0", "HC1", "HC2", "HAC".
    alpha : float, default=0.05
        Significance threshold for the significant flag in Stage2Result.

    Notes
    -----
    WHY HC3 IS NOT OPTIONAL:
        For binary targets, residuals are Ỹᵢ = yᵢ − p̂ᵢ where yᵢ ∈ {0,1}.
        Var(Ỹᵢ) = p̂ᵢ(1 − p̂ᵢ), observation-specific, never constant.
        OLS with homoskedastic SEs is misspecified. HC3 corrects this.

    The full statsmodels result is available on Stage2Result.model_result,
    so you can call result.model_result.summary() for the complete output.
    """

    def __init__(self, cov_type: str = "HC3", alpha: float = 0.05):
        self.cov_type = cov_type
        self.alpha = alpha

    def fit(self, T_k: np.ndarray, Y_resid: np.ndarray) -> Stage2Result:
        T_k = np.asarray(T_k, dtype=float)
        Y_resid = np.asarray(Y_resid, dtype=float)

        X_design = sm.add_constant(T_k, has_constant="add")
        ols_result = sm.OLS(Y_resid, X_design).fit(cov_type=self.cov_type)

        beta = float(ols_result.params[1])
        t_stat = float(ols_result.tvalues[1])
        pvalue = float(ols_result.pvalues[1])

        return Stage2Result(
            beta=beta,
            t_stat=t_stat,
            pvalue=pvalue,
            significant=pvalue < self.alpha,
            method=f"ols_{self.cov_type.lower()}",
            model_result=ols_result,
        )


# =============================================================================
# CUSTOM (user-supplied callable)
# =============================================================================

class CustomEstimator(Stage2Estimator):
    """
    Wraps a user-supplied callable as a Stage2Estimator.

    Parameters
    ----------
    fn : Callable[[np.ndarray, np.ndarray], Stage2Result]
        Signature: fn(T_k, Y_resid) -> Stage2Result.
    """

    def __init__(self, fn: Callable):
        self.fn = fn

    def fit(self, T_k: np.ndarray, Y_resid: np.ndarray) -> Stage2Result:
        result = self.fn(T_k, Y_resid)
        if not isinstance(result, Stage2Result):
            raise TypeError(
                f"CustomEstimator fn must return Stage2Result, got {type(result)}"
            )
        return result


class HSICEstimator(Stage2Estimator):
    """
    Stage-2 HSIC (Hilbert-Schmidt Independence Criterion) test.

    Kernel-based nonparametric independence test between Tₖ and Ỹ.
    Detects nonlinear and non-monotone dependence that OLS cannot capture.
    The statistic is unsigned (always ≥ 0), so use alongside OLSEstimator
    when direction of effect matters.

    Parameters
    ----------
    n_permutations : int, default=500
        Permutations for the p-value. 500 gives reliable results; use 1000+
        for publication-quality estimates.
    bandwidth : float or None, default=None
        RBF kernel bandwidth σ. None applies the median heuristic
        (σ = median of pairwise distances), which is a robust default.
    alpha : float, default=0.05
    random_state : int or None, default=None

    Notes
    -----
    beta and t_stat are both set to the raw HSIC statistic (scale depends on
    bandwidth and n). They are comparable across representations fitted with
    the same criterion instance but not across different datasets or bandwidths.

    Primary reference: Gretton, A., Bousquet, O., Smola, A. & Schölkopf, B.
    (2005), "Measuring Statistical Dependence with Hilbert-Schmidt Norms",
    ALT 2005. The permutation null follows Gretton et al. (2008), "A Kernel
    Statistical Test of Independence", NIPS 2008.

    SCALING:
        Each permutation is O(n²) in both time and memory: two n×n Gram
        matrices are materialised, and every permutation does an n×n
        elementwise product. Centering is hoisted out of the permutation loop
        and computed by mean subtraction, so no step is O(n³). Measured on one
        machine, n=2000 with n_permutations=500 takes ~17s end to end, at
        ~34ms per permutation; cost grows quadratically in n from there.

        This is fine into the low thousands but is not suitable much past
        n ≈ 10⁴. Two standard routes past that, neither implemented here
        (roadmap items):

        * The gamma approximation to the null (Gretton et al. 2008) replaces
          the permutation loop with a two-moment fit, turning n_permutations
          model passes into one. This is the cheaper and more directly
          applicable of the two for this codebase.
        * Block / Nyström HSIC (Zhang, Q., Filippi, S., Gretton, A. &
          Sejdinovic, D. (2018), "Large-scale kernel methods for independence
          testing", Statistics and Computing 28:113-130) trades a little power
          for near-linear scaling, and is the standard choice at large n.
    """

    def __init__(
        self,
        n_permutations: int = 500,
        bandwidth: float | None = None,
        alpha: float = 0.05,
        random_state: int | None = None,
    ):
        self.n_permutations = n_permutations
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, T_k: np.ndarray, Y_resid: np.ndarray) -> Stage2Result:
        T_k = np.asarray(T_k, dtype=float).ravel()
        Y_resid = np.asarray(Y_resid, dtype=float).ravel()

        n = len(Y_resid)
        if n > _HSIC_LARGE_N_WARN and self.n_permutations > 0:
            warnings.warn(
                f"HSIC on n={n} with n_permutations={self.n_permutations} builds "
                f"{n}x{n} Gram matrices and is O(n^2) per permutation; expect roughly "
                f"{self.n_permutations * (n / 2000.0) ** 2 * 0.03:.0f}s. Consider "
                "subsampling rows, or n_permutations=0 for the statistic alone.",
                UserWarning,
                stacklevel=2,
            )

        K = _rbf_kernel(T_k, self.bandwidth)
        L = _rbf_kernel(Y_resid, self.bandwidth)

        # Centering is hoisted out of the permutation loop below; see
        # _hsic_from_centered for why that is exact rather than an approximation.
        Kc = _center(K)
        observed = _hsic_from_centered(Kc, L)

        if self.n_permutations == 0:
            # Informational mode: statistic only, no p-value
            return Stage2Result(
                beta=observed,
                t_stat=observed,
                pvalue=float("nan"),
                significant=False,
                method="hsic",
                model_result=None,
            )

        rng = np.random.default_rng(self.random_state)
        null = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            perm = rng.permutation(n)
            null[i] = _hsic_from_centered(Kc, L[np.ix_(perm, perm)])

        # +1 conservative correction for finite permutations
        pvalue = float((np.sum(null >= observed) + 1) / (self.n_permutations + 1))

        return Stage2Result(
            beta=observed,
            t_stat=observed,
            pvalue=pvalue,
            significant=pvalue < self.alpha,
            method="hsic",
            model_result=None,
        )
