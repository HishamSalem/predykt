"""Stage-2 estimators: OLS/HC3, HSIC, CustomEstimator."""
import warnings

import numpy as np
import pytest
from predykt import OLSEstimator, HSICEstimator, CustomEstimator
from predykt import criteria
from predykt.criteria import (Stage2Result, _center, _hsic_from_centered,
                              _hsic_statistic, _rbf_kernel)

rng = np.random.default_rng(42)


class TestOLSEstimator:
    def test_recovers_known_beta(self):
        n = 2000
        T = rng.normal(size=n)
        Y = 0.5 * T + rng.normal(scale=0.5, size=n)
        res = OLSEstimator().fit(T, Y)
        assert res.method == "ols_hc3"
        assert res.significant
        assert abs(res.beta - 0.5) < 0.05
        assert res.pvalue < 1e-10
        # Full statsmodels result exposed as documented
        assert hasattr(res.model_result, "summary")

    def test_null_false_positive_rate_near_alpha(self):
        """Under H0 (independence) the rejection rate must track alpha=0.05."""
        est = OLSEstimator()
        local = np.random.default_rng(7)
        n_sims, n = 300, 200
        fp = sum(
            est.fit(local.normal(size=n), local.normal(size=n)).significant
            for _ in range(n_sims)
        )
        rate = fp / n_sims
        # 3-sigma band around 0.05 for 300 sims (sd ~ 0.0126)
        assert 0.01 <= rate <= 0.09, f"FP rate {rate:.3f} far from alpha"

    def test_cov_type_label(self):
        T, Y = rng.normal(size=200), rng.normal(size=200)
        assert OLSEstimator(cov_type="HC1").fit(T, Y).method == "ols_hc1"


class TestHSICEstimator:
    def test_detects_nonlinear_dependence_ols_misses(self):
        n = 250
        local = np.random.default_rng(3)
        T = local.normal(size=n)
        Y = T ** 2 + local.normal(scale=0.3, size=n)  # symmetric: corr ~ 0
        ols = OLSEstimator().fit(T, Y)
        hsic = HSICEstimator(n_permutations=300, random_state=0).fit(T, Y)
        assert not ols.significant, "OLS should miss pure quadratic dependence"
        assert hsic.significant, "HSIC should detect quadratic dependence"
        assert hsic.method == "hsic"

    def test_null_pvalue_not_extreme(self):
        local = np.random.default_rng(5)
        res = HSICEstimator(n_permutations=200, random_state=0).fit(
            local.normal(size=150), local.normal(size=150))
        assert res.pvalue > 0.005  # +1 correction bounds p >= 1/(B+1)

    def test_informational_mode_no_permutations(self):
        res = HSICEstimator(n_permutations=0).fit(
            rng.normal(size=100), rng.normal(size=100))
        assert np.isnan(res.pvalue) and not res.significant


def _hsic_naive(K: np.ndarray, L: np.ndarray) -> float:
    """Reference O(n³) HSIC: build H, center both sides, trace the product.

    This is the pre-optimisation implementation, kept here verbatim so the
    O(n²) version in predykt.criteria is pinned to it rather than to
    hardcoded magic numbers.
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    return float(np.trace(Kc @ Lc) / (n - 1) ** 2)


class TestHSICStatisticEquivalence:
    """The O(n²) rewrite must be non-behavioural: same values, less work."""

    @pytest.mark.parametrize("n", [7, 50, 137])
    def test_matches_naive_double_centered_trace(self, n):
        local = np.random.default_rng(20260729)
        x = local.normal(size=n)
        y = np.sin(2.0 * x) + 0.5 * local.normal(size=n)
        K = _rbf_kernel(x, None)
        L = _rbf_kernel(y, None)
        assert _hsic_statistic(K, L) == pytest.approx(_hsic_naive(K, L), rel=1e-12)

    def test_center_matches_explicit_HMH(self):
        local = np.random.default_rng(11)
        M = _rbf_kernel(local.normal(size=60), None)
        n = M.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        assert np.allclose(_center(M), H @ M @ H, atol=1e-12)

    def test_precentering_exact_under_permutation(self):
        """Hoisting the centering out of the permutation loop is exact:
        PᵀHP = H, so Pᵀ(HLH)P = H(PᵀLP)H."""
        local = np.random.default_rng(4)
        n = 80
        x = local.normal(size=n)
        y = np.cos(x) + 0.4 * local.normal(size=n)
        K = _rbf_kernel(x, None)
        L = _rbf_kernel(y, None)
        Kc = _center(K)
        for _ in range(50):
            perm = local.permutation(n)
            L_perm = L[np.ix_(perm, perm)]
            assert _hsic_from_centered(Kc, L_perm) == pytest.approx(
                _hsic_naive(K, L_perm), rel=1e-10
            )

    def test_pvalue_reproducible_for_fixed_random_state(self):
        local = np.random.default_rng(9)
        T, Y = local.normal(size=120), local.normal(size=120)
        a = HSICEstimator(n_permutations=50, random_state=3).fit(T, Y)
        b = HSICEstimator(n_permutations=50, random_state=3).fit(T, Y)
        assert a.pvalue == b.pvalue and a.beta == b.beta

    def test_large_n_warns_but_does_not_raise(self, monkeypatch):
        """A hard raise would break legitimate large-n exploratory use."""
        monkeypatch.setattr(criteria, "_HSIC_LARGE_N_WARN", 100)
        local = np.random.default_rng(2)
        with pytest.warns(UserWarning, match="O\\(n\\^2\\) per permutation"):
            res = HSICEstimator(n_permutations=5, random_state=0).fit(
                local.normal(size=150), local.normal(size=150))
        assert np.isfinite(res.pvalue)

    def test_no_warning_below_threshold(self):
        local = np.random.default_rng(2)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            HSICEstimator(n_permutations=5, random_state=0).fit(
                local.normal(size=120), local.normal(size=120))


class TestCustomEstimator:
    def test_wraps_callable(self):
        def fn(T, Y):
            return Stage2Result(beta=1.0, t_stat=1.0, pvalue=0.5,
                                significant=False, method="stub",
                                model_result=None)
        res = CustomEstimator(fn).fit(np.ones(10), np.ones(10))
        assert res.method == "stub"

    def test_rejects_wrong_return_type(self):
        with pytest.raises(TypeError, match="must return Stage2Result"):
            CustomEstimator(lambda T, Y: 0.5).fit(np.ones(10), np.ones(10))
