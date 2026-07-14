"""Stage-2 estimators: OLS/HC3, HSIC, CustomEstimator."""
import numpy as np
import pytest
from predykt import OLSEstimator, HSICEstimator, CustomEstimator
from predykt.criteria import Stage2Result

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
