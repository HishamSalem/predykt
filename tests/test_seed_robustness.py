"""SeedRobustnessValidator: statistical core vs references, verdict logic."""
import numpy as np
import pytest
from predykt import SeedRobustnessValidator


def _stable_eval(seed: int) -> float:
    """Deterministic per-seed metric with tiny variance around 0.75."""
    return 0.75 + np.random.default_rng(seed).normal(scale=0.002)


def _unstable_eval(seed: int) -> float:
    return 0.75 + np.random.default_rng(seed).normal(scale=0.05)


class TestStatisticalCore:
    def test_tolerance_k_matches_howe_reference(self):
        """Howe (1969) two-sided 95/95 k-factors: n=10 -> 3.379, n=30 -> 2.549,
        n=100 -> 2.233 (published tables, ISO 16269-6)."""
        for n, ref in [(10, 3.379), (30, 2.549), (100, 2.233)]:
            scores = np.random.default_rng(0).normal(size=n)
            _, _, k = SeedRobustnessValidator._tolerance_interval_95_95(scores)
            assert abs(k - ref) < 0.01, f"n={n}: k={k:.3f} vs ref {ref}"

    def test_chi2_variance_test_directionality(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(scale=0.01, size=50)
        # sigma_max far above observed std -> acceptable
        _, p_hi, ok_hi = SeedRobustnessValidator._chi2_variance_test(
            scores, sigma_max=0.10, alpha=0.05)
        # sigma_max far below observed std -> unacceptable
        _, p_lo, ok_lo = SeedRobustnessValidator._chi2_variance_test(
            scores, sigma_max=0.001, alpha=0.05)
        assert ok_hi and p_hi > 0.95
        assert not ok_lo and p_lo < 1e-6

    def test_bootstrap_std_ci_brackets_truth(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(scale=0.02, size=200)
        lo, hi = SeedRobustnessValidator._bootstrap_std_ci(
            scores, n_bootstrap=2000, alpha=0.05)
        assert lo < 0.02 < hi


class TestRunAndVerdict:
    def test_stable_config_accepted(self):
        v = SeedRobustnessValidator(_stable_eval, n_seeds=40,
                                    metric_name="AUC", sigma_max=0.01)
        report = v.run()
        assert abs(report.mean - 0.75) < 0.005
        assert report.std < 0.005
        assert report.variance_acceptable
        assert report.tol_lower < report.mean < report.tol_upper

    def test_unstable_config_flagged(self):
        v = SeedRobustnessValidator(_unstable_eval, n_seeds=40,
                                    metric_name="AUC", sigma_max=0.005)
        report = v.run()
        assert not report.variance_acceptable

    def test_nonfinite_eval_raises(self):
        v = SeedRobustnessValidator(lambda s: float("nan"), n_seeds=30)
        with pytest.raises(ValueError, match="non-finite"):
            v.run()

    def test_low_n_seeds_warns(self):
        with pytest.warns(UserWarning, match="below 30"):
            SeedRobustnessValidator(_stable_eval, n_seeds=10)
