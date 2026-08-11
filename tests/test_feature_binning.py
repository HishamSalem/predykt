"""FeatureBinningAnalyzer: IV ordering, uplift schema, error paths."""
import numpy as np
import pandas as pd
import pytest

from predykt import FeatureBinningAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    rng = np.random.default_rng(0)
    n = 1500
    y = rng.binomial(1, 0.3, size=n)
    X = pd.DataFrame({
        "informative": rng.normal(size=n) + 1.2 * y,
        "noise": rng.normal(size=n),
    })
    return FeatureBinningAnalyzer(X, y)


class TestIV:
    def test_informative_feature_outranks_noise(self, analyzer):
        iv_info = analyzer.get_feature_iv("informative")
        iv_noise = analyzer.get_feature_iv("noise")
        assert iv_info > 0.3
        assert iv_info > iv_noise

    def test_missing_feature_raises_keyerror(self, analyzer):
        with pytest.raises(KeyError, match="not found"):
            analyzer.get_feature_iv("ghost")


class TestCombinations:
    def test_analyze_schema_and_uplift(self, analyzer):
        res = analyzer.analyze_feature_combinations([("informative", "noise")])
        assert {"Feature1", "Feature2", "IV_1", "IV_2", "IV_2D",
                "Uplift", "Binning_Table"} <= set(res.columns)
        assert np.isfinite(res["Uplift"].iloc[0])
        top = analyzer.get_top_combinations()
        assert len(top) == 1
        details = analyzer.get_binning_details("informative", "noise")
        assert isinstance(details, pd.DataFrame)

    def test_methods_require_analyze_first(self):
        fresh = FeatureBinningAnalyzer(
            pd.DataFrame({"a": [1.0, 2.0]}), np.array([0, 1]))
        with pytest.raises(ValueError, match="analyze_feature_combinations"):
            fresh.get_top_combinations()
