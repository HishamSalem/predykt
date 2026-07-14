"""CyclicalBinner: recovery of a circular pattern, validation, sklearn API."""
import numpy as np
import pytest
from predykt import CyclicalBinner


@pytest.fixture(scope="module")
def hourly_data():
    """24-hour cycle with an event-rate spike spanning midnight (22:00-01:59):
    the wrap-around case linear binning cannot represent."""
    rng = np.random.default_rng(0)
    n = 6000
    hours = rng.integers(0, 24, size=n).astype(float)
    spike = np.isin(hours, [22, 23, 0, 1])
    p = np.where(spike, 0.35, 0.08)
    y = rng.binomial(1, p)
    return hours, y


@pytest.fixture(scope="module")
def fitted(hourly_data):
    hours, y = hourly_data
    b = CyclicalBinner(m=24, gamma=0.0, alpha_min=0.01, e_min=5, ne_min=5,
                       k_max=4)
    return b.fit(hours, y), hours, y


class TestFitRecovery:
    def test_finds_informative_binning(self, fitted):
        b, _, _ = fitted
        assert 2 <= b.n_bins_ <= 4
        assert b.iv_ > 0.3, f"IV={b.iv_:.3f}; spike should give strong IV"

    def test_circular_wraparound_grouped(self, fitted):
        """Hour 23 and hour 0 belong to the same spike; a correct circular
        binner must place them in one bin. This is the feature that
        distinguishes it from ordinary monotone binning."""
        b, _, _ = fitted
        bins = b.transform(np.array([23.0, 0.0]))
        assert bins[0] == bins[1]

    def test_woe_separates_spike_from_quiet(self, fitted):
        b, _, _ = fitted
        woe = b.transform_woe(np.array([23.0, 12.0]))
        assert np.all(np.isfinite(woe))
        assert abs(woe[0] - woe[1]) > 0.5

    def test_woe_encoder_covers_all_bins(self, fitted):
        b, _, _ = fitted
        enc = b.get_woe_encoder()
        assert len(enc) == b.n_bins_
        assert all(np.isfinite(v) for v in enc.values())

    def test_result_summary_and_dict(self, fitted):
        b, _, _ = fitted
        summary = b.result_.summary()
        assert summary is not None and len(summary) > 0  # DataFrame table
        d = b.result_.to_dict()
        assert isinstance(d, dict)
        assert any("iv" in k.lower() for k in d), f"no IV key in {list(d)}"


class TestValidation:
    def test_out_of_range_raises(self, hourly_data):
        hours, y = hourly_data
        with pytest.raises(ValueError, match=r"\[0"):
            CyclicalBinner(m=12).fit(hours, y)  # hours go to 23 > m-1

    def test_non_binary_target_raises(self, hourly_data):
        hours, _ = hourly_data
        with pytest.raises(ValueError, match="binary"):
            CyclicalBinner(m=24).fit(hours, np.full(len(hours), 2))

    def test_length_mismatch_raises(self, hourly_data):
        hours, y = hourly_data
        with pytest.raises(ValueError, match="same length"):
            CyclicalBinner(m=24).fit(hours, y[:-10])

    def test_transform_before_fit_raises(self):
        with pytest.raises(Exception):  # sklearn NotFittedError
            CyclicalBinner(m=24).transform(np.array([1.0]))
