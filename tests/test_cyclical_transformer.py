"""CyclicalBinner: recovery of a circular pattern, validation, sklearn API."""
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

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


class TestWOESignConvention:
    """WOE is ln(%non-event / %event), matching optbinning (Navas-Palencia 2020
    §2.1) — i.e. inversely related to the event rate. IV must be untouched by
    the convention, which is the part a naive one-sided flip breaks."""

    def test_woe_is_negative_above_base_rate_positive_below(self, fitted):
        b, _, y = fitted
        base_rate = y.mean()
        above = b.result_.event_rates > base_rate
        below = b.result_.event_rates < base_rate
        assert above.any() and below.any(), "fixture must span the base rate"
        assert np.all(b.woe_[above] < 0), (
            f"bins above the base rate must have negative WOE; got "
            f"{b.woe_[above]} for event rates {b.result_.event_rates[above]}"
        )
        assert np.all(b.woe_[below] > 0)

    def test_iv_positive_and_matches_manual_sum(self, fitted):
        """iv_smoothed is the value a one-sided flip silently negates."""
        b, _, _ = fitted
        assert b.iv_ > 0
        assert b.result_.iv_raw > 0
        assert b.result_.iv_smoothed > 0

        # Recompute iv_raw from counts, convention-free.
        ev = b.result_.event_counts
        ne = b.result_.non_event_counts
        p = ev / ev.sum()
        q = ne / ne.sum()
        expected = float(np.sum((p - q) * np.log(p / q)))
        assert b.result_.iv_raw == pytest.approx(expected, rel=1e-12)

    def test_summary_iv_column_positive(self, fitted):
        b, _, _ = fitted
        per_bin = b.result_.summary()["iv"].to_numpy()[:b.n_bins_]
        assert np.all(per_bin > 0), f"per-bin IV must stay positive, got {per_bin}"
        # Only a loose match to iv_raw: summary() pairs the *smoothed* woe_ with
        # *unsmoothed* p/q, so the column sums to neither iv_raw nor iv_smoothed
        # exactly. Pre-existing behaviour, unchanged by the sign flip.
        assert per_bin.sum() == pytest.approx(b.result_.iv_raw, rel=0.05)

    def test_woe_table_matches_woe_attribute(self, fitted):
        b, _, _ = fitted
        assert [r["woe"] for r in b.result_.woe_table()] == \
            pytest.approx(b.woe_.tolist())

    def test_woe_sign_convention_is_exact(self):
        """WOE must equal ln(%non-event / %event), computed from the binner's
        own reported counts.

        This replaces a cross-check against optbinning. That test only asserted
        directional agreement, skipped whenever optbinning was absent, and made
        optbinning the last thing in the repo carrying an ortools ceiling. The
        formula is arithmetic, so pin it directly: exact values, never skips,
        no dependency.
        """
        rng = np.random.default_rng(7)
        n = 8000
        x = rng.integers(0, 24, size=n)
        p = 0.02 + 0.012 * x
        y = rng.binomial(1, p)

        cb = CyclicalBinner(m=24, gamma=0.0, alpha_min=0.05, e_min=10,
                            ne_min=10, k_min=2, k_max=4).fit(x, y)

        rows = cb.result_.woe_table()
        total_e = sum(r["events"] for r in rows)
        total_ne = sum(r["non_events"] for r in rows)
        k, lam = len(rows), cb.lam

        for r in rows:
            p_j = (r["events"] + lam) / (total_e + k * lam)
            q_j = (r["non_events"] + lam) / (total_ne + k * lam)
            assert r["woe"] == pytest.approx(np.log(q_j / p_j), abs=1e-9)

        # The convention's observable consequence: a bin riskier than the base
        # rate carries a NEGATIVE woe. A sign flip anywhere breaks this.
        base_rate = y.mean()
        for r in rows:
            rate = r["events"] / (r["events"] + r["non_events"])
            if rate > base_rate:
                assert r["woe"] < 0, f"bin above base rate must have negative woe: {r}"
            elif rate < base_rate:
                assert r["woe"] > 0, f"bin below base rate must have positive woe: {r}"


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
        with pytest.raises(NotFittedError):
            CyclicalBinner(m=24).transform(np.array([1.0]))
