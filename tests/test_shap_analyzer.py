"""SHAPInteractionAnalyzer: layer math against independent hand computation,
validation, consistency warning, Mode A smoke."""
import numpy as np
import pandas as pd
import pytest
from predykt import SHAPInteractionAnalyzer

GROUPS = {"g1": ["f0", "f1"], "g2": ["f2", "f3"]}


@pytest.fixture(scope="module")
def synthetic_shap():
    """Symmetric interaction tensor + consistent shap_values.
    By the SHAP interaction decomposition, sv[i,f] = sum_h siv[i,f,h],
    so consistency holds exactly by construction."""
    rng = np.random.default_rng(0)
    n, p = 60, 4
    A = rng.normal(size=(n, p, p))
    siv = (A + A.transpose(0, 2, 1)) / 2.0     # symmetric off-diagonals
    sv = siv.sum(axis=2)                        # exact efficiency
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=["f0", "f1", "f2", "f3"])
    return sv, siv, X


class TestLayerMath:
    def test_layers_match_hand_computation(self, synthetic_shap):
        sv, siv, X = synthetic_shap
        an = SHAPInteractionAnalyzer(interaction_groups=GROUPS).fit(
            shap_values=sv, shap_interaction_values=siv, X=X)

        # Layer 1: independent loop-based group sums
        l1_g1 = sv[:, 0] + sv[:, 1]
        np.testing.assert_allclose(an.layer_1_group_total()["g1"], l1_g1)

        # Layer 2: layer1 minus half the cross-group interaction mass
        cross_g1 = np.zeros(len(sv))
        for f in (0, 1):
            for h in (2, 3):
                cross_g1 += siv[:, f, h]
        np.testing.assert_allclose(
            an.layer_2_net_group_effects()["g1"], l1_g1 - cross_g1 / 2.0)

        # Layer 3: pure main effects = diagonal
        np.testing.assert_allclose(
            an.layer_3_pure_main_effects()["f2"], siv[:, 2, 2])

    def test_summary_sorted_and_aggregates(self, synthetic_shap):
        sv, siv, X = synthetic_shap
        an = SHAPInteractionAnalyzer(interaction_groups=GROUPS).fit(
            shap_values=sv, shap_interaction_values=siv, X=X)
        s = an.summary(1, aggregate="mean_abs")
        assert list(s.columns) == ["group", "importance"]
        assert s["importance"].is_monotonic_decreasing
        with pytest.raises(ValueError, match="aggregate"):
            an.summary(1, aggregate="bogus")
        groups_df, feats_df = an.compare_layers()
        assert {"layer_1", "layer_2"} <= set(groups_df.columns)
        assert "layer_3" in feats_df.columns

    def test_consistency_warning_on_mismatched_inputs(self, synthetic_shap):
        sv, siv, X = synthetic_shap
        with pytest.warns(UserWarning, match="consistency check failed"):
            SHAPInteractionAnalyzer(interaction_groups=GROUPS).fit(
                shap_values=sv + 1.0, shap_interaction_values=siv, X=X)


class TestGroupValidation:
    def test_orphan_feature_raises(self, synthetic_shap):
        sv, siv, X = synthetic_shap
        with pytest.raises(ValueError, match="Unassigned"):
            SHAPInteractionAnalyzer(
                interaction_groups={"g1": ["f0", "f1", "f2"]}).fit(
                shap_values=sv, shap_interaction_values=siv, X=X)

    def test_duplicate_assignment_raises(self, synthetic_shap):
        sv, siv, X = synthetic_shap
        with pytest.raises(ValueError, match="multiple groups"):
            SHAPInteractionAnalyzer(interaction_groups={
                "g1": ["f0", "f1"], "g2": ["f1", "f2", "f3"]}).fit(
                shap_values=sv, shap_interaction_values=siv, X=X)

    def test_unknown_feature_raises(self, synthetic_shap):
        sv, siv, X = synthetic_shap
        with pytest.raises(ValueError, match="not in shap_values"):
            SHAPInteractionAnalyzer(interaction_groups={
                "g1": ["f0", "f1"], "g2": ["f2", "f3", "ghost"]}).fit(
                shap_values=sv, shap_interaction_values=siv, X=X)


class TestModeA:
    def test_internal_shap_computation_smoke(self):
        from sklearn.ensemble import GradientBoostingClassifier
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(120, 4)),
                         columns=["f0", "f1", "f2", "f3"])
        y = (X["f0"] + X["f2"] * X["f3"] > 0).astype(int)
        model = GradientBoostingClassifier(n_estimators=30,
                                           max_depth=2).fit(X, y)
        an = SHAPInteractionAnalyzer(interaction_groups=GROUPS).fit(
            model=model, X=X)
        assert an.layer_1_group_total().shape == (120, 2)
        assert an.layer_3_pure_main_effects().shape == (120, 4)
