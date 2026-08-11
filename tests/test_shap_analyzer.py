"""SHAPInteractionAnalyzer: layer math against independent hand computation,
validation, consistency warning, Mode A smoke."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

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


# =============================================================================
# Multiclass must be refused, not silently mis-answered
# =============================================================================

class TestMulticlassRefused:
    """shap's trailing axis means "multi-output", not "binary".

    A 3-class model produces (n, p, p, 3) on every model family. Taking index 1
    would report one arbitrary class as though it were the positive one, and
    return a plausible DataFrame while doing it. InteractionTester refuses such
    input; these pin that SHAPInteractionAnalyzer agrees, so the two modules
    cannot disagree about what they support.
    """

    GROUPS = {"g1": ["a", "b"], "g2": ["c", "d"]}

    @staticmethod
    def _data(seed=0, n=150, n_classes=3):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
        return X, rng.integers(0, n_classes, n)

    def test_mode_a_raises(self):
        X, y3 = self._data()
        model = RandomForestClassifier(n_estimators=8, max_depth=3,
                                       random_state=0).fit(X, y3)
        analyzer = SHAPInteractionAnalyzer(interaction_groups=self.GROUPS,
                                           layers=[1, 2, 3])
        with pytest.raises(ValueError, match="Multiclass"):
            analyzer.fit(model=model, X=X)

    def test_mode_b_raises(self):
        import shap
        X, y3 = self._data()
        model = RandomForestClassifier(n_estimators=8, max_depth=3,
                                       random_state=0).fit(X, y3)
        explainer = shap.TreeExplainer(model)
        analyzer = SHAPInteractionAnalyzer(interaction_groups=self.GROUPS,
                                           layers=[1, 2, 3])
        with pytest.raises(ValueError, match="Multiclass"):
            analyzer.fit(shap_values=explainer.shap_values(X),
                         shap_interaction_values=explainer.shap_interaction_values(X),
                         X=X)

    def test_binary_still_works(self):
        X, _ = self._data()
        rng = np.random.default_rng(1)
        y = (rng.random(len(X)) < 1 / (1 + np.exp(-(2 * X.a * X.b)))).astype(int)
        model = RandomForestClassifier(n_estimators=8, max_depth=3,
                                       random_state=0).fit(X, y)
        analyzer = SHAPInteractionAnalyzer(interaction_groups=self.GROUPS,
                                           layers=[1, 2, 3]).fit(model=model, X=X)
        assert analyzer.layer_2_net_group_effects().shape[1] == 2
