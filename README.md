# predykt

[![CI](https://github.com/HishamSalem/predykt/actions/workflows/ci.yml/badge.svg)](https://github.com/HishamSalem/predykt/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/predykt.svg)](https://pypi.org/project/predykt/)
[![Python](https://img.shields.io/pypi/pyversions/predykt.svg)](https://pypi.org/project/predykt/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#)

> ⚠️ **Alpha (0.x):** APIs may change between minor versions without deprecation. Pin a version in production.

A Python toolkit for rigorous feature interaction analysis in machine learning models. It brings together cyclical optimal binning, SHAP interaction stability testing, residual representation testing, and seed robustness validation as a **layered protocol** for tabular ML, where each tool strips out a different way a result can be an artifact of one arbitrary choice: one seed, one algorithm, one fit, one calendar encoding.

## Why predykt?

Standard ML libraries treat feature analysis as a single-pass operation: fit once, read SHAP values, done. This works poorly when:

- Your temporal features are **cyclical** (hour-of-day, month-of-year): standard binners don't know that 23:00 and 00:00 are adjacent
- Your SHAP interactions are **seed-dependent**: a strong-looking interaction from a single fit may vanish on the next random seed
- Your HPO result is **lucky**: the best config from your tuning run may only be best for that seed
- Your engineered feature needs **validation**: a candidate transformation may not explain residual structure the base model missed

predykt addresses each of these failure modes with a dedicated, statistically grounded tool.

## Installation

```bash
pip install predykt
```

**Core dependencies:** `numpy`, `numba`, `scikit-learn` (`>=1.6` on Python <=3.12, `>=1.1,<1.8` on 3.13+), `pandas`, `shap`, `scipy`, `statsmodels`, `optbinning`, `joblib`, `tqdm`

**Optional extras:**

```bash
pip install "predykt[plot]"   # matplotlib + seaborn, for the plot_* methods
pip install "predykt[test]"   # lightgbm, xgboost, catboost + pytest, to run the test suite
```

> **Why the version fork:** optbinning ≤0.20.1 calls an argument scikit-learn removed in 1.8, and optbinning ≥0.20.1 caps `ortools<9.12`, which has no wheel for Python 3.13+. Pinning the pair per Python version is what keeps predykt installable everywhere. On 3.13+ you will see a harmless `FutureWarning` about `force_all_finite` from optbinning. It is deliberately not suppressed, since a filter broad enough to catch it would hide the same warning from your own scikit-learn calls.

## Modules

| Module                         | What it does                                                                              |
| ------------------------------ | ----------------------------------------------------------------------------------------- |
| `CyclicalBinner`               | IV-maximizing optimal binning for circular temporal features                              |
| `InteractionTester`            | SHAP interaction testing against a simulated additive null                                 |
| `InteractionVoter`             | Cross-algorithm voting to distinguish data interactions from algorithm artifacts          |
| `SeedRobustnessValidator`      | Statistical validation of hyperparameter config robustness across seeds                   |
| `FeatureBinningAnalyzer`       | IV uplift screening for feature pair interactions via OptBinning                          |
| `ResidualRepresentationTester` | Residual-based test of whether an engineered representation explains base-model residuals |
| `SHAPInteractionAnalyzer`      | Three-layer SHAP attribution corrected for collinearity and cross-group aliasing          |
| `CatBoostAdapter` / `PandasCategoricalAdapter` | Let the residual tester cross-fit native-categorical models (CatBoost / LightGBM / XGBoost) |

## Quick Start

> **Prefer to just run it?** Every example below is available as an executable
> notebook: [`examples/predykt_quickstart.ipynb`](examples/predykt_quickstart.ipynb)
>, [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HishamSalem/predykt/blob/main/examples/predykt_quickstart.ipynb)
> It runs top to bottom on a free Colab CPU in roughly three minutes.

> **Note on runtime:** examples using large `n_null` / `n_bootstrap` / `n_seeds` or large `n_estimators` are illustrative of real production settings and can take minutes. For a fast smoke test, drop `n_null` and `n_bootstrap` to ~20 (and `SeedRobustnessValidator`'s `n_seeds` to ~20) and estimators to ~50. `InteractionTester` costs `1 + n_bootstrap + n_null + 1` model fits, each followed by a SHAP interaction pass. Reduce `n_bootstrap` first, since it does not affect `robust`.

### 0. The example dataset

Every example below runs against this one synthetic credit-risk frame. It is
built so `(age, income)` carries a **real** interaction: low income hurts the
young far more than the old, while the other pairs are additive, giving the
tools an honest negative control to fail to detect.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 4000

age              = rng.integers(21, 71, n).astype(float)
income           = np.round(rng.lognormal(10.9, 0.45, n), -2)
utilization_rate = rng.beta(2, 5, n)
delinquencies    = rng.poisson(0.4, n).astype(float)
loan_amount      = np.round(income * rng.uniform(0.2, 1.5, n), -2)
tenure           = rng.integers(0, 240, n).astype(float)
hour_bin         = rng.integers(0, 24, n).astype(float)
month            = rng.integers(1, 13, n).astype(float)

# Default risk. The (age, income) term is a genuine interaction: low income
# hurts the young far more than the old. Everything else enters additively,
# so the other pairs are honest negative controls.
z_age    = (age - age.mean()) / age.std()
z_income = (np.log(income) - np.log(income).mean()) / np.log(income).std()

# Risk also spikes overnight, 22:00-02:59, a window that wraps midnight, so
# no ordinary binner can express it as a single interval.
night = ((hour_bin >= 22) | (hour_bin <= 2)).astype(float)

logit = (-2.6
         - 0.35 * z_age
         - 0.55 * z_income
         + 1.60 * utilization_rate
         + 0.40 * delinquencies
         + 1.00 * night                      # <- the wrap-around effect
         - 1.10 * z_age * z_income)          # <- the interaction to recover

y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))), name="default")
X = pd.DataFrame({
    "age": age, "income": income, "utilization_rate": utilization_rate,
    "delinquencies": delinquencies, "loan_amount": loan_amount,
    "tenure": tenure, "hour_bin": hour_bin, "month": month,
})

# Categorical variant, used only by the adapter example in section 6.
X_cat = X.assign(
    state=rng.choice(["CA", "NY", "TX", "FL"], n),
    segment=rng.choice(["prime", "near_prime", "subprime"], n),
)
```

`X` is numeric-only, which `InteractionTester` requires. `X_cat` adds two string
columns and is used only by the adapter example in section 6.

### 1. Cyclical Optimal Binning

Standard binners treat hour 23 and hour 0 as maximally distant. `CyclicalBinner` treats the domain as circular and finds the IV-maximizing partition accordingly.

```python
from predykt import CyclicalBinner

binner = CyclicalBinner(m=24, gamma=0.02, k_max=6)
binner.fit(X["hour_bin"].to_numpy(int), y)

print(f"Optimal bins: {binner.n_bins_}")
print(f"Split points: {binner.split_points_}")
print(f"IV: {binner.iv_:.4f}")
print(binner.result_.summary())
```

**Transform to WOE for a scorecard:**

```python
binned      = binner.transform(X["hour_bin"].to_numpy(int))       # bin index
woe_encoded = binner.transform_woe(X["hour_bin"].to_numpy(int))   # WOE directly (for LR scorecards)
woe_table   = binner.result_.woe_table()    # WOE lookup table for documentation
```

**WOE sign convention:** `ln(%non-event / %event)`, matching [optbinning](https://github.com/guillermo-navas-palencia/optbinning) (Navas-Palencia 2020, §2.1) and Siddiqi's *Credit Risk Scorecards*. WOE is inversely related to the event rate, so a bin with an above-average event rate gets a **negative** WOE. Both conventions circulate in the credit-risk literature; the reason to pin one is that `FeatureBinningAnalyzer` delegates to optbinning directly, and mixing conventions inside a single scorecard silently sign-flips whichever features came from the odd one out. Information Value is unaffected, it is symmetric in the two factors.

> ⚠️ **Breaking change in 0.2.0.** `transform_woe()`, `get_woe_encoder()`, `woe_` and `result_.woe_table()` return the **opposite sign** to predykt ≤ 0.1.2, which used `ln(%event / %non-event)`. A scorecard fitted with `CyclicalBinner` WOE on ≤ 0.1.2 must be refit, or its coefficients on those features negated. `iv_`, `iv_smoothed` and the `iv` column of `summary()` are unchanged. See [`CHANGELOG.md`](CHANGELOG.md).

### 2. SHAP Interaction Testing Against an Additive Null

A SHAP interaction from a single model fit tells you nothing on its own: a pair of features with strong main effects and *no* interaction still produces a non-zero interaction value. `InteractionTester` measures the interaction magnitude and scores it against a null simulated from an **additive** surrogate fitted to the same data.

```python
import pandas as pd
from xgboost import XGBClassifier
from predykt import InteractionTester

tester = InteractionTester(
    model_class=XGBClassifier,
    base_params={
        "n_estimators": 200,
        "max_depth": 5,
        "eval_metric": "logloss",
        "verbosity": 0,
    },
    seed_param="random_state",
    n_null=100,           # additive-null replicates; drives the p-value
    n_bootstrap=100,      # descriptive interval only; cheap to reduce
    alpha=0.05,
    n_jobs=4,
)

# Step 1: cheap single-fit screen to identify candidate pairs
top_pairs = tester.get_top_n_interactions(X, y, n=10)

# Step 2: full test against the additive null
results = tester.test_pairs(X, y, top_pairs)

# Step 3: results with optional BH multiple-testing correction
df = tester.results_to_dataframe(results, correction_method="fdr_bh")
print(df[["Feature_i", "Feature_j", "Mean_Abs_Interaction", "P_Value",
          "OOF_Interaction_AUC", "Robust"]])
```

> `InteractionTester` requires **numeric-only** features. SHAP interaction values do not support native categorical splits. Encode categoricals (ordinal / target / WoE) before testing.

**How the null works.** Depth-1 stumps are additive by construction, a one-split tree is a function of a single feature, so no ensemble of them can carry an interaction. Fit those to `(X, y)`, draw `y* ~ Binomial(1, p_additive)`, refit the real model class on `(X, y*)`, and recompute `mean|Φ_ij|`. Repeating gives the distribution of interaction magnitude attributable to noise and to the estimator's own bias. The design follows the H-statistic's reference distribution in Friedman & Popescu (2008), §8.

- `p_value` = `(#{null ≥ observed} + 1) / (n_null + 1)`, bounded below by `1/(n_null + 1)`
- `robust` = `p_value < alpha` (**this is the decision rule**)
- `ci_low` / `ci_high` are a bootstrap **precision interval on the magnitude, descriptive only**. `mean|Φ_ij|` is strictly positive for any fitted tree ensemble, so this interval can never contain zero; `ci_low > 0` is a tautology that flags pure noise as significant. Do not use it as a criterion.

> ⚠️ **The interval runs above the point estimate**, and on a weak pair it excludes it. Measured on the reference DGP: true pair `mean_abs_interaction` 0.8301 against `[0.8398, 1.0907]`; null pair 0.1660 against `[0.1751, 0.2621]`. The bootstrap centres +13% and +27% high. This is a property of the resampling scheme, not of the sampling distribution: drawing rows with replacement duplicates them, and a tree ensemble fits duplicated rows more sharply, inflating `mean|Φ|`. The effect is larger on the weak pair, which is the signature of noise-fitting dominating when the true effect is near zero. Read the pair as the **spread of the bootstrap distribution**, not as an interval centred on the observed value. `robust` is unaffected, because it keys off `p_value`, which compares full-data fits against full-data fits.

> ⚠️ **Calibration limit.** The surrogate approximates the additive null; it is not the true null. The p-value is calibrated only insofar as depth-1 stumps capture the additive part of the data. Treat it as a principled screen with a real null, not an exact test.

**`oof_interaction_auc` is cross-fitted.** The interaction term's AUC is computed out of fold. Each fold fits on its training rows and explains only the held-out rows, so no row is scored by a model that saw it. Scoring in-sample is circular: on a DGP where `(x2, x3)` have main effects but **no** interaction, in-sample AUC read 0.72; cross-fitted it reads 0.53, and the truly-interacting pair reads 0.78. The old `max(auc, 1 - auc)` floor is also gone: it guaranteed ≥ 0.5 by construction, so the metric could never report "no discrimination" even when that was the truth. The direction is fixed **once** on the full-data fit and reused for every fold, choosing it per fold would reintroduce the same selection bias.

> This AUC answers "does this interaction term *alone* rank-order the target," which is narrower than "does adding this term improve the model." A cross-fitted ΔAUC would answer the latter and is the more standard choice; it is a roadmap item, not implemented. Note that DeLong et al. (1988) assumes fixed models, which cross-fitted predictions violate, so a paired bootstrap over folds would be the safer inference for this design.

```python
tester.plot_interaction_distribution(results[0])   # requires predykt[plot]
tester.plot_convergence(results[0])                # was n_bootstrap enough?
```

> **Removed in 0.2.0: `instability_score`.** It had no power under a deterministic learner, and reported `robust=True` for pure noise. See [`CHANGELOG.md`](CHANGELOG.md) for the migration.

### 3. Cross-Algorithm Voting

An interaction that is significant within XGBoost may be an artifact of gradient boosting's splitting strategy, not a property of the data. `InteractionVoter` runs the same test against each algorithm's own additive null and tallies votes.

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from predykt import InteractionVoter

configs = {
    "rf":   {"model_class": RandomForestClassifier,
             "params": {"n_estimators": 200, "max_depth": 5, "n_jobs": -1},
             "seed_param": "random_state"},
    "xgb":  {"model_class": XGBClassifier,
             "params": {"n_estimators": 200, "max_depth": 5, "eval_metric": "logloss", "verbosity": 0},
             "seed_param": "random_state"},
    "lgbm": {"model_class": LGBMClassifier,
             "params": {"n_estimators": 200, "max_depth": 5, "verbose": -1},
             "seed_param": "random_state"},
}

voter = InteractionVoter(configs, n_bootstrap=100, n_null=100, alpha=0.05, n_jobs=4)
vote_results = voter.vote(X, y, top_pairs)

summary = voter.summary(vote_results)
print(summary[["Feature_i", "Feature_j", "Votes", "Vote_Ratio", "Unanimous", "Mean_AUC"]])

voter.plot_vote_heatmap(vote_results)   # requires predykt[plot]
```

Unanimous interactions (every algorithm rejects its own additive null) are the most reliable candidates for feature engineering or regulatory documentation.

### 4. Seed Robustness Validation

HPO typically fixes a random seed during search, which means the "best" configuration may only be best for that initialization. `SeedRobustnessValidator` re-evaluates a fixed HP config across N seeds and runs formal statistical tests.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from predykt import SeedRobustnessValidator

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)

hp_config = {"n_estimators": 200, "max_depth": 6, "min_samples_split": 10}

def eval_fn(seed: int) -> float:
    clf = RandomForestClassifier(**hp_config, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    return roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

validator = SeedRobustnessValidator(
    eval_fn=eval_fn,
    n_seeds=100,
    metric_name="AUC",
    higher_is_better=True,
    sigma_max=0.005,   # domain-informed: 0.5% AUC std acceptable for production
)

report = validator.run()
validator.print_report(report)
validator.plot_diagnostics(report)   # requires predykt[plot]
```

**Statistical tests applied:**

| Test                                       | Purpose                                                        |
| ------------------------------------------ | -------------------------------------------------------------- |
| Shapiro-Wilk                               | Gates parametric vs bootstrap path                             |
| Chi-square variance test (one-sided upper) | H₀: σ² ≤ σ²_max                                                |
| 95/95 Tolerance interval                   | 95% confidence that 95% of future seed runs fall within [L, U] |
| Bootstrap CI for std                       | Non-parametric fallback when normality is violated             |
| Coefficient of Variation                   | Relative dispersion summary                                    |

**Verdict categories:** `ROBUST` / `MARGINAL` / `UNSTABLE`

> **Note on `sigma_max`:** if not set, defaults to 1% of the observed mean, a conservative auto-default. Override it with a domain-informed threshold. In credit scoring, 0.5% AUC std (`sigma_max=0.005`) is a reasonable production stability requirement.

### 5. Feature Binning IV Uplift

Quick screening for feature pair interactions using OptBinning's 2D binning. The uplift heuristic (`IV_2D - (IV_1 + IV_2)`) identifies pairs where joint information exceeds the sum of marginal information. This is a signal worth investigating further.

```python
from predykt import FeatureBinningAnalyzer

analyzer = FeatureBinningAnalyzer(X, y)

feature_pairs = [
    ("age", "income"),
    ("loan_amount", "tenure"),
    ("utilization_rate", "delinquencies"),
]

results = analyzer.analyze_feature_combinations(feature_pairs)
print(analyzer.get_top_combinations())

table = analyzer.get_binning_details("age", "income")
print(table)
```

> **Interpretation note:** IV uplift is a screening heuristic, not a formal interaction test. High-uplift pairs are candidates for the more rigorous `InteractionTester` / `InteractionVoter` pipeline.

### 6. Residual Representation Testing

After confirming an interaction pair is stable, `ResidualRepresentationTester` asks: does a specific engineered transformation of that pair explain structure the base model missed?

Stage 1 computes out-of-fold residuals `Ỹ = y − p̂` by K-fold cross-fitting. Stage 2 regresses `Ỹ` on the candidate representation `Tₖ` and tests `H₀: β₁ = 0`.

> **This is a feature-screening diagnostic, not a causal estimate.** A significant result is evidence that `Tₖ` correlates with structure the base model missed. Only the outcome is residualized, so validity depends on the base model being reasonably well specified. See the `predykt.residual_test` module docstring for the full statistical discussion and how this differs from FWL / DML. (The legacy `predykt.fwl` alias is deprecated for this reason and is removed in v0.4.0.)

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from predykt import ResidualRepresentationTester, OLSEstimator, HSICEstimator

# Candidate representations of the (age, income) interaction
reps = pd.DataFrame({
    "product":   X["age"] * X["income"],
    "ratio":     X["age"] / (X["income"] + 1),
    "log_ratio": np.log1p(X["age"]) - np.log1p(X["income"]),
})

tester = ResidualRepresentationTester(
    model=GradientBoostingClassifier(n_estimators=200, random_state=42),
    criterion=[OLSEstimator(), HSICEstimator(n_permutations=500)],
    n_folds=5,
    alpha=0.05,
)

tester.fit(feature_pairs=[("age", "income")], X=X, y=y, representations=reps)

# Summary table: beta, statistic, p-value, BH-corrected p-value, winner flag
print(tester.results_to_dataframe())

# Best representation per pair
winners = tester.winning_representations()

# Placebo + subsample refutation checks -> populates the `robust` column
tester.refute(n_permutations=100, n_bootstrap=50)
print(tester.results_to_dataframe()[["representation", "rejected", "robust"]])
```

> **Multiple testing:** BH correction (`pvalue_bh`) controls the false discovery rate **within each pair's set of representations**, not across many pairs screened in one run. When you test hundreds of pairs, treat the output as a ranked screen and apply a family-wide correction (or the `refute` step) before drawing firm conclusions.

**Criteria:**

| Criterion         | What it tests                                                           |
| ----------------- | ------------------------------------------------------------------------ |
| `OLSEstimator`    | Linear association (HC3 robust SE, handles heteroskedastic residuals)   |
| `HSICEstimator`   | Nonlinear / non-monotone dependence (kernel-based, permutation p-value) |
| `CustomEstimator` | Any user-supplied callable returning a `Stage2Result`                   |

**Precomputed residuals (Mode B):** if you already have OOF residuals, pass `Y_resid=` to skip Stage 1 cross-fitting.

```python
from sklearn.model_selection import cross_val_predict

# Any out-of-fold probability will do; this is just the cheapest illustration.
oof_p = cross_val_predict(
    GradientBoostingClassifier(n_estimators=200, random_state=42),
    X, y, cv=5, method="predict_proba",
)[:, 1]
precomputed_residuals = y - oof_p

tester.fit(
    feature_pairs=[("age", "income")],
    X=X, y=y,
    representations=reps,
    Y_resid=precomputed_residuals,
)
```

**Native-categorical models (adapters):** to cross-fit a CatBoost / LightGBM / XGBoost model with categorical columns, wrap it in an adapter so fit/predict dtype handling is applied consistently across folds.

```python
from catboost import CatBoostClassifier
from predykt import ResidualRepresentationTester, CatBoostAdapter

adapter = CatBoostAdapter(
    CatBoostClassifier(iterations=200, depth=5, verbose=0),
    cat_cols=["state", "segment"],
)
tester = ResidualRepresentationTester(model=adapter, n_folds=5)
tester.fit(feature_pairs=[("age", "income")], X=X_cat, y=y, representations=reps)
```

### 7. SHAP Interaction Analyzer

When a model contains both raw features and engineered interactions, raw SHAP values are aliased by collinearity. `SHAPInteractionAnalyzer` provides three progressively purer attribution layers.

```python
from predykt import SHAPInteractionAnalyzer

groups = {
    "demographics": ["age", "income"],
    "credit":       ["utilization_rate", "delinquencies", "loan_amount"],
    "temporal":     ["tenure", "hour_bin", "month"],
}

fitted_model = XGBClassifier(
    n_estimators=200, max_depth=4, eval_metric="logloss", verbosity=0,
).fit(X, y)

analyzer = SHAPInteractionAnalyzer(interaction_groups=groups, layers=[1, 2, 3])
analyzer.fit(model=fitted_model, X=X)

l1 = analyzer.layer_1_group_total()        # sum within group
l2 = analyzer.layer_2_net_group_effects()  # Layer 1 minus cross-group interactions
l3 = analyzer.layer_3_pure_main_effects()  # diagonal of shap_interaction_values

print(analyzer.summary(layer=1))
print(analyzer.summary(layer=2))

group_comparison, feature_effects = analyzer.compare_layers()
print(group_comparison)
```

**Reading the layers:**

| Comparison                          | What it tells you                                                               |
| ------------------------------------ | --------------------------------------------------------------------------------- |
| Layer 1 − Layer 2 per group          | How much of the group's apparent importance comes from cross-group interactions |
| Layer 2 − Σ(Layer 3 within group)    | Within-group collinearity aliasing even after cross-group correction            |

## Design Decisions

**Why simulate an additive null instead of permuting on a fixed model, or refitting across seeds?** Permuting on a fixed model tests whether the interaction is non-zero for that one fit, which is not the question. Refitting across seeds, which is what predykt ≤ 0.1.2 did, tests whether the interaction survives model randomness, but that has no power whenever the learner is deterministic: at `subsample=1.0` every seed gives a bit-identical fit, so the spread is exactly zero and every pair looks perfectly stable, noise included. Neither approach compares the interaction against anything. Simulating outcomes from an *additive* surrogate builds a reference distribution for "how big would this interaction look if there were no interaction at all," which is the comparison that licenses the word significant. Its cost is that the p-value inherits the surrogate's approximation error, stated plainly in §2 and in the module docstring.

**Why Numba for CyclicalBinner?** Exhaustive enumeration of all k-partitions of a circular domain of cardinality m is O(C(m, k)) per k. For m=24, k=6 that's C(24,6) = 134,596 partitions. Numba JIT brings this from seconds to milliseconds. The method is univariate and binary-target only; it is designed for low-cardinality circular domains (hours, months), not high-cardinality fields.

**Why the 95/95 tolerance interval in SeedRobustnessValidator?** A confidence interval on the mean tells you where the average seed lands. A tolerance interval tells you where individual seed runs land, which is what matters when you deploy a model trained on a single seed. The 95/95 interval follows the ISO 16269-6 convention for this use case.

**Why HC3 robust standard errors in OLSEstimator?** For binary targets, residuals Ỹ = y − p̂ have observation-specific variance p̂(1−p̂). OLS with homoskedastic standard errors is misspecified. HC3 (MacKinnon & White 1985) corrects this and is the default.

**Why HSIC alongside OLS?** OLS only detects linear association. HSIC (Hilbert–Schmidt Independence Criterion) is a kernel-based nonparametric test that detects any dependence structure, including nonlinear and non-monotone relationships. Running both gives a more complete picture of whether a representation carries signal.

## Testing

```bash
pip install -e ".[test,plot]"
pytest -q
```

CI runs the suite on Python 3.10 / 3.11 / 3.12 / 3.13 on every push.

## License

MIT

## Citation

If you use predykt in research or production systems, please cite it. Machine-readable
metadata lives in [`CITATION.cff`](CITATION.cff). GitHub renders a **Cite this repository**
button from it, which will give you BibTeX or APA directly.

```bibtex
@software{salem_predykt,
  author  = {Salem, Hisham},
  title   = {predykt: a toolkit for rigorous feature interaction analysis
             in machine learning},
  version = {0.2.0},
  year    = {2026},
  url     = {https://github.com/HishamSalem/predykt},
  license = {MIT}
}
```

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md). **0.2.0 contains breaking changes**: the
`CyclicalBinner` WOE sign convention and several `InteractionTester` field names.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development setup, the linting policy,
and the release process.
