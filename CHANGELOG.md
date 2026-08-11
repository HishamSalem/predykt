# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-31

### Changed (BREAKING)

- **Dependency floors raised: `scikit-learn>=1.6` and `optbinning>=0.21`**, replacing
  `scikit-learn>=1.1,<1.8` and `optbinning>=0.17`. The old ceiling made predykt
  uninstallable next to a current scikit-learn (1.9.0 at time of writing). On hosted
  notebooks that is not cosmetic: pip downgrades the preinstalled scikit-learn and the
  already-imported module stays in memory until the kernel restarts.

  The cap was aimed at the wrong package. optbinning up to 0.20.1 calls
  `check_array(force_all_finite=...)`, removed in scikit-learn 1.8; 0.21.0 is the first
  release using `ensure_all_finite`, and it declares `scikit-learn>=1.6.0`. Constraining
  optbinning instead of capping scikit-learn keeps predykt installable on current
  environments. Verified against scikit-learn 1.6.1, 1.7.2 and 1.9.0.

  Drops support for `scikit-learn` 1.1–1.5 and `optbinning` 0.17–0.20.


- **`CyclicalBinner` WOE sign convention flipped** to `ln(%non-event / %event)`, matching
  [optbinning](https://github.com/guillermo-navas-palencia/optbinning)
  (Navas-Palencia 2020, arXiv:2001.08025 §2.1) and Siddiqi, *Credit Risk Scorecards*.
  WOE is now inversely related to the event rate: a bin with an above-average event rate
  gets a **negative** WOE.

  `transform_woe()`, `get_woe_encoder()`, `woe_` and `result_.woe_table()` all return the
  opposite sign to ≤ 0.1.2.

  **Migration:** a scorecard fitted with `CyclicalBinner` WOE on ≤ 0.1.2 must be refit, or
  the coefficients on those features negated. There is no silent-failure warning available
  for this — a sign-flipped feature still trains, it just inverts that feature's
  contribution.

  `iv_`, `iv_smoothed` and the `iv` column of `summary()` are **unchanged** — verified
  identical to 8 decimal places. IV is symmetric in its two factors, and both were flipped
  together; flipping only the WOE would have negated `iv_smoothed`.

  This was previously inconsistent *within* predykt: `FeatureBinningAnalyzer` delegates to
  optbinning directly and always used optbinning's convention, so WOE-encoding one feature
  with `CyclicalBinner` and the rest with `FeatureBinningAnalyzer` put a sign-flipped
  feature into the same model.

- **`InteractionResult.instability_score` removed.** It measured the proportion of seeds on
  which the signed mean interaction flipped direction, and had no power under a
  deterministic learner: with XGBoost at `subsample=1.0` every seed produces a
  bit-identical fit, so the score was exactly `0.0` for every pair — including pure noise —
  and `robust` was `True` for everything. The underlying statistic was also signed, and
  SHAP interaction values are roughly sign-symmetric across rows, so the signed mean
  discarded ~95% of the magnitude it was meant to measure.

  Replaced by `p_value` against a simulated additive null, with
  `robust = p_value < alpha`. See **Added** below.

- **`per_interaction_auc` → `oof_interaction_auc`**, and it is now cross-fitted. The old
  value scored the SHAP interaction values of a model on the very rows it was fitted to;
  on a reference DGP that gave ~0.72 for a pair with **no** interaction, where the honest
  answer is ~0.50. Cross-fitted, the same pair reads 0.53 and the truly-interacting pair
  reads 0.78.

  `max(auc, 1 - auc)` was also removed. It floored the metric at 0.5 by construction, so it
  could never report "no discrimination" even when that was the truth. AUC below 0.5 is now
  reported honestly.

- **`mean_interaction` → `mean_abs_interaction`**, now the mean of `|Φ_ij|` rather than the
  signed mean.

- `InteractionTester.results_to_dataframe()` column names follow the above:
  `Instability_Score` → `P_Value`, `Mean_Interaction` → `Mean_Abs_Interaction`,
  `Per_Interaction_AUC` → `OOF_Interaction_AUC`. `Adjusted_Instability_Score` /
  `Robust_Adjusted` → `P_Value_Adjusted` / `Robust_Adjusted`. The Benjamini-Hochberg
  correction now applies to a real p-value; previously it was applied to a quantity that
  was not a p-value at all.

- `InteractionVoter.summary()` emits `{algo}_p_value` and `{algo}_oof_auc` in place of
  `{algo}_instability` and `{algo}_auc`.

### Deprecated

- `n_seeds` → `n_bootstrap` on `InteractionTester` and `InteractionVoter`. The constructor
  argument, the `n_seeds` property and the `seeds` argument to `test_pairs()` all still
  work and emit `DeprecationWarning`. Removal in v0.4.0.
  (`SeedRobustnessValidator.n_seeds` is a different, unaffected parameter.)
- `predykt.fwl` → `predykt.residual_test`. The old module remains as a shim that
  re-exports everything and emits `DeprecationWarning`. Removal in v0.4.0. The module was
  named after the Frisch-Waugh-Lovell theorem, which the procedure does not implement —
  FWL residualizes both sides of the regression, and only the outcome is residualized here.

### Added

- **`examples/predykt_quickstart.ipynb`** — every README example as an executable
  notebook, runnable on a free Colab CPU in about three minutes. A `FAST` switch
  trades sample size and replicate counts against fidelity to the README's
  production-scale settings.


- `InteractionTester(n_null=...)` — replicates drawn from the additive null, which drive
  `p_value`. Default 100. The smallest attainable p-value is `1/(n_null + 1)`; a warning
  is emitted when `alpha` is below it, since `robust` could then never be `True`.
- `InteractionTester(null_surrogate=...)` — inject your own additive surrogate. When left
  as `None` one is built from `model_class` with its depth parameter set to 1; a clear
  error is raised if the class has no recognisable depth parameter.
- `InteractionTester(n_folds=...)` — K for the cross-fitted `oof_interaction_auc`.
  Default 5.
- `InteractionResult.ci_low` / `ci_high` — 2.5/97.5 percentiles of the row bootstrap.
  **Descriptive only, not a decision rule:** `mean|Φ_ij|` is strictly positive for any
  fitted tree ensemble, so this interval can never contain zero and `ci_low > 0` is a
  tautology that flags pure noise as significant. It also runs *above*
  `mean_abs_interaction` and on a weak pair excludes it — bootstrap resampling duplicates
  rows, and a tree ensemble fits duplicated rows more sharply. Documented on the field and
  in the README.
- `InteractionResult.null_mean` and `null_distribution` — the additive null's location and
  full draw, for plotting and for reporting effect size as `observed / null_mean`.
- `InteractionTester(random_state=...)` — seeds the bootstrap resampling and null draws.

### Fixed

- **SHAP interaction values from scikit-learn ensembles were not reduced to the
  positive class.** `_shap_interaction_values` documents a return of `(n, p, p)`,
  but only handled the legacy list-per-class form. Binary classifiers also report
  per-class values as a single `(n, p, p, n_classes)` array, and on current shap
  that is the path scikit-learn's forests take — so a 4-D array reached the
  callers and `InteractionTester` / `InteractionVoter` died on
  `RandomForestClassifier` and `ExtraTreesClassifier` with

      ValueError: shape mismatch: value array of shape (n, 2) could not be
      broadcast to indexing result of shape (n,)

  The cross-algorithm voting example in the README is a RandomForest + XGBoost +
  LightGBM configuration, so it could not have run as published.
  `SHAPInteractionAnalyzer.fit` already normalised both forms; the two modules
  now agree.
- **The wheel shipped a top-level `tasks/` directory into `site-packages`.**
  `[tool.setuptools.packages.find]` defaults to `namespaces = true`, so it
  discovered any root directory as a namespace package even without an
  `__init__.py`. Replaced `exclude = ["tests*"]` with `include = ["predykt*"]`,
  which cannot regress when a new root directory is added. Published 0.1.2 is
  unaffected — `tasks/` postdates it. The sdist still carries `tests/` and
  `docs/`, which is correct.
- `HSICEstimator`: O(n³) → O(n²) per permutation. Centering is computed by mean
  subtraction and hoisted out of the permutation loop, and the trace of the product is read
  off as an elementwise sum. The statistic and the permutation p-value are unchanged —
  agreement to at least 15 decimal places. n=2000 with 500 permutations: 633s → 22s.
- `import predykt` no longer fails on a fresh clone before any build has run.
  `predykt/_version.py` is generated by setuptools-scm at build time and is gitignored, so
  a fresh checkout previously raised `ModuleNotFoundError` — which blocked collection of
  every test module, not just the import.
- Row-aligned `fit_params` (notably `sample_weight`) are now subset alongside the rows for
  both bootstrap draws and cross-fitting folds. Previously a full-length weight vector was
  passed to a resampled or held-out subset.
- `HSICEstimator` warns rather than raising above `n = 5000`, so large-n exploratory use
  still works.

### Documentation

- WOE sign convention stated explicitly in `CyclicalBinner`, `BinningResult`,
  `FeatureBinningAnalyzer` and the README — one statement per module, never assumed.
- `FeatureBinningAnalyzer`'s module docstring distinguishes it from `CyclicalBinner`
  (pairwise IV-uplift screening over arbitrary pairs, versus exact optimal binning of a
  single circular feature) and notes that it delegates to optbinning, so the sign flip
  above does not apply to its tables.
- `HSICEstimator` cites its primary reference (Gretton et al. 2005), which was absent, and
  records the gamma approximation and block HSIC as roadmap items.
- Cross-fitted ΔAUC recorded as the more standard alternative to `oof_interaction_auc`,
  with the caveat that DeLong et al. (1988) assumes fixed models — an assumption
  cross-fitted predictions violate.

- On Python 3.10, `shap.TreeExplainer` cannot read models from `xgboost >= 3.1`, failing
  with `could not convert string to float: '[5.4E-1]'`. shap fixed its `base_score` parsing
  in 0.50, but shap 0.50+ requires Python >= 3.11, so 3.10 is capped at shap 0.49.1. predykt
  now raises a `RuntimeError` naming the version clash instead of letting the raw error
  through, and the `test` extra pins `xgboost < 3.1` on Python 3.10. Users on 3.10 who need
  `xgboost >= 3.1` must upgrade to Python 3.11+.

### Known limitations

- The additive null's p-value is calibrated only insofar as the depth-1 surrogate
  approximates the additive part of the data. It is a principled screen with a real null,
  not an exact test.
- `oof_interaction_auc` answers "does this interaction term alone rank-order the target",
  which is narrower than "does adding this term improve the model".

## [0.1.2] - 2026-04-30

Released before this changelog existed.

## [0.1.1] - 2026-03-16

Released before this changelog existed.

[Unreleased]: https://github.com/HishamSalem/predykt/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/HishamSalem/predykt/releases/tag/v0.2.0
[0.1.2]: https://github.com/HishamSalem/predykt/releases/tag/v0.1.2
[0.1.1]: https://github.com/HishamSalem/predykt/releases/tag/v0.1.1
