# predykt fix list v2 — execution plan

Branch: `fix/interaction-and-release`. Baseline: `main` @ `ffd82dd`.
One commit per task, in the order below. `pytest -q` after every task.

---

## Task 0 — Fix version resolution (BLOCKING)

- [ ] `predykt/__init__.py`: defensive `_version` import → `importlib.metadata` → `"0.0.0.dev0"`.
- [ ] `pyproject.toml`: `[tool.setuptools_scm] fallback_version = "0.0.0.dev0"`.
- [ ] `.github/workflows/ci.yml`: `fetch-depth: 0` on both `actions/checkout@v4` steps.
- [ ] No git tag in this branch.
- Acceptance: fresh clone → `pip install -e .` → `import predykt` works; `pytest -q`
  collects all 8 modules (expect 60 passed, 1 failed — catboost absent locally).

## Task 1 — `HSICEstimator`: O(n²), pre-centered permutations

- [ ] `_center(M)` helper; `np.sum(Kc * Lc)` instead of `np.trace(Kc @ Lc)`.
- [ ] Center once outside the permutation loop; permute centered `Lc` via `np.ix_`.
      Exact, not approximate: `PᵀHP = H` ⇒ `Pᵀ(HLH)P = H(PᵀLP)H`. Document the why.
- [ ] Center one matrix only (`H` idempotent ⇒ `tr(Kc·Lc) = tr(Kc·L)`).
- [ ] `UserWarning` (not raise) when `n > 5000`.
- [ ] Notes: Gretton et al. 2005 (primary); Zhang et al. 2018 + gamma approx as roadmap.
- [ ] Test: statistic unchanged within float tolerance on fixed seed.

## Task 2 — `CyclicalBinner`: WOE sign convention

- [ ] Line 448 `woe[j] = np.log(q_j / p_j)`; flip the `(p−q)` factor at lines 449 and 211.
- [ ] Do NOT touch lines 92 (`_solve_for_k` kernel) or 456 (`iv_raw`) — self-contained.
- [ ] Docstrings: convention = ln(%non-event / %event), matching optbinning. Not "the standard".
- [ ] README §1 convention note.
- [ ] Test: directional agreement with `optbinning`; `iv_` / `iv_smoothed` / `n_bins_` /
      `split_points_` unchanged; only `woe_` and `woe_table()` flip.

## Task 3 — `InteractionTester`: replace the stability statistic

- [ ] Statistic → `mean(|Φ_ij|)`; `mean_interaction` → `mean_abs_interaction`.
- [ ] Bootstrap over rows → descriptive `ci_low`/`ci_high` only, NOT a decision rule.
      `n_seeds` → `n_bootstrap` with a deprecated alias.
- [ ] Friedman–Popescu null: additive depth-1 surrogate → `y* ~ Binomial(1, p_add)`
      → refit → null distribution → `p_value` → `robust = p_value < alpha`.
      `n_null: int = 100`; warn if `alpha < 1/(n_null+1)`; `null_surrogate=None` hook.
- [ ] Retire `_compute_instability_score`.
- [ ] Update `results_to_dataframe`, both plots, `InteractionVoter`.
- [ ] Docstrings: drop "instability_score" framing; cite Friedman & Popescu 2008 §8;
      state the surrogate-calibration limitation.
- [ ] Regression test: null pair not robust, true pair robust. Must fail on `main`.

## Task 4 — `InteractionTester`: fix `per_interaction_auc` circularity

- [ ] K-fold cross-fitting of SHAP interaction values (`n_folds: int = 5`),
      following `fwl.py::_compute_residuals`.
- [ ] Remove `max(auc, 1 - auc)`. Fix the sign ONCE outside the fold loop.
- [ ] `per_interaction_auc` → `oof_interaction_auc`; delete the duplicate `mean_auc` field.
- [ ] Notes: cross-fitted ΔAUC + DeLong 1988 with its fixed-model caveat, as roadmap.
- [ ] Tests: null pair OOF AUC ∈ [0.45, 0.55]; true pair ≥ 0.70.

## Task 5 — Rename `fwl.py` → `residual_test.py`

- [ ] `git mv`, update `predykt/__init__.py:17`, add `__all__` to `residual_test.py`.
- [ ] `predykt/fwl.py` deprecation shim (remove in v0.4.0).
- [ ] `git mv tests/test_fwl.py tests/test_residual_test.py`; logger name → `predykt.residual_test`.
- [ ] Do NOT rewrite README §6 / Design Decisions FWL paragraphs — they are correct.

## Task 6 — `FeatureBinningAnalyzer`: document, don't delete

- [ ] Module docstring: distinction from `CyclicalBinner`; delegates to optbinning so it
      already uses ln(%non-event / %event) — Task 2's flip does not apply.
- [ ] `pytest.importorskip("catboost")` on `test_catboost_mode_a`.
- [ ] No behaviour change.

## Task 7 — Release hygiene

- [ ] `CHANGELOG.md` (Keep a Changelog); 0.2.0 with `### Changed (BREAKING)` WOE flip.
- [ ] `CITATION.cff`.
- [ ] `CONTRIBUTING.md` release-process note: tag before `python -m build`.
- [ ] No tag, no release in this branch.

## Task 8 — Publish 0.2.0 (MANUAL, owner action — not part of the PR)

Documented only. Not executed here.
