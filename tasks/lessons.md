# Lessons / corrections to the v2 fix list

Log of places where the v2 fix list's stated facts did not match the repo as found.

---

## Task 0 — expected test count was `60 passed, 1 failed`; actual is `61 passed`

The fix list expected `tests/test_fwl.py::TestCategoricalAdapters::test_catboost_mode_a`
to fail locally because catboost is absent. In this checkout's `.venv` catboost **is**
installed, so the suite is fully green (`61 passed`) after Task 0 alone.

This does not invalidate Task 6's `pytest.importorskip("catboost")` — the guard is still
correct for any checkout without the `[test]` extra installed. It just means the guard
cannot be *demonstrated* to change the outcome here.

---

## Task 0 — `fallback_version` does not do what the fix list says it does

The fix list justified `fallback_version` as a "build-layer fallback so an untagged build
cannot emit a local version." It does not have that effect. `fallback_version` engages
only when setuptools-scm cannot detect a version *at all*. Measured on throwaway clones:

| Scenario | Resolved version |
|---|---|
| Fresh clone, untagged, `pip install -e .`, **before** this change | `0.1.dev50+gffd82dd94` |
| Fresh clone, untagged, `pip install -e .`, **after** this change | `0.0.0.dev51+g954a37bf6` |
| Same clone with `v0.2.0` tagged | `0.2.0` |
| Source tree with no `.git` at all | `0.0.0.dev0` |

The `+g<hash>` local segment survives on an untagged git checkout either way — only
tagging removes it. What `fallback_version` genuinely buys is row 4: without it, a build
from an sdist or a downloaded zip raises `LookupError` and fails outright.

Kept the setting for that reason, and corrected the comment in `pyproject.toml` to state
what it actually does. Deliberately did **not** add `local_scheme = "no-local-version"`,
which would suppress the suffix: PyPI's rejection of local versions is the safety net that
stops an untagged build being published, and Task 8's own verification step ("verify the
wheel version has no `+g<hash>` suffix") depends on that suffix still appearing when the
tag is missing. Tagging before `python -m build` is the real fix, documented in Task 7.

## Task 0 — the `import predykt` failure, and what actually proves it fixed

Reproduced on this checkout with `predykt/_version.py` absent (setuptools-scm generates it
at build time; it is gitignored):

- Before: `ModuleNotFoundError: No module named 'predykt._version'`, blocking collection of
  all 8 test modules.
- After: `61 passed`, with `_version.py` still absent.
- With predykt not installed anywhere *and* no `_version.py`: `0.0.0.dev0`, i.e. the
  terminal fallback resolves rather than raising.

Note the middle rung of the chain does real work: on a checkout where the package is
installed but unbuilt, the version comes from `importlib.metadata`, not from
`"0.0.0.dev0"`.

---

## Task 1 — the claimed 43.7× per-permutation speedup does not reproduce (the ~15s target does)

Measured on this machine, same script before and after:

| n | before | after | speedup |
|---|---|---|---|
| 1000 | 38.45 ms/perm | 8.49 ms/perm | 4.5× |
| 2000 | 182.32 ms/perm | 37.24 ms/perm | 4.9× |

The fix list quoted 1266 ms/perm at n=2000 before, and 43.7×. The *after* figures agree
(1266/43.7 ≈ 29 ms vs 37 ms measured); the *before* figures differ by ~7×. The baseline is
three n×n matrix products, so its wall time is set almost entirely by how many threads the
installed BLAS uses — 182 ms at n=2000 is ~264 GFLOPS, 1266 ms is ~38 GFLOPS, i.e. the fix
list's baseline looks effectively single-threaded. The ratio is a property of the
measuring machine, not of the change.

The headline acceptance criterion does hold: **n=2000 × 500 permutations, 17.0 s** end to
end (fix list said ~15 s), against a 91 s projection for the baseline's permutation loop
alone — and that projection excludes the baseline's per-permutation O(n³) centering, so the
true baseline is far worse.

Reporting 4.9× rather than 43.7×. The machine-independent claim is the order reduction:
O(n³) per permutation → O(n²) per permutation, with the O(n³) setup removed as well.

## Task 1 — items 3 and 4 of the task partly cancel each other

Item 3 says to center both matrices outside the loop and permute the already-centered `Lc`.
Item 4 says only one matrix needs centering. Taking item 4 makes item 3's mechanism moot:
the matrix that gets centered (`K`) is the one that is never permuted, so there is no
"permute the centered matrix" step left to justify. Implemented as `Kc = _center(K)` hoisted
out of the loop, with the raw `L` permuted inside it.

Both exactness arguments are still documented on `_hsic_from_centered`, because a reader
will ask why centering can be hoisted at all — idempotency of H for dropping L's centering,
and `PᵀHP = H` for the permutation commuting with centering.

## Task 1 — `_center` as literally specified would have kept the O(n³) cost

The task specifies `_center(M)` as "`H @ M @ H` with `H = I − 11ᵀ/n`". Building `H` and
evaluating that expression is two n×n matrix products, i.e. O(n³) — so hoisting it out of
the permutation loop would still leave an O(n³) term dominating `fit`, contradicting the
task's own title ("O(n²) instead of O(n³)"). Implemented instead by the equivalent
mean-subtraction identity `HMH = M − colmeans − rowmeans + grandmean`, which is O(n²).
`tests/test_criteria.py::test_center_matches_explicit_HMH` pins it to the explicit form.

## Task 3 — separation is much cleaner than the fix list reported

The fix list's verification used 20 null replicates and was therefore sitting on the
resolution floor. At `n_null=100` on the reference DGP:

| | fix list (20 reps) | measured (100 reps) |
|---|---|---|
| true pair | observed 0.8941, null mean 0.1345, p = 0.048 | observed 0.8301, null mean 0.1337, q95 0.1577, **p = 0.0099** |
| null pair | observed 0.1779, null mean 0.1566, p = 0.095 | observed 0.1660, null mean 0.1646, q95 0.1892, **p = 0.4851** |

The null pair's p-value moves from 0.095 (uncomfortably close to alpha) to 0.485 — the null
mean 0.1646 is essentially the observed 0.1660, which is exactly what "no interaction here"
should look like. The fix list's own advice to use >= 100 replicates for release is borne out.

Also reproduced the four defects on the current code before changing anything: signed-mean
ratio 5.9% (true) / 2.9% (null); std across 5 seeds exactly 0.0; bootstrap CI
[0.859, 1.056] and [0.163, 0.250] with `ci_low > 0` True for both — the tautology.

## Task 3 — `get_top_n_interactions` ranks the NULL pair first under a weak learner

Tried to strengthen `test_get_top_n` with "the true pair should rank first". It fails, and
correctly so. With LGBM at 15 estimators / depth 3 on n=600, the single-fit screen ranks:

```
x2 x x3: 0.06901   <- the NULL pair, first
x0 x x3: 0.02111
x1 x x2: 0.02016
x0 x x2: 0.01831
x1 x x3: 0.01810
x0 x x1: 0.00163   <- the TRUE pair, last
```

An underfit shallow model has not learned the `x0·x1` saddle at all, while `x2` and `x3`
have strong main effects that leak into their pairwise SHAP interaction term. This is
precisely the false-positive behaviour the method's own docstring warns about ("a cheap
pre-filter, not a final result"), so asserting the opposite would contradict the documented
contract. Assertion dropped, with the finding recorded in the test as a comment.

It is also the single best illustration of why Task 3 was necessary: the pair the screen
ranks first is the pair the additive null rejects hardest (p = 0.485).

## Task 3 — bootstrapping rows breaks row-aligned `fit_params`, which the task did not mention

`fit_params` is forwarded verbatim to every `fit()`. Once replicates resample rows, a
row-aligned entry such as `sample_weight` still has the right *length*, so nothing errors —
it just silently pairs each resampled row with a different row's weight. The pre-existing
code never resampled, so the bug could not arise before this task.

`sample_weight` is now reindexed with the bootstrap draw. It is special-cased rather than
handled generically because `fit_params` is an open dict with no way to distinguish a
row-aligned array from a scalar hyperparameter; any *other* entry that looks row-aligned
(len == n) is passed through with a `UserWarning`.

## Task 4 — cross-fitting broke row-aligned `fit_params` louder than bootstrapping did

The same `fit_params` problem from Task 3, but a CV fold is *shorter* than the full data, so
a full-length `sample_weight` does not merely misalign — LightGBM rejects it outright
("Length of weights differs from the length of #data"). Caught by the existing
`test_fit_params_full_length_sample_weight`.

Generalised the Task 3 helper from `_resampled_fit_params(idx)` to
`_subset_fit_params(idx, n_full)`, used by both the bootstrap draws and the fold fits. The
`n_full` argument matters: the old signature inferred the full row count from `len(idx)`,
which is right for a bootstrap draw (same size) but wrong for a fold, so the "looks
row-aligned" warning silently stopped firing for folds.

## Task 4 — "no `max(auc, 1 - auc)` anywhere in the file" cannot be taken literally

The acceptance criterion is a bare substring check, but the docstring explaining *why* the
floor was removed necessarily contains the expression. Scanning raw source therefore fails
on the very documentation the task asked for.

The test tokenises the module and strips `STRING` and `COMMENT` tokens before checking, so
it constrains executable code only. Measured values, reference DGP:

| | in-sample (old) | cross-fitted (new) |
|---|---|---|
| true pair | 0.8183 | **0.7781** (fix list predicted 0.777) |
| null pair | 0.7214 | **0.5296** (fix list predicted 0.518) |

---

## Task 2 — README §1 has no WOE example output table

The fix list said to update "the example output table's `woe` column" in README §1 and
named a specific value (`bin 2, event_rate 0.1495, woe = +1.1803`). No such table exists
in `README.md` at `ffd82dd`: `grep -n "1.1803\|0.1495" README.md` returns nothing, and
§1 only shows the code that *calls* `binner.result_.summary()`, never its output. There is
also no prose in the README that assumes a WOE sign.

Substituted action: added an explicit convention note to README §1 stating
ln(%non-event / %event) and its optbinning consistency, plus a breaking-change callout,
which is the substance the task was after.

## Task 2 — `summary()`'s per-bin `iv` column never summed to `iv_raw` (pre-existing)

Found while writing the acceptance test. `BinningResult.summary()` computes each bin's `iv`
as `(q−p)·woe[j]` using *unsmoothed* p/q but the *Laplace-smoothed* `woe_`, so the column
sums to neither `iv_raw` nor `iv_smoothed`. On the README's fixture the gap is small
(0.4010369 vs 0.4010618); on a fixture with `lam=0.5` and small bins it reaches ~0.15%.

Pre-existing and unrelated to the sign flip — visible in the pre-flip snapshot — and no
task authorises changing `summary()`'s output, so it is left alone. The acceptance test
asserts per-bin positivity and only a loose match to `iv_raw`. Flagged for the owner.
