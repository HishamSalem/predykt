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

## Task 2 — README §1 has no WOE example output table

The fix list said to update "the example output table's `woe` column" in README §1 and
named a specific value (`bin 2, event_rate 0.1495, woe = +1.1803`). No such table exists
in `README.md` at `ffd82dd`: `grep -n "1.1803\|0.1495" README.md` returns nothing, and
§1 only shows the code that *calls* `binner.result_.summary()`, never its output. There is
also no prose in the README that assumes a WOE sign.

Substituted action: added an explicit convention note to README §1 stating
ln(%non-event / %event) and its optbinning consistency, which is the substance the task
was after.
