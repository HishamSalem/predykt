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

## Task 2 — README §1 has no WOE example output table

The fix list said to update "the example output table's `woe` column" in README §1 and
named a specific value (`bin 2, event_rate 0.1495, woe = +1.1803`). No such table exists
in `README.md` at `ffd82dd`: `grep -n "1.1803\|0.1495" README.md` returns nothing, and
§1 only shows the code that *calls* `binner.result_.summary()`, never its output. There is
also no prose in the README that assumes a WOE sign.

Substituted action: added an explicit convention note to README §1 stating
ln(%non-event / %event) and its optbinning consistency, which is the substance the task
was after.
