# Development notes

Working notes from the 0.2.0 rework, kept because the measurements in them
are not recorded anywhere else. These are **not** user documentation — start
from the [top-level README](../../README.md) instead.

- [`lessons.md`](lessons.md) — places where a planned fix's stated facts did
  not survive measurement, and what was measured instead. Includes the
  shap/xgboost version bisection behind the `xgboost < 3.1` pin on Python
  3.10, and the Type-I rates behind the calibration tests' bounds.
- [`todo.md`](todo.md) — the execution plan for that rework, kept as a record
  of what was changed and why.
