"""Deprecated alias for predykt.residual_test.

The module was named `fwl` after the Frisch-Waugh-Lovell theorem, which this
procedure does not implement: FWL residualizes both sides of the regression,
and only the outcome is residualized here. See predykt.residual_test for the
full discussion.
"""

# TODO: remove in v0.4.0
import warnings

from .residual_test import *  # noqa: F401,F403
from .residual_test import __all__  # noqa: F401

warnings.warn(
    "predykt.fwl is deprecated; import from predykt.residual_test instead.",
    DeprecationWarning, stacklevel=2,
)
