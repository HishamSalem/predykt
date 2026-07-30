from .adapters import (
    CatBoostAdapter,
    ModelAdapter,
    PandasCategoricalAdapter,
    SklearnAdapter,
    resolve_adapter,
)
from .criteria import (
    CustomEstimator,
    HSICEstimator,
    OLSEstimator,
    Stage2Estimator,
    Stage2Result,
)
from .cyclical_transformer import CyclicalBinner
from .feature_binning import FeatureBinningAnalyzer
from .interaction_stability import InteractionTester, InteractionVoter
from .residual_test import ResidualRepresentationTester
from .seed_robustness import SeedRobustnessValidator
from .shap_analyzer import SHAPInteractionAnalyzer

try:
    from ._version import version as __version__
except ImportError:  # no build has run (fresh clone / editable dev checkout)
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version
    try:
        __version__ = _pkg_version("predykt")
    except PackageNotFoundError:
        __version__ = "0.0.0.dev0"

__author__ = "Hisham Salem"

__all__ = [
    "FeatureBinningAnalyzer",
    "CyclicalBinner",
    "InteractionTester",
    "InteractionVoter",
    "SeedRobustnessValidator",
    "Stage2Estimator",
    "Stage2Result",
    "OLSEstimator",
    "HSICEstimator",
    "CustomEstimator",
    "ResidualRepresentationTester",
    "SHAPInteractionAnalyzer",
    "ModelAdapter",
    "SklearnAdapter",
    "PandasCategoricalAdapter",
    "CatBoostAdapter",
    "resolve_adapter",
]
