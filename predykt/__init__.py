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
from .fwl import ResidualRepresentationTester
from .interaction_stability import InteractionTester, InteractionVoter
from .seed_robustness import SeedRobustnessValidator
from .shap_analyzer import SHAPInteractionAnalyzer

from ._version import version as __version__
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
