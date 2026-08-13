"""Evidence fusion with explicit dependence modelling.

The system's central claim is that weakly individuating traces combine into an
actionable one. This package is where that combination happens, and where the
assumption that would invalidate it is confronted rather than made.

``base``
    Contracts. Notably :class:`~viflap.analysis.fusion.base.FusionModel`, whose
    ``supports_pattern`` exists so a model can *decline* a comparison whose
    missing streams it cannot handle, rather than imputing them.
``models``
    Naive summation (the prohibited baseline), linear logistic fusion, a
    Gaussian latent model with exact marginalisation over absent streams, and a
    Gaussian copula separating marginal calibration from dependence.
``overstatement``
    How many orders of magnitude the independence assumption would have added.
    Measured on every comparison, reported with every result.

Two rules hold throughout. A stream that produced nothing is never given a
likelihood ratio of one — absence and neutrality are different states, and
conflating them fabricates an observation. And naive summation, though
implemented, may not produce a reported multi-stream result; the domain type
refuses to construct one.
"""

from viflap.analysis.fusion.base import (
    FusionModel,
    FusionObservation,
    FusionTrainingSet,
    StreamVector,
    to_matrix,
)
from viflap.analysis.fusion.models import (
    CommonFactorSummary,
    GaussianCopulaFusion,
    GaussianLatentFusion,
    LinearLogisticFusion,
    NaiveIndependentFusion,
)
from viflap.analysis.fusion.overstatement import (
    OverstatementReport,
    measure_overstatement,
)

__all__ = [
    "CommonFactorSummary",
    "FusionModel",
    "FusionObservation",
    "FusionTrainingSet",
    "GaussianCopulaFusion",
    "GaussianLatentFusion",
    "LinearLogisticFusion",
    "NaiveIndependentFusion",
    "OverstatementReport",
    "StreamVector",
    "measure_overstatement",
    "to_matrix",
]
