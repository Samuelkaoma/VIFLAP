"""Acoustic speaker comparison.

The stack, bottom to top, each stage answering a limitation of the one below:

``gmm``
    Diagonal-covariance Gaussian mixture and universal background model. The
    statistical description of speech in general, against which any particular
    recording is a deviation.
``ivector``
    Total variability modelling. Reduces a 15,000-dimensional supervector of
    adapted means to a few hundred latent factors, with a posterior that
    formally expresses how much a short recording failed to determine them.
``transforms``
    Length normalisation, LDA and WCCN. Makes the representation approximately
    Gaussian and removes directions dominated by session and channel
    variability.
``plda``
    Two-covariance probabilistic linear discriminant analysis. Models both the
    same-source and different-source distributions, which is what turns a
    similarity into a ratio.
``pipeline``
    The assembled system, trainable and serialisable, with a content-derived
    model identity.

The output of this package is an **uncalibrated score**. It is not evidence, and
nothing here returns a :class:`~viflap.domain.values.LogLikelihoodRatio`. The
PLDA score is derived as a ratio under Gaussian assumptions that are
approximations, from a training population that is not the relevant population,
so it requires empirical calibration on held-out data before it can be reported.
The fact that it is already *shaped* like a likelihood ratio makes that step
easier to skip, not less necessary.
"""

from viflap.analysis.speaker.gmm import (
    BaumWelchStatistics,
    GaussianMixture,
    GmmConfig,
    train_ubm,
)
from viflap.analysis.speaker.ivector import (
    IVectorExtractor,
    IVectorPosterior,
    TotalVariabilityConfig,
    train_total_variability,
)
from viflap.analysis.speaker.pipeline import (
    AcousticEmbedding,
    FrontEndConfig,
    SpeakerComparisonSystem,
    SpeakerSystemConfig,
    TrainingRecording,
)
from viflap.analysis.speaker.plda import PldaConfig, PldaModel, train_plda
from viflap.analysis.speaker.transforms import IVectorTransform, fit_transform_chain

__all__ = [
    "AcousticEmbedding",
    "BaumWelchStatistics",
    "FrontEndConfig",
    "GaussianMixture",
    "GmmConfig",
    "IVectorExtractor",
    "IVectorPosterior",
    "IVectorTransform",
    "PldaConfig",
    "PldaModel",
    "SpeakerComparisonSystem",
    "SpeakerSystemConfig",
    "TotalVariabilityConfig",
    "TrainingRecording",
    "fit_transform_chain",
    "train_plda",
    "train_total_variability",
    "train_ubm",
]
