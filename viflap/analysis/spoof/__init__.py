"""Synthetic speech detection and the validity gate.

This package conditions the admissibility of acoustic evidence. It contributes
no likelihood ratio of its own, because the question it answers is categorical:
whether the recording is the output of a human vocal tract at all.

``countermeasure``
    The LFCC-GMM detector, with an explicit out-of-domain indicator and
    first-class cross-attack evaluation.
``gate``
    The operating point as an auditable policy, converting a score into an
    admissibility verdict.
"""

from viflap.analysis.spoof.countermeasure import (
    CountermeasureConfig,
    CountermeasureScore,
    SpoofingCountermeasure,
    TrainingExample,
)
from viflap.analysis.spoof.gate import GatePolicy, ValidityGate

__all__ = [
    "CountermeasureConfig",
    "CountermeasureScore",
    "GatePolicy",
    "SpoofingCountermeasure",
    "TrainingExample",
    "ValidityGate",
]
