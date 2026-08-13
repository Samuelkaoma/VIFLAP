"""Statistical evidence models for the non-acoustic streams.

``conjugate``
    Marginal likelihood ratios under conjugate models — the foundation. Asks the
    question directly (one actor, or two?) and integrates out the unknown
    parameters, so rarity is handled by the model rather than by a chosen
    weight.
``streams``
    Temporal, transactional and device comparators built on that foundation.

The design point these share: a shared cash-out agent is evidence in proportion
to how unusual that agent is, and a set-overlap index cannot express that
distinction at all.
"""

from viflap.analysis.patterns.conjugate import (
    BackgroundPopulation,
    DirichletMultinomialComparator,
    NormalInverseGammaComparator,
    counts_to_vector,
)
from viflap.analysis.patterns.streams import (
    CallRecord,
    DeviceComparator,
    DeviceObservation,
    TemporalComparator,
    TemporalProfile,
    Transaction,
    TransactionalComparator,
    TransactionalProfile,
    circular_background,
)

__all__ = [
    "BackgroundPopulation",
    "CallRecord",
    "DeviceComparator",
    "DeviceObservation",
    "DirichletMultinomialComparator",
    "NormalInverseGammaComparator",
    "TemporalComparator",
    "TemporalProfile",
    "Transaction",
    "TransactionalComparator",
    "TransactionalProfile",
    "circular_background",
    "counts_to_vector",
]
