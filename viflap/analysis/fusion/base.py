"""Shared contracts for evidence fusion.

Fusion is where this system's central claim lives: that several weakly
individuating traces combine into an actionable one. It is also where the
easiest and most dangerous error lives, and the two facts are connected.

The error
---------
Summing log-likelihood ratios is correct if and only if the streams are
conditionally independent given each proposition. In this application they are
not, and the reason is structural rather than incidental: one person running one
scripted operation is the common cause of the acoustic, lexical, temporal and
transactional signatures alike. They co-vary because they share a cause, and
that is exactly the situation independence assumes away.

The direction of the resulting error is what makes it a safety issue.
Correlated evidence treated as independent counts the same underlying fact
several times, so the combined likelihood ratio is **overstated** — and
overstatement runs against the person the comparison concerns.

The response
------------
Every fuser here declares whether it models dependence, and the engine records
which was used on every result. The naive summation remains implemented, and is
prohibited from producing a reported multi-stream result: it exists so that the
cost of the assumption can be measured on real data rather than asserted. That
measurement — how many orders of magnitude independence would have added — is
reported alongside every comparison rather than computed once in an appendix.

Missing streams
---------------
A stream that produced nothing is not a stream reporting ``LR = 1``. Substituting
a neutral value asserts that the stream was computed and found the evidence
equally probable under both propositions, which is a fabricated observation.
Every fuser therefore takes a *pattern* of present streams and must either
marginalise over the absent ones analytically or hold a model fitted to that
pattern. This is what :attr:`FusionModel.supports_pattern` exists to answer, and
the engine will refuse rather than substitute.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from viflap.domain.errors import InvalidEvidenceError
from viflap.domain.evidence import EvidenceStream
from viflap.domain.linkage import FusionMethod

__all__ = [
    "FusionModel",
    "FusionObservation",
    "FusionTrainingSet",
    "StreamVector",
    "to_matrix",
]

StreamVector = Mapping[EvidenceStream, float]
"""Per-stream calibrated log-LRs for one comparison. Streams that produced
nothing are absent from the mapping — never present with a value of zero."""


@dataclass(frozen=True, slots=True)
class FusionObservation:
    """One training comparison: per-stream log-LRs and the known truth."""

    log_lrs: StreamVector
    is_same_source: bool
    group_id: str = ""
    """Identifier used to keep related comparisons together when splitting or
    resampling — normally the speaker or operation. Comparisons sharing an actor
    are not independent, and treating them as independent understates variance
    in exactly the way that produces confident wrong answers."""

    @property
    def pattern(self) -> frozenset[EvidenceStream]:
        return frozenset(self.log_lrs)


@dataclass(frozen=True, slots=True)
class FusionTrainingSet:
    """Training comparisons, indexed by which streams were present."""

    observations: Sequence[FusionObservation]

    def __post_init__(self) -> None:
        if not self.observations:
            raise InvalidEvidenceError("a fusion training set cannot be empty")

    @property
    def patterns(self) -> dict[frozenset[EvidenceStream], int]:
        """Count of comparisons per missingness pattern."""
        counts: dict[frozenset[EvidenceStream], int] = {}
        for observation in self.observations:
            counts[observation.pattern] = counts.get(observation.pattern, 0) + 1
        return counts

    @property
    def streams(self) -> tuple[EvidenceStream, ...]:
        """Every stream appearing anywhere, in canonical order.

        Canonical order — not order of first appearance — because it fixes the
        column layout of every design matrix and covariance estimate. A model
        trained today and loaded next year must interpret its own weights the
        same way, and ordering by discovery makes that depend on which
        comparison happened to be processed first.
        """
        present = {
            stream for observation in self.observations for stream in observation.log_lrs
        }
        return tuple(stream for stream in EvidenceStream.ordered() if stream in present)

    def labels(self) -> NDArray[np.int64]:
        return np.array(
            [1 if observation.is_same_source else 0 for observation in self.observations],
            dtype=np.int64,
        )

    def groups(self) -> NDArray[np.str_]:
        return np.array([observation.group_id for observation in self.observations])

    def counts(self) -> tuple[int, int]:
        labels = self.labels()
        return int(np.count_nonzero(labels == 1)), int(np.count_nonzero(labels == 0))

    def subset(self, pattern: frozenset[EvidenceStream]) -> FusionTrainingSet:
        """Comparisons whose present-stream pattern is exactly ``pattern``."""
        selected = [
            observation
            for observation in self.observations
            if observation.pattern == pattern
        ]
        if not selected:
            raise InvalidEvidenceError(
                "no training comparisons have this missingness pattern",
                pattern=sorted(stream.value for stream in pattern),
            )
        return FusionTrainingSet(selected)


def to_matrix(
    observations: Sequence[FusionObservation], streams: Sequence[EvidenceStream]
) -> NDArray[np.float64]:
    """Design matrix over ``streams``, with ``NaN`` where a stream was absent.

    ``NaN`` rather than zero, deliberately. Zero is a valid log-likelihood ratio
    meaning "uninformative", so using it as a missing marker makes the two
    indistinguishable downstream, and every subsequent operation silently treats
    absent evidence as neutral evidence. ``NaN`` propagates loudly instead:
    anything that fails to handle missingness produces ``NaN``, which is
    impossible to overlook, rather than a plausible number.
    """
    matrix = np.full((len(observations), len(streams)), np.nan, dtype=np.float64)
    for row, observation in enumerate(observations):
        for column, stream in enumerate(streams):
            value = observation.log_lrs.get(stream)
            if value is not None:
                matrix[row, column] = value
    return matrix


@runtime_checkable
class FusionModel(Protocol):
    """A fitted model combining per-stream log-LRs into one."""

    @property
    def method(self) -> FusionMethod: ...

    @property
    def is_fitted(self) -> bool: ...

    @property
    def model_id(self) -> str:
        """Identity of this fitted model, recorded on every result it produces."""
        ...

    def supports_pattern(self, pattern: frozenset[EvidenceStream]) -> bool:
        """Whether this model can fuse a comparison with exactly these streams.

        A model that cannot must say so, so the engine can decline. The
        alternative — imputing the missing streams and returning a number — is
        the failure this whole layer is arranged to prevent.
        """
        ...

    def fit(self, training: FusionTrainingSet) -> FusionModel: ...

    def fuse(self, log_lrs: StreamVector) -> float:
        """Fused log-likelihood ratio for one comparison."""
        ...
