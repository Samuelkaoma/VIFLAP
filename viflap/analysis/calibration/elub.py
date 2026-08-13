"""Empirical bounds on reportable likelihood ratios.

The problem
-----------
A calibration model will happily emit a likelihood ratio of ``10^12`` from a
validation set of five thousand trials. That number asserts the evidence is a
trillion times more probable under one proposition than the other. The
validation set contains no observation remotely capable of supporting such a
claim — it is an extrapolation of the fitted model far beyond the range of the
data, and its magnitude is a property of the functional form chosen, not of the
evidence.

This is not a hypothetical failure. It is the ordinary behaviour of every
parametric calibration method at the tails, and it is where the numbers that end
up in court come from.

The bound
---------
Vergeer and colleagues (2016) propose an empirical bound derived by asking what
the strongest defensible likelihood ratio is given the data actually observed.
The construction implemented here is their "devil's advocate": augment the
validation set with one hypothetical counterexample at each extreme — a
different-source trial scoring above everything observed, and a same-source
trial scoring below everything — then recompute the optimal calibration.

The effect is that the strongest attainable likelihood ratio becomes finite and
governed by the size of the validation set. Intuitively: with ``N`` same-source
trials and no different-source trial ever scoring that high, the strongest claim
the data supports is of order ``N`` to one, because one more trial could have
been the counterexample. Claiming ``10^12`` from five thousand trials is
claiming to have excluded a possibility that was never tested.

Applying the bound
------------------
Bounding is clipping, and clipping is honest here in a way it usually is not: it
does not discard information, because the information was never present. The
bounded value is what the validation data supports; the unbounded value is what
the model's algebra produced. Both are reported, so the size of the
extrapolation is visible rather than silently removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.calibration.pav import pool_adjacent_violators
from viflap.domain.errors import InsufficientDataError

__all__ = ["EmpiricalBounds", "apply_bounds", "empirical_bounds"]

_LN10 = math.log(10.0)


@dataclass(frozen=True, slots=True)
class EmpiricalBounds:
    """The strongest likelihood ratios a validation set can support."""

    lower_log_lr: float
    upper_log_lr: float
    n_same_source: int
    n_different_source: int

    @property
    def lower_log10(self) -> float:
        return self.lower_log_lr / _LN10

    @property
    def upper_log10(self) -> float:
        return self.upper_log_lr / _LN10

    def describe(self) -> str:
        return (
            f"Validation of {self.n_same_source:,} same-source and "
            f"{self.n_different_source:,} different-source trials supports "
            f"likelihood ratios between 10^{self.lower_log10:.2f} and "
            f"10^{self.upper_log10:.2f}. Values beyond this range are "
            f"extrapolations of the calibration model rather than findings "
            f"supported by the validation data."
        )

    def clips(self, log_lr: float) -> bool:
        """Whether a value would be bounded, and therefore is an extrapolation."""
        return not (self.lower_log_lr <= log_lr <= self.upper_log_lr)


def empirical_bounds(
    scores: NDArray[np.float64], labels: NDArray[np.int64]
) -> EmpiricalBounds:
    """Compute the empirical lower and upper bound for a validation set.

    Implementation of the devil's-advocate construction: one different-source
    trial is added above every observed score and one same-source trial below,
    and the optimal monotonic calibration is refitted. The extreme blocks of the
    refitted calibration then contain a counterexample by construction, so their
    likelihood ratios are finite, and those finite values are the bounds.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)

    n_same = int(np.count_nonzero(labels == 1))
    n_different = int(np.count_nonzero(labels == 0))
    if n_same == 0 or n_different == 0:
        raise InsufficientDataError(
            "empirical bounds require trials of both types",
            n_same_source=n_same,
            n_different_source=n_different,
        )

    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise InsufficientDataError("no finite scores to bound")
    span = float(np.max(finite) - np.min(finite))
    margin = max(span, 1.0)

    augmented_scores = np.concatenate(
        [scores, [float(np.max(finite)) + margin], [float(np.min(finite)) - margin]]
    )
    # The added trials are deliberately of the "wrong" class for their position:
    # a different-source trial scoring higher than anything observed, and a
    # same-source trial scoring lower. Each is the counterexample that the
    # validation set never happened to contain.
    augmented_labels = np.concatenate([labels, [0], [1]])

    result = pool_adjacent_violators(augmented_scores, augmented_labels)

    prior_log_odds = math.log(n_same + 1) - math.log(n_different + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        posterior_log_odds = np.log(result.block_values) - np.log1p(-result.block_values)
    block_log_lrs = posterior_log_odds - prior_log_odds

    usable = block_log_lrs[np.isfinite(block_log_lrs)]
    if usable.size == 0:
        raise InsufficientDataError(
            "the augmented calibration produced no finite blocks; the "
            "validation set does not support any likelihood ratio"
        )

    return EmpiricalBounds(
        lower_log_lr=float(np.min(usable)),
        upper_log_lr=float(np.max(usable)),
        n_same_source=n_same,
        n_different_source=n_different,
    )


def apply_bounds(
    log_lrs: NDArray[np.float64], bounds: EmpiricalBounds
) -> NDArray[np.float64]:
    """Clip log-likelihood ratios to the empirically supported range."""
    return np.clip(
        np.asarray(log_lrs, dtype=np.float64), bounds.lower_log_lr, bounds.upper_log_lr
    )
