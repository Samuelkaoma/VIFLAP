"""Pool-adjacent-violators: optimal monotonic calibration.

PAV solves isotonic regression — the best fit to the data subject to being
non-decreasing — and it does so exactly, in one pass, in linear time. It has two
distinct roles here and they should not be confused.

**As a metric.** Applying PAV to a system's scores gives the calibration that
minimises ``C_llr`` over all monotonic transformations. The resulting cost is
``C_llr_min``, and the gap between it and the system's actual ``C_llr`` is
precisely the cost of the system misrepresenting its own strength. This use is
diagnostic; the transform is discarded.

**As a calibrator.** The same transform can be fitted on development data and
applied to new scores. This is non-parametric calibration: it assumes only
monotonicity, where logistic calibration additionally assumes the relationship
is affine in the log-odds. The trade-off is real in both directions —
non-parametric calibration follows genuine structure that a line cannot, and it
also follows noise that a line would smooth away, so it needs more development
data to be worth using.

Why the implementation matters
------------------------------
The naive version rescans from the start after every merge, which is
``O(n^2)`` — on a validation set of a hundred thousand trials, minutes rather
than milliseconds, and this runs inside every bootstrap replicate. The stack
formulation below is ``O(n)``: each point is pushed once, and each merge removes
a block permanently, so the total merge count is bounded by the number of
points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from viflap.domain.errors import InsufficientDataError, InvalidEvidenceError

__all__ = ["PavResult", "pav_calibrate", "pool_adjacent_violators"]


@dataclass(frozen=True, slots=True)
class PavResult:
    """The isotonic fit, as blocks and as per-trial values."""

    fitted: NDArray[np.float64]
    """Fitted value for each input, in the original input order."""

    block_boundaries: NDArray[np.int64]
    """Index in score-sorted order at which each block starts."""

    block_values: NDArray[np.float64]
    """The pooled proportion within each block."""

    block_weights: NDArray[np.float64]

    @property
    def n_blocks(self) -> int:
        return self.block_values.shape[0]

    @property
    def resolution(self) -> float:
        """Blocks per trial, in ``(0, 1]``.

        A diagnostic of how much structure the scores actually contain. A value
        near one means almost every trial sits in its own block — the scores
        separate the classes finely. A value near zero means the scores collapse
        into a handful of distinguishable levels, so the system has few
        distinct strengths of evidence to offer regardless of the numbers it
        prints.
        """
        return self.n_blocks / max(self.fitted.shape[0], 1)


def pool_adjacent_violators(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    weights: NDArray[np.float64] | None = None,
) -> PavResult:
    """Fit an isotonic regression of ``labels`` on ``scores``.

    Parameters
    ----------
    scores:
        System scores. Only their order matters — PAV is invariant to any
        monotonic transformation of them, which is exactly why ``C_llr_min``
        measures discrimination alone and is unaffected by calibration.
    labels:
        1 for same-source, 0 for different-source.
    weights:
        Optional per-trial weights, for reweighting an unbalanced trial set to a
        target prior.

    Notes
    -----
    Ties in score are handled by sorting stably and letting the merge step pool
    them. Tied scores that receive different fitted values would violate
    monotonicity in any sensible reading, and the pooling is what prevents it.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.shape != labels.shape:
        raise InvalidEvidenceError(
            "scores and labels have different shapes",
            n_scores=int(scores.size),
            n_labels=int(labels.size),
        )
    if scores.size == 0:
        raise InsufficientDataError("cannot fit an isotonic regression to no data")

    if weights is None:
        weights = np.ones_like(scores, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != scores.shape:
            raise InvalidEvidenceError(
                "the number of weights differs from the number of trials"
            )
        if np.any(weights <= 0.0):
            raise InvalidEvidenceError("weights must be strictly positive")

    order = np.argsort(scores, kind="stable")
    sorted_labels = labels[order].astype(np.float64)
    sorted_weights = weights[order]

    fitted_sorted = _isotonic_fit(sorted_labels, sorted_weights)

    # Recover the block structure from the fitted values. Consecutive equal
    # values are one pooled block by construction, so the boundaries are simply
    # the positions where the fitted value changes. Deriving them this way keeps
    # the block bookkeeping and the fit from being two things that could
    # disagree.
    block_boundaries = np.concatenate(
        [[0], np.flatnonzero(np.diff(fitted_sorted) != 0.0) + 1]
    ).astype(np.int64)
    block_values = fitted_sorted[block_boundaries]

    block_edges = np.append(block_boundaries, fitted_sorted.size)
    block_weights = np.add.reduceat(sorted_weights, block_boundaries)
    del block_edges

    fitted = np.empty_like(fitted_sorted)
    fitted[order] = fitted_sorted

    return PavResult(
        fitted=fitted,
        block_boundaries=block_boundaries,
        block_values=block_values,
        block_weights=block_weights,
    )


def _isotonic_fit(
    values: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Weighted isotonic regression of ``values``, already in ascending score order.

    Delegates to SciPy's compiled implementation where available, falling back to
    the stack formulation below. The two agree exactly — that equivalence is
    asserted in the test suite against randomly generated problems — so the
    choice is purely one of speed, and speed matters here because this runs
    inside every bootstrap replicate of every evaluation.
    """
    try:
        from scipy.optimize import isotonic_regression
    except ImportError:  # pragma: no cover - very old SciPy
        return _isotonic_fit_stack(values, weights)
    return isotonic_regression(values, weights=weights, increasing=True).x


def _isotonic_fit_stack(
    values: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Reference implementation: the pool-adjacent-violators stack.

    Each point is pushed once and each merge permanently removes a block, so the
    total work is linear in the number of points. Blocks carry their weighted
    *sum* rather than their mean, which keeps every merge exact instead of
    accumulating rounding error through repeated averaging.
    """
    sums: list[float] = []
    totals: list[float] = []
    lengths: list[int] = []

    for index in range(values.size):
        sums.append(float(values[index] * weights[index]))
        totals.append(float(weights[index]))
        lengths.append(1)

        while len(sums) > 1 and (sums[-2] / totals[-2]) > (sums[-1] / totals[-1]):
            merged_sum = sums.pop() + sums[-1]
            merged_total = totals.pop() + totals[-1]
            merged_length = lengths.pop() + lengths[-1]
            sums[-1] = merged_sum
            totals[-1] = merged_total
            lengths[-1] = merged_length

    fitted = np.empty(values.size, dtype=np.float64)
    position = 0
    for block_sum, block_total, block_length in zip(sums, totals, lengths, strict=True):
        fitted[position : position + block_length] = block_sum / block_total
        position += block_length
    return fitted


def pav_calibrate(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Convert scores to optimally calibrated log-likelihood ratios.

    PAV gives posterior probabilities under the *empirical prior of the trial
    set*, which is an artefact of how the validation trials were constructed —
    typically many more different-source pairs than same-source ones, because
    that is how pairs combine. Dividing out that prior converts the posterior
    odds into a likelihood ratio, which is a property of the evidence rather
    than of the experiment:

    .. code-block:: text

        LR = [ p / (1 - p) ] / [ N_ss / N_ds ]

    Omitting the correction is a common and consequential error: it produces
    "likelihood ratios" that shift systematically with the trial-set
    composition, so the same system evaluated on two differently balanced sets
    appears to have different evidential strength.

    Infinities are returned rather than clipped. A block containing only
    same-source trials genuinely has infinite empirical odds — the data contains
    no counterexample. That is a statement about the *validation set being too
    small*, not about the evidence being infinitely strong, and the honest
    response is to bound the reportable range separately with an empirical
    bound (see :mod:`viflap.analysis.calibration.elub`) rather than to pick an
    arbitrary cap here and let it look like a measurement.
    """
    result = pool_adjacent_violators(scores, labels, weights)
    labels = np.asarray(labels)

    n_same = float(np.count_nonzero(labels == 1))
    n_different = float(np.count_nonzero(labels == 0))
    if n_same == 0.0 or n_different == 0.0:
        raise InsufficientDataError(
            "calibration requires trials of both types",
            n_same_source=int(n_same),
            n_different_source=int(n_different),
        )

    posterior = result.fitted
    with np.errstate(divide="ignore", invalid="ignore"):
        posterior_log_odds = np.log(posterior) - np.log1p(-posterior)
    prior_log_odds = np.log(n_same) - np.log(n_different)
    return posterior_log_odds - prior_log_odds
