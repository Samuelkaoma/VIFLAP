"""Forensic evaluation metrics.

``C_llr`` is the metric this system is judged by. It is the empirical
cross-entropy of the reported likelihood ratios under a flat prior, measured in
bits:

.. code-block:: text

    C_llr = 1/2 [ (1/N_ss) sum_{i in ss} log2(1 + 1/LR_i)
                + (1/N_ds) sum_{j in ds} log2(1 + LR_j) ]

Three properties earn it that position.

**It is strictly proper.** It is minimised only by reporting the likelihood
ratios the data actually supports. A system cannot improve its score by
exaggerating in the right direction, which is precisely the failure mode that
matters when the output goes to a court.

**It has an absolute reference point.** ``C_llr = 1`` is the cost of reporting
``LR = 1`` for everything — of contributing nothing. Above one, the system is
worse than useless: an investigator acting on it decides less well than one
ignoring it. No accuracy-based metric has this property; a system can be 95%
accurate and have ``C_llr`` above one.

**It decomposes.** ``C_llr_min``, obtained after optimal monotonic
recalibration, isolates discrimination. The remainder is calibration loss. The
two failures are different — one is "the system cannot tell these apart", the
other is "the system can tell them apart but is lying about how confidently" —
and they have different remedies.

Equal error rate is computed here as well, and reported, and used for nothing.
It exists so results can be placed alongside the speaker recognition literature,
which uses it as a primary metric. It requires a decision threshold and says
nothing at all about whether the reported strengths are warranted.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.calibration.elub import empirical_bounds
from viflap.analysis.calibration.pav import pav_calibrate
from viflap.domain.errors import InsufficientDataError, InvalidEvidenceError
from viflap.domain.metrics import CalibrationSummary

__all__ = [
    "compute_cllr",
    "compute_cllr_min",
    "compute_eer",
    "evaluate",
    "split_by_label",
]

_LN2 = np.log(2.0)


def split_by_label(
    values: NDArray[np.float64], labels: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Partition values into same-source and different-source."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    if values.shape != labels.shape:
        raise InvalidEvidenceError(
            "values and labels have different shapes",
            n_values=int(values.size),
            n_labels=int(labels.size),
        )
    return values[labels == 1], values[labels == 0]


def compute_cllr(log_lrs: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Log-likelihood-ratio cost of a set of reported log-LRs.

    Computed as ``log2(1 + exp(-x))`` via :func:`numpy.logaddexp2`, which is the
    stable form. Evaluating ``log2(1 + exp(-x))`` directly overflows for ``x``
    below about ``-745``, and a fused multi-stream result routinely exceeds that
    — so the naive form fails exactly on the strong results whose cost matters
    most.

    The two class averages are taken separately and then averaged with equal
    weight. This makes the metric independent of how many trials of each type
    the validation set happens to contain, which is an artefact of pair
    construction rather than a property of the system.
    """
    same, different = split_by_label(log_lrs, labels)
    if same.size == 0 or different.size == 0:
        raise InsufficientDataError(
            "C_llr is undefined without trials of both types",
            n_same_source=int(same.size),
            n_different_source=int(different.size),
        )

    # log2(1 + exp(-x)) = logaddexp2(0, -x / ln2)
    same_cost = float(np.mean(np.logaddexp2(0.0, -same / _LN2)))
    different_cost = float(np.mean(np.logaddexp2(0.0, different / _LN2)))
    return 0.5 * (same_cost + different_cost)


def compute_cllr_min(scores: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Minimum ``C_llr`` achievable by any monotonic recalibration of ``scores``.

    Pure discrimination. Invariant to any monotonic transformation of the
    scores, so a system's ``C_llr_min`` is unchanged by calibrating it well or
    badly — which is what makes the decomposition meaningful.
    """
    calibrated = pav_calibrate(scores, labels)
    return compute_cllr(calibrated, labels)


def compute_eer(scores: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Equal error rate, by exact interpolation of the crossing point.

    Sweeps every distinct score as a threshold, computes the false-acceptance
    and false-rejection rates, and interpolates linearly where they cross. The
    common alternative — taking the threshold minimising ``|FAR - FRR|`` and
    reporting their average — is biased on discrete score sets, because the
    crossing usually falls between two attainable thresholds and neither
    endpoint is the crossing.

    The error rates are obtained by binary search into the sorted score arrays
    rather than by counting at each threshold. The counting form is ``O(n^2)``:
    on a validation set of a few hundred thousand trials it takes minutes, and
    it is called inside every bootstrap replicate, so the quadratic term is
    multiplied by a thousand. This form is ``O(n log n)``.

    Reported only for comparability with the speaker recognition literature. No
    decision in this system uses a threshold, and none uses this number.
    """
    same, different = split_by_label(scores, labels)
    if same.size == 0 or different.size == 0:
        raise InsufficientDataError(
            "EER is undefined without trials of both types",
            n_same_source=int(same.size),
            n_different_source=int(different.size),
        )

    sorted_same = np.sort(same)
    sorted_different = np.sort(different)

    thresholds = np.unique(np.concatenate([same, different]))
    # Extend below the minimum so the sweep starts from FAR = 1, FRR = 0.
    thresholds = np.concatenate([[thresholds[0] - 1.0], thresholds])

    # #{different >= t} and #{same < t}, both by insertion point.
    false_acceptance = (
        sorted_different.size - np.searchsorted(sorted_different, thresholds, side="left")
    ) / sorted_different.size
    false_rejection = (
        np.searchsorted(sorted_same, thresholds, side="left") / sorted_same.size
    )

    difference = false_acceptance - false_rejection
    crossings = np.flatnonzero(np.diff(np.sign(difference)) != 0)
    if crossings.size == 0:
        index = int(np.argmin(np.abs(difference)))
        return float(0.5 * (false_acceptance[index] + false_rejection[index]))

    index = int(crossings[0])
    before, after = difference[index], difference[index + 1]
    fraction = 0.0 if before == after else before / (before - after)

    far = false_acceptance[index] + fraction * (
        false_acceptance[index + 1] - false_acceptance[index]
    )
    frr = false_rejection[index] + fraction * (
        false_rejection[index + 1] - false_rejection[index]
    )
    return float(0.5 * (far + frr))


def evaluate(
    log_lrs: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    include_bounds: bool = True,
) -> CalibrationSummary:
    """Full evaluation of a set of reported log-likelihood ratios.

    ``C_llr`` is computed on the values as reported; ``C_llr_min`` on the same
    values treated as bare scores. That asymmetry is the point of the
    decomposition: the first asks whether the numbers are right, the second asks
    whether their ordering is.
    """
    log_lrs = np.asarray(log_lrs, dtype=np.float64)
    labels = np.asarray(labels)

    c_llr = compute_cllr(log_lrs, labels)
    c_llr_min = compute_cllr_min(log_lrs, labels)
    eer = compute_eer(log_lrs, labels)

    lower = upper = None
    if include_bounds:
        bounds = empirical_bounds(log_lrs, labels)
        lower = bounds.lower_log10
        upper = bounds.upper_log10

    return CalibrationSummary(
        c_llr=c_llr,
        # Floating-point noise can put C_llr_min a few ulps above C_llr when a
        # system is already optimally calibrated. The domain object treats that
        # ordering as an invariant violation, and rightly — so it is corrected
        # here, where the cause is known, rather than by loosening the invariant.
        c_llr_min=min(c_llr_min, c_llr),
        n_same_source=int(np.count_nonzero(labels == 1)),
        n_different_source=int(np.count_nonzero(labels == 0)),
        eer=eer,
        elub_lower_log10=lower,
        elub_upper_log10=upper,
    )
