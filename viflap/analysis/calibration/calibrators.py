"""Score-to-likelihood-ratio calibration.

Every evidence stream produces a score whose scale is an accident of its
construction: a PLDA log-ratio, a divergence between two distributions, a
count-based similarity. None of them is a likelihood ratio, and reporting one as
though it were is the specific error the forensic literature exists to warn
against. Calibration is the mapping from a score to the likelihood ratio the
data warrants, fitted on held-out trials with known ground truth.

Three calibrators, with a clear default
---------------------------------------
:class:`LogisticCalibrator`
    An affine map in the log-odds domain, with parameters fitted to minimise
    ``C_llr`` directly. This is the standard method (Brümmer's FoCal), it is the
    default, and it should be preferred unless there is a reason not to. Two
    parameters cannot overfit, the objective is convex so the optimum is unique
    and reproducible, and it extrapolates smoothly.

:class:`IsotonicCalibrator`
    Non-parametric, assuming only monotonicity. Follows genuine curvature that
    an affine map cannot — and follows noise that an affine map would smooth
    away. Needs substantially more development data. Its extrapolation beyond
    the observed score range is a constant, which is the honest behaviour but
    means it cannot distinguish "very strong" from "extremely strong" outside
    that range.

:class:`KernelDensityCalibrator`
    Models the same-source and different-source score densities separately and
    takes their ratio. The approach used in much forensic practice, and the
    right choice when the two distributions have genuinely different shapes
    rather than differing by a location shift — which is common for the
    non-acoustic streams, where a "score" is often a count-based quantity with a
    spike at zero.

All three implement :class:`Calibrator`, and all three return a
:class:`~viflap.domain.values.LogLikelihoodRatio` — the point at which a number
becomes evidence. Below this layer nothing does.

Empirical bounding is applied by default
----------------------------------------
Every calibrator records the empirical bounds of its training data and clips its
output to them. A likelihood ratio beyond what the validation set can support is
an extrapolation of the fitted curve, not a finding, and the unbounded value is
retained alongside so the size of the extrapolation is visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from viflap.analysis.calibration.elub import EmpiricalBounds, empirical_bounds
from viflap.analysis.calibration.pav import pool_adjacent_violators
from viflap.domain.errors import (
    CalibrationError,
    InsufficientDataError,
    ModelNotTrainedError,
)
from viflap.domain.values import LogLikelihoodRatio

__all__ = [
    "CalibratedOutput",
    "Calibrator",
    "IsotonicCalibrator",
    "KernelDensityCalibrator",
    "LogisticCalibrator",
]

_LN2 = float(np.log(2.0))
_LN10 = float(np.log(10.0))

#: Below this many trials of either class, a fitted calibration describes the
#: development set rather than the system. Chosen as the point at which the
#: empirical bound itself becomes the binding constraint on any reportable
#: likelihood ratio.
MIN_TRIALS_PER_CLASS = 30


@dataclass(frozen=True, slots=True)
class CalibratedOutput:
    """A calibrated likelihood ratio, with what it took to produce it."""

    log_lr: LogLikelihoodRatio
    unbounded_log_lr: float
    """Before empirical bounding. Equal to the reported value unless the
    calibration extrapolated beyond what the development data supports."""

    was_bounded: bool
    calibrator_id: str

    @property
    def extrapolation_log10(self) -> float:
        """Orders of magnitude removed by bounding.

        Zero when the value was supported by the data. A large value means the
        calibration model was asked for a strength the validation set has no
        basis to confirm, which is worth surfacing rather than silently fixing.
        """
        return abs(self.unbounded_log_lr - self.log_lr.value) / _LN10


@runtime_checkable
class Calibrator(Protocol):
    """Maps a raw score to a calibrated likelihood ratio."""

    @property
    def is_fitted(self) -> bool: ...

    @property
    def calibrator_id(self) -> str: ...

    def fit(self, scores: NDArray[np.float64], labels: NDArray[np.int64]) -> Calibrator: ...

    def transform(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calibrated log-LRs for an array of scores, before bounding."""
        ...

    def calibrate(self, score: float) -> CalibratedOutput:
        """Calibrate a single score, with bounding applied."""
        ...


class _BaseCalibrator:
    """Shared fitting protocol, validation and bounding."""

    def __init__(self) -> None:
        self._bounds: EmpiricalBounds | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def bounds(self) -> EmpiricalBounds:
        if self._bounds is None:
            raise ModelNotTrainedError("calibrator has not been fitted")
        return self._bounds

    @property
    def calibrator_id(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _validate(
        self, scores: NDArray[np.float64], labels: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels).ravel()
        if scores.shape != labels.shape:
            raise CalibrationError(
                "scores and labels have different lengths",
                n_scores=int(scores.size),
                n_labels=int(labels.size),
            )
        if not np.all(np.isfinite(scores)):
            raise CalibrationError(
                "scores contain non-finite values; a calibration fitted through "
                "them would be determined by whichever infinity dominates"
            )

        n_same = int(np.count_nonzero(labels == 1))
        n_different = int(np.count_nonzero(labels == 0))
        if n_same < MIN_TRIALS_PER_CLASS or n_different < MIN_TRIALS_PER_CLASS:
            raise InsufficientDataError(
                "too few development trials to fit a calibration that describes "
                "the system rather than the development set",
                n_same_source=n_same,
                n_different_source=n_different,
                required_per_class=MIN_TRIALS_PER_CLASS,
            )
        return scores, labels

    def _finish_fit(self, scores: NDArray[np.float64], labels: NDArray[np.int64]) -> None:
        """Record empirical bounds from the *calibrated* development scores."""
        self._fitted = True
        self._bounds = empirical_bounds(self.transform(scores), labels)  # type: ignore[attr-defined]

    def calibrate(self, score: float) -> CalibratedOutput:
        if not self._fitted:
            raise ModelNotTrainedError(
                "calibrator has not been fitted; an uncalibrated score reported "
                "as a likelihood ratio carries a strength it does not have"
            )
        unbounded = float(self.transform(np.array([score]))[0])  # type: ignore[attr-defined]
        bounds = self.bounds
        bounded = float(np.clip(unbounded, bounds.lower_log_lr, bounds.upper_log_lr))
        return CalibratedOutput(
            log_lr=LogLikelihoodRatio(bounded),
            unbounded_log_lr=unbounded,
            was_bounded=bounded != unbounded,
            calibrator_id=self.calibrator_id,
        )


class LogisticCalibrator(_BaseCalibrator):
    """Affine calibration fitted by direct minimisation of ``C_llr``.

    The map is ``log LR = a * score + b``. Fitting minimises

    .. code-block:: text

        C_llr(a, b) = 1/2 [ mean_ss log2(1 + exp(-(a s + b)))
                          + mean_ds log2(1 + exp( (a s + b))) ]

    which is convex in ``(a, b)``: ``log(1 + exp(-x))`` is convex and its
    argument is affine in the parameters. So the optimum is unique, and
    refitting on the same data gives the same answer — a property that matters
    when a result may have to be reproduced years later.

    The gradient is supplied analytically rather than estimated by finite
    differences. This is not only faster: near the optimum the objective is flat,
    and a finite-difference gradient there is dominated by its own truncation
    error, so the optimiser stops in a slightly different place on each run and
    the calibration becomes irreproducible at the third decimal.

    ``a`` is constrained to be non-negative. A negative slope would mean the
    calibration had inverted the score's direction — concluding that higher
    similarity supports *different* sources. That can genuinely fit better on a
    development set where the underlying system is uninformative, and it is
    always a symptom to investigate rather than a calibration to deploy.
    """

    def __init__(self) -> None:
        super().__init__()
        self._slope = 1.0
        self._intercept = 0.0

    @property
    def calibrator_id(self) -> str:
        return f"logistic(a={self._slope:.6g},b={self._intercept:.6g})"

    @property
    def parameters(self) -> tuple[float, float]:
        return (self._slope, self._intercept)

    def fit(
        self, scores: NDArray[np.float64], labels: NDArray[np.int64]
    ) -> LogisticCalibrator:
        scores, labels = self._validate(scores, labels)
        same = scores[labels == 1]
        different = scores[labels == 0]

        # Standardise before optimising, then transform the solution back. The
        # raw scales differ wildly between streams — a PLDA log-ratio spans
        # hundreds, a divergence spans a fraction of one — and an optimiser
        # given a badly scaled problem converges to different places depending
        # on the units, which is not a property a calibration should have.
        centre = float(np.mean(scores))
        scale = float(np.std(scores))
        if scale <= 0.0:
            raise CalibrationError(
                "development scores have zero variance; the system produced the "
                "same score for every trial and there is nothing to calibrate"
            )
        same_z = (same - centre) / scale
        different_z = (different - centre) / scale

        result = minimize(
            _cllr_objective_and_gradient,
            x0=np.array([1.0, 0.0]),
            args=(same_z, different_z),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, None), (None, None)],
        )
        if not result.success and result.status != 1:
            raise CalibrationError(
                "logistic calibration failed to converge", message=str(result.message)
            )

        slope_z, intercept_z = float(result.x[0]), float(result.x[1])
        self._slope = slope_z / scale
        self._intercept = intercept_z - slope_z * centre / scale

        self._finish_fit(scores, labels)
        return self

    def transform(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._slope * np.asarray(scores, dtype=np.float64) + self._intercept


def _cllr_objective_and_gradient(
    parameters: NDArray[np.float64],
    same: NDArray[np.float64],
    different: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """``C_llr`` and its analytic gradient with respect to ``(slope, intercept)``.

    Using ``d/dx log2(1 + exp(-x)) = -sigmoid(-x) / ln2`` and
    ``d/dx log2(1 + exp(x)) = sigmoid(x) / ln2``, with the logistic evaluated in
    its stable branching form so neither tail overflows.
    """
    slope, intercept = parameters
    same_llr = slope * same + intercept
    different_llr = slope * different + intercept

    same_cost = np.logaddexp2(0.0, -same_llr / _LN2)
    different_cost = np.logaddexp2(0.0, different_llr / _LN2)
    objective = 0.5 * (float(np.mean(same_cost)) + float(np.mean(different_cost)))

    same_weight = -_stable_logistic(-same_llr) / _LN2
    different_weight = _stable_logistic(different_llr) / _LN2

    gradient_slope = 0.5 * (
        float(np.mean(same_weight * same)) + float(np.mean(different_weight * different))
    )
    gradient_intercept = 0.5 * (
        float(np.mean(same_weight)) + float(np.mean(different_weight))
    )
    return objective, np.array([gradient_slope, gradient_intercept])


def _stable_logistic(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Elementwise logistic, branching so ``exp`` never sees a positive argument."""
    out = np.empty_like(x)
    positive = x >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_negative = np.exp(x[~positive])
    out[~positive] = exp_negative / (1.0 + exp_negative)
    return out


class IsotonicCalibrator(_BaseCalibrator):
    """Non-parametric calibration by pool-adjacent-violators.

    Assumes only that a higher score means stronger support. Where the true
    score-to-LR relationship is genuinely curved, this follows it; where the
    development set is small, it follows the noise instead, which is why the
    minimum trial count is enforced more strictly here in practice.

    Outside the range of development scores the calibration is constant at the
    nearest fitted value. That is the honest behaviour — the data says nothing
    about scores it never saw — but it means this calibrator cannot rank two
    results that both lie beyond the development range, and for a system whose
    strongest results matter most, that is a real limitation rather than a
    technicality.
    """

    def __init__(self) -> None:
        super().__init__()
        self._knot_scores: NDArray[np.float64] | None = None
        self._knot_log_lrs: NDArray[np.float64] | None = None

    @property
    def calibrator_id(self) -> str:
        knots = 0 if self._knot_scores is None else self._knot_scores.size
        return f"isotonic(knots={knots})"

    def fit(
        self, scores: NDArray[np.float64], labels: NDArray[np.int64]
    ) -> IsotonicCalibrator:
        scores, labels = self._validate(scores, labels)
        result = pool_adjacent_violators(scores, labels)

        n_same = float(np.count_nonzero(labels == 1))
        n_different = float(np.count_nonzero(labels == 0))
        prior_log_odds = float(np.log(n_same) - np.log(n_different))

        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        sorted_fitted = result.fitted[order]

        # One knot per block, placed at the block's highest score. Interpolating
        # between block *values* rather than stepping between them keeps the
        # calibration continuous, which matters because a step discontinuity
        # means two nearly identical recordings can receive materially different
        # likelihood ratios.
        boundaries = np.append(result.block_boundaries, sorted_scores.size)
        knot_scores = []
        knot_posteriors = []
        for index in range(result.n_blocks):
            start, end = int(boundaries[index]), int(boundaries[index + 1])
            knot_scores.append(float(sorted_scores[end - 1]))
            knot_posteriors.append(float(sorted_fitted[start]))

        posteriors = np.clip(
            np.array(knot_posteriors, dtype=np.float64),
            1.0 / (2.0 * (n_same + n_different)),
            1.0 - 1.0 / (2.0 * (n_same + n_different)),
        )
        self._knot_scores = np.array(knot_scores, dtype=np.float64)
        self._knot_log_lrs = np.log(posteriors) - np.log1p(-posteriors) - prior_log_odds

        self._finish_fit(scores, labels)
        return self

    def transform(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._knot_scores is None or self._knot_log_lrs is None:
            raise ModelNotTrainedError("isotonic calibrator has not been fitted")
        return np.interp(
            np.asarray(scores, dtype=np.float64),
            self._knot_scores,
            self._knot_log_lrs,
            left=float(self._knot_log_lrs[0]),
            right=float(self._knot_log_lrs[-1]),
        )


class KernelDensityCalibrator(_BaseCalibrator):
    """Calibration by the ratio of estimated class-conditional score densities.

    Estimates ``p(score | H_ss)`` and ``p(score | H_ds)`` separately by Gaussian
    kernel density estimation and reports their log ratio. This is the
    construction closest to the definition of a likelihood ratio, and it is the
    approach used in a good deal of forensic practice.

    Its advantage over an affine map is that it does not require the two
    distributions to be related by a location shift. For several of the
    non-acoustic streams they are not: a transactional similarity score has a
    spike at zero for pairs sharing no counterparty and a broad distribution
    above it, and no affine calibration of that has the right shape anywhere.

    Its weakness is the tails. A kernel density estimate beyond the range of the
    data decays at a rate set by the bandwidth rather than by evidence, and
    ratios of two such extrapolations are unstable. Empirical bounding, applied
    to all calibrators here, is what keeps that from reaching a report.

    Bandwidth is chosen per class by Silverman's rule on that class's own
    scores. Using a single pooled bandwidth would apply the resolution
    appropriate to the broad different-source distribution to the typically much
    narrower same-source one, over-smoothing exactly the region where the two
    separate.
    """

    def __init__(self, bandwidth_factor: float = 1.0) -> None:
        super().__init__()
        self._bandwidth_factor = bandwidth_factor
        self._same: NDArray[np.float64] | None = None
        self._different: NDArray[np.float64] | None = None
        self._same_bandwidth = 1.0
        self._different_bandwidth = 1.0

    @property
    def calibrator_id(self) -> str:
        return f"kde(h_ss={self._same_bandwidth:.4g},h_ds={self._different_bandwidth:.4g})"

    def fit(
        self, scores: NDArray[np.float64], labels: NDArray[np.int64]
    ) -> KernelDensityCalibrator:
        scores, labels = self._validate(scores, labels)
        self._same = scores[labels == 1]
        self._different = scores[labels == 0]
        self._same_bandwidth = self._silverman(self._same)
        self._different_bandwidth = self._silverman(self._different)
        self._finish_fit(scores, labels)
        return self

    def _silverman(self, values: NDArray[np.float64]) -> float:
        """Silverman's rule, using the more robust spread estimate.

        ``0.9 * min(sd, IQR / 1.34) * n^(-1/5)``. The interquartile term guards
        against a heavy-tailed score distribution, where the standard deviation
        is inflated by a few extreme trials and the resulting bandwidth
        over-smooths the bulk of the data.
        """
        n = values.size
        spread = float(np.std(values))
        iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
        if iqr > 0.0:
            spread = min(spread, iqr / 1.34)
        if spread <= 0.0:
            spread = 1e-6
        return float(self._bandwidth_factor * 0.9 * spread * n ** (-0.2))

    def transform(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._same is None or self._different is None:
            raise ModelNotTrainedError("kernel density calibrator has not been fitted")
        query = np.asarray(scores, dtype=np.float64).ravel()
        same_log_density = _gaussian_kde_log_density(
            query, self._same, self._same_bandwidth
        )
        different_log_density = _gaussian_kde_log_density(
            query, self._different, self._different_bandwidth
        )
        return same_log_density - different_log_density


#: Largest kernel matrix held at once, in elements. At eight bytes each this is
#: about 64 MB, which is small enough to be irrelevant on any machine that can
#: hold the scores themselves and large enough that the chunking overhead does
#: not show up in a profile.
_KDE_BLOCK_ELEMENTS: Final[int] = 8_000_000


def _gaussian_kde_log_density(
    query: NDArray[np.float64], sample: NDArray[np.float64], bandwidth: float
) -> NDArray[np.float64]:
    """Log density of a Gaussian KDE, evaluated stably and in bounded memory.

    Computed through :func:`scipy.special.logsumexp` rather than by summing
    kernels. In the tails every kernel contributes an underflowing exponential,
    their sum is exactly zero, and the logarithm of that is negative infinity —
    which then propagates into the likelihood ratio as an infinity. The
    log-domain form gives the correct small number instead.

    Evaluated in blocks over the query axis. The kernel matrix is
    ``n_query x n_sample``, and both grow with the size of the validation set:
    twenty thousand of each is a matrix of four hundred million doubles, three
    and a half gigabytes, for a calculation whose output is twenty thousand
    numbers. Allocating it whole made this calibrator fail on precisely the
    validation sets large enough to justify using it.

    Each row's ``logsumexp`` is independent of every other, so blocking changes
    nothing about the result — it is the same arithmetic in the same order,
    performed on a slice at a time.
    """
    from scipy.special import logsumexp

    query = np.asarray(query, dtype=np.float64).ravel()
    sample = np.asarray(sample, dtype=np.float64).ravel()
    if sample.size == 0:
        raise CalibrationError("cannot estimate a density from an empty sample")

    block = max(1, _KDE_BLOCK_ELEMENTS // sample.size)
    output = np.empty(query.size, dtype=np.float64)
    constant = 0.5 * np.log(2.0 * np.pi) + np.log(bandwidth)

    for start in range(0, query.size, block):
        stop = min(start + block, query.size)
        z = (query[start:stop, None] - sample[None, :]) / bandwidth
        output[start:stop] = logsumexp(-0.5 * z**2 - constant, axis=1)

    return output - np.log(sample.size)
