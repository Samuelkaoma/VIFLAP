"""Forensic evaluation metrics as domain objects.

Two decisions here are worth stating explicitly.

**Derived quantities are properties, not fields.** ``calibration_loss`` is
``C_llr - C_llr_min`` by definition. Storing it as a field creates the
possibility of a value that disagrees with its own definition, which then has to
be validated, which then has a tolerance, which then has an edge case. Deriving
it removes the class of defect rather than checking for it.

**Accuracy, precision, recall and F1 are absent, deliberately.** They require a
decision threshold. Placing a threshold inside the system pre-empts a judgement
that belongs to the investigator and ultimately to the court, and it hides the
quantity that matters: whether the strength the system claims is the strength it
has. A system can be highly accurate and still systematically overstate, and
threshold metrics cannot see that. ``C_llr`` can, because it is a proper scoring
rule.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from viflap.domain.errors import MetricInvariantError

__all__ = [
    "CalibrationSummary",
    "Estimate",
    "PerformanceGrade",
]


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with a confidence interval and the resampling unit.

    ``resampling_unit`` is mandatory because it changes what the interval means.
    Intervals from resampling *trials* are narrower than intervals from
    resampling *speakers*, and are wrong: trials sharing a speaker are not
    independent, so trial-level resampling treats correlated observations as
    fresh information and understates variance. Two intervals computed different
    ways are not comparable, and without this field they are indistinguishable.
    """

    value: float
    lower: float
    upper: float
    confidence_level: float
    resampling_unit: str
    n_resamples: int

    interval_method: str = "percentile"
    """How the bounds were derived. Recorded for the same reason as
    ``resampling_unit``: two intervals computed by different methods are not
    comparable, and for a biased statistic they are not even close. A
    resubstitution minimum — ``C_llr_min`` fits its PAV transform on the trials
    it scores — is optimistically biased, and the bias grows as the effective
    sample shrinks, which is exactly what a bootstrap resample does. The
    percentile method corrects for none of that; ``bca`` corrects for both the
    median bias and the skew. On this project's data the difference is about
    0.05, which is enough to move a verdict across a threshold."""

    n_discarded: int = 0
    """Resamples that produced no computable value, usually because a draw
    contained a single class. A high rate means the estimate rests on a handful
    of speakers and the interval, however computed, is optimistic — so it is
    carried with the estimate rather than left in a log."""

    p_value_vs_zero: float | None = None
    """Two-sided bootstrap p-value against a null of zero, where that null means
    something — a *difference* between two systems, not a level.

    Carried because "the interval excludes zero" is a hypothesis test whichever
    way it is written, and a document reporting six of them is testing a family.
    Without a p-value there is nothing for a multiplicity correction to consume,
    and the correction silently never happens. ``None`` where a null of zero is
    meaningless, which is every single-system estimate."""

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (self.value, self.lower, self.upper)):
            raise MetricInvariantError(
                "estimate and its bounds must be finite",
                value=self.value,
                lower=self.lower,
                upper=self.upper,
            )
        if self.lower > self.upper:
            raise MetricInvariantError(
                "estimate interval is inverted", lower=self.lower, upper=self.upper
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise MetricInvariantError(
                "confidence level must lie strictly within (0, 1)",
                confidence_level=self.confidence_level,
            )
        if not self.resampling_unit.strip():
            raise MetricInvariantError(
                "the resampling unit must be recorded; intervals from trial-level "
                "and speaker-level resampling are not comparable"
            )

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def excludes(self, reference: float, /) -> bool:
        """Whether ``reference`` lies outside the interval."""
        return not (self.lower <= reference <= self.upper)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        pct = int(round(self.confidence_level * 100))
        return (
            f"{self.value:.4f} [{self.lower:.4f}, {self.upper:.4f}] "
            f"({pct}% CI, {self.n_resamples:,} resamples over {self.resampling_unit})"
        )


class PerformanceGrade(Enum):
    """Qualitative reading of ``C_llr``.

    The boundary that matters is one. ``C_llr = 1`` is the cost of a system that
    reports a likelihood ratio of one for everything — that is, of contributing
    nothing. Above one, the system is not merely unhelpful, it is *misleading*:
    an investigator acting on its output does worse than one ignoring it.
    Threshold-based accuracy metrics cannot express this, which is why they are
    not used here.
    """

    INFORMATIVE = "informative"
    MARGINAL = "marginal"
    UNINFORMATIVE = "uninformative"
    MISLEADING = "misleading"

    @classmethod
    def for_cllr(cls, c_llr: float, /) -> PerformanceGrade:
        if c_llr > 1.0:
            return cls.MISLEADING
        if math.isclose(c_llr, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            return cls.UNINFORMATIVE
        if c_llr > 0.9:
            return cls.MARGINAL
        return cls.INFORMATIVE


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """Evaluation of a set of likelihood ratios against known ground truth.

    ``C_llr`` is the empirical cross-entropy of the reported likelihood ratios
    under a flat prior, in bits:

    .. code-block:: text

        C_llr = 1/2 [ (1/N_ss) sum_{i in ss} log2(1 + 1/LR_i)
                    + (1/N_ds) sum_{j in ds} log2(1 + LR_j) ]

    It is a strictly proper scoring rule, so it is minimised only by reporting
    the likelihood ratios the data actually supports. Overstating is penalised
    even when the direction is right, which is the property that makes it the
    correct metric for this application.

    Its decomposition separates two failures that accuracy conflates:

    ``C_llr_min``
        Cost after optimal monotonic recalibration (pool-adjacent-violators).
        What the system could achieve if its numbers were honest — pure
        discrimination.

    ``calibration_loss``
        The remainder. The system can tell these apart to some degree but is
        misrepresenting how confidently.
    """

    c_llr: float
    c_llr_min: float
    n_same_source: int
    n_different_source: int
    eer: float | None = None
    """Equal error rate. Reported only for comparability with the speaker
    recognition literature, which uses it as a primary metric. It requires a
    threshold and says nothing about calibration; it is not used to make any
    decision in this system."""

    elub_lower_log10: float | None = None
    elub_upper_log10: float | None = None
    """Empirical lower and upper bounds on the likelihood ratios this validation
    set can support. A reported ``LR`` beyond these is an extrapolation: the
    data contains no counterexample strong enough to justify it. Bounding to
    these is the honest response to a system that would otherwise emit
    ``10^12`` from a validation set of a few thousand trials."""

    def __post_init__(self) -> None:
        for name, value in (("c_llr", self.c_llr), ("c_llr_min", self.c_llr_min)):
            if not math.isfinite(value):
                raise MetricInvariantError(f"{name} must be finite", **{name: value})
        # Tolerance absorbs floating-point noise only. A genuine violation
        # indicates a defect in the metric implementation, since C_llr_min is by
        # construction the minimum of C_llr over monotonic recalibrations.
        tolerance = 1e-9
        if self.c_llr_min < -tolerance:
            raise MetricInvariantError(
                "C_llr_min is negative, which is impossible for a cross-entropy",
                c_llr_min=self.c_llr_min,
            )
        if self.c_llr < self.c_llr_min - tolerance:
            raise MetricInvariantError(
                "C_llr is below C_llr_min; since C_llr_min minimises C_llr over "
                "monotonic recalibrations this indicates a defect in the metric "
                "implementation, not a property of the system evaluated",
                c_llr=self.c_llr,
                c_llr_min=self.c_llr_min,
            )
        if self.n_same_source < 1 or self.n_different_source < 1:
            raise MetricInvariantError(
                "C_llr is undefined without trials of both types",
                n_same_source=self.n_same_source,
                n_different_source=self.n_different_source,
            )
        if self.eer is not None and not 0.0 <= self.eer <= 1.0:
            raise MetricInvariantError("EER must lie in [0, 1]", eer=self.eer)
        if (
            self.elub_lower_log10 is not None
            and self.elub_upper_log10 is not None
            and self.elub_lower_log10 > self.elub_upper_log10
        ):
            raise MetricInvariantError(
                "empirical LR bounds are inverted",
                lower=self.elub_lower_log10,
                upper=self.elub_upper_log10,
            )

    @property
    def calibration_loss(self) -> float:
        """``C_llr - C_llr_min``: the cost of misrepresenting strength."""
        return max(0.0, self.c_llr - self.c_llr_min)

    @property
    def discrimination_loss(self) -> float:
        """``C_llr_min``: the cost the system cannot remove by recalibrating."""
        return self.c_llr_min

    @property
    def grade(self) -> PerformanceGrade:
        return PerformanceGrade.for_cllr(self.c_llr)

    @property
    def n_trials(self) -> int:
        return self.n_same_source + self.n_different_source

    @property
    def is_well_calibrated(self) -> bool:
        """Whether calibration loss is small relative to what is achievable.

        Judged as a fraction of the total cost rather than of ``C_llr_min``,
        because a system with excellent discrimination has a small
        ``C_llr_min``, and dividing by it makes any calibration error look
        catastrophic.
        """
        if self.c_llr <= 0.0:
            return True
        return (self.calibration_loss / self.c_llr) < 0.05

    @property
    def is_misleading(self) -> bool:
        """Whether an investigator would do better ignoring this system."""
        return self.grade is PerformanceGrade.MISLEADING

    def describe(self) -> str:
        """One-paragraph statement of what this evaluation shows."""
        parts = [
            f"C_llr = {self.c_llr:.4f} over {self.n_trials:,} trials "
            f"({self.n_same_source:,} same-source, {self.n_different_source:,} "
            f"different-source), decomposing into discrimination "
            f"C_llr_min = {self.c_llr_min:.4f} and calibration loss "
            f"{self.calibration_loss:.4f}."
        ]
        if self.is_misleading:
            parts.append(
                "This exceeds 1.0: the system is worse than uninformative, and "
                "an investigator relying on it would decide less well than one "
                "ignoring it."
            )
        elif not self.is_well_calibrated:
            parts.append(
                "Discrimination is not the limiting factor; the reported "
                "strengths are not warranted by the data and require "
                "recalibration before use."
            )
        if self.elub_upper_log10 is not None:
            parts.append(
                f"Likelihood ratios are bounded to "
                f"[10^{self.elub_lower_log10:.2f}, 10^{self.elub_upper_log10:.2f}] "
                f"by the size and separation of this validation set."
            )
        return " ".join(parts)


def aggregate_trial_counts(summaries: Iterable[CalibrationSummary]) -> tuple[int, int]:
    """Total same-source and different-source trial counts across summaries."""
    same = 0
    different = 0
    for summary in summaries:
        same += summary.n_same_source
        different += summary.n_different_source
    return same, different
