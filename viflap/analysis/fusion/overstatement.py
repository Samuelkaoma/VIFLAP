"""Quantifying the cost of assuming conditional independence.

The research proposal names this as an explicit deliverable, and it is worth
being clear about why it deserves that status rather than being a footnote to
the fusion results.

Suppose the sophisticated dependence models turn out to gain little over naive
summation. That would be a disappointing result for the fusion contribution.
It would *not* make this quantity uninteresting — because the question "how
badly does the standard method mislead" is answered either way, and the standard
method is what deployed multimodal forensic systems actually use. A finding that
naive summation overstates by two orders of magnitude on real data is a safety
result about an entire class of systems, and it holds whether or not the
correction is worth its complexity.

So the measurement is made and reported on every comparison, not only in
aggregate at the end of a study.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.fusion.base import FusionModel, FusionTrainingSet
from viflap.analysis.fusion.models import NaiveIndependentFusion
from viflap.domain.errors import InsufficientDataError
from viflap.domain.values import EvidentialStrength

__all__ = ["OverstatementReport", "measure_overstatement"]

_LN10 = math.log(10.0)


@dataclass(frozen=True, slots=True)
class OverstatementReport:
    """Distribution of the exaggeration produced by independence, over a trial set."""

    exaggeration_log10: NDArray[np.float64]
    """Per-comparison ``|log10 LR_naive| - |log10 LR_corrected|``. Positive means
    independence inflated the apparent strength."""

    same_source_mask: NDArray[np.bool_]
    n_comparisons: int
    n_band_changes: int
    """Comparisons where independence would have placed the result in a
    different verbal strength band. The figure that matters for a report,
    because the band is what gets read aloud."""

    naive_cllr: float
    corrected_cllr: float

    @property
    def median_exaggeration_log10(self) -> float:
        return float(np.median(self.exaggeration_log10))

    @property
    def upper_decile_exaggeration_log10(self) -> float:
        """The ninetieth percentile.

        Reported alongside the median because the distribution is heavily
        right-skewed: most comparisons are barely affected and a minority are
        affected enormously, and a median alone conceals the minority. Those are
        the strongly supporting comparisons — which are the ones acted upon.
        """
        return float(np.percentile(self.exaggeration_log10, 90))

    @property
    def worst_exaggeration_log10(self) -> float:
        return float(np.max(self.exaggeration_log10))

    @property
    def fraction_overstated(self) -> float:
        return float(np.mean(self.exaggeration_log10 > 0.0))

    @property
    def false_support_exaggeration_log10(self) -> float:
        """Median exaggeration restricted to different-source comparisons.

        The harm-bearing subset. Overstating the evidence on a genuinely linked
        pair wastes nothing; overstating it on an unlinked pair is what directs
        an investigation at the wrong person.
        """
        different = self.exaggeration_log10[~self.same_source_mask]
        if different.size == 0:
            return float("nan")
        return float(np.median(different))

    def describe(self) -> str:
        return (
            f"Across {self.n_comparisons:,} comparisons, assuming the evidence "
            f"streams conditionally independent inflates the reported strength "
            f"by a median of {self.median_exaggeration_log10:.2f} orders of "
            f"magnitude, rising to {self.upper_decile_exaggeration_log10:.2f} at "
            f"the ninetieth percentile and {self.worst_exaggeration_log10:.2f} at "
            f"worst. {self.n_band_changes:,} comparisons "
            f"({self.n_band_changes / max(self.n_comparisons, 1):.1%}) would have "
            f"been reported in a different verbal strength band. On "
            f"different-source comparisons — where overstatement directs an "
            f"investigation at the wrong person — the median inflation is "
            f"{self.false_support_exaggeration_log10:.2f} orders of magnitude. "
            f"Independence also costs accuracy overall: C_llr rises from "
            f"{self.corrected_cllr:.4f} to {self.naive_cllr:.4f}."
        )


def measure_overstatement(
    corrected: FusionModel,
    evaluation: FusionTrainingSet,
) -> OverstatementReport:
    """Compare a dependence-corrected model against naive summation.

    Both models are applied to the same held-out comparisons, so the difference
    isolates the effect of the independence assumption rather than confounding
    it with a different training set or a different set of streams.
    """
    from viflap.analysis.calibration.metrics import compute_cllr

    naive_model = NaiveIndependentFusion()
    naive_values: list[float] = []
    corrected_values: list[float] = []
    labels: list[int] = []
    band_changes = 0

    for observation in evaluation.observations:
        if not observation.log_lrs:
            continue
        pattern = observation.pattern
        if not corrected.supports_pattern(pattern):
            continue

        naive = naive_model.fuse(observation.log_lrs)
        fixed = corrected.fuse(observation.log_lrs)

        naive_values.append(naive)
        corrected_values.append(fixed)
        labels.append(1 if observation.is_same_source else 0)

        if EvidentialStrength.for_log10_lr(
            naive / _LN10
        ) is not EvidentialStrength.for_log10_lr(fixed / _LN10):
            band_changes += 1

    if len(naive_values) < 2:
        raise InsufficientDataError(
            "too few evaluable comparisons to measure overstatement",
            n_comparisons=len(naive_values),
        )

    naive_array = np.array(naive_values)
    corrected_array = np.array(corrected_values)
    label_array = np.array(labels)

    exaggeration = (np.abs(naive_array) - np.abs(corrected_array)) / _LN10

    return OverstatementReport(
        exaggeration_log10=exaggeration,
        same_source_mask=label_array == 1,
        n_comparisons=len(naive_values),
        n_band_changes=band_changes,
        naive_cllr=compute_cllr(naive_array, label_array),
        corrected_cllr=compute_cllr(corrected_array, label_array),
    )
