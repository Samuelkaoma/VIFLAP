"""The unit of comparison and the result it produces.

A :class:`ComparisonResult` is the system's primary output. It is designed so
that it cannot be constructed without everything required to interpret it
correctly, and so that no field can be read in isolation and mistaken for a
conclusion:

- the fused log-likelihood-ratio, with its uncertainty interval;
- what each stream contributed, including the streams that contributed nothing
  and why;
- the prior odds of the search that produced it, and hence the posterior;
- what naive conditional-independence summation would have claimed instead, so
  the cost of the assumption is visible rather than assumed away;
- the identity of every model involved, so a result can be reproduced.

There is no constructor that omits the prior. A result stripped of its search
context is the raw material of the prosecutor's fallacy, and this type does not
permit one to exist.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from viflap.domain.errors import InvalidEvidenceError
from viflap.domain.evidence import (
    AbsenceReason,
    EvidenceStream,
    StreamAbsent,
    StreamEvidence,
    StreamOutcome,
    ValidityAssessment,
    absent_streams,
    missingness_pattern,
    present_evidence,
)
from viflap.domain.governance import CaseReference
from viflap.domain.hypotheses import PosteriorAssessment, PriorOdds
from viflap.domain.values import (
    EvidentialStrength,
    LogLikelihoodRatio,
    UncertaintyInterval,
)

__all__ = [
    "ComparisonResult",
    "FusionMethod",
    "IncidentId",
    "IncidentPair",
    "OverstatementEstimate",
]

_LN10 = math.log(10.0)


@dataclass(frozen=True, slots=True, order=True)
class IncidentId:
    """Identifier of a single reported incident."""

    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise InvalidEvidenceError("incident identifier must be non-empty")

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return self.value


@dataclass(frozen=True, slots=True)
class IncidentPair:
    """Two incidents being compared, under the authority of a case.

    The pair is unordered in meaning — evidence for linkage is symmetric — and
    is normalised to a canonical order at construction so that the same
    comparison has one identity regardless of which incident the investigator
    started from. Without this, a cache or a deduplication step silently treats
    ``(A, B)`` and ``(B, A)`` as different work.
    """

    first: IncidentId
    second: IncidentId
    authorised_by: CaseReference

    def __post_init__(self) -> None:
        if self.first == self.second:
            raise InvalidEvidenceError(
                "an incident cannot be compared with itself",
                incident=self.first.value,
            )
        if self.second < self.first:
            object.__setattr__(self, "first", self.second)
            object.__setattr__(self, "second", self.first)

    @property
    def key(self) -> str:
        """Stable identity of this comparison, independent of argument order."""
        return f"{self.first.value}|{self.second.value}"

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return f"{self.first} <-> {self.second}"


class FusionMethod(Enum):
    """How per-stream likelihood ratios were combined.

    Recorded on every result. The methods disagree, sometimes by orders of
    magnitude, and a result that does not say which produced it cannot be
    compared with another.
    """

    NAIVE_INDEPENDENT = "naive_independent"
    """Summation of log-LRs. Valid only under conditional independence, which is
    false here. Retained exclusively as the baseline against which overstatement
    is measured; it is not selectable for operational use."""

    LINEAR_LOGISTIC = "linear_logistic"
    """Weighted linear fusion with weights trained to minimise ``C_llr``. Absorbs
    systematic dependence into the weights. Standard practice, and provably
    insufficient where dependence is strong and non-linear."""

    GAUSSIAN_LATENT = "gaussian_latent"
    """Multivariate Gaussian models of the joint log-LR vector under each
    proposition, with the shared cause represented explicitly. Marginalises over
    absent streams analytically."""

    GAUSSIAN_COPULA = "gaussian_copula"
    """Empirical marginals with a Gaussian copula for the dependence structure.
    Separates marginal calibration from dependence; sensitive to
    misspecification in the tails, which is where the results that matter live."""

    @property
    def models_dependence(self) -> bool:
        return self is not FusionMethod.NAIVE_INDEPENDENT

    @property
    def is_operational(self) -> bool:
        """Whether the method may be used to produce a reported result."""
        return self.models_dependence


@dataclass(frozen=True, slots=True)
class OverstatementEstimate:
    """How far naive conditional-independence summation would have overstated.

    This is a safety quantity, reported with every result rather than computed
    once in an appendix. The direction of the error is what makes it worth
    reporting: correlated evidence treated as independent inflates the apparent
    weight, and inflation runs against the person the comparison concerns.

    Two figures, because they answer different questions:

    ``displacement_log10``
        Signed difference, ``log10(LR_naive) - log10(LR_corrected)``. Where the
        naive figure sits relative to the corrected one.

    ``exaggeration_log10``
        Difference in *magnitude*, ``|log10 LR_naive| - |log10 LR_corrected|``.
        How many orders of magnitude of apparent strength the independence
        assumption manufactured, irrespective of which proposition is supported.
        Negative values mean the assumption understated, which does occur when
        streams are negatively dependent.
    """

    naive_log_lr: LogLikelihoodRatio
    corrected_log_lr: LogLikelihoodRatio

    @property
    def displacement_log10(self) -> float:
        return (self.naive_log_lr.value - self.corrected_log_lr.value) / _LN10

    @property
    def exaggeration_log10(self) -> float:
        return (abs(self.naive_log_lr.value) - abs(self.corrected_log_lr.value)) / _LN10

    @property
    def is_material(self) -> bool:
        """Whether the assumption would have changed the verbal strength band."""
        return self.naive_log_lr.strength is not self.corrected_log_lr.strength


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """A fully contextualised comparison of two incidents."""

    pair: IncidentPair
    fused_log_lr: LogLikelihoodRatio
    uncertainty: UncertaintyInterval
    outcomes: Mapping[EvidenceStream, StreamOutcome]
    posterior: PosteriorAssessment
    fusion_method: FusionMethod
    fusion_model_id: str
    validity_assessments: tuple[ValidityAssessment, ...]
    computed_at: datetime
    overstatement: OverstatementEstimate | None = None
    """``None`` only where fewer than two streams contributed, in which case
    there is no independence assumption to test."""

    notes: tuple[str, ...] = field(default_factory=tuple)
    """Caveats attached during computation — extrapolation beyond the calibration
    domain, a stream near its data minimum. Surfaced with the result, never
    logged and discarded."""

    def __post_init__(self) -> None:
        if self.posterior.log_lr != self.fused_log_lr:
            raise InvalidEvidenceError(
                "posterior was derived from a different likelihood ratio than "
                "the one recorded on this result",
                result_log_lr=self.fused_log_lr.value,
                posterior_log_lr=self.posterior.log_lr.value,
            )
        if not self.uncertainty.contains(self.fused_log_lr.value):
            raise InvalidEvidenceError(
                "fused point estimate lies outside its uncertainty interval",
                log_lr=self.fused_log_lr.value,
                lower=self.uncertainty.lower,
                upper=self.uncertainty.upper,
            )
        if not self.fusion_method.is_operational and self.contributing_stream_count > 1:
            raise InvalidEvidenceError(
                "naive independent summation may not be used to produce a "
                "reported multi-stream result; it exists only as a baseline",
                method=self.fusion_method.value,
            )
        if self.computed_at.tzinfo is None:
            raise InvalidEvidenceError(
                "result timestamp must be timezone-aware; a naive timestamp "
                "cannot be reconciled with an audit record"
            )

    # -- Stream decomposition ---------------------------------------------

    @property
    def contributing(self) -> dict[EvidenceStream, StreamEvidence]:
        return present_evidence(self.outcomes)

    @property
    def absent(self) -> dict[EvidenceStream, StreamAbsent]:
        return absent_streams(self.outcomes)

    @property
    def contributing_stream_count(self) -> int:
        return len(self.contributing)

    @property
    def missingness_pattern(self) -> frozenset[EvidenceStream]:
        return missingness_pattern(self.outcomes)

    @property
    def rests_on_single_stream(self) -> bool:
        """Whether one stream carries the entire result.

        Interfaces are required to distinguish this case visually. A single
        stream at ``LR = 10^4`` and four streams agreeing at ``10^1`` each reach
        the same fused figure and warrant very different confidence: the second
        survives the failure of any one model, the first does not.
        """
        return self.contributing_stream_count == 1

    @property
    def acoustic_was_excluded(self) -> bool:
        """Whether the validity gate voided the acoustic evidence."""
        outcome = self.outcomes.get(EvidenceStream.ACOUSTIC)
        return (
            isinstance(outcome, StreamAbsent)
            and outcome.reason is AbsenceReason.EXCLUDED_BY_VALIDITY_GATE
        )

    def dominant_stream(self) -> EvidenceStream | None:
        """The stream contributing the largest magnitude, if any contributed."""
        contributing = self.contributing
        if not contributing:
            return None
        return max(contributing, key=lambda stream: abs(contributing[stream].log_lr.value))

    # -- Interpretation ---------------------------------------------------

    @property
    def strength(self) -> EvidentialStrength:
        return self.fused_log_lr.strength

    @property
    def supported_hypothesis_is_same_source(self) -> bool:
        return self.fused_log_lr.supports_same_source

    @property
    def is_weakly_determined(self) -> bool:
        """Whether the uncertainty interval crosses neutral."""
        return self.uncertainty.spans_neutral

    @property
    def prior(self) -> PriorOdds:
        return self.posterior.prior

    @property
    def requires_prominent_prior_warning(self) -> bool:
        """Whether the result looks strong but remains improbable given the prior.

        The ordinary outcome of a large database search, and the one most often
        lost between the report and the reader.
        """
        return self.posterior.is_dominated_by_prior

    def verbal_summary(self) -> str:
        """A complete, policy-compliant statement of what this result shows.

        Exists because :attr:`strength` alone is dangerous. The verbal band is a
        function of *magnitude*, so a likelihood ratio of ``10^-4`` is "very
        strong" — support for the **different-source** proposition. Rendering
        the band without the direction inverts the finding while sounding
        confident, and it is the single easiest way for a correct system to
        produce a false accusation.

        The direction is therefore not available separately in the rendered
        form: the only method that produces a sentence produces the whole
        sentence.
        """
        proposition = (
            "that the incidents were conducted by the same actor"
            if self.supported_hypothesis_is_same_source
            else "that the incidents were conducted by different actors"
        )
        strength = self.strength.value
        if self.strength is EvidentialStrength.NO_SUPPORT:
            return (
                f"The observed evidence {strength} in distinguishing the two "
                f"propositions (log10 LR = {self.fused_log_lr.log10:+.2f})."
            )
        return (
            f"The observed evidence {strength} the proposition {proposition} "
            f"(log10 LR = {self.fused_log_lr.log10:+.2f}, "
            f"{self.contributing_stream_count} of "
            f"{len(self.outcomes)} streams contributing)."
        )

    def caveats(self) -> tuple[str, ...]:
        """Every condition an investigator must be shown alongside this result.

        Assembled from the result's own structure rather than written by the
        caller, so that a presentation layer cannot omit one by forgetting it.
        """
        caveats: list[str] = list(self.notes)

        if self.rests_on_single_stream:
            stream = next(iter(self.contributing))
            caveats.append(
                f"The whole of this result rests on the {stream.display_name.lower()} "
                f"stream. It does not survive a failure of that one model."
            )
        if self.is_weakly_determined:
            caveats.append(
                "The uncertainty interval spans the neutral point, so the "
                "direction of support is not reliably determined."
            )
        if self.requires_prominent_prior_warning:
            caveats.append(
                f"Given the prior odds of this search ({self.prior.log_odds}), the "
                f"same-source proposition remains less probable than not: "
                f"posterior probability {self.posterior.probability}."
            )
        if self.acoustic_was_excluded:
            caveats.append(
                "Acoustic evidence was excluded because the recording was judged "
                "synthetic. It carries no information about a human speaker and "
                "was removed from fusion rather than down-weighted."
            )
        if self.overstatement is not None and self.overstatement.is_material:
            caveats.append(
                f"Assuming the streams independent would have placed this result "
                f"{self.overstatement.exaggeration_log10:+.1f} orders of magnitude "
                f"from its dependence-corrected value, in a different strength band."
            )
        if self.prior.basis.value == "uniform_over_database":
            caveats.append(
                "The prior is uniform over the searched population. Sensitivity "
                "to that choice should be reviewed before the result is relied on."
            )
        return tuple(caveats)
