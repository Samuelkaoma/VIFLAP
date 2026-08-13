"""Evidence streams and their outcomes.

The central modelling decision here is that a stream's outcome is a **sum
type**, not a record with optional fields:

.. code-block:: text

    StreamOutcome = StreamEvidence | StreamAbsent

The previous shape — ``(log_lr: float | None, uncertainty: Interval | None,
is_available: bool)`` — admits eight combinations of which only two are
meaningful, and pushes the burden of checking onto every consumer. Worse, the
easy mistake it invites is exactly the dangerous one: treating a missing stream
as ``log_lr = 0``. That is not a conservative default. It is the assertion that
the stream was computed and found the evidence equally probable under both
propositions, which is a fabricated observation.

With a sum type, a caller that wants the log-LR must first establish which case
it holds, and "absent" carries a reason that survives into the audit record and
the interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from viflap.domain.errors import InvalidEvidenceError
from viflap.domain.values import LogLikelihoodRatio, UncertaintyInterval

__all__ = [
    "AbsenceReason",
    "DisguiseCondition",
    "EvidenceStream",
    "StreamAbsent",
    "StreamEvidence",
    "StreamOutcome",
    "StreamOutcomes",
    "ValidityAssessment",
    "ValidityVerdict",
]


class EvidenceStream(Enum):
    """The evidence streams VIFLAP extracts and fuses.

    Each stream emits an independently calibrated likelihood ratio. They are
    *not* conditionally independent of one another — the same operator running
    the same operation is the common cause of all of them — which is why fusion
    models the dependence rather than summing.

    Order is fixed and meaningful: it determines the column order of every
    design matrix and every covariance estimate in the fusion layer, so a
    trained model remains interpretable across releases.
    """

    ACOUSTIC = "acoustic"
    BEHAVIOURAL = "behavioural"
    TEMPORAL = "temporal"
    TRANSACTIONAL = "transactional"
    DEVICE = "device"

    @property
    def display_name(self) -> str:
        return self.value.capitalize()

    @property
    def is_gated_by_validity(self) -> bool:
        """Whether the synthetic-speech gate can exclude this stream.

        Only the acoustic stream. A synthetic recording carries no information
        about a human speaker's anatomy, so its acoustic evidence is
        meaningless rather than weak. The behavioural stream survives synthesis
        in part — a text-to-speech rendering of a scripted pretext still carries
        the script — and is handled by its own admissibility logic.
        """
        return self is EvidenceStream.ACOUSTIC

    @classmethod
    def ordered(cls) -> tuple[EvidenceStream, ...]:
        """Canonical ordering, used for all vectorised representations."""
        return tuple(cls)


class DisguiseCondition(Enum):
    """Disguise state under which a recording was produced.

    Every performance claim in this system is conditional on disguise state. A
    ``C_llr`` reported without one is not a weak result, it is an uninterpretable
    one, because the same system spans an order of magnitude across these
    conditions.

    ``VOICE_CONVERSION`` and ``NEURAL_SYNTHESIS`` are categorically different
    from the rest: they do not degrade the acoustic evidence, they void it. They
    appear here so that corpora can be labelled with them, and the validity gate
    is what acts on them.
    """

    NATURAL = "natural"
    PITCH_RAISED = "pitch_raised"
    PITCH_LOWERED = "pitch_lowered"
    ACCENT_ADOPTED = "accent_adopted"
    NASAL_OCCLUDED = "nasal_occluded"
    WHISPERED = "whispered"
    ELECTRONIC_PITCH_SHIFT = "electronic_pitch_shift"
    VOICE_CONVERSION = "voice_conversion"
    NEURAL_SYNTHESIS = "neural_synthesis"
    UNKNOWN = "unknown"

    @property
    def voids_acoustic_evidence(self) -> bool:
        """Whether acoustic evidence is meaningless rather than merely degraded."""
        return self in (
            DisguiseCondition.VOICE_CONVERSION,
            DisguiseCondition.NEURAL_SYNTHESIS,
        )

    @property
    def is_deliberate(self) -> bool:
        """Whether the condition implies intent to defeat speaker comparison."""
        return self not in (DisguiseCondition.NATURAL, DisguiseCondition.UNKNOWN)


class ValidityVerdict(Enum):
    """Outcome of the synthetic-speech validity gate.

    The gate conditions admissibility; it does not contribute evidence. Its
    output is therefore a verdict, not a likelihood ratio, and it is recorded
    separately from the streams it governs.
    """

    ADMITTED = "admitted"
    EXCLUDED = "excluded"
    INDETERMINATE = "indeterminate"

    @property
    def permits_acoustic_evidence(self) -> bool:
        """Only an explicit admission permits acoustic evidence into fusion.

        ``INDETERMINATE`` does not. Where the detector cannot decide, the
        conservative action is to withhold the stream and flag the recording for
        human assessment — the asymmetry being that admitting synthetic speech
        contaminates a result presented to a court, whereas withholding genuine
        speech only weakens it.
        """
        return self is ValidityVerdict.ADMITTED


class AbsenceReason(Enum):
    """Why a stream produced no evidence.

    Carried through to the interface and the audit record. "No transaction
    records exist for this pair" and "the acoustic evidence was excluded because
    the recording is synthetic" are both absences, but an investigator must be
    able to tell them apart, and only the second is a finding.
    """

    NO_DATA = "no_data"
    """The underlying records do not exist for at least one incident."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Records exist but fall below the minimum the model requires."""

    EXCLUDED_BY_VALIDITY_GATE = "excluded_by_validity_gate"
    """Acoustic evidence voided: the recording was judged synthetic."""

    QUALITY_BELOW_THRESHOLD = "quality_below_threshold"
    """Data exists but its quality places it outside the calibrated domain."""

    MODEL_UNAVAILABLE = "model_unavailable"
    """No trained, calibrated model is deployed for this stream."""

    OUT_OF_CALIBRATION_DOMAIN = "out_of_calibration_domain"
    """Conditions differ so far from the calibration set that any LR would be
    an extrapolation. Reporting one would be a false precision."""

    @property
    def is_finding(self) -> bool:
        """Whether the absence is itself investigatively meaningful.

        Exclusion by the validity gate says something about the recording.
        Absent transaction records usually say something about data access.
        """
        return self is AbsenceReason.EXCLUDED_BY_VALIDITY_GATE


@dataclass(frozen=True, slots=True)
class ValidityAssessment:
    """The validity gate's judgement on one recording."""

    recording_id: str
    verdict: ValidityVerdict
    countermeasure_log_lr: float
    """Log-likelihood-ratio of ``bona fide`` versus ``spoofed``, from the
    countermeasure. Positive supports genuine speech. Reported so that the
    operating point can be audited and, if policy changes, re-applied to
    historical assessments without re-running the detector."""

    threshold: float
    """The operating point in force when the verdict was issued."""

    detector_id: str
    """Identifier of the countermeasure model and version, so that a verdict
    can be traced to the system that produced it."""

    def __post_init__(self) -> None:
        if not self.recording_id.strip():
            raise InvalidEvidenceError("validity assessment requires a recording id")
        if not self.detector_id.strip():
            raise InvalidEvidenceError(
                "validity assessment requires a detector identifier so the "
                "verdict can be traced to a specific model version",
                recording_id=self.recording_id,
            )


@dataclass(frozen=True, slots=True)
class StreamEvidence:
    """A likelihood ratio produced by one stream for one comparison."""

    stream: EvidenceStream
    log_lr: LogLikelihoodRatio
    uncertainty: UncertaintyInterval
    model_id: str
    """Identifier of the scoring and calibration models that produced this
    value. Two results from different model versions are not comparable, and
    without this they are indistinguishable."""

    diagnostics: Mapping[str, float] = field(default_factory=dict)
    """Stream-specific quantities that explain the value: speech duration,
    number of transactions, SNR. Used by the interface and by post-hoc analysis;
    never used in fusion, which sees only the calibrated log-LR."""

    def __post_init__(self) -> None:
        if not self.uncertainty.contains(self.log_lr.value):
            raise InvalidEvidenceError(
                "point estimate lies outside its own uncertainty interval, "
                "which indicates a defect in interval estimation",
                stream=self.stream.value,
                log_lr=self.log_lr.value,
                lower=self.uncertainty.lower,
                upper=self.uncertainty.upper,
            )
        if not self.model_id.strip():
            raise InvalidEvidenceError(
                "stream evidence must identify the model that produced it",
                stream=self.stream.value,
            )

    @property
    def is_present(self) -> bool:
        return True

    @property
    def is_weakly_determined(self) -> bool:
        """Whether the uncertainty interval crosses neutral.

        Such a result must be presented differently from one whose interval lies
        wholly on one side. The point estimate alone would suggest a direction
        the data does not support.
        """
        return self.uncertainty.spans_neutral


@dataclass(frozen=True, slots=True)
class StreamAbsent:
    """A stream produced no evidence, with the reason it did not.

    Deliberately *not* interchangeable with a neutral likelihood ratio. Fusion
    must marginalise over the absent stream using a model fitted to comparisons
    where that stream was also absent, rather than imputing a value.
    """

    stream: EvidenceStream
    reason: AbsenceReason
    detail: str = ""
    """Operator-facing explanation. Passed through the output-language policy
    before display, like any other text leaving the system."""

    @property
    def is_present(self) -> bool:
        return False


StreamOutcome: TypeAlias = StreamEvidence | StreamAbsent
"""Either evidence, or a reasoned absence. There is no third case."""

StreamOutcomes: TypeAlias = Mapping[EvidenceStream, StreamOutcome]
"""Outcomes for every stream considered in a comparison.

Every stream the deployment is configured to attempt appears as a key, including
those that produced nothing. A stream that is silently omitted is indexed
identically to one that was never attempted, and the difference matters to an
investigator reading the result.
"""


def present_evidence(outcomes: StreamOutcomes) -> dict[EvidenceStream, StreamEvidence]:
    """Narrow a mapping of outcomes to those that carry evidence."""
    return {
        stream: outcome
        for stream, outcome in outcomes.items()
        if isinstance(outcome, StreamEvidence)
    }


def absent_streams(outcomes: StreamOutcomes) -> dict[EvidenceStream, StreamAbsent]:
    """Narrow a mapping of outcomes to those that produced nothing."""
    return {
        stream: outcome
        for stream, outcome in outcomes.items()
        if isinstance(outcome, StreamAbsent)
    }


def missingness_pattern(outcomes: StreamOutcomes) -> frozenset[EvidenceStream]:
    """The set of streams that produced evidence.

    Fusion is conditioned on this set. A model fitted where all five streams are
    present cannot be applied when two are missing: the weights absorb
    dependence between the streams that were present when it was trained, and
    those dependencies do not survive the removal of a stream.
    """
    return frozenset(present_evidence(outcomes))
