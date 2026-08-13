"""Adapters presenting each analysis component as a :class:`StreamComparator`.

This is the seam between the science and the application. Each adapter does the
same three things, and the third is the one that matters:

1. Pull its own payload out of the evidence bundle.
2. Compute an uncalibrated score with the analysis code.
3. **Pass that score through a fitted calibrator** to obtain a likelihood ratio,
   with empirical bounds applied.

Step three is why these adapters exist as a layer rather than the use case
calling the analysis code directly. It is the single place where a score becomes
evidence, so the rule "no uncalibrated number is ever reported" is enforced by
the structure of the code rather than by everyone remembering it. An adapter
constructed without a fitted calibrator produces absence, not numbers.

The uncertainty interval on every stream comes from the calibrator's empirical
bounds and its residual spread on development data — not from a fixed
``+/- 0.5`` placeholder, which would state a precision nobody measured.
"""

from __future__ import annotations

from collections.abc import Mapping

from viflap.analysis.behaviour.profile import BehaviouralComparator, BehaviouralProfile
from viflap.analysis.calibration.calibrators import Calibrator
from viflap.analysis.patterns.streams import (
    DeviceComparator,
    DeviceObservation,
    TemporalComparator,
    TemporalProfile,
    TransactionalComparator,
    TransactionalProfile,
)
from viflap.analysis.speaker.pipeline import AcousticEmbedding, SpeakerComparisonSystem
from viflap.application.ports import EvidenceBundle
from viflap.domain.errors import InsufficientDataError, ModelNotTrainedError
from viflap.domain.evidence import (
    AbsenceReason,
    EvidenceStream,
    StreamAbsent,
    StreamEvidence,
    StreamOutcome,
)
from viflap.domain.values import UncertaintyInterval

__all__ = [
    "AcousticStreamComparator",
    "BehaviouralStreamComparator",
    "CalibratedStreamComparator",
    "DeviceStreamComparator",
    "TemporalStreamComparator",
    "TransactionalStreamComparator",
]


class CalibratedStreamComparator:
    """Base adapter: score, calibrate, bound, and attach an interval.

    Subclasses supply only ``_score`` and the payload type they expect.
    Everything about turning a score into reportable evidence — calibration,
    empirical bounding, the uncertainty interval, the model identity — happens
    here once, so no stream can accidentally omit it.
    """

    def __init__(
        self,
        stream: EvidenceStream,
        calibrator: Calibrator,
        scorer_id: str,
        residual_spread: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        residual_spread:
            Standard deviation of the calibrated log-LR on development trials of
            the same class, in nats. Sets the width of the reported interval.
            Zero means it was never measured, and the adapter then reports an
            interval derived from the empirical bounds alone — wide, and
            honestly so.
        """
        self._stream = stream
        self._calibrator = calibrator
        self._scorer_id = scorer_id
        self._residual_spread = max(residual_spread, 0.0)

    @property
    def stream(self) -> EvidenceStream:
        return self._stream

    @property
    def model_id(self) -> str:
        return f"{self._scorer_id}+{self._calibrator.calibrator_id}"

    def compare(self, first: EvidenceBundle, second: EvidenceBundle) -> StreamOutcome:
        if not self._calibrator.is_fitted:
            return StreamAbsent(
                stream=self._stream,
                reason=AbsenceReason.MODEL_UNAVAILABLE,
                detail=(
                    f"No fitted calibration is deployed for the "
                    f"{self._stream.display_name.lower()} stream. An uncalibrated "
                    f"score carries an implied strength it does not have, so no "
                    f"value is reported."
                ),
            )

        try:
            score, diagnostics = self._score(first, second)
        except InsufficientDataError as exc:
            return StreamAbsent(
                stream=self._stream,
                reason=AbsenceReason.INSUFFICIENT_DATA,
                detail=exc.message,
            )
        except ModelNotTrainedError as exc:
            return StreamAbsent(
                stream=self._stream,
                reason=AbsenceReason.MODEL_UNAVAILABLE,
                detail=exc.message,
            )

        calibrated = self._calibrator.calibrate(score)

        if calibrated.was_bounded:
            diagnostics = {
                **diagnostics,
                "extrapolation_log10": calibrated.extrapolation_log10,
            }

        return StreamEvidence(
            stream=self._stream,
            log_lr=calibrated.log_lr,
            uncertainty=self._interval(calibrated.log_lr.value),
            model_id=self.model_id,
            diagnostics={**diagnostics, "raw_score": float(score)},
        )

    def _interval(self, log_lr: float) -> UncertaintyInterval:
        """Interval around the point estimate, clipped to what data supports.

        Built from the measured residual spread where one is available, and
        clipped to the empirical bounds in every case. The clip is what stops an
        interval extending into a region the development set says nothing about
        — a point estimate can sit at the bound, but the interval must not run
        past it and imply the system could have said more.
        """
        bounds = self._calibrator.bounds  # type: ignore[attr-defined]
        if self._residual_spread > 0.0:
            half_width = 1.96 * self._residual_spread
        else:
            half_width = 0.5 * (bounds.upper_log_lr - bounds.lower_log_lr)

        interval = UncertaintyInterval(lower=log_lr - half_width, upper=log_lr + half_width)
        return interval.clipped_to(bounds.lower_log_lr, bounds.upper_log_lr)

    def _score(
        self, first: EvidenceBundle, second: EvidenceBundle
    ) -> tuple[float, Mapping[str, float]]:  # pragma: no cover - overridden
        raise NotImplementedError


class AcousticStreamComparator(CalibratedStreamComparator):
    """Acoustic comparison via the i-vector/PLDA system."""

    def __init__(
        self,
        system: SpeakerComparisonSystem,
        calibrator: Calibrator,
        residual_spread: float = 0.0,
    ) -> None:
        super().__init__(
            EvidenceStream.ACOUSTIC, calibrator, system.model_id, residual_spread
        )
        self._system = system

    def _score(
        self, first: EvidenceBundle, second: EvidenceBundle
    ) -> tuple[float, Mapping[str, float]]:
        left: AcousticEmbedding = first.payload(EvidenceStream.ACOUSTIC)
        right: AcousticEmbedding = second.payload(EvidenceStream.ACOUSTIC)

        # Duration governs how much a recording can carry, and shrinkage is the
        # formal expression of it. Surfacing it means "eleven seconds of speech"
        # appears in the result rather than only in the likelihood ratio.
        diagnostics = {
            "shrinkage_first": left.posterior_shrinkage,
            "shrinkage_second": right.posterior_shrinkage,
            "speech_seconds_first": float(left.diagnostics.get("speech_seconds", 0.0)),
            "speech_seconds_second": float(right.diagnostics.get("speech_seconds", 0.0)),
        }
        if not (left.is_well_determined and right.is_well_determined):
            raise InsufficientDataError(
                "at least one recording was too short to determine its acoustic "
                "representation; the resulting comparison would largely reflect "
                "the model's prior rather than the recordings",
                shrinkage_first=round(left.posterior_shrinkage, 3),
                shrinkage_second=round(right.posterior_shrinkage, 3),
            )
        return self._system.score(left, right), diagnostics


class BehaviouralStreamComparator(CalibratedStreamComparator):
    """Behavioural comparison over transcript profiles.

    Reports the idiolect and script components separately in its diagnostics,
    even though the calibrated likelihood ratio is computed from their total.
    "Same operation, different speaker" is a finding, and it is visible only in
    the decomposition.
    """

    def __init__(
        self,
        comparator: BehaviouralComparator,
        calibrator: Calibrator,
        scorer_id: str = "behavioural-v1",
        residual_spread: float = 0.0,
    ) -> None:
        super().__init__(EvidenceStream.BEHAVIOURAL, calibrator, scorer_id, residual_spread)
        self._comparator = comparator

    def _score(
        self, first: EvidenceBundle, second: EvidenceBundle
    ) -> tuple[float, Mapping[str, float]]:
        left: BehaviouralProfile = first.payload(EvidenceStream.BEHAVIOURAL)
        right: BehaviouralProfile = second.payload(EvidenceStream.BEHAVIOURAL)
        score = self._comparator.score(left, right)
        return score.total_log_lr, {
            **score.diagnostics,
            "idiolect_component": score.idiolect_log_lr,
            "script_component": score.script_log_lr,
            "suggests_delegation": float(score.suggests_shared_operation_not_speaker),
        }


class TemporalStreamComparator(CalibratedStreamComparator):
    """Temporal comparison over calling rhythm."""

    def __init__(
        self,
        comparator: TemporalComparator,
        calibrator: Calibrator,
        scorer_id: str = "temporal-v1",
        residual_spread: float = 0.0,
    ) -> None:
        super().__init__(EvidenceStream.TEMPORAL, calibrator, scorer_id, residual_spread)
        self._comparator = comparator

    def _score(
        self, first: EvidenceBundle, second: EvidenceBundle
    ) -> tuple[float, Mapping[str, float]]:
        left: TemporalProfile = first.payload(EvidenceStream.TEMPORAL)
        right: TemporalProfile = second.payload(EvidenceStream.TEMPORAL)
        return self._comparator.score(left, right), self._comparator.diagnostics(
            left, right
        )


class TransactionalStreamComparator(CalibratedStreamComparator):
    """Transactional comparison over mobile-money behaviour."""

    def __init__(
        self,
        comparator: TransactionalComparator,
        calibrator: Calibrator,
        scorer_id: str = "transactional-v1",
        residual_spread: float = 0.0,
    ) -> None:
        super().__init__(
            EvidenceStream.TRANSACTIONAL, calibrator, scorer_id, residual_spread
        )
        self._comparator = comparator

    def _score(
        self, first: EvidenceBundle, second: EvidenceBundle
    ) -> tuple[float, Mapping[str, float]]:
        left: TransactionalProfile = first.payload(EvidenceStream.TRANSACTIONAL)
        right: TransactionalProfile = second.payload(EvidenceStream.TRANSACTIONAL)
        return self._comparator.score(left, right), self._comparator.diagnostics(
            left, right
        )


class DeviceStreamComparator(CalibratedStreamComparator):
    """Device and network identifier comparison."""

    def __init__(
        self,
        comparator: DeviceComparator,
        calibrator: Calibrator,
        scorer_id: str = "device-v1",
        residual_spread: float = 0.0,
    ) -> None:
        super().__init__(EvidenceStream.DEVICE, calibrator, scorer_id, residual_spread)
        self._comparator = comparator

    def _score(
        self, first: EvidenceBundle, second: EvidenceBundle
    ) -> tuple[float, Mapping[str, float]]:
        left: DeviceObservation = first.payload(EvidenceStream.DEVICE)
        right: DeviceObservation = second.payload(EvidenceStream.DEVICE)
        return self._comparator.score(left, right), self._comparator.diagnostics(
            left, right
        )
