"""The validity gate: admissibility of acoustic evidence.

The gate sits between the countermeasure and fusion. It converts a detector
score into an admissibility verdict, and it is the only place in the system
where that conversion happens — so the operating point is one decision, made
once, recorded, and auditable, rather than a threshold repeated at several call
sites that drift apart.

Three verdicts, and the middle one matters
------------------------------------------
``ADMITTED``
    The recording is confidently genuine. Acoustic evidence proceeds to fusion.

``EXCLUDED``
    The recording is confidently synthetic or converted. Acoustic evidence is
    **removed** from fusion, not down-weighted. A synthesised utterance carries
    no information about a human vocal tract, so there is no small amount of it
    to include.

``INDETERMINATE``
    Everything else — a score in the uncertain band, or a recording the detector
    has no basis to judge because it is unlike anything it was trained on.

``INDETERMINATE`` does not admit acoustic evidence. That is a policy choice and
it rests on an asymmetry worth stating: admitting synthetic speech puts
fabricated evidence in front of a court, while excluding genuine speech only
weakens a case that other streams may still support. Those costs are not
comparable, so the threshold is not placed at the point that minimises total
error.

The operating point is a policy decision
----------------------------------------
Not an engineering one. It is expressed as a configuration object that is stored
with every verdict, so that a verdict can be re-derived if policy changes and so
that "why was this recording excluded" has an answer that does not require
re-running the detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.spoof.countermeasure import (
    CountermeasureScore,
    SpoofingCountermeasure,
)
from viflap.domain.errors import InvalidEvidenceError
from viflap.domain.evidence import ValidityAssessment, ValidityVerdict

__all__ = ["GatePolicy", "ValidityGate"]


@dataclass(frozen=True, slots=True)
class GatePolicy:
    """The operating point, as an explicit and auditable policy.

    Thresholds are on the countermeasure's calibrated log-likelihood ratio of
    genuine against spoofed, so they read directly as strengths of evidence: an
    ``admit_above`` of 2.3 means "admit only when the recording is about ten
    times more probable under the genuine model".
    """

    admit_above: float = 2.3
    """Score above which a recording is confidently genuine."""

    exclude_below: float = -2.3
    """Score below which a recording is confidently synthetic."""

    max_out_of_domain_fraction: float = 0.25
    """Above this, the detector is judging a recording unlike anything it was
    trained on. The verdict is ``INDETERMINATE`` regardless of the score,
    because the score in that regime is an extrapolation. This is the concrete
    handling of the known failure of countermeasures against unseen synthesis
    methods: rather than reporting a confident number, the system says it does
    not know."""

    max_dispersion_ratio: float = 2.5
    """Above this multiple of the detector's typical within-recording dispersion,
    per-frame scores are inconsistent enough to suggest the recording is not
    homogeneous — a synthetic segment spliced into genuine audio, for instance.
    A mean score across such a recording describes neither part of it.

    Expressed as a ratio rather than an absolute dispersion. The absolute figure
    depends on the feature dimension and the calibration slope, neither of which
    says anything about the recording, so a threshold on it is a threshold on
    the configuration rather than on the evidence."""

    def __post_init__(self) -> None:
        if self.admit_above <= self.exclude_below:
            raise InvalidEvidenceError(
                "the admit threshold must lie above the exclude threshold; "
                "otherwise a single score could be both",
                admit_above=self.admit_above,
                exclude_below=self.exclude_below,
            )

    @property
    def uncertainty_band(self) -> tuple[float, float]:
        return (self.exclude_below, self.admit_above)

    def describe(self) -> str:
        """Statement of the operating point for the audit record."""
        return (
            f"Admit above {self.admit_above:+.2f}, exclude below "
            f"{self.exclude_below:+.2f} (countermeasure log-LR, genuine against "
            f"spoofed); indeterminate between, or where more than "
            f"{self.max_out_of_domain_fraction:.0%} of frames fall outside the "
            f"detector's training domain."
        )

    @classmethod
    def conservative(cls) -> GatePolicy:
        """A wider uncertainty band, for deployment where evidence reaches court.

        Widening the band converts confident errors into acknowledged
        uncertainty. It costs cases; it does not cost anyone their liberty on
        the strength of a vocoder.
        """
        return cls(admit_above=4.6, exclude_below=-1.0)


class ValidityGate:
    """Applies a policy to a countermeasure score."""

    def __init__(
        self, countermeasure: SpoofingCountermeasure, policy: GatePolicy | None = None
    ) -> None:
        self._countermeasure = countermeasure
        self._policy = policy or GatePolicy()

    @property
    def policy(self) -> GatePolicy:
        return self._policy

    @property
    def detector_id(self) -> str:
        return self._countermeasure.detector_id

    def assess(
        self, recording_id: str, signal: NDArray[np.float64], sample_rate: int
    ) -> ValidityAssessment:
        """Produce an admissibility verdict for one recording."""
        score = self._countermeasure.score(signal, sample_rate)
        return self.assess_score(recording_id, score)

    def assess_score(
        self, recording_id: str, score: CountermeasureScore
    ) -> ValidityAssessment:
        """Apply the policy to an already-computed score.

        Separated from :meth:`assess` so that a stored score can be re-judged
        under a revised policy without re-running the detector — which matters
        because the policy is expected to change as the threat landscape does,
        and re-deriving historical verdicts is otherwise impossible.
        """
        verdict = self._verdict(score)
        return ValidityAssessment(
            recording_id=recording_id,
            verdict=verdict,
            countermeasure_log_lr=score.log_likelihood_ratio,
            threshold=self._policy.admit_above,
            detector_id=score.detector_id,
        )

    def _verdict(self, score: CountermeasureScore) -> ValidityVerdict:
        # Domain checks first. A score from outside the detector's training
        # domain is not evidence of anything, so it is not compared against a
        # threshold at all — comparing it would give the extrapolation the same
        # standing as a measurement.
        if score.out_of_domain_fraction > self._policy.max_out_of_domain_fraction:
            return ValidityVerdict.INDETERMINATE
        if score.dispersion_ratio > self._policy.max_dispersion_ratio:
            return ValidityVerdict.INDETERMINATE

        if score.log_likelihood_ratio >= self._policy.admit_above:
            return ValidityVerdict.ADMITTED
        if score.log_likelihood_ratio <= self._policy.exclude_below:
            return ValidityVerdict.EXCLUDED
        return ValidityVerdict.INDETERMINATE

    def explain(self, assessment: ValidityAssessment) -> str:
        """Operator-facing explanation of a verdict.

        Passed through the output-language policy by the caller like any other
        text that leaves the system.
        """
        if assessment.verdict is ValidityVerdict.EXCLUDED:
            return (
                f"Acoustic evidence for recording {assessment.recording_id} was "
                f"excluded from fusion. The recording is judged to be "
                f"synthesised or voice-converted, and such a recording carries "
                f"no information about a human speaker's vocal anatomy. It was "
                f"removed entirely rather than given reduced weight."
            )
        if assessment.verdict is ValidityVerdict.INDETERMINATE:
            return (
                f"Acoustic evidence for recording {assessment.recording_id} was "
                f"withheld. The detector could not determine whether the "
                f"recording is genuine, either because the score falls within "
                f"the uncertainty band or because the recording is unlike the "
                f"material the detector was trained on. Withholding is the "
                f"conservative action and the recording should be assessed by a "
                f"person."
            )
        return (
            f"Acoustic evidence for recording {assessment.recording_id} was "
            f"admitted to fusion. Operating point: {self._policy.describe()}"
        )
