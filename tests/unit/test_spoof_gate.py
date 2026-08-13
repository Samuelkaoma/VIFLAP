"""The validity gate: when acoustic evidence is allowed to count.

The gate decides whether a recording's acoustic evidence reaches fusion at all.
Its verdict *removes* the stream rather than down-weighting it, on the reasoning
that a synthesised utterance carries no information about a human vocal tract —
so there is no small amount of it to include.

Two properties carry the safety of this module and both are tested here.

**Domain checks precede thresholds.** A score computed on a recording unlike
anything the detector was trained on is an extrapolation, not a measurement, and
must not be compared against a threshold at all. This is the concrete handling
of the known failure of spoofing countermeasures against synthesis methods they
have never seen: the honest output is "I do not know", and the dangerous output
is a confident ADMITTED on a deepfake produced by a vocoder released after
training.

**The uncertain band is a verdict, not a gap.** INDETERMINATE is a first-class
outcome. A gate with only two verdicts would have to resolve every borderline
recording as genuine or synthetic, and would be wrong in the direction of
whichever error the threshold happened to favour.
"""

from __future__ import annotations

import pytest

from viflap.analysis.spoof.countermeasure import CountermeasureScore
from viflap.analysis.spoof.gate import GatePolicy, ValidityGate
from viflap.domain.errors import InvalidEvidenceError
from viflap.domain.evidence import ValidityVerdict
from viflap.domain.governance import OutputLanguagePolicy

DETECTOR = "cm-test-0001"


class _StubCountermeasure:
    """Stands in for the detector. The gate under test only applies a policy."""

    detector_id = DETECTOR


def _score(
    log_lr: float,
    out_of_domain: float = 0.0,
    dispersion: float = 1.0,
) -> CountermeasureScore:
    return CountermeasureScore(
        log_likelihood_ratio=log_lr,
        raw_score=log_lr,
        n_frames=500,
        out_of_domain_fraction=out_of_domain,
        frame_score_std=1.0,
        dispersion_ratio=dispersion,
        detector_id=DETECTOR,
    )


@pytest.fixture
def gate() -> ValidityGate:
    return ValidityGate(_StubCountermeasure())  # type: ignore[arg-type]


class TestGatePolicy:
    def test_thresholds_must_not_cross(self) -> None:
        """Otherwise one score could be both admitted and excluded."""
        with pytest.raises(InvalidEvidenceError):
            GatePolicy(admit_above=-1.0, exclude_below=1.0)

    def test_equal_thresholds_are_refused(self) -> None:
        """Equal thresholds leave no uncertain band, which is the point of it."""
        with pytest.raises(InvalidEvidenceError):
            GatePolicy(admit_above=2.0, exclude_below=2.0)

    def test_the_uncertainty_band_is_reported_low_to_high(self) -> None:
        assert GatePolicy().uncertainty_band == (-2.3, 2.3)

    def test_the_operating_point_is_stateable_for_the_audit_record(self) -> None:
        described = GatePolicy().describe()
        assert "+2.30" in described
        assert "-2.30" in described

    def test_the_conservative_policy_admits_less_readily(self) -> None:
        """For deployment where the evidence reaches a court."""
        default = GatePolicy()
        conservative = GatePolicy.conservative()

        assert conservative.admit_above > default.admit_above
        assert conservative.exclude_below > default.exclude_below

    def test_the_conservative_band_is_wider(self) -> None:
        """It converts confident errors into acknowledged uncertainty."""
        low, high = GatePolicy.conservative().uncertainty_band
        default_low, default_high = GatePolicy().uncertainty_band
        assert (high - low) > (default_high - default_low)


class TestVerdicts:
    def test_a_confidently_genuine_recording_is_admitted(self, gate: ValidityGate) -> None:
        assessment = gate.assess_score("rec-1", _score(5.0))
        assert assessment.verdict is ValidityVerdict.ADMITTED

    def test_a_confidently_synthetic_recording_is_excluded(
        self, gate: ValidityGate
    ) -> None:
        assessment = gate.assess_score("rec-2", _score(-5.0))
        assert assessment.verdict is ValidityVerdict.EXCLUDED

    def test_a_score_in_the_band_is_indeterminate(self, gate: ValidityGate) -> None:
        assessment = gate.assess_score("rec-3", _score(0.0))
        assert assessment.verdict is ValidityVerdict.INDETERMINATE

    @pytest.mark.parametrize(
        ("log_lr", "expected"),
        [
            (2.3, ValidityVerdict.ADMITTED),
            (2.2999, ValidityVerdict.INDETERMINATE),
            (-2.3, ValidityVerdict.EXCLUDED),
            (-2.2999, ValidityVerdict.INDETERMINATE),
        ],
    )
    def test_the_boundaries_are_inclusive_on_the_confident_side(
        self, gate: ValidityGate, log_lr: float, expected: ValidityVerdict
    ) -> None:
        """Pinned because a flipped comparison here is invisible in every other
        test: the verdict is still one of three, just the wrong one, and only
        for scores sitting exactly on a threshold."""
        assert gate.assess_score("rec", _score(log_lr)).verdict is expected

    def test_the_assessment_records_what_produced_it(self, gate: ValidityGate) -> None:
        assessment = gate.assess_score("rec-4", _score(5.0))

        assert assessment.recording_id == "rec-4"
        assert assessment.detector_id == DETECTOR
        assert assessment.countermeasure_log_lr == 5.0
        assert assessment.threshold == GatePolicy().admit_above


class TestDomainChecksPrecedeThresholds:
    """The failure mode this module exists to prevent."""

    def test_an_out_of_domain_recording_is_not_admitted_however_high_the_score(
        self, gate: ValidityGate
    ) -> None:
        """A deepfake from a vocoder released after training scores high and
        means nothing. Admitting it is the worst available outcome."""
        assessment = gate.assess_score("rec-5", _score(50.0, out_of_domain=0.9))
        assert assessment.verdict is ValidityVerdict.INDETERMINATE

    def test_an_out_of_domain_recording_is_not_excluded_either(
        self, gate: ValidityGate
    ) -> None:
        """Symmetry matters. Not knowing is not evidence of forgery, and a
        wrongly excluded genuine recording removes real evidence from a case."""
        assessment = gate.assess_score("rec-6", _score(-50.0, out_of_domain=0.9))
        assert assessment.verdict is ValidityVerdict.INDETERMINATE

    def test_a_heterogeneous_recording_is_indeterminate(self, gate: ValidityGate) -> None:
        """A synthetic segment spliced into genuine audio. The mean score
        describes neither part of it."""
        assessment = gate.assess_score("rec-7", _score(50.0, dispersion=9.0))
        assert assessment.verdict is ValidityVerdict.INDETERMINATE

    def test_domain_thresholds_are_exclusive(self, gate: ValidityGate) -> None:
        """Exactly at the limit is still in domain, so the score is judged."""
        at_limit = gate.assess_score("rec-8", _score(5.0, out_of_domain=0.25))
        just_over = gate.assess_score("rec-9", _score(5.0, out_of_domain=0.2501))

        assert at_limit.verdict is ValidityVerdict.ADMITTED
        assert just_over.verdict is ValidityVerdict.INDETERMINATE


class TestRejudgingStoredScores:
    def test_a_stored_score_can_be_rejudged_under_a_revised_policy(self) -> None:
        """The policy is expected to change as the threat landscape does, and
        re-deriving historical verdicts would otherwise be impossible."""
        score = _score(3.0)

        default = ValidityGate(_StubCountermeasure())  # type: ignore[arg-type]
        conservative = ValidityGate(
            _StubCountermeasure(),  # type: ignore[arg-type]
            GatePolicy.conservative(),
        )

        assert default.assess_score("rec", score).verdict is ValidityVerdict.ADMITTED
        assert (
            conservative.assess_score("rec", score).verdict is ValidityVerdict.INDETERMINATE
        )

    def test_the_policy_in_force_is_readable(self) -> None:
        gate = ValidityGate(
            _StubCountermeasure(),  # type: ignore[arg-type]
            GatePolicy.conservative(),
        )
        assert gate.policy.admit_above == 4.6


class TestExplanationsSatisfyTheOutputPolicy:
    """Anything the gate says to an operator still crosses the system boundary.

    The output-language policy forbids the vocabulary of identity everywhere,
    and a module that explains itself in prose is exactly where a phrase like
    "identified as synthetic" slips in. These would raise if it did.
    """

    @pytest.mark.parametrize(
        ("log_lr", "out_of_domain"),
        [(5.0, 0.0), (-5.0, 0.0), (0.0, 0.0), (5.0, 0.9)],
    )
    def test_every_verdict_explains_itself_within_the_policy(
        self, gate: ValidityGate, log_lr: float, out_of_domain: float
    ) -> None:
        assessment = gate.assess_score("rec-10", _score(log_lr, out_of_domain))
        OutputLanguagePolicy.assert_permitted(
            gate.explain(assessment), origin="spoof.gate.explain"
        )

    def test_the_exclusion_explanation_says_removed_not_down_weighted(
        self, gate: ValidityGate
    ) -> None:
        """The distinction is the module's substantive claim, so the prose has
        to carry it — an operator who reads "reduced weight" will assume some
        acoustic evidence survived, and none did."""
        assessment = gate.assess_score("rec-11", _score(-5.0))
        explanation = gate.explain(assessment).lower()

        assert "removed" in explanation
        assert "reduced weight" not in explanation.replace(
            "rather than given reduced weight", ""
        )
