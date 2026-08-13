"""Domain value objects: the invariants that make invalid states unrepresentable."""

from __future__ import annotations

import math

import pytest

from viflap.domain.errors import InvalidEvidenceError, MetricInvariantError
from viflap.domain.evidence import (
    AbsenceReason,
    EvidenceStream,
    StreamAbsent,
    StreamEvidence,
    absent_streams,
    missingness_pattern,
    present_evidence,
)
from viflap.domain.hypotheses import (
    PosteriorAssessment,
    PriorBasis,
    PriorOdds,
    SearchMode,
)
from viflap.domain.metrics import CalibrationSummary, PerformanceGrade
from viflap.domain.values import (
    EvidentialStrength,
    LikelihoodRatio,
    LogLikelihoodRatio,
    LogOdds,
    Probability,
    UncertaintyInterval,
    logistic,
)


class TestLogLikelihoodRatio:
    def test_rejects_non_finite(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with pytest.raises(InvalidEvidenceError):
                LogLikelihoodRatio(value)

    def test_neutral_is_exactly_zero(self) -> None:
        assert LogLikelihoodRatio.neutral().value == 0.0

    @pytest.mark.parametrize("log10_lr", [-6.0, -1.0, 0.0, 1.0, 3.0, 8.0])
    def test_base_conversions_round_trip(self, log10_lr: float) -> None:
        llr = LogLikelihoodRatio.from_log10(log10_lr)
        assert llr.log10 == pytest.approx(log10_lr)
        assert llr.log2 == pytest.approx(log10_lr * math.log2(10.0))

    def test_extreme_value_refuses_linear_conversion(self) -> None:
        """A log-LR beyond float range is a calibration failure, not strong evidence.

        Saturating to the largest float would present the failure as an
        extremely strong result, which is the specific misreading this refusal
        prevents.
        """
        extreme = LogLikelihoodRatio(2000.0)
        assert not extreme.is_representable_as_ratio
        with pytest.raises(InvalidEvidenceError, match="calibration failure"):
            extreme.to_likelihood_ratio()
        # The log-scale value remains exact and usable.
        assert extreme.log10 == pytest.approx(868.588963, abs=1e-4)

    def test_negation_inverts_the_propositions(self) -> None:
        llr = LogLikelihoodRatio.from_log10(3.0)
        assert (-llr).log10 == pytest.approx(-3.0)

    @pytest.mark.parametrize(
        "log10_lr,expected",
        [
            (0.0, EvidentialStrength.NO_SUPPORT),
            (0.5, EvidentialStrength.WEAK),
            (1.5, EvidentialStrength.MODERATE),
            (2.5, EvidentialStrength.MODERATELY_STRONG),
            (3.5, EvidentialStrength.STRONG),
            (5.0, EvidentialStrength.VERY_STRONG),
            (9.0, EvidentialStrength.EXTREMELY_STRONG),
        ],
    )
    def test_strength_bands(self, log10_lr: float, expected: EvidentialStrength) -> None:
        assert LogLikelihoodRatio.from_log10(log10_lr).strength is expected

    def test_strength_is_symmetric_in_direction(self) -> None:
        """The band depends on magnitude; direction is reported separately.

        This is why rendering a band without its direction inverts the finding,
        and why the domain provides no way to render one alone.
        """
        for magnitude in (0.5, 2.0, 5.0):
            assert (
                LogLikelihoodRatio.from_log10(magnitude).strength
                is LogLikelihoodRatio.from_log10(-magnitude).strength
            )


class TestLikelihoodRatio:
    @pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_impossible_values(self, value: float) -> None:
        with pytest.raises(InvalidEvidenceError):
            LikelihoodRatio(value)

    def test_inversion(self) -> None:
        assert LikelihoodRatio(100.0).inverted().value == pytest.approx(0.01)


class TestLogOdds:
    def test_bayes_rule_is_addition(self) -> None:
        prior = LogOdds(-11.5)
        posterior = prior.updated_with(LogLikelihoodRatio(6.9))
        assert posterior.value == pytest.approx(-4.6)

    def test_probability_exact_in_both_tails(self) -> None:
        """The reason posteriors are computed in log-odds.

        The linear form underflows for small priors and overflows for strong
        evidence, which is precisely the regime a national-scale search occupies.
        """
        assert LogOdds(-40.0).to_probability().value == pytest.approx(
            logistic(-40.0), rel=1e-12
        )
        assert LogOdds(-40.0).to_probability().value > 0.0
        assert LogOdds(800.0).to_probability().value == 1.0

    def test_certainty_has_no_log_odds(self) -> None:
        for value in (0.0, 1.0):
            with pytest.raises(InvalidEvidenceError):
                Probability(value).to_log_odds()


class TestUncertaintyInterval:
    def test_rejects_inverted_interval(self) -> None:
        with pytest.raises(InvalidEvidenceError):
            UncertaintyInterval(lower=2.0, upper=1.0)

    def test_spans_neutral_detects_crossing_zero(self) -> None:
        assert UncertaintyInterval(-1.0, 3.0).spans_neutral
        assert not UncertaintyInterval(1.0, 3.0).spans_neutral

    def test_clipping_respects_bounds(self) -> None:
        clipped = UncertaintyInterval(-10.0, 10.0).clipped_to(-2.0, 2.0)
        assert clipped.lower == -2.0 and clipped.upper == 2.0


class TestPriorOdds:
    def test_uniform_prior_uses_n_minus_one(self) -> None:
        """The questioned incident is not a candidate for linkage with itself."""
        prior = PriorOdds.uniform_over_database(100_000)
        assert prior.log_odds.value == pytest.approx(-math.log(99_999))

    def test_proposal_worked_example_reproduces(self) -> None:
        """Section 6.4: LR = 1,000 against N = 100,000 gives about 1%."""
        prior = PriorOdds.uniform_over_database(100_000)
        posterior = PosteriorAssessment.from_evidence(
            prior, LogLikelihoodRatio.from_log10(3.0)
        )
        assert posterior.probability.percent == pytest.approx(0.99, abs=0.01)
        assert posterior.is_dominated_by_prior

    def test_proposal_fusion_example_reproduces(self) -> None:
        """Section 6.4: fused LR = 5e6 against the same prior gives about 98%."""
        prior = PriorOdds.uniform_over_database(100_000)
        posterior = PosteriorAssessment.from_evidence(
            prior, LogLikelihoodRatio(math.log(5e6))
        )
        assert posterior.probability.percent == pytest.approx(98.0, abs=0.1)

    def test_prior_requires_justification(self) -> None:
        with pytest.raises(InvalidEvidenceError, match="justification"):
            PriorOdds(
                log_odds=LogOdds(-5.0),
                basis=PriorBasis.RESTRICTED_POPULATION,
                search_mode=SearchMode.DATABASE_SEARCH,
                justification="   ",
                population_size=100,
                supplied_by="inv-1",
            )

    def test_only_uniform_prior_may_be_attributed_to_the_system(self) -> None:
        """Any other prior is a judgement and must name whoever made it."""
        with pytest.raises(InvalidEvidenceError, match="attributed to the system"):
            PriorOdds(
                log_odds=LogOdds(-5.0),
                basis=PriorBasis.RESTRICTED_POPULATION,
                search_mode=SearchMode.DATABASE_SEARCH,
                justification="Corridor narrowed by cell-site analysis.",
                population_size=100,
                supplied_by="system",
            )

    def test_posterior_is_derived_not_supplied(self) -> None:
        """Consistency is a property of construction, not something checked."""
        prior = PriorOdds.uniform_over_database(1000)
        llr = LogLikelihoodRatio(4.0)
        posterior = PosteriorAssessment.from_evidence(prior, llr)
        assert posterior.posterior_log_odds.value == pytest.approx(
            prior.log_odds.value + llr.value
        )


class TestStreamOutcomes:
    def _evidence(self, stream: EvidenceStream) -> StreamEvidence:
        return StreamEvidence(
            stream=stream,
            log_lr=LogLikelihoodRatio(2.0),
            uncertainty=UncertaintyInterval(1.0, 3.0),
            model_id="test-model",
        )

    def test_point_estimate_must_lie_in_its_interval(self) -> None:
        with pytest.raises(InvalidEvidenceError, match="outside its own"):
            StreamEvidence(
                stream=EvidenceStream.ACOUSTIC,
                log_lr=LogLikelihoodRatio(9.0),
                uncertainty=UncertaintyInterval(1.0, 3.0),
                model_id="test-model",
            )

    def test_evidence_requires_a_model_identity(self) -> None:
        with pytest.raises(InvalidEvidenceError, match="identify the model"):
            StreamEvidence(
                stream=EvidenceStream.ACOUSTIC,
                log_lr=LogLikelihoodRatio(2.0),
                uncertainty=UncertaintyInterval(1.0, 3.0),
                model_id="",
            )

    def test_absence_and_evidence_are_distinguishable(self) -> None:
        outcomes = {
            EvidenceStream.ACOUSTIC: self._evidence(EvidenceStream.ACOUSTIC),
            EvidenceStream.DEVICE: StreamAbsent(
                EvidenceStream.DEVICE, AbsenceReason.NO_DATA
            ),
        }
        assert set(present_evidence(outcomes)) == {EvidenceStream.ACOUSTIC}
        assert set(absent_streams(outcomes)) == {EvidenceStream.DEVICE}
        assert missingness_pattern(outcomes) == frozenset({EvidenceStream.ACOUSTIC})

    def test_gate_exclusion_is_a_finding(self) -> None:
        assert AbsenceReason.EXCLUDED_BY_VALIDITY_GATE.is_finding
        assert not AbsenceReason.NO_DATA.is_finding

    def test_only_acoustic_is_gated(self) -> None:
        assert EvidenceStream.ACOUSTIC.is_gated_by_validity
        for stream in EvidenceStream.ordered():
            if stream is not EvidenceStream.ACOUSTIC:
                assert not stream.is_gated_by_validity


class TestCalibrationSummary:
    def test_calibration_loss_is_derived(self) -> None:
        summary = CalibrationSummary(
            c_llr=0.5, c_llr_min=0.3, n_same_source=100, n_different_source=500
        )
        assert summary.calibration_loss == pytest.approx(0.2)

    def test_rejects_cllr_below_cllr_min(self) -> None:
        with pytest.raises(MetricInvariantError, match="defect in the metric"):
            CalibrationSummary(
                c_llr=0.2, c_llr_min=0.5, n_same_source=10, n_different_source=10
            )

    def test_requires_both_trial_types(self) -> None:
        with pytest.raises(MetricInvariantError):
            CalibrationSummary(
                c_llr=0.5, c_llr_min=0.3, n_same_source=0, n_different_source=100
            )

    @pytest.mark.parametrize(
        "c_llr,grade",
        [
            (0.3, PerformanceGrade.INFORMATIVE),
            (0.95, PerformanceGrade.MARGINAL),
            (1.0, PerformanceGrade.UNINFORMATIVE),
            (1.4, PerformanceGrade.MISLEADING),
        ],
    )
    def test_grade_boundaries(self, c_llr: float, grade: PerformanceGrade) -> None:
        summary = CalibrationSummary(
            c_llr=c_llr, c_llr_min=min(c_llr, 0.2), n_same_source=10, n_different_source=10
        )
        assert summary.grade is grade
