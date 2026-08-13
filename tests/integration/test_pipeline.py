"""End-to-end: enrolment, comparison, search, audit and retention.

Exercises the real use cases against in-memory adapters. The stream comparators
are minimal but *genuinely calibrated* — the point of these tests is the
orchestration and the governance, and substituting an uncalibrated comparator
would test a system that the architecture forbids.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from viflap.analysis.calibration.calibrators import LogisticCalibrator
from viflap.analysis.fusion.base import FusionObservation, FusionTrainingSet
from viflap.analysis.fusion.models import GaussianLatentFusion
from viflap.application.comparison import CompareIncidents, ComparisonRequest
from viflap.application.ingestion import IngestIncident, IngestionRequest
from viflap.application.ports import EvidenceBundle
from viflap.application.search import SearchDatabase, SearchRequest
from viflap.domain.errors import (
    AuthorityViolation,
    GovernanceViolation,
    InsufficientDataError,
    InvalidEvidenceError,
)
from viflap.domain.evidence import (
    AbsenceReason,
    EvidenceStream,
    StreamAbsent,
    ValidityAssessment,
    ValidityVerdict,
)
from viflap.domain.governance import AnalystRole, OutputLanguagePolicy, Principal
from viflap.domain.hypotheses import PriorOdds
from viflap.domain.linkage import IncidentId, IncidentPair
from viflap.infrastructure.audit import FileAuditLog
from viflap.infrastructure.comparators import CalibratedStreamComparator
from viflap.infrastructure.fusion_provider import FittedFusionProvider
from viflap.infrastructure.memory_repositories import InMemoryUnitOfWork
from viflap.infrastructure.retention import RetentionSchedule, RetentionSweep

STREAMS = [
    EvidenceStream.ACOUSTIC,
    EvidenceStream.BEHAVIOURAL,
    EvidenceStream.TEMPORAL,
]


class SignatureComparator(CalibratedStreamComparator):
    """Compares a scalar per-incident signature. Real calibration, trivial scorer."""

    def _score(self, first, second):
        return float(
            -abs(first.payload(self.stream) - second.payload(self.stream)) + 2.0
        ), {}


def _fitted_calibrator(seed: int) -> LogisticCalibrator:
    gen = np.random.default_rng(seed)
    scores, labels = [], []
    for _ in range(500):
        same = gen.random() < 0.5
        a = gen.normal(0.0, 1.0)
        b = a + gen.normal(0.0, 0.4 if same else 2.5)
        scores.append(float(-abs(a - b) + 2.0))
        labels.append(1 if same else 0)
    return LogisticCalibrator().fit(np.array(scores), np.array(labels))


def _fitted_fusion() -> FittedFusionProvider:
    gen = np.random.default_rng(9)
    observations = []
    for _ in range(1200):
        same = gen.random() < 0.5
        common = gen.standard_normal()
        values = {}
        for index, stream in enumerate(STREAMS):
            error = math.sqrt(0.6) * common + math.sqrt(0.4) * gen.standard_normal()
            values[stream] = float(
                (0.8 + 0.1 * index) * ((1.4 if same else -1.4) + error) * 2.0
            )
        observations.append(FusionObservation(values, same, group_id="g"))
    return FittedFusionProvider(
        GaussianLatentFusion().fit(FusionTrainingSet(observations)),
        stream_spreads=dict.fromkeys(STREAMS, 0.8),
    )


@pytest.fixture
def system(tmp_path: Path, clock):
    """A fully assembled system with four actors and four incidents each."""
    unit_of_work = InMemoryUnitOfWork()
    audit = FileAuditLog(tmp_path / "audit.jsonl")

    comparators = [
        SignatureComparator(
            stream, _fitted_calibrator(100 + index), f"sig-{stream.value}", 0.8
        )
        for index, stream in enumerate(STREAMS)
    ]
    compare = CompareIncidents(comparators, _fitted_fusion(), audit, clock)
    search = SearchDatabase(
        compare, unit_of_work.incidents, unit_of_work.evidence, audit, clock
    )
    ingest = IngestIncident(
        extractors={
            stream: (lambda request, i=index: request.metadata["signatures"][i])
            for index, stream in enumerate(STREAMS)
        },
        unit_of_work=unit_of_work,
        audit=audit,
        clock=clock,
    )
    return {
        "uow": unit_of_work,
        "audit": audit,
        "compare": compare,
        "search": search,
        "ingest": ingest,
        "clock": clock,
    }


@pytest.fixture
def enrolled(system, case_reference, enrolment_officer):
    """Enrol sixteen incidents from four actors."""
    gen = np.random.default_rng(21)
    incidents: list[tuple[IncidentId, int]] = []
    for actor in range(4):
        for repetition in range(4):
            incident_id = IncidentId(f"ZP-2025-{actor:02d}{repetition:03d}")
            signatures = [
                float(actor * 3.0 + gen.normal(0.0, 0.35) + index * 0.01)
                for index in range(len(STREAMS))
            ]
            system["ingest"].execute(
                IngestionRequest(
                    incident_id=incident_id,
                    case_reference=case_reference,
                    submitted_by=enrolment_officer,
                    metadata={"signatures": signatures, "actor": actor},
                )
            )
            incidents.append((incident_id, actor))
    return incidents


class TestGovernance:
    def test_separation_of_duties_blocks_incompatible_roles(self) -> None:
        for combination in (
            {AnalystRole.ENROLMENT_OFFICER, AnalystRole.INVESTIGATOR},
            {AnalystRole.INVESTIGATOR, AnalystRole.DISCLOSURE_OFFICER},
            {AnalystRole.OVERSIGHT_AUDITOR, AnalystRole.INVESTIGATOR},
        ):
            with pytest.raises(Exception, match="incompatible authorities"):
                Principal("x", frozenset(combination))

    def test_investigator_cannot_enrol(self, system, case_reference, investigator) -> None:
        with pytest.raises(AuthorityViolation):
            system["ingest"].execute(
                IngestionRequest(
                    incident_id=IncidentId("ZP-2025-99999"),
                    case_reference=case_reference,
                    submitted_by=investigator,
                    metadata={"signatures": [0.0, 0.0, 0.0]},
                )
            )

    def test_enroller_cannot_query(
        self, system, enrolled, case_reference, enrolment_officer
    ) -> None:
        first, second = enrolled[0][0], enrolled[1][0]
        with pytest.raises(AuthorityViolation):
            system["compare"].execute(
                ComparisonRequest(
                    pair=IncidentPair(first, second, case_reference),
                    prior=PriorOdds.uniform_over_database(16),
                    requested_by=enrolment_officer,
                ),
                system["uow"].store.evidence[first.value],
                system["uow"].store.evidence[second.value],
            )

    def test_re_enrolment_is_refused(
        self, system, enrolled, case_reference, enrolment_officer
    ) -> None:
        """Replacing evidence would make earlier results unreproducible."""
        with pytest.raises(GovernanceViolation, match="already enrolled"):
            system["ingest"].execute(
                IngestionRequest(
                    incident_id=enrolled[0][0],
                    case_reference=case_reference,
                    submitted_by=enrolment_officer,
                    metadata={"signatures": [0.0, 0.0, 0.0]},
                )
            )


class TestComparison:
    def test_same_actor_outranks_different_actors(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        def compare(a: IncidentId, b: IncidentId) -> float:
            result = system["compare"].execute(
                ComparisonRequest(
                    pair=IncidentPair(a, b, case_reference),
                    prior=PriorOdds.uniform_over_database(16),
                    requested_by=investigator,
                ),
                system["uow"].store.evidence[a.value],
                system["uow"].store.evidence[b.value],
            )
            return result.fused_log_lr.log10

        same_actor = compare(enrolled[0][0], enrolled[1][0])
        different_actor = compare(enrolled[0][0], enrolled[12][0])
        assert same_actor > different_actor

    def test_absent_stream_is_not_a_neutral_stream(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        """Substituting LR = 1 would assert an observation that was never made."""
        first, second = enrolled[0][0], enrolled[1][0]
        full = system["uow"].store.evidence[second.value]
        partial = EvidenceBundle(
            incident_id=second,
            payloads={EvidenceStream.ACOUSTIC: full.payload(EvidenceStream.ACOUSTIC)},
        )
        result = system["compare"].execute(
            ComparisonRequest(
                pair=IncidentPair(first, second, case_reference),
                prior=PriorOdds.uniform_over_database(16),
                requested_by=investigator,
            ),
            system["uow"].store.evidence[first.value],
            partial,
        )
        assert result.rests_on_single_stream
        assert result.overstatement is None
        for stream in (EvidenceStream.BEHAVIOURAL, EvidenceStream.TEMPORAL):
            outcome = result.outcomes[stream]
            assert isinstance(outcome, StreamAbsent)
            assert outcome.reason is AbsenceReason.NO_DATA

    def test_validity_gate_removes_the_acoustic_stream(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        first, second = enrolled[0][0], enrolled[1][0]
        gated = EvidenceBundle(
            incident_id=second,
            payloads=dict(system["uow"].store.evidence[second.value].payloads),
            validity=ValidityAssessment(
                recording_id="REC-991",
                verdict=ValidityVerdict.EXCLUDED,
                countermeasure_log_lr=-8.4,
                threshold=2.3,
                detector_id="lfcc-gmm-test",
            ),
        )
        result = system["compare"].execute(
            ComparisonRequest(
                pair=IncidentPair(first, second, case_reference),
                prior=PriorOdds.uniform_over_database(16),
                requested_by=investigator,
            ),
            system["uow"].store.evidence[first.value],
            gated,
        )
        assert result.acoustic_was_excluded
        outcome = result.outcomes[EvidenceStream.ACOUSTIC]
        assert isinstance(outcome, StreamAbsent)
        assert outcome.reason is AbsenceReason.EXCLUDED_BY_VALIDITY_GATE

    def test_indeterminate_verdict_also_withholds_acoustic_evidence(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        """ "Cannot tell" does not admit. The asymmetry is deliberate."""
        first, second = enrolled[0][0], enrolled[1][0]
        gated = EvidenceBundle(
            incident_id=second,
            payloads=dict(system["uow"].store.evidence[second.value].payloads),
            validity=ValidityAssessment(
                recording_id="REC-992",
                verdict=ValidityVerdict.INDETERMINATE,
                countermeasure_log_lr=0.4,
                threshold=2.3,
                detector_id="lfcc-gmm-test",
            ),
        )
        result = system["compare"].execute(
            ComparisonRequest(
                pair=IncidentPair(first, second, case_reference),
                prior=PriorOdds.uniform_over_database(16),
                requested_by=investigator,
            ),
            system["uow"].store.evidence[first.value],
            gated,
        )
        assert isinstance(result.outcomes[EvidenceStream.ACOUSTIC], StreamAbsent)

    def test_declines_when_no_stream_produces_evidence(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        """A neutral result would be indistinguishable from a real finding."""
        first = enrolled[0][0]
        empty = EvidenceBundle(incident_id=enrolled[1][0], payloads={})
        with pytest.raises(InsufficientDataError):
            system["compare"].execute(
                ComparisonRequest(
                    pair=IncidentPair(first, enrolled[1][0], case_reference),
                    prior=PriorOdds.uniform_over_database(16),
                    requested_by=investigator,
                ),
                system["uow"].store.evidence[first.value],
                empty,
            )

    def test_result_carries_its_own_caveats(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        first, second = enrolled[0][0], enrolled[1][0]
        result = system["compare"].execute(
            ComparisonRequest(
                pair=IncidentPair(first, second, case_reference),
                prior=PriorOdds.uniform_over_database(16),
                requested_by=investigator,
            ),
            system["uow"].store.evidence[first.value],
            system["uow"].store.evidence[second.value],
        )
        caveats = result.caveats()
        assert any("uniform over the searched population" in c for c in caveats)
        for caveat in caveats:
            OutputLanguagePolicy.assert_permitted(caveat)

    def test_verbal_summary_states_direction_with_strength(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        """A band without its direction inverts the finding while sounding sure."""
        first, second = enrolled[0][0], enrolled[12][0]
        result = system["compare"].execute(
            ComparisonRequest(
                pair=IncidentPair(first, second, case_reference),
                prior=PriorOdds.uniform_over_database(16),
                requested_by=investigator,
            ),
            system["uow"].store.evidence[first.value],
            system["uow"].store.evidence[second.value],
        )
        summary = result.verbal_summary()
        assert "different actors" in summary
        OutputLanguagePolicy.assert_permitted(summary)


class TestSearch:
    def test_ranks_same_actor_incidents_first(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        probe, probe_actor = enrolled[0]
        results = system["search"].execute(
            SearchRequest(probe, case_reference, investigator, max_results=3)
        )
        actor_of = {incident.value: actor for incident, actor in enrolled}
        for result in results.results:
            other = result.pair.second if result.pair.first == probe else result.pair.first
            assert actor_of[other.value] == probe_actor

    def test_prior_comes_from_the_population_not_the_caller(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        """A caller who could set N could set the posterior."""
        results = system["search"].execute(
            SearchRequest(enrolled[0][0], case_reference, investigator, max_results=1)
        )
        assert results.prior.population_size == 16
        assert results.prior.supplied_by == "system"

    def test_restriction_is_attributed_to_the_investigator(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        results = system["search"].execute(
            SearchRequest(
                enrolled[0][0],
                case_reference,
                investigator,
                max_results=1,
                restricted_population=50,
                restriction_justification="Corridor narrowed by cell-site analysis.",
            )
        )
        assert results.prior.supplied_by == investigator.identifier
        assert results.prior.population_size == 50

    def test_restriction_without_justification_is_refused(
        self, enrolled, case_reference, investigator
    ) -> None:
        with pytest.raises(InvalidEvidenceError, match="justification"):
            SearchRequest(
                enrolled[0][0], case_reference, investigator, restricted_population=50
            )

    def test_mandatory_caveat_names_the_search_size_and_prior(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        results = system["search"].execute(
            SearchRequest(enrolled[0][0], case_reference, investigator)
        )
        caveat = results.caveat()
        assert "15 enrolled candidates" in caveat
        OutputLanguagePolicy.assert_permitted(caveat)


class TestAuditTrail:
    def test_chain_verifies_after_a_full_session(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        system["search"].execute(
            SearchRequest(enrolled[0][0], case_reference, investigator)
        )
        verification = system["audit"].verify()
        assert verification.is_intact
        assert verification.n_entries > 16

    def test_refused_operations_are_recorded(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        """A query that was refused is exactly what oversight needs to see."""
        first = enrolled[0][0]
        with pytest.raises(InsufficientDataError):
            system["compare"].execute(
                ComparisonRequest(
                    pair=IncidentPair(first, enrolled[1][0], case_reference),
                    prior=PriorOdds.uniform_over_database(16),
                    requested_by=investigator,
                ),
                system["uow"].store.evidence[first.value],
                EvidenceBundle(incident_id=enrolled[1][0], payloads={}),
            )
        failures = [e for e in system["audit"].query() if e.outcome != "ok"]
        assert failures

    def test_prior_is_recorded_with_every_query(
        self, system, enrolled, case_reference, investigator
    ) -> None:
        """Reconstructing a historical result's prior is otherwise guesswork."""
        system["search"].execute(
            SearchRequest(enrolled[0][0], case_reference, investigator, max_results=1)
        )
        comparisons = [
            e for e in system["audit"].query() if e.action == "compare_incidents"
        ]
        assert comparisons
        assert "prior_log_odds" in comparisons[0].parameters
        assert "prior_basis" in comparisons[0].parameters

    def test_tampering_is_detected_at_the_altered_entry(
        self, system, enrolled, tmp_path
    ) -> None:
        path = system["audit"].path
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[2] = lines[2].replace('"outcome":"ok"', '"outcome":"tampered"')
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        verification = FileAuditLog(path).verify()
        assert not verification.is_intact
        assert verification.first_broken_index == 2

    def test_removing_an_entry_is_detected(self, system, enrolled) -> None:
        """Deletion breaks the sequence as well as the hash chain."""
        path = system["audit"].path
        lines = path.read_text(encoding="utf-8").splitlines()
        del lines[3]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        verification = FileAuditLog(path).verify()
        assert not verification.is_intact


class TestRetention:
    def test_expired_data_is_deleted_and_the_deletion_audited(
        self, system, enrolled, tmp_path
    ) -> None:
        audit = FileAuditLog(tmp_path / "retention-audit.jsonl")
        system["clock"].set(datetime(2027, 6, 1, tzinfo=UTC))
        sweep = RetentionSweep(
            RetentionSchedule.default(), system["uow"], audit, system["clock"]
        )
        report = sweep.run()

        assert report.expired == 16
        assert report.deleted == 16
        assert system["uow"].incidents.count() == 0

        deletions = [e for e in audit.query() if e.action == "retention_delete"]
        assert len(deletions) == 16
        assert deletions[0].parameters["legal_basis"]

    def test_data_within_its_retention_period_is_untouched(
        self, system, enrolled, tmp_path
    ) -> None:
        audit = FileAuditLog(tmp_path / "retention-audit.jsonl")
        system["clock"].advance(days=30)
        report = RetentionSweep(
            RetentionSchedule.default(), system["uow"], audit, system["clock"]
        ).run()
        assert report.deleted == 0
        assert system["uow"].incidents.count() == 16
