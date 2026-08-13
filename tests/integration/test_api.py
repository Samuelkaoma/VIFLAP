"""API contract tests.

The properties checked here are the ones a client cannot be trusted to get
right, so the transport layer must guarantee them: no likelihood ratio without
its prior, no strength band without its direction, and a refusal that explains
what to do instead.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from viflap.analysis.calibration.calibrators import LogisticCalibrator
from viflap.analysis.fusion.base import FusionObservation, FusionTrainingSet
from viflap.analysis.fusion.models import GaussianLatentFusion
from viflap.application.comparison import CompareIncidents
from viflap.application.ingestion import IngestIncident, IngestionRequest
from viflap.application.search import SearchDatabase
from viflap.domain.evidence import EvidenceStream
from viflap.domain.governance import AnalystRole, CaseReference, Principal
from viflap.infrastructure.audit import FileAuditLog
from viflap.infrastructure.comparators import CalibratedStreamComparator
from viflap.infrastructure.fusion_provider import FittedFusionProvider
from viflap.infrastructure.memory_repositories import InMemoryUnitOfWork
from viflap.interfaces.api.app import create_app
from viflap.interfaces.api.dependencies import (
    ApplicationContainer,
    HeaderPrincipalResolver,
)

STREAMS = [EvidenceStream.ACOUSTIC, EvidenceStream.BEHAVIOURAL]


class SignatureComparator(CalibratedStreamComparator):
    def _score(self, first, second):
        return float(
            -abs(first.payload(self.stream) - second.payload(self.stream)) + 2.0
        ), {}


def _calibrator(seed: int) -> LogisticCalibrator:
    gen = np.random.default_rng(seed)
    scores, labels = [], []
    for _ in range(400):
        same = gen.random() < 0.5
        a = gen.normal(0.0, 1.0)
        b = a + gen.normal(0.0, 0.4 if same else 2.5)
        scores.append(float(-abs(a - b) + 2.0))
        labels.append(1 if same else 0)
    return LogisticCalibrator().fit(np.array(scores), np.array(labels))


def _fusion() -> FittedFusionProvider:
    gen = np.random.default_rng(3)
    observations = []
    for _ in range(900):
        same = gen.random() < 0.5
        common = gen.standard_normal()
        values = {
            stream: float(
                ((1.4 if same else -1.4) + 0.7 * common + 0.7 * gen.standard_normal()) * 2.0
            )
            for stream in STREAMS
        }
        observations.append(FusionObservation(values, same, group_id="g"))
    return FittedFusionProvider(
        GaussianLatentFusion().fit(FusionTrainingSet(observations)),
        stream_spreads=dict.fromkeys(STREAMS, 0.8),
    )


@pytest.fixture
def client(tmp_path: Path, clock) -> TestClient:
    unit_of_work = InMemoryUnitOfWork()
    audit = FileAuditLog(tmp_path / "audit.jsonl")

    comparators = [
        SignatureComparator(stream, _calibrator(50 + index), f"sig-{stream.value}", 0.8)
        for index, stream in enumerate(STREAMS)
    ]
    compare = CompareIncidents(comparators, _fusion(), audit, clock)
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

    officer = Principal("enrol-1", frozenset({AnalystRole.ENROLMENT_OFFICER}))
    case = CaseReference.parse("ZP-2025-01847")
    gen = np.random.default_rng(4)
    for actor in range(3):
        for repetition in range(3):
            ingest.execute(
                IngestionRequest(
                    incident_id=__import__(
                        "viflap.domain.linkage", fromlist=["IncidentId"]
                    ).IncidentId(f"ZP-2025-{actor:02d}{repetition:03d}"),
                    case_reference=case,
                    submitted_by=officer,
                    metadata={
                        "signatures": [
                            float(actor * 3.0 + gen.normal(0.0, 0.3)) for _ in STREAMS
                        ]
                    },
                )
            )

    container = ApplicationContainer(
        compare=compare,
        search=search,
        ingest=ingest,
        incidents=unit_of_work.incidents,
        evidence=unit_of_work.evidence,
        audit=audit,
        clock=clock,
        principals=HeaderPrincipalResolver(),
    )
    return TestClient(create_app(container), raise_server_exceptions=False)


INVESTIGATOR = {"X-Analyst-Id": "inv-1", "X-Analyst-Roles": "investigator"}
AUDITOR = {"X-Analyst-Id": "aud-1", "X-Analyst-Roles": "oversight_auditor"}
CASE = {"X-Case-Reference": "ZP-2025-01847"}


class TestHealth:
    def test_reports_the_facts_needed_to_trust_the_deployment(self, client) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["enrolled_incidents"] == 9
        assert body["audit_chain_intact"] is True


class TestGovernanceAtTheTransportLayer:
    def test_query_without_a_case_reference_is_refused(self, client) -> None:
        response = client.post(
            "/api/v1/comparisons",
            headers=INVESTIGATOR,
            params={"incident_a": "ZP-2025-00000", "incident_b": "ZP-2025-00001"},
        )
        assert response.status_code == 403
        body = response.json()
        assert body["error_type"] == "CaseBindingViolation"
        assert "X-Case-Reference" in body["remedy"]

    def test_malformed_case_reference_is_refused(self, client) -> None:
        response = client.post(
            "/api/v1/comparisons",
            headers={**INVESTIGATOR, "X-Case-Reference": "not-a-case"},
            params={"incident_a": "ZP-2025-00000", "incident_b": "ZP-2025-00001"},
        )
        assert response.status_code == 403

    def test_missing_identity_is_refused(self, client) -> None:
        response = client.post(
            "/api/v1/comparisons",
            headers=CASE,
            params={"incident_a": "ZP-2025-00000", "incident_b": "ZP-2025-00001"},
        )
        assert response.status_code == 403

    def test_incompatible_roles_are_refused_before_any_evidence_is_read(
        self, client
    ) -> None:
        response = client.post(
            "/api/v1/comparisons",
            headers={
                "X-Analyst-Id": "bad-1",
                "X-Analyst-Roles": "investigator,disclosure_officer",
                **CASE,
            },
            params={"incident_a": "ZP-2025-00000", "incident_b": "ZP-2025-00001"},
        )
        assert response.status_code == 403
        assert response.json()["error_type"] == "SeparationOfDutiesViolation"

    def test_investigator_cannot_read_the_audit_trail(self, client) -> None:
        assert client.get("/api/v1/audit", headers=INVESTIGATOR).status_code == 403

    def test_auditor_can_read_the_audit_trail(self, client) -> None:
        response = client.get("/api/v1/audit", headers=AUDITOR)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_auditor_can_verify_the_chain(self, client) -> None:
        body = client.get("/api/v1/audit/verification", headers=AUDITOR).json()
        assert body["is_intact"] is True


class TestComparisonResponse:
    @pytest.fixture
    def comparison(self, client) -> dict:
        response = client.post(
            "/api/v1/comparisons",
            headers={**INVESTIGATOR, **CASE},
            params={"incident_a": "ZP-2025-00000", "incident_b": "ZP-2025-00001"},
        )
        assert response.status_code == 200, response.text
        return response.json()

    def test_a_likelihood_ratio_never_appears_without_its_prior(self, comparison) -> None:
        """The schema has no shape that omits it."""
        assert "fused_log10_lr" in comparison
        prior = comparison["prior"]
        for field in ("log_odds", "basis", "justification", "supplied_by"):
            assert prior[field] not in (None, "")
        assert "posterior_probability" in comparison

    def test_posterior_is_consistent_with_the_prior_and_the_ratio(self, comparison) -> None:
        expected = comparison["prior"]["log_odds"] + comparison[
            "fused_log10_lr"
        ] * math.log(10.0)
        assert comparison["posterior_log_odds"] == pytest.approx(expected, abs=1e-6)

    def test_strength_is_rendered_with_its_direction(self, comparison) -> None:
        summary = comparison["verbal_summary"]
        assert "same actor" in summary or "different actors" in summary

    def test_every_configured_stream_is_accounted_for(self, comparison) -> None:
        """A silently omitted stream is indexed identically to one never tried."""
        reported = {item["stream"] for item in comparison["streams"]}
        assert reported == {stream.value for stream in STREAMS}
        for item in comparison["streams"]:
            assert item["status"] in {"evidence", "absent"}
            if item["status"] == "absent":
                assert item["absence_reason"]

    def test_caveats_are_present_and_policy_compliant(self, comparison) -> None:
        from viflap.domain.governance import OutputLanguagePolicy

        assert comparison["caveats"]
        for caveat in comparison["caveats"]:
            OutputLanguagePolicy.assert_permitted(caveat)

    def test_independence_inflation_is_reported(self, comparison) -> None:
        """On every multi-stream comparison, not only in an appendix."""
        assert comparison["contributing_stream_count"] >= 2
        assert comparison["independence_inflation_log10"] is not None
        assert comparison["naive_log10_lr"] is not None

    def test_unknown_incident_gives_a_clean_not_found(self, client) -> None:
        response = client.post(
            "/api/v1/comparisons",
            headers={**INVESTIGATOR, **CASE},
            params={"incident_a": "ZP-2025-00000", "incident_b": "ZP-2025-99999"},
        )
        assert response.status_code == 404


class TestSearchResponse:
    @pytest.fixture
    def search(self, client) -> dict:
        response = client.post(
            "/api/v1/searches",
            headers=INVESTIGATOR,
            json={
                "probe_incident_id": "ZP-2025-00000",
                "case_reference": "ZP-2025-01847",
                "max_results": 3,
            },
        )
        assert response.status_code == 200, response.text
        return response.json()

    def test_carries_the_mandatory_caveat(self, search) -> None:
        assert search["mandatory_caveat"]
        assert "enrolled candidates" in search["mandatory_caveat"]

    def test_reports_how_many_could_not_be_compared(self, search) -> None:
        """Declined candidates have not been excluded on the evidence."""
        assert "n_declined" in search
        assert search["n_candidates_compared"] == 8

    def test_every_result_carries_its_own_prior(self, search) -> None:
        for result in search["results"]:
            assert result["prior"]["population_size"] == 9

    def test_results_are_ordered_by_posterior(self, search) -> None:
        posteriors = [item["posterior_log_odds"] for item in search["results"]]
        assert posteriors == sorted(posteriors, reverse=True)

    def test_restriction_requires_a_justification(self, client) -> None:
        response = client.post(
            "/api/v1/searches",
            headers=INVESTIGATOR,
            json={
                "probe_incident_id": "ZP-2025-00000",
                "case_reference": "ZP-2025-01847",
                "restricted_population": 50,
            },
        )
        assert response.status_code in (400, 422)


class TestOpenApiContract:
    def test_no_endpoint_accepts_a_live_audio_stream(self, client) -> None:
        """The absence is the control, so it is asserted rather than assumed."""
        paths = client.get("/openapi.json").json()["paths"]
        for path in paths:
            assert "stream" not in path.lower()
            assert "live" not in path.lower()
            assert "bulk" not in path.lower()

    def test_the_description_states_what_a_likelihood_ratio_is_not(self, client) -> None:
        description = client.get("/openapi.json").json()["info"]["description"]
        assert (
            "not** the probability" in description or "not the probability" in description
        )
