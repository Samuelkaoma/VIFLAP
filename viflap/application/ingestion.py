"""The ingestion use case.

Enrolment is the point at which material enters the searchable corpus, and it is
the point at which the architectural constraints on *what this system can be*
are enforced. Three of them live here.

**Every recording arrives bound to a filed complaint.** The signature requires a
:class:`~viflap.domain.governance.CaseReference`, and that type cannot hold an
invalid value. There is no ingestion path taking a bare file, so bulk import of
unattributed audio is not a use that is discouraged — it is an operation the
system has no function for.

**There is no live path.** Ingestion takes a complete recording. Nothing here
accepts a stream, a socket or a partial buffer, and the absence is the control.
A capability the software does not have cannot be enabled by an operator who
finds its absence inconvenient.

**Enrolment and query are separate authorities**, and the domain forbids one
principal holding both — a principal who could enrol and query could plant a
reference and then produce a linkage to it, and nothing in the audit trail would
distinguish that from an investigation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from viflap.application.ports import (
    AuditRecord,
    AuditSink,
    Clock,
    EvidenceBundle,
    IncidentRecord,
    UnitOfWork,
    ValidityAssessor,
)
from viflap.domain.errors import GovernanceViolation, InvalidEvidenceError
from viflap.domain.evidence import EvidenceStream, ValidityAssessment, ValidityVerdict
from viflap.domain.governance import Authority, CaseReference, Principal
from viflap.domain.linkage import IncidentId

__all__ = ["IngestIncident", "IngestionOutcome", "IngestionRequest"]


@dataclass(frozen=True, slots=True)
class IngestionRequest:
    """A request to enrol one incident's evidence."""

    incident_id: IncidentId
    case_reference: CaseReference
    submitted_by: Principal
    audio: Any | None = None
    sample_rate: int | None = None
    recording_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.audio is not None and not self.sample_rate:
            raise InvalidEvidenceError(
                "audio was supplied without its sample rate; every downstream "
                "measurement depends on it and guessing would silently change "
                "every frequency in the analysis"
            )
        if self.audio is not None and not self.recording_id.strip():
            raise InvalidEvidenceError(
                "audio must be enrolled under a recording identifier so that a "
                "validity verdict can be traced back to the material it judged"
            )


@dataclass(frozen=True, slots=True)
class IngestionOutcome:
    """What enrolment produced."""

    incident_id: IncidentId
    extracted_streams: tuple[EvidenceStream, ...]
    validity: ValidityAssessment | None
    warnings: tuple[str, ...] = ()

    @property
    def acoustic_admitted(self) -> bool:
        return self.validity is not None and self.validity.verdict.permits_acoustic_evidence


class IngestIncident:
    """Extracts and stores evidence for one incident."""

    def __init__(
        self,
        extractors: Mapping[EvidenceStream, Any],
        unit_of_work: UnitOfWork,
        audit: AuditSink,
        clock: Clock,
        validity_assessor: ValidityAssessor | None = None,
    ) -> None:
        """
        Parameters
        ----------
        extractors:
            Per-stream callables turning raw material into a stored payload.
            Each takes the request and returns a payload, or raises to decline.
        validity_assessor:
            Runs the synthetic-speech gate at enrolment rather than at query
            time. Deliberate: the verdict is a property of the recording, so
            computing it once at enrolment means every later query sees the same
            verdict, and a change of policy is visible as a re-assessment rather
            than as query results that silently differ from last week's.
        """
        self._extractors = dict(extractors)
        self._uow = unit_of_work
        self._audit = audit
        self._clock = clock
        self._validity = validity_assessor

    def execute(self, request: IngestionRequest) -> IngestionOutcome:
        """Enrol one incident.

        Raises
        ------
        AuthorityViolation
            If the principal may not enrol.
        GovernanceViolation
            If the incident is already enrolled. Re-enrolment under the same
            identifier would replace evidence that earlier results were computed
            from, leaving those results unreproducible and their audit entries
            pointing at material that no longer exists.
        """
        request.submitted_by.require(Authority.ENROL)

        warnings: list[str] = []
        validity: ValidityAssessment | None = None
        payloads: dict[EvidenceStream, Any] = {}

        with self._uow as work:
            if work.incidents.exists(request.incident_id):
                self._record(request, "failed: already enrolled", {})
                raise GovernanceViolation(
                    "this incident is already enrolled; replacing its evidence "
                    "would make results computed from the previous material "
                    "unreproducible",
                    incident=request.incident_id.value,
                )

            if request.audio is not None and self._validity is not None:
                validity = self._validity.assess(
                    request.recording_id, request.audio, int(request.sample_rate or 0)
                )
                if validity.verdict is ValidityVerdict.EXCLUDED:
                    warnings.append(
                        "The recording was judged synthetic or voice-converted. "
                        "It is enrolled with that verdict attached, and its "
                        "acoustic evidence will be excluded from every "
                        "comparison rather than down-weighted."
                    )
                elif validity.verdict is ValidityVerdict.INDETERMINATE:
                    warnings.append(
                        "The validity gate could not determine whether the "
                        "recording is genuine. Acoustic evidence will be "
                        "withheld pending assessment by a person."
                    )

            for stream, extractor in self._extractors.items():
                try:
                    payload = extractor(request)
                except Exception as exc:
                    # One stream failing must not prevent the others being
                    # enrolled. A recording with no transaction records is still
                    # worth enrolling for its acoustic and behavioural evidence,
                    # and refusing the lot would discard usable material.
                    warnings.append(
                        f"{stream.display_name} evidence was not extracted: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if payload is not None:
                    payloads[stream] = payload

            work.incidents.add(
                IncidentRecord(
                    incident_id=request.incident_id,
                    case_reference=request.case_reference,
                    enrolled_at=self._clock.now(),
                    enrolled_by=request.submitted_by.identifier,
                    metadata=dict(request.metadata),
                )
            )
            work.evidence.store(
                EvidenceBundle(
                    incident_id=request.incident_id,
                    payloads=payloads,
                    validity=validity,
                    metadata=dict(request.metadata),
                )
            )
            work.commit()

        self._record(
            request,
            "ok",
            {
                "extracted_streams": sorted(stream.value for stream in payloads),
                "validity_verdict": validity.verdict.value if validity else None,
                "n_warnings": len(warnings),
            },
        )

        return IngestionOutcome(
            incident_id=request.incident_id,
            extracted_streams=tuple(
                stream for stream in EvidenceStream.ordered() if stream in payloads
            ),
            validity=validity,
            warnings=tuple(warnings),
        )

    def _record(
        self, request: IngestionRequest, outcome: str, extra: Mapping[str, object]
    ) -> None:
        self._audit.record(
            AuditRecord(
                timestamp=self._clock.now(),
                actor_id=request.submitted_by.identifier,
                actor_roles=tuple(
                    sorted(role.value for role in request.submitted_by.roles)
                ),
                action="ingest_incident",
                case_reference=request.case_reference.value,
                parameters={
                    "incident_id": request.incident_id.value,
                    "recording_id": request.recording_id,
                    "has_audio": request.audio is not None,
                    **extra,
                },
                outcome=outcome,
            )
        )
