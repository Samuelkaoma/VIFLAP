"""Ports: the interfaces the application layer depends on.

Every one is a :class:`typing.Protocol` rather than an abstract base class.
Structural typing means an adapter satisfies a port by having the right shape,
without importing anything from this module — so the dependency arrow points
inward from infrastructure to application, and never the other way. An in-memory
repository written for a test is not a subclass of anything; it simply has the
methods.

The abstraction that carries the design is :class:`StreamComparator`. Each
evidence stream is wrapped in an adapter with the same three-method shape, and
the comparison use case iterates over whatever comparators it was given. Adding
a sixth stream requires no change to the use case, and — more importantly for
this system — a stream can be *withdrawn* from a deployment by not registering
it, which is how a jurisdiction that has not authorised transaction access runs
the same software without it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from viflap.domain.evidence import EvidenceStream, StreamOutcome, ValidityAssessment
from viflap.domain.governance import CaseReference
from viflap.domain.linkage import ComparisonResult, IncidentId

__all__ = [
    "AuditRecord",
    "AuditSink",
    "ChainVerification",
    "Clock",
    "ComparisonResultRepository",
    "EvidenceBundle",
    "EvidenceRepository",
    "FusionProvider",
    "IncidentRecord",
    "IncidentRepository",
    "StreamComparator",
    "UnitOfWork",
    "ValidityAssessor",
]


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


@runtime_checkable
class Clock(Protocol):
    """Source of the current time.

    Injected rather than called directly so that results and audit entries have
    reproducible timestamps under test. A use case that calls
    ``datetime.now()`` cannot be tested for the ordering guarantees the audit
    chain depends on.
    """

    def now(self) -> datetime:
        """Current time, timezone-aware."""
        ...


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Everything extracted from one incident, keyed by stream.

    Payloads are deliberately opaque to the application layer. Each
    :class:`StreamComparator` knows how to interpret its own — an acoustic
    embedding, a behavioural profile, a set of transactions — and nothing else
    needs to. This keeps the comparison use case free of any knowledge of what
    a stream's evidence looks like, so the streams can change independently of
    it.

    A stream absent from ``payloads`` was not extracted. That is different from
    being extracted and found uninformative, and the distinction survives into
    the result.
    """

    incident_id: IncidentId
    payloads: Mapping[EvidenceStream, Any] = field(default_factory=dict)
    validity: ValidityAssessment | None = None
    """The validity gate's verdict on this incident's recording, where one
    exists. Held on the bundle rather than inside the acoustic payload because
    it governs admissibility rather than contributing evidence."""

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def has(self, stream: EvidenceStream) -> bool:
        return stream in self.payloads

    def payload(self, stream: EvidenceStream) -> Any:
        return self.payloads.get(stream)

    @property
    def available_streams(self) -> frozenset[EvidenceStream]:
        return frozenset(self.payloads)


@runtime_checkable
class StreamComparator(Protocol):
    """Compares one evidence stream across two incidents.

    Implementations return a :class:`~viflap.domain.evidence.StreamOutcome` —
    either calibrated evidence or a reasoned absence. They must **not** raise
    when the stream is inapplicable: absence is a normal outcome carrying
    investigative meaning, and an exception would force every caller to decide
    what an absent stream means, which is exactly the decision that must be made
    in one place.
    """

    @property
    def stream(self) -> EvidenceStream: ...

    @property
    def model_id(self) -> str:
        """Identity of the scoring and calibration models in use."""
        ...

    def compare(self, first: EvidenceBundle, second: EvidenceBundle) -> StreamOutcome:
        """Compare the two incidents on this stream."""
        ...


@runtime_checkable
class ValidityAssessor(Protocol):
    """Judges whether a recording's acoustic evidence is admissible."""

    @property
    def detector_id(self) -> str: ...

    def assess(
        self, recording_id: str, audio: Any, sample_rate: int
    ) -> ValidityAssessment: ...


@runtime_checkable
class FusionProvider(Protocol):
    """Supplies the fitted fusion model and its naive comparator."""

    @property
    def model_id(self) -> str: ...

    def fuse(self, log_lrs: Mapping[EvidenceStream, float]) -> float: ...

    def fuse_naive(self, log_lrs: Mapping[EvidenceStream, float]) -> float:
        """Naive summation, for measuring what independence would have added."""
        ...

    def supports_pattern(self, pattern: frozenset[EvidenceStream]) -> bool: ...

    def uncertainty_for(
        self, log_lrs: Mapping[EvidenceStream, float], fused: float
    ) -> tuple[float, float]:
        """Lower and upper bounds on the fused log-LR."""
        ...


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IncidentRecord:
    """An enrolled incident, as persistence sees it."""

    incident_id: IncidentId
    case_reference: CaseReference
    enrolled_at: datetime
    enrolled_by: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class IncidentRepository(Protocol):
    def add(self, record: IncidentRecord) -> None: ...

    def get(self, incident_id: IncidentId) -> IncidentRecord: ...

    def exists(self, incident_id: IncidentId) -> bool: ...

    def list_enrolled(self, limit: int | None = None) -> Sequence[IncidentRecord]: ...

    def count(self) -> int:
        """Number of enrolled incidents.

        Not a convenience. This is the ``N`` from which the prior odds of a
        database search are derived, so it is a load-bearing quantity: a search
        reported against the wrong ``N`` reports the wrong posterior.
        """
        ...

    def delete(self, incident_id: IncidentId) -> None:
        """Remove an incident, for retention enforcement."""
        ...


@runtime_checkable
class EvidenceRepository(Protocol):
    def store(self, bundle: EvidenceBundle) -> None: ...

    def load(self, incident_id: IncidentId) -> EvidenceBundle: ...

    def load_many(self, incident_ids: Sequence[IncidentId]) -> Sequence[EvidenceBundle]: ...

    def delete(self, incident_id: IncidentId) -> None: ...


@runtime_checkable
class ComparisonResultRepository(Protocol):
    def store(self, result: ComparisonResult) -> str:
        """Persist a result, returning its identifier."""
        ...

    def get(self, result_id: str) -> ComparisonResult: ...

    def list_for_case(
        self, case_reference: CaseReference
    ) -> Sequence[ComparisonResult]: ...


@runtime_checkable
class UnitOfWork(Protocol):
    """Transaction boundary.

    Use cases that both write evidence and record an audit entry need the two to
    commit together. An audit log that survives a failed enrolment records
    something that did not happen; evidence that survives a failed audit write is
    unaccounted for. Neither is acceptable, so both go through one boundary.
    """

    incidents: IncidentRepository
    evidence: EvidenceRepository
    results: ComparisonResultRepository

    def __enter__(self) -> UnitOfWork: ...

    def __exit__(self, *exc_info: object) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One entry in the audit trail.

    Records the query as well as the result. A log that captures only what the
    system returned cannot answer the question an oversight body actually asks,
    which is what the system was *asked*. Queries that returned nothing are the
    ones most worth being able to see.
    """

    timestamp: datetime
    actor_id: str
    actor_roles: tuple[str, ...]
    action: str
    case_reference: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    outcome: str = "ok"


@dataclass(frozen=True, slots=True)
class ChainVerification:
    """Result of verifying the audit chain's integrity."""

    is_intact: bool
    n_entries: int
    first_broken_index: int | None = None
    detail: str = ""


@runtime_checkable
class AuditSink(Protocol):
    """Append-only, tamper-evident audit trail."""

    def record(self, entry: AuditRecord) -> str:
        """Append an entry, returning its hash."""
        ...

    def verify(self) -> ChainVerification: ...

    def query(
        self,
        case_reference: str | None = None,
        actor_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> Sequence[AuditRecord]: ...
