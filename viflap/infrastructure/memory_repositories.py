"""In-memory repositories and unit of work.

The reference implementation of the persistence ports. Two uses, and the second
is the reason they are written carefully rather than as a stub.

They are the test double for every use case, so a defect here shows up as a
passing test over broken behaviour. And they are a complete, working deployment
for a single-analyst prototype — which is the fidelity at which the research
proposal scopes the investigator study — so a prototype needs no database.

Transactional semantics are real, not simulated. Writes go to a staging area
inside the unit of work and are applied on commit; an abandoned block leaves the
store untouched. Without that, a use case that fails between writing evidence
and writing its audit entry would leave the two disagreeing, and the tests
asserting they cannot disagree would pass against an implementation where they
can.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from viflap.application.ports import (
    ComparisonResultRepository,
    EvidenceBundle,
    EvidenceRepository,
    IncidentRecord,
    IncidentRepository,
)
from viflap.domain.errors import EntityNotFoundError, RepositoryError
from viflap.domain.governance import CaseReference
from viflap.domain.linkage import ComparisonResult, IncidentId

__all__ = [
    "InMemoryComparisonResultRepository",
    "InMemoryEvidenceRepository",
    "InMemoryIncidentRepository",
    "InMemoryStore",
    "InMemoryUnitOfWork",
]


@dataclass
class InMemoryStore:
    """The committed state, shared by every unit of work opened against it."""

    incidents: dict[str, IncidentRecord] = field(default_factory=dict)
    evidence: dict[str, EvidenceBundle] = field(default_factory=dict)
    results: dict[str, ComparisonResult] = field(default_factory=dict)
    _next_result_id: int = 1

    def allocate_result_id(self) -> str:
        identifier = f"res-{self._next_result_id:08d}"
        self._next_result_id += 1
        return identifier


class InMemoryIncidentRepository:
    """Incident records, staged until commit."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store
        self._staged_adds: dict[str, IncidentRecord] = {}
        self._staged_deletes: set[str] = set()

    def add(self, record: IncidentRecord) -> None:
        key = record.incident_id.value
        if self.exists(record.incident_id):
            raise RepositoryError("incident already enrolled", incident=key)
        self._staged_adds[key] = record

    def get(self, incident_id: IncidentId) -> IncidentRecord:
        key = incident_id.value
        if key in self._staged_deletes:
            raise EntityNotFoundError("incident not found", incident=key)
        record = self._staged_adds.get(key) or self._store.incidents.get(key)
        if record is None:
            raise EntityNotFoundError("incident not found", incident=key)
        return record

    def exists(self, incident_id: IncidentId) -> bool:
        key = incident_id.value
        if key in self._staged_deletes:
            return False
        return key in self._staged_adds or key in self._store.incidents

    def list_enrolled(self, limit: int | None = None) -> Sequence[IncidentRecord]:
        combined = {**self._store.incidents, **self._staged_adds}
        for key in self._staged_deletes:
            combined.pop(key, None)
        # Sorted by identifier so a search enumerates candidates in a stable
        # order. Without it, two runs of the same search can differ in which
        # equally-ranked candidate appears first, and a result that changes
        # between runs cannot be defended.
        records = [combined[key] for key in sorted(combined)]
        return records[:limit] if limit is not None else records

    def count(self) -> int:
        combined = set(self._store.incidents) | set(self._staged_adds)
        return len(combined - self._staged_deletes)

    def delete(self, incident_id: IncidentId) -> None:
        key = incident_id.value
        self._staged_adds.pop(key, None)
        self._staged_deletes.add(key)

    def apply(self) -> None:
        self._store.incidents.update(self._staged_adds)
        for key in self._staged_deletes:
            self._store.incidents.pop(key, None)
        self.discard()

    def discard(self) -> None:
        self._staged_adds.clear()
        self._staged_deletes.clear()


class InMemoryEvidenceRepository:
    """Evidence bundles, staged until commit."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store
        self._staged: dict[str, EvidenceBundle] = {}
        self._staged_deletes: set[str] = set()

    def store_bundle(self, bundle: EvidenceBundle) -> None:
        self._staged[bundle.incident_id.value] = bundle

    # The port names this ``store``; the attribute name above avoids shadowing
    # the constructor argument while keeping the port satisfied.
    store = store_bundle

    def load(self, incident_id: IncidentId) -> EvidenceBundle:
        key = incident_id.value
        if key in self._staged_deletes:
            raise EntityNotFoundError("no evidence for incident", incident=key)
        bundle = self._staged.get(key) or self._store.evidence.get(key)
        if bundle is None:
            raise EntityNotFoundError("no evidence for incident", incident=key)
        return bundle

    def load_many(self, incident_ids: Sequence[IncidentId]) -> Sequence[EvidenceBundle]:
        bundles: list[EvidenceBundle] = []
        for incident_id in incident_ids:
            try:
                bundles.append(self.load(incident_id))
            except EntityNotFoundError:
                continue
        return bundles

    def delete(self, incident_id: IncidentId) -> None:
        key = incident_id.value
        self._staged.pop(key, None)
        self._staged_deletes.add(key)

    def apply(self) -> None:
        self._store.evidence.update(self._staged)
        for key in self._staged_deletes:
            self._store.evidence.pop(key, None)
        self.discard()

    def discard(self) -> None:
        self._staged.clear()
        self._staged_deletes.clear()


class InMemoryComparisonResultRepository:
    """Comparison results, staged until commit."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store
        self._staged: dict[str, ComparisonResult] = {}

    def store(self, result: ComparisonResult) -> str:
        identifier = self._store.allocate_result_id()
        self._staged[identifier] = result
        return identifier

    def get(self, result_id: str) -> ComparisonResult:
        result = self._staged.get(result_id) or self._store.results.get(result_id)
        if result is None:
            raise EntityNotFoundError("result not found", result_id=result_id)
        return result

    def list_for_case(self, case_reference: CaseReference) -> Sequence[ComparisonResult]:
        combined = {**self._store.results, **self._staged}
        return [
            result
            for _, result in sorted(combined.items())
            if result.pair.authorised_by == case_reference
        ]

    def apply(self) -> None:
        self._store.results.update(self._staged)
        self._staged.clear()

    def discard(self) -> None:
        self._staged.clear()


class InMemoryUnitOfWork:
    """Transaction boundary over the in-memory store."""

    def __init__(self, store: InMemoryStore | None = None) -> None:
        self._store = store if store is not None else InMemoryStore()
        # Annotated with the port types, not the concrete ones. ``UnitOfWork``
        # declares these as mutable attributes, and a protocol's mutable
        # attributes are invariant: a narrower type here means this class
        # silently stops satisfying the port it exists to implement.
        self._incidents = InMemoryIncidentRepository(self._store)
        self._evidence = InMemoryEvidenceRepository(self._store)
        self._results = InMemoryComparisonResultRepository(self._store)
        # Exposed under the port types, not the concrete ones. ``UnitOfWork``
        # declares these as mutable attributes, and a protocol's mutable
        # attributes are invariant: narrowing them here makes this class stop
        # satisfying the port it exists to implement. The concrete references
        # are kept privately because staging is not part of the port.
        self.incidents: IncidentRepository = self._incidents
        self.evidence: EvidenceRepository = self._evidence
        self.results: ComparisonResultRepository = self._results
        self._committed = False

    @property
    def store(self) -> InMemoryStore:
        return self._store

    def __enter__(self) -> InMemoryUnitOfWork:
        self._committed = False
        return self

    def __exit__(self, *exc_info: object) -> None:
        # Roll back unless commit was called explicitly. A block that raises, or
        # one that simply forgets to commit, must leave nothing behind — the
        # failure mode being guarded against is a half-written enrolment that no
        # audit entry describes.
        if not self._committed:
            self.rollback()

    def commit(self) -> None:
        self._incidents.apply()
        self._evidence.apply()
        self._results.apply()
        self._committed = True

    def rollback(self) -> None:
        self._incidents.discard()
        self._evidence.discard()
        self._results.discard()
