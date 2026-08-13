"""Use cases, expressed against ports.

Orchestrates the domain and analysis layers without knowing what implements the
ports it depends on. No framework imports, no database, no file paths.

``ports``
    The interfaces. Protocols, so adapters satisfy them structurally and the
    dependency arrow points inward.
``ingestion``
    Enrolment. Where "no live path" and "no unattributed bulk import" are
    architectural facts rather than policies — there is no function for either.
``comparison``
    Two incidents in, one contextualised result out. Authority checked before
    evidence is read; the validity gate applied before the acoustic stream;
    absence recorded with a reason; a prior required; overstatement measured.
``search``
    Database search, with the prior derived from the actual enrolled population
    rather than from anything the caller supplies.
"""

from viflap.application.comparison import CompareIncidents, ComparisonRequest
from viflap.application.ingestion import (
    IngestIncident,
    IngestionOutcome,
    IngestionRequest,
)
from viflap.application.ports import (
    AuditRecord,
    AuditSink,
    ChainVerification,
    Clock,
    ComparisonResultRepository,
    EvidenceBundle,
    EvidenceRepository,
    FusionProvider,
    IncidentRecord,
    IncidentRepository,
    StreamComparator,
    UnitOfWork,
    ValidityAssessor,
)
from viflap.application.search import SearchDatabase, SearchRequest, SearchResults

__all__ = [
    "AuditRecord",
    "AuditSink",
    "ChainVerification",
    "Clock",
    "CompareIncidents",
    "ComparisonRequest",
    "ComparisonResultRepository",
    "EvidenceBundle",
    "EvidenceRepository",
    "FusionProvider",
    "IncidentRecord",
    "IncidentRepository",
    "IngestIncident",
    "IngestionOutcome",
    "IngestionRequest",
    "SearchDatabase",
    "SearchRequest",
    "SearchResults",
    "StreamComparator",
    "UnitOfWork",
    "ValidityAssessor",
]
