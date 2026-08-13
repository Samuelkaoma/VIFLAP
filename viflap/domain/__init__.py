"""The VIFLAP domain layer.

Pure. Depends on the standard library and nothing else — no numpy, no
framework, no I/O. Every other layer depends on this one and it depends on none
of them, so the concepts that a court would recognise (a likelihood ratio, a
prior, a case reference, an authority) are defined once and cannot drift with a
change of database or web framework.

The layering, outermost last:

``viflap.domain``
    Concepts and their invariants. This module.
``viflap.analysis``
    The science: signal processing, statistical models, calibration, fusion.
    Depends on the domain and on numpy/scipy. Contains no I/O.
``viflap.application``
    Use cases, expressed against ports (Protocols). Orchestrates the domain and
    analysis layers without knowing what implements the ports.
``viflap.infrastructure``
    Adapters that satisfy the ports: files, databases, codecs, model artefacts.
``viflap.interfaces``
    Delivery: HTTP API and command line.

Dependencies point inward only. A violation of that rule is a design defect, and
``tests/architecture`` enforces it mechanically rather than by convention.
"""

from viflap.domain.errors import (
    AuditIntegrityError,
    AuthorityViolation,
    CalibrationError,
    CaseBindingViolation,
    ConfigurationError,
    ConvergenceError,
    DomainError,
    EntityNotFoundError,
    GovernanceViolation,
    InsufficientDataError,
    InvalidEvidenceError,
    MetricInvariantError,
    ModelError,
    ModelNotTrainedError,
    OutputConstraintViolation,
    RepositoryError,
    RetentionViolation,
    SeparationOfDutiesViolation,
    StreamUnavailableError,
    ViflapError,
)
from viflap.domain.evidence import (
    AbsenceReason,
    DisguiseCondition,
    EvidenceStream,
    StreamAbsent,
    StreamEvidence,
    StreamOutcome,
    StreamOutcomes,
    ValidityAssessment,
    ValidityVerdict,
    absent_streams,
    missingness_pattern,
    present_evidence,
)
from viflap.domain.governance import (
    AnalystRole,
    Authority,
    CaseReference,
    CaseReferenceFormat,
    OutputLanguagePolicy,
    Principal,
    assert_separation_of_duties,
)
from viflap.domain.hypotheses import (
    Hypothesis,
    PosteriorAssessment,
    PriorBasis,
    PriorOdds,
    SearchMode,
)
from viflap.domain.linkage import (
    ComparisonResult,
    FusionMethod,
    IncidentId,
    IncidentPair,
    OverstatementEstimate,
)
from viflap.domain.metrics import CalibrationSummary, Estimate, PerformanceGrade
from viflap.domain.values import (
    EvidentialStrength,
    LikelihoodRatio,
    LogLikelihoodRatio,
    LogOdds,
    Probability,
    UncertaintyInterval,
    log_logistic,
    logistic,
)

__all__ = [
    # errors
    "AuditIntegrityError",
    "AuthorityViolation",
    "CalibrationError",
    "CaseBindingViolation",
    "ConfigurationError",
    "ConvergenceError",
    "DomainError",
    "EntityNotFoundError",
    "GovernanceViolation",
    "InsufficientDataError",
    "InvalidEvidenceError",
    "MetricInvariantError",
    "ModelError",
    "ModelNotTrainedError",
    "OutputConstraintViolation",
    "RepositoryError",
    "RetentionViolation",
    "SeparationOfDutiesViolation",
    "StreamUnavailableError",
    "ViflapError",
    # evidence
    "AbsenceReason",
    "DisguiseCondition",
    "EvidenceStream",
    "StreamAbsent",
    "StreamEvidence",
    "StreamOutcome",
    "StreamOutcomes",
    "ValidityAssessment",
    "ValidityVerdict",
    "absent_streams",
    "missingness_pattern",
    "present_evidence",
    # governance
    "AnalystRole",
    "Authority",
    "CaseReference",
    "CaseReferenceFormat",
    "OutputLanguagePolicy",
    "Principal",
    "assert_separation_of_duties",
    # hypotheses
    "Hypothesis",
    "PosteriorAssessment",
    "PriorBasis",
    "PriorOdds",
    "SearchMode",
    # linkage
    "ComparisonResult",
    "FusionMethod",
    "IncidentId",
    "IncidentPair",
    "OverstatementEstimate",
    # metrics
    "CalibrationSummary",
    "Estimate",
    "PerformanceGrade",
    # values
    "EvidentialStrength",
    "LikelihoodRatio",
    "LogLikelihoodRatio",
    "LogOdds",
    "Probability",
    "UncertaintyInterval",
    "log_logistic",
    "logistic",
]
