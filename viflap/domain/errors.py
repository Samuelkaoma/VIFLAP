"""Exception hierarchy for VIFLAP.

Several of these exceptions are not error handling in the ordinary sense; they
are the runtime expression of an architectural constraint. A
:class:`GovernanceViolation` is what "queries without a case reference are
structurally impossible" *means* in code — the operation cannot proceed, and the
exception must never be caught and suppressed.

The hierarchy is deliberately shallow. Every exception carries enough structured
context to be logged and audited without re-parsing its message.
"""

from __future__ import annotations

from typing import Any

__all__ = [
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
]


class ViflapError(Exception):
    """Base class for every error raised by VIFLAP.

    Parameters
    ----------
    message:
        Human-readable description. This string may be surfaced to an operator,
        so it must not contain personal data or raw evidence values.
    context:
        Structured detail for logs and the audit trail. Keys should be stable
        identifiers, not prose.
    """

    def __init__(self, message: str, /, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context)

    def __str__(self) -> str:  # pragma: no cover - trivial
        if not self.context:
            return self.message
        rendered = ", ".join(
            f"{key}={value!r}" for key, value in sorted(self.context.items())
        )
        return f"{self.message} ({rendered})"


# ---------------------------------------------------------------------------
# Domain invariants
# ---------------------------------------------------------------------------


class DomainError(ViflapError):
    """A domain invariant was violated."""


class InvalidEvidenceError(DomainError):
    """Evidence violates a mathematical invariant.

    Examples
    --------
    - A likelihood ratio that is zero, negative or non-finite.
    - An uncertainty interval whose lower bound exceeds its upper bound.
    - A posterior that is inconsistent with the likelihood ratio and prior that
      allegedly produced it.
    """


class MetricInvariantError(DomainError):
    """An evaluation metric violates an invariant that must hold by construction.

    ``C_llr >= C_llr_min >= 0`` is not a convention; it follows from the
    definition of ``C_llr_min`` as the minimum of ``C_llr`` over monotonic
    recalibrations. A violation indicates a defect in the metric implementation,
    never a property of the system under evaluation.
    """


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------


class GovernanceViolation(ViflapError):
    """An operation violated a structural governance constraint.

    This is a hard stop, not a warning. Governance implemented as a constraint
    that callers may catch and ignore is governance that survives until the
    first inconvenience.
    """


class CaseBindingViolation(GovernanceViolation):
    """An operation was attempted without a valid, well-formed case reference."""


class AuthorityViolation(GovernanceViolation):
    """An actor attempted an action outside the authority of its role."""


class SeparationOfDutiesViolation(GovernanceViolation):
    """A single principal was assigned an incompatible combination of roles."""


class OutputConstraintViolation(GovernanceViolation):
    """Text leaving the system boundary contained prohibited language.

    The system must be technically incapable of asserting identity. This
    exception is how that incapability is realised: any code path constructing
    operator-facing text passes it through the output policy, and prohibited
    phrasing raises rather than being silently rewritten. Silent rewriting would
    hide the defect that produced the phrasing in the first place.
    """


class AuditIntegrityError(GovernanceViolation):
    """The audit log's hash chain does not verify.

    Indicates tampering, storage corruption, or a defect in the chaining
    implementation. In every case, audit-dependent operations must halt until
    the breach is investigated.
    """


class RetentionViolation(GovernanceViolation):
    """A data retention rule was violated or could not be enforced."""


# ---------------------------------------------------------------------------
# Models and estimation
# ---------------------------------------------------------------------------


class ModelError(ViflapError):
    """Base class for statistical model failures."""


class ModelNotTrainedError(ModelError):
    """A fitted quantity was requested from a model that has not been trained.

    Uncalibrated output is worse than no output: it carries an implied strength
    it does not possess. Models therefore refuse to score before they are
    trained rather than falling back to a default transform.
    """


class InsufficientDataError(ModelError):
    """Training or calibration data is too small for the estimate to be meaningful."""


class ConvergenceError(ModelError):
    """An iterative estimation procedure failed to converge."""


class CalibrationError(ModelError):
    """Calibration could not be fitted, applied, or evaluated."""


class StreamUnavailableError(ModelError):
    """An evidence stream cannot produce a result for this comparison.

    This is a control-flow signal, not a fault. The fusion engine treats the
    stream as absent data — which is emphatically not the same as neutral
    evidence with a likelihood ratio of one.
    """


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


class ConfigurationError(ViflapError):
    """Configuration is missing, malformed, or internally inconsistent."""


class RepositoryError(ViflapError):
    """A persistence operation failed."""


class EntityNotFoundError(RepositoryError):
    """A requested entity does not exist."""
