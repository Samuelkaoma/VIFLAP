"""Enforced retention expiry.

Storage limitation is a data protection principle, and the difference between
observing it and enforcing it is whether deletion happens when nobody is
watching. This module deletes on a schedule, and audits the deletion.

Two properties are load-bearing.

**Deletion is logged before it happens, not after.** A deletion recorded
afterwards is a deletion that was not recorded if the process died in between,
and the missing record is indistinguishable from a deletion that never occurred.
Logging first can produce a record of a deletion that did not complete, which is
the failure to prefer: it is visible and it can be re-run.

**The audit log is never subject to retention deletion by this module.** Audit
entries outlive the data they describe — that is what makes them auditable. A
retention policy applied to the audit trail would let an operator remove the
record of an access by waiting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from viflap.application.ports import (
    AuditRecord,
    AuditSink,
    Clock,
    UnitOfWork,
)
from viflap.domain.errors import RetentionViolation
from viflap.domain.linkage import IncidentId

__all__ = ["RetentionPolicy", "RetentionSchedule", "RetentionStatus", "RetentionSweep"]


class RetentionStatus(Enum):
    ACTIVE = "active"
    EXPIRING_SOON = "expiring_soon"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """How long one class of data may be held, and on what authority."""

    data_class: str
    max_days: int
    legal_basis: str
    """The specific authority for this period. Not decoration: a retention
    period without a stated basis cannot be defended when questioned, and a
    period chosen because it seemed reasonable is the first thing an
    inspection finds."""

    warn_days_before: int = 30

    def __post_init__(self) -> None:
        if self.max_days < 1:
            raise RetentionViolation(
                "a retention period must be at least one day", data_class=self.data_class
            )
        if not self.legal_basis.strip():
            raise RetentionViolation(
                "a retention policy requires a stated legal basis",
                data_class=self.data_class,
            )

    def status_at(self, created: datetime, now: datetime) -> RetentionStatus:
        age = (now - created).days
        if age >= self.max_days:
            return RetentionStatus.EXPIRED
        if age >= self.max_days - self.warn_days_before:
            return RetentionStatus.EXPIRING_SOON
        return RetentionStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class RetentionSchedule:
    """The policies in force for a deployment."""

    policies: Mapping[str, RetentionPolicy]

    def for_class(self, data_class: str) -> RetentionPolicy:
        policy = self.policies.get(data_class)
        if policy is None:
            raise RetentionViolation(
                "no retention policy covers this data class; data held under no "
                "policy is held indefinitely, which is the condition storage "
                "limitation exists to prevent",
                data_class=data_class,
            )
        return policy

    @classmethod
    def default(cls) -> RetentionSchedule:
        """A starting schedule, to be replaced by one with real legal advice.

        The periods are placeholders and say so. Setting them correctly is a
        legal question about a specific jurisdiction and a specific authority,
        and the proposal is explicit that legal interpretation is outside the
        author's competence.
        """
        return cls(
            policies={
                "recording": RetentionPolicy(
                    "recording",
                    365,
                    "PLACEHOLDER — requires legal opinion for the deployment "
                    "jurisdiction before use",
                ),
                "evidence": RetentionPolicy(
                    "evidence",
                    365,
                    "PLACEHOLDER — derived representations, tied to the "
                    "retention of the material they derive from",
                ),
                "result": RetentionPolicy(
                    "result",
                    180,
                    "PLACEHOLDER — comparison outputs, retained for the life of "
                    "the investigation",
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class SweepReport:
    """What a retention sweep did."""

    examined: int
    expired: int
    deleted: int
    expiring_soon: tuple[str, ...]
    failures: tuple[str, ...]


class RetentionSweep:
    """Deletes expired data and records having done so."""

    def __init__(
        self,
        schedule: RetentionSchedule,
        unit_of_work: UnitOfWork,
        audit: AuditSink,
        clock: Clock,
    ) -> None:
        self._schedule = schedule
        self._uow = unit_of_work
        self._audit = audit
        self._clock = clock

    def run(self, actor_id: str = "system") -> SweepReport:
        """Examine every enrolled incident and delete what has expired."""
        now = self._clock.now()
        policy = self._schedule.for_class("evidence")

        examined = 0
        expired: list[IncidentId] = []
        expiring: list[str] = []
        failures: list[str] = []

        with self._uow as work:
            for record in work.incidents.list_enrolled():
                examined += 1
                status = policy.status_at(record.enrolled_at, now)
                if status is RetentionStatus.EXPIRED:
                    expired.append(record.incident_id)
                elif status is RetentionStatus.EXPIRING_SOON:
                    expiring.append(record.incident_id.value)

            deleted = 0
            for incident_id in expired:
                # Logged before deletion. A record of a deletion that did not
                # complete is recoverable; a deletion with no record is not.
                self._audit.record(
                    AuditRecord(
                        timestamp=self._clock.now(),
                        actor_id=actor_id,
                        actor_roles=("system",),
                        action="retention_delete",
                        case_reference="SYSTEM",
                        parameters={
                            "incident_id": incident_id.value,
                            "data_class": policy.data_class,
                            "max_days": policy.max_days,
                            "legal_basis": policy.legal_basis,
                        },
                        outcome="pending",
                    )
                )
                try:
                    work.evidence.delete(incident_id)
                    work.incidents.delete(incident_id)
                    deleted += 1
                except Exception as exc:
                    failures.append(f"{incident_id.value}: {exc}")

            work.commit()

        self._audit.record(
            AuditRecord(
                timestamp=self._clock.now(),
                actor_id=actor_id,
                actor_roles=("system",),
                action="retention_sweep",
                case_reference="SYSTEM",
                parameters={
                    "examined": examined,
                    "expired": len(expired),
                    "deleted": deleted,
                    "expiring_soon": len(expiring),
                    "failures": len(failures),
                },
                outcome="ok" if not failures else "partial",
            )
        )

        return SweepReport(
            examined=examined,
            expired=len(expired),
            deleted=deleted,
            expiring_soon=tuple(expiring),
            failures=tuple(failures),
        )
