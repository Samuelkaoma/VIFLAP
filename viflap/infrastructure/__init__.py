"""Adapters satisfying the application layer's ports.

Nothing here is imported by the application layer. The ports are Protocols, so
an adapter satisfies one by having the right shape, and the dependency arrow
points inward only.

``audit``
    Hash-chained, append-only audit log with O(1) appends and durable writes.
``memory_repositories``
    In-memory persistence with real transactional semantics — the test double,
    and a complete deployment at prototype fidelity.
``comparators``
    Each analysis component presented as a ``StreamComparator``. The single
    place where a score becomes a calibrated likelihood ratio.
``fusion_provider``
    A fitted fusion model, plus the naive comparator and a fused uncertainty
    interval obtained by correlated resampling.
``retention``
    Enforced expiry, with deletion logged before it occurs.
``clock``
    Injectable time.
``settings``
    Typed configuration from the environment.
"""

from viflap.infrastructure.audit import FileAuditLog
from viflap.infrastructure.clock import FixedClock, SystemClock
from viflap.infrastructure.comparators import (
    AcousticStreamComparator,
    BehaviouralStreamComparator,
    CalibratedStreamComparator,
    DeviceStreamComparator,
    TemporalStreamComparator,
    TransactionalStreamComparator,
)
from viflap.infrastructure.fusion_provider import FittedFusionProvider
from viflap.infrastructure.memory_repositories import (
    InMemoryStore,
    InMemoryUnitOfWork,
)
from viflap.infrastructure.retention import (
    RetentionPolicy,
    RetentionSchedule,
    RetentionStatus,
    RetentionSweep,
)

__all__ = [
    "AcousticStreamComparator",
    "BehaviouralStreamComparator",
    "CalibratedStreamComparator",
    "DeviceStreamComparator",
    "FileAuditLog",
    "FittedFusionProvider",
    "FixedClock",
    "InMemoryStore",
    "InMemoryUnitOfWork",
    "RetentionPolicy",
    "RetentionSchedule",
    "RetentionStatus",
    "RetentionSweep",
    "SystemClock",
    "TemporalStreamComparator",
    "TransactionalStreamComparator",
]
