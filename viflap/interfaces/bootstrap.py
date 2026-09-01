"""Assembling a runnable deployment.

Composition root: the one place that knows about every layer at once. Everything
else receives what it needs and constructs nothing, which is what keeps the
dependency arrows pointing inward.

Two assemblies are offered, and the distinction between them is enforced rather
than documented.

:func:`build_demonstration_container` builds a system with **untrained**
calibrators. It exists so the API, the governance constraints and the interface
can be exercised without model files. Every comparison it performs returns
*absence* for every stream, because an unfitted calibrator refuses to produce a
likelihood ratio. That is the designed behaviour: a demonstration deployment
cannot be mistaken for an operational one, because it produces no evidence at
all rather than plausible-looking numbers.

:func:`build_container` requires trained artefacts and refuses to start without
them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from viflap.application.comparison import CompareIncidents
from viflap.application.ingestion import IngestIncident
from viflap.application.ports import EvidenceBundle, StreamComparator
from viflap.application.search import SearchDatabase
from viflap.domain.errors import ConfigurationError
from viflap.domain.evidence import EvidenceStream
from viflap.infrastructure.audit import FileAuditLog
from viflap.infrastructure.clock import SystemClock
from viflap.infrastructure.fusion_provider import FittedFusionProvider
from viflap.infrastructure.memory_repositories import InMemoryStore, InMemoryUnitOfWork
from viflap.interfaces.api.dependencies import (
    ApplicationContainer,
    HeaderPrincipalResolver,
)

__all__ = ["build_container", "build_demonstration_container"]


def build_container(
    comparators: Sequence[StreamComparator],
    fusion: FittedFusionProvider,
    audit_path: Path | str,
    extractors: Mapping[EvidenceStream, object] | None = None,
    store: InMemoryStore | None = None,
) -> ApplicationContainer:
    """Assemble a container from trained components.

    Raises
    ------
    ConfigurationError
        If no comparator is supplied. A deployment with no evidence streams
        would start, accept queries, and decline every one — which looks like a
        data problem and is a configuration problem, so it fails at startup
        instead.
    """
    if not comparators:
        raise ConfigurationError(
            "no stream comparators were supplied; the deployment would accept "
            "queries and decline every one"
        )

    unit_of_work = InMemoryUnitOfWork(store)
    audit = FileAuditLog(audit_path)
    clock = SystemClock()

    compare = CompareIncidents(
        comparators=list(comparators), fusion=fusion, audit=audit, clock=clock
    )
    search = SearchDatabase(
        comparison=compare,
        incidents=unit_of_work.incidents,
        evidence=unit_of_work.evidence,
        audit=audit,
        clock=clock,
    )
    ingest = (
        IngestIncident(
            extractors=dict(extractors or {}),
            unit_of_work=unit_of_work,
            audit=audit,
            clock=clock,
        )
        if extractors is not None
        else None
    )

    return ApplicationContainer(
        compare=compare,
        search=search,
        ingest=ingest,
        incidents=unit_of_work.incidents,
        evidence=unit_of_work.evidence,
        audit=audit,
        clock=clock,
        principals=HeaderPrincipalResolver(),
    )


def build_demonstration_container(audit_path: Path | str) -> ApplicationContainer:
    """Assemble a container with untrained models, for interface work only.

    Produces a system that runs and answers, and whose every answer is an
    absence. This is deliberate. The alternative — seeding it with a plausible
    calibration so the demonstration produces numbers — would create a
    deployment whose output is indistinguishable in form from an operational
    one and meaningless in content, and someone would eventually screenshot it.
    """
    from viflap.analysis.calibration.calibrators import LogisticCalibrator
    from viflap.analysis.fusion.models import LinearLogisticFusion
    from viflap.infrastructure.comparators import CalibratedStreamComparator

    class _UnfittedComparator(CalibratedStreamComparator):
        def _score(
            self, first: EvidenceBundle, second: EvidenceBundle
        ) -> tuple[float, Mapping[str, float]]:  # pragma: no cover - never reached
            raise AssertionError("an unfitted comparator must refuse before scoring")

    comparators = [
        _UnfittedComparator(stream, LogisticCalibrator(), f"demonstration-{stream.value}")
        for stream in EvidenceStream.ordered()
    ]

    class _UnfittedFusion(FittedFusionProvider):
        def supports_pattern(self, pattern: object) -> bool:
            return bool(pattern)

    fusion = _UnfittedFusion(LinearLogisticFusion())
    return build_container(comparators, fusion, audit_path, extractors={})
