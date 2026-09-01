"""Dependency wiring and request authentication.

The container is built once at application startup and injected into handlers.
Handlers therefore construct nothing and reach for no globals, which is what
makes them testable against an in-memory container without a database, a model
file or a network.

On authentication
-----------------
This module resolves a request to a :class:`~viflap.domain.governance.Principal`
using headers. That is **not** authentication and it is labelled as such in the
one place it could be mistaken for it: a deployment must place this service
behind an authenticating proxy, or replace :class:`HeaderPrincipalResolver` with
one that verifies a signed assertion.

The distinction matters because the domain's separation-of-duties rules are
enforced against whatever principal is resolved. Those rules are only as strong
as the claim that the principal is who it says it is, and stating the boundary
here is more useful than a header-based scheme that looks like security.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Protocol, cast

from fastapi import Depends, Header, Request

from viflap.application.comparison import CompareIncidents
from viflap.application.ingestion import IngestIncident
from viflap.application.ports import (
    AuditSink,
    Clock,
    EvidenceRepository,
    IncidentRepository,
)
from viflap.application.search import SearchDatabase
from viflap.domain.errors import AuthorityViolation
from viflap.domain.governance import (
    DEFAULT_CASE_REFERENCE_FORMAT,
    AnalystRole,
    CaseReference,
    CaseReferenceFormat,
    Principal,
)

__all__ = [
    "ApplicationContainer",
    "HeaderPrincipalResolver",
    "PrincipalResolver",
    "get_container",
    "resolve_principal",
]


class PrincipalResolver(Protocol):
    """Turns request headers into an authenticated principal."""

    def resolve(self, actor_id: str | None, roles: str | None) -> Principal: ...


class HeaderPrincipalResolver:
    """Resolves a principal from ``X-Analyst-Id`` and ``X-Analyst-Roles``.

    For development and for deployment behind an authenticating proxy that sets
    these headers and strips any the client supplied. It performs no
    verification whatever, and a deployment that exposes this service directly
    has no access control at all.

    It does still enforce separation of duties, because
    :class:`~viflap.domain.governance.Principal` cannot be constructed with an
    incompatible role combination — so even an unverified claim cannot assert a
    combination the architecture forbids.
    """

    def __init__(self, allowed_roles: Sequence[AnalystRole] | None = None) -> None:
        self._allowed = frozenset(allowed_roles or list(AnalystRole))

    def resolve(self, actor_id: str | None, roles: str | None) -> Principal:
        if not actor_id or not actor_id.strip():
            raise AuthorityViolation(
                "no analyst identity was supplied; every operation on evidence "
                "is attributed to a person and there is no anonymous path"
            )
        if not roles or not roles.strip():
            raise AuthorityViolation(
                "no roles were supplied for this analyst", actor_id=actor_id
            )

        parsed: set[AnalystRole] = set()
        for token in roles.split(","):
            name = token.strip().lower()
            if not name:
                continue
            try:
                role = AnalystRole(name)
            except ValueError as exc:
                raise AuthorityViolation(
                    "unknown analyst role",
                    role=name,
                    known=sorted(item.value for item in AnalystRole),
                ) from exc
            if role not in self._allowed:
                raise AuthorityViolation(
                    "this role is not enabled in this deployment", role=name
                )
            parsed.add(role)

        # Principal's constructor applies the separation-of-duties rules, so an
        # incompatible combination is rejected here rather than at the point of
        # use — before any evidence is touched.
        return Principal(identifier=actor_id.strip(), roles=frozenset(parsed))


@dataclass
class ApplicationContainer:
    """Everything the API needs, assembled once at startup."""

    compare: CompareIncidents
    search: SearchDatabase
    ingest: IngestIncident | None
    incidents: IncidentRepository
    evidence: EvidenceRepository
    audit: AuditSink
    clock: Clock
    principals: PrincipalResolver
    case_format: CaseReferenceFormat = DEFAULT_CASE_REFERENCE_FORMAT

    def parse_case_reference(self, raw: str | None) -> CaseReference:
        """Parse a case reference under this deployment's format."""
        return CaseReference.parse(raw, self.case_format)


def get_container(request: Request) -> ApplicationContainer:
    """Retrieve the container from application state."""
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - startup misconfiguration
        raise RuntimeError(
            "the application container was not installed; build the app with "
            "create_app(container=...)"
        )
    # ``app.state`` is untyped, so what comes back is Any. The None check
    # above is what actually establishes the type; the cast records it.
    return cast(ApplicationContainer, container)


def resolve_principal(
    container: Annotated[ApplicationContainer, Depends(get_container)],
    x_analyst_id: Annotated[str | None, Header()] = None,
    x_analyst_roles: Annotated[str | None, Header()] = None,
) -> Principal:
    """Resolve the calling principal, or refuse."""
    return container.principals.resolve(x_analyst_id, x_analyst_roles)


def require_case_reference(
    container: Annotated[ApplicationContainer, Depends(get_container)],
    x_case_reference: Annotated[str | None, Header()] = None,
) -> CaseReference:
    """Require a valid case reference on the request.

    Applied as a dependency rather than checked inside handlers. A handler that
    forgot the check would still work; a handler missing this dependency does
    not receive the parameter it needs, so the omission is a failure at import
    rather than a governance gap discovered later.
    """
    return container.parse_case_reference(x_case_reference)
