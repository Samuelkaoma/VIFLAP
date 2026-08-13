"""The HTTP interface.

An application factory, not a module-level ``app``. A module-level instance
constructs its dependencies at import time, which means importing the module to
read one route opens a database connection and loads model files — and means a
test cannot substitute anything.

Routes are thin. Every one resolves a principal, parses a case reference, calls
a use case, and renders the result through :mod:`presenters`. Nothing here
computes evidence, checks authority, or decides what a result means: those are
the use case's and the domain's, and duplicating any of them at the transport
layer would create a second place for the rules to be enforced differently.

What has deliberately not been implemented
------------------------------------------
There is no endpoint accepting a live audio stream, no bulk import, and no route
returning a likelihood ratio without its prior. Those absences are the
architectural constraints of section 9.3 of the proposal, expressed as missing
functions rather than as configuration an operator could change.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, Query, Request, status
from fastapi.responses import JSONResponse

from viflap.application.comparison import ComparisonRequest
from viflap.application.search import SearchRequest
from viflap.domain.errors import (
    AuthorityViolation,
    CaseBindingViolation,
    EntityNotFoundError,
    GovernanceViolation,
    InsufficientDataError,
    InvalidEvidenceError,
    OutputConstraintViolation,
    SeparationOfDutiesViolation,
    ViflapError,
)
from viflap.domain.governance import Authority, CaseReference, Principal
from viflap.domain.hypotheses import PriorOdds
from viflap.domain.linkage import IncidentId, IncidentPair
from viflap.interfaces.api.dependencies import (
    ApplicationContainer,
    get_container,
    require_case_reference,
    resolve_principal,
)
from viflap.interfaces.api.presenters import present_comparison, present_search
from viflap.interfaces.api.schemas import (
    AuditEntryResponse,
    AuditVerificationResponse,
    ComparisonResponse,
    ErrorResponse,
    SearchRequestBody,
    SearchResponse,
)

__all__ = ["create_app"]

API_DESCRIPTION = """
VIFLAP evaluates the weight of evidence that two reported incidents were
conducted by the same actor. It reports a **calibrated likelihood ratio** and
never an identification.

Reading a result
----------------
A likelihood ratio states how much more probable the observed evidence is if the
incidents share an actor than if they do not. It is **not** the probability that
they share an actor. Converting one to the other requires prior odds, which
depend on the size and composition of the population searched.

Every response carrying a likelihood ratio also carries the prior odds that
applied, the resulting posterior, and the caveats that must accompany it. That
is not a convention this API follows; the response schema has no shape that
omits them.

For a search against 100,000 enrolled incidents, a likelihood ratio of 1,000
leaves the same-source proposition at a posterior probability of about **one
percent**. A ranking is a list of things to look at, not a list of answers.

Authentication
--------------
This service performs none. Deploy it behind an authenticating proxy that sets
the analyst headers and strips any supplied by the client.
"""


def create_app(container: ApplicationContainer, *, debug: bool = False) -> FastAPI:
    """Build the application around an assembled container."""
    app = FastAPI(
        title="VIFLAP",
        version="1.0.0",
        description=API_DESCRIPTION,
        debug=debug,
    )
    app.state.container = container

    _install_error_handlers(app)
    _install_routes(app)
    return app


def _install_error_handlers(app: FastAPI) -> None:
    """Map domain errors to status codes, preserving their structure.

    Governance violations are 403 rather than 400: the request was
    well-formed and the system refused it, and the distinction matters to
    whoever reads the logs. An output-constraint violation is 500, because it
    means a defect in this system produced language it must not produce — that
    is not the caller's error and must not be presented as one.
    """

    def respond(status_code: int, exc: ViflapError, remedy: str = "") -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error_type=type(exc).__name__,
                message=exc.message,
                context={key: str(value) for key, value in exc.context.items()},
                remedy=remedy,
            ).model_dump(),
        )

    @app.exception_handler(CaseBindingViolation)
    async def _case_binding(request: Request, exc: CaseBindingViolation) -> JSONResponse:
        return respond(
            status.HTTP_403_FORBIDDEN,
            exc,
            "Supply a valid complaint reference in the X-Case-Reference header. "
            "Every operation on evidence is bound to a filed complaint; there is "
            "no unbound path.",
        )

    @app.exception_handler(SeparationOfDutiesViolation)
    async def _separation(
        request: Request, exc: SeparationOfDutiesViolation
    ) -> JSONResponse:
        return respond(
            status.HTTP_403_FORBIDDEN,
            exc,
            "This combination of roles concentrates incompatible authority. "
            "Split the work between principals.",
        )

    @app.exception_handler(AuthorityViolation)
    async def _authority(request: Request, exc: AuthorityViolation) -> JSONResponse:
        return respond(
            status.HTTP_403_FORBIDDEN,
            exc,
            "Use a principal holding the authority this operation requires.",
        )

    @app.exception_handler(GovernanceViolation)
    async def _governance(request: Request, exc: GovernanceViolation) -> JSONResponse:
        return respond(status.HTTP_403_FORBIDDEN, exc)

    @app.exception_handler(EntityNotFoundError)
    async def _not_found(request: Request, exc: EntityNotFoundError) -> JSONResponse:
        return respond(status.HTTP_404_NOT_FOUND, exc)

    @app.exception_handler(InsufficientDataError)
    async def _insufficient(request: Request, exc: InsufficientDataError) -> JSONResponse:
        return respond(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            exc,
            "The system declines to report a value it cannot support. This is "
            "not a failure: a result produced from insufficient data would be "
            "indistinguishable from one produced from adequate data.",
        )

    @app.exception_handler(InvalidEvidenceError)
    async def _invalid(request: Request, exc: InvalidEvidenceError) -> JSONResponse:
        return respond(status.HTTP_400_BAD_REQUEST, exc)

    @app.exception_handler(OutputConstraintViolation)
    async def _output(request: Request, exc: OutputConstraintViolation) -> JSONResponse:
        # Deliberately a server error. Text asserting identity means a defect
        # upstream formed a conclusion it had no basis to form, and returning a
        # 4xx would attribute that to the caller.
        return respond(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            exc,
            "This is a defect in VIFLAP, not in the request. The system "
            "constructed text asserting identity and refused to emit it.",
        )


# Injected-dependency aliases, defined at module level.
#
# They must be module-level. ``from __future__ import annotations`` makes every
# annotation a string, and FastAPI resolves those strings against the *module*
# namespace when it builds each route's signature. An alias defined inside the
# route-installing function is a local, so FastAPI cannot resolve it, falls back
# to treating the parameter as an ordinary query parameter, and every request
# fails with "field required" — a failure that looks like a client error and is
# a wiring error.
Container = Annotated[ApplicationContainer, Depends(get_container)]
Caller = Annotated[Principal, Depends(resolve_principal)]
Case = Annotated[CaseReference, Depends(require_case_reference)]


def _install_routes(app: FastAPI) -> None:
    @app.get("/health", tags=["operations"])
    async def health(container: Container) -> dict[str, Any]:
        """Liveness and the facts an operator needs to trust the deployment."""
        verification = container.audit.verify()
        return {
            "status": "ok",
            "enrolled_incidents": container.incidents.count(),
            "audit_chain_intact": verification.is_intact,
            "audit_entries": verification.n_entries,
            "case_reference_format": container.case_format.description,
        }

    @app.post(
        "/api/v1/comparisons",
        response_model=ComparisonResponse,
        tags=["comparison"],
        summary="Compare two incidents",
    )
    async def compare(
        container: Container,
        caller: Caller,
        case: Case,
        incident_a: Annotated[str, Query(min_length=1)],
        incident_b: Annotated[str, Query(min_length=1)],
        restricted_population: Annotated[int | None, Query(ge=2)] = None,
        restriction_justification: Annotated[str, Query()] = "",
    ) -> ComparisonResponse:
        """Evaluate the evidence that two incidents share an actor.

        The prior is the uniform prior over the enrolled population unless the
        caller narrows the relevant population, which requires a justification
        and is attributed to them rather than to the system.
        """
        pair = IncidentPair(
            first=IncidentId(incident_a),
            second=IncidentId(incident_b),
            authorised_by=case,
        )
        if restricted_population is not None:
            if not restriction_justification.strip():
                raise InvalidEvidenceError(
                    "restricting the relevant population changes the prior and "
                    "therefore the posterior; a justification is required and is "
                    "attributed to the person who supplied it"
                )
            prior = PriorOdds.restricted_population(
                population_size=restricted_population,
                justification=restriction_justification,
                supplied_by=caller.identifier,
            )
        else:
            prior = PriorOdds.uniform_over_database(container.incidents.count())

        first = container.evidence.load(pair.first)
        second = container.evidence.load(pair.second)
        result = container.compare.execute(
            ComparisonRequest(pair=pair, prior=prior, requested_by=caller),
            first,
            second,
        )
        return present_comparison(result)

    @app.post(
        "/api/v1/searches",
        response_model=SearchResponse,
        tags=["comparison"],
        summary="Search the enrolled population",
    )
    async def search(
        container: Container,
        caller: Caller,
        body: SearchRequestBody,
    ) -> SearchResponse:
        """Rank enrolled incidents by the evidence that they share an actor.

        The response carries a mandatory caveat naming the search size and the
        prior. Presenting the ranking without it invites the reader to treat the
        top entry as an answer, which for a large search it is usually not.
        """
        case = container.parse_case_reference(body.case_reference)
        results = container.search.execute(
            SearchRequest(
                probe=IncidentId(body.probe_incident_id),
                case_reference=case,
                requested_by=caller,
                max_results=body.max_results,
                restricted_population=body.restricted_population,
                restriction_justification=body.restriction_justification,
            )
        )
        return present_search(results, case.value)

    @app.get(
        "/api/v1/audit",
        response_model=list[AuditEntryResponse],
        tags=["oversight"],
        summary="Read the audit trail",
    )
    async def read_audit(
        container: Container,
        caller: Caller,
        case_reference: Annotated[str | None, Query()] = None,
        actor_id: Annotated[str | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> list[AuditEntryResponse]:
        """Read audit entries.

        Requires the audit authority, which the domain forbids combining with
        any operational role. An auditor cannot enrol, query or export, and no
        operational principal can read this.
        """
        caller.require(Authority.AUDIT)
        entries = container.audit.query(
            case_reference=case_reference, actor_id=actor_id, limit=limit
        )
        return [
            AuditEntryResponse(
                timestamp=entry.timestamp.isoformat(),
                actor_id=entry.actor_id,
                actor_roles=list(entry.actor_roles),
                action=entry.action,
                case_reference=entry.case_reference,
                outcome=entry.outcome,
                parameters={k: str(v) for k, v in entry.parameters.items()},
            )
            for entry in entries
        ]

    @app.get(
        "/api/v1/audit/verification",
        response_model=AuditVerificationResponse,
        tags=["oversight"],
        summary="Verify the audit chain",
    )
    async def verify_audit(
        container: Container, caller: Caller
    ) -> AuditVerificationResponse:
        """Recompute the audit chain from its published genesis value."""
        caller.require(Authority.AUDIT)
        verification = container.audit.verify()
        return AuditVerificationResponse(
            is_intact=verification.is_intact,
            n_entries=verification.n_entries,
            first_broken_index=verification.first_broken_index,
            detail=verification.detail,
        )
