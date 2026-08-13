"""The database search use case.

Search is the operation this system exists to support and the one most capable
of doing harm, and the two facts have the same cause. A search against ``N``
enrolled incidents produces ``N`` comparisons, so it produces the extreme tail of
the score distribution by construction. The strongest-looking result in any large
search is, more often than not, the most extreme *different-source* comparison
rather than a genuine linkage.

The arithmetic, restated because it governs everything here: against 100,000
enrolled entries under a uniform prior, an acoustic likelihood ratio of 1,000
leaves the same-source proposition at a posterior probability of about one
percent. Ranked first, presented without its prior, it looks like an answer.

What this use case does about it
--------------------------------
- The prior is derived from the **actual** enrolled population size at query
  time, not from a figure supplied by the caller. A caller who could set ``N``
  could set the posterior.
- Ranking is by posterior, never by likelihood ratio. Two candidates with the
  same likelihood ratio have the same posterior only under the same prior, and
  ranking by the ratio invites reading the top of the list as the answer.
- Every returned candidate carries its own full context, so a result cannot be
  detached from the search that produced it.
- The number of comparisons performed is reported with the results, because it
  is what makes an extreme value unsurprising.
- Results are not truncated to a threshold by default. Imposing one inside the
  system pre-empts a judgement belonging to the investigator; where the caller
  asks for one, it is recorded in the audit entry.
"""

from __future__ import annotations

from dataclasses import dataclass

from viflap.application.comparison import CompareIncidents, ComparisonRequest
from viflap.application.ports import (
    AuditRecord,
    AuditSink,
    Clock,
    EvidenceRepository,
    IncidentRepository,
)
from viflap.domain.errors import (
    EntityNotFoundError,
    InsufficientDataError,
    InvalidEvidenceError,
)
from viflap.domain.governance import Authority, CaseReference, Principal
from viflap.domain.hypotheses import PriorOdds, SearchMode
from viflap.domain.linkage import ComparisonResult, IncidentId, IncidentPair

__all__ = ["SearchDatabase", "SearchRequest", "SearchResults"]


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """A request to search the enrolled population for linkages."""

    probe: IncidentId
    case_reference: CaseReference
    requested_by: Principal
    max_results: int = 50
    restricted_population: int | None = None
    """If the investigator has narrowed the relevant population on
    non-acoustic grounds, its size and their justification. Supplying it changes
    the prior, so it must be accompanied by a reason and is attributed to them
    rather than to the system."""

    restriction_justification: str = ""

    def __post_init__(self) -> None:
        # Domain errors, not ValueError. These are governance rules rather than
        # argument-validation accidents, so they carry structured context and are
        # mapped to a meaningful status by the API's error handlers. A bare
        # ValueError reaches the transport layer unhandled and becomes a 500,
        # presenting a correctly refused request as a server fault.
        if self.max_results < 1:
            raise InvalidEvidenceError(
                "max_results must be at least one", max_results=self.max_results
            )
        if (
            self.restricted_population is not None
            and not self.restriction_justification.strip()
        ):
            raise InvalidEvidenceError(
                "restricting the relevant population changes the prior and "
                "therefore the posterior; a justification is required and is "
                "attributed to the person who supplied it",
                restricted_population=self.restricted_population,
            )


@dataclass(frozen=True, slots=True)
class SearchResults:
    """Ranked candidates, with the context that makes them interpretable."""

    probe: IncidentId
    results: tuple[ComparisonResult, ...]
    n_candidates_compared: int
    n_declined: int
    """Candidates that produced no comparison — no shared evidence stream, or
    too little data. Reported because a search over 900 candidates that could
    only compare 40 of them is a different search from one that compared all
    900, and the difference is invisible from the results alone."""

    prior: PriorOdds
    searched_at: str

    @property
    def is_empty(self) -> bool:
        return not self.results

    def caveat(self) -> str:
        """The statement that must accompany any presentation of these results.

        Assembled from the search's own parameters so that a presentation layer
        cannot omit it by forgetting to.
        """
        top = self.results[0] if self.results else None
        lines = [
            f"This search compared the probe incident against "
            f"{self.n_candidates_compared:,} enrolled candidates. The prior odds "
            f"applied to each were {self.prior.log_odds} "
            f"({self.prior.justification})."
        ]
        if top is not None:
            lines.append(
                f"The highest-ranked candidate has a fused likelihood ratio of "
                f"10^{top.fused_log_lr.log10:.2f}, giving a posterior probability "
                f"of {top.posterior.probability}."
            )
            if top.requires_prominent_prior_warning:
                lines.append(
                    "Even so, the same-source proposition remains less probable "
                    "than not. A search of this size is expected to produce its "
                    "most extreme result from an unrelated incident."
                )
        if self.n_declined:
            lines.append(
                f"{self.n_declined:,} candidates could not be compared and are "
                f"absent from this ranking; they have not been excluded on the "
                f"evidence."
            )
        return " ".join(lines)


class SearchDatabase:
    """Searches the enrolled population for linkages to a probe incident."""

    def __init__(
        self,
        comparison: CompareIncidents,
        incidents: IncidentRepository,
        evidence: EvidenceRepository,
        audit: AuditSink,
        clock: Clock,
    ) -> None:
        self._comparison = comparison
        self._incidents = incidents
        self._evidence = evidence
        self._audit = audit
        self._clock = clock

    def execute(self, request: SearchRequest) -> SearchResults:
        """Run the search.

        Raises
        ------
        AuthorityViolation
            Before any evidence is read.
        EntityNotFoundError
            If the probe is not enrolled.
        InsufficientDataError
            If the enrolled population is too small for a search to mean
            anything.
        """
        request.requested_by.require(Authority.QUERY)

        if not self._incidents.exists(request.probe):
            self._record(request, outcome="failed: probe not enrolled", extra={})
            raise EntityNotFoundError(
                "the probe incident is not enrolled", incident=request.probe.value
            )

        population = self._incidents.count()
        if population < 2:
            raise InsufficientDataError(
                "a database search requires at least two enrolled incidents",
                enrolled=population,
            )

        prior = self._build_prior(request, population)
        probe_bundle = self._evidence.load(request.probe)

        results: list[ComparisonResult] = []
        compared = 0
        declined = 0

        for candidate in self._incidents.list_enrolled():
            if candidate.incident_id == request.probe:
                continue
            compared += 1
            try:
                bundle = self._evidence.load(candidate.incident_id)
                comparison_request = ComparisonRequest(
                    pair=IncidentPair(
                        first=request.probe,
                        second=candidate.incident_id,
                        authorised_by=request.case_reference,
                    ),
                    prior=prior,
                    requested_by=request.requested_by,
                )
                results.append(
                    self._comparison.execute(comparison_request, probe_bundle, bundle)
                )
            except (InsufficientDataError, EntityNotFoundError):
                # A candidate sharing no evidence stream with the probe is not a
                # candidate that was ruled out. Counting it separately keeps the
                # two apart in the report.
                declined += 1

        # Ranked by posterior, not by likelihood ratio. Under a single prior the
        # two orderings coincide, but ranking by the posterior is what keeps the
        # prior attached to the ordering rather than applied to it afterwards.
        results.sort(key=lambda item: item.posterior.posterior_log_odds.value, reverse=True)
        top = tuple(results[: request.max_results])

        self._record(
            request,
            outcome="ok",
            extra={
                "n_candidates_compared": compared,
                "n_declined": declined,
                "n_returned": len(top),
                "population_size": population,
                "prior_log_odds": round(prior.log_odds.value, 6),
                "top_log10_lr": (round(top[0].fused_log_lr.log10, 4) if top else None),
            },
        )

        return SearchResults(
            probe=request.probe,
            results=top,
            n_candidates_compared=compared,
            n_declined=declined,
            prior=prior,
            searched_at=self._clock.now().isoformat(),
        )

    def _build_prior(self, request: SearchRequest, population: int) -> PriorOdds:
        """Derive the prior from the actual population, or from a stated restriction.

        The unrestricted case uses the enrolled count read at query time. Taking
        it from the caller would let the caller choose the posterior, which is
        the one parameter a requester must not control.
        """
        if request.restricted_population is None:
            return PriorOdds.uniform_over_database(population)
        return PriorOdds.restricted_population(
            population_size=request.restricted_population,
            justification=request.restriction_justification,
            supplied_by=request.requested_by.identifier,
            search_mode=SearchMode.DATABASE_SEARCH,
        )

    def _record(
        self, request: SearchRequest, outcome: str, extra: dict[str, object]
    ) -> None:
        self._audit.record(
            AuditRecord(
                timestamp=self._clock.now(),
                actor_id=request.requested_by.identifier,
                actor_roles=tuple(
                    sorted(role.value for role in request.requested_by.roles)
                ),
                action="search_database",
                case_reference=request.case_reference.value,
                parameters={
                    "probe": request.probe.value,
                    "max_results": request.max_results,
                    "restricted_population": request.restricted_population,
                    "restriction_justification": request.restriction_justification,
                    **extra,
                },
                outcome=outcome,
            )
        )
