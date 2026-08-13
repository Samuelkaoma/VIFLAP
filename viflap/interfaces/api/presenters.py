"""Converting domain results into wire responses.

Kept out of the route handlers so there is exactly one place where a
:class:`~viflap.domain.linkage.ComparisonResult` becomes JSON. If two routes
each built their own response, one of them would eventually omit the prior, and
the omission would be invisible in review because both would look complete.
"""

from __future__ import annotations

import math

from viflap.application.search import SearchResults
from viflap.domain.evidence import EvidenceStream, StreamAbsent, StreamEvidence
from viflap.domain.hypotheses import PriorOdds
from viflap.domain.linkage import ComparisonResult
from viflap.interfaces.api.schemas import (
    ComparisonResponse,
    PriorContext,
    SearchResponse,
    StreamContribution,
)

__all__ = ["present_comparison", "present_prior", "present_search"]

_LN10 = math.log(10.0)


def present_prior(prior: PriorOdds) -> PriorContext:
    return PriorContext(
        log_odds=prior.log_odds.value,
        odds_description=str(prior.log_odds),
        basis=prior.basis.value,
        justification=prior.justification,
        supplied_by=prior.supplied_by,
        population_size=prior.population_size,
    )


def present_comparison(result: ComparisonResult) -> ComparisonResponse:
    """Render a comparison result, with nothing load-bearing left out."""
    streams: list[StreamContribution] = []
    for stream in EvidenceStream.ordered():
        outcome = result.outcomes.get(stream)
        if outcome is None:
            continue
        if isinstance(outcome, StreamEvidence):
            streams.append(
                StreamContribution(
                    stream=stream.value,
                    status="evidence",
                    log10_lr=outcome.log_lr.log10,
                    interval_lower_log10=outcome.uncertainty.lower / _LN10,
                    interval_upper_log10=outcome.uncertainty.upper / _LN10,
                    model_id=outcome.model_id,
                    diagnostics={
                        key: float(value)
                        for key, value in outcome.diagnostics.items()
                        if isinstance(value, (int, float)) and math.isfinite(float(value))
                    },
                )
            )
        elif isinstance(outcome, StreamAbsent):
            streams.append(
                StreamContribution(
                    stream=stream.value,
                    status="absent",
                    absence_reason=outcome.reason.value,
                    absence_detail=outcome.detail,
                )
            )

    probability_bounds = result.posterior.probability_interval
    inflation = (
        result.overstatement.exaggeration_log10
        if result.overstatement is not None
        else None
    )
    naive_log10 = (
        result.overstatement.naive_log_lr.log10
        if result.overstatement is not None
        else None
    )

    return ComparisonResponse(
        incident_a=result.pair.first.value,
        incident_b=result.pair.second.value,
        case_reference=result.pair.authorised_by.value,
        fused_log10_lr=result.fused_log_lr.log10,
        interval_lower_log10=result.uncertainty.lower / _LN10,
        interval_upper_log10=result.uncertainty.upper / _LN10,
        verbal_summary=result.verbal_summary(),
        prior=present_prior(result.prior),
        posterior_log_odds=result.posterior.posterior_log_odds.value,
        posterior_probability=result.posterior.probability.value,
        posterior_probability_lower=(
            probability_bounds[0].value if probability_bounds else None
        ),
        posterior_probability_upper=(
            probability_bounds[1].value if probability_bounds else None
        ),
        streams=streams,
        contributing_stream_count=result.contributing_stream_count,
        rests_on_single_stream=result.rests_on_single_stream,
        acoustic_excluded_by_validity_gate=result.acoustic_was_excluded,
        fusion_method=result.fusion_method.value,
        fusion_model_id=result.fusion_model_id,
        naive_log10_lr=naive_log10,
        independence_inflation_log10=inflation,
        caveats=list(result.caveats()),
        computed_at=result.computed_at.isoformat(),
    )


def present_search(results: SearchResults, case_reference: str) -> SearchResponse:
    return SearchResponse(
        probe_incident_id=results.probe.value,
        case_reference=case_reference,
        results=[present_comparison(item) for item in results.results],
        n_candidates_compared=results.n_candidates_compared,
        n_declined=results.n_declined,
        prior=present_prior(results.prior),
        mandatory_caveat=results.caveat(),
        searched_at=results.searched_at,
    )
