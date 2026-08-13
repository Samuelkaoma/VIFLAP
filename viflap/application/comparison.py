"""The comparison use case: two incidents in, one contextualised result out.

This is where the pieces meet, and where the system's safety properties are
enforced as control flow rather than as documentation:

1. **Authority is checked before anything is read.** Not before the result is
   returned — before the evidence is touched. An unauthorised query must not
   even cause an evidence read, because the read itself is a disclosure.
2. **The validity gate runs before the acoustic stream is consulted**, and its
   verdict removes that stream rather than reducing it.
3. **Streams that produce nothing are recorded as absent, with a reason.** They
   are never given a likelihood ratio of one.
4. **A prior is required.** There is no code path producing a result without
   one, because the type that carries the result cannot be constructed without a
   posterior, and a posterior cannot be constructed without a prior.
5. **What naive summation would have claimed is computed and attached** to every
   multi-stream result.
6. **The query is audited whether or not it succeeded.** A failed or empty query
   is exactly what an oversight body needs to see.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from viflap.application.ports import (
    AuditRecord,
    AuditSink,
    Clock,
    EvidenceBundle,
    FusionProvider,
    StreamComparator,
)
from viflap.domain.errors import (
    InsufficientDataError,
    ModelNotTrainedError,
    StreamUnavailableError,
)
from viflap.domain.evidence import (
    AbsenceReason,
    EvidenceStream,
    StreamAbsent,
    StreamEvidence,
    StreamOutcome,
)
from viflap.domain.governance import Authority, Principal
from viflap.domain.hypotheses import PosteriorAssessment, PriorOdds
from viflap.domain.linkage import (
    ComparisonResult,
    FusionMethod,
    IncidentPair,
    OverstatementEstimate,
)
from viflap.domain.values import LogLikelihoodRatio, UncertaintyInterval

__all__ = ["CompareIncidents", "ComparisonRequest"]


@dataclass(frozen=True, slots=True)
class ComparisonRequest:
    """A request to compare two incidents.

    The prior is part of the request, not a default supplied by the service.
    That placement is deliberate: choosing a prior is a judgement, the judgement
    belongs to the investigator, and making it a parameter of the request means
    it is recorded with the query rather than inferred afterwards from
    configuration.
    """

    pair: IncidentPair
    prior: PriorOdds
    requested_by: Principal


class CompareIncidents:
    """Compares two incidents across all registered streams."""

    def __init__(
        self,
        comparators: Sequence[StreamComparator],
        fusion: FusionProvider,
        audit: AuditSink,
        clock: Clock,
    ) -> None:
        if not comparators:
            raise ValueError("at least one stream comparator is required")
        duplicates = [
            stream
            for stream in EvidenceStream.ordered()
            if sum(1 for c in comparators if c.stream is stream) > 1
        ]
        if duplicates:
            raise ValueError(
                f"more than one comparator registered for: "
                f"{[stream.value for stream in duplicates]}"
            )
        self._comparators = tuple(comparators)
        self._fusion = fusion
        self._audit = audit
        self._clock = clock

    @property
    def registered_streams(self) -> tuple[EvidenceStream, ...]:
        return tuple(comparator.stream for comparator in self._comparators)

    def execute(
        self,
        request: ComparisonRequest,
        first: EvidenceBundle,
        second: EvidenceBundle,
    ) -> ComparisonResult:
        """Run the comparison.

        Raises
        ------
        AuthorityViolation
            If the principal may not run queries. Raised before any evidence is
            read.
        InsufficientDataError
            If no stream produced evidence. The system declines rather than
            returning a neutral result, because a neutral result is
            indistinguishable from a genuine finding of no support and this is
            not that.
        """
        request.requested_by.require(Authority.QUERY)

        try:
            outcomes = self._collect_outcomes(first, second)
            result = self._fuse(request, outcomes, first, second)
        except Exception as exc:
            self._record(request, outcome=f"failed: {type(exc).__name__}", extra={})
            raise

        self._record(
            request,
            outcome="ok",
            extra={
                "fused_log10_lr": round(result.fused_log_lr.log10, 4),
                "contributing_streams": sorted(
                    stream.value for stream in result.contributing
                ),
                "absent_streams": sorted(stream.value for stream in result.absent),
                "fusion_method": result.fusion_method.value,
                "posterior_probability": round(result.posterior.probability.value, 6),
            },
        )
        return result

    # -- Internals --------------------------------------------------------

    def _collect_outcomes(
        self, first: EvidenceBundle, second: EvidenceBundle
    ) -> dict[EvidenceStream, StreamOutcome]:
        """Run every registered comparator, converting refusals into absences."""
        outcomes: dict[EvidenceStream, StreamOutcome] = {}

        for comparator in self._comparators:
            stream = comparator.stream

            gated = self._validity_absence(stream, first, second)
            if gated is not None:
                outcomes[stream] = gated
                continue

            if not (first.has(stream) and second.has(stream)):
                outcomes[stream] = StreamAbsent(
                    stream=stream,
                    reason=AbsenceReason.NO_DATA,
                    detail=self._describe_missing(stream, first, second),
                )
                continue

            try:
                outcomes[stream] = comparator.compare(first, second)
            except InsufficientDataError as exc:
                outcomes[stream] = StreamAbsent(
                    stream=stream,
                    reason=AbsenceReason.INSUFFICIENT_DATA,
                    detail=exc.message,
                )
            except (ModelNotTrainedError, StreamUnavailableError) as exc:
                # A stream whose model is not deployed is absent, not neutral.
                # Failing the whole comparison would be worse: the other streams
                # have evidence to offer and an investigator should get it.
                outcomes[stream] = StreamAbsent(
                    stream=stream,
                    reason=AbsenceReason.MODEL_UNAVAILABLE,
                    detail=exc.message,
                )
        return outcomes

    def _validity_absence(
        self,
        stream: EvidenceStream,
        first: EvidenceBundle,
        second: EvidenceBundle,
    ) -> StreamAbsent | None:
        """Exclude a gated stream when either recording fails the validity gate.

        Either, not both. If one of the two recordings is synthetic there is no
        pair of human vocal tracts to compare, so the acoustic evidence for the
        *pair* is void regardless of the other recording's provenance.
        """
        if not stream.is_gated_by_validity:
            return None

        for bundle in (first, second):
            assessment = bundle.validity
            if assessment is None:
                continue
            if not assessment.verdict.permits_acoustic_evidence:
                return StreamAbsent(
                    stream=stream,
                    reason=AbsenceReason.EXCLUDED_BY_VALIDITY_GATE,
                    detail=(
                        f"Recording {assessment.recording_id} was not admitted by "
                        f"the validity gate (verdict: "
                        f"{assessment.verdict.value}). Acoustic evidence was "
                        f"removed from fusion rather than down-weighted, because "
                        f"a recording that is not human speech carries no "
                        f"information about a human speaker."
                    ),
                )
        return None

    @staticmethod
    def _describe_missing(
        stream: EvidenceStream, first: EvidenceBundle, second: EvidenceBundle
    ) -> str:
        missing = [
            bundle.incident_id.value for bundle in (first, second) if not bundle.has(stream)
        ]
        return (
            f"No {stream.display_name.lower()} evidence was extracted for "
            f"{' and '.join(missing)}."
        )

    def _fuse(
        self,
        request: ComparisonRequest,
        outcomes: Mapping[EvidenceStream, StreamOutcome],
        first: EvidenceBundle,
        second: EvidenceBundle,
    ) -> ComparisonResult:
        """Combine the stream outcomes into a contextualised result."""
        contributing = {
            stream: outcome
            for stream, outcome in outcomes.items()
            if isinstance(outcome, StreamEvidence)
        }

        if not contributing:
            raise InsufficientDataError(
                "no evidence stream produced a result for this pair; the system "
                "declines rather than reporting a neutral likelihood ratio, "
                "which would be indistinguishable from a finding of no support",
                absent=sorted(stream.value for stream in outcomes),
            )

        log_lrs = {
            stream: evidence.log_lr.value for stream, evidence in contributing.items()
        }
        pattern = frozenset(log_lrs)

        notes: list[str] = []
        if not self._fusion.supports_pattern(pattern):
            raise InsufficientDataError(
                "the fusion model has no fitted handling for this combination of "
                "available streams, and will not impute the missing ones",
                pattern=sorted(stream.value for stream in pattern),
            )

        fused_value = self._fusion.fuse(log_lrs)
        fused = LogLikelihoodRatio(fused_value)

        lower, upper = self._fusion.uncertainty_for(log_lrs, fused_value)
        uncertainty = UncertaintyInterval(
            lower=min(lower, fused_value), upper=max(upper, fused_value)
        )

        overstatement: OverstatementEstimate | None = None
        method = FusionMethod.LINEAR_LOGISTIC
        if len(log_lrs) > 1:
            overstatement = OverstatementEstimate(
                naive_log_lr=LogLikelihoodRatio(self._fusion.fuse_naive(log_lrs)),
                corrected_log_lr=fused,
            )

        posterior = PosteriorAssessment.from_evidence(request.prior, fused, uncertainty)

        validity_assessments = tuple(
            bundle.validity for bundle in (first, second) if bundle.validity is not None
        )

        for stream, evidence in contributing.items():
            if evidence.is_weakly_determined:
                notes.append(
                    f"The {stream.display_name.lower()} stream's interval spans "
                    f"the neutral point; its direction of support is not "
                    f"reliably determined."
                )

        return ComparisonResult(
            pair=request.pair,
            fused_log_lr=fused,
            uncertainty=uncertainty,
            outcomes=dict(outcomes),
            posterior=posterior,
            fusion_method=getattr(self._fusion, "method", method),
            fusion_model_id=self._fusion.model_id,
            validity_assessments=validity_assessments,
            computed_at=self._clock.now(),
            overstatement=overstatement,
            notes=tuple(notes),
        )

    def _record(
        self,
        request: ComparisonRequest,
        outcome: str,
        extra: Mapping[str, object],
    ) -> None:
        """Write the audit entry for this query.

        The prior is recorded with the query. Reconstructing later what prior a
        historical result rested on is otherwise guesswork, and the prior is the
        difference between a likelihood ratio and a conclusion.
        """
        self._audit.record(
            AuditRecord(
                timestamp=self._clock.now(),
                actor_id=request.requested_by.identifier,
                actor_roles=tuple(
                    sorted(role.value for role in request.requested_by.roles)
                ),
                action="compare_incidents",
                case_reference=request.pair.authorised_by.value,
                parameters={
                    "incident_a": request.pair.first.value,
                    "incident_b": request.pair.second.value,
                    "prior_log_odds": round(request.prior.log_odds.value, 6),
                    "prior_basis": request.prior.basis.value,
                    "prior_supplied_by": request.prior.supplied_by,
                    "search_mode": request.prior.search_mode.value,
                    **extra,
                },
                outcome=outcome,
            )
        )
