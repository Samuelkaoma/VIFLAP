"""Wire schemas.

Two properties are enforced by these types rather than by the handlers that use
them, so that no route can accidentally omit either.

**A likelihood ratio never appears without its prior.** There is no response
model containing ``fused_log10_lr`` and not containing the prior odds, the
posterior, and the search context. A client cannot render a bare likelihood
ratio from this API without having deliberately discarded the fields beside it.

**Strength is never rendered without direction.** The verbal band is a function
of magnitude, so ``10^-4`` is "very strong" — support for the *different-source*
proposition. The response carries a complete pre-rendered sentence rather than a
band the client is trusted to combine with a sign correctly.

Both are the prosecutor's fallacy prevented at the transport layer, which is
where an interface built by someone who has not read the proposal will first
encounter it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from viflap.domain.governance import OutputLanguagePolicy

__all__ = [
    "AuditEntryResponse",
    "AuditVerificationResponse",
    "ComparisonResponse",
    "ErrorResponse",
    "IngestRequest",
    "IngestResponse",
    "PriorContext",
    "SearchRequestBody",
    "SearchResponse",
    "StreamContribution",
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ErrorResponse(_Base):
    """A refusal, with enough context to act on it."""

    error_type: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    remedy: str = ""
    """What the caller should do differently. Governance refusals are the
    common case and are usually actionable — obtain a case reference, use a
    principal with the right authority — so saying so turns a rejection into
    instruction rather than an obstacle to route around."""


class PriorContext(_Base):
    """The prior odds of the search that produced a result.

    Mandatory on every response carrying a likelihood ratio. Not optional, not
    nullable: a result without this is the raw material of the prosecutor's
    fallacy, and the schema does not permit one.
    """

    log_odds: float
    odds_description: str
    basis: str
    justification: str
    supplied_by: str
    population_size: int | None = None


class StreamContribution(_Base):
    """What one stream contributed, or why it contributed nothing."""

    stream: str
    status: Literal["evidence", "absent"]
    log10_lr: float | None = None
    interval_lower_log10: float | None = None
    interval_upper_log10: float | None = None
    model_id: str | None = None
    absence_reason: str | None = None
    absence_detail: str | None = None
    diagnostics: dict[str, float] = Field(default_factory=dict)


class ComparisonResponse(_Base):
    """A comparison of two incidents, with everything needed to read it."""

    incident_a: str
    incident_b: str
    case_reference: str

    fused_log10_lr: float
    interval_lower_log10: float
    interval_upper_log10: float

    verbal_summary: str
    """A complete, policy-compliant sentence stating strength *and* direction.
    Clients should render this rather than composing their own from the numeric
    fields."""

    prior: PriorContext
    posterior_log_odds: float
    posterior_probability: float
    posterior_probability_lower: float | None = None
    posterior_probability_upper: float | None = None

    streams: list[StreamContribution]
    contributing_stream_count: int
    rests_on_single_stream: bool
    acoustic_excluded_by_validity_gate: bool

    fusion_method: str
    fusion_model_id: str
    naive_log10_lr: float | None = None
    independence_inflation_log10: float | None = None
    """Orders of magnitude that assuming conditional independence would have
    added. Reported on every multi-stream comparison, not only in aggregate."""

    caveats: list[str]
    """Conditions that must accompany any presentation of this result. Derived
    from the result's own structure, so a client cannot omit one by forgetting
    it existed."""

    computed_at: str

    @field_validator("verbal_summary", "caveats")
    @classmethod
    def _check_language(cls, value: str | list[str]) -> str | list[str]:
        """Refuse to serialise text that asserts identity.

        The last checkpoint before bytes leave the process. If a defect upstream
        produced the vocabulary of identity, this turns it into a 500 rather than
        a conclusion on someone's screen.
        """
        if isinstance(value, str):
            OutputLanguagePolicy.assert_permitted(value, origin="api.response")
        else:
            for item in value:
                OutputLanguagePolicy.assert_permitted(item, origin="api.response")
        return value


class SearchRequestBody(_Base):
    """A database search request."""

    probe_incident_id: str = Field(min_length=1)
    case_reference: str = Field(min_length=1)
    max_results: int = Field(default=25, ge=1, le=200)
    restricted_population: int | None = Field(default=None, ge=2)
    restriction_justification: str = ""


class SearchResponse(_Base):
    """Ranked candidates, with the search context that makes them readable."""

    probe_incident_id: str
    case_reference: str
    results: list[ComparisonResponse]

    n_candidates_compared: int
    n_declined: int
    """Candidates that could not be compared. They have *not* been excluded on
    the evidence, and a client that omits this number implies they were."""

    prior: PriorContext
    mandatory_caveat: str
    """Must be displayed with the results. Names the search size, the prior, and
    — where it applies — that the top-ranked candidate remains more likely than
    not to be unrelated."""

    searched_at: str


class IngestRequest(_Base):
    """An enrolment request."""

    incident_id: str = Field(min_length=1)
    case_reference: str = Field(min_length=1)
    recording_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(_Base):
    """What enrolment produced."""

    incident_id: str
    extracted_streams: list[str]
    validity_verdict: str | None = None
    acoustic_admitted: bool
    warnings: list[str] = Field(default_factory=list)


class AuditEntryResponse(_Base):
    """One audit entry."""

    timestamp: str
    actor_id: str
    actor_roles: list[str]
    action: str
    case_reference: str
    outcome: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class AuditVerificationResponse(_Base):
    """The result of verifying the audit chain."""

    is_intact: bool
    n_entries: int
    first_broken_index: int | None = None
    detail: str
