"""Propositions, prior odds, and the posterior.

This module is where the system's central safety property lives. The likelihood
ratio answers "how much more probable is this evidence if the incidents share an
actor?". The question an investigator actually wants answered — "how probable is
it that they share an actor?" — requires a prior that the system does not
possess and cannot invent. Reporting the first as though it were the second is
the prosecutor's fallacy.

The type system enforces the distinction. A :class:`PosteriorAssessment` cannot
be constructed except from an explicit :class:`PriorOdds`, and a
:class:`PriorOdds` cannot be constructed without recording where it came from
and who is answerable for it. There is no default prior anywhere in this
package.

The arithmetic that matters
---------------------------
For a database search over ``N`` enrolled entries under a uniform prior, the
prior odds on any single entry are ``1 / (N - 1)``. With ``N = 100,000`` and an
acoustic ``LR`` of ``1,000``:

.. code-block:: text

    prior odds     = 1 / 99,999          ~ 1.0e-5
    posterior odds = 1,000 x 1.0e-5      ~ 0.01
    posterior prob = 0.01 / 1.01         ~ 0.99%

A thousand-to-one result from a national-scale search is about ninety-nine
percent likely to be wrong. This is not a caveat. It is the reason the platform
fuses streams instead of comparing voices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from viflap.domain.errors import InvalidEvidenceError
from viflap.domain.values import (
    LogLikelihoodRatio,
    LogOdds,
    Probability,
    UncertaintyInterval,
)

__all__ = [
    "Hypothesis",
    "PosteriorAssessment",
    "PriorBasis",
    "PriorOdds",
    "SearchMode",
]


class Hypothesis(Enum):
    """The two competing propositions.

    Stated at the *activity* level — did the same person conduct both incidents
    — rather than at the source level of a single recording. This matters: the
    behavioural and transactional streams speak to the operation, and an
    organised group that rotates who speaks can defeat a source-level
    proposition while leaving an activity-level one intact.
    """

    SAME_SOURCE = "H_ss"
    """The two incidents were conducted by the same actor."""

    DIFFERENT_SOURCE = "H_ds"
    """The two incidents were conducted by different actors."""

    @property
    def description(self) -> str:
        if self is Hypothesis.SAME_SOURCE:
            return "the incidents were conducted by the same actor"
        return "the incidents were conducted by different actors"


class SearchMode(Enum):
    """The kind of comparison that produced a result.

    Determines what prior odds are defensible, and is therefore recorded with
    every result. The same likelihood ratio means something entirely different
    depending on which of these produced it.
    """

    VERIFICATION = "verification"
    """One questioned incident against one named suspect. Prior odds come from
    the non-acoustic case circumstances and must be supplied explicitly."""

    DATABASE_SEARCH = "database_search"
    """One questioned incident against ``N`` enrolled entries. Prior odds on any
    individual entry are small and must be applied, or every strong-looking
    result is an artefact of the search size."""

    INTELLIGENCE_TRIAGE = "intelligence_triage"
    """Ranking for investigative attention only, with no evidential claim. The
    prior is still recorded, because a ranking without one invites the reader to
    supply an implicit prior of their own."""


class PriorBasis(Enum):
    """Where a prior came from, and hence who is answerable for it."""

    UNIFORM_OVER_DATABASE = "uniform_over_database"
    """``1 / (N - 1)``, assuming the actor is enrolled and every entry is
    equally probable a priori. Both assumptions are usually false; the first
    inflates the prior, the second ignores population structure. Used as a
    reference point, and its sensitivity is reported alongside it."""

    RESTRICTED_POPULATION = "restricted_population"
    """The investigator narrowed the relevant population — a corridor, a period,
    an operator network — and supplied its size with a justification."""

    CASE_CIRCUMSTANCES = "case_circumstances"
    """Odds supplied directly from non-acoustic case circumstances. Only
    defensible in verification, and only by the investigator, never the
    system."""


@dataclass(frozen=True, slots=True)
class PriorOdds:
    """Prior odds on the same-source proposition, with their provenance.

    Carried on the log scale so that a prior of ``1 / 10^7`` is exact rather
    than a denormal, and so that applying it is addition.

    Every field is mandatory. A prior without a justification is a number
    somebody will later have to defend in cross-examination without knowing
    where it came from.
    """

    log_odds: LogOdds
    basis: PriorBasis
    search_mode: SearchMode
    justification: str
    """Why this prior, in language an investigator would give a court. Not
    optional, and not defaulted."""

    population_size: int | None = None
    """Size of the relevant population where the basis implies one."""

    supplied_by: str = "system"
    """Identifier of the principal who chose this prior. The system may only
    appear here for :attr:`PriorBasis.UNIFORM_OVER_DATABASE`, where the value
    follows mechanically from the search size."""

    def __post_init__(self) -> None:
        if not self.justification.strip():
            raise InvalidEvidenceError(
                "prior odds require a justification; an unexplained prior "
                "cannot be defended and must not be recorded",
                basis=self.basis.value,
            )
        if self.basis is not PriorBasis.CASE_CIRCUMSTANCES and self.population_size is None:
            raise InvalidEvidenceError(
                "this prior basis requires the size of the relevant population",
                basis=self.basis.value,
            )
        if self.population_size is not None and self.population_size < 2:
            raise InvalidEvidenceError(
                "a relevant population must contain at least two candidates for "
                "a comparison to be meaningful",
                population_size=self.population_size,
            )
        if (
            self.basis is not PriorBasis.UNIFORM_OVER_DATABASE
            and self.supplied_by == "system"
        ):
            raise InvalidEvidenceError(
                "only a uniform prior over the searched database may be "
                "attributed to the system; any other prior is a judgement and "
                "must be attributed to the person who made it",
                basis=self.basis.value,
            )

    @property
    def probability(self) -> Probability:
        """The prior as a probability, for display alongside the odds."""
        return self.log_odds.to_probability()

    @classmethod
    def uniform_over_database(cls, database_size: int, /) -> PriorOdds:
        """Uniform prior for a search over ``database_size`` enrolled entries.

        The relevant population is ``N - 1``: the questioned incident is not a
        candidate for linkage with itself.

        This is the only prior the system may originate, and even then it is a
        modelling choice rather than a fact. Sensitivity to it is reported with
        every result that uses it.
        """
        if database_size < 2:
            raise InvalidEvidenceError(
                "a database search requires at least two enrolled entries",
                database_size=database_size,
            )
        return cls(
            log_odds=LogOdds(-math.log(database_size - 1)),
            basis=PriorBasis.UNIFORM_OVER_DATABASE,
            search_mode=SearchMode.DATABASE_SEARCH,
            justification=(
                f"Uniform prior over {database_size - 1:,} candidate entries, "
                f"assuming the actor is enrolled and no entry is favoured a "
                f"priori. Both assumptions require review for this case."
            ),
            population_size=database_size,
            supplied_by="system",
        )

    @classmethod
    def restricted_population(
        cls,
        population_size: int,
        justification: str,
        supplied_by: str,
        search_mode: SearchMode = SearchMode.DATABASE_SEARCH,
    ) -> PriorOdds:
        """Prior over an investigator-narrowed population."""
        if population_size < 2:
            raise InvalidEvidenceError(
                "a restricted population must contain at least two candidates",
                population_size=population_size,
            )
        return cls(
            log_odds=LogOdds(-math.log(population_size - 1)),
            basis=PriorBasis.RESTRICTED_POPULATION,
            search_mode=search_mode,
            justification=justification,
            population_size=population_size,
            supplied_by=supplied_by,
        )

    @classmethod
    def from_case_circumstances(
        cls, odds: float, justification: str, supplied_by: str
    ) -> PriorOdds:
        """Prior supplied directly by an investigator for a verification task."""
        return cls(
            log_odds=LogOdds.from_odds(odds),
            basis=PriorBasis.CASE_CIRCUMSTANCES,
            search_mode=SearchMode.VERIFICATION,
            justification=justification,
            population_size=None,
            supplied_by=supplied_by,
        )

    def apply(self, log_lr: LogLikelihoodRatio) -> LogOdds:
        """Bayes' rule on the log scale."""
        return self.log_odds.updated_with(log_lr)


@dataclass(frozen=True, slots=True)
class PosteriorAssessment:
    """Posterior odds and probability, inseparable from the prior that produced them.

    Constructed only via :meth:`from_evidence`, so the posterior is *derived*
    from the prior and the likelihood ratio rather than supplied alongside them.
    Consistency is therefore a property of construction, not something checked
    after the fact and hoped for.
    """

    prior: PriorOdds
    log_lr: LogLikelihoodRatio
    posterior_log_odds: LogOdds
    interval: UncertaintyInterval | None
    """Posterior log-odds interval, obtained by translating the evidential
    interval by the prior. ``None`` when no interval was estimated."""

    @classmethod
    def from_evidence(
        cls,
        prior: PriorOdds,
        log_lr: LogLikelihoodRatio,
        interval: UncertaintyInterval | None = None,
    ) -> PosteriorAssessment:
        """Apply a prior to a likelihood ratio."""
        return cls(
            prior=prior,
            log_lr=log_lr,
            posterior_log_odds=prior.apply(log_lr),
            interval=(
                interval.shifted_by(prior.log_odds.value) if interval is not None else None
            ),
        )

    @property
    def probability(self) -> Probability:
        """Posterior probability of the same-source proposition.

        Exact across the whole range: a posterior log-odds of ``-11.5`` gives
        ``1.0e-5`` rather than underflowing, and ``+40`` gives a value
        indistinguishable from one without ever producing ``inf/inf``.
        """
        return self.posterior_log_odds.to_probability()

    @property
    def probability_interval(self) -> tuple[Probability, Probability] | None:
        """Posterior probability bounds, or ``None`` if no interval was estimated."""
        if self.interval is None:
            return None
        return (
            LogOdds(self.interval.lower).to_probability(),
            LogOdds(self.interval.upper).to_probability(),
        )

    @property
    def is_dominated_by_prior(self) -> bool:
        """Whether the prior, not the evidence, drives the result.

        True when the posterior remains below even odds despite evidence
        supporting linkage. This is the ordinary condition for a database search
        and the fact most often lost in presentation: strong evidence against a
        small prior still leaves the proposition improbable.
        """
        return self.log_lr.value > 0.0 and self.posterior_log_odds.value < 0.0
