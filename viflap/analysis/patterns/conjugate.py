"""Marginal likelihood ratios under conjugate models.

This module is the statistical foundation of every non-acoustic stream, and it
replaces the approach those streams would otherwise take.

The approach it replaces
------------------------
The obvious way to compare two incidents on, say, their cash-out agents is a
Jaccard index, combined with other similarities in a weighted sum, and then
calibrated. Two things are wrong with it.

First, it ignores **rarity**. Two incidents sharing an agent who handles four
transactions a month is strong evidence. Two incidents sharing an agent who
handles four thousand is nearly none. A Jaccard index gives both the same value,
and no amount of downstream calibration can recover a distinction the statistic
discarded.

Second, the weights are chosen rather than estimated, and every arbitrary
constant in a forensic system is a place where an assumption hides.

The approach taken instead
--------------------------
Ask the question directly. Under the same-source proposition the two incidents'
observations are draws from **one** actor's distribution; under the
different-source proposition they are draws from **two** independent actors'
distributions. Both actors' parameters are unknown, so integrate them out
against a prior fitted to the background population. The likelihood ratio is
then

.. code-block:: text

    LR = p(x_A, x_B | one actor) / [ p(x_A | actor 1) p(x_B | actor 2) ]

With a conjugate prior each term has a closed form, so this is exact rather than
approximated, and it is a genuine likelihood ratio by construction rather than a
similarity that has been calibrated into resembling one.

Rarity is handled automatically and correctly. The prior's concentration
parameters are proportional to background frequencies, so sharing a category
that the background rarely produces contributes far more than sharing a common
one — without anyone choosing a weight.

Two families cover the evidence types
-------------------------------------
:class:`DirichletMultinomialComparator`
    Categorical counts: cash-out agents, cell sites, handset models, function
    words, script n-grams, disfluency types, discretised hour of day.

:class:`NormalInverseGammaComparator`
    Continuous measurements: log transaction amounts, call durations,
    inter-arrival times.

Both are used across several streams, which means the dependence between those
streams is a property to be modelled by the fusion layer rather than an artefact
of each having invented its own scoring scheme.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from viflap.domain.errors import InsufficientDataError, InvalidEvidenceError

__all__ = [
    "BackgroundPopulation",
    "DirichletMultinomialComparator",
    "NormalInverseGammaComparator",
    "counts_to_vector",
]


@dataclass(frozen=True, slots=True)
class BackgroundPopulation:
    """Relative frequencies of categories in the relevant population.

    The denominator of every likelihood ratio this module produces. Its choice
    is a modelling decision with real consequences and it is recorded so that it
    can be examined: a background estimated from one province's agent network
    says something different about a shared agent than one estimated nationally.

    ``unseen_mass`` reserves probability for categories absent from the
    background sample. Without it, an agent identifier never seen in the
    background has frequency zero, giving an infinite likelihood ratio on the
    strength of a sampling gap. Good-Turing-style reservation is what keeps the
    strongest results from being artefacts of what the background happened to
    contain.
    """

    frequencies: Mapping[str, float]
    total_observations: int
    description: str
    unseen_mass: float = 0.01

    def __post_init__(self) -> None:
        if not self.frequencies:
            raise InvalidEvidenceError("background population cannot be empty")
        total = sum(self.frequencies.values())
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise InvalidEvidenceError(
                "background frequencies must sum to one", total=total
            )
        if not 0.0 <= self.unseen_mass < 0.5:
            raise InvalidEvidenceError(
                "unseen mass must lie in [0, 0.5)", unseen_mass=self.unseen_mass
            )
        if self.total_observations < 1:
            raise InvalidEvidenceError("background must rest on at least one observation")

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted(self.frequencies))

    def frequency_of(self, category: str) -> float:
        """Frequency of a category, with unseen categories given a floor.

        The floor is the reserved unseen mass spread over a plausible number of
        unobserved categories, approximated by the number observed. This is
        crude, and it is deliberately conservative: it makes an unseen category
        look about as rare as the rarest observed one rather than infinitely
        rare.
        """
        known = self.frequencies.get(category)
        if known is not None:
            return known * (1.0 - self.unseen_mass)
        return self.unseen_mass / max(len(self.frequencies), 1)

    @classmethod
    def from_counts(
        cls,
        counts: Mapping[str, int],
        description: str,
        unseen_mass: float = 0.01,
    ) -> BackgroundPopulation:
        """Estimate a background from observed category counts."""
        total = sum(counts.values())
        if total < 1:
            raise InsufficientDataError(
                "cannot estimate a background population from no observations"
            )
        return cls(
            frequencies={
                category: count / total for category, count in counts.items() if count > 0
            },
            total_observations=total,
            description=description,
            unseen_mass=unseen_mass,
        )

    @classmethod
    def uniform_over(
        cls, categories: Sequence[str], description: str
    ) -> BackgroundPopulation:
        """A uniform background, for use only where no better estimate exists.

        Uniformity is almost always wrong — agent transaction volumes, cell site
        usage and word frequencies are all heavily skewed — and it understates
        the value of sharing a rare category while overstating a common one.
        Provided so that a system can run before background data is available,
        and named so that dependence on it is visible in the record.
        """
        if not categories:
            raise InvalidEvidenceError("cannot build a background over no categories")
        weight = 1.0 / len(categories)
        return cls(
            frequencies=dict.fromkeys(categories, weight),
            total_observations=len(categories),
            description=f"{description} (UNIFORM — no background data available)",
        )


def counts_to_vector(
    counts: Mapping[str, int], categories: Sequence[str]
) -> NDArray[np.float64]:
    """Project a count mapping onto a fixed category ordering."""
    return np.array([float(counts.get(category, 0)) for category in categories])


class DirichletMultinomialComparator:
    """Same-source likelihood ratio for categorical count data.

    Model: an actor draws categories from a multinomial whose probabilities are
    unknown and drawn from a Dirichlet prior concentrated on the background
    frequencies. Integrating out the multinomial probabilities gives the
    Dirichlet-multinomial marginal likelihood

    .. code-block:: text

        p(n | alpha) = Gamma(A) / Gamma(N + A) * prod_c Gamma(n_c + a_c) / Gamma(a_c)

    with ``A = sum_c a_c`` and ``N = sum_c n_c``. The likelihood ratio for two
    count vectors follows by comparing one pooled draw against two independent
    ones, and the ``Gamma(a_c)`` terms cancel:

    .. code-block:: text

        log LR = logG(N_A + A) + logG(N_B + A) - logG(A) - logG(N_A + N_B + A)
               + sum_c [ logG(n_Ac + n_Bc + a_c) + logG(a_c)
                       - logG(n_Ac + a_c) - logG(n_Bc + a_c) ]

    Exact, in closed form, with no fitted weights.

    What the concentration parameter controls
    -----------------------------------------
    ``a_c = concentration * background_frequency(c)``. The concentration governs
    how strongly an actor is expected to resemble the population. Low
    concentration means actors are idiosyncratic — each has strong preferences
    of their own — which makes agreement between two incidents more surprising
    and therefore stronger evidence. High concentration means actors mostly
    behave like the population, and agreement says little.

    It is a real parameter with a real effect, and it is estimated from data
    (:meth:`fit_concentration`) rather than chosen. Where it cannot be
    estimated, the default is deliberately high, because that is the
    conservative direction: it understates evidence rather than overstating it.
    """

    def __init__(
        self, background: BackgroundPopulation, concentration: float = 50.0
    ) -> None:
        if concentration <= 0.0:
            raise InvalidEvidenceError(
                "concentration must be positive", concentration=concentration
            )
        self._background = background
        self._concentration = concentration
        self._categories = background.categories
        self._alpha = np.array(
            [background.frequency_of(category) for category in self._categories]
        )
        self._alpha = self._alpha / self._alpha.sum() * concentration

    @property
    def background(self) -> BackgroundPopulation:
        return self._background

    @property
    def concentration(self) -> float:
        return self._concentration

    def _align(self, counts: Mapping[str, int]) -> tuple[NDArray[np.float64], float]:
        """Project counts onto the background categories, pooling unknown ones.

        Categories absent from the background are pooled into a single
        "unseen" bucket rather than dropped. Dropping them would silently
        discard evidence; treating each as unique would make any two incidents
        sharing an unseen category appear identical.
        """
        vector = np.zeros(self._alpha.size + 1)
        index = {category: position for position, category in enumerate(self._categories)}
        unseen = 0.0
        for category, count in counts.items():
            position = index.get(category)
            if position is None:
                unseen += float(count)
            else:
                vector[position] += float(count)
        vector[-1] = unseen
        return vector, unseen

    @property
    def _full_alpha(self) -> NDArray[np.float64]:
        """Concentration parameters including the pooled unseen bucket."""
        unseen_alpha = self._concentration * self._background.unseen_mass
        return np.append(self._alpha, max(unseen_alpha, 1e-6))

    def log_likelihood_ratio(
        self, counts_a: Mapping[str, int], counts_b: Mapping[str, int]
    ) -> float:
        """Log-likelihood ratio that both count vectors came from one actor."""
        vector_a, _ = self._align(counts_a)
        vector_b, _ = self._align(counts_b)

        total_a = float(vector_a.sum())
        total_b = float(vector_b.sum())
        if total_a == 0.0 or total_b == 0.0:
            raise InsufficientDataError(
                "both incidents must have at least one observation",
                n_a=int(total_a),
                n_b=int(total_b),
            )

        alpha = self._full_alpha
        alpha_total = float(alpha.sum())

        pooled = (
            gammaln(alpha_total)
            - gammaln(total_a + total_b + alpha_total)
            + float(np.sum(gammaln(vector_a + vector_b + alpha) - gammaln(alpha)))
        )
        separate_a = (
            gammaln(alpha_total)
            - gammaln(total_a + alpha_total)
            + float(np.sum(gammaln(vector_a + alpha) - gammaln(alpha)))
        )
        separate_b = (
            gammaln(alpha_total)
            - gammaln(total_b + alpha_total)
            + float(np.sum(gammaln(vector_b + alpha) - gammaln(alpha)))
        )
        return float(pooled - separate_a - separate_b)

    def fit_concentration(
        self,
        same_source_pairs: Sequence[tuple[Mapping[str, int], Mapping[str, int]]],
        candidates: Sequence[float] | None = None,
    ) -> DirichletMultinomialComparator:
        """Estimate the concentration from known same-source pairs.

        Chooses the value maximising the total likelihood ratio over pairs known
        to share an actor. This is a legitimate estimator: the concentration
        describes how much actors resemble the population, and pairs known to
        share an actor are precisely the observations that identify it.

        Uses a grid rather than a continuous optimiser. The likelihood surface in
        this parameter is flat over wide ranges and has local optima, and a
        gradient method lands in a different place depending on where it starts —
        which is not a property a forensic parameter should have.
        """
        if not same_source_pairs:
            raise InsufficientDataError(
                "concentration estimation requires known same-source pairs"
            )
        grid = candidates or [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 1000.0]

        best_value = self._concentration
        best_total = -math.inf
        for candidate in grid:
            trial = DirichletMultinomialComparator(self._background, candidate)
            total = 0.0
            usable = 0
            for counts_a, counts_b in same_source_pairs:
                try:
                    total += trial.log_likelihood_ratio(counts_a, counts_b)
                    usable += 1
                except InsufficientDataError:
                    continue
            if usable == 0:
                continue
            average = total / usable
            if average > best_total:
                best_total = average
                best_value = candidate

        return DirichletMultinomialComparator(self._background, best_value)


class NormalInverseGammaComparator:
    """Same-source likelihood ratio for continuous measurements.

    The continuous counterpart of the Dirichlet-multinomial. An actor's
    measurements are normal with unknown mean and variance, drawn from a
    normal-inverse-gamma prior fitted to the background population. Integrating
    both out gives a closed-form marginal likelihood, and the ratio compares one
    pooled actor against two independent ones.

    Used on quantities that are approximately normal after transformation.
    Transaction amounts and call durations are strongly right-skewed and are
    passed through a logarithm first, where they are close to normal — a
    substantive modelling choice, not a numerical convenience, and one that
    matters because the untransformed distribution has no meaningful mean.

    The marginal likelihood of ``n`` observations under a normal-inverse-gamma
    prior ``(mu_0, kappa_0, alpha_0, beta_0)`` is

    .. code-block:: text

        p(x) = (1 / pi^(n/2)) * sqrt(k_0 / k_n)
             * Gamma(a_n) / Gamma(a_0) * b_0^a_0 / b_n^a_n

    with the usual conjugate updates.
    """

    def __init__(
        self,
        prior_mean: float,
        prior_precision: float,
        prior_shape: float,
        prior_scale: float,
        description: str = "",
    ) -> None:
        if prior_precision <= 0.0 or prior_shape <= 0.0 or prior_scale <= 0.0:
            raise InvalidEvidenceError(
                "normal-inverse-gamma hyperparameters must be positive",
                precision=prior_precision,
                shape=prior_shape,
                scale=prior_scale,
            )
        self._mean = prior_mean
        self._precision = prior_precision
        self._shape = prior_shape
        self._scale = prior_scale
        self._description = description

    @property
    def description(self) -> str:
        return self._description

    @classmethod
    def from_background(
        cls,
        background_values: NDArray[np.float64],
        within_actor_fraction: float = 0.4,
        description: str = "",
    ) -> NormalInverseGammaComparator:
        """Fit hyperparameters from a background sample.

        ``within_actor_fraction`` is the share of total variance attributable to
        variation *within* one actor rather than between actors. It is the
        parameter that determines evidential value, and it must be supplied or
        estimated: if actors differ little from one another, two incidents
        agreeing says nothing.

        The default of 0.4 is a placeholder to be replaced by an estimate from
        labelled data, and it errs high — attributing more variation to within-
        actor noise understates the evidence rather than overstating it.
        """
        values = np.asarray(background_values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 10:
            raise InsufficientDataError(
                "too few background observations to fit a continuous model",
                n_observations=int(values.size),
            )
        if not 0.0 < within_actor_fraction < 1.0:
            raise InvalidEvidenceError(
                "within-actor variance fraction must lie in (0, 1)",
                fraction=within_actor_fraction,
            )

        total_variance = float(np.var(values, ddof=1))
        within_variance = max(total_variance * within_actor_fraction, 1e-9)
        between_variance = max(total_variance - within_variance, 1e-9)

        # kappa_0 is the prior's weight in units of observations: the ratio of
        # within- to between-actor variance. A small ratio means actors are well
        # separated, so a single observation locates one precisely.
        precision = within_variance / between_variance
        shape = 2.5  # finite variance for the marginal, weakly informative
        scale = within_variance * (shape - 1.0)

        return cls(
            prior_mean=float(np.mean(values)),
            prior_precision=precision,
            prior_shape=shape,
            prior_scale=scale,
            description=description,
        )

    def _log_marginal(self, values: NDArray[np.float64]) -> float:
        """Log marginal likelihood of one actor's observations."""
        n = values.size
        if n == 0:
            return 0.0
        sample_mean = float(np.mean(values))
        sum_squares = float(np.sum((values - sample_mean) ** 2))

        precision_n = self._precision + n
        shape_n = self._shape + n / 2.0
        scale_n = (
            self._scale
            + 0.5 * sum_squares
            + (self._precision * n * (sample_mean - self._mean) ** 2) / (2.0 * precision_n)
        )

        return float(
            -0.5 * n * math.log(math.pi)
            + 0.5 * (math.log(self._precision) - math.log(precision_n))
            + gammaln(shape_n)
            - gammaln(self._shape)
            + self._shape * math.log(self._scale)
            - shape_n * math.log(scale_n)
        )

    def log_likelihood_ratio(
        self, values_a: Iterable[float], values_b: Iterable[float]
    ) -> float:
        """Log-likelihood ratio that both samples came from one actor."""
        array_a = np.asarray(list(values_a), dtype=np.float64)
        array_b = np.asarray(list(values_b), dtype=np.float64)
        array_a = array_a[np.isfinite(array_a)]
        array_b = array_b[np.isfinite(array_b)]

        if array_a.size == 0 or array_b.size == 0:
            raise InsufficientDataError(
                "both incidents must have at least one measurement",
                n_a=int(array_a.size),
                n_b=int(array_b.size),
            )

        pooled = self._log_marginal(np.concatenate([array_a, array_b]))
        return float(pooled - self._log_marginal(array_a) - self._log_marginal(array_b))
