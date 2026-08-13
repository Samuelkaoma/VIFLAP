"""Diagonal-covariance Gaussian mixture model and universal background model.

The UBM is the statistical description of "speech in general" against which any
particular speaker is described as a deviation. Everything above it in the
speaker stack — Baum-Welch statistics, i-vectors, and hence the acoustic
likelihood ratio — is defined relative to it, so its properties propagate.

Diagonal covariance, and why
----------------------------
A full-covariance mixture with ``C`` components in ``D`` dimensions has
``C * D * (D + 1) / 2`` covariance parameters: for ``C = 256`` and ``D = 60``,
468,480 of them. Estimating that from the tens of hours of speech realistically
available for a low-resource language yields covariance matrices that are
singular or nearly so, and the log-determinant of a nearly singular matrix
dominates the likelihood in a way that has nothing to do with the speaker.

The diagonal restriction is not a loss of generality in the way it first
appears. A mixture of enough diagonal Gaussians approximates any density,
including one with correlated dimensions, by using more components. Trading
covariance parameters for components spends the same budget on something that
can be estimated reliably. This is why the entire speaker recognition literature
uses diagonal UBMs.

Variance flooring
-----------------
A component that captures very few frames drives its variance toward zero, the
likelihood of those frames toward infinity, and the whole model toward a
degenerate solution. Flooring at a fraction of the global variance prevents it.
This is not a numerical detail bolted on afterwards: without it, EM on real
speech reliably diverges within a few tens of iterations, and the failure is
silent because the likelihood keeps increasing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from viflap.domain.errors import (
    ConvergenceError,
    InsufficientDataError,
    InvalidEvidenceError,
)

__all__ = ["BaumWelchStatistics", "GaussianMixture", "GmmConfig", "train_ubm"]


@dataclass(frozen=True, slots=True)
class GmmConfig:
    """Training parameters for the universal background model."""

    n_components: int = 256
    max_iterations: int = 100
    tolerance: float = 1e-4
    """Relative improvement in average log-likelihood below which training stops."""

    variance_floor_factor: float = 0.01
    """Variance floor as a fraction of the global per-dimension variance. Without
    a floor, EM collapses components onto individual frames and diverges."""

    min_component_occupancy: float = 1.0
    """Components accumulating less than this expected frame count are
    reinitialised by splitting the heaviest component. Left alone, a starved
    component contributes nothing but still consumes a mixture weight, and its
    parameters drift to wherever the last frame it saw happened to be."""

    seed: int = 0

    chunk_frames: int = 16_384
    """Frames per block in the E-step and in Lloyd refinement.

    Every quantity these two steps need is a sum over frames, so they can run
    blockwise for the same answer at a bounded working set. Held whole, the
    ``(n_frames, n_components)`` responsibility matrix is 1.2 GB at 600k frames
    and 256 components, and the E-step keeps several of that size alive at once;
    the machine pages and the run takes longer than the arithmetic warrants.
    Raising this trades memory for a small gain in BLAS efficiency."""

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise InvalidEvidenceError(
                "a mixture needs at least one component", n_components=self.n_components
            )
        if self.variance_floor_factor <= 0.0:
            raise InvalidEvidenceError(
                "variance floor factor must be positive; without a floor EM "
                "collapses components onto single frames"
            )
        if self.chunk_frames < 1:
            raise InvalidEvidenceError(
                "chunk_frames must be at least one frame", chunk_frames=self.chunk_frames
            )


@dataclass(frozen=True, slots=True)
class BaumWelchStatistics:
    """Sufficient statistics of one utterance with respect to a UBM.

    These are the interface between the frame-level front-end and everything
    above it. An utterance of any length reduces to a fixed-size summary:

    ``zeroth``
        ``N_c = sum_t gamma_tc`` — expected frames per component. Shape ``(C,)``.
    ``first``
        ``F_c = sum_t gamma_tc (x_t - mu_c)`` — first moment, **centred on the
        UBM means**. Shape ``(C, D)``.

    Centring at accumulation time rather than later matters: the i-vector model
    is defined on deviations from the UBM supervector, and carrying uncentred
    statistics means every consumer must remember to subtract, which one of them
    eventually will not.
    """

    zeroth: NDArray[np.float64]
    first: NDArray[np.float64]
    n_frames: int

    @property
    def n_components(self) -> int:
        return self.zeroth.shape[0]

    @property
    def n_dimensions(self) -> int:
        return self.first.shape[1]

    @property
    def effective_occupancy(self) -> float:
        """Fraction of components that saw meaningful occupancy.

        A short utterance touches only a few components, so its statistics are
        sparse and the i-vector extracted from them is dominated by the prior.
        Reported so that duration-driven degradation is visible in the
        diagnostics rather than appearing as an unexplained weak likelihood
        ratio.
        """
        return float(np.mean(self.zeroth > 1.0))


@dataclass(frozen=True, slots=True)
class GaussianMixture:
    """A trained diagonal-covariance Gaussian mixture.

    Immutable. Training returns a new instance rather than mutating in place, so
    an intermediate model cannot be accidentally used for scoring and a model
    loaded from disk cannot be modified by the code that scores with it.
    """

    weights: NDArray[np.float64]
    means: NDArray[np.float64]
    variances: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.weights.ndim != 1 or self.means.ndim != 2 or self.variances.ndim != 2:
            raise InvalidEvidenceError("mixture parameters have inconsistent rank")
        if not (self.weights.shape[0] == self.means.shape[0] == self.variances.shape[0]):
            raise InvalidEvidenceError("mixture parameters disagree on component count")
        if self.means.shape != self.variances.shape:
            raise InvalidEvidenceError("means and variances disagree on dimension")
        if np.any(self.variances <= 0.0):
            raise InvalidEvidenceError("variances must be strictly positive")
        if not np.isclose(self.weights.sum(), 1.0, atol=1e-6):
            raise InvalidEvidenceError(
                "mixture weights must sum to one", total=float(self.weights.sum())
            )

    @property
    def n_components(self) -> int:
        return self.weights.shape[0]

    @property
    def n_dimensions(self) -> int:
        return self.means.shape[1]

    @property
    def supervector(self) -> NDArray[np.float64]:
        """Means concatenated into a single ``C * D`` vector.

        The representation the total-variability model operates on. Ordering is
        component-major, matching the flattening of ``means``, and every other
        module depends on that ordering being this one.
        """
        return self.means.reshape(-1)

    def log_component_likelihoods(
        self, features: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Per-frame, per-component log density plus log weight.

        Shape ``(n_frames, n_components)``.

        Expanded into matrix products rather than evaluated as a loop over
        components:

        .. code-block:: text

            -0.5 sum_d (x_d - mu_d)^2 / var_d
              = -0.5 [ sum_d x_d^2/var_d - 2 sum_d x_d mu_d/var_d + sum_d mu_d^2/var_d ]

        The first term is ``X^2 @ (1/var)^T``, the second ``X @ (mu/var)^T``, and
        the third is constant per component. Three BLAS calls instead of ``C``
        Python iterations; on a three-minute call against a 256-component UBM
        that is the difference between milliseconds and tens of seconds, and a
        database search runs it over every enrolled recording.
        """
        features = np.atleast_2d(np.asarray(features, dtype=np.float64))
        if features.shape[1] != self.n_dimensions:
            raise InvalidEvidenceError(
                "feature dimension differs from the model's",
                expected=self.n_dimensions,
                received=features.shape[1],
            )

        precision = 1.0 / self.variances
        constant = -0.5 * (
            self.n_dimensions * np.log(2.0 * np.pi)
            + np.sum(np.log(self.variances), axis=1)
            + np.sum(self.means**2 * precision, axis=1)
        )
        quadratic = (features**2) @ precision.T - 2.0 * features @ (
            self.means * precision
        ).T
        return np.log(self.weights) + constant - 0.5 * quadratic

    def log_likelihood(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-frame log likelihood under the mixture."""
        return logsumexp(self.log_component_likelihoods(features), axis=1)

    def responsibilities(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Posterior probability of each component for each frame."""
        joint = self.log_component_likelihoods(features)
        return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))

    def baum_welch(self, features: NDArray[np.float64]) -> BaumWelchStatistics:
        """Accumulate centred sufficient statistics for one utterance."""
        features = np.atleast_2d(np.asarray(features, dtype=np.float64))
        gamma = self.responsibilities(features)
        zeroth = gamma.sum(axis=0)
        # sum_t gamma_tc x_t, then subtract N_c mu_c to centre on the UBM.
        first = gamma.T @ features - zeroth[:, None] * self.means
        return BaumWelchStatistics(
            zeroth=zeroth, first=first, n_frames=int(features.shape[0])
        )

    def map_adapt(
        self,
        features: NDArray[np.float64],
        relevance_factor: float = 16.0,
        adapt_variances: bool = False,
    ) -> GaussianMixture:
        """Maximum a posteriori adaptation of the means toward an utterance.

        Each component's mean moves from the UBM value toward the utterance's
        own estimate by

        .. code-block:: text

            mu_hat_c = alpha_c * (F_c / N_c) + (1 - alpha_c) * mu_c
            alpha_c  = N_c / (N_c + r)

        The relevance factor ``r`` sets how much evidence a component needs
        before it is allowed to move. Components the utterance barely visits
        stay at the UBM value, which is the correct behaviour: their utterance
        estimate is one or two frames and moving to it would be fitting noise.

        Only the means are adapted by default. Adapting weights and variances
        from a single utterance is a well-documented way to make performance
        worse, because those parameters need far more data than a mean does.
        """
        if relevance_factor <= 0.0:
            raise InvalidEvidenceError(
                "relevance factor must be positive", relevance_factor=relevance_factor
            )
        features = np.atleast_2d(np.asarray(features, dtype=np.float64))
        gamma = self.responsibilities(features)
        zeroth = gamma.sum(axis=0)
        first = gamma.T @ features

        alpha = (zeroth / (zeroth + relevance_factor))[:, None]
        # Where a component saw no frames the ratio is undefined; alpha is zero
        # there, so the UBM mean is retained and the placeholder never matters.
        utterance_means = np.divide(
            first, zeroth[:, None], out=np.zeros_like(first), where=zeroth[:, None] > 0.0
        )
        means = alpha * utterance_means + (1.0 - alpha) * self.means

        variances = self.variances
        if adapt_variances:
            second = gamma.T @ (features**2)
            utterance_variances = np.divide(
                second,
                zeroth[:, None],
                out=np.zeros_like(second),
                where=zeroth[:, None] > 0.0,
            )
            variances = (
                alpha * utterance_variances
                + (1.0 - alpha) * (self.variances + self.means**2)
                - means**2
            )
            variances = np.maximum(variances, self.variances * 0.01)

        return replace(self, means=means, variances=variances)


def train_ubm(
    feature_batches: Iterable[NDArray[np.float64]],
    config: GmmConfig | None = None,
) -> GaussianMixture:
    """Train a universal background model by expectation-maximisation.

    Parameters
    ----------
    feature_batches:
        Per-utterance feature matrices. Consumed once and concatenated. For
        corpora too large to hold in memory the caller should subsample frames
        before calling; a UBM does not need every frame of every recording, and
        a stratified sample across speakers and channels is a better use of the
        same budget than all frames from a few recordings.

    Notes
    -----
    Initialisation is k-means++ followed by a short Lloyd refinement. Random
    initialisation of a 256-component mixture in 60 dimensions reliably converges
    to a poor local optimum; k-means++ seeding costs one pass and removes most of
    that variance.

    The average log-likelihood is asserted to be non-decreasing across
    iterations. That is a mathematical property of EM, so a decrease is a defect
    in this implementation and is raised rather than tolerated.
    """
    config = config or GmmConfig()
    features = _stack_batches(feature_batches)

    n_frames = features.shape[0]
    if n_frames < config.n_components * 10:
        raise InsufficientDataError(
            "too few frames to estimate this many mixture components; a "
            "component fitted to a handful of frames describes those frames "
            "rather than the population",
            n_frames=n_frames,
            n_components=config.n_components,
            recommended_minimum=config.n_components * 10,
        )

    rng = np.random.default_rng(config.seed)
    global_variance = np.var(features, axis=0)
    variance_floor = np.maximum(global_variance * config.variance_floor_factor, 1e-10)

    means = _kmeans_plus_plus(features, config.n_components, rng)
    means = _lloyd_refine(features, means, iterations=5, chunk_frames=config.chunk_frames)

    weights = np.full(config.n_components, 1.0 / config.n_components)
    variances = np.tile(
        np.maximum(global_variance, variance_floor), (config.n_components, 1)
    )
    model = GaussianMixture(weights=weights, means=means, variances=variances)

    previous = -np.inf
    reseeded_last_iteration = False

    for iteration in range(config.max_iterations):
        statistics = _expectation_step(model, features, config.chunk_frames)
        current = statistics.mean_log_likelihood

        if not np.isfinite(current):
            raise ConvergenceError(
                "UBM training produced a non-finite log-likelihood",
                iteration=iteration,
            )
        # EM cannot decrease the likelihood, so a decrease is a defect in the
        # update equations rather than a property of the data — with one
        # legitimate exception. Reseeding a starved component is a deliberate
        # discontinuous jump outside the EM update, and it may cost likelihood
        # in the iteration that follows. Only the exception is exempted; an
        # unexplained decrease still fails loudly.
        if current < previous - 1e-6 and not reseeded_last_iteration:
            raise ConvergenceError(
                "UBM log-likelihood decreased without a component being reseeded, "
                "which expectation-maximisation cannot do",
                iteration=iteration,
                previous=previous,
                current=current,
            )
        if (
            iteration > 0
            and not reseeded_last_iteration
            and (current - previous) < config.tolerance * abs(previous)
        ):
            break
        previous = current

        model, reseeded_last_iteration = _maximisation_step(
            n_frames, statistics, variance_floor, config, rng
        )

    return model


class _ExpectationStatistics(NamedTuple):
    """What the M-step needs from a pass over the data.

    ``occupancy`` is ``sum_t gamma_tc``, ``first_moment`` is
    ``sum_t gamma_tc x_td`` and ``second_moment`` is ``sum_t gamma_tc x_td^2``,
    all summed over every frame. Sizes depend on the model, not on the corpus.
    """

    mean_log_likelihood: float
    occupancy: NDArray[np.float64]
    first_moment: NDArray[np.float64]
    second_moment: NDArray[np.float64]


def _expectation_step(
    model: GaussianMixture, features: NDArray[np.float64], chunk_frames: int
) -> _ExpectationStatistics:
    """Accumulate the E-step statistics a block of frames at a time.

    Equivalent to forming the whole responsibility matrix and reducing it, since
    every statistic here is a sum over frames — but the working set is set by
    ``chunk_frames`` rather than by the size of the corpus. See
    :attr:`GmmConfig.chunk_frames` for why that matters.

    Blockwise summation is not bit-identical to summing the whole matrix at
    once, because floating-point addition does not associate. The difference is
    at the level of accumulation error and is if anything smaller, a long
    running sum being the less accurate of the two.
    """
    occupancy = np.zeros(model.n_components)
    first_moment = np.zeros((model.n_components, model.n_dimensions))
    second_moment = np.zeros((model.n_components, model.n_dimensions))
    total_log_likelihood = 0.0

    for start in range(0, features.shape[0], chunk_frames):
        block = features[start : start + chunk_frames]
        joint = model.log_component_likelihoods(block)
        block_log_likelihood = logsumexp(joint, axis=1)
        total_log_likelihood += float(block_log_likelihood.sum())

        gamma = np.exp(joint - block_log_likelihood[:, None])
        occupancy += gamma.sum(axis=0)
        first_moment += gamma.T @ block
        second_moment += gamma.T @ (block**2)

    return _ExpectationStatistics(
        mean_log_likelihood=total_log_likelihood / features.shape[0],
        occupancy=occupancy,
        first_moment=first_moment,
        second_moment=second_moment,
    )


def _maximisation_step(
    n_frames: int,
    statistics: _ExpectationStatistics,
    variance_floor: NDArray[np.float64],
    config: GmmConfig,
    rng: np.random.Generator,
) -> tuple[GaussianMixture, bool]:
    """Closed-form parameter update, with starved components reseeded.

    Returns the updated model and whether any component was reseeded, so the
    caller can distinguish a legitimate likelihood decrease from a defect.
    """
    occupancy = statistics.occupancy
    safe_occupancy = np.maximum(occupancy, 1e-10)[:, None]

    means = statistics.first_moment / safe_occupancy
    second_moment = statistics.second_moment / safe_occupancy
    variances = np.maximum(second_moment - means**2, variance_floor)
    weights = occupancy / n_frames

    starved = occupancy < config.min_component_occupancy
    reseeded = bool(np.any(starved))
    if reseeded:
        means, variances, weights = _reseed_starved_components(
            means, variances, weights, occupancy, starved, variance_floor, rng
        )

    # Floor the weights before normalising. A weight of exactly zero makes
    # log_component_likelihoods produce -inf for that component, which is
    # harmless in the logsumexp but raises a numpy warning on every call and
    # makes any genuine numerical problem harder to notice.
    weights = np.maximum(weights, 1e-10)
    weights = weights / weights.sum()
    return GaussianMixture(weights=weights, means=means, variances=variances), reseeded


def _reseed_starved_components(
    means: NDArray[np.float64],
    variances: NDArray[np.float64],
    weights: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    starved: NDArray[np.bool_],
    variance_floor: NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Split the heaviest components to replace ones that have starved.

    A starved component is dead weight: it holds a mixture weight, contributes
    nothing to the likelihood, and its mean sits wherever the last frame it saw
    happened to be. Splitting the heaviest component along its principal
    direction of variance puts the capacity where the data actually is. This is
    the standard remedy and it is what makes EM on real speech converge to a
    usable model rather than one with a third of its components inert.
    """
    means = means.copy()
    variances = variances.copy()
    weights = weights.copy()

    donors = np.argsort(occupancy)[::-1]
    donor_index = 0
    for target in np.flatnonzero(starved):
        while donor_index < donors.size and starved[donors[donor_index]]:
            donor_index += 1
        if donor_index >= donors.size:
            break
        donor = donors[donor_index]
        donor_index += 1

        perturbation = (
            0.25 * np.sqrt(variances[donor]) * rng.standard_normal(means.shape[1])
        )
        means[target] = means[donor] + perturbation
        means[donor] = means[donor] - perturbation
        variances[target] = np.maximum(variances[donor], variance_floor)
        weights[target] = weights[donor] = max(weights[donor], 1e-6) / 2.0

    return means, variances, weights


def _kmeans_plus_plus(
    features: NDArray[np.float64], k: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """k-means++ seeding: choose centres far from those already chosen.

    Each new centre is drawn with probability proportional to its squared
    distance from the nearest existing centre. The result is a spread of centres
    across the data rather than a cluster of them in the densest region, which
    is what uniform random selection produces and which costs EM many iterations
    to undo — if it undoes it at all.
    """
    n_frames = features.shape[0]
    centres = np.empty((k, features.shape[1]), dtype=np.float64)
    centres[0] = features[rng.integers(n_frames)]

    closest = np.sum((features - centres[0]) ** 2, axis=1)
    for index in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:
            # All remaining points coincide with a chosen centre. Fill the rest
            # with jittered copies rather than dividing by zero.
            centres[index:] = (
                centres[index - 1]
                + rng.standard_normal((k - index, features.shape[1])) * 1e-3
            )
            break
        probabilities = closest / total
        centres[index] = features[rng.choice(n_frames, p=probabilities)]
        closest = np.minimum(closest, np.sum((features - centres[index]) ** 2, axis=1))
    return centres


def _lloyd_refine(
    features: NDArray[np.float64],
    centres: NDArray[np.float64],
    iterations: int,
    chunk_frames: int,
) -> NDArray[np.float64]:
    """A few Lloyd iterations to settle the seeds before EM takes over.

    Assignment is blocked for the same reason the E-step is: the distance matrix
    is ``(n_frames, n_components)`` and only the per-frame argmin survives it.
    """
    centres = centres.copy()
    n_frames = features.shape[0]
    assignment = np.empty(n_frames, dtype=np.intp)

    for _ in range(iterations):
        squared_norms = np.sum(centres**2, axis=1)[None, :]
        for start in range(0, n_frames, chunk_frames):
            block = features[start : start + chunk_frames]
            # ||x - c||^2 expanded so the assignment is a single matrix product;
            # the ||x||^2 term is constant across centres and can be dropped.
            distances = squared_norms - 2.0 * block @ centres.T
            assignment[start : start + chunk_frames] = np.argmin(distances, axis=1)
        for index in range(centres.shape[0]):
            members = features[assignment == index]
            if members.shape[0] > 0:
                centres[index] = members.mean(axis=0)
    return centres


def _stack_batches(batches: Iterable[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Concatenate feature batches, validating shape consistency."""
    collected = [np.atleast_2d(np.asarray(batch, dtype=np.float64)) for batch in batches]
    collected = [batch for batch in collected if batch.size > 0]
    if not collected:
        raise InsufficientDataError("no features supplied for training")
    dimensions = {batch.shape[1] for batch in collected}
    if len(dimensions) != 1:
        raise InvalidEvidenceError(
            "feature batches have inconsistent dimension",
            dimensions=sorted(dimensions),
        )
    return np.concatenate(collected, axis=0)
