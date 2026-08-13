"""Two-covariance probabilistic linear discriminant analysis.

This is the component that turns a geometric similarity into an evidential
quantity. Everything before it produces vectors; a cosine similarity between two
of them is a number with no interpretation, because it says nothing about how
similar two recordings of *different* speakers would have been. PLDA is a
generative model of both distributions, so the ratio it produces is a
likelihood ratio in the proper sense — the same quantity the forensic
likelihood-ratio framework requires, arrived at by construction rather than by
calibrating a similarity after the fact.

The model
---------
An observation decomposes into a speaker term and a session term:

.. code-block:: text

    x = mu + y + e,     y ~ N(0, B),  e ~ N(0, W)

``y`` is the speaker's true position, drawn once per speaker; ``e`` is
everything that varies between recordings of that speaker — channel, handset,
health, affect, phonetic content. ``B`` is the between-speaker covariance and
``W`` the within-speaker covariance.

Scoring
-------
For two observations, the competing propositions are that they share a ``y``
(same source) or have independent draws of it (different source). Both
likelihoods are Gaussian and the ratio has a closed form.

Simultaneously diagonalising ``B`` and ``W`` — whiten by ``W``, then
eigendecompose the transformed ``B`` — gives a space in which ``W = I`` and
``B = diag(psi)``. Every dimension then separates, and for dimension ``d`` with
between-speaker variance ``psi``, observations ``a`` and ``b``:

.. code-block:: text

    Sigma_ss = [[psi+1, psi], [psi, psi+1]]        (shared y)
    Sigma_ds = [[psi+1, 0  ], [0,   psi+1]]        (independent y)

    log LR_d = 0.5 [ log|Sigma_ds| - log|Sigma_ss| ]
             - 0.5 (a,b) Sigma_ss^-1 (a,b)^T
             + 0.5 (a,b) Sigma_ds^-1 (a,b)^T

which reduces to the expression implemented below. Two limits are worth checking
against intuition, and both hold: as ``psi -> 0`` the dimension carries no
between-speaker variability and contributes exactly zero, and as ``psi -> inf``
the contribution tends to ``0.5 log(psi/2) - (a-b)^2/4``, large when the two
observations agree and strongly negative when they do not.

What this does and does not give
--------------------------------
The output is a likelihood ratio *under the model's assumptions*: Gaussian
speaker and session distributions, and a training population representative of
the relevant population. Neither holds exactly. It is therefore treated
throughout as an uncalibrated score that must pass through empirical
calibration on held-out data before it is reported as evidence. Reporting a raw
PLDA score as a likelihood ratio is the specific error the forensic literature
warns against, and the fact that the number is *derived* as a ratio makes the
error easier to commit, not harder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

from viflap.domain.errors import (
    ConvergenceError,
    InsufficientDataError,
    InvalidEvidenceError,
)

__all__ = ["PldaConfig", "PldaModel", "train_plda"]


@dataclass(frozen=True, slots=True)
class PldaConfig:
    """Training parameters for the two-covariance model."""

    max_iterations: int = 50
    tolerance: float = 1e-6
    regularisation: float = 1e-8
    seed: int = 0

    min_speakers: int = 20
    """Below this the between-speaker covariance is an estimate from too few
    draws to be meaningful. The system refuses rather than producing likelihood
    ratios whose denominator rests on a dozen people."""

    min_recordings_per_speaker: int = 2
    """Speakers with a single recording contribute nothing to within-speaker
    variability. They still inform the between-speaker term and are retained for
    that, but at least some speakers must have repeats."""


@dataclass(frozen=True, slots=True)
class PldaModel:
    """A trained two-covariance PLDA model in its diagonalised form.

    Stored diagonalised rather than as raw ``B`` and ``W``. Scoring then needs
    no matrix operations at all — only elementwise arithmetic on the transformed
    vectors — which turns a database search from ``N`` matrix solves into one
    vectorised expression.
    """

    mean: NDArray[np.float64]
    transform: NDArray[np.float64]
    """Maps an observation into the space where ``W = I`` and ``B = diag(psi)``."""

    psi: NDArray[np.float64]
    """Between-speaker variance per diagonalised dimension."""

    n_training_speakers: int
    n_training_recordings: int

    def __post_init__(self) -> None:
        if np.any(self.psi < 0.0):
            raise InvalidEvidenceError(
                "between-speaker variances must be non-negative",
                minimum=float(np.min(self.psi)),
            )

    @property
    def dimension(self) -> int:
        return self.psi.shape[0]

    @property
    def effective_dimension(self) -> int:
        """Dimensions carrying meaningful between-speaker variability.

        Dimensions with ``psi`` near zero contribute nothing to any score. Their
        count is a useful diagnostic: if most dimensions are inert, the
        representation is not separating speakers and no amount of calibration
        will fix it.
        """
        return int(np.count_nonzero(self.psi > 1e-6))

    def project(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map observations into the diagonalised space."""
        single = np.asarray(vectors, dtype=np.float64).ndim == 1
        matrix = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        if matrix.shape[1] != self.mean.shape[0]:
            raise InvalidEvidenceError(
                "observation dimension differs from the model's",
                expected=int(self.mean.shape[0]),
                received=int(matrix.shape[1]),
            )
        projected = (matrix - self.mean) @ self.transform
        return projected[0] if single else projected

    def score(self, first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
        """Log-likelihood ratio for one pair, under the model.

        Symmetric in its arguments by construction, which is a property forensic
        comparison requires: the evidential value of a pair cannot depend on
        which recording the investigator happened to open first.
        """
        a = self.project(np.asarray(first, dtype=np.float64))
        b = self.project(np.asarray(second, dtype=np.float64))
        return float(np.sum(self._per_dimension_llr(a, b)))

    def score_many(
        self, probe: NDArray[np.float64], gallery: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score one observation against many, vectorised.

        The database search path. Broadcasting the closed form across the
        gallery keeps a search over a large enrolled population to a handful of
        array operations rather than one Python call per candidate.
        """
        a = self.project(np.asarray(probe, dtype=np.float64))
        others = self.project(np.atleast_2d(np.asarray(gallery, dtype=np.float64)))
        return np.sum(self._per_dimension_llr(a[None, :], others), axis=1)

    def _per_dimension_llr(
        self, a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Closed-form per-dimension log-likelihood ratio.

        Derived in the module docstring. Written out in full rather than
        factored, because every rearrangement of it that looks simpler turns out
        to lose the ``psi -> 0`` limit, where the dimension must contribute
        exactly zero and any residual accumulates across hundreds of inert
        dimensions into a large constant offset.
        """
        psi = self.psi
        total_variance = psi + 1.0
        determinant_ss = 2.0 * psi + 1.0

        log_determinant_term = 0.5 * (2.0 * np.log(total_variance) - np.log(determinant_ss))

        sum_of_squares = a**2 + b**2
        cross = a * b

        quadratic_ss = (
            total_variance * sum_of_squares - 2.0 * psi * cross
        ) / determinant_ss
        quadratic_ds = sum_of_squares / total_variance

        return log_determinant_term - 0.5 * quadratic_ss + 0.5 * quadratic_ds

    def diagnostics(self) -> dict[str, float]:
        """Model properties an analyst needs in order to trust a score."""
        return {
            "dimension": float(self.dimension),
            "effective_dimension": float(self.effective_dimension),
            "psi_mean": float(np.mean(self.psi)),
            "psi_max": float(np.max(self.psi)),
            "n_training_speakers": float(self.n_training_speakers),
            "n_training_recordings": float(self.n_training_recordings),
        }


def train_plda(
    vectors: NDArray[np.float64],
    speaker_labels: NDArray[np.int64],
    config: PldaConfig | None = None,
) -> PldaModel:
    """Train the two-covariance model by expectation-maximisation.

    Each iteration computes, for every speaker, the posterior over their latent
    position given their recordings:

    .. code-block:: text

        Precision_s = B^-1 + n_s W^-1
        Cov_s       = Precision_s^-1
        mean_s      = Cov_s W^-1 sum_i (x_si - mu)

    and then re-estimates

    .. code-block:: text

        B = (1/S) sum_s [ Cov_s + mean_s mean_s^T ]
        W = (1/N) sum_s sum_i [ (x_si - mu - mean_s)(...)^T + Cov_s ]

    The ``Cov_s`` term in the ``W`` update is what distinguishes this from
    naively splitting the total covariance. Omitting it treats each speaker's
    estimated position as if it were known exactly, which attributes the
    uncertainty in that estimate to within-speaker variability. The resulting
    ``W`` is too large and ``B`` too small, and the model systematically
    *understates* the evidential value of a genuine same-source pair —
    conservative in direction, but wrong, and wrong by an amount that grows as
    the recordings per speaker fall.
    """
    config = config or PldaConfig()
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    speaker_labels = np.asarray(speaker_labels)

    if vectors.shape[0] != speaker_labels.shape[0]:
        raise InvalidEvidenceError(
            "vector count and label count differ",
            n_vectors=int(vectors.shape[0]),
            n_labels=int(speaker_labels.shape[0]),
        )

    unique_speakers, counts = np.unique(speaker_labels, return_counts=True)
    if unique_speakers.size < config.min_speakers:
        raise InsufficientDataError(
            "too few speakers to estimate between-speaker variability; the "
            "denominator of every likelihood ratio this model produces would "
            "rest on this population",
            n_speakers=int(unique_speakers.size),
            required=config.min_speakers,
        )
    if np.max(counts) < config.min_recordings_per_speaker:
        raise InsufficientDataError(
            "no speaker has repeat recordings, so within-speaker variability "
            "cannot be estimated",
            max_recordings_per_speaker=int(np.max(counts)),
        )

    dimension = vectors.shape[1]
    mean = vectors.mean(axis=0)
    centred = vectors - mean

    groups = [np.flatnonzero(speaker_labels == speaker) for speaker in unique_speakers]

    between, within = _initialise_covariances(centred, groups, dimension, config)

    previous = -np.inf
    for iteration in range(config.max_iterations):
        between_inverse = _stable_inverse(between, config.regularisation)
        within_inverse = _stable_inverse(within, config.regularisation)

        accumulated_between = np.zeros((dimension, dimension))
        accumulated_within = np.zeros((dimension, dimension))
        log_likelihood = 0.0
        n_recordings = 0

        # Speakers with the same recording count share a posterior precision, so
        # the inverse is computed once per distinct count rather than once per
        # speaker. On a corpus where most speakers have two or three recordings
        # this removes almost all of the linear algebra from the E-step.
        posterior_cache: dict[int, NDArray[np.float64]] = {}

        for indices in groups:
            count = indices.size
            if count not in posterior_cache:
                posterior_cache[count] = _stable_inverse(
                    between_inverse + count * within_inverse, config.regularisation
                )
            posterior_covariance = posterior_cache[count]

            observations = centred[indices]
            posterior_mean = posterior_covariance @ (
                within_inverse @ observations.sum(axis=0)
            )

            accumulated_between += posterior_covariance + np.outer(
                posterior_mean, posterior_mean
            )

            residuals = observations - posterior_mean
            accumulated_within += residuals.T @ residuals + count * posterior_covariance

            log_likelihood += float(-0.5 * np.sum(residuals @ within_inverse * residuals))
            n_recordings += count

        between = accumulated_between / len(groups)
        within = accumulated_within / n_recordings

        between = _symmetrise(between)
        within = _symmetrise(within)

        if not np.all(np.isfinite(between)) or not np.all(np.isfinite(within)):
            raise ConvergenceError(
                "PLDA training produced non-finite covariances", iteration=iteration
            )

        objective = log_likelihood / max(n_recordings, 1)
        if iteration > 0 and abs(objective - previous) < config.tolerance * max(
            abs(previous), 1.0
        ):
            break
        previous = objective

    transform, psi = _simultaneous_diagonalisation(between, within, config.regularisation)

    return PldaModel(
        mean=mean,
        transform=transform,
        psi=psi,
        n_training_speakers=int(unique_speakers.size),
        n_training_recordings=int(vectors.shape[0]),
    )


def _initialise_covariances(
    centred: NDArray[np.float64],
    groups: list[NDArray[np.int64]],
    dimension: int,
    config: PldaConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Moment-based starting point for the EM iteration.

    Within-speaker covariance from the pooled within-speaker scatter, and
    between-speaker as the remainder of the total. Starting EM from a sensible
    decomposition rather than from the identity saves iterations and, more
    importantly, avoids a starting point from which the model can converge to a
    solution attributing everything to one term.
    """
    within = np.zeros((dimension, dimension))
    contributing = 0
    for indices in groups:
        if indices.size < 2:
            continue
        observations = centred[indices]
        deviations = observations - observations.mean(axis=0)
        within += deviations.T @ deviations / (indices.size - 1)
        contributing += 1

    if contributing == 0:
        raise InsufficientDataError(
            "no speaker has repeat recordings, so within-speaker variability "
            "cannot be initialised"
        )
    within /= contributing

    total = centred.T @ centred / centred.shape[0]
    between = total - within
    # The subtraction can leave small negative eigenvalues from estimation
    # noise. Projecting onto the positive semi-definite cone is the standard
    # repair; leaving them makes the first inverse ill-conditioned.
    between = _project_positive_semidefinite(between)

    ridge = np.eye(dimension) * config.regularisation
    return between + ridge, within + ridge


def _simultaneous_diagonalisation(
    between: NDArray[np.float64], within: NDArray[np.float64], regularisation: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Find the basis in which ``W`` is the identity and ``B`` is diagonal.

    Solves the generalised symmetric eigenproblem ``B v = lambda W v``. The
    resulting transform ``A`` satisfies ``A^T W A = I`` and ``A^T B A =
    diag(lambda)`` exactly, so the scoring formula's assumption is not an
    approximation but a property of the coordinates.
    """
    dimension = between.shape[0]
    ridge = np.eye(dimension) * regularisation * max(np.trace(within) / dimension, 1e-12)

    eigenvalues, eigenvectors = eigh(_symmetrise(between), _symmetrise(within) + ridge)

    ordering = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[ordering]
    eigenvectors = eigenvectors[:, ordering]

    # Clip at zero rather than raising. Small negative eigenvalues are the
    # signature of estimation noise in a direction carrying no between-speaker
    # variability; the correct value is zero, and a dimension with psi = 0
    # contributes exactly zero to every score, which is the right outcome.
    psi = np.maximum(eigenvalues, 0.0)
    return eigenvectors, psi


def _symmetrise(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Force exact symmetry, which accumulated arithmetic erodes."""
    return 0.5 * (matrix + matrix.T)


def _project_positive_semidefinite(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Nearest positive semi-definite matrix, by clipping eigenvalues at zero."""
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetrise(matrix))
    return eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def _stable_inverse(
    matrix: NDArray[np.float64], regularisation: float
) -> NDArray[np.float64]:
    """Invert a symmetric matrix with a scale-aware ridge.

    The ridge is proportional to the matrix's own trace rather than a fixed
    absolute value. A fixed ridge either dominates a small-scale covariance or
    fails to stabilise a large-scale one, and which of those happens depends on
    the arbitrary scaling of the i-vectors.
    """
    dimension = matrix.shape[0]
    scale = max(np.trace(matrix) / dimension, 1e-12)
    ridge = np.eye(dimension) * regularisation * scale
    try:
        return np.linalg.inv(_symmetrise(matrix) + ridge)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(_symmetrise(matrix) + ridge)
