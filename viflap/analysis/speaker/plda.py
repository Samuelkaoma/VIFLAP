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


#: Below this between-speaker variance a diagonalised dimension is treated as
#: inert. ``W = I`` in that space, so ``psi`` is a between-to-within variance
#: ratio: at 0.1 the between-speaker standard deviation is under a third of the
#: within-speaker one, and two recordings of the same person differ along the
#: dimension by more than two different people do. Such a dimension is not
#: literally worthless — nothing here removes it from the score — but counting
#: it as evidence of a healthy subspace is what the previous 1e-6 threshold did,
#: and that is what this constant exists to stop.
INERT_PSI = 0.1


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

    n_iterations: int = 0
    """EM iterations actually run, which is not the configured maximum.

    Zero on models trained before this was recorded, which means unknown rather
    than none."""

    final_log_likelihood: float = float("nan")
    """Observed-data log-likelihood per recording at the parameters returned.

    The quantity the stopping rule reads. Recorded because a stopping rule is
    only interpretable beside the value it stopped at: a run that halted after
    three iterations at a likelihood still climbing steeply and a run that
    halted after forty at a flat one are different models, and the iteration
    count alone does not distinguish them."""

    converged: bool = False
    """Whether the tolerance was met, as against the iteration cap being hit."""

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
        """Dimensions carrying enough between-speaker variability to matter.

        Dimensions with ``psi`` near zero contribute nothing to any score. Their
        count is a useful diagnostic: if most dimensions are inert, the
        representation is not separating speakers and no amount of calibration
        will fix it.

        .. note::

           This tested ``psi > 1e-6`` until now, which is a test for exact
           numerical collapse and effectively never fires. It reported 100 of
           100 dimensions healthy on every model this project has trained,
           including ones where 26 to 40 dimensions carry ``psi`` below 0.1 —
           so the one diagnostic meant to detect a degenerate back-end
           certified every model as sound, and did so by construction rather
           than by measurement.

        The threshold is absolute rather than relative to ``psi[0]``, and the
        choice is not arbitrary. In the diagonalised space ``W = I``, so ``psi``
        *is* the between-speaker variance expressed in units of the
        within-speaker variance, and it means the same thing in every model
        regardless of the i-vector scale. A relative threshold would call a
        dimension inert merely because some other dimension is strong, which is
        wrong: a dimension with ``psi = 1`` separates speakers usefully whether
        the leading dimension sits at 5 or at 50.

        See :data:`INERT_PSI` for what the level itself is derived from.
        """
        return int(np.count_nonzero(self.psi > INERT_PSI))

    @property
    def psi_ratio(self) -> float:
        """Leading between-speaker variance over the second.

        A spectrum whose first dimension dwarfs its second is the signature of
        one dominant axis of between-speaker variation — which is what a
        nuisance factor absorbed into the speaker subspace looks like, since a
        confound shared across a speaker's recordings is indistinguishable from
        the speaker as far as the model is concerned. Reported because it is the
        cheapest available check on that, and because it needs a number beside
        it rather than an eigenvalue plot nobody generates.
        """
        if self.psi.shape[0] < 2:
            return float("nan")
        ordered = np.sort(self.psi)[::-1]
        return float(ordered[0] / ordered[1]) if ordered[1] > 0.0 else float("inf")

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
        ordered = np.sort(self.psi)[::-1]
        return {
            "dimension": float(self.dimension),
            "effective_dimension": float(self.effective_dimension),
            "inert_psi_threshold": float(INERT_PSI),
            "psi_mean": float(np.mean(self.psi)),
            "psi_max": float(np.max(self.psi)),
            "psi_1": float(ordered[0]),
            "psi_2": float(ordered[1]) if ordered.size > 1 else float("nan"),
            "psi_ratio": self.psi_ratio,
            "n_training_speakers": float(self.n_training_speakers),
            "n_training_recordings": float(self.n_training_recordings),
            "n_iterations": float(self.n_iterations),
            "final_log_likelihood": float(self.final_log_likelihood),
            "converged": float(self.converged),
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
    objective = float("nan")
    converged = False
    iterations_run = 0

    for iteration in range(config.max_iterations):
        between_inverse = _stable_inverse(between, config.regularisation)
        within_inverse = _stable_inverse(within, config.regularisation)

        # Evaluated at the parameters this iteration starts from, so the
        # recorded sequence is L(theta_0), L(theta_1), ... and EM's guarantee
        # applies to consecutive entries. Computing it after the M-step instead
        # would interleave two different parameter sets and the monotonicity
        # check below would be testing nothing.
        objective = _observed_log_likelihood(
            centred, groups, between, within, between_inverse, within_inverse, config
        )
        iterations_run = iteration + 1

        # EM cannot decrease the observed-data likelihood. A decrease is a
        # defect in the update equations, not a property of the data, and it is
        # the one check that distinguishes a correct M-step from a plausible
        # wrong one — so it is enforced rather than logged. The tolerance
        # absorbs the ridge added to both covariances, which perturbs the exact
        # EM by a little and can cost a few units in the last place.
        if iteration > 0 and objective < previous - 1e-6 * max(abs(previous), 1.0):
            raise ConvergenceError(
                "the PLDA observed-data log-likelihood decreased, which "
                "expectation-maximisation cannot do; the update equations are "
                "wrong rather than the data being difficult",
                iteration=iteration,
                previous=round(previous, 9),
                current=round(objective, 9),
            )

        if iteration > 0 and abs(objective - previous) < config.tolerance * max(
            abs(previous), 1.0
        ):
            converged = True
            break
        previous = objective

        accumulated_between = np.zeros((dimension, dimension))
        accumulated_within = np.zeros((dimension, dimension))
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
            n_recordings += count

        between = accumulated_between / len(groups)
        within = accumulated_within / n_recordings

        between = _symmetrise(between)
        within = _symmetrise(within)

        if not np.all(np.isfinite(between)) or not np.all(np.isfinite(within)):
            raise ConvergenceError(
                "PLDA training produced non-finite covariances", iteration=iteration
            )
    else:
        # The cap was reached rather than the tolerance met, so the last thing
        # the loop did was an M-step and the parameters have moved past the last
        # objective evaluated. Re-evaluating costs one pass and keeps the
        # reported likelihood a property of the model actually returned, which
        # is the only reading of it that means anything.
        objective = _observed_log_likelihood(
            centred,
            groups,
            between,
            within,
            _stable_inverse(between, config.regularisation),
            _stable_inverse(within, config.regularisation),
            config,
        )

    transform, psi = _simultaneous_diagonalisation(between, within, config.regularisation)

    return PldaModel(
        mean=mean,
        transform=transform,
        psi=psi,
        n_training_speakers=int(unique_speakers.size),
        n_training_recordings=int(vectors.shape[0]),
        n_iterations=iterations_run,
        final_log_likelihood=objective,
        converged=converged,
    )


def _observed_log_likelihood(
    centred: NDArray[np.float64],
    groups: list[NDArray[np.int64]],
    between: NDArray[np.float64],
    within: NDArray[np.float64],
    between_inverse: NDArray[np.float64],
    within_inverse: NDArray[np.float64],
    config: PldaConfig,
) -> float:
    """Observed-data log-likelihood per recording, marginalising the latent term.

    The quantity that was previously tracked was the quadratic data term alone —
    no ``log|W|``, no ``log|B|``, no latent prior and no posterior-covariance
    trace. That is neither the observed-data likelihood nor the evidence lower
    bound, and nothing guarantees it increases, so the stopping rule was halting
    wherever a quantity with no monotonicity property happened to settle and the
    standard sanity check on an EM implementation was unavailable.

    For speaker ``s`` the ``n_s`` observations are jointly Gaussian with
    covariance ``I ⊗ W + 11ᵀ ⊗ B``. The determinant lemma and Woodbury reduce
    that to quantities the E-step already forms:

    .. code-block:: text

        log|Σ_s| = n_s log|W| + log|B| - log|Cov_s|
        xᵀ Σ_s⁻¹ x = Σ_i x_iᵀ W⁻¹ x_i - (Σ_i x_i)ᵀ W⁻¹ Cov_s W⁻¹ (Σ_i x_i)

    with ``Cov_s = (B⁻¹ + n_s W⁻¹)⁻¹``. So the exact marginal costs one extra
    determinant per distinct recording count, not per speaker, and the cache
    that already exists for the posterior covariance serves both.

    Divided by the recording count so the number is comparable across corpora
    and the tolerance means the same thing on 600 recordings as on 1,500.
    """
    dimension = centred.shape[1]
    # The ridged matrices, because those are the ones the E-step inverted. See
    # _ridged: scoring the unridged determinants against ridged inverses is
    # scoring one model with another's sufficient statistics.
    sign_within, log_det_within = np.linalg.slogdet(_ridged(within, config.regularisation))
    sign_between, log_det_between = np.linalg.slogdet(
        _ridged(between, config.regularisation)
    )
    if sign_within <= 0 or sign_between <= 0:
        raise ConvergenceError(
            "a PLDA covariance stopped being positive definite during training",
            within_determinant_sign=float(sign_within),
            between_determinant_sign=float(sign_between),
        )

    posterior_cache: dict[int, tuple[NDArray[np.float64], float]] = {}
    total = 0.0
    n_recordings = 0

    for indices in groups:
        count = int(indices.size)
        if count not in posterior_cache:
            covariance = _stable_inverse(
                between_inverse + count * within_inverse, config.regularisation
            )
            _, log_det_covariance = np.linalg.slogdet(covariance)
            posterior_cache[count] = (covariance, float(log_det_covariance))
        covariance, log_det_covariance = posterior_cache[count]

        observations = centred[indices]
        summed = observations.sum(axis=0)
        quadratic = float(np.sum(observations @ within_inverse * observations)) - float(
            summed @ within_inverse @ covariance @ within_inverse @ summed
        )
        log_determinant = count * log_det_within + log_det_between - log_det_covariance
        total += -0.5 * (
            count * dimension * float(np.log(2.0 * np.pi)) + log_determinant + quadratic
        )
        n_recordings += count

    return total / max(n_recordings, 1)


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
    ridged = _ridged(matrix, regularisation)
    try:
        return np.asarray(np.linalg.inv(ridged), dtype=np.float64)
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.pinv(ridged), dtype=np.float64)


def _ridged(matrix: NDArray[np.float64], regularisation: float) -> NDArray[np.float64]:
    """The symmetrised matrix the inverse is actually taken of.

    Exposed rather than buried inside :func:`_stable_inverse` because the
    likelihood has to be evaluated on the *same* matrix the E-step inverts. The
    ridge makes the algorithm exact expectation-maximisation for the ridged
    model and only approximate for the unridged one, so scoring the unridged
    determinant against ridged inverses mixes two models — and the monotonicity
    that EM guarantees then fails by a small amount that looks exactly like a
    defect in the update equations. On this project's compact test
    configuration it failed by 7 parts in 100,000, which is far too large to
    dismiss as arithmetic and was entirely this mismatch.
    """
    dimension = matrix.shape[0]
    scale = max(float(np.trace(matrix)) / dimension, 1e-12)
    ridge = np.eye(dimension, dtype=np.float64) * (regularisation * scale)
    return np.asarray(_symmetrise(matrix) + ridge, dtype=np.float64)
