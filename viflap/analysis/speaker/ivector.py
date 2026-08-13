"""Total variability modelling: the i-vector.

The problem this solves
-----------------------
MAP adaptation of a UBM gives a supervector of adapted means — for a
256-component model over 60-dimensional features, 15,360 numbers per recording.
That representation is far too large to model directly: estimating a covariance
in that space needs more data than exists, and most of its dimensions are
occupied by variation that has nothing to do with the speaker.

Joint factor analysis attacked this by modelling speaker and channel variability
in separate subspaces. Dehak's insight (Dehak et al., 2011) was that the
separation is unnecessary and in fact harmful: the "channel" subspace turns out
to contain speaker information too. Model *all* the variability in a single
low-dimensional subspace, and leave the separation of speaker from channel to a
discriminative back-end that can be trained for exactly that.

The model
---------
A recording's adapted mean supervector is

.. code-block:: text

    M(u) = m + T w(u),      w(u) ~ N(0, I)

with ``m`` the UBM supervector, ``T`` the total variability matrix of shape
``(C*D, R)`` for subspace rank ``R``, and ``w`` the latent factor. The i-vector
is the MAP point estimate of ``w`` — its posterior mean.

Given centred Baum-Welch statistics ``N_c`` and ``F_c``, the posterior of ``w``
is Gaussian with precision and mean

.. code-block:: text

    L(u)     = I + sum_c N_c(u) T_c^T Sigma_c^-1 T_c
    w_hat(u) = L(u)^-1 sum_c T_c^T Sigma_c^-1 F_c(u)

Note what ``L`` depends on: only the zeroth-order statistics. A short recording
has small ``N_c``, so ``L`` is close to the identity and ``w_hat`` is shrunk
toward zero — the prior. This is the correct behaviour and it is the reason
i-vectors from short utterances are systematically less informative rather than
merely noisier. The magnitude of that shrinkage is recoverable from the
posterior covariance and is reported, so that duration effects appear in the
diagnostics rather than as an unexplained weak likelihood ratio.

Training ``T`` is expectation-maximisation over the same posterior, and it is
unsupervised: no speaker labels are required, only a corpus spanning the
variability the deployment will encounter. Speaker labels enter later, in the
back-end.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from viflap.analysis.speaker.gmm import BaumWelchStatistics, GaussianMixture
from viflap.domain.errors import (
    ConvergenceError,
    InsufficientDataError,
    InvalidEvidenceError,
)

__all__ = [
    "IVectorExtractor",
    "IVectorPosterior",
    "TotalVariabilityConfig",
    "train_total_variability",
]


@dataclass(frozen=True, slots=True)
class TotalVariabilityConfig:
    """Training parameters for the total variability matrix."""

    rank: int = 200
    """Subspace dimension. The literature uses 400-600 for wideband corpora with
    thousands of hours. For narrowband telephony with a low-resource training
    set, a lower rank is not a compromise but the correct choice: an
    over-parameterised ``T`` fits channel idiosyncrasies of the training
    recordings, and the back-end then has to remove what the front-end should
    not have encoded."""

    max_iterations: int = 20
    tolerance: float = 1e-4
    seed: int = 0
    minimum_divergence: bool = True
    """Apply minimum-divergence re-estimation after each M-step. Rescales ``T``
    so the empirical distribution of the latent factors matches the standard
    normal the model assumes. Without it the prior is misspecified — the
    posterior shrinkage is then wrong, and wrong in a duration-dependent way,
    which is precisely the dependency this system needs to characterise."""

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise InvalidEvidenceError("subspace rank must be positive", rank=self.rank)


@dataclass(frozen=True, slots=True)
class IVectorPosterior:
    """The posterior over the latent factor for one recording."""

    mean: NDArray[np.float64]
    """The i-vector: the posterior mean, shape ``(rank,)``."""

    covariance: NDArray[np.float64]
    """Posterior covariance, shape ``(rank, rank)``. Retained because it
    quantifies how much the estimate was shrunk toward the prior, which is the
    formal expression of "this recording was too short to say much"."""

    n_frames: int
    occupancy: float

    @property
    def shrinkage(self) -> float:
        """How far the estimate was pulled toward the prior, in ``[0, 1]``.

        The mean posterior variance across dimensions. The prior has unit
        variance, so a value near one means the data contributed almost nothing
        and the i-vector is essentially the prior mean; near zero means the
        recording determined the factor well.

        Reported with every acoustic likelihood ratio. An investigator is
        entitled to know that a result came from eleven seconds of speech.
        """
        return float(np.clip(np.mean(np.diag(self.covariance)), 0.0, 1.0))

    @property
    def is_well_determined(self) -> bool:
        return self.shrinkage < 0.5


class IVectorExtractor:
    """Extracts i-vectors given a UBM and a trained total variability matrix.

    Immutable after construction. Extraction is a pure function of the
    statistics, which matters for reproducibility: the same recording and the
    same model must give the same i-vector on any machine and in any order.
    """

    def __init__(
        self,
        ubm: GaussianMixture,
        total_variability: NDArray[np.float64],
    ) -> None:
        n_components, n_dimensions = ubm.means.shape
        expected = n_components * n_dimensions
        if total_variability.shape[0] != expected:
            raise InvalidEvidenceError(
                "total variability matrix disagrees with the UBM supervector size",
                expected_rows=expected,
                received_rows=total_variability.shape[0],
            )
        self._ubm = ubm
        self._t = np.asarray(total_variability, dtype=np.float64)
        self._rank = self._t.shape[1]

        # Reshape to (C, D, R) so component blocks can be indexed directly, and
        # precompute the per-component terms that do not depend on the utterance.
        self._t_blocks = self._t.reshape(n_components, n_dimensions, self._rank)
        precision = 1.0 / ubm.variances  # (C, D)
        self._t_precision = self._t_blocks * precision[:, :, None]  # (C, D, R)

        # T_c^T Sigma_c^-1 T_c for every component, shape (C, R, R). This is the
        # expensive part of extraction and it is identical for every recording,
        # so computing it once at construction turns each extraction into a
        # weighted sum of C precomputed matrices.
        self._t_precision_t = np.einsum("cdr,cds->crs", self._t_blocks, self._t_precision)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def ubm(self) -> GaussianMixture:
        return self._ubm

    @property
    def matrix(self) -> NDArray[np.float64]:
        return self._t

    def extract(self, statistics: BaumWelchStatistics) -> IVectorPosterior:
        """Compute the posterior over the latent factor for one recording."""
        if statistics.n_components != self._ubm.n_components:
            raise InvalidEvidenceError(
                "statistics were accumulated against a different UBM",
                expected=self._ubm.n_components,
                received=statistics.n_components,
            )

        precision = np.eye(self._rank) + np.einsum(
            "c,crs->rs", statistics.zeroth, self._t_precision_t
        )
        linear = np.einsum("cdr,cd->r", self._t_precision, statistics.first)

        # Solve rather than invert: forming L^-1 explicitly and multiplying is
        # both slower and numerically worse than a Cholesky solve, and the
        # covariance is needed anyway so it is obtained from the same
        # factorisation.
        cholesky = np.linalg.cholesky(precision)
        mean = _cholesky_solve(cholesky, linear)
        covariance = _cholesky_inverse(cholesky)

        return IVectorPosterior(
            mean=mean,
            covariance=covariance,
            n_frames=statistics.n_frames,
            occupancy=statistics.effective_occupancy,
        )

    def extract_from_features(self, features: NDArray[np.float64]) -> IVectorPosterior:
        """Convenience: accumulate statistics and extract in one step."""
        return self.extract(self._ubm.baum_welch(features))


def train_total_variability(
    statistics: Sequence[BaumWelchStatistics],
    ubm: GaussianMixture,
    config: TotalVariabilityConfig | None = None,
) -> NDArray[np.float64]:
    """Train the total variability matrix by expectation-maximisation.

    Unsupervised: the recordings need not be labelled by speaker. What they must
    do is *span the variability of the deployment* — the handsets, networks,
    bitrates, languages and acoustic environments the system will encounter. A
    ``T`` trained on clean wideband speech has no basis on which to represent
    the variability of a coded narrowband call, and no amount of back-end
    training recovers information the subspace never encoded.

    The E-step accumulates, over recordings ``u``:

    .. code-block:: text

        A_c = sum_u N_c(u) E[w w^T](u)          for each component c
        C_c = sum_u F_c(u) E[w](u)^T

    and the M-step solves ``T_c = C_c A_c^-1`` per component. The per-component
    structure is what keeps this tractable: the full ``(C*D, R)`` least-squares
    problem factorises into ``C`` independent ``(D, R)`` problems.
    """
    config = config or TotalVariabilityConfig()
    if not statistics:
        raise InsufficientDataError("no statistics supplied for training")

    n_components, n_dimensions = ubm.means.shape
    rank = config.rank
    if len(statistics) < rank:
        raise InsufficientDataError(
            "fewer training recordings than subspace dimensions; the matrix "
            "would be determined by an arbitrary choice among infinitely many "
            "exact fits",
            n_recordings=len(statistics),
            rank=rank,
        )

    rng = np.random.default_rng(config.seed)
    # Initialise at a small random matrix scaled by the UBM variances, so the
    # initial subspace is dimensionally commensurate with the data. Starting
    # from a matrix of the wrong magnitude wastes iterations rescaling and can
    # produce an ill-conditioned first precision matrix.
    scale = np.sqrt(ubm.variances).reshape(-1)[:, None]
    t_matrix = rng.standard_normal((n_components * n_dimensions, rank)) * scale * 0.1

    zeroth = np.stack([item.zeroth for item in statistics])
    first = np.stack([item.first for item in statistics])

    previous_objective = -np.inf
    for iteration in range(config.max_iterations):
        extractor = IVectorExtractor(ubm, t_matrix)

        first_moments = np.zeros((len(statistics), rank))
        second_moments = np.zeros((len(statistics), rank, rank))

        for index, item in enumerate(statistics):
            posterior = extractor.extract(item)
            first_moments[index] = posterior.mean
            # E[w w^T] = Cov + mean mean^T. Omitting the covariance term is the
            # classic mistake here: it turns EM into an alternating least
            # squares that ignores posterior uncertainty, systematically
            # over-fitting short recordings whose factors are barely determined.
            second_moments[index] = posterior.covariance + np.outer(
                posterior.mean, posterior.mean
            )

        # A_c = sum_u N_c(u) E[w w^T](u), shape (C, R, R)
        accumulated_second = np.einsum("uc,urs->crs", zeroth, second_moments)
        # C_c = sum_u F_c(u) E[w](u)^T, shape (C, D, R)
        accumulated_cross = np.einsum("ucd,ur->cdr", first, first_moments)

        new_blocks = np.zeros((n_components, n_dimensions, rank))
        for component in range(n_components):
            # Ridge term for components that no recording visited often enough
            # to determine their block. Without it the solve is singular and the
            # whole matrix becomes NaN because of a few rare components.
            regularised = accumulated_second[component] + np.eye(rank) * 1e-6
            new_blocks[component] = np.linalg.solve(
                regularised, accumulated_cross[component].T
            ).T

        t_matrix = new_blocks.reshape(n_components * n_dimensions, rank)

        if config.minimum_divergence:
            t_matrix = _minimum_divergence_update(t_matrix, second_moments)

        objective = float(np.mean(np.sum(first_moments**2, axis=1)))
        if not np.isfinite(objective):
            raise ConvergenceError(
                "total variability training diverged", iteration=iteration
            )
        if iteration > 0 and abs(objective - previous_objective) < config.tolerance * max(
            abs(previous_objective), 1.0
        ):
            break
        previous_objective = objective

    return t_matrix


def _minimum_divergence_update(
    t_matrix: NDArray[np.float64], second_moments: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Rescale ``T`` so the empirical latent distribution matches its prior.

    The model asserts ``w ~ N(0, I)``. After an M-step the empirical second
    moment of the posterior means is generally not the identity, so the prior is
    misspecified. Absorbing the empirical covariance into ``T`` — replacing
    ``T`` with ``T L`` where ``L L^T`` is the empirical second moment — restores
    the assumption without changing the model's likelihood.

    This is not cosmetic. The prior is what produces the shrinkage of short
    recordings toward zero, and a misspecified prior gives the wrong shrinkage.
    Since duration is one of the principal factors governing how much evidence a
    recording carries, getting that relationship wrong distorts the very
    quantity this system exists to report.
    """
    empirical = np.mean(second_moments, axis=0)
    empirical = 0.5 * (empirical + empirical.T)
    empirical += np.eye(empirical.shape[0]) * 1e-9
    try:
        factor = np.linalg.cholesky(empirical)
    except np.linalg.LinAlgError:
        return t_matrix
    return t_matrix @ factor


def _cholesky_solve(
    cholesky: NDArray[np.float64], rhs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Solve ``L L^T x = rhs`` by forward and back substitution."""
    from scipy.linalg import solve_triangular

    intermediate = solve_triangular(cholesky, rhs, lower=True)
    return solve_triangular(cholesky.T, intermediate, lower=False)


def _cholesky_inverse(cholesky: NDArray[np.float64]) -> NDArray[np.float64]:
    """Invert a matrix from its Cholesky factor."""
    from scipy.linalg import solve_triangular

    identity = np.eye(cholesky.shape[0])
    intermediate = solve_triangular(cholesky, identity, lower=True)
    return solve_triangular(cholesky.T, intermediate, lower=False)
