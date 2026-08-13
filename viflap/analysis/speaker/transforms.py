"""Inter-session compensation applied to i-vectors before the back-end.

Three transforms, applied in this order, each addressing a different way in
which raw i-vectors violate the assumptions of the back-end that consumes them.

**Centring and length normalisation.** The PLDA back-end assumes Gaussian
distributions. Raw i-vectors are not Gaussian: their length varies
systematically with utterance duration, producing heavy tails, and the tails are
exactly where the strong likelihood ratios come from. Projecting onto the unit
sphere after centring — radial Gaussianisation (Garcia-Romero and
Espy-Wilson, 2011) — removes the duration-driven length variation and brings the
distribution close enough to Gaussian for the back-end's closed form to be
meaningful rather than merely computable.

**Linear discriminant analysis.** Projects onto directions maximising
between-speaker over within-speaker variance. This is not primarily a
dimensionality reduction: it removes directions dominated by channel and session
variability, in which the narrowband telephony setting is unusually rich.

**Within-class covariance normalisation.** Whitens by the within-speaker
covariance so that a unit of distance means the same thing in every direction.
Without it, the back-end's estimate of within-speaker variability has to absorb
a scale that varies across dimensions, which it does by inflating the average.

The order is not arbitrary. Length normalisation before LDA, so the scatter
matrices are estimated on the Gaussianised representation; WCCN after LDA, so it
whitens the space the back-end actually sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

from viflap.domain.errors import InsufficientDataError, InvalidEvidenceError

__all__ = ["IVectorTransform", "fit_transform_chain"]


@dataclass(frozen=True, slots=True)
class IVectorTransform:
    """A fitted centring, projection and whitening chain.

    Stored as a single object because the components must be applied together
    and in order. Splitting them into separate fitted objects invites a caller
    to apply LDA without the centring it was estimated with, which produces a
    projection that is subtly wrong and entirely plausible-looking.
    """

    mean: NDArray[np.float64]
    projection: NDArray[np.float64]
    """Combined LDA and WCCN transform, shape ``(input_dim, output_dim)``."""

    length_normalise: bool = True

    @property
    def output_dimension(self) -> int:
        return self.projection.shape[1]

    def apply(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform one or more i-vectors.

        Accepts a single vector or a matrix of them and returns the same rank,
        so callers do not have to reshape around it.
        """
        single = np.asarray(vectors, dtype=np.float64).ndim == 1
        matrix = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        if matrix.shape[1] != self.mean.shape[0]:
            raise InvalidEvidenceError(
                "i-vector dimension differs from the fitted transform's",
                expected=int(self.mean.shape[0]),
                received=int(matrix.shape[1]),
            )

        centred = matrix - self.mean
        if self.length_normalise:
            centred = _length_normalise(centred)
        projected = centred @ self.projection
        if self.length_normalise:
            # Normalise again after projection. The first normalisation makes
            # the distribution isotropic in the input space; the projection
            # changes the space, so the property has to be re-established in the
            # one the back-end will actually work in.
            projected = _length_normalise(projected)
        return projected[0] if single else projected


def _length_normalise(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project onto the unit sphere, leaving zero vectors untouched."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=vectors.copy(), where=norms > 1e-12)


def fit_transform_chain(
    vectors: NDArray[np.float64],
    speaker_labels: NDArray[np.int64],
    lda_dimension: int | None = None,
    *,
    length_normalise: bool = True,
    regularisation: float = 1e-6,
) -> IVectorTransform:
    """Fit the centring, LDA and WCCN chain from labelled training i-vectors.

    Parameters
    ----------
    lda_dimension:
        Output dimension. Defaults to ``min(rank, n_speakers - 1)``, the latter
        being a hard mathematical ceiling: the between-speaker scatter of ``S``
        speakers has rank at most ``S - 1``, so any further dimension is
        spanned by numerical noise. Requesting more silently yields directions
        that look discriminative on the training set and are meaningless off it.

    Raises
    ------
    InsufficientDataError
        If there are too few speakers, or too few recordings per speaker, for
        the within-speaker scatter to be estimable. Both scatter matrices are
        required; a system trained where only one is estimable would report
        likelihood ratios with no basis for the variability it claims to be
        comparing against.
    """
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    speaker_labels = np.asarray(speaker_labels)
    if vectors.shape[0] != speaker_labels.shape[0]:
        raise InvalidEvidenceError(
            "vector count and label count differ",
            n_vectors=int(vectors.shape[0]),
            n_labels=int(speaker_labels.shape[0]),
        )

    unique_speakers, counts = np.unique(speaker_labels, return_counts=True)
    n_speakers = unique_speakers.size
    if n_speakers < 3:
        raise InsufficientDataError(
            "at least three speakers are required to estimate between-speaker variability",
            n_speakers=int(n_speakers),
        )
    if np.sum(counts > 1) < 2:
        raise InsufficientDataError(
            "at least two speakers must have more than one recording, or "
            "within-speaker variability cannot be estimated at all",
            n_speakers_with_repeats=int(np.sum(counts > 1)),
        )

    mean = vectors.mean(axis=0)
    centred = vectors - mean
    if length_normalise:
        centred = _length_normalise(centred)

    dimension = centred.shape[1]
    maximum_lda = min(dimension, n_speakers - 1)
    target = maximum_lda if lda_dimension is None else min(lda_dimension, maximum_lda)
    if target < 1:
        raise InsufficientDataError(
            "no discriminative dimensions are available", n_speakers=int(n_speakers)
        )

    between, within = _scatter_matrices(centred, speaker_labels, unique_speakers)

    # Regularise the within-class scatter before the generalised eigenproblem.
    # With more dimensions than recordings per speaker it is rank-deficient, and
    # the smallest eigenvalues — which are numerical noise — would otherwise
    # dominate the ratio being maximised and select pure noise directions.
    within = within + np.eye(dimension) * regularisation * np.trace(within) / dimension

    eigenvalues, eigenvectors = eigh(between, within)
    ordering = np.argsort(eigenvalues)[::-1][:target]
    lda = eigenvectors[:, ordering]

    projected = centred @ lda
    wccn = _wccn_transform(projected, speaker_labels, unique_speakers, regularisation)

    return IVectorTransform(
        mean=mean, projection=lda @ wccn, length_normalise=length_normalise
    )


def _scatter_matrices(
    vectors: NDArray[np.float64],
    labels: NDArray[np.int64],
    unique_speakers: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Between- and within-speaker scatter.

    The between-speaker scatter weights each speaker by their recording count.
    Weighting speakers equally instead would let a speaker with two recordings
    influence the estimated between-speaker variability as much as one with two
    hundred, which inflates the estimate with the sampling noise of the
    under-represented speakers.
    """
    dimension = vectors.shape[1]
    grand_mean = vectors.mean(axis=0)
    between = np.zeros((dimension, dimension))
    within = np.zeros((dimension, dimension))
    total = 0

    for speaker in unique_speakers:
        member_vectors = vectors[labels == speaker]
        count = member_vectors.shape[0]
        speaker_mean = member_vectors.mean(axis=0)

        offset = (speaker_mean - grand_mean)[:, None]
        between += count * (offset @ offset.T)

        deviations = member_vectors - speaker_mean
        within += deviations.T @ deviations
        total += count

    return between / total, within / total


def _wccn_transform(
    vectors: NDArray[np.float64],
    labels: NDArray[np.int64],
    unique_speakers: NDArray[np.int64],
    regularisation: float,
) -> NDArray[np.float64]:
    """Within-class covariance normalisation.

    Returns ``B`` such that ``x B`` has identity within-speaker covariance,
    obtained from the Cholesky factor of the inverse within-class covariance.

    Speakers are weighted equally here, unlike in the scatter estimate above.
    The quantity being estimated is the *typical* within-speaker covariance, and
    weighting by recording count would make it the covariance of whichever
    speaker happens to be over-represented in the training corpus — which, in a
    corpus assembled from operational case material, is generally the most
    prolific offender rather than a representative person.
    """
    dimension = vectors.shape[1]
    covariance = np.zeros((dimension, dimension))
    contributing = 0

    for speaker in unique_speakers:
        member_vectors = vectors[labels == speaker]
        if member_vectors.shape[0] < 2:
            continue
        deviations = member_vectors - member_vectors.mean(axis=0)
        covariance += (deviations.T @ deviations) / (member_vectors.shape[0] - 1)
        contributing += 1

    if contributing == 0:
        return np.eye(dimension)

    covariance /= contributing
    covariance += np.eye(dimension) * regularisation * np.trace(covariance) / dimension
    return np.linalg.cholesky(np.linalg.inv(covariance))
