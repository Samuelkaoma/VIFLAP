"""Fusion models, from the prohibited baseline to dependence-corrected forms.

Four implementations, corresponding to the three approaches set out in the
research proposal plus the baseline they are measured against.

:class:`NaiveIndependentFusion`
    Summation. Correct only under conditional independence, which is false here.
    Implemented so the cost of the assumption can be measured, and structurally
    barred from producing a reported result.

:class:`LinearLogisticFusion`
    Weighted linear combination with weights fitted to minimise ``C_llr``. The
    weights absorb *systematic* dependence: two streams that largely repeat each
    other end up sharing the weight one of them would have had alone. Cheap,
    standard, and provably insufficient where the dependence is strong or varies
    with the strength of the evidence — which is precisely the regime that
    produces the results anyone acts on.

:class:`GaussianLatentFusion`
    Models the joint distribution of the log-LR vector under each proposition as
    multivariate Gaussian, with an explicit common-factor decomposition
    representing the shared cause. Marginalises over absent streams analytically
    — exact for a Gaussian, and the reason this handles missingness properly
    rather than by imputation.

:class:`GaussianCopulaFusion`
    Separates each stream's marginal calibration from the dependence structure
    between them. Attractive because the two can be estimated from different
    amounts of data, and fragile because a misspecified copula family misleads
    most in the tails, which is where the strong results are.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

from viflap.analysis.fusion.base import (
    FusionTrainingSet,
    StreamVector,
    to_matrix,
)
from viflap.domain.errors import (
    CalibrationError,
    InsufficientDataError,
    InvalidEvidenceError,
    ModelNotTrainedError,
)
from viflap.domain.evidence import EvidenceStream
from viflap.domain.linkage import FusionMethod

__all__ = [
    "GaussianCopulaFusion",
    "GaussianLatentFusion",
    "LinearLogisticFusion",
    "NaiveIndependentFusion",
]

_LN2 = float(np.log(2.0))

#: Minimum comparisons of each class before a missingness pattern gets its own
#: fitted model. Below this the pattern is handled by marginalising the full
#: model, which is less flexible and far less likely to be fitting noise.
MIN_PATTERN_TRIALS = 40


def _hash_arrays(label: str, *arrays: NDArray[np.float64]) -> str:
    digest = hashlib.sha256(label.encode())
    for array in arrays:
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return f"{label}-{digest.hexdigest()[:12]}"


class NaiveIndependentFusion:
    """Summation of log-likelihood ratios. The baseline, not an option.

    Requires no training, because the conditional-independence assumption
    contains no free parameters — which is exactly why it is attractive and
    exactly why it is wrong. Its output is used in one place: as the comparator
    against which overstatement is measured.
    """

    @property
    def method(self) -> FusionMethod:
        return FusionMethod.NAIVE_INDEPENDENT

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def model_id(self) -> str:
        return "naive-independent"

    def supports_pattern(self, pattern: frozenset[EvidenceStream]) -> bool:
        return len(pattern) > 0

    def fit(self, training: FusionTrainingSet) -> NaiveIndependentFusion:
        return self

    def fuse(self, log_lrs: StreamVector) -> float:
        if not log_lrs:
            raise InvalidEvidenceError("no streams to fuse")
        return float(sum(log_lrs.values()))


class LinearLogisticFusion:
    """Weighted linear fusion with weights minimising ``C_llr``.

    The fused value is ``w_0 + sum_i w_i * l_i``. Weights are fitted on held-out
    development comparisons, not assumed.

    A separate model is fitted **per missingness pattern** where enough
    comparisons exist. This is not a refinement; it is required for correctness.
    A weight vector fitted where all five streams were present encodes how much
    each stream repeats the others *in that configuration*. Applying it to a
    comparison where two streams are absent leaves the remaining weights
    discounted for redundancy with evidence that is not there, systematically
    understating. Patterns with too few comparisons fall back to the full model
    restricted to the available columns, which is documented as an approximation
    rather than presented as the same thing.

    The objective is convex, so the fit is unique and reproducible. Weights are
    constrained to be non-negative: a negative weight means the model has
    concluded that a stream's support for linkage is evidence *against* it,
    which is occasionally the best fit to a development set and never something
    to deploy without understanding why.
    """

    def __init__(self, allow_per_pattern: bool = True) -> None:
        self._allow_per_pattern = allow_per_pattern
        self._streams: tuple[EvidenceStream, ...] = ()
        self._global_weights: NDArray[np.float64] | None = None
        self._global_bias = 0.0
        self._pattern_models: dict[
            frozenset[EvidenceStream], tuple[NDArray[np.float64], float]
        ] = {}
        self._fitted = False

    @property
    def method(self) -> FusionMethod:
        return FusionMethod.LINEAR_LOGISTIC

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def model_id(self) -> str:
        if self._global_weights is None:
            return "linear-logistic-unfitted"
        return _hash_arrays(
            "linear-logistic",
            self._global_weights,
            np.array([self._global_bias]),
        )

    @property
    def weights(self) -> dict[EvidenceStream, float]:
        """Global weights, for inspection and reporting.

        Interpretable and worth inspecting: a weight far below one means the
        stream is largely redundant with the others, which is a finding about
        the dependence structure rather than a tuning detail.
        """
        if self._global_weights is None:
            raise ModelNotTrainedError("fusion model has not been fitted")
        return dict(
            zip(
                self._streams,
                (float(weight) for weight in self._global_weights),
                strict=True,
            )
        )

    def supports_pattern(self, pattern: frozenset[EvidenceStream]) -> bool:
        return self._fitted and bool(pattern) and pattern <= set(self._streams)

    def fit(self, training: FusionTrainingSet) -> LinearLogisticFusion:
        self._streams = training.streams
        if not self._streams:
            raise InsufficientDataError("training set contains no streams")

        n_same, n_different = training.counts()
        if n_same < 10 or n_different < 10:
            raise InsufficientDataError(
                "fusion requires comparisons of both types",
                n_same_source=n_same,
                n_different_source=n_different,
            )

        matrix = to_matrix(training.observations, self._streams)
        labels = training.labels()

        # The global fit treats an absent stream as contributing nothing to the
        # linear combination for that comparison. This is a compromise and is
        # named as one: it is used only as the fallback for rare patterns, and
        # for common patterns the per-pattern fit below avoids it entirely.
        self._global_weights, self._global_bias = _fit_cllr_linear(
            np.nan_to_num(matrix, nan=0.0), labels, ~np.isnan(matrix)
        )

        if self._allow_per_pattern:
            for pattern, count in training.patterns.items():
                if count < 2 * MIN_PATTERN_TRIALS:
                    continue
                subset = training.subset(pattern)
                subset_same, subset_different = subset.counts()
                if min(subset_same, subset_different) < MIN_PATTERN_TRIALS:
                    continue
                ordered = tuple(
                    stream for stream in EvidenceStream.ordered() if stream in pattern
                )
                sub_matrix = to_matrix(subset.observations, ordered)
                weights, bias = _fit_cllr_linear(
                    sub_matrix, subset.labels(), np.ones_like(sub_matrix, dtype=bool)
                )
                self._pattern_models[pattern] = (weights, bias)

        self._fitted = True
        return self

    def fuse(self, log_lrs: StreamVector) -> float:
        if not self._fitted or self._global_weights is None:
            raise ModelNotTrainedError("fusion model has not been fitted")
        if not log_lrs:
            raise InvalidEvidenceError("no streams to fuse")

        pattern = frozenset(log_lrs)
        if pattern in self._pattern_models:
            weights, bias = self._pattern_models[pattern]
            ordered = tuple(
                stream for stream in EvidenceStream.ordered() if stream in pattern
            )
            return float(
                bias + sum(w * log_lrs[s] for w, s in zip(weights, ordered, strict=True))
            )

        total = self._global_bias
        for index, stream in enumerate(self._streams):
            value = log_lrs.get(stream)
            if value is not None:
                total += float(self._global_weights[index]) * value
        return float(total)

    def has_dedicated_model(self, pattern: frozenset[EvidenceStream]) -> bool:
        """Whether this pattern has its own fitted weights, or falls back."""
        return pattern in self._pattern_models


def _fit_cllr_linear(
    matrix: NDArray[np.float64],
    labels: NDArray[np.int64],
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], float]:
    """Fit weights and bias minimising ``C_llr``, with analytic gradient.

    ``mask`` marks which entries are genuinely observed; masked-out entries
    contribute nothing to the combination *or* to the gradient, so an absent
    stream neither pushes the fused value nor influences its own weight.
    """
    n_features = matrix.shape[1]
    same = labels == 1
    different = labels == 0
    design = np.where(mask, matrix, 0.0)

    def objective_and_gradient(
        parameters: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        weights = parameters[:-1]
        bias = parameters[-1]
        fused = design @ weights + bias

        same_fused = fused[same]
        different_fused = fused[different]
        objective = 0.5 * (
            float(np.mean(np.logaddexp2(0.0, -same_fused / _LN2)))
            + float(np.mean(np.logaddexp2(0.0, different_fused / _LN2)))
        )

        same_weight = -_logistic(-same_fused) / (_LN2 * same_fused.size)
        different_weight = _logistic(different_fused) / (_LN2 * different_fused.size)

        per_trial = np.zeros(fused.size)
        per_trial[same] = same_weight
        per_trial[different] = different_weight
        per_trial *= 0.5

        gradient_weights = design.T @ per_trial
        gradient_bias = float(np.sum(per_trial))
        return objective, np.append(gradient_weights, gradient_bias)

    start = np.append(np.ones(n_features), 0.0)
    result = minimize(
        objective_and_gradient,
        x0=start,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * n_features + [(None, None)],
    )
    if not result.success and result.status != 1:
        raise CalibrationError(
            "linear fusion failed to converge", message=str(result.message)
        )
    return result.x[:-1].copy(), float(result.x[-1])


def _logistic(x: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.empty_like(x)
    positive = x >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


@dataclass(frozen=True, slots=True)
class CommonFactorSummary:
    """How much of the joint variation is attributable to a single shared cause.

    Obtained by a one-factor decomposition of the same-source covariance,
    ``Sigma = lambda lambda^T + Psi``. The loadings say how strongly each stream
    responds to the common factor — which, in this application, *is* the shared
    actor running the shared operation.

    This is the quantity the research proposal's section 6.3 asks for. A high
    shared-variance fraction means the streams are largely restating one fact,
    and that summing them counts it repeatedly.
    """

    loadings: dict[EvidenceStream, float]
    shared_variance_fraction: float

    def describe(self) -> str:
        ordered = sorted(self.loadings.items(), key=lambda item: -abs(item[1]))
        parts = ", ".join(f"{stream.value} {loading:+.2f}" for stream, loading in ordered)
        return (
            f"A single common factor accounts for "
            f"{self.shared_variance_fraction:.0%} of the joint variation of the "
            f"stream log-likelihood ratios under the same-source proposition "
            f"(loadings: {parts}). Evidence sharing this much common cause "
            f"cannot be summed."
        )


class GaussianLatentFusion:
    """Multivariate Gaussian models of the joint log-LR vector under each proposition.

    .. code-block:: text

        log LR_fused = log N(l; mu_ss, Sigma_ss) - log N(l; mu_ds, Sigma_ds)

    Three properties make this the primary operational method.

    **Marginalisation over absent streams is exact.** The marginal of a Gaussian
    is a Gaussian with the corresponding sub-vector and sub-matrix. So a
    comparison missing two streams is handled by dropping two rows and columns —
    not by imputing values, and not by needing a separately fitted model.

    **Missing data in training is handled by expectation-maximisation**, not by
    mean imputation. Imputing the mean drives every estimated correlation toward
    zero, which makes the streams look more independent than they are — the
    exact error this model exists to correct, reintroduced during its own
    fitting.

    **Covariance is estimated with shrinkage.** Ten streams-worth of covariance
    is fifteen parameters per proposition; with a few hundred development
    comparisons the sample covariance is badly conditioned, and its inverse —
    which is what scoring uses — is worse. Ledoit-Wolf shrinkage toward a scaled
    identity is the standard remedy and it has a closed form, so it adds no
    tuning parameter.
    """

    def __init__(self, shrinkage: float | None = None, max_em_iterations: int = 50) -> None:
        self._shrinkage = shrinkage
        self._max_em_iterations = max_em_iterations
        self._streams: tuple[EvidenceStream, ...] = ()
        self._mean_same: NDArray[np.float64] | None = None
        self._cov_same: NDArray[np.float64] | None = None
        self._mean_different: NDArray[np.float64] | None = None
        self._cov_different: NDArray[np.float64] | None = None
        self._fitted = False

    @property
    def method(self) -> FusionMethod:
        return FusionMethod.GAUSSIAN_LATENT

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def model_id(self) -> str:
        if self._cov_same is None or self._cov_different is None:
            return "gaussian-latent-unfitted"
        return _hash_arrays(
            "gaussian-latent",
            self._mean_same,  # type: ignore[arg-type]
            self._cov_same,
            self._mean_different,  # type: ignore[arg-type]
            self._cov_different,
        )

    def supports_pattern(self, pattern: frozenset[EvidenceStream]) -> bool:
        return self._fitted and bool(pattern) and pattern <= set(self._streams)

    def fit(self, training: FusionTrainingSet) -> GaussianLatentFusion:
        self._streams = training.streams
        if not self._streams:
            raise InsufficientDataError("training set contains no streams")

        matrix = to_matrix(training.observations, self._streams)
        labels = training.labels()

        for name, selector in (
            ("same-source", labels == 1),
            ("different-source", labels == 0),
        ):
            if int(np.count_nonzero(selector)) < len(self._streams) + 5:
                raise InsufficientDataError(
                    f"too few {name} comparisons to estimate a joint distribution "
                    f"over {len(self._streams)} streams",
                    n_comparisons=int(np.count_nonzero(selector)),
                )

        self._mean_same, self._cov_same = _fit_gaussian_with_missing(
            matrix[labels == 1], self._shrinkage, self._max_em_iterations
        )
        self._mean_different, self._cov_different = _fit_gaussian_with_missing(
            matrix[labels == 0], self._shrinkage, self._max_em_iterations
        )
        self._fitted = True
        return self

    def fuse(self, log_lrs: StreamVector) -> float:
        if not self._fitted:
            raise ModelNotTrainedError("fusion model has not been fitted")
        if not log_lrs:
            raise InvalidEvidenceError("no streams to fuse")

        indices = [index for index, stream in enumerate(self._streams) if stream in log_lrs]
        if not indices:
            raise InvalidEvidenceError(
                "none of the supplied streams is known to this model",
                supplied=sorted(stream.value for stream in log_lrs),
                known=sorted(stream.value for stream in self._streams),
            )

        observation = np.array(
            [log_lrs[self._streams[index]] for index in indices], dtype=np.float64
        )
        grid = np.ix_(indices, indices)

        same = _gaussian_log_density(
            observation,
            self._mean_same[indices],  # type: ignore[index]
            self._cov_same[grid],  # type: ignore[index]
        )
        different = _gaussian_log_density(
            observation,
            self._mean_different[indices],  # type: ignore[index]
            self._cov_different[grid],  # type: ignore[index]
        )
        return float(same - different)

    def common_factor(self) -> CommonFactorSummary:
        """One-factor decomposition of the same-source covariance.

        Quantifies the shared cause directly, which is what makes the dependence
        argument concrete rather than assertional.
        """
        if not self._fitted or self._cov_same is None:
            raise ModelNotTrainedError("fusion model has not been fitted")

        correlation = _to_correlation(self._cov_same)
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        leading = int(np.argmax(eigenvalues))
        loading_vector = eigenvectors[:, leading] * np.sqrt(max(eigenvalues[leading], 0.0))

        # Sign is arbitrary in an eigendecomposition; orient so the majority of
        # loadings are positive, which makes the summary readable rather than
        # flipping between runs.
        if float(np.sum(loading_vector)) < 0.0:
            loading_vector = -loading_vector

        shared = float(np.sum(loading_vector**2) / correlation.shape[0])
        return CommonFactorSummary(
            loadings={
                stream: float(value)
                for stream, value in zip(self._streams, loading_vector, strict=True)
            },
            shared_variance_fraction=shared,
        )


def _fit_gaussian_with_missing(
    matrix: NDArray[np.float64], shrinkage: float | None, max_iterations: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit a multivariate Gaussian to data with missing entries, by EM.

    Each iteration imputes the conditional expectation of the missing entries
    given the observed ones under the current estimate, and — critically — adds
    the conditional *covariance* of those entries to the scatter. Omitting that
    second term is mean imputation wearing a different name: it treats each
    imputed value as if it were observed, and drives every estimated correlation
    involving a partly missing stream toward zero.
    """
    n_rows, n_columns = matrix.shape
    observed = ~np.isnan(matrix)

    # Initialise from per-column complete cases.
    mean = np.array(
        [
            float(np.nanmean(matrix[:, column])) if np.any(observed[:, column]) else 0.0
            for column in range(n_columns)
        ]
    )
    variance = np.array(
        [
            float(np.nanvar(matrix[:, column])) if np.any(observed[:, column]) else 1.0
            for column in range(n_columns)
        ]
    )
    covariance = np.diag(np.maximum(variance, 1e-6))

    if np.all(observed):
        return _shrink_covariance(matrix, shrinkage)

    for _ in range(max_iterations):
        completed = matrix.copy()
        extra_scatter = np.zeros((n_columns, n_columns))

        for row in range(n_rows):
            present = observed[row]
            missing = ~present
            if not np.any(missing):
                continue
            if not np.any(present):
                completed[row] = mean
                extra_scatter += covariance
                continue

            present_index = np.flatnonzero(present)
            missing_index = np.flatnonzero(missing)

            cov_mm = covariance[np.ix_(missing_index, missing_index)]
            cov_mp = covariance[np.ix_(missing_index, present_index)]
            cov_pp = covariance[np.ix_(present_index, present_index)]

            solved = np.linalg.solve(
                cov_pp + np.eye(present_index.size) * 1e-10,
                (matrix[row, present_index] - mean[present_index]),
            )
            completed[row, missing_index] = mean[missing_index] + cov_mp @ solved

            conditional = cov_mm - cov_mp @ np.linalg.solve(
                cov_pp + np.eye(present_index.size) * 1e-10, cov_mp.T
            )
            extra_scatter[np.ix_(missing_index, missing_index)] += conditional

        new_mean = completed.mean(axis=0)
        centred = completed - new_mean
        new_covariance = (centred.T @ centred + extra_scatter) / n_rows

        if np.allclose(new_mean, mean, atol=1e-8) and np.allclose(
            new_covariance, covariance, atol=1e-8
        ):
            mean, covariance = new_mean, new_covariance
            break
        mean, covariance = new_mean, new_covariance

    _, shrunk = _shrink_covariance_from_moments(covariance, n_rows, shrinkage)
    return mean, shrunk


def _shrink_covariance(
    matrix: NDArray[np.float64], shrinkage: float | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample mean and shrunk covariance for a complete matrix."""
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    sample = centred.T @ centred / matrix.shape[0]
    _, shrunk = _shrink_covariance_from_moments(sample, matrix.shape[0], shrinkage)
    return mean, shrunk


def _shrink_covariance_from_moments(
    sample: NDArray[np.float64], n_samples: int, shrinkage: float | None
) -> tuple[float, NDArray[np.float64]]:
    """Shrink a covariance estimate toward a scaled identity.

    ``Sigma_shrunk = (1 - a) Sigma_sample + a * (trace / p) * I``.

    With ``shrinkage=None`` the intensity is chosen by a Ledoit-Wolf-style rule
    based on the ratio of parameters to observations, which requires no tuning
    and degrades gracefully: it approaches zero when observations are plentiful
    and approaches full shrinkage when they are scarce.
    """
    dimension = sample.shape[0]
    target_scale = float(np.trace(sample) / dimension) if dimension else 1.0
    target = np.eye(dimension) * max(target_scale, 1e-12)

    if shrinkage is None:
        n_parameters = dimension * (dimension + 1) / 2.0
        intensity = float(np.clip(n_parameters / max(n_samples, 1), 0.0, 0.9))
    else:
        intensity = float(np.clip(shrinkage, 0.0, 1.0))

    shrunk = (1.0 - intensity) * sample + intensity * target
    shrunk = 0.5 * (shrunk + shrunk.T)
    # A final ridge guarantees invertibility even when every stream in the
    # development set was perfectly correlated with another.
    shrunk += np.eye(dimension) * max(target_scale, 1e-12) * 1e-6
    return intensity, shrunk


def _gaussian_log_density(
    observation: NDArray[np.float64],
    mean: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> float:
    """Multivariate Gaussian log density, via Cholesky.

    Cholesky rather than an explicit inverse: it gives the log-determinant and
    the quadratic form from one factorisation, and it fails loudly on a
    non-positive-definite covariance instead of silently returning a
    plausible-looking number computed from a meaningless inverse.
    """
    dimension = observation.size
    deviation = observation - mean
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded by shrinkage
        raise CalibrationError(
            "fusion covariance is not positive definite", dimension=dimension
        ) from exc

    solved = np.linalg.solve(factor, deviation)
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor))))
    quadratic = float(np.dot(solved, solved))
    return -0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant + quadratic)


def _to_correlation(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    deviation = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    return covariance / np.outer(deviation, deviation)


class GaussianCopulaFusion:
    """Empirical marginals with a Gaussian copula for the dependence structure.

    Sklar's theorem factorises any joint distribution into its marginals and a
    copula capturing the dependence between them. Applied here:

    .. code-block:: text

        log LR = sum_i [log f_ss,i(l_i) - log f_ds,i(l_i)]
               + log c_ss(u_ss) - log c_ds(u_ds)

    The first term is the sum of *marginal* log-likelihood ratios and the second
    is the dependence correction. The attraction is that the two are estimated
    separately: the marginals need only per-stream data, and the copula is a
    single correlation matrix.

    One implementation detail matters more than it appears to. The marginal
    log-LRs are computed from the **fitted marginal densities**, not taken to be
    the input values. Using the inputs directly assumes every stream is already
    perfectly calibrated, in which case the marginal log-LR and the input
    coincide. They do not coincide in practice, and the difference is silently
    absorbed into the dependence correction, which is then measuring stream
    miscalibration rather than dependence.

    The known weakness is the tail. A Gaussian copula has no tail dependence: it
    asserts that extreme values in two streams become asymptotically
    independent. If the truth is that extreme acoustic and behavioural evidence
    tend to co-occur — which is what one operator running one script would
    produce — this understates dependence exactly where the evidence is
    strongest, and understating dependence overstates the fused result.
    """

    def __init__(self, bandwidth_factor: float = 1.0) -> None:
        self._bandwidth_factor = bandwidth_factor
        self._streams: tuple[EvidenceStream, ...] = ()
        self._same_samples: dict[EvidenceStream, NDArray[np.float64]] = {}
        self._different_samples: dict[EvidenceStream, NDArray[np.float64]] = {}
        self._same_bandwidth: dict[EvidenceStream, float] = {}
        self._different_bandwidth: dict[EvidenceStream, float] = {}
        self._same_correlation: NDArray[np.float64] | None = None
        self._different_correlation: NDArray[np.float64] | None = None
        self._fitted = False

    @property
    def method(self) -> FusionMethod:
        return FusionMethod.GAUSSIAN_COPULA

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def model_id(self) -> str:
        if self._same_correlation is None or self._different_correlation is None:
            return "gaussian-copula-unfitted"
        return _hash_arrays(
            "gaussian-copula", self._same_correlation, self._different_correlation
        )

    def supports_pattern(self, pattern: frozenset[EvidenceStream]) -> bool:
        return self._fitted and bool(pattern) and pattern <= set(self._streams)

    def fit(self, training: FusionTrainingSet) -> GaussianCopulaFusion:
        self._streams = training.streams
        matrix = to_matrix(training.observations, self._streams)
        labels = training.labels()

        for index, stream in enumerate(self._streams):
            column = matrix[:, index]
            same = column[(labels == 1) & ~np.isnan(column)]
            different = column[(labels == 0) & ~np.isnan(column)]
            if same.size < 10 or different.size < 10:
                raise InsufficientDataError(
                    "too few observations to estimate marginal distributions",
                    stream=stream.value,
                    n_same_source=int(same.size),
                    n_different_source=int(different.size),
                )
            self._same_samples[stream] = np.sort(same)
            self._different_samples[stream] = np.sort(different)
            self._same_bandwidth[stream] = self._bandwidth(same)
            self._different_bandwidth[stream] = self._bandwidth(different)

        self._same_correlation = self._rank_correlation(matrix[labels == 1])
        self._different_correlation = self._rank_correlation(matrix[labels == 0])
        self._fitted = True
        return self

    def _bandwidth(self, values: NDArray[np.float64]) -> float:
        spread = float(np.std(values))
        iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
        if iqr > 0.0:
            spread = min(spread, iqr / 1.34)
        return max(self._bandwidth_factor * 0.9 * spread * values.size ** (-0.2), 1e-6)

    def _rank_correlation(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Correlation matrix estimated through Kendall's tau.

        ``rho = sin(pi * tau / 2)`` is the exact relationship between Kendall's
        tau and the Pearson correlation of the underlying Gaussian, for a
        Gaussian copula. Estimating tau rather than Pearson correlation directly
        makes the estimate invariant to the marginal distributions — which is
        the entire premise of the copula approach, and is lost if the
        correlation is estimated on the raw values.
        """
        from scipy.stats import kendalltau

        dimension = matrix.shape[1]
        correlation = np.eye(dimension)
        for i in range(dimension):
            for j in range(i + 1, dimension):
                both = ~np.isnan(matrix[:, i]) & ~np.isnan(matrix[:, j])
                if int(np.count_nonzero(both)) < 10:
                    continue
                tau = kendalltau(matrix[both, i], matrix[both, j]).statistic
                if not np.isfinite(tau):
                    continue
                rho = float(np.sin(np.pi * tau / 2.0))
                correlation[i, j] = correlation[j, i] = rho

        # Project onto the positive-definite cone. A correlation matrix built
        # pairwise need not be positive definite, and inverting one that is not
        # produces a copula density that is not a density.
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        eigenvalues = np.maximum(eigenvalues, 1e-4)
        repaired = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        scale = np.sqrt(np.diag(repaired))
        return repaired / np.outer(scale, scale)

    def fuse(self, log_lrs: StreamVector) -> float:
        if not self._fitted:
            raise ModelNotTrainedError("fusion model has not been fitted")
        if not log_lrs:
            raise InvalidEvidenceError("no streams to fuse")

        indices = [index for index, stream in enumerate(self._streams) if stream in log_lrs]
        present = [self._streams[index] for index in indices]
        if not present:
            raise InvalidEvidenceError("none of the supplied streams is known")

        marginal_total = 0.0
        same_u = np.empty(len(present))
        different_u = np.empty(len(present))

        for position, stream in enumerate(present):
            value = float(log_lrs[stream])
            same_density = _kde_log_density(
                value, self._same_samples[stream], self._same_bandwidth[stream]
            )
            different_density = _kde_log_density(
                value, self._different_samples[stream], self._different_bandwidth[stream]
            )
            marginal_total += same_density - different_density
            same_u[position] = _empirical_cdf(value, self._same_samples[stream])
            different_u[position] = _empirical_cdf(value, self._different_samples[stream])

        if len(present) == 1:
            return float(marginal_total)

        grid = np.ix_(indices, indices)
        same_copula = _gaussian_copula_log_density(
            same_u,
            self._same_correlation[grid],  # type: ignore[index]
        )
        different_copula = _gaussian_copula_log_density(
            different_u,
            self._different_correlation[grid],  # type: ignore[index]
        )
        return float(marginal_total + same_copula - different_copula)


def _kde_log_density(value: float, sample: NDArray[np.float64], bandwidth: float) -> float:
    """Gaussian kernel density at one point, evaluated in the log domain."""
    z = (value - sample) / bandwidth
    log_kernels = -0.5 * z**2 - 0.5 * np.log(2.0 * np.pi) - np.log(bandwidth)
    return float(logsumexp(log_kernels) - np.log(sample.size))


def _empirical_cdf(value: float, sorted_sample: NDArray[np.float64]) -> float:
    """Empirical CDF with a continuity correction.

    ``(rank + 0.5) / (n + 1)`` rather than ``rank / n``. The uncorrected form
    returns exactly zero or one at the extremes, whose Gaussian quantile is
    infinite, and the copula density then evaluates to ``nan`` for precisely the
    strongest observations.
    """
    position = float(np.searchsorted(sorted_sample, value, side="left"))
    return (position + 0.5) / (sorted_sample.size + 1.0)


def _gaussian_copula_log_density(
    u: NDArray[np.float64], correlation: NDArray[np.float64]
) -> float:
    """Log density of a Gaussian copula at ``u``.

    ``log c(u) = -0.5 log|R| - 0.5 z^T (R^-1 - I) z`` with ``z = Phi^-1(u)``.

    The ``-I`` term is what makes this a *copula* density rather than a Gaussian
    density: it divides out the standard normal marginals, leaving only the
    dependence. Dropping it — an easy omission — turns the correction into a
    second, incompatible model of the marginals.
    """
    z = norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
    sign, log_determinant = np.linalg.slogdet(correlation)
    if sign <= 0:  # pragma: no cover - guarded by the eigenvalue repair
        return 0.0
    inverse = np.linalg.inv(correlation)
    quadratic = float(z @ (inverse - np.eye(correlation.shape[0])) @ z)
    return float(-0.5 * log_determinant - 0.5 * quadratic)
