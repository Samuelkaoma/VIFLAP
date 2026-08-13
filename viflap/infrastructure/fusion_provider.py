"""Adapter presenting a fitted fusion model as a :class:`FusionProvider`.

Adds two things the raw fusion model does not carry: the naive comparator used
to measure overstatement, and an uncertainty interval on the fused value.

Where the fused interval comes from
-----------------------------------
Not from a fixed width, and not from summing the per-stream intervals. Summing
them would assume the streams' *errors* independent — the same assumption the
fusion layer exists to correct, reintroduced one level up, and in the same
direction: it would produce an interval that is too narrow, implying more
precision than the system has.

Instead the interval is obtained by resampling. Each stream's log-LR is
perturbed within its own interval, correlated according to the dependence the
fusion model estimated, and the fused value recomputed. The spread of those
recomputations is the interval. This propagates dependence through the
uncertainty as well as through the point estimate, which is what makes the two
consistent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from viflap.analysis.fusion.base import FusionModel
from viflap.analysis.fusion.models import NaiveIndependentFusion
from viflap.domain.evidence import EvidenceStream
from viflap.domain.linkage import FusionMethod

__all__ = ["FittedFusionProvider"]


class FittedFusionProvider:
    """Wraps a fitted fusion model for use by the application layer."""

    def __init__(
        self,
        model: FusionModel,
        stream_spreads: Mapping[EvidenceStream, float] | None = None,
        correlation: Mapping[tuple[EvidenceStream, EvidenceStream], float] | None = None,
        n_resamples: int = 400,
        confidence_level: float = 0.95,
        seed: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        stream_spreads:
            Per-stream standard deviation of the calibrated log-LR on
            development trials, in nats. Absent streams get a conservative
            default; a stream whose spread was never measured should widen the
            interval, not narrow it.
        correlation:
            Estimated correlation between stream errors. Defaults to a moderate
            positive value between every pair, which is the conservative choice
            here: the streams share a cause, so assuming zero correlation would
            produce an interval narrower than the evidence supports.
        """
        self._model = model
        self._naive = NaiveIndependentFusion()
        self._spreads = dict(stream_spreads or {})
        self._correlation = dict(correlation or {})
        self._n_resamples = n_resamples
        self._confidence_level = confidence_level
        self._rng = np.random.default_rng(seed)

    @property
    def model_id(self) -> str:
        return self._model.model_id

    @property
    def method(self) -> FusionMethod:
        return self._model.method

    def supports_pattern(self, pattern: frozenset[EvidenceStream]) -> bool:
        return self._model.supports_pattern(pattern)

    def fuse(self, log_lrs: Mapping[EvidenceStream, float]) -> float:
        return self._model.fuse(dict(log_lrs))

    def fuse_naive(self, log_lrs: Mapping[EvidenceStream, float]) -> float:
        return self._naive.fuse(dict(log_lrs))

    def uncertainty_for(
        self, log_lrs: Mapping[EvidenceStream, float], fused: float
    ) -> tuple[float, float]:
        """Interval on the fused log-LR, by correlated resampling."""
        streams = sorted(log_lrs, key=lambda stream: stream.value)
        if len(streams) == 1:
            spread = self._spread_for(streams[0])
            return (fused - 1.96 * spread, fused + 1.96 * spread)

        covariance = self._covariance(streams)
        centre = np.array([log_lrs[stream] for stream in streams])

        try:
            perturbations = self._rng.multivariate_normal(
                np.zeros(len(streams)), covariance, size=self._n_resamples
            )
        except np.linalg.LinAlgError:  # pragma: no cover - guarded by the ridge
            spread = float(np.mean([self._spread_for(s) for s in streams]))
            return (fused - 1.96 * spread, fused + 1.96 * spread)

        fused_samples = []
        for perturbation in perturbations:
            perturbed = {
                stream: float(centre[index] + perturbation[index])
                for index, stream in enumerate(streams)
            }
            try:
                fused_samples.append(self._model.fuse(perturbed))
            except Exception:
                continue

        if len(fused_samples) < 20:
            spread = float(np.mean([self._spread_for(s) for s in streams]))
            return (fused - 1.96 * spread, fused + 1.96 * spread)

        alpha = 1.0 - self._confidence_level
        samples = np.array(fused_samples)
        return (
            float(np.percentile(samples, 100 * alpha / 2)),
            float(np.percentile(samples, 100 * (1 - alpha / 2))),
        )

    def _spread_for(self, stream: EvidenceStream) -> float:
        """Per-stream spread, defaulting wide where it was never measured."""
        return self._spreads.get(stream, 1.5)

    def _covariance(self, streams: Sequence[EvidenceStream]) -> np.ndarray:
        """Covariance of the stream errors, with a ridge for conditioning."""
        size = len(streams)
        spreads = np.array([self._spread_for(stream) for stream in streams])
        covariance = np.eye(size)

        for i in range(size):
            for j in range(i + 1, size):
                rho = self._correlation.get(
                    (streams[i], streams[j]),
                    self._correlation.get((streams[j], streams[i]), 0.4),
                )
                covariance[i, j] = covariance[j, i] = float(np.clip(rho, -0.95, 0.95))

        covariance = covariance * np.outer(spreads, spreads)
        return covariance + np.eye(size) * 1e-9
