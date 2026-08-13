"""Stream ablation: what each stream is actually worth.

Evaluates every non-empty subset of the available streams. With five streams
that is thirty-one fusion models, and the exhaustive sweep is the point: the
marginal value of a stream depends on which others are present, so a single
"leave one out" run answers a different and much less useful question.

The quantity that matters is not each stream's solo performance. It is how much
a stream adds *given the others*, which is small precisely when the streams
share a cause — and sharing a cause is the premise of this whole application. A
stream with excellent solo performance and near-zero marginal contribution is a
stream that is repeating what the others already said, and the ablation is what
makes that visible.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from viflap.analysis.calibration.metrics import compute_cllr
from viflap.analysis.fusion.base import FusionModel, FusionObservation, FusionTrainingSet
from viflap.domain.errors import InsufficientDataError
from viflap.domain.evidence import EvidenceStream

__all__ = ["AblationReport", "SubsetResult", "run_ablation"]


@dataclass(frozen=True, slots=True)
class SubsetResult:
    """Performance of one subset of streams."""

    streams: tuple[EvidenceStream, ...]
    c_llr: float
    c_llr_naive: float
    n_comparisons: int

    @property
    def size(self) -> int:
        return len(self.streams)

    @property
    def label(self) -> str:
        return "+".join(stream.value for stream in self.streams)


@dataclass(frozen=True, slots=True)
class AblationReport:
    """Every subset's performance, and what each stream contributes."""

    subsets: tuple[SubsetResult, ...]
    marginal_contributions: Mapping[EvidenceStream, float]
    """Mean reduction in ``C_llr`` from adding this stream to a subset lacking
    it, averaged over every such subset. The Shapley-style average, which is the
    defensible way to attribute credit among interacting contributors: a stream's
    value genuinely depends on which others are present, and any single ordering
    would report the artefact of that ordering."""

    redundancy: Mapping[EvidenceStream, float]
    """One minus the ratio of marginal to solo contribution. Near one means the
    stream is almost entirely redundant given the others — it performs well
    alone and adds nothing to the ensemble, which is the signature of shared
    cause rather than of a weak stream."""

    def best_subset(self) -> SubsetResult:
        return min(self.subsets, key=lambda subset: subset.c_llr)

    def best_single(self) -> SubsetResult:
        singles = [subset for subset in self.subsets if subset.size == 1]
        if not singles:
            raise InsufficientDataError("no single-stream subsets were evaluated")
        return min(singles, key=lambda subset: subset.c_llr)

    def describe(self) -> str:
        best = self.best_subset()
        single = self.best_single()
        lines = [
            f"Best subset: {best.label} at C_llr {best.c_llr:.4f}, against the "
            f"best single stream {single.label} at {single.c_llr:.4f} — an "
            f"improvement of {single.c_llr - best.c_llr:.4f}.",
            "Marginal contribution of each stream, averaged over the subsets it "
            "could be added to:",
        ]
        for stream, value in sorted(
            self.marginal_contributions.items(), key=lambda item: -item[1]
        ):
            lines.append(
                f"  {stream.value:14s} {value:+.4f}  "
                f"(redundancy {self.redundancy.get(stream, float('nan')):.0%})"
            )
        return "\n".join(lines)


def run_ablation(
    training: FusionTrainingSet,
    evaluation: FusionTrainingSet,
    model_factory: Callable[[], FusionModel],
    min_comparisons: int = 50,
) -> AblationReport:
    """Fit and evaluate a fusion model on every non-empty subset of streams.

    A fresh model is fitted per subset. Fitting once on all streams and then
    zeroing the excluded ones would measure something else entirely — the
    remaining weights would still be discounted for redundancy with evidence
    that is no longer present, systematically understating every subset.
    """
    streams = training.streams
    if not streams:
        raise InsufficientDataError("the training set contains no streams")

    results: list[SubsetResult] = []

    for size in range(1, len(streams) + 1):
        for subset in combinations(streams, size):
            subset_set = set(subset)

            train_obs = [
                FusionObservation(
                    log_lrs={s: v for s, v in o.log_lrs.items() if s in subset_set},
                    is_same_source=o.is_same_source,
                    group_id=o.group_id,
                )
                for o in training.observations
                if subset_set & set(o.log_lrs)
            ]
            eval_obs = [
                FusionObservation(
                    log_lrs={s: v for s, v in o.log_lrs.items() if s in subset_set},
                    is_same_source=o.is_same_source,
                    group_id=o.group_id,
                )
                for o in evaluation.observations
                if subset_set & set(o.log_lrs)
            ]
            train_obs = [o for o in train_obs if o.log_lrs]
            eval_obs = [o for o in eval_obs if o.log_lrs]

            if len(train_obs) < min_comparisons or len(eval_obs) < min_comparisons:
                continue

            try:
                model = model_factory().fit(FusionTrainingSet(train_obs))
            except Exception:
                continue

            fused, naive, labels = [], [], []
            for observation in eval_obs:
                if not model.supports_pattern(observation.pattern):
                    continue
                fused.append(model.fuse(observation.log_lrs))
                naive.append(sum(observation.log_lrs.values()))
                labels.append(1 if observation.is_same_source else 0)

            label_array = np.array(labels)
            if label_array.size < min_comparisons or np.unique(label_array).size < 2:
                continue

            results.append(
                SubsetResult(
                    streams=subset,
                    c_llr=compute_cllr(np.array(fused), label_array),
                    c_llr_naive=compute_cllr(np.array(naive), label_array),
                    n_comparisons=int(label_array.size),
                )
            )

    if not results:
        raise InsufficientDataError(
            "no stream subset had enough comparisons to evaluate",
            min_comparisons=min_comparisons,
        )

    by_streams = {result.streams: result for result in results}
    marginal: dict[EvidenceStream, float] = {}
    redundancy: dict[EvidenceStream, float] = {}

    for stream in streams:
        gains: list[float] = []
        for subset, result in by_streams.items():
            if stream in subset:
                continue
            extended = tuple(sorted({*subset, stream}, key=lambda s: s.value))
            extended = tuple(s for s in streams if s in set(extended))
            with_stream = by_streams.get(extended)
            if with_stream is not None:
                gains.append(result.c_llr - with_stream.c_llr)
        marginal[stream] = float(np.mean(gains)) if gains else 0.0

        solo = by_streams.get((stream,))
        if solo is not None and gains:
            # Solo contribution measured against contributing nothing at all,
            # which is C_llr = 1 by definition.
            solo_gain = 1.0 - solo.c_llr
            redundancy[stream] = (
                float(np.clip(1.0 - marginal[stream] / solo_gain, 0.0, 1.0))
                if solo_gain > 1e-9
                else 1.0
            )

    return AblationReport(
        subsets=tuple(sorted(results, key=lambda r: (r.size, r.label))),
        marginal_contributions=marginal,
        redundancy=redundancy,
    )
