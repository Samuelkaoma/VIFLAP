"""Speaker-disjoint splitting and speaker-level resampling.

Two rules govern every number this system reports, and violating either produces
results that are optimistic, confident, and wrong.

**Splits are disjoint by speaker, not by recording.** A speaker appearing in both
training and evaluation lets the model recognise *that speaker* rather than
recognise speakers. Recording-level splitting is the standard way this happens
by accident, and the resulting error rates are often several times too good —
good enough to be believed, which is the problem.

**Resampling is over speakers, not over trials.** Trials sharing a speaker are
not independent: they share a vocal tract, a handset, a recording session. A
bootstrap over trials treats correlated observations as fresh information and
produces confidence intervals that are too narrow, sometimes by a factor of
three. An interval that excludes the truth most of the time is worse than no
interval, because a stated interval is believed.

The second rule extends past the acoustic stream, and that is easy to miss. The
same operator produces the behavioural, temporal and transactional evidence too,
so *every* stream's evaluation resamples over operators. A behavioural
evaluation resampling over transcripts has the same defect for the same reason.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from viflap.domain.errors import InsufficientDataError, InvalidEvidenceError
from viflap.domain.metrics import Estimate

__all__ = [
    "SpeakerDisjointSplit",
    "bootstrap_contrast_over_speakers",
    "bootstrap_over_speakers",
    "make_folds",
    "paired_bootstrap_over_speakers",
    "verify_disjoint",
]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SpeakerDisjointSplit:
    """One train/evaluation split with no speaker in both parts."""

    train_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]
    train_speakers: frozenset[Hashable]
    eval_speakers: frozenset[Hashable]
    fold: int

    def __post_init__(self) -> None:
        overlap = self.train_speakers & self.eval_speakers
        if overlap:
            raise InvalidEvidenceError(
                "speaker leakage between training and evaluation; any metric "
                "computed from this split measures memorisation rather than "
                "generalisation",
                fold=self.fold,
                leaked=sorted(str(speaker) for speaker in overlap)[:10],
                n_leaked=len(overlap),
            )

    @property
    def n_train(self) -> int:
        return len(self.train_indices)

    @property
    def n_eval(self) -> int:
        return len(self.eval_indices)


def make_folds(
    speakers: Sequence[Hashable],
    n_folds: int = 5,
    seed: int = 0,
) -> list[SpeakerDisjointSplit]:
    """Partition items into speaker-disjoint cross-validation folds.

    Speakers are assigned to folds in descending order of how many items they
    contribute, each going to the currently smallest fold. This balances fold
    sizes despite the heavily uneven contribution typical of operational
    material, where a few prolific offenders account for most incidents. Random
    assignment can put several of them in one fold, leaving it dominated by two
    or three people.
    """
    if n_folds < 2:
        raise InvalidEvidenceError("cross-validation needs at least two folds")

    by_speaker: dict[Hashable, list[int]] = defaultdict(list)
    for index, speaker in enumerate(speakers):
        by_speaker[speaker].append(index)

    if len(by_speaker) < n_folds:
        raise InsufficientDataError(
            "fewer speakers than folds; a fold with no speakers cannot be "
            "evaluated, and reducing the fold count is the correct response "
            "rather than allowing a speaker into two folds",
            n_speakers=len(by_speaker),
            n_folds=n_folds,
        )

    # The seed breaks ties among speakers contributing equally many items. The
    # assignment itself is deterministic — largest contributor to the currently
    # smallest fold — so without a tie-breaker the folds would depend on the
    # string ordering of speaker identifiers, which is an arbitrary property of
    # how the corpus happened to be labelled and would make two runs over the
    # same speakers under different identifier schemes produce different folds.
    rng = np.random.default_rng(seed)
    jitter = {speaker: float(rng.random()) for speaker in by_speaker}
    ordered = sorted(by_speaker, key=lambda s: (-len(by_speaker[s]), jitter[s], str(s)))

    fold_of: dict[Hashable, int] = {}
    fold_sizes = [0] * n_folds
    for speaker in ordered:
        target = int(np.argmin(fold_sizes))
        fold_of[speaker] = target
        fold_sizes[target] += len(by_speaker[speaker])

    splits: list[SpeakerDisjointSplit] = []
    for fold in range(n_folds):
        eval_speakers = {s for s, f in fold_of.items() if f == fold}
        train_speakers = set(fold_of) - eval_speakers
        eval_indices = [i for s in eval_speakers for i in by_speaker[s]]
        train_indices = [i for s in train_speakers for i in by_speaker[s]]
        splits.append(
            SpeakerDisjointSplit(
                train_indices=tuple(sorted(train_indices)),
                eval_indices=tuple(sorted(eval_indices)),
                train_speakers=frozenset(train_speakers),
                eval_speakers=frozenset(eval_speakers),
                fold=fold,
            )
        )
    return splits


def verify_disjoint(
    train_speakers: Sequence[Hashable], eval_speakers: Sequence[Hashable]
) -> None:
    """Assert that no speaker appears in both sets.

    Called at the start of every evaluation, not only when constructing splits.
    Leakage most often enters later — a corpus extended, two sources merged, a
    speaker appearing under two identifiers — and the check is cheap enough to
    run every time.
    """
    overlap = set(train_speakers) & set(eval_speakers)
    if overlap:
        raise InvalidEvidenceError(
            "speaker leakage detected",
            n_leaked=len(overlap),
            examples=sorted(str(s) for s in overlap)[:10],
        )


def bootstrap_contrast_over_speakers(
    statistic: Callable[[Mapping[str, NDArray[np.float64]], NDArray[np.int64]], float],
    score_sets: Mapping[str, NDArray[np.float64]],
    labels: NDArray[np.int64],
    speakers: Sequence[Hashable],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
    *,
    with_p_value: bool = False,
) -> Estimate:
    """Interval for any statistic computed from several aligned score vectors.

    The general form of the two functions below, and the one to reach for when
    the quantity of interest is not a metric or a difference of two but a
    contrast of contrasts — for instance, *how much a front-end change moves the
    gap between two durations*, which is a difference of two differences over
    four score vectors.

    Every vector in ``score_sets`` must be scored on the same trials in the same
    order, since one set of ``labels`` and ``speakers`` indexes all of them. The
    caller is responsible for that alignment: a system that refused a different
    subset of recordings produces a different trial list, and the misalignment
    is silent rather than an error.

    ``statistic`` receives the resampled vectors under the same keys and the
    resampled labels, and returns one number. Resamples on which it raises, and
    resamples containing only one class, are discarded and counted rather than
    substituted for.

    ``with_p_value`` attaches a two-sided p-value against zero, derived through
    the same BCa transformation as the interval so that the two cannot disagree.
    It is meaningful only where zero is the null — a difference or a contrast —
    which is why it is off by default rather than always computed.
    """
    prepared = {
        name: np.asarray(vector, dtype=np.float64) for name, vector in score_sets.items()
    }
    if not prepared:
        raise InvalidEvidenceError("no score vectors were supplied to resample")

    labels = np.asarray(labels)
    lengths = {name: int(vector.shape[0]) for name, vector in prepared.items()}
    if set(lengths.values()) != {labels.shape[0]} or labels.shape[0] != len(speakers):
        raise InvalidEvidenceError(
            "every score vector, the labels and the speaker identifiers must be "
            "the same length; a mismatch means the vectors are not scored on the "
            "same trials and no contrast between them is meaningful",
            lengths=lengths,
            n_labels=int(labels.shape[0]),
            n_speakers=len(speakers),
        )

    by_speaker: dict[Hashable, list[int]] = defaultdict(list)
    for index, speaker in enumerate(speakers):
        by_speaker[speaker].append(index)

    unique = sorted(by_speaker, key=str)
    if len(unique) < 3:
        raise InsufficientDataError(
            "resampling over speakers needs at least three speakers; with fewer, "
            "the interval describes those speakers rather than the population",
            n_speakers=len(unique),
        )

    point = float(statistic(prepared, labels))
    rng = np.random.default_rng(seed)
    values: list[float] = []

    for _ in range(n_resamples):
        drawn = rng.choice(len(unique), size=len(unique), replace=True)
        indices = [i for position in drawn for i in by_speaker[unique[position]]]
        if not indices:
            continue
        resampled_labels = labels[indices]
        if np.unique(resampled_labels).size < 2:
            continue
        try:
            values.append(
                float(
                    statistic(
                        {name: vector[indices] for name, vector in prepared.items()},
                        resampled_labels,
                    )
                )
            )
        except Exception:
            continue

    if len(values) < max(30, n_resamples // 20):
        raise InsufficientDataError(
            "too few resamples produced a computable value; the speaker set is "
            "too small or too unbalanced for a meaningful interval",
            n_usable=len(values),
            n_attempted=n_resamples,
        )

    # Delete-one-speaker jackknife, for the acceleration term. One extra
    # evaluation per speaker — negligible beside the resampling, and without it
    # the interval cannot be corrected for the skew that a resubstitution
    # statistic guarantees.
    jackknife: list[float] = []
    for omitted in range(len(unique)):
        indices = [
            i
            for position, speaker in enumerate(unique)
            if position != omitted
            for i in by_speaker[speaker]
        ]
        kept_labels = labels[indices]
        if np.unique(kept_labels).size < 2:
            continue
        try:
            jackknife.append(
                float(
                    statistic(
                        {name: vector[indices] for name, vector in prepared.items()},
                        kept_labels,
                    )
                )
            )
        except Exception:
            continue

    array = np.array(values)
    accelerations = np.array(jackknife)
    lower, upper, method = _bca_bounds(point, array, accelerations, confidence_level)
    return Estimate(
        value=point,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        resampling_unit="speakers",
        n_resamples=len(values),
        interval_method=method,
        n_discarded=n_resamples - len(values),
        p_value_vs_zero=(
            _bca_p_value(array, accelerations, point) if with_p_value else None
        ),
    )


def bootstrap_over_speakers(
    metric: Callable[[NDArray[np.float64], NDArray[np.int64]], float],
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    speakers: Sequence[Hashable],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Bootstrap confidence interval, resampling **speakers** with replacement.

    Each resample draws speakers, then takes *all* of each drawn speaker's
    trials. This preserves the correlation structure within a speaker, which is
    what makes the interval honest: the effective sample size is the number of
    speakers, not the number of trials, and this is the estimator that reflects
    it.

    Resamples yielding only one class are discarded rather than counted, and the
    number surviving is recorded. On small speaker sets a large discard rate is
    itself a finding — it means the estimate rests on a handful of speakers and
    the interval, however computed, is optimistic.
    """
    return bootstrap_contrast_over_speakers(
        lambda vectors, resampled: metric(vectors["scores"], resampled),
        {"scores": scores},
        labels,
        speakers,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )


def _bca_p_value(
    replicates: NDArray[np.float64],
    jackknife: NDArray[np.float64],
    point: float,
) -> float:
    """Two-sided p-value against zero, on the same scale as the BCa interval.

    The obvious p-value — the fraction of replicates on the far side of zero —
    is a *percentile* statement, and pairing it with a BCa interval reports two
    different inference procedures as though they agreed. They need not: a BCa
    interval can exclude zero while the raw proportion exceeds 0.05, which
    happened here and is exactly the internal contradiction a reader is entitled
    to attack.

    So the same correction is applied to the p-value. The BCa map sends a
    nominal probability to an adjusted one; this inverts it, asking what nominal
    level would place its bound exactly at zero. Interval and p-value then agree
    by construction: the interval excludes zero at 95% precisely when this
    returns below 0.05.
    """
    parameters = _bca_parameters(point, replicates, jackknife)
    raw = float(np.mean(replicates < 0.0))
    floor = 1.0 / max(len(replicates), 1)

    if parameters is None or raw <= 0.0 or raw >= 1.0:
        one_sided = min(max(raw, floor), 1.0 - floor)
        return float(min(1.0, 2.0 * min(one_sided, 1.0 - one_sided)))

    z0, acceleration = parameters
    u = float(norm.ppf(min(max(raw, floor), 1.0 - floor)))
    denominator = 1.0 + acceleration * (u - z0)
    if abs(denominator) < 1e-12:
        one_sided = raw
    else:
        w = (u - z0) / denominator
        one_sided = float(norm.cdf(w - z0))

    one_sided = min(max(one_sided, floor), 1.0 - floor)
    return float(min(1.0, 2.0 * min(one_sided, 1.0 - one_sided)))


def _bca_parameters(
    point: float, replicates: NDArray[np.float64], jackknife: NDArray[np.float64]
) -> tuple[float, float] | None:
    """Bias ``z0`` and acceleration ``a``, or ``None`` where undefined."""
    proportion = float(np.mean(replicates < point))
    if proportion <= 0.0 or proportion >= 1.0 or jackknife.size == 0:
        return None

    z0 = float(norm.ppf(proportion))
    centred = jackknife.mean() - jackknife
    denominator = float(np.sum(centred**2)) ** 1.5
    acceleration = (
        0.0 if denominator <= 0.0 else float(np.sum(centred**3) / (6.0 * denominator))
    )
    return z0, acceleration


def _bca_bounds(
    point: float,
    replicates: NDArray[np.float64],
    jackknife: NDArray[np.float64],
    confidence_level: float,
) -> tuple[float, float, str]:
    """Bias-corrected and accelerated interval bounds, with a stated fallback.

    The percentile method assumes the bootstrap distribution is centred on the
    estimate and symmetric. Neither holds here. ``C_llr_min`` fits its PAV
    transform on the very trials it scores, so it is a resubstitution minimum
    and optimistically biased; a resample contains about 63% of the distinct
    speakers, so the bias is *larger* in the replicates than in the point
    estimate and the whole distribution sits low. Reading percentiles off it
    produces an interval shifted toward zero — which, for a rule that declares
    falsification when a lower bound clears a threshold, is the direction that
    withdraws findings that should stand.

    Two corrections, both from the replicate distribution itself:

    ``z0`` (bias) is the normal quantile of the fraction of replicates below the
    point estimate. At perfect symmetry that fraction is a half and ``z0`` is
    zero, so the correction costs nothing when it is not needed.

    ``a`` (acceleration) comes from the skew of the delete-one-speaker jackknife
    and handles the variance changing with the estimate's level, which it does
    here — a resample of easy speakers has both a lower value and a tighter
    spread.

    Falls back to percentile bounds, and says so in the returned label, when
    ``z0`` is not finite. That happens when every replicate falls on one side of
    the point estimate, where the correction is undefined rather than merely
    large.
    """
    alpha = 1.0 - confidence_level
    parameters = _bca_parameters(point, replicates, jackknife)
    if parameters is None:
        return (
            float(np.percentile(replicates, 100 * alpha / 2)),
            float(np.percentile(replicates, 100 * (1 - alpha / 2))),
            "percentile (BCa undefined: every replicate on one side)",
        )

    z0, acceleration = parameters

    def adjusted(probability: float) -> float:
        z = float(norm.ppf(probability))
        shifted = z0 + z
        divisor = 1.0 - acceleration * shifted
        if abs(divisor) < 1e-12:
            return probability
        return float(norm.cdf(z0 + shifted / divisor))

    lower_probability = adjusted(alpha / 2)
    upper_probability = adjusted(1 - alpha / 2)
    if not (0.0 < lower_probability < upper_probability < 1.0):
        return (
            float(np.percentile(replicates, 100 * alpha / 2)),
            float(np.percentile(replicates, 100 * (1 - alpha / 2))),
            "percentile (BCa adjustment left the unit interval)",
        )

    return (
        float(np.percentile(replicates, 100 * lower_probability)),
        float(np.percentile(replicates, 100 * upper_probability)),
        "bca",
    )


def paired_bootstrap_over_speakers(
    metric: Callable[[NDArray[np.float64], NDArray[np.int64]], float],
    baseline_scores: NDArray[np.float64],
    variant_scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    speakers: Sequence[Hashable],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Interval for ``metric(variant) - metric(baseline)`` on the *same* trials.

    Two systems compared by their separate intervals are compared far more
    weakly than the data allow. Those intervals are wide because speakers differ
    from one another, and both systems saw the same speakers — so most of that
    width is shared and cancels in the difference. Overlapping marginal
    intervals therefore do not mean the systems are indistinguishable, and
    reading them that way is how a real difference gets reported as a null.

    Each resample draws speakers once and evaluates **both** score vectors on
    the identical trials, which is what makes the difference paired. The two
    vectors must be aligned trial for trial; the caller is responsible for
    restricting both systems to the recordings both were willing to embed,
    because a system that refuses a different subset produces a different trial
    list and the pairing would be silently broken.

    A negative value means the variant scores better than the baseline, for a
    metric like ``C_llr_min`` where lower is better.

    The p-value it carries is floored at 1/B rather than allowed to reach zero:
    with B resamples the smallest observable p-value is 1/B, and reporting zero
    would claim a precision the resampling cannot deliver. It is carried at all
    so a family of these comparisons can be corrected for multiplicity —
    reporting only "the interval excludes zero" leaves nothing for a correction
    to consume, and the correction then silently never happens.
    """
    return bootstrap_contrast_over_speakers(
        lambda vectors, resampled: (
            metric(vectors["variant"], resampled) - metric(vectors["baseline"], resampled)
        ),
        {"baseline": baseline_scores, "variant": variant_scores},
        labels,
        speakers,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        with_p_value=True,
    )
