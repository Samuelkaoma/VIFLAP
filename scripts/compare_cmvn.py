"""How much of the duration effect is the duration, and how much is the front-end?

The largest effect in the H1 sweep is duration: ``C_llr_min`` rises by roughly
0.23 from 30 s to 5 s. That number is confounded, and the confound is in the
front-end rather than in the channel.

Cepstral mean and variance normalisation runs over a *fixed* window of 300
frames. At 30 s a recording carries around 2,650 speech frames, so the window
covers 11% of it and the normalisation is genuinely local. At 5 s it carries
around 450, the window covers 67%, and the operation is most of the way to
utterance-level normalisation — which subtracts a mean and divides by a standard
deviation estimated over the whole utterance, removing between-speaker variance
along with the channel it was meant to remove.

So the sweep varied two things at once. Shortening the recording removed speech,
*and* it changed what the front-end did to what was left. This script separates
them by rerunning the durations against a model whose normalisation window does
not change character with duration, and reporting the difference of the two
duration gaps with an interval.

What is compared, and why it has to be paired twice over
--------------------------------------------------------
Four score vectors: two front-ends at two durations, all on one set of trials.
The quantity of interest is a difference of differences —

    (variant at 5 s - variant at 30 s) - (baseline at 5 s - baseline at 30 s)

— which is negative when the fixed window shrinks the duration gap, that is,
when part of the reported duration effect was the front-end changing behaviour.

Every vector must be scored on the identical trials or the contrast compares
different pairs of recordings. Refusal is what threatens that: the front-end
declines recordings with under three seconds of net speech, and it declines
different ones at 5 s than at 30 s and, in principle, different ones under each
model. Everything is therefore restricted to the recordings **every** model
embedded at **every** duration. The 30 s figures here are consequently computed
over the subset that also survived at 5 s, and differ slightly from the same
cell evaluated on its own — that is the price of a paired contrast and it is
reported rather than absorbed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.corpus import materialise, scan_corpora, split_by_speaker
from scripts.experiment import (
    DegradedRecording,
    TrialSet,
    build_trials,
    degrade_many,
    embed_many,
    worker_count,
)
from viflap.analysis.calibration.metrics import compute_cllr_min, compute_eer
from viflap.analysis.channel.degradation import DegradationCondition, NoiseType
from viflap.analysis.speaker.pipeline import SpeakerComparisonSystem
from viflap.domain.errors import InsufficientDataError
from viflap.evaluation.hypotheses import holm_bonferroni
from viflap.evaluation.splits import (
    bootstrap_contrast_over_speakers,
    bootstrap_over_speakers,
    paired_bootstrap_over_speakers,
    verify_disjoint,
)


@dataclass(slots=True)
class DurationCell:
    """One duration, scored by both front-ends on the shared trial list."""

    duration_seconds: float

    baseline_c_llr_min: float
    baseline_lower: float
    baseline_upper: float
    baseline_eer: float

    variant_c_llr_min: float
    variant_lower: float
    variant_upper: float
    variant_eer: float

    #: variant - baseline. Negative means the fixed window discriminates better.
    difference: float
    difference_lower: float
    difference_upper: float
    difference_p_value: float

    n_refused_baseline: int
    n_refused_variant: int


@dataclass(slots=True)
class DurationGap:
    """The duration effect itself, under each front-end and as a contrast."""

    duration_seconds: float
    reference_seconds: float

    #: C_llr_min(short) - C_llr_min(reference), per front-end. This is the
    #: quantity §5 reported as "+0.23 from 30 s to 5 s".
    baseline_gap: float
    baseline_gap_lower: float
    baseline_gap_upper: float

    variant_gap: float
    variant_gap_lower: float
    variant_gap_upper: float

    #: variant gap minus baseline gap. Negative means part of the duration
    #: effect the baseline reported was the normalisation window changing
    #: character, not the loss of speech.
    contrast: float
    contrast_lower: float
    contrast_upper: float
    contrast_p_value: float
    contrast_excludes_zero: bool

    #: Share of the baseline's duration gap that survives the fixed window.
    #: Undefined and left as None where the baseline gap is not positive.
    fraction_surviving: float | None

    survives_multiplicity: bool = False


def restrict_to_common_survivors(
    sets: Mapping[str, Sequence[tuple[DegradedRecording, object]]],
) -> dict[str, list[tuple[DegradedRecording, object]]]:
    """Cut every set down to the recordings present in all of them, in one order.

    The two-way version in ``compare_capacity`` reconciles one model against
    another at a single duration. Here the pairing has to hold across durations
    as well, because the contrast subtracts a 5 s figure from a 30 s one and a
    recording refused at 5 s would otherwise leave the two computed over
    different populations.

    Order is imposed by sorting on the recording identifier rather than
    inherited from any input, so every returned list pairs index for index.
    """
    if not sets:
        raise InsufficientDataError("no embedding sets were supplied to reconcile")

    shared: set[str] | None = None
    for embedded in sets.values():
        identifiers = {recording.recording_id for recording, _ in embedded}
        shared = identifiers if shared is None else shared & identifiers

    assert shared is not None
    return {
        name: sorted(
            (item for item in embedded if item[0].recording_id in shared),
            key=lambda item: item[0].recording_id,
        )
        for name, embedded in sets.items()
    }


def _check_alignment(trials: Mapping[str, TrialSet]) -> None:
    """Refuse to contrast trial sets that are not the same trials.

    Cheap, and the failure it catches is silent: mismatched trial lists produce
    a number of the right magnitude computed over different pairs of
    recordings.
    """
    reference_name, reference = next(iter(trials.items()))
    for name, candidate in trials.items():
        if (
            candidate.labels.shape != reference.labels.shape
            or not (candidate.labels == reference.labels).all()
        ):
            raise RuntimeError(
                f"trial labels for {name} do not match {reference_name}; the "
                "score vectors are not scored on the same trials and no "
                "contrast between them is meaningful"
            )
        if candidate.speakers != reference.speakers:
            raise RuntimeError(
                f"trial speaker sequence for {name} differs from {reference_name}"
            )


#: A quantity computed from several aligned score vectors and their labels.
Statistic = Callable[[Mapping[str, NDArray[np.float64]], NDArray[np.int64]], float]


def _gap(short: str, long: str) -> Statistic:
    """A statistic returning ``C_llr_min(short) - C_llr_min(long)``."""

    def statistic(
        vectors: Mapping[str, NDArray[np.float64]], labels: NDArray[np.int64]
    ) -> float:
        return float(compute_cllr_min(vectors[short], labels)) - float(
            compute_cllr_min(vectors[long], labels)
        )

    return statistic


def _contrast_of_gaps(
    variant_short: str, variant_long: str, baseline_short: str, baseline_long: str
) -> Statistic:
    """The difference of the two front-ends' duration gaps."""

    def statistic(
        vectors: Mapping[str, NDArray[np.float64]], labels: NDArray[np.int64]
    ) -> float:
        variant = float(compute_cllr_min(vectors[variant_short], labels)) - float(
            compute_cllr_min(vectors[variant_long], labels)
        )
        baseline = float(compute_cllr_min(vectors[baseline_short], labels)) - float(
            compute_cllr_min(vectors[baseline_long], labels)
        )
        return variant - baseline

    return statistic


@dataclass(slots=True)
class Comparison:
    """Everything one condition produced."""

    condition: str
    codec_mode: str
    n_evaluation_speakers: int
    kish_effective_speakers: float
    n_same_source: int
    n_different_source: int
    n_paired_recordings: int
    cells: list[DurationCell] = field(default_factory=list)
    gaps: list[DurationGap] = field(default_factory=list)


def compare(
    condition: DegradationCondition,
    degraded: Sequence[DegradedRecording],
    baseline: SpeakerComparisonSystem,
    variant: SpeakerComparisonSystem,
    baseline_path: Path,
    variant_path: Path,
    durations: Sequence[float],
    *,
    reference_duration: float,
    n_resamples: int,
    workers: int | None,
) -> Comparison:
    """Score every duration with both front-ends and contrast the duration gaps."""
    embedded: dict[str, list[tuple[DegradedRecording, object]]] = {}
    refusals: dict[str, int] = {}

    for duration in durations:
        truncated = [r.truncated(duration) for r in degraded]
        for name, path in (("baseline", baseline_path), ("variant", variant_path)):
            key = f"{name}@{duration:g}"
            results, failures = embed_many(truncated, path, workers=workers)
            embedded[key] = list(results)
            refusals[key] = len(failures)
            print(
                f"    {key:>16}: embedded {len(results)}, refused {len(failures)}",
                flush=True,
            )

    aligned = restrict_to_common_survivors(embedded)
    n_paired = len(next(iter(aligned.values())))
    print(f"    common survivors across all {len(aligned)} sets: {n_paired}", flush=True)

    systems = {"baseline": baseline, "variant": variant}
    trials = {
        key: build_trials(items, systems[key.split("@")[0]], seed=22)  # type: ignore[arg-type]
        for key, items in aligned.items()
    }
    _check_alignment(trials)

    reference_trials = next(iter(trials.values()))
    vectors = {key: trial_set.scores for key, trial_set in trials.items()}
    labels = reference_trials.labels
    speakers = reference_trials.speakers

    cells: list[DurationCell] = []
    for duration in durations:
        base_key, variant_key = f"baseline@{duration:g}", f"variant@{duration:g}"
        base_estimate = bootstrap_over_speakers(
            compute_cllr_min, vectors[base_key], labels, speakers, n_resamples=n_resamples
        )
        variant_estimate = bootstrap_over_speakers(
            compute_cllr_min,
            vectors[variant_key],
            labels,
            speakers,
            n_resamples=n_resamples,
        )
        difference = paired_bootstrap_over_speakers(
            compute_cllr_min,
            vectors[base_key],
            vectors[variant_key],
            labels,
            speakers,
            n_resamples=n_resamples,
        )
        cells.append(
            DurationCell(
                duration_seconds=duration,
                baseline_c_llr_min=base_estimate.value,
                baseline_lower=base_estimate.lower,
                baseline_upper=base_estimate.upper,
                baseline_eer=compute_eer(vectors[base_key], labels),
                variant_c_llr_min=variant_estimate.value,
                variant_lower=variant_estimate.lower,
                variant_upper=variant_estimate.upper,
                variant_eer=compute_eer(vectors[variant_key], labels),
                difference=difference.value,
                difference_lower=difference.lower,
                difference_upper=difference.upper,
                difference_p_value=float(difference.p_value_vs_zero or 1.0),
                n_refused_baseline=refusals[base_key],
                n_refused_variant=refusals[variant_key],
            )
        )
        print(
            f"    {duration:>4g}s  baseline {base_estimate.value:.3f}  "
            f"fixed {variant_estimate.value:.3f}  "
            f"diff {difference.value:+.4f} "
            f"[{difference.lower:+.4f}, {difference.upper:+.4f}]",
            flush=True,
        )

    gaps: list[DurationGap] = []
    for duration in durations:
        if duration == reference_duration:
            continue
        base_short, base_long = f"baseline@{duration:g}", f"baseline@{reference_duration:g}"
        var_short, var_long = f"variant@{duration:g}", f"variant@{reference_duration:g}"

        baseline_gap = bootstrap_contrast_over_speakers(
            _gap(base_short, base_long),
            vectors,
            labels,
            speakers,
            n_resamples=n_resamples,
        )
        variant_gap = bootstrap_contrast_over_speakers(
            _gap(var_short, var_long),
            vectors,
            labels,
            speakers,
            n_resamples=n_resamples,
        )
        contrast = bootstrap_contrast_over_speakers(
            _contrast_of_gaps(var_short, var_long, base_short, base_long),
            vectors,
            labels,
            speakers,
            n_resamples=n_resamples,
            with_p_value=True,
        )
        gaps.append(
            DurationGap(
                duration_seconds=duration,
                reference_seconds=reference_duration,
                baseline_gap=baseline_gap.value,
                baseline_gap_lower=baseline_gap.lower,
                baseline_gap_upper=baseline_gap.upper,
                variant_gap=variant_gap.value,
                variant_gap_lower=variant_gap.lower,
                variant_gap_upper=variant_gap.upper,
                contrast=contrast.value,
                contrast_lower=contrast.lower,
                contrast_upper=contrast.upper,
                contrast_p_value=float(contrast.p_value_vs_zero or 1.0),
                contrast_excludes_zero=contrast.lower > 0.0 or contrast.upper < 0.0,
                fraction_surviving=(
                    variant_gap.value / baseline_gap.value
                    if baseline_gap.value > 0.0
                    else None
                ),
            )
        )
        surviving = gaps[-1].fraction_surviving
        print(
            f"    gap {reference_duration:g}s->{duration:g}s  "
            f"baseline {baseline_gap.value:+.3f}  fixed {variant_gap.value:+.3f}  "
            f"contrast {contrast.value:+.4f} "
            f"[{contrast.lower:+.4f}, {contrast.upper:+.4f}]  "
            f"surviving {'n/a' if surviving is None else f'{surviving * 100:.0f}%'}",
            flush=True,
        )

    codec_modes = {r.codec_mode for r in degraded}
    return Comparison(
        condition=condition.label,
        codec_mode=sorted(codec_modes)[0] if codec_modes else "unknown",
        n_evaluation_speakers=reference_trials.n_speakers,
        kish_effective_speakers=round(reference_trials.kish_effective_sample_size, 2),
        n_same_source=reference_trials.n_same_source,
        n_different_source=reference_trials.n_different_source,
        n_paired_recordings=n_paired,
        cells=cells,
        gaps=gaps,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument("--baseline", type=Path, default=Path("models/acoustic_pooled.npz"))
    parser.add_argument(
        "--variant", type=Path, default=Path("models/acoustic_pooled_cmvn_utt.npz")
    )
    parser.add_argument("--output", type=Path, default=Path("data/reports/h1_cmvn.json"))
    parser.add_argument("--bitrate", type=float, default=12.20)
    parser.add_argument("--noise", default=None, choices=[n.value for n in NoiseType])
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--durations", type=float, nargs="+", default=[30.0, 15.0, 5.0])
    parser.add_argument("--reference-duration", type=float, default=30.0)
    parser.add_argument("--recording-seconds", type=float, default=30.0)
    parser.add_argument("--recordings-per-session", type=int, default=2)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20250601)
    arguments = parser.parse_args(argv)

    if arguments.reference_duration not in arguments.durations:
        parser.error("--reference-duration must be one of --durations")

    baseline = SpeakerComparisonSystem.load(arguments.baseline)
    variant = SpeakerComparisonSystem.load(arguments.variant)
    baseline_window = baseline.config.front_end.sliding_cmvn_frames
    variant_window = variant.config.front_end.sliding_cmvn_frames
    print(f"baseline {baseline.model_id}  CMVN window {baseline_window}", flush=True)
    print(f"variant  {variant.model_id}  CMVN window {variant_window}", flush=True)
    if baseline_window == variant_window:
        parser.error(
            "both models normalise over the same window, so there is no "
            "front-end contrast to measure"
        )

    corpora = arguments.corpus or [Path("data/corpus/librispeech")]
    plans = scan_corpora(
        corpora,
        target_seconds=arguments.recording_seconds,
        max_recordings_per_session=arguments.recordings_per_session,
    )
    split = split_by_speaker(plans)
    print(f"split: {json.dumps(split.summary())}", flush=True)

    # Both models were trained on the same split, so one evaluation partition
    # serves both — but that is checked against what each model recorded rather
    # than assumed from the fact that the same command produced them.
    evaluation_speakers = [r.speaker_id for r in split.evaluation]
    for name, system in (("baseline", baseline), ("variant", variant)):
        if system.training_speakers:
            verify_disjoint(sorted(system.training_speakers), evaluation_speakers)
            print(
                f"  {name}: verified disjoint from its "
                f"{len(system.training_speakers)} training speakers",
                flush=True,
            )
        else:
            print(
                f"  WARNING: the {name} model does not record its training "
                f"speakers, so this evaluation cannot be verified disjoint from "
                f"them",
                flush=True,
            )

    evaluation = materialise(list(split.evaluation))
    print(
        f"materialised {len(evaluation)} evaluation recordings, "
        f"{worker_count(arguments.workers)} workers",
        flush=True,
    )

    condition = DegradationCondition(
        bitrate_kbps=arguments.bitrate,
        noise_type=NoiseType(arguments.noise) if arguments.noise else None,
        snr_db=arguments.snr if arguments.noise else None,
    )
    started = time.monotonic()
    print(f"{condition.label}: degrading {len(evaluation)} recordings", flush=True)
    degraded = degrade_many(
        evaluation, [condition], seed=arguments.seed, workers=arguments.workers
    )
    print(f"    degraded in {time.monotonic() - started:.0f}s", flush=True)

    comparison = compare(
        condition,
        degraded,
        baseline,
        variant,
        arguments.baseline,
        arguments.variant,
        arguments.durations,
        reference_duration=arguments.reference_duration,
        n_resamples=arguments.resamples,
        workers=arguments.workers,
    )

    # Holm-Bonferroni over the duration gaps. Two or three contrasts on one
    # dataset is a family, and reporting them uncorrected reports the chance of
    # a spurious one as a finding.
    decisions = holm_bonferroni(
        {f"{g.duration_seconds:g}s": g.contrast_p_value for g in comparison.gaps}
    )
    for gap in comparison.gaps:
        gap.survives_multiplicity = decisions.get(f"{gap.duration_seconds:g}s", False)

    payload = {
        "baseline_model_id": baseline.model_id,
        "variant_model_id": variant.model_id,
        "baseline_cmvn_frames": baseline_window,
        "variant_cmvn_frames": variant_window,
        "baseline_describe": dict(baseline.describe()),
        "variant_describe": dict(variant.describe()),
        "split": split.summary(),
        "comparison": (
            "variant minus baseline, paired over trials shared by both models at "
            "every duration"
        ),
        "elapsed_minutes": round((time.monotonic() - started) / 60, 1),
        "multiplicity_correction": "holm-bonferroni, family = the duration gaps in this run",
        "condition": asdict(comparison),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
