"""Does more model capacity recover any of the information the channel leaves?

The sweep in ``data/reports/h1_sweep.json`` left one explanation standing. H1 is
not supported anywhere, calibration was ruled out as the constraint, and the
remaining question is whether ``C_llr_min`` of roughly 0.34 is the *channel*
having destroyed the speaker information or the *model* having been too small to
find what survived. This script answers that by scoring two models — the 128
component / rank 100 baseline and a 256 / 200 variant — on identical trials.

Why the comparison is paired
----------------------------
Comparing the two models by their separate bootstrap intervals wastes most of
the data. Those intervals are wide because 42 speakers differ from one another,
and both models saw the same 42 speakers, so nearly all of that width is common
to the two and cancels in a difference. Marginal intervals that overlap would
therefore say almost nothing, and reporting that overlap as "no difference"
would be a null manufactured by the choice of estimator.

Each cell here degrades the audio **once** and embeds that same audio with both
models, so the difference is bootstrapped over speakers on paired trials.

Refusals have to be reconciled before pairing
---------------------------------------------
The two models do not necessarily refuse the same recordings — refusal depends
on the front-end's voice-activity decision and on the component count through
the effective-occupancy check. If each model's trials were built from its own
survivors, the two trial lists would differ and the pairing would be broken
silently, in the direction that flatters whichever model refused more of the
hard recordings. Both models are therefore restricted to the recordings **both**
embedded, and the per-model refusals are reported so the restriction is visible.

What a null result here would and would not show
------------------------------------------------
If the difference interval covers zero, that shows this *corpus* does not
support a larger model — 125 training speakers caps PLDA quality whatever the
component count. It does not show the channel is responsible. Separating those
needs more speakers, not more parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from scripts.corpus import materialise, scan_corpora, split_by_speaker
from scripts.experiment import (
    DegradedRecording,
    build_trials,
    degrade_many,
    embed_many,
    worker_count,
)
from viflap.analysis.calibration.metrics import compute_cllr_min
from viflap.analysis.channel.degradation import DegradationCondition, NoiseType
from viflap.analysis.speaker.pipeline import SpeakerComparisonSystem
from viflap.domain.errors import InsufficientDataError
from viflap.evaluation.hypotheses import holm_bonferroni
from viflap.evaluation.splits import (
    bootstrap_over_speakers,
    paired_bootstrap_over_speakers,
    verify_disjoint,
)


@dataclass(slots=True)
class CapacityCell:
    """One condition at one duration, scored by both models on shared trials."""

    condition: str
    duration_seconds: float
    codec_mode: str

    n_evaluation_speakers: int
    kish_effective_speakers: float
    n_same_source: int
    n_different_source: int

    n_refused_baseline: int
    n_refused_variant: int
    n_paired_recordings: int

    baseline_c_llr_min: float
    baseline_lower: float
    baseline_upper: float

    variant_c_llr_min: float
    variant_lower: float
    variant_upper: float

    #: variant - baseline. Negative means the larger model discriminates better.
    difference: float
    difference_lower: float
    difference_upper: float
    difference_excludes_zero: bool

    #: Two-sided bootstrap p-value against no difference, and the verdict
    #: after Holm-Bonferroni over this run's family of cells. Both are kept:
    #: the raw verdict is what the interval shows, the adjusted one is what
    #: survives having tested several cells on one dataset.
    p_value: float

    notes: list[str]

    survives_multiplicity: bool = False


#: The embedding type is left free so the reconciliation can be tested with a
#: stub, and so mypy still checks that whatever goes in reaches ``build_trials``
#: unchanged rather than being widened to ``object`` on the way through.
_Embedding = TypeVar("_Embedding")


def _common_survivors(
    embedded_a: Sequence[tuple[DegradedRecording, _Embedding]],
    embedded_b: Sequence[tuple[DegradedRecording, _Embedding]],
) -> tuple[
    list[tuple[DegradedRecording, _Embedding]],
    list[tuple[DegradedRecording, _Embedding]],
]:
    """Restrict both models to the recordings both embedded, in one order.

    Order is fixed by sorting on the recording identifier rather than inherited
    from either model's result list, so the two trial lists are built from the
    same sequence and pair index for index.
    """
    shared = {r.recording_id for r, _ in embedded_a} & {
        r.recording_id for r, _ in embedded_b
    }
    keep_a = sorted(
        (item for item in embedded_a if item[0].recording_id in shared),
        key=lambda item: item[0].recording_id,
    )
    keep_b = sorted(
        (item for item in embedded_b if item[0].recording_id in shared),
        key=lambda item: item[0].recording_id,
    )
    return keep_a, keep_b


def evaluate_cell(
    condition: DegradationCondition,
    duration: float,
    degraded: Sequence[DegradedRecording],
    baseline: SpeakerComparisonSystem,
    variant: SpeakerComparisonSystem,
    baseline_path: Path,
    variant_path: Path,
    *,
    n_resamples: int,
    workers: int | None,
) -> CapacityCell | None:
    """Score one cell with both models and bootstrap their difference."""
    notes: list[str] = []
    truncated = [r.truncated(duration) for r in degraded]

    embedded_baseline, failed_baseline = embed_many(
        truncated, baseline_path, workers=workers
    )
    embedded_variant, failed_variant = embed_many(truncated, variant_path, workers=workers)

    keep_baseline, keep_variant = _common_survivors(embedded_baseline, embedded_variant)
    if len(keep_baseline) != len(embedded_baseline) or len(keep_variant) != len(
        embedded_variant
    ):
        notes.append(
            "the two models refused different recordings; both were restricted to "
            f"the {len(keep_baseline)} embedded by both, so the trials pair"
        )

    codec_modes = {r.codec_mode for r in degraded}
    codec_mode = sorted(codec_modes)[0] if codec_modes else "unknown"

    try:
        baseline_trials = build_trials(keep_baseline, baseline, seed=22)
        variant_trials = build_trials(keep_variant, variant, seed=22)
    except InsufficientDataError as exc:
        print(f"    {duration:>4g}s  not evaluable: {exc.message}", flush=True)
        return None

    # The pairing is an assumption everything downstream rests on, so it is
    # checked rather than trusted: same trials, same order, same labels.
    if (
        baseline_trials.labels.shape != variant_trials.labels.shape
        or not (baseline_trials.labels == variant_trials.labels).all()
    ):
        raise RuntimeError(
            f"{condition.label} at {duration:g}s: the two models produced "
            "non-corresponding trial lists, so a paired difference would compare "
            "different pairs of recordings"
        )
    if baseline_trials.speakers != variant_trials.speakers:
        raise RuntimeError(
            f"{condition.label} at {duration:g}s: trial speaker sequences differ "
            "between the two models despite identical inputs"
        )

    try:
        baseline_estimate = bootstrap_over_speakers(
            compute_cllr_min,
            baseline_trials.scores,
            baseline_trials.labels,
            baseline_trials.speakers,
            n_resamples=n_resamples,
        )
        variant_estimate = bootstrap_over_speakers(
            compute_cllr_min,
            variant_trials.scores,
            variant_trials.labels,
            variant_trials.speakers,
            n_resamples=n_resamples,
        )
        difference = paired_bootstrap_over_speakers(
            compute_cllr_min,
            baseline_trials.scores,
            variant_trials.scores,
            baseline_trials.labels,
            baseline_trials.speakers,
            n_resamples=n_resamples,
        )
    except InsufficientDataError as exc:
        print(f"    {duration:>4g}s  not evaluable: {exc.message}", flush=True)
        return None

    excludes_zero = difference.lower > 0.0 or difference.upper < 0.0
    cell = CapacityCell(
        condition=condition.label,
        duration_seconds=duration,
        codec_mode=codec_mode,
        n_evaluation_speakers=baseline_trials.n_speakers,
        kish_effective_speakers=round(baseline_trials.kish_effective_sample_size, 2),
        n_same_source=baseline_trials.n_same_source,
        n_different_source=baseline_trials.n_different_source,
        n_refused_baseline=len(failed_baseline),
        n_refused_variant=len(failed_variant),
        n_paired_recordings=len(keep_baseline),
        baseline_c_llr_min=baseline_estimate.value,
        baseline_lower=baseline_estimate.lower,
        baseline_upper=baseline_estimate.upper,
        variant_c_llr_min=variant_estimate.value,
        variant_lower=variant_estimate.lower,
        variant_upper=variant_estimate.upper,
        difference=difference.value,
        difference_lower=difference.lower,
        difference_upper=difference.upper,
        difference_excludes_zero=excludes_zero,
        p_value=float(difference.p_value_vs_zero or 1.0),
        notes=notes,
    )
    print(
        f"    {duration:>4g}s  base {cell.baseline_c_llr_min:.3f}  "
        f"large {cell.variant_c_llr_min:.3f}  "
        f"diff {cell.difference:+.4f} "
        f"[{cell.difference_lower:+.4f}, {cell.difference_upper:+.4f}]  "
        f"{'DIFFERENT' if excludes_zero else 'covers zero'}",
        flush=True,
    )
    return cell


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument(
        "--evaluation-speakers",
        type=Path,
        default=None,
        help=(
            "file of speaker identifiers, one per line, to evaluate on. Use "
            "when the two models were trained on different splits: each "
            "model's own held-out speakers may sit inside the other's "
            "training set, and only a set held out by both is a fair test."
        ),
    )
    parser.add_argument("--baseline", type=Path, default=Path("models/acoustic.npz"))
    parser.add_argument("--variant", type=Path, default=Path("models/acoustic_large.npz"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/h1_capacity_paired.json")
    )
    parser.add_argument("--bitrates", type=float, nargs="+", default=[12.20])
    parser.add_argument(
        "--noise", nargs="+", default=[], choices=[n.value for n in NoiseType]
    )
    parser.add_argument("--snrs", type=float, nargs="+", default=[20.0])
    parser.add_argument("--durations", type=float, nargs="+", default=[30.0, 15.0, 5.0])
    parser.add_argument("--recording-seconds", type=float, default=30.0)
    parser.add_argument("--recordings-per-session", type=int, default=2)
    parser.add_argument("--resamples", type=int, default=300)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20250601)
    arguments = parser.parse_args(argv)

    baseline = SpeakerComparisonSystem.load(arguments.baseline)
    variant = SpeakerComparisonSystem.load(arguments.variant)
    print(f"baseline {baseline.model_id}", flush=True)
    print(f"variant  {variant.model_id}", flush=True)

    corpora = arguments.corpus or [Path("data/corpus/librispeech")]
    plans = scan_corpora(
        corpora,
        target_seconds=arguments.recording_seconds,
        max_recordings_per_session=arguments.recordings_per_session,
    )

    if arguments.evaluation_speakers is not None:
        # Two models trained on different splits have different held-out sets,
        # and each one's own evaluation speakers are liable to sit inside the
        # other's training set. Scoring both on either model's split would
        # reward whichever had seen the speakers. The caller supplies the set
        # held out by both, and it is recorded in the report so the comparison
        # can be audited rather than trusted.
        wanted = {
            line.strip()
            for line in arguments.evaluation_speakers.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }
        evaluation_plans = [plan for plan in plans if plan.speaker_id in wanted]
        found = {plan.speaker_id for plan in evaluation_plans}
        missing = wanted - found
        if missing:
            raise InsufficientDataError(
                "evaluation speakers were requested that the corpus does not contain",
                n_missing=len(missing),
                examples=sorted(missing)[:5],
            )
        split_summary = {
            "evaluation_recordings": len(evaluation_plans),
            "evaluation_speakers": len(found),
            "source": str(arguments.evaluation_speakers),
        }
    else:
        split = split_by_speaker(plans)
        verify_disjoint(
            [r.speaker_id for r in split.train], [r.speaker_id for r in split.evaluation]
        )
        evaluation_plans = list(split.evaluation)
        split_summary = split.summary()

    print(f"split: {json.dumps(split_summary)}", flush=True)

    evaluation = materialise(evaluation_plans)
    print(
        f"materialised {len(evaluation)} evaluation recordings, "
        f"{worker_count(arguments.workers)} workers",
        flush=True,
    )

    conditions: list[DegradationCondition] = []
    for bitrate in arguments.bitrates:
        conditions.append(DegradationCondition(bitrate_kbps=bitrate))
        for noise in arguments.noise:
            for snr in arguments.snrs:
                conditions.append(
                    DegradationCondition(
                        bitrate_kbps=bitrate, noise_type=NoiseType(noise), snr_db=snr
                    )
                )

    started = time.monotonic()
    cells: list[CapacityCell] = []
    for index, condition in enumerate(conditions, start=1):
        print(f"[{index}/{len(conditions)}] {condition.label}: degrading", flush=True)
        degraded = degrade_many(
            evaluation, [condition], seed=arguments.seed, workers=arguments.workers
        )
        for duration in arguments.durations:
            cell = evaluate_cell(
                condition,
                duration,
                degraded,
                baseline,
                variant,
                arguments.baseline,
                arguments.variant,
                n_resamples=arguments.resamples,
                workers=arguments.workers,
            )
            if cell is not None:
                cells.append(cell)

    # Holm-Bonferroni over this run's cells. Six comparisons on one dataset
    # is a family, and reporting six uncorrected verdicts reports the chance
    # of a spurious one as a finding. Both verdicts are kept: the raw one is
    # what the interval shows, the adjusted one is what survives the family.
    decisions = holm_bonferroni(
        {f"{c.condition}@{c.duration_seconds:g}": c.p_value for c in cells}
    )
    for cell in cells:
        cell.survives_multiplicity = decisions.get(
            f"{cell.condition}@{cell.duration_seconds:g}", False
        )
    surviving = sum(1 for c in cells if c.survives_multiplicity)
    raw = sum(1 for c in cells if c.difference_excludes_zero)
    print(
        f"\nHolm-Bonferroni over {len(cells)} cells: {raw} raw, {surviving} survive",
        flush=True,
    )

    payload = {
        "baseline_model_id": baseline.model_id,
        "variant_model_id": variant.model_id,
        "baseline_describe": dict(baseline.describe()),
        "variant_describe": dict(variant.describe()),
        "split": split_summary,
        "comparison": "variant minus baseline, paired over shared trials",
        "elapsed_minutes": round((time.monotonic() - started) / 60, 1),
        "n_cells_differing": sum(1 for c in cells if c.difference_excludes_zero),
        "n_cells_surviving_multiplicity": sum(1 for c in cells if c.survives_multiplicity),
        "multiplicity_correction": "holm-bonferroni, family = the cells in this run",
        "cells": [asdict(c) for c in cells],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
