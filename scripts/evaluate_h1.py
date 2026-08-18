"""H1: what speaker-discriminative information survives the telephony channel.

Runs the degradation sweep — bitrate x noise type x SNR x duration — and reports
``C_llr``, ``C_llr_min`` and calibration loss for every cell, each with a
bootstrap interval resampled over **speakers**.

The decision rule is not chosen here. It lives in
:class:`viflap.evaluation.hypotheses.H1ChannelViability`, was fixed before the
experiment, and is applied to the interval rather than the point estimate.

The three quantities and why all three are reported
---------------------------------------------------
``C_llr_min`` is discrimination: the cost that would remain if the scores were
optimally recalibrated. It is invariant to any monotonic transformation, so it
measures whether the system's *ordering* of trials carries information. H1 is
decided on this, because a system that orders well but reports badly is a
calibration problem, and calibration is fixable.

``C_llr`` is what a reported likelihood ratio would actually cost. It is the
number that matters operationally, and it is always at least ``C_llr_min``.

Their difference is calibration loss. Reporting it separately is what keeps a
bad result diagnosable: a large ``C_llr`` with a small ``C_llr_min`` means the
mapping is wrong, and a large ``C_llr_min`` means the evidence is not there.

Two calibrations, deliberately
------------------------------
Each cell is calibrated twice. Once **matched** — a calibrator fitted on
development speakers under that same condition — which is the optimistic case
and the one H1's calibration loss refers to. Once **transferred** — a single
calibrator fitted under the cleanest condition and applied everywhere — which is
what happens in service, where the channel a call arrived over is not known in
advance and no condition-specific development set exists for it.

The gap between the two is the cost of not knowing the channel, and it is
reported because a system deployed on the strength of the matched figure alone
would be deployed on a number nobody will ever see again.

Speakers are disjoint three ways
--------------------------------
Training, development (calibration) and evaluation partitions share no speaker.
The split is reconstructed from the same corpus with the same seed the training
script used, and re-verified here rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.corpus import Recording, materialise, scan_corpora, split_by_speaker
from scripts.experiment import (
    DegradedRecording,
    TrialSet,
    build_trials,
    degrade_many,
    embed_many,
    summarise_codec_modes,
    worker_count,
)
from viflap.analysis.calibration.calibrators import LogisticCalibrator, as_reported
from viflap.analysis.calibration.metrics import compute_cllr, compute_eer
from viflap.analysis.channel.degradation import DegradationCondition, NoiseType
from viflap.analysis.speaker.pipeline import SpeakerComparisonSystem
from viflap.domain.errors import CalibrationError, InsufficientDataError
from viflap.domain.metrics import Estimate
from viflap.evaluation.hypotheses import H1ChannelViability
from viflap.evaluation.splits import bootstrap_over_speakers, verify_disjoint

#: The reference condition whose calibrator is transferred to every other cell.
#: The highest rate with no added noise — the most favourable channel in the
#: design, and therefore the one a development set would most likely have been
#: collected under.
REFERENCE_CONDITION = DegradationCondition(bitrate_kbps=12.20)


def build_sweep(
    bitrates: Sequence[float],
    noise_types: Sequence[NoiseType],
    snrs_db: Sequence[float],
) -> list[DegradationCondition]:
    """Enumerate the channel conditions, clean first at each rate."""
    conditions: list[DegradationCondition] = []
    for bitrate in bitrates:
        conditions.append(DegradationCondition(bitrate_kbps=bitrate))
        for noise_type in noise_types:
            for snr in snrs_db:
                conditions.append(
                    DegradationCondition(
                        bitrate_kbps=bitrate, noise_type=noise_type, snr_db=snr
                    )
                )
    return conditions


@dataclass(slots=True)
class CellResult:
    """One cell of the sweep: a channel condition at one duration."""

    condition: str
    bitrate_kbps: float
    noise_type: str | None
    snr_db: float | None
    duration_seconds: float
    codec_mode: str

    n_evaluation_speakers: int
    kish_effective_speakers: float
    """Effective resampling units, given how unevenly speakers own trials.

    Reported beside the nominal count because the bootstrap treats speakers as
    exchangeable and they are not exactly so: a cell whose 42 speakers behave
    like 31 has a wider true interval than its count suggests, and nothing else
    in the output shows it."""

    n_same_source: int
    n_different_source: int
    n_refused: int
    refusal_rate: float

    c_llr_min: float
    c_llr_min_lower: float
    c_llr_min_upper: float

    c_llr_matched: float | None
    c_llr_matched_lower: float | None
    c_llr_matched_upper: float | None
    calibration_loss_matched: float | None

    c_llr_transferred: float | None
    calibration_loss_transferred: float | None

    # The same two quantities before empirical bounding. Recorded for diagnosis
    # only: they describe a mapping the system never emits. Reporting them as
    # C_llr overstates the loss substantially, which is a mistake this schema
    # now makes visible rather than possible.
    c_llr_matched_unbounded: float | None
    c_llr_transferred_unbounded: float | None

    equal_error_rate: float
    h1_supported: bool
    h1_falsified: bool

    notes: list[str] = field(default_factory=list)


def _fit_calibrator(trials: TrialSet) -> LogisticCalibrator | None:
    """Fit a calibrator on development trials, or report that it could not be."""
    try:
        return LogisticCalibrator().fit(trials.scores, trials.labels)
    except (CalibrationError, InsufficientDataError):
        return None


def _bootstrap(
    metric: Any,
    scores: NDArray[np.float64],
    trials: TrialSet,
    n_resamples: int,
) -> Estimate | None:
    try:
        return bootstrap_over_speakers(
            metric, scores, trials.labels, trials.speakers, n_resamples=n_resamples
        )
    except InsufficientDataError:
        return None


def evaluate_cell(
    condition: DegradationCondition,
    duration: float,
    development: Sequence[DegradedRecording],
    evaluation: Sequence[DegradedRecording],
    system: SpeakerComparisonSystem,
    model_path: Path,
    transferred: LogisticCalibrator | None,
    *,
    n_resamples: int,
    workers: int | None,
    hypothesis: H1ChannelViability,
    scores_sink: dict[str, TrialSet] | None = None,
) -> tuple[CellResult, TrialSet | None]:
    """Embed, score, calibrate and summarise one cell.

    Returns the cell result and the development trial set, the latter so that
    the reference condition's trials can be reused to fit the transferred
    calibrator.

    ``scores_sink``, when supplied, receives this cell's **evaluation** trials
    keyed by ``{condition}@{duration:g}``. It is a sink rather than a third
    return value because the return tuple already means something else and a
    third element would be silently ignored by both existing callers. Persisting
    these is what makes a paired comparison against another system possible at
    all: this script's expensive half is degrading and embedding the corpus, and
    without the scores on disk any pairing costs a full rerun.
    """
    notes: list[str] = []
    truncated_dev = [r.truncated(duration) for r in development]
    truncated_eval = [r.truncated(duration) for r in evaluation]

    # Both partitions embed in one pool. Spawning a pool costs a numpy import in
    # every worker, and the sweep runs this once per cell — two pools per cell
    # doubles a fixed cost that is already comparable to the work itself.
    # Splitting afterwards on the identifiers keeps the partitions separate.
    development_ids = {r.recording_id for r in truncated_dev}
    embedded, failures = embed_many(
        [*truncated_dev, *truncated_eval], model_path, workers=workers
    )
    dev_embedded = [item for item in embedded if item[0].recording_id in development_ids]
    eval_embedded = [
        item for item in embedded if item[0].recording_id not in development_ids
    ]

    n_refused = sum(1 for f in failures if f.recording_id not in development_ids)
    refusal_rate = n_refused / max(len(truncated_eval), 1)

    codec_modes = summarise_codec_modes(list(development) + list(evaluation))
    codec_mode = codec_modes["codec_modes"][0] if codec_modes["codec_modes"] else "unknown"

    def unevaluable(reason: str) -> tuple[CellResult, None]:
        notes.append(reason)
        return (
            CellResult(
                condition=condition.label,
                bitrate_kbps=condition.bitrate_kbps,
                noise_type=condition.noise_type.value if condition.noise_type else None,
                snr_db=condition.snr_db,
                duration_seconds=duration,
                codec_mode=codec_mode,
                n_evaluation_speakers=len({r.speaker_id for r, _ in eval_embedded}),
                kish_effective_speakers=float("nan"),
                n_same_source=0,
                n_different_source=0,
                n_refused=n_refused,
                refusal_rate=refusal_rate,
                c_llr_min=float("nan"),
                c_llr_min_lower=float("nan"),
                c_llr_min_upper=float("nan"),
                c_llr_matched=None,
                c_llr_matched_lower=None,
                c_llr_matched_upper=None,
                calibration_loss_matched=None,
                c_llr_transferred=None,
                calibration_loss_transferred=None,
                c_llr_matched_unbounded=None,
                c_llr_transferred_unbounded=None,
                equal_error_rate=float("nan"),
                h1_supported=False,
                h1_falsified=False,
                notes=notes,
            ),
            None,
        )

    # Every way this cell can fail to be measurable is caught here, and all of
    # them are the same failure: refusal removed so much of the evaluation set
    # that no honest interval can be computed over what remains.
    #
    # The count that matters is **speakers**, not recordings. The bootstrap
    # resamples speakers, so five surviving recordings from two speakers is not
    # a usable cell however many trials they generate — an interval over two
    # speakers describes those two people. Guarding on recordings, as this did
    # first, passes that case straight into the resampler.
    try:
        dev_trials = build_trials(dev_embedded, system, seed=11)
        eval_trials = build_trials(eval_embedded, system, seed=22)

        verify_disjoint(
            [r.speaker_id for r, _ in dev_embedded],
            [r.speaker_id for r, _ in eval_embedded],
        )

        if scores_sink is not None:
            scores_sink[f"{condition.label}@{duration:g}"] = eval_trials

        # Discrimination and the H1 decision, both on the raw scores. C_llr_min
        # is invariant to monotonic recalibration, so calibrating first would
        # change nothing and would only invite the reader to think it had.
        outcome = hypothesis.run(
            eval_trials.scores,
            eval_trials.labels,
            eval_trials.speakers,
            condition=f"{condition.label} @ {duration:g}s",
            n_resamples=n_resamples,
        )
    except InsufficientDataError as exc:
        return unevaluable(
            f"the cell could not be evaluated after refusal: {exc.message} "
            f"({n_refused} of {len(truncated_eval)} evaluation recordings were "
            f"refused by the front-end)"
        )

    c_llr_min_point = outcome.estimate.value

    matched = _fit_calibrator(dev_trials)
    c_llr_matched = matched_lower = matched_upper = None
    calibration_loss_matched = None
    c_llr_matched_unbounded = None
    if matched is None:
        notes.append(
            "no calibrator could be fitted on the development speakers for this "
            "cell; C_llr is not reported, because an uncalibrated score presented "
            "as a likelihood ratio would carry a strength it does not have"
        )
    else:
        calibrated = as_reported(matched, eval_trials.scores)
        estimate = _bootstrap(compute_cllr, calibrated, eval_trials, n_resamples)
        if estimate is not None:
            c_llr_matched = estimate.value
            matched_lower, matched_upper = estimate.lower, estimate.upper
            calibration_loss_matched = estimate.value - c_llr_min_point
        c_llr_matched_unbounded = compute_cllr(
            matched.transform(eval_trials.scores), eval_trials.labels
        )

    c_llr_transferred = calibration_loss_transferred = None
    c_llr_transferred_unbounded = None
    if transferred is not None:
        c_llr_transferred = compute_cllr(
            as_reported(transferred, eval_trials.scores), eval_trials.labels
        )
        calibration_loss_transferred = c_llr_transferred - c_llr_min_point
        c_llr_transferred_unbounded = compute_cllr(
            transferred.transform(eval_trials.scores), eval_trials.labels
        )

    return (
        CellResult(
            condition=condition.label,
            bitrate_kbps=condition.bitrate_kbps,
            noise_type=condition.noise_type.value if condition.noise_type else None,
            snr_db=condition.snr_db,
            duration_seconds=duration,
            codec_mode=codec_mode,
            n_evaluation_speakers=eval_trials.n_speakers,
            kish_effective_speakers=round(eval_trials.kish_effective_sample_size, 2),
            n_same_source=eval_trials.n_same_source,
            n_different_source=eval_trials.n_different_source,
            n_refused=n_refused,
            refusal_rate=round(refusal_rate, 4),
            c_llr_min=c_llr_min_point,
            c_llr_min_lower=outcome.estimate.lower,
            c_llr_min_upper=outcome.estimate.upper,
            c_llr_matched=c_llr_matched,
            c_llr_matched_lower=matched_lower,
            c_llr_matched_upper=matched_upper,
            calibration_loss_matched=calibration_loss_matched,
            c_llr_transferred=c_llr_transferred,
            calibration_loss_transferred=calibration_loss_transferred,
            c_llr_matched_unbounded=c_llr_matched_unbounded,
            c_llr_transferred_unbounded=c_llr_transferred_unbounded,
            equal_error_rate=compute_eer(eval_trials.scores, eval_trials.labels),
            h1_supported=outcome.supported,
            h1_falsified=outcome.falsified,
            notes=notes,
        ),
        dev_trials,
    )


def run_sweep(
    development: Sequence[Recording],
    evaluation: Sequence[Recording],
    system: SpeakerComparisonSystem,
    model_path: Path,
    conditions: Sequence[DegradationCondition],
    durations: Sequence[float],
    *,
    n_resamples: int,
    workers: int | None,
    seed: int,
    checkpoint: Callable[[list[CellResult]], None] | None = None,
    scores_sink: dict[str, TrialSet] | None = None,
) -> list[CellResult]:
    """Degrade once per condition, then evaluate every duration through it.

    ``checkpoint`` is invoked with the results so far after each condition. A
    sweep is hours long and the expensive part — degrading several hundred
    recordings through a sequential coder — cannot be recovered by rerunning
    only the tail. Writing the partial result after every condition means a
    failure in the twenty-eighth cell costs that cell rather than everything
    before it.
    """
    hypothesis = H1ChannelViability()
    results: list[CellResult] = []
    transferred: LogisticCalibrator | None = None

    # The reference condition runs first so its development trials are available
    # to fit the transferred calibrator for every later cell.
    ordered = sorted(
        conditions, key=lambda c: (c.label != REFERENCE_CONDITION.label, c.label)
    )

    for index, condition in enumerate(ordered, start=1):
        started = time.monotonic()
        print(
            f"[{index}/{len(ordered)}] {condition.label}: degrading "
            f"{len(development) + len(evaluation)} recordings",
            flush=True,
        )
        degraded_dev = degrade_many(development, [condition], seed=seed, workers=workers)
        degraded_eval = degrade_many(evaluation, [condition], seed=seed, workers=workers)
        print(f"    degraded in {time.monotonic() - started:.0f}s", flush=True)

        for duration in durations:
            cell, dev_trials = evaluate_cell(
                condition,
                duration,
                degraded_dev,
                degraded_eval,
                system,
                model_path,
                transferred,
                n_resamples=n_resamples,
                workers=workers,
                hypothesis=hypothesis,
                scores_sink=scores_sink,
            )
            results.append(cell)

            if (
                transferred is None
                and condition.label == REFERENCE_CONDITION.label
                and duration == max(durations)
                and dev_trials is not None
            ):
                transferred = _fit_calibrator(dev_trials)
                if transferred is None:
                    print(
                        "    warning: the reference condition produced no usable "
                        "calibrator; transferred C_llr will not be reported",
                        flush=True,
                    )

            if not np.isfinite(cell.c_llr_min):
                verdict = "NOT EVALUABLE"
            elif cell.h1_supported:
                verdict = "supported"
            elif cell.h1_falsified:
                verdict = "falsified"
            else:
                verdict = "inconclusive"
            print(
                f"    {duration:>4g}s  C_llr_min {cell.c_llr_min:.3f} "
                f"[{cell.c_llr_min_lower:.3f}, {cell.c_llr_min_upper:.3f}]  "
                f"EER {cell.equal_error_rate * 100:5.2f}%  "
                f"refused {cell.refusal_rate * 100:4.1f}%  {verdict}",
                flush=True,
            )

        if checkpoint is not None:
            checkpoint(results)
    return results


def overall_verdict(
    results: Sequence[CellResult], hypothesis: H1ChannelViability
) -> dict[str, Any]:
    """Decide H1 over the sweep as a whole.

    Reported per cell *and* aggregated, because the two answer different
    questions. A per-cell verdict says whether the channel at that operating
    point is usable. The aggregate says whether the acoustic stream is worth
    including at all, and it is decided on the condition the system would
    actually meet rather than on the best cell — reporting the best cell as the
    system's performance is how a sweep becomes a selection effect.
    """
    usable = [r for r in results if np.isfinite(r.c_llr_min)]
    if not usable:
        return {"verdict": "inconclusive", "reason": "no cell produced a usable metric"}

    supported = [r for r in usable if r.h1_supported]
    falsified = [r for r in usable if r.h1_falsified]
    best = min(usable, key=lambda r: r.c_llr_min)
    worst = max(usable, key=lambda r: r.c_llr_min)

    return {
        "n_cells": len(usable),
        "n_supported": len(supported),
        "n_falsified": len(falsified),
        "n_inconclusive": len(usable) - len(supported) - len(falsified),
        "best_cell": {
            "condition": best.condition,
            "duration": best.duration_seconds,
            "c_llr_min": best.c_llr_min,
            "interval": [best.c_llr_min_lower, best.c_llr_min_upper],
        },
        "worst_cell": {
            "condition": worst.condition,
            "duration": worst.duration_seconds,
            "c_llr_min": worst.c_llr_min,
            "interval": [worst.c_llr_min_lower, worst.c_llr_min_upper],
        },
        "decision_rule": (
            f"per cell: supported if the upper bound of C_llr_min <= "
            f"{hypothesis.support_below}; falsified if the lower bound > "
            f"{hypothesis.falsify_above}"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        default=None,
        help="corpus root; repeatable, so partitions can be pooled.",
    )
    parser.add_argument("--model", type=Path, default=Path("models/acoustic.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/h1_sweep.json"))
    parser.add_argument("--bitrates", type=float, nargs="+", default=[12.20, 4.75])
    parser.add_argument(
        "--noise",
        nargs="+",
        default=["babble", "vehicle"],
        choices=[n.value for n in NoiseType],
    )
    parser.add_argument("--snrs", type=float, nargs="+", default=[20.0, 10.0, 5.0])
    parser.add_argument("--durations", type=float, nargs="+", default=[30.0, 15.0, 5.0])
    parser.add_argument("--recording-seconds", type=float, default=30.0)
    parser.add_argument("--recordings-per-session", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--max-eval-recordings", type=int, default=None)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help=(
            "Persist per-trial evaluation scores, labels, owners and "
            "recording-id pairs to this .npz. The expensive half of this script "
            "is degrading and embedding the corpus; without the scores on disk, "
            "pairing this system against another costs a full rerun."
        ),
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20250601)
    arguments = parser.parse_args(argv)

    system = SpeakerComparisonSystem.load(arguments.model)
    print(f"loaded model {system.model_id}", flush=True)

    # Scan first, split on the plans, and read audio only for the two partitions
    # this run uses. Materialising the training partition as well would add
    # gigabytes that are never touched, and on a machine where that does not fit
    # the worker processes stop computing and start waiting on the pager.
    plans = scan_corpora(
        arguments.corpus or [Path("data/corpus/librispeech")],
        target_seconds=arguments.recording_seconds,
        max_recordings_per_session=arguments.recordings_per_session,
        max_speakers=arguments.max_speakers,
    )
    split = split_by_speaker(plans)
    print(f"split: {json.dumps(split.summary())}", flush=True)

    verify_disjoint(
        [r.speaker_id for r in split.train], [r.speaker_id for r in split.evaluation]
    )
    verify_disjoint(
        [r.speaker_id for r in split.development], [r.speaker_id for r in split.evaluation]
    )

    # The check above compares this split against itself and therefore always
    # passes. It cannot see the failure that actually happens: a correct split
    # computed over the *wrong corpus*. A model trained on two pooled partitions
    # and evaluated with --corpus left at its default gets an internally
    # disjoint split of the smaller partition, every check passes, and the model
    # is scored on speakers it memorised. Only the model knows who it trained
    # on, so the model is asked.
    if system.training_speakers:
        verify_disjoint(
            sorted(system.training_speakers),
            [r.speaker_id for r in split.evaluation],
        )
        print(
            f"  verified against the model's {len(system.training_speakers)} "
            f"training speakers",
            flush=True,
        )
    else:
        print(
            "  WARNING: this model does not record its training speakers, so "
            "evaluation cannot be verified disjoint from them. Retrain to "
            "enable the check; until then the split above is the only evidence.",
            flush=True,
        )

    development_plans = list(split.development)
    evaluation_plans = list(split.evaluation)
    if arguments.max_eval_recordings is not None:
        development_plans = development_plans[: arguments.max_eval_recordings]
        evaluation_plans = evaluation_plans[: arguments.max_eval_recordings]

    development = materialise(development_plans)
    evaluation = materialise(evaluation_plans)
    print(
        f"materialised {len(development)} development and {len(evaluation)} "
        f"evaluation recordings",
        flush=True,
    )

    conditions = build_sweep(
        arguments.bitrates, [NoiseType(n) for n in arguments.noise], arguments.snrs
    )
    print(
        f"sweep: {len(conditions)} channel conditions x "
        f"{len(arguments.durations)} durations = "
        f"{len(conditions) * len(arguments.durations)} cells, "
        f"{worker_count(arguments.workers)} workers",
        flush=True,
    )

    started = time.monotonic()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def write(results: Sequence[CellResult], *, complete: bool) -> None:
        payload = {
            "model_id": system.model_id,
            "model_describe": dict(system.describe()),
            "split": split.summary(),
            "durations": list(arguments.durations),
            "reference_condition": REFERENCE_CONDITION.label,
            "elapsed_minutes": round((time.monotonic() - started) / 60, 1),
            # Stated in the artefact itself. A partial sweep read as a complete
            # one would be reported as a verdict over the whole design when it
            # is a verdict over whichever conditions happened to finish.
            "complete": complete,
            "n_conditions_planned": len(conditions),
            "n_cells_planned": len(conditions) * len(arguments.durations),
            "verdict": overall_verdict(list(results), H1ChannelViability()),
            "cells": [asdict(r) for r in results],
        }
        arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    scores_sink: dict[str, TrialSet] | None = {} if arguments.scores else None
    results = run_sweep(
        development,
        evaluation,
        system,
        arguments.model,
        conditions,
        arguments.durations,
        n_resamples=arguments.resamples,
        workers=arguments.workers,
        seed=arguments.seed,
        checkpoint=lambda partial: write(partial, complete=False),
        scores_sink=scores_sink,
    )
    elapsed = time.monotonic() - started
    write(results, complete=True)
    print(f"\nwrote {arguments.output} after {elapsed / 60:.1f} min", flush=True)

    if arguments.scores is not None and scores_sink is not None:
        persisted: dict[str, NDArray[np.generic]] = {}
        for cell, trials in scores_sink.items():
            persisted[f"{cell}|scores"] = trials.scores
            persisted[f"{cell}|labels"] = trials.labels
            persisted[f"{cell}|owners"] = np.array(trials.speakers, dtype=np.str_)
            persisted[f"{cell}|pairs"] = np.array(
                ["\t".join(pair) for pair in trials.pairs], dtype=np.str_
            )
        arguments.scores.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arguments.scores, **persisted)
        print(f"wrote {arguments.scores} ({len(scores_sink)} cells)", flush=True)

    print(
        json.dumps(overall_verdict(results, H1ChannelViability()), indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
