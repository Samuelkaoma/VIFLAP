"""Is the calibrator or the channel the binding constraint?

The H1 sweep showed reported likelihood ratios barely improving with duration
even though discrimination improved substantially: mean matched calibration loss
of 0.171 bits at 30 seconds against 0.056 at 5 seconds. The loss was *largest*
where the system worked best, which points at the form of the calibration rather
than at the amount of evidence.

The suspicion is specific. :class:`LogisticCalibrator` is affine in the score —
one slope and one intercept. Where discrimination is good the two class-
conditional score distributions are well separated and not well described by any
single affine map, so information the ordering carries is discarded on the way to
a likelihood ratio. Two calibrators in the same module are not so constrained:
isotonic regression follows whatever monotonic shape the development data shows,
and the kernel-density calibrator estimates the two densities and reports their
ratio directly.

This script fits all three on the same development trials and reports each on the
same evaluation trials, so the only thing that varies is the mapping.
``C_llr_min`` is identical across all three by construction — it is invariant to
any monotonic transformation — which makes it the control: if it moves, something
is wrong with the experiment rather than interesting about the calibrators.

Both columns are persisted, and the first version persisted the wrong one
------------------------------------------------------------------------
This script originally scored ``calibrator.transform`` — the raw mapping, before
empirical bounding — and wrote it to the report under the field name ``c_llr``.
The results document quotes the *bounded* figures, which are the ones a
deployment emits and which had been recomputed by hand: 0.416 / 0.426 / 0.420 at
12.2 kbit/s clean and 30 s, against the 0.603 / 0.451 / 9.929 the artefact held.
The document was right and the artefact was not, which is the worse way round —
an examiner opening the JSON would reasonably conclude the numbers had been
edited rather than measured.

Both are now computed, by the same ``as_reported`` helper ``evaluate_h1.py``
uses, so the two scripts cannot disagree about what "the reported likelihood
ratio" means. ``c_llr`` is the bounded, as-reported quantity throughout;
``c_llr_unbounded`` is retained beside it for diagnosis, as §15 asks.

The clip is also measured rather than left implicit
---------------------------------------------------
§15 established that the bound is doing most of the work: at the best cell it
replaces the reported likelihood ratio for 60.6% of trials and removes 72% of
the calibration loss. So the clipped fraction is recorded per calibrator, split
by class, and so are the bounds themselves.

That last field settles a question §5 raised and could not answer from its own
output. ``empirical_bounds`` is computed from a PAV fit, which reads only the
*rank order* of the scores. A logistic map is affine-increasing and an isotonic
map is monotone, so both preserve that order and both must produce identical
bounds. Recording the bounds makes the identity visible in the artefact instead
of being an argument in prose — and it is why three calibrator families landing
within 0.01 of each other post-clip is close to a mathematical necessity rather
than a finding about calibration.

Scores are written to disk
--------------------------
Every per-trial score is saved alongside the report. Regenerating them costs a
full pass of codec simulation over several hundred recordings, and the first
version of this investigation had to redo exactly that because the sweep
persisted only its aggregates. Any further calibration question — a different
family, a different prior, a duration-aware gate — can now be answered from the
saved scores in seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.corpus import materialise, scan_corpus, split_by_speaker
from scripts.experiment import build_trials, degrade_many, embed_many
from viflap.analysis.calibration.calibrators import (
    Calibrator,
    IsotonicCalibrator,
    KernelDensityCalibrator,
    LogisticCalibrator,
    as_reported,
)
from viflap.analysis.calibration.metrics import compute_cllr, compute_cllr_min
from viflap.analysis.channel.degradation import DegradationCondition, NoiseType
from viflap.analysis.speaker.pipeline import SpeakerComparisonSystem
from viflap.domain.errors import CalibrationError, InsufficientDataError
from viflap.evaluation.splits import bootstrap_over_speakers

#: The cells that carry the question. The reference condition is where the loss
#: was largest; the other two check that whatever is found is not peculiar to a
#: clean channel at the highest rate.
DEFAULT_CONDITIONS: tuple[DegradationCondition, ...] = (
    DegradationCondition(bitrate_kbps=12.20),
    DegradationCondition(bitrate_kbps=12.20, noise_type=NoiseType.VEHICLE, snr_db=5.0),
    DegradationCondition(bitrate_kbps=4.75),
)

CALIBRATORS: dict[str, type] = {
    "logistic": LogisticCalibrator,
    "isotonic": IsotonicCalibrator,
    "kernel_density": KernelDensityCalibrator,
}


@dataclass(slots=True)
class CalibratorResult:
    """One calibrator's cost on one cell.

    ``c_llr`` is the as-reported, bounded quantity everywhere in this project.
    The unbounded companion is a diagnostic and is named so that it cannot be
    mistaken for the other.
    """

    name: str
    calibrator_id: str
    c_llr: float | None
    c_llr_lower: float | None
    c_llr_upper: float | None
    calibration_loss: float | None

    c_llr_unbounded: float | None = None
    calibration_loss_unbounded: float | None = None

    #: Share of evaluation trials whose reported log-LR is the bound rather than
    #: the calibration's own output, overall and by class. The asymmetry is the
    #: finding: §15 measured 61.5% of different-source against 3.0% of
    #: same-source, so the bound is almost entirely capping how negative the
    #: report may go.
    clipped_fraction: float | None = None
    clipped_fraction_different_source: float | None = None
    clipped_fraction_same_source: float | None = None

    #: The bounds themselves, base 10. Identical between any two calibrators
    #: whose maps preserve the score order, which is most of why the families
    #: agree post-clip.
    lower_log10_lr: float | None = None
    upper_log10_lr: float | None = None

    note: str = ""


def evaluate_calibrators(
    dev_scores: NDArray[np.float64],
    dev_labels: NDArray[np.int64],
    eval_scores: NDArray[np.float64],
    eval_labels: NDArray[np.int64],
    eval_speakers: Sequence[str],
    *,
    n_resamples: int,
) -> tuple[float, list[CalibratorResult]]:
    """Fit each calibrator on development trials and cost it on evaluation ones."""
    c_llr_min = compute_cllr_min(eval_scores, eval_labels)
    results: list[CalibratorResult] = []

    for name, factory in CALIBRATORS.items():
        try:
            calibrator = factory().fit(dev_scores, dev_labels)
        except (CalibrationError, InsufficientDataError) as exc:
            results.append(
                CalibratorResult(
                    name=name,
                    calibrator_id="",
                    c_llr=None,
                    c_llr_lower=None,
                    c_llr_upper=None,
                    calibration_loss=None,
                    note=f"could not be fitted: {exc.message}",
                )
            )
            continue

        unbounded = calibrator.transform(eval_scores)
        bounded = as_reported(calibrator, eval_scores)
        clip = _clipping(calibrator, unbounded, bounded, eval_labels)

        try:
            estimate = bootstrap_over_speakers(
                compute_cllr,
                bounded,
                eval_labels,
                eval_speakers,
                n_resamples=n_resamples,
            )
        except InsufficientDataError as exc:
            results.append(
                CalibratorResult(
                    name=name,
                    calibrator_id=calibrator.calibrator_id,
                    c_llr=None,
                    c_llr_lower=None,
                    c_llr_upper=None,
                    calibration_loss=None,
                    c_llr_unbounded=compute_cllr(unbounded, eval_labels),
                    note=f"no interval: {exc.message}",
                    **clip,
                )
            )
            continue

        c_llr_unbounded = compute_cllr(unbounded, eval_labels)
        results.append(
            CalibratorResult(
                name=name,
                calibrator_id=calibrator.calibrator_id,
                c_llr=estimate.value,
                c_llr_lower=estimate.lower,
                c_llr_upper=estimate.upper,
                calibration_loss=estimate.value - c_llr_min,
                c_llr_unbounded=c_llr_unbounded,
                calibration_loss_unbounded=c_llr_unbounded - c_llr_min,
                **clip,
            )
        )
    return c_llr_min, results


def _clipping(
    calibrator: Calibrator,
    unbounded: NDArray[np.float64],
    bounded: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> dict[str, float]:
    """How much of the report is the bound rather than the calibration."""
    clipped = unbounded != bounded
    same = labels == 1
    different = labels == 0
    bounds = calibrator.bounds
    return {
        "clipped_fraction": float(np.mean(clipped)),
        "clipped_fraction_different_source": float(np.mean(clipped[different]))
        if different.any()
        else 0.0,
        "clipped_fraction_same_source": float(np.mean(clipped[same]))
        if same.any()
        else 0.0,
        "lower_log10_lr": bounds.lower_log10,
        "upper_log10_lr": bounds.upper_log10,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/librispeech"))
    parser.add_argument("--model", type=Path, default=Path("models/acoustic.npz"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/calibrator_comparison.json")
    )
    parser.add_argument(
        "--scores", type=Path, default=Path("data/reports/calibration_scores.npz")
    )
    parser.add_argument("--durations", type=float, nargs="+", default=[30.0, 15.0, 5.0])
    parser.add_argument("--resamples", type=int, default=300)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20250601)
    arguments = parser.parse_args(argv)

    system = SpeakerComparisonSystem.load(arguments.model)
    print(f"loaded model {system.model_id}", flush=True)

    split = split_by_speaker(scan_corpus(arguments.corpus))
    development = materialise(split.development)
    evaluation = materialise(split.evaluation)
    print(
        f"{len(development)} development and {len(evaluation)} evaluation recordings",
        flush=True,
    )

    cells: list[dict[str, object]] = []
    saved: dict[str, NDArray[np.float64]] = {}
    started = time.monotonic()

    for index, condition in enumerate(DEFAULT_CONDITIONS, start=1):
        print(f"[{index}/{len(DEFAULT_CONDITIONS)}] {condition.label}", flush=True)
        degraded_dev = degrade_many(
            development, [condition], seed=arguments.seed, workers=arguments.workers
        )
        degraded_eval = degrade_many(
            evaluation, [condition], seed=arguments.seed, workers=arguments.workers
        )

        for duration in arguments.durations:
            dev_ids = {r.recording_id for r in degraded_dev}
            embedded, _ = embed_many(
                [r.truncated(duration) for r in [*degraded_dev, *degraded_eval]],
                arguments.model,
                workers=arguments.workers,
            )
            dev_embedded = [i for i in embedded if i[0].recording_id in dev_ids]
            eval_embedded = [i for i in embedded if i[0].recording_id not in dev_ids]

            try:
                dev_trials = build_trials(dev_embedded, system, seed=11)
                eval_trials = build_trials(eval_embedded, system, seed=22)
                c_llr_min, results = evaluate_calibrators(
                    dev_trials.scores,
                    dev_trials.labels,
                    eval_trials.scores,
                    eval_trials.labels,
                    eval_trials.speakers,
                    n_resamples=arguments.resamples,
                )
            except InsufficientDataError as exc:
                print(f"    {duration:>4g}s  not evaluable: {exc.message}", flush=True)
                cells.append(
                    {
                        "condition": condition.label,
                        "duration_seconds": duration,
                        "evaluable": False,
                        "note": exc.message,
                    }
                )
                continue

            key = f"{condition.label}@{duration:g}"
            saved[f"{key}|dev_scores"] = dev_trials.scores
            saved[f"{key}|dev_labels"] = dev_trials.labels
            saved[f"{key}|eval_scores"] = eval_trials.scores
            saved[f"{key}|eval_labels"] = eval_trials.labels
            saved[f"{key}|eval_speakers"] = np.array(eval_trials.speakers)

            cells.append(
                {
                    "condition": condition.label,
                    "duration_seconds": duration,
                    "evaluable": True,
                    "c_llr_min": c_llr_min,
                    "n_evaluation_speakers": eval_trials.n_speakers,
                    "calibrators": [asdict(r) for r in results],
                }
            )
            # Bounded first and unbounded in parentheses, always together. The
            # figure that matters operationally is the bounded one; printing it
            # alone is how the difference between the two stopped being visible.
            summary = "  ".join(
                f"{r.name}={r.c_llr:.3f}({r.c_llr_unbounded:.3f})"
                if r.c_llr is not None and r.c_llr_unbounded is not None
                else f"{r.name}=n/a"
                for r in results
            )
            clipped = results[0].clipped_fraction if results else None
            print(
                f"    {duration:>4g}s  C_llr_min {c_llr_min:.3f} | {summary}"
                + (f" | clipped {clipped * 100:.1f}%" if clipped is not None else ""),
                flush=True,
            )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "model_id": system.model_id,
                "conditions": [c.label for c in DEFAULT_CONDITIONS],
                "durations": list(arguments.durations),
                "elapsed_minutes": round((time.monotonic() - started) / 60, 1),
                "cells": cells,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(arguments.scores, **saved)
    print(
        f"\nwrote {arguments.output} and {arguments.scores} "
        f"({len(saved) // 5} cells of raw scores)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
