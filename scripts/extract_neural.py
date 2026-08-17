"""Embed the corpus with a borrowed extractor, and keep only the embeddings.

§12 concluded that the gap against a published forensic system is speaker count
rather than architecture, in the extractor as well as the back-end. This script
performs the move that follows: run a VoxCeleb2-trained network over the same
corpus, through the same channel, and hand the result to the same back-end.

Why extraction is a separate script from training
-------------------------------------------------
Embedding the corpus costs hours. Fitting LDA and PLDA to the result costs
seconds. Putting both in one script means every back-end question — a different
LDA dimension, a different PLDA rank, a scoring bug found on the third pass —
pays the extraction cost again. `compare_calibrators.py` learned this the hard
way and now persists its per-trial scores for the same reason.

So this writes an ``.npz`` of embeddings and labels and stops. Everything
downstream reads that file.

Memory is bounded by batching, not by hope
-------------------------------------------
2,578 recordings of 30 seconds at 16 kHz is about 10 GB held as float64, on a
machine with 12 GB. `train_acoustic.py` materialises 1,539 of them and reaches
8.5 GB resident, which is close enough to the edge that a second pass over a
larger set would page rather than compute — and a paging job on this machine has
previously sat at six percent CPU for hours. Recordings are therefore
materialised, degraded, embedded and **discarded** in batches, so peak memory is
set by the batch rather than by the corpus.

What is deliberately not here
------------------------------
No scores, no ``C_llr``, no verdict. This produces vectors. Comparing them
against the i-vector system is a separate step on the same held-out speakers,
because §7 is a standing reminder that a plausible improvement can be
significantly worse in every cell, and the only way to know is to pair them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.corpus import RecordingPlan, materialise, scan_corpora, split_by_speaker
from scripts.experiment import degrade_many, worker_count
from scripts.train_acoustic import (
    DEGRADATION_SEED,
    TRAINING_CONDITIONS,
    assign_conditions,
)
from viflap.analysis.channel.degradation import DegradationCondition, NoiseType
from viflap.domain.errors import InsufficientDataError

#: Recordings held in memory at once. 120 at 30 seconds is roughly 460 MB of
#: float64 audio before degradation and about the same after, which leaves the
#: extractor and the operating system room on a 12 GB machine.
BATCH = 120


def extract(
    plans: Sequence[RecordingPlan],
    condition: DegradationCondition | Sequence[DegradationCondition],
    durations: Sequence[float | None],
    extractor: object,
    *,
    seed: int,
    workers: int | None,
    label: str,
) -> dict[float | None, tuple[NDArray[np.float64], list[str], list[str], int]]:
    """Embed a partition under one condition, at every duration, in batches.

    **All durations come off one pass of degradation**, which is the whole shape
    of this function. Truncation is free and coding is not: a smoke test that
    degraded once per duration spent 4.37 seconds per recording where two of
    every three were re-coding audio it already had, and would have added about
    three hours to the full run. `evaluate_h1.py` has always degraded once per
    condition and truncated afterwards; this now does the same.

    It also makes the durations *nested* rather than merely equal-cost: the 15 s
    embedding is taken from the front of the same coded signal as the 30 s one,
    so a duration contrast is not confounded with a fresh noise realisation.

    Returns, per duration, the embeddings with their speaker and recording ids
    and how many recordings the extractor refused. Refusals are counted rather
    than dropped silently: §6 records that at short durations they are not
    random with respect to difficulty — they remove the recordings carrying
    least speech, which are the hardest — so the count is part of the result.
    """
    collected: dict[
        float | None, tuple[list[NDArray[np.float64]], list[str], list[str], list[int]]
    ] = {duration: ([], [], [], [0]) for duration in durations}
    started = time.monotonic()

    per_recording = not isinstance(condition, DegradationCondition)
    for start in range(0, len(plans), BATCH):
        chunk = list(plans[start : start + BATCH])
        conditions_for_chunk = (
            list(condition)[start : start + BATCH] if per_recording else [condition]
        )
        degraded = degrade_many(
            materialise(chunk), conditions_for_chunk, seed=seed, workers=workers
        )
        for item in degraded:
            for duration in durations:
                piece = item.truncated(duration) if duration is not None else item
                vectors, speakers, ids, refused = collected[duration]
                try:
                    embedding = extractor.embed(piece.signal, piece.sample_rate)  # type: ignore[attr-defined]
                except InsufficientDataError:
                    refused[0] += 1
                    continue
                vectors.append(embedding)
                speakers.append(piece.speaker_id)
                ids.append(piece.recording_id)
        done = min(start + BATCH, len(plans))
        elapsed = time.monotonic() - started
        print(
            f"    {label}: {done}/{len(plans)} in {elapsed / 60:.1f} min "
            f"({elapsed / max(done, 1):.2f} s/rec across {len(durations)} durations)",
            flush=True,
        )

    return {
        duration: (
            np.stack(vectors) if vectors else np.zeros((0, 0)),
            speakers,
            ids,
            refused[0],
        )
        for duration, (vectors, speakers, ids, refused) in collected.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/neural_embeddings.npz")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/reports/neural_extraction.json")
    )
    parser.add_argument("--durations", type=float, nargs="+", default=[30.0, 15.0, 5.0])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEGRADATION_SEED)
    arguments = parser.parse_args(argv)

    corpora = arguments.corpus or [
        Path("data/corpus/librispeech"),
        Path("data/corpus/librispeech-360"),
    ]
    print(f"scanning {', '.join(str(root) for root in corpora)}", flush=True)
    plans = scan_corpora(corpora, target_seconds=30.0, max_recordings_per_session=2)
    split = split_by_speaker(plans)
    print(f"  split: {json.dumps(split.summary())}", flush=True)

    from viflap.infrastructure.neural_extractor import NeuralEmbeddingExtractor

    extractor = NeuralEmbeddingExtractor()
    print(f"  extractor: {extractor.extractor_id}", flush=True)
    print(f"  workers for degradation: {worker_count(arguments.workers)}", flush=True)

    saved: dict[str, NDArray[np.float64]] = {}
    summary: list[dict[str, object]] = []
    started = time.monotonic()

    # Training first, multi-condition and at full duration. The back-end is
    # fitted on the same channel mixture the i-vector system was, so the two
    # differ in the extractor and in nothing else.
    print("[train] multi-condition, 30 s", flush=True)
    train_conditions = assign_conditions(
        [p.speaker_id for p in split.train], seed=arguments.seed
    )
    vectors, speakers, ids, refused = extract(
        split.train,
        train_conditions,
        [None],
        extractor,
        seed=arguments.seed,
        workers=arguments.workers,
        label="train",
    )[None]
    saved["train|vectors"] = vectors
    saved["train|speakers"] = np.array(speakers, dtype=np.str_)
    saved["train|recordings"] = np.array(ids, dtype=np.str_)
    summary.append(
        {
            "partition": "train",
            "condition": "multi",
            "duration_seconds": 30.0,
            "n_embeddings": int(vectors.shape[0]),
            "n_refused": refused,
        }
    )

    evaluation_conditions = [
        DegradationCondition(bitrate_kbps=12.20),
        DegradationCondition(bitrate_kbps=12.20, noise_type=NoiseType.BABBLE, snr_db=20.0),
    ]
    for partition, partition_plans in (
        ("development", split.development),
        ("evaluation", split.evaluation),
    ):
        for condition in evaluation_conditions:
            print(f"[{partition}|{condition.label}] all durations", flush=True)
            per_duration = extract(
                partition_plans,
                condition,
                list(arguments.durations),
                extractor,
                seed=arguments.seed,
                workers=arguments.workers,
                label=f"{partition}|{condition.label}",
            )
            for duration, (vectors, speakers, ids, refused) in per_duration.items():
                key = f"{partition}|{condition.label}@{duration:g}"
                saved[f"{key}|vectors"] = vectors
                saved[f"{key}|speakers"] = np.array(speakers, dtype=np.str_)
                saved[f"{key}|recordings"] = np.array(ids, dtype=np.str_)
                summary.append(
                    {
                        "partition": partition,
                        "condition": condition.label,
                        "duration_seconds": duration,
                        "n_embeddings": int(vectors.shape[0]),
                        "n_refused": refused,
                    }
                )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **saved)
    arguments.report.write_text(
        json.dumps(
            {
                "extractor_id": extractor.extractor_id,
                "split": split.summary(),
                "training_conditions": [c.label for c in TRAINING_CONDITIONS],
                "durations": list(arguments.durations),
                "elapsed_minutes": round((time.monotonic() - started) / 60, 1),
                "cells": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {arguments.output} and {arguments.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
