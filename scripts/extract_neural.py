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

Progress is on disk, not only in the process
---------------------------------------------
This run takes about ten hours on the 936-speaker corpus, and the first attempt
at it died in the fifth of five blocks having written nothing at all: the script
saved once, at the end. Ten hours of embeddings existed only as Python lists in
a process that no longer existed.

So a checkpoint is written after **every batch** — not every block. Block
granularity would have saved four fifths of that run, but the training block
alone is over five hours and losing it is still the largest single loss
available. At batch granularity the worst case is one batch, about fifteen
minutes. The checkpoint is uncompressed because it is rewritten roughly 120
times over a run and compression would cost more than the crash insurance is
worth; only the final artefact is compressed.

Resuming is guarded by a fingerprint of everything that would change the answer
— extractor, seed, durations, corpora, split and conditions. A checkpoint that
does not match is refused rather than silently continued, because a resumed run
that mixes two configurations produces an artefact no section could describe.

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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

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

#: Bumped when the checkpoint layout changes in a way that makes an older file
#: unreadable. A stale checkpoint is refused on this before anything else.
CHECKPOINT_VERSION = 1

#: Key under which the checkpoint's JSON metadata is stored inside the ``.npz``.
META_KEY = "__meta__"

#: Per-duration accumulators: vectors, speaker ids, recording ids, and a
#: one-element list holding the refusal count so it can be mutated in place.
Collected: TypeAlias = dict[
    "float | None", tuple[list[NDArray[np.float64]], list[str], list[str], list[int]]
]


@dataclass(frozen=True, slots=True)
class Block:
    """One condition over one partition — the unit a checkpoint completes.

    ``condition`` is either a single condition for the whole partition or one
    per recording, which is what the multi-condition training block needs.
    ``report_condition`` is what the extraction report calls it, and differs
    from ``name`` for training because "multi" is not a condition label.
    """

    name: str
    condition: DegradationCondition | Sequence[DegradationCondition]
    durations: list[float | None]
    plans: Sequence[RecordingPlan]
    partition: str
    report_condition: str


def cell_key(block: str, duration: float | None) -> str:
    """The ``.npz`` key prefix for one block at one duration.

    Duration ``None`` means the block ran at full length and carries no suffix,
    which is what the training block does and what every downstream reader
    already expects.
    """
    return block if duration is None else f"{block}@{duration:g}"


def _empty(durations: Sequence[float | None]) -> Collected:
    return {duration: ([], [], [], [0]) for duration in durations}


def _finalise(
    collected: Collected,
) -> dict[float | None, tuple[NDArray[np.float64], list[str], list[str], int]]:
    """Stack the accumulators into arrays without consuming them."""
    return {
        duration: (
            np.stack(vectors) if vectors else np.zeros((0, 0)),
            speakers,
            ids,
            refused[0],
        )
        for duration, (vectors, speakers, ids, refused) in collected.items()
    }


def extract(
    plans: Sequence[RecordingPlan],
    condition: DegradationCondition | Sequence[DegradationCondition],
    durations: Sequence[float | None],
    extractor: object,
    *,
    seed: int,
    workers: int | None,
    label: str,
    resume: Collected | None = None,
    start_index: int = 0,
    on_batch: Callable[[int, Collected], None] | None = None,
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

    ``start_index`` and ``resume`` restart a block part-way through. Batches are
    aligned to plan order and the degradation seed does not vary with the batch,
    so recording *n* is degraded identically whether the run reached it in one
    pass or two. ``on_batch`` is called after every batch with the number of
    plans consumed so far and the live accumulators, which is what makes the
    checkpoint a record of the current batch rather than of the current block.
    """
    collected: Collected = resume if resume is not None else _empty(durations)
    started = time.monotonic()

    per_recording = not isinstance(condition, DegradationCondition)
    for start in range(start_index, len(plans), BATCH):
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
        if on_batch is not None:
            on_batch(done, collected)
        elapsed = time.monotonic() - started
        fresh = max(done - start_index, 1)
        print(
            f"    {label}: {done}/{len(plans)} in {elapsed / 60:.1f} min "
            f"({elapsed / fresh:.2f} s/rec across {len(durations)} durations)",
            flush=True,
        )

    return _finalise(collected)


def checkpoint_path_for(output: Path, override: Path | None) -> Path:
    """Where the running checkpoint lives, next to the artefact it will become."""
    if override is not None:
        return override
    return output.with_name(output.stem + ".checkpoint.npz")


def save_checkpoint(
    path: Path,
    *,
    fingerprint: dict[str, object],
    arrays: dict[str, NDArray[Any]],
    summary: list[dict[str, object]],
    completed: Sequence[str],
    partial: dict[str, Any] | None,
    elapsed_minutes: float,
) -> None:
    """Write the checkpoint atomically, so a crash mid-write cannot corrupt it.

    Uncompressed on purpose: this is rewritten after every batch and compression
    would spend minutes over a run protecting against a failure that costs one
    batch. The final artefact is compressed; this is not the final artefact.

    ``elapsed_minutes`` is carried so that a resumed run reports the total cost
    of producing the artefact rather than the cost of its last attempt, which
    would understate it every time and is the sort of figure that gets quoted.
    """
    meta = json.dumps(
        {
            "version": CHECKPOINT_VERSION,
            "fingerprint": fingerprint,
            "summary": summary,
            "completed": list(completed),
            "partial": partial,
            "elapsed_minutes": elapsed_minutes,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays, **{META_KEY: np.array(meta, dtype=np.str_)})
    temporary.replace(path)


def load_checkpoint(
    path: Path, fingerprint: dict[str, object]
) -> tuple[
    dict[str, NDArray[Any]],
    list[dict[str, object]],
    set[str],
    dict[str, Any] | None,
    float,
]:
    """Read a checkpoint, refusing one that does not match this run.

    Refusing rather than warning is deliberate. A checkpoint from a different
    seed, corpus or extractor would resume into an artefact that is half one
    configuration and half another, and nothing downstream could detect it — the
    file would be well-formed and the numbers would be wrong.
    """
    with np.load(path, allow_pickle=False) as loaded:
        meta = json.loads(str(loaded[META_KEY]))
        arrays = {key: loaded[key] for key in loaded.files if key != META_KEY}

    if meta.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} was written by checkpoint version {meta.get('version')}, "
            f"this script writes version {CHECKPOINT_VERSION}; delete it to restart"
        )
    stored = meta.get("fingerprint")
    if stored != fingerprint:
        differing = sorted(
            key
            for key in set(fingerprint) | set(stored or {})
            if (stored or {}).get(key) != fingerprint.get(key)
        )
        raise ValueError(
            f"{path} was written for a different configuration (differs in "
            f"{', '.join(differing)}); point --checkpoint elsewhere or delete it"
        )
    return (
        arrays,
        list(meta["summary"]),
        set(meta["completed"]),
        meta["partial"],
        float(meta.get("elapsed_minutes", 0.0)),
    )


def restore_block(
    arrays: dict[str, NDArray[Any]],
    block: str,
    durations: Sequence[float | None],
    refused: dict[str, int],
) -> Collected:
    """Turn the checkpoint's arrays back into live accumulators for one block."""
    collected: Collected = {}
    for duration in durations:
        key = cell_key(block, duration)
        vectors = arrays.get(f"{key}|vectors")
        speakers = arrays.get(f"{key}|speakers")
        ids = arrays.get(f"{key}|recordings")
        if vectors is None or speakers is None or ids is None:
            collected[duration] = ([], [], [], [refused.get(key, 0)])
            continue
        collected[duration] = (
            [np.asarray(row, dtype=np.float64) for row in np.atleast_2d(vectors)]
            if vectors.size
            else [],
            [str(value) for value in speakers],
            [str(value) for value in ids],
            [refused.get(key, 0)],
        )
    return collected


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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="where to keep resumable progress; defaults to <output>.checkpoint.npz",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="ignore and overwrite any existing checkpoint",
    )
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

    evaluation_conditions = [
        DegradationCondition(bitrate_kbps=12.20),
        DegradationCondition(bitrate_kbps=12.20, noise_type=NoiseType.BABBLE, snr_db=20.0),
    ]
    train_conditions = assign_conditions(
        [p.speaker_id for p in split.train], seed=arguments.seed
    )

    # Training first, multi-condition and at full duration. The back-end is
    # fitted on the same channel mixture the i-vector system was, so the two
    # differ in the extractor and in nothing else.
    blocks: list[Block] = [
        Block("train", train_conditions, [None], split.train, "train", "multi")
    ]
    for partition, partition_plans in (
        ("development", split.development),
        ("evaluation", split.evaluation),
    ):
        for condition in evaluation_conditions:
            blocks.append(
                Block(
                    f"{partition}|{condition.label}",
                    condition,
                    [float(d) for d in arguments.durations],
                    partition_plans,
                    partition,
                    condition.label,
                )
            )

    fingerprint: dict[str, object] = {
        "extractor_id": extractor.extractor_id,
        "seed": int(arguments.seed),
        "durations": [float(d) for d in arguments.durations],
        "corpora": [str(root) for root in corpora],
        "split": split.summary(),
        "batch": BATCH,
        "training_conditions": [c.label for c in TRAINING_CONDITIONS],
        "blocks": [block.name for block in blocks],
    }

    checkpoint = checkpoint_path_for(arguments.output, arguments.checkpoint)
    saved: dict[str, NDArray[Any]] = {}
    summary: list[dict[str, object]] = []
    completed: set[str] = set()
    partial: dict[str, Any] | None = None
    accrued_minutes = 0.0

    if arguments.restart and checkpoint.exists():
        checkpoint.unlink()
        print(f"  discarded {checkpoint} at --restart", flush=True)
    if checkpoint.exists():
        saved, summary, completed, partial, accrued_minutes = load_checkpoint(
            checkpoint, fingerprint
        )
        print(
            f"  resuming from {checkpoint}: {len(completed)} of {len(blocks)} blocks "
            f"complete after {accrued_minutes:.1f} min, partial={json.dumps(partial)}",
            flush=True,
        )

    started = time.monotonic()

    def elapsed_minutes() -> float:
        return round(accrued_minutes + (time.monotonic() - started) / 60, 1)

    for block in blocks:
        if block.name in completed:
            print(f"[{block.name}] already in the checkpoint, skipping", flush=True)
            continue

        resume: Collected | None = None
        start_index = 0
        if partial is not None and partial.get("block") == block.name:
            start_index = int(partial["done"])
            refused = {str(key): int(value) for key, value in partial["refused"].items()}
            resume = restore_block(saved, block.name, block.durations, refused)
            print(
                f"[{block.name}] resuming at recording {start_index}/{len(block.plans)}",
                flush=True,
            )
        else:
            print(f"[{block.name}] {len(block.durations)} duration(s)", flush=True)

        def record(done: int, collected: Collected, block: Block = block) -> None:
            arrays = dict(saved)
            refused_now: dict[str, int] = {}
            for duration, cell in _finalise(collected).items():
                key = cell_key(block.name, duration)
                arrays[f"{key}|vectors"] = cell[0]
                arrays[f"{key}|speakers"] = np.array(cell[1], dtype=np.str_)
                arrays[f"{key}|recordings"] = np.array(cell[2], dtype=np.str_)
                refused_now[key] = cell[3]
            save_checkpoint(
                checkpoint,
                fingerprint=fingerprint,
                arrays=arrays,
                summary=summary,
                completed=sorted(completed),
                partial={"block": block.name, "done": done, "refused": refused_now},
                elapsed_minutes=elapsed_minutes(),
            )

        per_duration = extract(
            block.plans,
            block.condition,
            block.durations,
            extractor,
            seed=arguments.seed,
            workers=arguments.workers,
            label=block.name,
            resume=resume,
            start_index=start_index,
            on_batch=record,
        )

        for duration, (vectors, speakers, ids, refused_count) in per_duration.items():
            key = cell_key(block.name, duration)
            saved[f"{key}|vectors"] = vectors
            saved[f"{key}|speakers"] = np.array(speakers, dtype=np.str_)
            saved[f"{key}|recordings"] = np.array(ids, dtype=np.str_)
            summary.append(
                {
                    "partition": block.partition,
                    "condition": block.report_condition,
                    "duration_seconds": 30.0 if duration is None else duration,
                    "n_embeddings": int(vectors.shape[0]),
                    "n_refused": refused_count,
                }
            )
        completed.add(block.name)
        partial = None
        save_checkpoint(
            checkpoint,
            fingerprint=fingerprint,
            arrays=saved,
            summary=summary,
            completed=sorted(completed),
            partial=None,
            elapsed_minutes=elapsed_minutes(),
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
                "elapsed_minutes": elapsed_minutes(),
                "cells": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    checkpoint.unlink(missing_ok=True)
    print(f"\nwrote {arguments.output} and {arguments.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
