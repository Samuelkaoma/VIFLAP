"""Train the acoustic stack on real speech.

Trains UBM, total variability matrix, session compensation and PLDA on
speaker-disjoint data, and writes the result as a single archive whose
identifier is derived from its own parameters.

Multi-condition training, and why it is not optional
----------------------------------------------------
The training recordings do not go through one channel. Each is assigned its own
condition from the degradation design, so the corpus spans bitrates, noise types
and signal-to-noise ratios.

Training on clean audio and evaluating on degraded audio measures the mismatch
between two channels rather than the discriminability of speakers, and it
measures it pessimistically — the front-end statistics differ before any speaker
information is considered. Training through the same family of channels the
system will meet puts the model in the right domain, and it is what every
operational telephony system does.

The cost is that the model is specific to that family. It is not a general
speaker recognition system; it is one fitted to simulated narrowband telephony,
and the archive records which codec produced its training material.

What this script does not do
----------------------------
It does not fit a calibrator. A calibrator maps scores to likelihood ratios and
must be fitted on speakers disjoint from both training and evaluation; it
belongs to the evaluation, where the condition it is fitted for is known. The
model this script writes produces uncalibrated scores, and nothing downstream is
permitted to report one as evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

from scripts.corpus import (
    Recording,
    materialise,
    scan_corpora,
    split_by_speaker,
    write_split_manifest,
)
from scripts.corpus_zambian import scan_unlabelled
from scripts.experiment import (
    LazyBackgroundCorpus,
    degrade_many,
    summarise_codec_modes,
    worker_count,
)
from viflap.analysis.channel.degradation import DegradationCondition, NoiseType
from viflap.analysis.speaker.gmm import GmmConfig
from viflap.analysis.speaker.ivector import TotalVariabilityConfig
from viflap.analysis.speaker.pipeline import (
    FrontEndConfig,
    SpeakerComparisonSystem,
    SpeakerSystemConfig,
    TrainingRecording,
)
from viflap.analysis.speaker.plda import PldaConfig

#: Seeds every noise realisation and condition assignment in a training run.
#: One constant rather than a default repeated at each call site, so the
#: labelled and background corpora cannot drift onto different streams and
#: quietly stop being reproducible together.
DEGRADATION_SEED: int = 20250601

#: The channel conditions training material is drawn from. Spans the rate
#: extremes and both noise characters, so the model is not fitted to one point
#: of the design it will be evaluated over.
TRAINING_CONDITIONS: tuple[DegradationCondition, ...] = (
    DegradationCondition(bitrate_kbps=12.20),
    DegradationCondition(bitrate_kbps=7.40),
    DegradationCondition(bitrate_kbps=4.75),
    DegradationCondition(bitrate_kbps=12.20, noise_type=NoiseType.BABBLE, snr_db=20.0),
    DegradationCondition(bitrate_kbps=7.40, noise_type=NoiseType.BABBLE, snr_db=10.0),
    DegradationCondition(bitrate_kbps=4.75, noise_type=NoiseType.VEHICLE, snr_db=15.0),
    DegradationCondition(bitrate_kbps=12.20, noise_type=NoiseType.VEHICLE, snr_db=10.0),
    DegradationCondition(bitrate_kbps=5.90, noise_type=NoiseType.BABBLE, snr_db=15.0),
)


def assign_conditions_globally(
    n_recordings: int, seed: int = DEGRADATION_SEED
) -> list[DegradationCondition]:
    """Cycle a permuted list, ignoring who is speaking.

    Exactly balanced corpus-wide, which is what it was written for: independent
    draws leave one condition over-represented by chance, and condition is
    confounded with whichever speakers received it.

    It does not balance *within* a speaker, and that is the defect
    :func:`assign_conditions` exists to fix. Retained so the two can be compared
    on the same corpus rather than the old behaviour being replaced by
    assertion — see :func:`assign_conditions` for what is at stake.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_recordings)
    conditions = list(TRAINING_CONDITIONS)
    return [conditions[int(position) % len(conditions)] for position in order]


def assign_conditions(
    speaker_ids: Sequence[str], seed: int = DEGRADATION_SEED
) -> list[DegradationCondition]:
    """Give each training recording a channel condition, balanced within speaker.

    Corpus-wide balance is not enough, and the reason is specific to what PLDA
    estimates. A speaker's recordings are the only evidence the model has about
    within-speaker variability; whatever is *common* to them is, as far as the
    model can tell, the speaker. Under a global permutation each speaker draws
    their conditions at random from the balanced pool, so each carries a mean
    channel offset of their own — one speaker happens to be mostly 4.75 kbit/s
    in babble, another mostly 12.2 clean — and LDA and PLDA absorb that offset
    as between-speaker variance. The model learns a channel and reports it as a
    person.

    The allocation here removes the offset instead of hoping it averages out.
    Speakers are taken in a fixed order and each is given a *contiguous block*
    of the condition cycle, the cycle continuing across speaker boundaries:

    .. code-block:: text

        conditions   c0 c1 c2 c3 c4 c5 c6 c7 c0 c1 c2 c3 ...
        speaker      |--- A (5) ---|--- B (4) --|-- C (3) --

    Every speaker therefore receives ``min(n_recordings, 8)`` *distinct*
    conditions with no repeat until the design is exhausted, which is the Latin
    square's within-block property carried over to blocks of unequal size. The
    corpus-wide counts are unchanged — the cycle is contiguous, so the totals
    are the same as running it straight through — and the starting point rotates
    across speakers, so no condition attaches preferentially to the speakers
    sorting early.

    The order *within* a speaker is then permuted. Without that, condition would
    track position in the recording list, which for LibriSpeech means chapter
    order, and a systematic association between condition and session is the
    same defect one level down.

    Whether this matters is an empirical question and is not settled by the
    argument above: the whole point is to be able to retrain with it and see
    whether the leading between-speaker variance falls. ``psi[0]`` runs 5.1 to
    7.0 times ``psi[1]`` across every model trained so far, which is what one
    dominant nuisance axis looks like, but a spike consistent with a confound is
    not a confound measured.
    """
    rng = np.random.default_rng(seed)
    conditions = list(TRAINING_CONDITIONS)
    n_conditions = len(conditions)

    positions_by_speaker: dict[str, list[int]] = {}
    for index, speaker in enumerate(speaker_ids):
        positions_by_speaker.setdefault(speaker, []).append(index)

    assigned: list[DegradationCondition | None] = [None] * len(speaker_ids)
    offset = 0
    # Sorted rather than insertion-ordered so the allocation depends on the
    # corpus and not on the order a scan happened to walk the filesystem in.
    for speaker in sorted(positions_by_speaker):
        positions = positions_by_speaker[speaker]
        block = [conditions[(offset + i) % n_conditions] for i in range(len(positions))]
        shuffled = rng.permutation(len(positions))
        for slot, position in enumerate(positions):
            assigned[position] = block[int(shuffled[slot])]
        offset += len(positions)

    if any(condition is None for condition in assigned):
        raise RuntimeError(
            "some recordings were left without a channel condition, which means "
            "the allocation did not cover every position it was given"
        )
    return cast("list[DegradationCondition]", assigned)


def condition_balance(
    speaker_ids: Sequence[str], conditions: Sequence[DegradationCondition]
) -> dict[str, float]:
    """How evenly the channel conditions fell across and within speakers.

    Recorded in the training report so the allocation is visible in the artefact
    rather than being a property of a function nobody reads. Every model this
    project has trained was built under an allocation that is exactly balanced
    corpus-wide and unbalanced within speaker, and nothing written down said so.

    ``speaker_mean_bitrate_sd`` is the figure that matters. It is the standard
    deviation across speakers of each speaker's mean training bitrate — that is,
    the size of the per-speaker channel offset that LDA and PLDA have no way to
    distinguish from the speaker. Corpus-wide balance leaves it non-zero; the
    stratified allocation is what drives it toward zero.
    """
    by_speaker: dict[str, list[DegradationCondition]] = {}
    for speaker, condition in zip(speaker_ids, conditions, strict=True):
        by_speaker.setdefault(speaker, []).append(condition)

    distinct = [len({c.label for c in items}) for items in by_speaker.values()]
    reachable = [min(len(items), len(TRAINING_CONDITIONS)) for items in by_speaker.values()]
    mean_bitrates = [
        float(np.mean([c.bitrate_kbps for c in items])) for items in by_speaker.values()
    ]
    counts = np.array(
        [
            sum(1 for c in conditions if c.label == reference.label)
            for reference in TRAINING_CONDITIONS
        ],
        dtype=float,
    )

    return {
        "n_speakers": float(len(by_speaker)),
        "mean_distinct_conditions_per_speaker": float(np.mean(distinct)),
        "mean_reachable_conditions_per_speaker": float(np.mean(reachable)),
        "speakers_with_a_repeated_condition": float(
            np.mean([d < r for d, r in zip(distinct, reachable, strict=True)])
        ),
        "speaker_mean_bitrate_sd": float(np.std(mean_bitrates)),
        "corpus_condition_count_min": float(counts.min()),
        "corpus_condition_count_max": float(counts.max()),
    }


def build_config(
    n_components: int,
    rank: int,
    lda_dimension: int | None,
    ubm_frame_budget: int | None,
    min_speakers: int,
    background_recording_budget: int | None = None,
    cmvn_frames: int | None = None,
) -> SpeakerSystemConfig:
    """Assemble a training configuration.

    ``cmvn_frames`` overrides the front-end's normalisation window, and it is
    exposed here rather than left at its default because the window is not
    duration-neutral: a fixed window spans a different fraction of a 30 s
    recording than of a 5 s one, so a model trained at one setting and swept
    across durations varies its front-end alongside the factor under test. The
    override exists to hold that fixed. It has to be set at *training* time —
    the window is part of the front-end the model was fitted to and travels
    inside the archive — so evaluating a model under a window it was not trained
    with would measure a mismatch instead.
    """
    front_end = (
        FrontEndConfig()
        if cmvn_frames is None
        else FrontEndConfig(sliding_cmvn_frames=cmvn_frames)
    )
    return SpeakerSystemConfig(
        front_end=front_end,
        ubm=GmmConfig(n_components=n_components, max_iterations=60),
        total_variability=TotalVariabilityConfig(rank=rank, max_iterations=10),
        plda=PldaConfig(max_iterations=40, min_speakers=min_speakers),
        lda_dimension=lda_dimension,
        ubm_frame_budget=ubm_frame_budget,
        background_recording_budget=background_recording_budget,
    )


def train(
    training: Sequence[Recording],
    config: SpeakerSystemConfig,
    output: Path,
    split_summary: dict[str, int],
    *,
    workers: int | None = None,
    seed: int = DEGRADATION_SEED,
    background: Sequence[TrainingRecording] | None = None,
    condition_allocation: str = "stratified",
) -> dict[str, object]:
    """Degrade the training partition, train on it, and save the model.

    ``split_summary`` is passed in rather than derived from ``training``,
    because this function only ever sees the training partition — the other two
    are never loaded here — and the report needs to record all three.

    ``background`` supplies the unsupervised stages from a separate corpus. It
    is never materialised here: the sequence is handed to the trainer as given,
    which loads each recording as it reaches it. Pass a
    :class:`~scripts.experiment.LazyBackgroundCorpus` rather than a list unless
    the pool is small enough to hold.
    """
    training = list(training)
    if condition_allocation == "stratified":
        conditions = assign_conditions([r.speaker_id for r in training], seed=seed)
    elif condition_allocation == "global":
        conditions = assign_conditions_globally(len(training), seed=seed)
    else:
        raise ValueError(f"unknown condition allocation {condition_allocation!r}")

    print(
        f"degrading {len(training)} training recordings across "
        f"{len(TRAINING_CONDITIONS)} channel conditions "
        f"({condition_allocation}) on "
        f"{worker_count(workers)} workers",
        flush=True,
    )
    started = time.monotonic()
    degraded = degrade_many(training, conditions, seed=seed, workers=workers)
    degrade_seconds = time.monotonic() - started
    codec_summary = summarise_codec_modes(degraded)
    print(
        f"  degraded in {degrade_seconds / 60:.1f} min "
        f"(codec: {', '.join(codec_summary['codec_modes'])})",
        flush=True,
    )

    if not codec_summary["is_single_mode"]:
        raise RuntimeError(
            "training material was produced by more than one codec implementation; "
            f"modes seen: {codec_summary['codec_modes']}. Pooling them would make "
            "the model's training domain undefined."
        )

    recordings = [
        TrainingRecording(
            signal=item.signal,
            sample_rate=item.sample_rate,
            speaker_id=item.speaker_id,
            recording_id=item.recording_id,
        )
        for item in degraded
    ]

    print(
        f"training: {config.ubm.n_components} components, rank "
        f"{config.total_variability.rank}, LDA {config.lda_dimension}",
        flush=True,
    )
    started = time.monotonic()
    system = SpeakerComparisonSystem.train(recordings, config, background=background)
    train_seconds = time.monotonic() - started

    output.parent.mkdir(parents=True, exist_ok=True)
    system.save(output)

    report: dict[str, object] = {
        "model_id": system.model_id,
        "model_path": str(output),
        "split": split_summary,
        "training_conditions": [c.label for c in TRAINING_CONDITIONS],
        "condition_allocation": condition_allocation,
        "condition_balance": condition_balance(
            [r.speaker_id for r in training], conditions
        ),
        "codec": codec_summary,
        "degrade_seconds": round(degrade_seconds, 1),
        "train_seconds": round(train_seconds, 1),
        "describe": dict(system.describe()),
    }
    print(
        f"  trained in {train_seconds / 60:.1f} min -> {system.model_id}",
        flush=True,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        default=None,
        help=(
            "corpus root. Repeatable: corpora ship in partitions and the "
            "speaker count is what bounds this system, so pooling them is the "
            "cheapest way to raise the LDA ceiling. Roots reusing a speaker "
            "identifier are refused rather than merged."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("models/acoustic.npz"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/training.json"))
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--rank", type=int, default=100)
    parser.add_argument("--lda-dimension", type=int, default=None)
    parser.add_argument("--ubm-frame-budget", type=int, default=600_000)
    parser.add_argument("--min-speakers", type=int, default=20)
    parser.add_argument(
        "--cmvn-frames",
        type=int,
        default=None,
        help=(
            "cepstral normalisation window in frames; 0 or below normalises over "
            "the whole utterance. Defaults to the front-end's 300 (three "
            "seconds). A fixed window spans a different fraction of a short "
            "recording than of a long one, so hold this at a duration-invariant "
            "setting when duration is the factor under test."
        ),
    )
    parser.add_argument(
        "--condition-allocation",
        choices=["stratified", "global"],
        default="stratified",
        help=(
            "how training channel conditions are dealt out. 'stratified' gives "
            "each speaker a contiguous block of the condition cycle, so every "
            "speaker sees as many distinct conditions as they have recordings "
            "and carries no mean channel offset. 'global' permutes across the "
            "whole corpus, which is balanced corpus-wide and leaves each speaker "
            "with an offset of their own for LDA and PLDA to absorb as "
            "between-speaker variance. Every model trained before this flag "
            "existed used 'global', which is the only reason it is still here."
        ),
    )
    parser.add_argument("--recording-seconds", type=float, default=30.0)
    parser.add_argument("--recordings-per-session", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--max-train-recordings", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--background-corpus",
        type=Path,
        default=None,
        help=(
            "directory of unlabelled .wav audio for the UBM and total "
            "variability matrix. Those stages are unsupervised, so this material "
            "needs no speaker labels — which is what allows a corpus matched to "
            "the deployment population to train the front-end while the "
            "back-end takes its speaker subspace from --corpus. Any speaker id "
            "on this audio is ignored, and it never reaches LDA or PLDA."
        ),
    )
    parser.add_argument(
        "--background-recording-budget",
        type=int,
        default=None,
        help=(
            "cap on background recordings contributing statistics to the total "
            "variability matrix. Must exceed --rank."
        ),
    )
    parser.add_argument(
        "--max-background-recordings",
        type=int,
        default=None,
        help="cap on background recordings scanned at all.",
    )
    arguments = parser.parse_args(argv)

    corpora = arguments.corpus or [Path("data/corpus/librispeech")]
    print(f"scanning corpus at {', '.join(str(root) for root in corpora)}", flush=True)
    plans = scan_corpora(
        corpora,
        target_seconds=arguments.recording_seconds,
        max_recordings_per_session=arguments.recordings_per_session,
        max_speakers=arguments.max_speakers,
    )
    print(
        f"  {len(plans)} recordings from {len({p.speaker_id for p in plans})} speakers",
        flush=True,
    )

    plan_split = split_by_speaker(plans)
    write_split_manifest(plan_split, Path("data/reports/split.json"))
    print(f"  split: {json.dumps(plan_split.summary())}", flush=True)

    train_plans = plan_split.train
    if arguments.max_train_recordings is not None:
        train_plans = train_plans[: arguments.max_train_recordings]

    # Only the training partition is read. Development and evaluation audio
    # belongs to the evaluation run and is deliberately never loaded here, which
    # both saves the memory and makes it structurally impossible for this script
    # to touch the speakers its model will be judged on.
    training = materialise(train_plans)
    print(f"  materialised {len(training)} training recordings", flush=True)

    background: LazyBackgroundCorpus | None = None
    if arguments.background_corpus is not None:
        background_plans = scan_unlabelled(
            arguments.background_corpus,
            max_files=arguments.max_background_recordings,
        )
        print(
            f"  background: {len(background_plans)} unlabelled recordings from "
            f"{arguments.background_corpus}",
            flush=True,
        )
        # Not materialised. The pool is read twice during training and each
        # recording is loaded and degraded as it is reached, so a corpus larger
        # than memory costs I/O rather than failing.
        background = LazyBackgroundCorpus(
            background_plans, TRAINING_CONDITIONS, seed=DEGRADATION_SEED
        )

    config = build_config(
        n_components=arguments.components,
        rank=arguments.rank,
        lda_dimension=arguments.lda_dimension,
        ubm_frame_budget=arguments.ubm_frame_budget,
        min_speakers=arguments.min_speakers,
        background_recording_budget=arguments.background_recording_budget,
        cmvn_frames=arguments.cmvn_frames,
    )
    print(
        f"  front-end: CMVN window "
        f"{config.front_end.sliding_cmvn_frames or 'utterance-level'}",
        flush=True,
    )
    report = train(
        training,
        config,
        arguments.output,
        plan_split.summary(),
        workers=arguments.workers,
        background=background,
        condition_allocation=arguments.condition_allocation,
    )

    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2))
    print(f"wrote {arguments.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
