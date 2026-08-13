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


def assign_conditions(
    n_recordings: int, seed: int = DEGRADATION_SEED
) -> list[DegradationCondition]:
    """Give each training recording a channel condition.

    Assigned by cycling a permuted list rather than by independent draws, so the
    conditions are exactly balanced. Independent draws leave one condition
    over-represented by chance, and since condition is confounded with whichever
    speakers received it, that imbalance becomes a speaker effect.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_recordings)
    conditions = list(TRAINING_CONDITIONS)
    return [conditions[int(position) % len(conditions)] for position in order]


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
    conditions = assign_conditions(len(training), seed=seed)

    print(
        f"degrading {len(training)} training recordings across "
        f"{len(TRAINING_CONDITIONS)} channel conditions on "
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
    )

    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2))
    print(f"wrote {arguments.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
