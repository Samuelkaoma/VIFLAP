"""Turning a read-speech corpus into something shaped like telephony evidence.

LibriSpeech is not telephony. It is 16 kHz read audiobook speech recorded over a
good microphone by volunteers who were not trying to defraud anyone. Every
number derived from it inherits that, and the report says so. What it does
provide, and what is genuinely hard to obtain otherwise, is a few hundred
speakers each recorded in several distinct sessions — which is the minimum
structure any speaker-comparison system needs in order to be trained and
evaluated honestly.

Three decisions here shape everything computed downstream.

**A chapter is a session.** LibriSpeech chapters are separate recording sessions
for the same reader: different day, different room state, often a different
microphone gain. Treating the chapter as the session unit means within-speaker
variation in the training data includes between-session variation, which is what
PLDA's within-class covariance is supposed to model. Building recordings that
never cross a chapter boundary, and splitting so that a speaker's sessions stay
together, keeps that structure intact.

**A recording is a concatenation, not an utterance.** A single LibriSpeech
utterance averages 12 seconds, of which perhaps 9 survive voice activity
detection. An i-vector from 9 seconds is dominated by its prior. Concatenating
consecutive utterances from one session into a fixed-length recording produces
something with the net speech of a short call, and the pauses between utterances
are a reasonable stand-in for conversational silence — which the VAD must handle
anyway.

**Amplitude is normalised per recording, before anything else.** Readers differ
in recording level by more than 20 dB, and level is a property of the volunteer's
microphone, not their vocal tract. Left alone it is a strong, entirely spurious
speaker cue, and one the codec and the SNR calculation would both then inherit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Protocol, TypeVar

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from viflap.domain.errors import InsufficientDataError, InvalidEvidenceError

__all__ = [
    "CorpusSplit",
    "Recording",
    "RecordingPlan",
    "load_corpus",
    "materialise",
    "scan_corpora",
    "scan_corpus",
    "split_by_speaker",
]

#: Audio is held as float32 rather than float64.
#:
#: The source is 16-bit FLAC, so float32's 24-bit mantissa represents it
#: exactly; float64 stores the same values in twice the memory. The corpus is
#: several gigabytes either way and the difference decides whether a run fits in
#: memory or pages — and a paging run is not slow at the margin, it is an order
#: of magnitude slower, because the worker processes spend their time waiting
#: rather than computing. Everything downstream converts to float64 on entry.
STORAGE_DTYPE = np.float32


class _HasSpeaker(Protocol):
    @property
    def speaker_id(self) -> str: ...


ItemT = TypeVar("ItemT", bound=_HasSpeaker)


@dataclass(frozen=True, slots=True)
class Recording:
    """One constructed recording, with the provenance needed to split on it."""

    signal: NDArray[np.float64]
    sample_rate: int
    speaker_id: str
    session_id: str
    """The chapter. Distinct sessions of one speaker are what within-speaker
    covariance is estimated from."""

    recording_id: str
    source_utterances: tuple[str, ...]

    @property
    def duration_seconds(self) -> float:
        return self.signal.size / self.sample_rate


def _normalise(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale to a fixed peak, leaving silence alone.

    Peak rather than RMS: RMS normalisation over a recording whose pause
    fraction varies would change the level of the *speech* according to how much
    silence surrounded it, which is a worse artefact than the one being removed.
    """
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak <= 0.0:
        return signal
    return signal * (0.7 / peak)


@dataclass(frozen=True, slots=True)
class RecordingPlan:
    """A recording that has been decided on but not yet read.

    Carries only what the split needs — who spoke, in which session — plus the
    files the audio will come from. Deciding the whole corpus this way and then
    reading only the partitions a run actually uses is the difference between
    holding four gigabytes of audio and holding one.
    """

    speaker_id: str
    session_id: str
    recording_id: str
    sources: tuple[Path, ...]
    sample_rate: int
    target_samples: int


def scan_corpus(
    root: Path,
    target_seconds: float = 30.0,
    max_recordings_per_session: int = 2,
    max_speakers: int | None = None,
    min_sessions_per_speaker: int = 2,
) -> list[RecordingPlan]:
    """Decide the recordings without reading any audio.

    Durations come from the FLAC headers via :func:`soundfile.info`, which reads
    a few dozen bytes per file rather than decoding it. The grouping is
    identical to what reading everything would produce.

    Speakers with fewer than ``min_sessions_per_speaker`` usable sessions are
    dropped. They would still be usable for the UBM, which is unsupervised, but
    mixing them in means the set of speakers behind the between-speaker term is
    not the set behind the within-speaker term, and the ratio of the two is
    exactly what a PLDA likelihood ratio is.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"corpus root {root} does not exist; run scripts/fetch_corpus.py first"
        )

    plans: list[RecordingPlan] = []
    speakers = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if max_speakers is not None:
        speakers = speakers[:max_speakers]

    for speaker_directory in speakers:
        speaker_id = speaker_directory.name
        per_speaker: list[RecordingPlan] = []
        sessions_used = 0

        for session_directory in sorted(
            (p for p in speaker_directory.iterdir() if p.is_dir()),
            key=lambda p: p.name,
        ):
            files = sorted(session_directory.glob("*.flac"), key=lambda p: p.name)
            if not files:
                continue

            built = _plan_session(
                files,
                speaker_id,
                session_directory.name,
                target_seconds,
                max_recordings_per_session,
            )
            if built:
                sessions_used += 1
                per_speaker.extend(built)

        if sessions_used >= min_sessions_per_speaker:
            plans.extend(per_speaker)

    return plans


def scan_corpora(
    roots: Sequence[Path],
    target_seconds: float = 30.0,
    max_recordings_per_session: int = 2,
    max_speakers: int | None = None,
    min_sessions_per_speaker: int = 2,
) -> list[RecordingPlan]:
    """Scan several corpus roots as one corpus, refusing to merge identities.

    Corpora arrive in partitions — LibriSpeech ships ``train-clean-100`` and
    ``train-clean-360`` separately — and the speaker count is what bounds this
    system, so pooling partitions is the cheapest way to raise the ceiling.

    The collision check is the reason this is a function rather than a list
    concatenation at the call site. Speaker identifiers here are bare numbers,
    unique only within the partition that issued them, and two partitions that
    reuse one are not hypothetical. Merging them would place two different
    people under one label, which produces same-source trials between strangers,
    inflates the within-speaker covariance PLDA is estimated from, and is
    undetectable downstream: every count still looks plausible, and
    ``verify_disjoint`` still passes because the identifier really does appear on
    one side of the split only.

    ``max_speakers`` applies to the pool rather than to each root, so the cap
    means what it says regardless of how the material was partitioned.
    """
    if not roots:
        raise InsufficientDataError("no corpus roots supplied")

    plans: list[RecordingPlan] = []
    seen: dict[str, Path] = {}
    for root in roots:
        root = Path(root)
        scanned = scan_corpus(
            root,
            target_seconds=target_seconds,
            max_recordings_per_session=max_recordings_per_session,
            max_speakers=None,
            min_sessions_per_speaker=min_sessions_per_speaker,
        )
        for speaker in {plan.speaker_id for plan in scanned}:
            if speaker in seen:
                raise InvalidEvidenceError(
                    "the same speaker identifier appears in two corpus roots; "
                    "pooling them would merge two people into one speaker",
                    speaker_id=speaker,
                    first_root=str(seen[speaker]),
                    second_root=str(root),
                )
            seen[speaker] = root
        plans.extend(scanned)

    if max_speakers is not None:
        keep = set(sorted({plan.speaker_id for plan in plans})[:max_speakers])
        plans = [plan for plan in plans if plan.speaker_id in keep]

    return plans


def _plan_session(
    files: Sequence[Path],
    speaker_id: str,
    session_id: str,
    target_seconds: float,
    max_per_session: int,
) -> list[RecordingPlan]:
    """Group a session's files into fixed-length recordings, by header alone."""
    plans: list[RecordingPlan] = []
    group: list[Path] = []
    held = 0
    sample_rate = 0
    target_samples = 0

    for path in files:
        try:
            info = sf.info(path)
        except Exception:
            # Unreadable, truncated, or still being written. Skipping is the
            # documented posture for the rest of this pipeline — operational
            # corpora always contain some unusable files — and it also makes a
            # scan safe to run against a corpus that is still downloading.
            continue
        if sample_rate and info.samplerate != sample_rate:
            continue
        sample_rate = info.samplerate
        target_samples = int(round(target_seconds * sample_rate))

        group.append(path)
        held += info.frames
        if held < target_samples:
            continue

        plans.append(
            RecordingPlan(
                speaker_id=speaker_id,
                session_id=session_id,
                recording_id=f"{speaker_id}-{session_id}-r{len(plans):02d}",
                sources=tuple(group),
                sample_rate=sample_rate,
                target_samples=target_samples,
            )
        )
        group, held = [], 0
        if len(plans) >= max_per_session:
            break
    return plans


def materialise(plans: Sequence[RecordingPlan]) -> list[Recording]:
    """Read the audio for the given plans and nothing else."""
    recordings: list[Recording] = []
    for plan in plans:
        pieces: list[NDArray[np.float64]] = []
        for path in plan.sources:
            audio, _ = sf.read(path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            pieces.append(audio)

        joined = np.concatenate(pieces)[: plan.target_samples]
        if joined.size < plan.target_samples:
            # The header-based plan said this group was long enough. If the
            # audio disagrees the file changed under us, and a short recording
            # would silently contribute a different duration condition from the
            # one its label claims.
            continue

        recordings.append(
            Recording(
                signal=_normalise(joined),
                sample_rate=plan.sample_rate,
                speaker_id=plan.speaker_id,
                session_id=plan.session_id,
                recording_id=plan.recording_id,
                source_utterances=tuple(p.stem for p in plan.sources),
            )
        )
    return recordings


def load_corpus(
    root: Path,
    target_seconds: float = 30.0,
    max_recordings_per_session: int = 2,
    max_speakers: int | None = None,
    min_sessions_per_speaker: int = 2,
) -> list[Recording]:
    """Scan and materialise the whole corpus.

    Convenience for a run that genuinely needs every partition in memory at
    once. A run that needs two of the three should scan, split, and materialise
    the partitions it uses.
    """
    return materialise(
        scan_corpus(
            root,
            target_seconds=target_seconds,
            max_recordings_per_session=max_recordings_per_session,
            max_speakers=max_speakers,
            min_sessions_per_speaker=min_sessions_per_speaker,
        )
    )


@dataclass(frozen=True, slots=True)
class CorpusSplit(Generic[ItemT]):
    """A three-way speaker-disjoint partition.

    Three parts rather than two, because two is not enough. The calibrator is a
    fitted model like any other: fitting it on the evaluation speakers and then
    reporting ``C_llr`` on those same speakers reports how well a calibration
    describes the data it was fitted to, which is not the quantity of interest
    and is always better than the truth.

    Generic over what it holds, so the same split can be computed over plans —
    which carry no audio — and only the needed partitions then materialised.
    """

    train: tuple[ItemT, ...]
    development: tuple[ItemT, ...]
    evaluation: tuple[ItemT, ...]

    def summary(self) -> dict[str, int]:
        return {
            "train_recordings": len(self.train),
            "train_speakers": len({r.speaker_id for r in self.train}),
            "development_recordings": len(self.development),
            "development_speakers": len({r.speaker_id for r in self.development}),
            "evaluation_recordings": len(self.evaluation),
            "evaluation_speakers": len({r.speaker_id for r in self.evaluation}),
        }


def split_by_speaker(
    recordings: Sequence[ItemT],
    development_fraction: float = 0.2,
    evaluation_fraction: float = 0.2,
    seed: int = 20250601,
) -> CorpusSplit[ItemT]:
    """Partition recordings three ways with no speaker in more than one part.

    Speakers are ordered by a seeded hash of their identifier rather than by
    name. LibriSpeech speaker identifiers are assigned in a way that correlates
    with when the reader joined the project, and taking a contiguous block would
    select a cohort rather than a sample.

    Accepts plans or loaded recordings alike: only ``speaker_id`` is consulted,
    and computing the split before reading any audio is what allows a run that
    needs two partitions to avoid loading the third.
    """
    by_speaker: dict[str, list[ItemT]] = {}
    for recording in recordings:
        by_speaker.setdefault(recording.speaker_id, []).append(recording)

    rng = np.random.default_rng(seed)
    speakers = sorted(by_speaker)
    order = rng.permutation(len(speakers))
    shuffled = [speakers[int(i)] for i in order]

    n_total = len(shuffled)
    n_eval = int(round(n_total * evaluation_fraction))
    n_dev = int(round(n_total * development_fraction))
    if n_eval < 3 or n_dev < 3 or n_total - n_eval - n_dev < 1:
        raise ValueError(
            f"corpus has too few speakers ({n_total}) to form a three-way "
            f"speaker-disjoint split with usable development and evaluation parts"
        )

    eval_speakers = set(shuffled[:n_eval])
    dev_speakers = set(shuffled[n_eval : n_eval + n_dev])
    train_speakers = set(shuffled[n_eval + n_dev :])

    overlaps = (
        eval_speakers & dev_speakers,
        eval_speakers & train_speakers,
        dev_speakers & train_speakers,
    )
    if any(overlaps):
        raise AssertionError("speaker leakage while constructing the split")

    def gather(chosen: set[str]) -> tuple[ItemT, ...]:
        return tuple(
            recording for speaker in sorted(chosen) for recording in by_speaker[speaker]
        )

    return CorpusSplit(
        train=gather(train_speakers),
        development=gather(dev_speakers),
        evaluation=gather(eval_speakers),
    )


def write_split_manifest(split: CorpusSplit, path: Path) -> None:
    """Record which speakers went where, so a later run can be checked.

    Without this, a re-run with a different corpus subset silently produces a
    different split, and two results that look comparable are not.
    """
    payload = {
        "summary": split.summary(),
        "train_speakers": sorted({r.speaker_id for r in split.train}),
        "development_speakers": sorted({r.speaker_id for r in split.development}),
        "evaluation_speakers": sorted({r.speaker_id for r in split.evaluation}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
