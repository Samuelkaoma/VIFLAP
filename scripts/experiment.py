"""Shared apparatus for the acoustic experiments.

Holds the parts that training and evaluation both need, so that the two scripts
cannot drift apart in how they degrade a signal or form a trial. A difference
between the channel a model was trained through and the channel it is evaluated
through is invisible in the output and changes the result by more than most of
the effects being measured.

The expensive step, by a wide margin, is the codec: the parametric CELP model
costs roughly a quarter of a second per second of audio, because an
analysis-by-synthesis coder is a sequential search and there is no way to
express it as one array operation. Two consequences are designed around here.

**Degrade once per channel condition, truncate afterwards.** Duration is a
factor in the sweep, and the naive implementation encodes the signal separately
at every duration. Encoding the full-length signal once and truncating the
*decoded* output gives the same thing — the coder is causal, so the first ten
seconds of the decoded 30-second signal are what the decoded 10-second signal
would have been — and makes duration nearly free instead of tripling the cost.

**Work is distributed across processes, not threads.** The inner loops are
Python-level, so they hold the interpreter lock and threads buy nothing.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.corpus import Recording, RecordingPlan, materialise
from viflap.analysis.channel.codec import Codec, CodecMode, resolve_codec
from viflap.analysis.channel.degradation import DegradationCondition, apply_condition
from viflap.analysis.speaker.pipeline import (
    AcousticEmbedding,
    SpeakerComparisonSystem,
    TrainingRecording,
)
from viflap.domain.errors import InsufficientDataError

__all__ = [
    "DegradedRecording",
    "EmbeddingFailure",
    "Trial",
    "TrialSet",
    "build_trials",
    "degrade_many",
    "embed_many",
    "worker_count",
]


def worker_count(requested: int | None = None) -> int:
    """Processes to use, leaving one core for the rest of the machine."""
    if requested is not None and requested > 0:
        return requested
    return max(1, (os.cpu_count() or 2) - 1)


#: Thread limits exported to worker processes before they import numpy.
#:
#: Without these, every worker's BLAS opens a thread pool sized to the whole
#: machine, and seven workers on eight cores contend for those cores badly
#: enough to cost more than the parallelism gains. The work here is already
#: parallel at the recording level, which is the coarser and more efficient
#: place to split it, so each worker should be single-threaded.
_THREAD_LIMIT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _limit_worker_threads() -> None:
    """Set thread limits in this process so spawned children inherit them.

    Set in the parent rather than in the pool initialiser because the initialiser
    runs after the child has already imported numpy, by which point the thread
    pool is fixed. A spawned child inherits the environment and imports numpy
    fresh, so the limit takes effect.
    """
    for name in _THREAD_LIMIT_VARIABLES:
        os.environ.setdefault(name, "1")


def _stable_seed(identifier: str) -> int:
    """A per-recording seed offset that is the same in every interpreter."""
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0xFFFF


@dataclass(frozen=True, slots=True)
class DegradedRecording:
    """A recording after passage through a channel condition."""

    signal: NDArray[np.float64]
    sample_rate: int
    speaker_id: str
    session_id: str
    recording_id: str
    condition_label: str
    codec_mode: str
    """``CodecMode`` value. Carried per recording rather than per run, so that a
    result assembled from a run in which ffmpeg appeared or vanished part-way
    through is detectable rather than silently pooled."""

    def truncated(self, seconds: float | None) -> DegradedRecording:
        """The same recording restricted to its first ``seconds``."""
        if seconds is None:
            return self
        keep = int(round(seconds * self.sample_rate))
        if keep >= self.signal.size:
            return self
        return DegradedRecording(
            signal=self.signal[:keep],
            sample_rate=self.sample_rate,
            speaker_id=self.speaker_id,
            session_id=self.session_id,
            recording_id=self.recording_id,
            condition_label=self.condition_label,
            codec_mode=self.codec_mode,
        )


@dataclass(frozen=True, slots=True)
class EmbeddingFailure:
    """A recording the front-end refused, and why.

    Counted and reported rather than dropped. At short durations the refusal
    rate *is* the result: a system that declines half its trials at five seconds
    has a property worth stating, and a metric computed over only the surviving
    half would conceal it.
    """

    recording_id: str
    speaker_id: str
    reason: str


# -- Degradation ---------------------------------------------------------

_CODEC: dict[str, Codec] = {}


def _init_codec(seed: int) -> None:
    _CODEC["codec"] = resolve_codec(seed=seed)


def _degrade_one(
    payload: tuple[Recording, DegradationCondition, int],
) -> DegradedRecording:
    recording, condition, seed = payload
    codec = _CODEC["codec"]
    result = apply_condition(
        recording.signal,
        recording.sample_rate,
        condition,
        codec=codec,
        rng=np.random.default_rng(seed),
    )
    return DegradedRecording(
        signal=result.signal,
        sample_rate=result.sample_rate,
        speaker_id=recording.speaker_id,
        session_id=recording.session_id,
        recording_id=recording.recording_id,
        condition_label=condition.label,
        codec_mode=result.mode.value,
    )


def degrade_many(
    recordings: Sequence[Recording],
    conditions: Sequence[DegradationCondition],
    *,
    seed: int = 0,
    workers: int | None = None,
) -> list[DegradedRecording]:
    """Apply conditions to recordings in parallel.

    ``conditions`` is either one condition applied to every recording, or a
    per-recording sequence of the same length — which is how multi-condition
    training assigns a different channel to each training recording.
    """
    if len(conditions) == 1:
        paired = [(r, conditions[0]) for r in recordings]
    elif len(conditions) == len(recordings):
        paired = list(zip(recordings, conditions, strict=True))
    else:
        raise ValueError(
            "conditions must be a single condition or one per recording, "
            f"got {len(conditions)} for {len(recordings)} recordings"
        )

    # The seed is derived from the recording identity rather than drawn from a
    # shared stream, so that the noise added to a given recording under a given
    # condition is the same however the work happened to be scheduled. Without
    # it, two runs of the same experiment differ.
    #
    # A content hash, not the built-in `hash`: string hashing is salted per
    # interpreter, so `hash` would give the same recording a different noise
    # realisation on every run and quietly break the reproducibility this is
    # here to provide.
    payloads = [
        (recording, condition, seed + _stable_seed(recording.recording_id))
        for recording, condition in paired
    ]

    n_workers = worker_count(workers)
    if n_workers == 1:
        _init_codec(seed)
        return [_degrade_one(p) for p in payloads]

    _limit_worker_threads()
    # chunksize 1: recordings cost a few seconds each and vary in length, so
    # handing out larger chunks up front leaves workers idle at the end of the
    # run waiting for whichever one drew the slowest chunk.
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_codec, initargs=(seed,)
    ) as pool:
        return list(pool.map(_degrade_one, payloads, chunksize=1))


class LazyBackgroundCorpus(Sequence[TrainingRecording]):
    """Background recordings read and degraded one at a time, on access.

    The unsupervised stages take a corpus far larger than the labelled one —
    hundreds of hours against tens — and
    :meth:`SpeakerComparisonSystem.train` traverses it twice without retaining
    audio, so that its size does not become the peak memory of a training run.
    Handing it a materialised list would throw that away. This loads each
    recording when it is asked for and lets the caller drop it.

    The material is degraded through the same channel as the labelled corpus,
    which is not optional. A UBM estimated on clean audio and a labelled set
    estimated on codec-passed audio describe different distributions, and the
    front-end would be modelling a channel the system never encounters.

    Conditions cycle across the pool rather than one being applied throughout,
    so the background spans the same channels as the labelled set. Degradation
    is seeded from the recording identity, so both traversals yield identical
    audio and the trained model stays reproducible.
    """

    def __init__(
        self,
        plans: Sequence[RecordingPlan],
        conditions: Sequence[DegradationCondition],
        *,
        seed: int = 0,
    ) -> None:
        if not conditions:
            raise ValueError("at least one degradation condition is required")
        self._plans = list(plans)
        self._conditions = list(conditions)
        self._seed = seed

    def __len__(self) -> int:
        return len(self._plans)

    def __getitem__(self, index: int) -> TrainingRecording:  # type: ignore[override]
        if isinstance(index, slice):
            raise TypeError(
                "slicing a lazy background corpus would read every recording it "
                "spans; index it or iterate it instead"
            )

        plan = self._plans[index]
        materialised = materialise([plan])
        if not materialised:
            # materialise drops a plan whose audio is shorter than its header
            # promised. Silent here would mean a hole in the sequence.
            raise RuntimeError(
                f"audio for {plan.recording_id} is shorter than its plan; the "
                f"file changed after the corpus was scanned"
            )

        condition = self._conditions[index % len(self._conditions)]
        degraded = degrade_many(materialised, [condition], seed=self._seed, workers=1)[0]
        return TrainingRecording(
            signal=degraded.signal,
            sample_rate=degraded.sample_rate,
            speaker_id=degraded.speaker_id,
            recording_id=degraded.recording_id,
        )


# -- Embedding -----------------------------------------------------------

_SYSTEM: dict[str, SpeakerComparisonSystem] = {}


def _init_system(model_path: str) -> None:
    _SYSTEM["system"] = SpeakerComparisonSystem.load(model_path)


def _embed_one(
    recording: DegradedRecording,
) -> tuple[DegradedRecording, AcousticEmbedding | None, str]:
    system = _SYSTEM["system"]
    try:
        embedding = system.embed(recording.signal, recording.sample_rate)
    except InsufficientDataError as exc:
        return recording, None, exc.message
    return recording, embedding, ""


def embed_many(
    recordings: Sequence[DegradedRecording],
    model_path: Path | str,
    *,
    workers: int | None = None,
) -> tuple[list[tuple[DegradedRecording, AcousticEmbedding]], list[EmbeddingFailure]]:
    """Embed degraded recordings, separating refusals from results."""
    n_workers = worker_count(workers)
    if n_workers == 1:
        _init_system(str(model_path))
        outcomes = [_embed_one(r) for r in recordings]
    else:
        _limit_worker_threads()
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_system, initargs=(str(model_path),)
        ) as pool:
            outcomes = list(pool.map(_embed_one, recordings, chunksize=4))

    embedded: list[tuple[DegradedRecording, AcousticEmbedding]] = []
    failures: list[EmbeddingFailure] = []
    for recording, embedding, reason in outcomes:
        if embedding is None:
            failures.append(
                EmbeddingFailure(
                    recording_id=recording.recording_id,
                    speaker_id=recording.speaker_id,
                    reason=reason,
                )
            )
        else:
            embedded.append((recording, embedding))
    return embedded, failures


# -- Trials --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Trial:
    """One comparison of two recordings."""

    first: str
    second: str
    is_same_source: bool
    speaker: str
    """The resampling unit. For a same-source trial this is the shared speaker;
    for a different-source trial it is one of the two, chosen by
    :func:`_different_source_owner`. Either way a trial belongs to exactly one
    resampling unit, which is what the speaker bootstrap requires."""


def _pair_key(first_id: str, second_id: str) -> tuple[str, str]:
    """The identity of a trial: its two recordings, order removed.

    Sorted for the same reason ``_different_source_owner`` hashes an unordered
    pair — which recording is "first" is an artefact of enumeration order, and
    two systems that both iterate ``i < j`` over differently ordered inputs would
    otherwise produce keys that never match. This is the join key that lets a
    paired comparison be built between two scripts' outputs, so if it depended on
    ordering the pairing would fail silently, aligning unrelated trials rather
    than raising.
    """
    return (first_id, second_id) if first_id <= second_id else (second_id, first_id)


def _different_source_owner(first: str, second: str, first_id: str, second_id: str) -> str:
    """Which of two speakers a different-source trial is attributed to.

    A different-source trial genuinely belongs to two speakers, and the cluster
    bootstrap needs it to belong to one. Attributing it to the *first*
    recording's speaker — the obvious choice, and the one this used — is not
    neutral, because the recordings arrive grouped by speaker and sorted by
    identifier. Under that rule the first speaker in the sort owns a trial
    against every later recording in the set and the last speaker owns almost
    none. Measured on ``amr12.2_clean`` at 30 s: 1,314 trials for the heaviest
    speaker against 16 for the lightest, and a Kish effective sample size of 31
    against a nominal 42.

    That skew is a defect rather than a nuisance. The bootstrap draws speakers
    as exchangeable units, and units carrying a hundredfold difference in
    influence are not exchangeable — so the resampling distribution is driven by
    whether a handful of early speakers were drawn.

    The owner is therefore chosen by hashing the two recording identifiers as an
    **unordered** pair, which makes the choice independent of which side of the
    comparison a recording happened to fall on, splits each speaker pair's
    trials roughly evenly between the two, and is stable across interpreters and
    runs. ``hashlib`` rather than the built-in ``hash``, which is salted per
    process and would give a different attribution — and therefore a different
    interval — on every run.

    This changes no point estimate. Ownership enters only the resampling, so
    every ``C_llr``, ``C_llr_min`` and EER computed on a full trial set is
    exactly what it was; the intervals around them change.
    """
    # Both the hash key and the choice it indexes are built from the pair
    # *sorted*, so the result names a speaker rather than a side. Hashing the
    # sorted key but indexing the arguments in the order given would still pick
    # by position, which is the defect this exists to remove.
    ordered = sorted(((first_id, first), (second_id, second)))
    digest = hashlib.sha256("|".join(name for name, _ in ordered).encode("utf-8")).digest()
    return ordered[digest[0] & 1][1]


@dataclass(frozen=True, slots=True)
class TrialSet:
    """Scored trials with everything needed to compute an interval."""

    scores: NDArray[np.float64]
    labels: NDArray[np.int64]
    speakers: tuple[str, ...]

    pairs: tuple[tuple[str, str], ...] = ()
    """The two recording identifiers behind each trial, in sorted order.

    Carried so that two systems scoring the same corpus can be **joined trial
    for trial** rather than compared through their separate intervals.
    ``paired_bootstrap_over_speakers`` requires vectors aligned trial for trial
    and says so, but nothing until now recorded what a trial *was* — the row
    index depends on the order recordings came off the scan, and two scripts
    that both iterate ``i < j`` still produce different orderings if either
    system refused a different subset. Sorted, so the key does not depend on
    which recording happened to be enumerated first.

    Defaulted to empty because ``compare_capacity`` and ``compare_cmvn`` build
    their own trial sets and do not need it.
    """

    @property
    def n_same_source(self) -> int:
        return int(np.count_nonzero(self.labels == 1))

    @property
    def n_different_source(self) -> int:
        return int(np.count_nonzero(self.labels == 0))

    @property
    def n_speakers(self) -> int:
        return len(set(self.speakers))

    @property
    def kish_effective_sample_size(self) -> float:
        """Effective number of resampling units, given how unevenly they own trials.

        ``(sum n_i)^2 / sum n_i^2`` over the trials each speaker owns. It equals
        the speaker count when every speaker owns equally and falls toward one
        as a single speaker comes to own everything, so it is the direct measure
        of whether the units the bootstrap treats as exchangeable actually are.

        Worth reporting beside the speaker count rather than instead of it: a
        cell with 42 speakers and an effective 31 has a wider true interval than
        its nominal count suggests, and nothing else in the output shows that.
        """
        if not self.speakers:
            return 0.0
        counts = np.array(list(Counter(self.speakers).values()), dtype=np.float64)
        return float(counts.sum() ** 2 / np.sum(counts**2))


def build_trials(
    embedded: Sequence[tuple[DegradedRecording, AcousticEmbedding]],
    system: SpeakerComparisonSystem,
    *,
    cross_session_only: bool = True,
    max_different_source_per_recording: int | None = None,
    seed: int = 0,
) -> TrialSet:
    """Score every admissible pair.

    ``cross_session_only`` excludes same-speaker pairs drawn from one session,
    and it is on by default because leaving them in measures the wrong thing.
    Two recordings from one session share a microphone, a room and a voice on
    one day; a system scores them alike partly for reasons that have nothing to
    do with the speaker. The operational question is whether two *separate calls*
    can be linked, so a same-source trial is a pair of separate sessions.
    """
    if len(embedded) < 2:
        raise InsufficientDataError(
            "at least two embedded recordings are needed to form a trial",
            n_embedded=len(embedded),
        )

    matrix = np.stack([embedding.vector for _, embedding in embedded])
    metadata = [recording for recording, _ in embedded]
    rng = np.random.default_rng(seed)

    scores: list[float] = []
    labels: list[int] = []
    speakers: list[str] = []
    pairs: list[tuple[str, str]] = []

    for index, probe in enumerate(metadata):
        if index + 1 >= len(metadata):
            break
        # One vectorised call against every later recording, rather than a
        # pairwise loop: the whole point of the diagonalised PLDA form.
        row = system.plda.score_many(matrix[index], matrix[index + 1 :])
        candidates = metadata[index + 1 :]

        different_indices: list[int] = []
        for offset, candidate in enumerate(candidates):
            same_speaker = candidate.speaker_id == probe.speaker_id
            if same_speaker:
                if cross_session_only and candidate.session_id == probe.session_id:
                    continue
                scores.append(float(row[offset]))
                labels.append(1)
                speakers.append(probe.speaker_id)
                pairs.append(_pair_key(probe.recording_id, candidate.recording_id))
            else:
                different_indices.append(offset)

        if (
            max_different_source_per_recording is not None
            and len(different_indices) > max_different_source_per_recording
        ):
            chosen = rng.choice(
                len(different_indices),
                size=max_different_source_per_recording,
                replace=False,
            )
            different_indices = [different_indices[int(i)] for i in chosen]

        for offset in different_indices:
            candidate = candidates[offset]
            scores.append(float(row[offset]))
            labels.append(0)
            speakers.append(
                _different_source_owner(
                    probe.speaker_id,
                    candidate.speaker_id,
                    probe.recording_id,
                    candidate.recording_id,
                )
            )
            pairs.append(_pair_key(probe.recording_id, candidate.recording_id))

    return TrialSet(
        scores=np.array(scores, dtype=np.float64),
        labels=np.array(labels, dtype=np.int64),
        speakers=tuple(speakers),
        pairs=tuple(pairs),
    )


def summarise_codec_modes(recordings: Iterable[DegradedRecording]) -> dict[str, Any]:
    """Confirm one codec produced everything, and say which.

    Results obtained through the reference coder and through the parametric
    model are different quantities and must never be pooled. This is the check
    that makes pooling detectable rather than a matter of remembering.
    """
    modes = sorted({recording.codec_mode for recording in recordings})
    return {
        "codec_modes": modes,
        "is_single_mode": len(modes) == 1,
        "is_reference_codec": modes == [CodecMode.REFERENCE_AMR.value],
    }
