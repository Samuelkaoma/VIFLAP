"""Scanners for the Zambian corpora, and the speaker identity they do or do not carry.

Three corpora cover the target population, and only one of them can be used for
anything supervised. The distinction is not a detail of ingestion — it decides
which stage of training each corpus is allowed to feed — so it is enforced here
rather than left to the caller to remember.

**BembaSpeech** (also released as the Bemba partition of Zambezi Voice) ships a
speaker roster and encodes the speaker in the filename, ``01-200921-192247_bem_
d31_elicit_16.wav`` giving speaker ``01`` and session ``200921-192247``. Parsing
the prefix reproduces the roster's per-speaker utterance counts exactly for all
seventeen speakers, so the identity is real and checkable — :func:`scan_labelled`
checks it.

**Zambezi Voice's Nyanja, Tonga and Lozi** partitions carry no speaker identity
at all. Their filenames begin with the session timestamp rather than a speaker,
``221102-102320_nya_510_elicit_0.wav``, no manifest column names a speaker, and
no roster is published. The paper reports twelve, nine and six speakers from the
authors' records; the release does not expose which recording belongs to whom,
and no field or combination of fields reproduces those counts.

**BIG-C** has a ``speaker_id`` column that is not a speaker. It takes 74 values,
of which 37 carry both male and female labels in near-equal proportion — it
indexes the participants *within* a conversation and is reused across
conversations by different people.

So the corpora that cannot be labelled are still perfectly good audio, and the
unsupervised stages of training — the UBM and the total variability matrix —
need audio rather than labels. :func:`scan_unlabelled` prepares them for that
and marks them with :data:`UNLABELLED_SPEAKER`, which
:func:`reject_unlabelled` refuses. The failure being guarded against is
specific: feeding unlabelled plans into a labelled split collapses every
recording into one enormous fake speaker, and nothing downstream would complain.
PLDA would train, the run would finish, and the model would be nonsense.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from scripts.corpus import RecordingPlan

__all__ = [
    "UNLABELLED_SPEAKER",
    "RosterEntry",
    "parse_labelled_name",
    "read_speaker_roster",
    "reject_unlabelled",
    "scan_labelled",
    "scan_unlabelled",
]

UNLABELLED_SPEAKER = "<unlabelled>"
"""Speaker id for material with no identity. The angle brackets are deliberate:
no real identifier in any of these corpora can collide with it, so a plan
carrying it can always be told apart from one carrying a speaker."""

#: ``{speaker}-{yymmdd}-{hhmmss}_{lang}_...``. The speaker group is not
#: restricted to two digits, so that a filename whose prefix is a six-digit date
#: still parses — and is then rejected by the roster check rather than silently
#: becoming a speaker called "200630".
_LABELLED_NAME = re.compile(r"^(?P<speaker>[^-]+)-(?P<session>\d{6}-\d{6})_")

_ROSTER_ROW = re.compile(
    r"^\s*\|\s*(?P<id>\w+)\s*\|\s*(?P<sex>[MF])\s*\|\s*(?P<utterances>\d+)\s*\|"
    r"\s*(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)\s*\|"
)


@dataclass(frozen=True, slots=True)
class RosterEntry:
    """One speaker as the corpus itself describes them."""

    speaker_id: str
    sex: str
    utterances: int
    duration_seconds: int


def read_speaker_roster(path: Path) -> dict[str, RosterEntry]:
    """Parse ``speaker_info.txt`` into a speaker table.

    The roster is the authority on who exists, and the second of two gates. The
    first is the filename pattern: the corpus's 1,051 unattributed recordings —
    material added after the roster was written — are named
    ``200701-160335_bem_d16_elicit_40.wav``, session first and no speaker at
    all, so they never parse as speaker-prefixed. The roster catches anything
    that does parse but names someone the corpus does not document.

    The file also carries speakers' first names. They are not read: nothing here
    needs them, and a pipeline that never loads them cannot leak them into a
    report, a log or a serialised model.
    """
    roster: dict[str, RosterEntry] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = _ROSTER_ROW.match(line)
        if match is None:
            continue
        roster[match["id"]] = RosterEntry(
            speaker_id=match["id"],
            sex=match["sex"],
            utterances=int(match["utterances"]),
            duration_seconds=(
                int(match["hours"]) * 3600
                + int(match["minutes"]) * 60
                + int(match["seconds"])
            ),
        )
    if not roster:
        raise ValueError(f"no speaker rows found in {path}; is this a roster file?")
    return roster


def parse_labelled_name(name: str) -> tuple[str, str] | None:
    """Split a BembaSpeech filename into ``(speaker, session)``, or ``None``.

    ``None`` means the name does not carry a speaker. That covers both the
    Zambezi Voice layout, ``221102-102320_nya_510_elicit_0.wav``, and
    BembaSpeech's own unattributed recordings, which use the same session-first
    form — a six-digit date followed by a time is not a speaker followed by a
    session, and the pattern requires the full ``yymmdd-hhmmss`` after the
    speaker for exactly that reason.

    Returning the pair unvalidated when it does parse is deliberate: whether the
    prefix names a real speaker is the roster's call, not the pattern's.
    """
    match = _LABELLED_NAME.match(name)
    if match is None:
        return None
    return match["speaker"], match["session"]


def scan_labelled(
    audio_root: Path,
    roster: dict[str, RosterEntry],
    target_seconds: float = 30.0,
    max_recordings_per_session: int = 2,
    min_sessions_per_speaker: int = 2,
) -> list[RecordingPlan]:
    """Plan labelled recordings from a corpus whose filenames name the speaker.

    The corpus's own ``train``/``dev``/``test`` manifests are ignored, and not
    by oversight. They are split by utterance for speech recognition, and they
    are not speaker-disjoint — one speaker appears in both train and test,
    another in both train and dev. Reusing them would put the same voice on both
    sides of an evaluation. The plans this returns are unsplit; pass them to
    :func:`scripts.corpus.split_by_speaker`.

    Speakers with fewer than ``min_sessions_per_speaker`` sessions are dropped,
    for the reason :func:`scripts.corpus.scan_corpus` drops them: within-speaker
    covariance cannot be estimated from a single session, so a speaker who only
    appears once contributes to the between-speaker term alone and unbalances
    the ratio the PLDA likelihood ratio is made of.
    """
    audio_root = Path(audio_root)
    if not audio_root.exists():
        raise FileNotFoundError(f"audio root {audio_root} does not exist")

    by_session: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(audio_root.glob("*.wav"), key=lambda p: p.name):
        parsed = parse_labelled_name(path.name)
        if parsed is None:
            continue
        speaker, session = parsed
        if speaker not in roster:
            # Anything the roster does not document. The corpus's own
            # unattributed files are already gone by here — they are named
            # session-first and fail to parse — so this catches the case the
            # pattern cannot: a well-formed prefix naming an unknown speaker.
            continue
        by_session.setdefault((speaker, session), []).append(path)

    plans: list[RecordingPlan] = []
    for speaker in sorted({speaker for speaker, _ in by_session}):
        sessions = sorted(session for spk, session in by_session if spk == speaker)
        per_speaker: list[RecordingPlan] = []
        sessions_used = 0

        for session in sessions:
            built = _plan_group(
                by_session[(speaker, session)],
                speaker_id=speaker,
                session_id=session,
                target_seconds=target_seconds,
                max_per_session=max_recordings_per_session,
            )
            if built:
                sessions_used += 1
                per_speaker.extend(built)

        if sessions_used >= min_sessions_per_speaker:
            plans.extend(per_speaker)

    return plans


def scan_unlabelled(
    audio_roots: Path | Sequence[Path],
    max_files: int | None = None,
) -> list[RecordingPlan]:
    """Plan background recordings from audio with no speaker identity.

    One plan per file, at the file's own length. No grouping into fixed
    durations, because nothing that consumes this material cares: the UBM sees
    a frame sample and the total variability matrix sees per-recording
    statistics, and neither is a duration-controlled comparison.

    Every plan is marked :data:`UNLABELLED_SPEAKER`. That is what makes the
    material safe to hold alongside labelled plans — it can be told apart at any
    point, and :func:`reject_unlabelled` refuses it where identity is required.
    """
    roots = [audio_roots] if isinstance(audio_roots, Path) else list(audio_roots)

    plans: list[RecordingPlan] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"audio root {root} does not exist")

        for path in sorted(root.glob("*.wav"), key=lambda p: p.name):
            if max_files is not None and len(plans) >= max_files:
                return plans
            info = sf.info(path)
            plans.append(
                RecordingPlan(
                    speaker_id=UNLABELLED_SPEAKER,
                    session_id=path.stem,
                    recording_id=f"bg-{root.name}-{path.stem}",
                    sources=(path,),
                    sample_rate=info.samplerate,
                    target_samples=int(info.frames),
                )
            )
    return plans


def reject_unlabelled(plans: Iterable[RecordingPlan]) -> list[RecordingPlan]:
    """Return the plans, or raise if any of them lacks a speaker.

    Call this on anything about to be split by speaker or used to fit LDA or
    PLDA. Unlabelled plans all share one sentinel id, so a split would treat
    hundreds of hours of many people as a single speaker with an enormous number
    of sessions. Every downstream stage would accept that quietly — the split
    succeeds, ``verify_disjoint`` passes because the id genuinely appears on one
    side only, PLDA trains, and the model is meaningless.
    """
    plans = list(plans)
    offenders = [p.recording_id for p in plans if p.speaker_id == UNLABELLED_SPEAKER]
    if offenders:
        raise ValueError(
            f"{len(offenders)} plan(s) carry {UNLABELLED_SPEAKER} and have no speaker "
            f"identity; they may train the UBM and total variability matrix but not "
            f"LDA or PLDA. First few: {offenders[:3]}"
        )
    return plans


def _plan_group(
    files: Sequence[Path],
    speaker_id: str,
    session_id: str,
    target_seconds: float,
    max_per_session: int,
) -> list[RecordingPlan]:
    """Group one session's utterances into fixed-length recordings, by header.

    Mirrors :func:`scripts.corpus._plan_session`, which cannot be reused
    directly because it takes its speaker and session from the directory tree
    and these corpora are flat, with both encoded in the filename.
    """
    plans: list[RecordingPlan] = []
    group: list[Path] = []
    held = 0
    sample_rate = 0
    target_samples = 0

    for path in files:
        info = sf.info(path)
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
