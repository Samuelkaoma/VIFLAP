"""Fetch a bounded subset of LibriSpeech by streaming the archive.

The published ``train-clean-100`` archive is 6.4 GB and expands to about the
same again. Neither the whole archive nor the whole expansion is needed: the
i-vector/PLDA stack is trained on a few tens of minutes per speaker at most,
and what governs the quality of the result is the number of *speakers* and the
number of distinct *sessions* per speaker, not the total hours.

So the archive is consumed as a stream and discarded as it goes. ``tarfile``
in stream mode (``r|gz``) reads members sequentially without seeking, which is
exactly what an HTTP response body supports, and members that are not wanted
are skipped rather than written. Peak disk cost is the retained subset.

Two selection rules, both of which matter downstream:

**Cap per chapter, and take several chapters.** A LibriSpeech chapter is one
recording session. A speaker represented by one chapter contributes no
within-speaker between-session variation, and PLDA trained on such a corpus
models session identity as though it were speaker identity — it learns that
same-session means same-speaker, which is true in training and false in
service. Taking from several chapters per speaker is what gives the
within-class covariance something real to estimate.

**Keep every speaker, cap every speaker equally.** An unequal cap would let a
handful of speakers dominate the UBM and the total variability matrix.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"

#: Read speech is 16 kHz in LibriSpeech. Kept as-is on disk; the channel
#: simulation resamples to 8 kHz itself, and doing it once at that point keeps
#: a single resampling stage in the signal path rather than two.
SOURCE_RATE = 16_000


@dataclass(slots=True)
class Selection:
    """How much to keep, per speaker and per session."""

    max_speakers: int | None = None
    chapters_per_speaker: int = 3
    utterances_per_chapter: int = 10

    def __post_init__(self) -> None:
        if self.chapters_per_speaker < 2:
            raise ValueError(
                "at least two chapters per speaker are required; a single-session "
                "speaker contributes no within-speaker between-session variation "
                "and PLDA trained on such a corpus confuses session with speaker"
            )
        if self.utterances_per_chapter < 1:
            raise ValueError("utterances_per_chapter must be positive")


@dataclass(slots=True)
class _Progress:
    """Counters for the streaming pass."""

    bytes_read: int = 0
    members_seen: int = 0
    kept: int = 0
    started: float = field(default_factory=time.monotonic)

    def line(self, total_bytes: int | None) -> str:
        elapsed = max(time.monotonic() - self.started, 1e-9)
        mb = self.bytes_read / 1e6
        rate = mb / elapsed
        if total_bytes:
            pct = 100.0 * self.bytes_read / total_bytes
            return (
                f"  {mb:8.1f} MB ({pct:5.1f}%)  {rate:5.1f} MB/s  kept {self.kept:5d} files"
            )
        return f"  {mb:8.1f} MB  {rate:5.1f} MB/s  kept {self.kept:5d} files"


class _CountingReader:
    """Wrap a stream so bytes consumed can be reported during a long fetch."""

    def __init__(self, stream: object, progress: _Progress) -> None:
        self._stream = stream
        self._progress = progress

    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)  # type: ignore[attr-defined]
        self._progress.bytes_read += len(chunk)
        return chunk

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()


class _ResilientSource:
    """A byte stream over HTTP that reconnects and carries on where it stopped.

    A ``.tar.gz`` cannot be resumed at the archive level: gzip is one continuous
    stream, so decompression cannot restart from an arbitrary offset. It can be
    resumed at the *byte* level, which is enough. Reopening the request with
    ``Range: bytes=N-`` delivers exactly the bytes the decompressor was waiting
    for, and the gzip and tar layers above never learn that the connection
    dropped.

    Why this is safe rather than merely convenient: ``_position`` advances only
    over bytes that have actually been *returned* to the caller. A read that
    fails part-way therefore reconnects at the last byte the caller saw, not at
    whatever the socket managed to buffer, so the delivered sequence has no gap
    and no overlap. If that reasoning were ever wrong the gzip CRC would fail at
    the end of the archive — the corruption cannot pass silently.

    This exists because the link this tool runs over is not reliable. A 23 GB
    archive fetched over a connection that drops every few minutes will never
    complete on a single socket, and a fetch that has to succeed in one
    uninterrupted attempt is not a fetch that works outside a data centre.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 120.0,
        max_attempts: int = 10,
        backoff_seconds: float = 2.0,
    ) -> None:
        self._url = url
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._position = 0
        self._response: object | None = None
        self.reconnects = 0
        self.content_length: int | None = None
        self._connect()

    def _connect(self) -> None:
        """Open the stream at the current position, retrying transient failures."""
        headers = {"User-Agent": "viflap-corpus-fetch"}
        if self._position:
            headers["Range"] = f"bytes={self._position}-"

        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                request = urllib.request.Request(self._url, headers=headers)
                response = urllib.request.urlopen(request, timeout=self._timeout)
            except (OSError, http.client.HTTPException) as error:
                last_error = error
                time.sleep(min(self._backoff_seconds * (attempt + 1), 30.0))
                continue

            if self._position and getattr(response, "status", None) != 206:
                # A 200 here means the server ignored the range and restarted
                # the archive. Continuing would splice the beginning of the file
                # into the middle of the stream.
                response.close()
                raise RuntimeError(
                    f"server did not honour a range request at byte {self._position}; "
                    f"resuming is not possible against this host"
                )
            if not self._position:
                length = response.headers.get("Content-Length")
                self.content_length = int(length) if length else None
            self._response = response
            return

        raise RuntimeError(
            f"could not open {self._url} at byte {self._position} after "
            f"{self._max_attempts} attempts"
        ) from last_error

    def read(self, size: int = -1) -> bytes:
        last_error: Exception | None = None
        for _ in range(self._max_attempts):
            try:
                chunk: bytes = self._response.read(size)  # type: ignore[attr-defined]
            except (OSError, http.client.HTTPException) as error:
                last_error = error
                self.reconnects += 1
                print(
                    f"  connection lost at {self._position / 1e9:.2f} GB "
                    f"({type(error).__name__}); reconnecting",
                    flush=True,
                )
                self._close_response()
                self._connect()
                continue
            self._position += len(chunk)
            return chunk

        raise RuntimeError(
            f"stream failed at byte {self._position} after {self._max_attempts} "
            f"reconnection attempts"
        ) from last_error

    def _close_response(self) -> None:
        close = getattr(self._response, "close", None)
        if close is not None:
            # Suppressed because this connection is already being discarded:
            # it failed, that is why we are here, and a second failure while
            # closing it says nothing the caller can act on.
            with contextlib.suppress(Exception):
                close()

    def close(self) -> None:
        self._close_response()

    def __enter__(self) -> _ResilientSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _parse_member(name: str) -> tuple[str, str, str] | None:
    """Split a LibriSpeech member path into (speaker, chapter, stem).

    Paths look like ``LibriSpeech/train-clean-100/19/198/19-198-0001.flac``.
    Anything else — the licence, the transcript files, directory entries — is
    not audio and is skipped.
    """
    if not name.endswith(".flac"):
        return None
    parts = name.split("/")
    if len(parts) < 4:
        return None
    speaker, chapter = parts[-3], parts[-2]
    if not speaker.isdigit() or not chapter.isdigit():
        return None
    return speaker, chapter, Path(parts[-1]).stem


def fetch(
    destination: Path,
    url: str = DEFAULT_URL,
    selection: Selection | None = None,
    progress_every: int = 2000,
    max_attempts: int = 10,
) -> dict[str, object]:
    """Stream ``url`` and retain the selected subset under ``destination``.

    Returns a manifest describing exactly what was kept, which is written
    alongside the audio so that a later run can be checked against it rather
    than re-derived by listing the directory.

    The stream reconnects and resumes on network failure — see
    :class:`_ResilientSource`. The number of reconnections is recorded in the
    manifest rather than kept quiet: a fetch that needed forty reconnections
    completed, but it also says something about the link it ran over that the
    next person deserves to know.
    """
    selection = selection or Selection()
    destination.mkdir(parents=True, exist_ok=True)

    progress = _Progress()
    kept_per_chapter: dict[tuple[str, str], int] = defaultdict(int)
    chapters_of: dict[str, set[str]] = defaultdict(set)
    manifest: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    with _ResilientSource(url, max_attempts=max_attempts) as source:
        total_bytes = source.content_length
        print(
            f"streaming {url}\n  {(total_bytes or 0) / 1e9:.2f} GB, "
            f"keeping <= {selection.chapters_per_speaker} chapters x "
            f"{selection.utterances_per_chapter} utterances per speaker",
            flush=True,
        )

        reader = _CountingReader(source, progress)
        # Stream mode: sequential, no seeking, so the archive is never
        # materialised on disk.
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:  # type: ignore[arg-type]
            for member in archive:
                progress.members_seen += 1
                if progress.members_seen % progress_every == 0:
                    print(progress.line(total_bytes), flush=True)

                if not member.isfile():
                    continue
                parsed = _parse_member(member.name)
                if parsed is None:
                    continue
                speaker, chapter, stem = parsed

                if (
                    selection.max_speakers is not None
                    and speaker not in chapters_of
                    and len(chapters_of) >= selection.max_speakers
                ):
                    continue
                if (
                    chapter not in chapters_of[speaker]
                    and len(chapters_of[speaker]) >= selection.chapters_per_speaker
                ):
                    continue
                if kept_per_chapter[(speaker, chapter)] >= selection.utterances_per_chapter:
                    continue

                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                target = destination / speaker / chapter
                target.mkdir(parents=True, exist_ok=True)
                (target / f"{stem}.flac").write_bytes(extracted.read())

                chapters_of[speaker].add(chapter)
                kept_per_chapter[(speaker, chapter)] += 1
                progress.kept += 1
                manifest[speaker][chapter].append(stem)

    # Speakers that yielded only one usable session are recorded but flagged:
    # they are usable for the UBM and the total variability matrix, which are
    # unsupervised, and unusable for estimating within-speaker covariance.
    single_session = sorted(s for s, c in chapters_of.items() if len(c) < 2)

    summary: dict[str, object] = {
        "source_url": url,
        "source_sample_rate": SOURCE_RATE,
        "n_speakers": len(chapters_of),
        "n_recordings": progress.kept,
        "chapters_per_speaker": selection.chapters_per_speaker,
        "utterances_per_chapter": selection.utterances_per_chapter,
        "single_session_speakers": single_session,
        "bytes_streamed": progress.bytes_read,
        # Kept because it describes the link, not the corpus. A run that needed
        # many reconnections produced the same audio as one that needed none,
        # and the next person planning a fetch should know which it was.
        "stream_reconnects": source.reconnects,
        "speakers": {s: dict(c) for s, c in manifest.items()},
    }
    (destination / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\nkept {progress.kept} files from {len(chapters_of)} speakers "
        f"after streaming {progress.bytes_read / 1e9:.2f} GB",
        flush=True,
    )
    if single_session:
        print(
            f"warning: {len(single_session)} speakers have a single session and "
            f"cannot contribute within-speaker variation",
            flush=True,
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--destination", type=Path, default=Path("data/corpus/librispeech"))
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--chapters-per-speaker", type=int, default=3)
    parser.add_argument("--utterances-per-chapter", type=int, default=10)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help=(
            "reconnection attempts before giving up. The default suits an "
            "unreliable link; raise it for a worse one."
        ),
    )
    arguments = parser.parse_args(argv)

    fetch(
        destination=arguments.destination,
        url=arguments.url,
        selection=Selection(
            max_speakers=arguments.max_speakers,
            chapters_per_speaker=arguments.chapters_per_speaker,
            utterances_per_chapter=arguments.utterances_per_chapter,
        ),
        max_attempts=arguments.max_attempts,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
