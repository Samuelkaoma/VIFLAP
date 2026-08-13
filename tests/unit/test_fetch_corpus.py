"""Fetching a corpus over a link that drops.

The archive is 23 GB of gzip, which cannot be resumed at the archive level —
decompression has no restart point. It is resumed at the byte level instead, and
the property that makes that sound is narrow enough to be worth testing
directly: the bytes handed to the decompressor after a reconnection must be
exactly the bytes it would have received had nothing gone wrong. A gap or an
overlap of even one byte produces a corrupt corpus.
"""

from __future__ import annotations

import urllib.request
from typing import Any

import pytest

from scripts.fetch_corpus import _ResilientSource

URL = "https://example.invalid/train-clean-360.tar.gz"


class _FakeResponse:
    """Serves a slice of the payload, optionally dying part-way through."""

    def __init__(
        self,
        payload: bytes,
        status: int,
        headers: dict[str, str],
        fail_after: int | None,
    ) -> None:
        self._payload = payload
        self._position = 0
        self._served = 0
        self._fail_after = fail_after
        self.status = status
        self.headers = headers
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if self._fail_after is not None and self._served >= self._fail_after:
            raise OSError("connection reset by peer")
        if size is None or size < 0:
            size = len(self._payload) - self._position
        if self._fail_after is not None:
            size = min(size, self._fail_after - self._served)
        chunk = self._payload[self._position : self._position + size]
        self._position += len(chunk)
        self._served += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class _FlakyHost:
    """Stands in for ``urlopen``. Honours Range, fails on a fixed schedule."""

    def __init__(
        self,
        payload: bytes,
        failures: list[int | None],
        honour_range: bool = True,
        connect_errors: int = 0,
    ) -> None:
        self.payload = payload
        self._failures = list(failures)
        self._honour_range = honour_range
        self._connect_errors = connect_errors
        self.starts: list[int] = []

    def __call__(self, request: Any, timeout: float | None = None) -> _FakeResponse:
        if self._connect_errors > 0:
            self._connect_errors -= 1
            raise OSError("connection attempt failed")

        header = request.headers.get("Range")
        start = int(header.split("=")[1].split("-")[0]) if header else 0
        self.starts.append(start)

        fail_after = self._failures.pop(0) if self._failures else None
        if start and not self._honour_range:
            # A server that ignores the range and restarts the archive.
            return _FakeResponse(self.payload, 200, {}, fail_after)

        headers = {"Content-Length": str(len(self.payload))} if not start else {}
        status = 206 if start else 200
        return _FakeResponse(self.payload[start:], status, headers, fail_after)


def _drain(source: _ResilientSource, chunk_size: int = 4096) -> bytes:
    pieces: list[bytes] = []
    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            return b"".join(pieces)
        pieces.append(chunk)


@pytest.fixture
def payload() -> bytes:
    return bytes(range(256)) * 400


class TestResilientSource:
    def test_delivers_the_exact_byte_sequence_across_reconnections(
        self, payload: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The property everything else rests on."""
        host = _FlakyHost(payload, failures=[1000, 2000, 500, None])
        monkeypatch.setattr(urllib.request, "urlopen", host)

        source = _ResilientSource(URL, backoff_seconds=0.0)
        assert _drain(source) == payload

    def test_resumes_from_the_last_byte_the_caller_saw(
        self, payload: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not from whatever the socket happened to buffer."""
        host = _FlakyHost(payload, failures=[1000, None])
        monkeypatch.setattr(urllib.request, "urlopen", host)

        source = _ResilientSource(URL, backoff_seconds=0.0)
        _drain(source, chunk_size=250)

        assert host.starts[0] == 0
        assert host.starts[1] == 1000

    def test_counts_reconnections(
        self, payload: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        host = _FlakyHost(payload, failures=[1000, 2000, 500, None])
        monkeypatch.setattr(urllib.request, "urlopen", host)

        source = _ResilientSource(URL, backoff_seconds=0.0)
        _drain(source)

        assert source.reconnects == 3

    def test_reads_content_length_from_the_first_response_only(
        self, payload: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 206 reports the length of the remainder, which is not the total."""
        host = _FlakyHost(payload, failures=[1000, None])
        monkeypatch.setattr(urllib.request, "urlopen", host)

        source = _ResilientSource(URL, backoff_seconds=0.0)
        _drain(source)

        assert source.content_length == len(payload)

    def test_refuses_a_server_that_ignores_the_range(
        self, payload: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Splicing the start of the archive into the middle would corrupt it."""
        host = _FlakyHost(payload, failures=[1000, None], honour_range=False)
        monkeypatch.setattr(urllib.request, "urlopen", host)

        source = _ResilientSource(URL, backoff_seconds=0.0)
        with pytest.raises(RuntimeError, match="did not honour a range request"):
            _drain(source)

    def test_retries_a_failing_initial_connection(
        self, payload: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The observed failure mode: the first connect times out, later ones work."""
        host = _FlakyHost(payload, failures=[None], connect_errors=3)
        monkeypatch.setattr(urllib.request, "urlopen", host)

        source = _ResilientSource(URL, backoff_seconds=0.0)
        assert _drain(source) == payload

    def test_gives_up_after_max_attempts(
        self, payload: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        host = _FlakyHost(payload, failures=[], connect_errors=99)
        monkeypatch.setattr(urllib.request, "urlopen", host)

        with pytest.raises(RuntimeError, match="could not open"):
            _ResilientSource(URL, max_attempts=3, backoff_seconds=0.0)

    def test_stops_reconnecting_on_an_endlessly_failing_stream(
        self, payload: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A link that never delivers must fail, not spin forever."""
        host = _FlakyHost(payload, failures=[0] * 50)
        monkeypatch.setattr(urllib.request, "urlopen", host)

        source = _ResilientSource(URL, max_attempts=4, backoff_seconds=0.0)
        with pytest.raises(RuntimeError, match="stream failed"):
            _drain(source)

    def test_closes_the_dead_response_before_reconnecting(
        self, payload: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise a long fetch leaks a socket per drop."""
        opened: list[_FakeResponse] = []
        host = _FlakyHost(payload, failures=[1000, None])

        def _recording(request: Any, timeout: float | None = None) -> _FakeResponse:
            response = host(request, timeout)
            opened.append(response)
            return response

        monkeypatch.setattr(urllib.request, "urlopen", _recording)

        source = _ResilientSource(URL, backoff_seconds=0.0)
        _drain(source)
        source.close()

        assert all(response.closed for response in opened)
