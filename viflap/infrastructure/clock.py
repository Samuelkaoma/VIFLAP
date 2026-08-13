"""Clock implementations.

Injected everywhere rather than called directly, so that use cases can be tested
for the ordering guarantees the audit chain depends on. A test that cannot fix
the clock cannot assert that an audit entry precedes the deletion it describes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

__all__ = ["FixedClock", "SystemClock"]


class SystemClock:
    """The wall clock, in UTC.

    Always UTC, never local time. Local timestamps are ambiguous across a
    daylight-saving transition — one hour occurs twice — and an audit trail that
    cannot order two entries within that hour has a gap exactly where an
    investigation would look.
    """

    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """A controllable clock, for tests and for reproducing a historical run."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("a fixed clock requires a timezone-aware start time")
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, **delta: float) -> datetime:
        self._now = self._now + timedelta(**delta)
        return self._now

    def set(self, moment: datetime) -> None:
        if moment.tzinfo is None:
            raise ValueError("a fixed clock requires a timezone-aware time")
        self._now = moment
