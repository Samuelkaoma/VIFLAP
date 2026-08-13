"""Append-only, hash-chained audit log.

The proposal states the requirement plainly: an audit log the operator can edit
is not an audit log. This implementation makes editing *detectable*, which is
the strongest property software alone can provide — detection, not prevention.
Prevention requires the log to be held somewhere the operator cannot write, and
that is a deployment question this module supports rather than solves.

The chain
---------
Each entry carries the SHA-256 hash of its own canonical serialisation
concatenated with the previous entry's hash. Altering entry ``k`` changes its
hash, which breaks the link to entry ``k+1``, and so on to the end. An editor
must therefore recompute every subsequent hash, and if a hash from the tail has
been published or countersigned externally, even that fails.

Three implementation details that a naive version gets wrong
-------------------------------------------------------------
**Canonical serialisation.** The hash is taken over a deterministic encoding —
sorted keys, no insertion-order dependence, fixed separators, explicit UTF-8.
Hashing ``json.dumps`` with default settings makes verification depend on
dictionary ordering, so a log written by one process can fail verification in
another for reasons having nothing to do with tampering. A verifier that raises
false alarms is a verifier that gets switched off.

**Appending is O(1).** The previous hash and sequence number are held in memory
and updated on write. Re-reading the file to find the tail on every append is
``O(n)`` per entry and ``O(n^2)`` over a session — which on a busy day means
audit writes come to dominate query latency, and the first proposal to "make
auditing optional for performance" follows shortly after.

**Writes are durable before they are acknowledged.** The file is flushed and
synced before ``record`` returns. An audit entry lost in a buffer when the
process dies is an unrecorded query, which is the one failure mode this
component exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from viflap.application.ports import AuditRecord, ChainVerification
from viflap.domain.errors import AuditIntegrityError

__all__ = ["GENESIS", "FileAuditLog", "canonical_json"]

#: Seed of the chain. A fixed, published value, so that the first entry's hash
#: is verifiable by anyone rather than depending on a secret.
GENESIS: str = hashlib.sha256(b"VIFLAP-AUDIT-CHAIN-v1").hexdigest()


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Deterministic JSON encoding, for hashing.

    Sorted keys, no whitespace, no ASCII escaping, and non-serialisable values
    coerced rather than raising. The coercion matters: an audit write must not
    fail because a caller put an unusual type in the parameters. Losing the
    entry is worse than recording a stringified value.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


class FileAuditLog:
    """A hash-chained audit log backed by a JSON Lines file.

    Thread-safe. Concurrent appends from one process are serialised by a lock;
    concurrent appends from *several* processes would interleave and break the
    chain, so a deployment must write through a single process. That constraint
    is stated rather than defended against, because defending against it in the
    file system means a lock file, and a lock file introduces a failure mode
    (stale locks blocking all auditing) worse than the one it prevents.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._previous_hash, self._sequence = self._read_tail()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def sequence(self) -> int:
        """Number of entries written."""
        return self._sequence

    def _read_tail(self) -> tuple[str, int]:
        """Recover the chain head once, at construction."""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return GENESIS, 0

        last_line = ""
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_line = line
        if not last_line:
            return GENESIS, 0

        try:
            record = json.loads(last_line)
            return str(record["entry_hash"]), int(record["sequence"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise AuditIntegrityError(
                "the audit log's final entry is unreadable; the chain head "
                "cannot be established and appending would silently start a "
                "second chain",
                path=str(self._path),
            ) from exc

    def record(self, entry: AuditRecord) -> str:
        """Append an entry and return its hash."""
        with self._lock:
            sequence = self._sequence + 1
            body = self._body(entry, sequence)
            entry_hash = self._hash(self._previous_hash, body)

            line = canonical_json({**body, "entry_hash": entry_hash})
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                # fsync, not just flush: flush moves the bytes to the operating
                # system, fsync moves them to the disk. A power loss between the
                # two loses the entry while the caller believes it recorded.
                os.fsync(handle.fileno())

            self._previous_hash = entry_hash
            self._sequence = sequence
            return entry_hash

    @staticmethod
    def _body(entry: AuditRecord, sequence: int) -> dict[str, Any]:
        timestamp = entry.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return {
            "sequence": sequence,
            "timestamp": timestamp.astimezone(UTC).isoformat(),
            "actor_id": entry.actor_id,
            "actor_roles": list(entry.actor_roles),
            "action": entry.action,
            "case_reference": entry.case_reference,
            "parameters": dict(entry.parameters),
            "outcome": entry.outcome,
        }

    @staticmethod
    def _hash(previous_hash: str, body: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            (previous_hash + canonical_json(body)).encode("utf-8")
        ).hexdigest()

    def verify(self) -> ChainVerification:
        """Recompute the whole chain from the genesis value.

        Reports the index of the first broken link rather than a bare boolean.
        An oversight body investigating a breach needs to know *where* the chain
        was altered: entries before that point are still trustworthy, and
        narrowing the window is most of the investigation.
        """
        if not self._path.exists() or self._path.stat().st_size == 0:
            return ChainVerification(is_intact=True, n_entries=0, detail="log is empty")

        previous = GENESIS
        count = 0

        with self._path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    return ChainVerification(
                        is_intact=False,
                        n_entries=count,
                        first_broken_index=index,
                        detail="entry is not valid JSON",
                    )

                stored = record.pop("entry_hash", None)
                if stored is None:
                    return ChainVerification(
                        is_intact=False,
                        n_entries=count,
                        first_broken_index=index,
                        detail="entry carries no hash",
                    )

                if record.get("sequence") != index + 1:
                    return ChainVerification(
                        is_intact=False,
                        n_entries=count,
                        first_broken_index=index,
                        detail=(
                            f"sequence number {record.get('sequence')} disagrees "
                            f"with position {index + 1}; an entry has been "
                            f"removed or reordered"
                        ),
                    )

                expected = self._hash(previous, record)
                if expected != stored:
                    return ChainVerification(
                        is_intact=False,
                        n_entries=count,
                        first_broken_index=index,
                        detail=(
                            "recomputed hash differs from the stored hash; "
                            "this entry or one before it has been altered"
                        ),
                    )
                previous = stored
                count += 1

        return ChainVerification(
            is_intact=True, n_entries=count, detail="all entries verify against genesis"
        )

    def query(
        self,
        case_reference: str | None = None,
        actor_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> Sequence[AuditRecord]:
        """Read entries matching the given filters.

        Streams the file rather than loading it. An audit log retained for seven
        years does not fit comfortably in memory, and a query that exhausts
        memory takes the auditing capability down with it.
        """
        results: list[AuditRecord] = []
        for record in self._iter_records():
            if case_reference and record.case_reference != case_reference:
                continue
            if actor_id and record.actor_id != actor_id:
                continue
            if since and record.timestamp < since:
                continue
            if until and record.timestamp > until:
                continue
            results.append(record)
            if limit is not None and len(results) >= limit:
                break
        return results

    def _iter_records(self) -> Iterator[AuditRecord]:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield AuditRecord(
                    timestamp=datetime.fromisoformat(payload["timestamp"]),
                    actor_id=payload["actor_id"],
                    actor_roles=tuple(payload.get("actor_roles", ())),
                    action=payload["action"],
                    case_reference=payload["case_reference"],
                    parameters=payload.get("parameters", {}),
                    outcome=payload.get("outcome", "ok"),
                )
