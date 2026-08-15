# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-local dynamic-content pool for graph traces.

Captured assistant responses keyed ``trace_id -> ordinal ->
value`` so a later credit of the SAME trace on the SAME worker (guaranteed by
per-trace sticky routing) can splice a predecessor's actual response into its
prompt. Content never crosses back to the timing plane; this pool is
the graph twin of the worker-cached linear ``UserSession`` state.

Values are :class:`GraphCapturedReply` (the reply's joined replayable text
plus, for tool_calls / structured replies, the verbatim orjson-serialized
assistant message) or a :class:`GraphPoolSentinel`: ``FAILED`` (dispatch error
/ timeout / cancellation) and ``EMPTY`` (successful response with no
replayable content at all) both splice as omission downstream but stay
distinct for observability. A tool-calls-only reply is NOT ``EMPTY``: it is a
structured capture whose ``message_json`` splices the recorded assistant
message -- ``tool_calls`` included -- into successor prompts verbatim.

Lifecycle: the router forwards ``GraphTraceEnd`` when the strategy reaps a
trace's dispatch adapter; eviction is DEFERRED while the worker still has
in-flight credits for the trace (their capture writes may land after the end
message on cancelled paths). The byte cap is a load-bearing LRU backstop for
lost end messages (worker death re-routes, version skew): evicting a live
trace's entry surfaces as a loud pool-missing error at its consumer, never a
silent content truncation.

Disambiguation -- "pool" names three unrelated things in the graph subsystem.
Here it is the RUNTIME capture cache (:class:`GraphDynamicPool`,
:class:`GraphPoolSentinel`), and :class:`GraphPoolMissingError` refers to THIS
pool. It is NOT ``dataset/graph/segment_trie/pool.py`` (the build-plane
interned content store: ``SegmentPool`` and its dynamo shims) and NOT
``dataset/graph/adapters/shared/pool.py`` (a multiprocessing worker pool for
parse dispatch).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from itertools import count

__all__ = [
    "GraphCapturedReply",
    "GraphDynamicPool",
    "GraphPoolMissingError",
    "GraphPoolSentinel",
]


class GraphPoolMissingError(RuntimeError):
    """A dynamic slot referenced a pool entry this worker does not hold.

    Broken stickiness (worker death re-route) or backstop eviction; the worker
    converts it to a ``credit_context.error`` with the
    ``GraphErrorCode.POOL_MISSING`` code, which the dispatch adapter raises
    as a non-containable trace error — never a silent omission.
    """

    def __init__(self, src_ordinal: int) -> None:
        super().__init__(f"missing dynamic pool value for producer {src_ordinal}")
        self.src_ordinal = src_ordinal


class GraphPoolSentinel(str, Enum):
    FAILED = "failed"
    EMPTY = "empty"


@dataclass(slots=True, frozen=True)
class GraphCapturedReply:
    """One captured assistant reply, structured so splices can reproduce it."""

    text: str
    """Joined replayable text of the reply (``''`` when the reply is
    tool_calls-only). Consumed by ``{"sv"}`` block parts (string
    concatenation) and as the fallback ``{"s"}`` splice content."""

    message_json: str | None = None
    """Verbatim orjson-serialized assistant message when the endpoint's
    ``build_assistant_turn`` returned ``raw_messages`` (tool_calls /
    structured replies); ``None`` for plain-text replies. ``{"s"}`` splices
    ``orjson.loads(message_json)`` verbatim, byte-matching the legacy
    child-seed rendering."""


PoolValue = GraphCapturedReply | GraphPoolSentinel

_SENTINEL_COST_BYTES = 8

# Default bound on the recently-ended-key tombstone FIFO (see
# `GraphDynamicPool.__init__`). Keys are two short strings (~100 B each), so
# the worst-case footprint is a few hundred KiB per worker.
_TOMBSTONE_CAPACITY = 4096


def _value_bytes(value: PoolValue) -> int:
    if isinstance(value, GraphPoolSentinel):
        return _SENTINEL_COST_BYTES
    return len(value.text.encode()) + (
        len(value.message_json.encode()) if value.message_json else 0
    )


@dataclass(slots=True)
class _TraceEntry:
    """One trace execution's captured values plus its lifecycle state."""

    values: dict[int, PoolValue] = field(default_factory=dict)
    """Captured value per node ordinal."""
    total_bytes: int = 0
    """Byte cost of ``values`` (UTF-8 text + message JSON length; flat cost
    per sentinel)."""
    inflight: int = 0
    """Credits of this trace currently processing on this worker."""
    end_pending: bool = False
    """A ``GraphTraceEnd`` arrived while credits were still in flight."""
    last_touch: int = 0
    """Monotonic LRU stamp (pool-level counter, not wall time)."""


class GraphDynamicPool:
    """Byte-capped, LRU-backstopped store of captured graph responses."""

    def __init__(
        self, max_bytes: int, tombstone_capacity: int = _TOMBSTONE_CAPACITY
    ) -> None:
        self._max_bytes = max_bytes
        self._entries: dict[str, _TraceEntry] = {}
        self._total_bytes = 0
        self._clock = count()
        # FIFO tombstones of recently-ENDED keys: `credit_started` / `put` for
        # a tombstoned key is a no-op, so a straggler event racing behind its
        # `GraphTraceEnd` (cancelled dispatch's start/capture landing after the
        # end) cannot resurrect a zero-byte immortal entry. Bounded so recycle-
        # heavy runs don't trade the entry leak for a tombstone leak; a key
        # aged out of the FIFO could in principle resurrect, but by then its
        # straggler window is long past.
        self._tombstone_capacity = tombstone_capacity
        self._tombstones: OrderedDict[str, None] = OrderedDict()

    @property
    def total_bytes(self) -> int:
        """Live byte cost of every held entry, as measured against ``max_bytes``.

        Accounting only -- UTF-8 text plus message JSON length, with a flat
        per-sentinel cost. Not the pool's real RSS.
        """
        return self._total_bytes

    def get(self, trace_id: str, node_ordinal: int) -> PoolValue | None:
        """The captured value for one node, or ``None`` when absent.

        ``None`` means missing (never captured, or evicted) -- the Phase 3
        consumer converts it to a loud pool-missing error, never a default.
        """
        entry = self._entries.get(trace_id)
        if entry is None:
            return None
        entry.last_touch = next(self._clock)
        return entry.values.get(node_ordinal)

    def put(
        self,
        trace_id: str,
        node_ordinal: int,
        value: PoolValue,
    ) -> None:
        """Store one node's captured reply for a successor node to splice.

        Silently a NO-OP, not an error, when the trace has already ENDED: a
        straggler capture racing behind ``GraphTraceEnd`` must not resurrect an
        entry no consumer will ever read. An overwrite rebalances the byte
        accounting before re-running the LRU cap with this trace protected, so
        the write just made is the last entry evicted, never the first.
        """
        key = trace_id
        entry = self._entries.get(key)
        if entry is None:
            if key in self._tombstones:
                # Straggler capture for an already-ENDED trace: nothing will
                # ever read it (the end already reaped the consumers), so
                # storing it would recreate an immortal entry.
                return
            entry = _TraceEntry()
            self._entries[key] = entry
        previous = entry.values.get(node_ordinal)
        if previous is not None:
            delta = _value_bytes(previous)
            entry.total_bytes -= delta
            self._total_bytes -= delta
        entry.values[node_ordinal] = value
        added = _value_bytes(value)
        entry.total_bytes += added
        self._total_bytes += added
        entry.last_touch = next(self._clock)
        self._enforce_cap(protect=key)

    def credit_started(self, trace_id: str) -> None:
        """Open the in-flight bracket for one credit; pins the trace in the pool.

        MUST be paired with :meth:`credit_finished` on every exit path,
        including cancellation -- the worker brackets it in ``try/finally``
        around graph dispatch. An unbalanced call leaves ``inflight > 0``
        forever, so a later :meth:`trace_end` defers eviction indefinitely and
        the entry can only leave via the byte-cap LRU backstop.

        Creates the entry when absent so an early capture has somewhere to
        land, EXCEPT on an already-ENDED trace, where it is a silent no-op (a
        cancelled dispatch racing behind ``GraphTraceEnd`` must not resurrect a
        zero-byte entry with no end message left to evict it).
        """
        key = trace_id
        entry = self._entries.get(key)
        if entry is None:
            if key in self._tombstones:
                # The trace already ENDED; a late credit_started (cancelled
                # dispatch racing behind GraphTraceEnd) must not resurrect a
                # zero-byte entry with no end message left to evict it.
                return
            entry = _TraceEntry()
            self._entries[key] = entry
        entry.inflight += 1
        entry.last_touch = next(self._clock)

    def credit_finished(self, trace_id: str) -> None:
        """Close the in-flight bracket; run the deferred eviction if it drains.

        This is the eviction trigger for a trace whose ``GraphTraceEnd``
        arrived while credits were still in flight -- when the count reaches
        zero and an end is pending, the entry is evicted and the key
        tombstoned here rather than in :meth:`trace_end`. Silently a no-op for
        a trace this pool does not hold, and the count floors at zero, so an
        extra call cannot drive it negative.
        """
        key = trace_id
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.inflight = max(0, entry.inflight - 1)
        if entry.end_pending and entry.inflight == 0:
            self._evict(key)
            self._mark_ended(key)

    def trace_end(self, trace_id: str) -> None:
        """Handle ``GraphTraceEnd``: evict now, or defer until in-flight drains.

        Deferral covers the cancelled paths where a credit's capture write can
        land AFTER the end message (the strategy reaps on adapter drain, but a
        locally-cancelled dispatch leaves the worker task still running).
        Idempotent; unknown traces still tombstone the key so straggler events
        arriving after the end cannot resurrect an entry.
        """
        key = trace_id
        entry = self._entries.get(key)
        if entry is not None and entry.inflight > 0:
            entry.end_pending = True
            return
        self._evict(key)
        self._mark_ended(key)

    def _mark_ended(self, key: str) -> None:
        """Tombstone ``key`` as recently ended (bounded FIFO, oldest dropped)."""
        self._tombstones[key] = None
        self._tombstones.move_to_end(key)
        while len(self._tombstones) > self._tombstone_capacity:
            self._tombstones.popitem(last=False)

    def _evict(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._total_bytes -= entry.total_bytes

    def _enforce_cap(self, protect: str) -> None:
        """LRU-evict whole trace entries until under the byte cap.

        The entry just written is evicted only as the last resort (it alone
        exceeds the cap); its consumers then fail loudly at splice time.
        """
        if self._total_bytes <= self._max_bytes:
            return
        by_lru = sorted(self._entries.items(), key=lambda kv: kv[1].last_touch)
        for key, _entry in by_lru:
            if self._total_bytes <= self._max_bytes:
                return
            if key == protect:
                continue
            self._evict(key)
        if self._total_bytes > self._max_bytes:
            self._evict(protect)
