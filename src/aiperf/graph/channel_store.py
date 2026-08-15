# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VersionedChannelStore - per-trace channel state for the async-dataflow executor.

A versioned log per channel. Writes are linearized via a single monotonic
`_next_seq` counter; readers capture the per-channel versions at firing time and
reducers consume them in (write_seq, writer_node_id) order on read. No deepcopy
anywhere - values are stored share-by-ref.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aiperf.dataset.graph.models import ChannelRequirement, ChannelSpec, ReducerName
from aiperf.graph.reducers import (
    UNSET,
    OverwriteConflictError,
    Reducer,
    get_reducer,
)

__all__ = [
    "ChannelCaptureMissingError",
    "ChannelOrphanedError",
    "ReducerRegistry",
    "UnknownChannelError",
    "VersionedChannelStore",
]


class UnknownChannelError(KeyError):
    """Raised when writing or reading a channel not declared in `channel_specs`."""


class ChannelCaptureMissingError(KeyError):
    """Raised when a read's requirement set and its version capture disagree.

    Distinct from `UnknownChannelError`: the channel IS declared, it just has
    no entry in the capture the caller passed to `read`. Both are built by the
    executor from the same firing, so a mismatch is an executor invariant
    violation, not a missing write and not bad workload input. `KeyError` is
    kept as the base so any caller catching the broad type still catches this.
    """


class ChannelOrphanedError(RuntimeError):
    """Raised when an `await_inputs` requirement can no longer be satisfied.

    Surfaces when a channel's arrivals-so-far plus its still-live producers can
    no longer reach the waiter's count target -- which can happen while
    producers are still running, not only after every producer is done.
    """


# Type alias for the reducer dispatch dependency. The store does not depend
# on a concrete registry class - any callable that maps reducer-name to a
# `(current, [(writer_id, value), ...]) -> reduced` reducer works. Defaults
# to `aiperf.graph.reducers.get_reducer`.
ReducerRegistry = Callable[[ReducerName], Reducer]


@dataclass(slots=True, frozen=True)
class _LogEntry:
    """One committed write to a value channel."""

    write_seq: int
    """Monotonic global sequence number assigned at commit time."""
    writer_node_id: str
    """Identifier of the node that produced this write; secondary sort key."""
    value: Any
    """The raw write value; stored by reference, never deep-copied."""


@dataclass(slots=True, frozen=True)
class _VersionCapture:
    """Immutable snapshot of which channel versions count for one firing."""

    channel: str
    """Channel this capture refers to."""
    captured_seqs: tuple[int, ...]
    """`write_seq` values included in this capture, in commit order."""


@dataclass(slots=True)
class _Waiter:
    """One pending `await_inputs` requirement on a single channel."""

    channel: str
    required_count: int
    event: asyncio.Event = field(default_factory=asyncio.Event)
    orphaned_reason: str | None = None


class VersionedChannelStore:
    """Per-trace channel store backed by per-channel append-only logs.

    Writers call `write(channels, value, writer_node_id=...)`; readers call
    `await await_inputs(reqs)` to capture versions, then `read(reqs, capture)`
    to materialize reduced values. Cancellation is communicated by the
    executor via `mark_producer_done(channel, success=...)`.
    """

    __slots__ = (
        "_specs",
        "_logs",
        "_arrival_count",
        "_producers_remaining",
        "_producers_declared",
        "_initial",
        "_next_seq",
        "_waiters",
        "_reducers",
        "_orphaned",
        "_overwrite_writer",
    )

    def __init__(
        self,
        initial: dict[str, Any],
        channel_specs: dict[str, ChannelSpec],
        producers_per_channel: dict[str, int],
    ) -> None:
        self._specs: dict[str, ChannelSpec] = dict(channel_specs)
        self._logs: dict[str, list[_LogEntry]] = {ch: [] for ch in channel_specs}
        self._arrival_count: dict[str, int] = dict.fromkeys(channel_specs, 0)
        self._producers_remaining: dict[str, int] = {
            ch: int(producers_per_channel.get(ch, 0)) for ch in channel_specs
        }
        self._producers_declared: dict[str, int] = {
            ch: int(producers_per_channel.get(ch, 0)) for ch in channel_specs
        }
        self._initial: dict[str, Any] = dict(initial)
        # `_next_seq` is incremented before every commit. Reserve 0 for
        # synthetic initial-state writes; node writes start at 1.
        self._next_seq: int = 0
        self._waiters: dict[str, list[_Waiter]] = {ch: [] for ch in channel_specs}
        self._reducers: ReducerRegistry = get_reducer
        self._orphaned: dict[str, str] = {}
        # Tracks the writer id of the first non-init write to each overwrite
        # channel so a different writer can be rejected with
        # OverwriteConflictError across the trace's lifetime (cross-spec
        # decision 1: strict over the whole trace, not per-dispatch).
        self._overwrite_writer: dict[str, str] = {}

        for ch, value in self._initial.items():
            if ch not in self._specs:
                raise UnknownChannelError(
                    f"initial state names unknown channel: {ch!r}"
                )
            self._logs[ch].append(
                _LogEntry(write_seq=0, writer_node_id="__init__", value=value)
            )
            # Note: init seed does NOT bump `_arrival_count`. arrival_count
            # counts node writes only, so `count=N` requirements measure
            # producer arrivals.

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------
    def write(
        self,
        channel_names: list[str],
        value: Any,
        *,
        writer_node_id: str,
    ) -> None:
        """Commit `value` to every channel in `channel_names`."""
        if not channel_names:
            return
        for ch in channel_names:
            self._validate_write_channel(ch, writer_node_id=writer_node_id)

        for ch in channel_names:
            self._commit_write_channel(ch, value, writer_node_id=writer_node_id)
            self._wake_waiters(ch)

    def _validate_write_channel(self, channel: str, *, writer_node_id: str) -> None:
        if channel not in self._specs:
            raise UnknownChannelError(f"unknown channel: {channel!r}")
        if self._specs[channel].reducer is not ReducerName.OVERWRITE:
            return
        prior = self._overwrite_writer.get(channel)
        if prior is None:
            return
        # Both cases are errors -- the executor fires each node at most once per
        # instance, so even a repeat write from the SAME node is an invariant
        # violation -- but they are different bugs and must not share a message.
        # The second-writer wording named one node as both the incumbent and the
        # interloper ("already written by 'n1'; rejecting second writer 'n1'"),
        # sending the reader hunting for a writer that does not exist.
        if prior == writer_node_id:
            raise OverwriteConflictError(
                f"overwrite channel {channel!r} written twice by the same node "
                f"{writer_node_id!r}: each node fires at most once per instance "
                "run, so this is an executor invariant violation, not a "
                "multi-writer conflict"
            )
        raise OverwriteConflictError(
            f"overwrite channel {channel!r} already written by "
            f"{prior!r}; rejecting second writer {writer_node_id!r}"
        )

    def has_overwrite_writer(self, channel: str) -> bool:
        """Return True when `channel` is OVERWRITE-reduced and already written.

        A subsequent `write` to such a channel raises `OverwriteConflictError`.
        Callers that must not raise (the executor's dispatch-failure containment
        path, which writes a content-neutral sentinel to the failed node's own
        output channels) check this first and skip the write instead.
        """
        if channel not in self._specs:
            raise UnknownChannelError(f"unknown channel: {channel!r}")
        return (
            self._specs[channel].reducer is ReducerName.OVERWRITE
            and channel in self._overwrite_writer
        )

    def _commit_write_channel(
        self, channel: str, value: Any, *, writer_node_id: str
    ) -> None:
        self._next_seq += 1
        entry = _LogEntry(
            write_seq=self._next_seq,
            writer_node_id=writer_node_id,
            value=value,
        )
        self._logs[channel].append(entry)
        self._arrival_count[channel] += 1
        if self._specs[channel].reducer is ReducerName.OVERWRITE:
            self._overwrite_writer.setdefault(channel, writer_node_id)

    # ------------------------------------------------------------------
    # await_inputs
    # ------------------------------------------------------------------
    async def await_inputs(
        self, requirements: list[ChannelRequirement]
    ) -> dict[str, _VersionCapture]:
        """Block until every requirement is satisfied; return frozen captures.

        For each requirement: `count=N` waits for the N-th arrival; `count="all"`
        resolves to the static topology count at call time and then behaves
        identically. Raises `ChannelOrphanedError` if any required channel can
        no longer fulfil its count.
        """
        captures: dict[str, _VersionCapture] = {}
        for req in requirements:
            ch = req.channel
            if ch not in self._specs:
                raise UnknownChannelError(f"unknown channel: {ch!r}")
            if ch in self._orphaned:
                raise ChannelOrphanedError(
                    f"channel {ch!r} orphaned: {self._orphaned[ch]}"
                )
            target = self._resolve_count(req)
            await self._await_count(ch, target)
            captures[ch] = self._capture(ch, target)
        return captures

    def _resolve_count(self, req: ChannelRequirement) -> int:
        if req.count == "all":
            # "all" resolves to the static topology count at call time. Late
            # producer cancellation may render this unreachable; orphan check
            # inside _await_count handles that case.
            return self._producers_declared[req.channel]
        return int(req.count)

    async def _await_count(self, channel: str, target: int) -> None:
        if target <= 0:
            return
        while self._arrival_count[channel] < target:
            if channel in self._orphaned:
                raise ChannelOrphanedError(
                    f"channel {channel!r} orphaned: {self._orphaned[channel]}"
                )
            # Check reachability: arrivals so far + still-live producers must
            # reach target, else orphan immediately.
            reachable = (
                self._arrival_count[channel] + self._producers_remaining[channel]
            )
            if reachable < target:
                self._orphaned[channel] = "insufficient_producers_remaining"
                raise ChannelOrphanedError(
                    f"channel {channel!r} cannot reach count={target}: "
                    f"only {reachable} arrivals possible "
                    f"({self._arrival_count[channel]} so far, "
                    f"{self._producers_remaining[channel]} producers remain)"
                )
            waiter = _Waiter(channel=channel, required_count=target)
            self._waiters[channel].append(waiter)
            try:
                await waiter.event.wait()
            finally:
                with contextlib.suppress(ValueError):
                    self._waiters[channel].remove(waiter)
            if waiter.orphaned_reason is not None:
                raise ChannelOrphanedError(
                    f"channel {channel!r} orphaned: {waiter.orphaned_reason}"
                )

    def _capture(self, channel: str, target: int) -> _VersionCapture:
        entries = self._logs[channel]
        if target <= 0:
            return _VersionCapture(channel=channel, captured_seqs=())
        # Init seed (write_seq=0) is the reducer seed, not an "arrival"; the
        # count-N requirement measures node writes only. The init entry is
        # always implicitly part of read (handled in _reduce_value_channel).
        non_init = [e for e in entries if e.write_seq != 0]
        sorted_entries = sorted(non_init, key=lambda e: (e.write_seq, e.writer_node_id))
        chosen = sorted_entries[:target]
        return _VersionCapture(
            channel=channel,
            captured_seqs=tuple(e.write_seq for e in chosen),
        )

    def _wake_waiters(self, channel: str) -> None:
        if not self._waiters[channel]:
            return
        current = self._arrival_count[channel]
        for waiter in list(self._waiters[channel]):
            if current >= waiter.required_count:
                waiter.event.set()

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------
    def read(
        self,
        requirements: list[ChannelRequirement],
        capture: dict[str, _VersionCapture],
    ) -> dict[str, Any]:
        """Return the reduced value per channel at the captured versions."""
        out: dict[str, Any] = {}
        for req in requirements:
            ch = req.channel
            if ch not in self._specs:
                raise UnknownChannelError(f"unknown channel: {ch!r}")
            cap = capture.get(ch)
            if cap is None:
                raise ChannelCaptureMissingError(
                    f"channel {ch!r} is required by this node but absent from "
                    "its version capture: the requirement set and the capture "
                    "were built from different reads, which is an executor "
                    "invariant violation, not a missing write"
                )
            out[ch] = self._reduce_value_channel(ch, cap.captured_seqs)
        return out

    def _reduce(self, channel: str, chosen: list[_LogEntry]) -> Any:
        """Seed with the init write, order `chosen`, and apply the reducer.

        The single reduce sequence shared by both read paths. `chosen` holds the
        non-init entries taking part in this reduction: the captured subset for
        `read`, every node write for `snapshot`. Everything else -- which entry
        seeds the reducer, the `(write_seq, writer_node_id)` order, the
        no-writes shortcut -- is identical between the two by construction.
        """
        entries = self._logs[channel]
        # Init seed (write_seq=0) is the reducer's starting value; it is not
        # itself a participating write. Capture sets hold node writes only.
        init_entry = next((e for e in entries if e.write_seq == 0), None)
        current: Any = init_entry.value if init_entry is not None else UNSET
        if not chosen:
            return current
        ordered = sorted(chosen, key=lambda e: (e.write_seq, e.writer_node_id))
        reducer = self._reducers(self._specs[channel].reducer)
        try:
            return reducer(current, [(e.writer_node_id, e.value) for e in ordered])
        except (OverwriteConflictError, TypeError) as e:
            # Reducers are channel-agnostic by signature, so their messages
            # carry writer node ids only. The channel is the one fact that
            # localizes the failure in a graph with dozens of channels and a
            # shared writer set. Re-raised as the same class so callers
            # catching OverwriteConflictError / TypeError are unaffected.
            raise type(e)(f"channel {channel!r}: {e}") from e

    def _reduce_value_channel(
        self, channel: str, captured_seqs: tuple[int, ...]
    ) -> Any:
        seq_set = set(captured_seqs)
        return self._reduce(
            channel, [e for e in self._logs[channel] if e.write_seq in seq_set]
        )

    # ------------------------------------------------------------------
    # snapshot
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Return the final user-visible view of every channel."""
        return {
            ch: self._reduce(ch, [e for e in self._logs[ch] if e.write_seq != 0])
            for ch in self._specs
        }

    # ------------------------------------------------------------------
    # producer accounting / orphan propagation
    # ------------------------------------------------------------------
    def mark_producer_done(
        self,
        channel: str,
        *,
        success: bool,
    ) -> None:
        """Inform the store that one producer of `channel` has terminated.

        Decrements the remaining-producer count, then -- regardless of
        `success` -- wakes every waiter whose count target is now unreachable
        (arrivals + remaining producers), stamping `orphaned_reason`. `success`
        only picks the label: `insufficient_producers_remaining` vs
        `all_producers_cancelled`.

        The CHANNEL itself is marked orphaned (failing later `await_inputs`
        immediately) only when `success=False` leaves it with no producers
        remaining, no arrivals, and no init-state seed. Whether this particular
        producer wrote is irrelevant: arrivals are already counted by `write`.
        """
        if channel not in self._specs:
            raise UnknownChannelError(f"unknown channel: {channel!r}")
        if self._producers_remaining[channel] > 0:
            self._producers_remaining[channel] -= 1
        reachable = self._arrival_count[channel] + self._producers_remaining[channel]
        orphan_reason = (
            "insufficient_producers_remaining" if success else "all_producers_cancelled"
        )
        for waiter in list(self._waiters[channel]):
            if reachable < waiter.required_count:
                waiter.orphaned_reason = orphan_reason
                waiter.event.set()
        has_init_seed = channel in self._logs and any(
            entry.write_seq == 0 for entry in self._logs[channel]
        )
        if (
            not success
            and self._producers_remaining[channel] == 0
            and self._arrival_count[channel] == 0
            and not has_init_seed
        ):
            self._orphaned[channel] = orphan_reason
