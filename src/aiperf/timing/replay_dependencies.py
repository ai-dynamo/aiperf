# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorded interval-order dependencies for agentic replay."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.common.models import DatasetMetadata
    from aiperf.credit.structs import Credit, TurnToSend

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, order=True)
class ReplayTurnKey:
    """Stable dataset identity for one replayed request."""

    conversation_id: str
    turn_index: int


@dataclass(frozen=True, slots=True, order=True)
class ReplayResumeBoundary:
    """Completed prefix of one replay stream at a phase boundary."""

    conversation_id: str
    next_turn_index: int


@dataclass(frozen=True, slots=True)
class RecordedTurnInterval:
    """One request interval on a logical replay stream."""

    key: ReplayTurnKey
    stream_id: str
    start_ms: float | None
    api_time_ms: float | None

    @property
    def normalized_interval(self) -> tuple[float, float] | None:
        """Return ``[start, end]`` using the Weka duration fallback policy."""
        if self.start_ms is None or not math.isfinite(self.start_ms):
            return None
        duration_ms = self.api_time_ms
        if duration_ms is None or not math.isfinite(duration_ms) or duration_ms < 0:
            duration_ms = 0.0
        return self.start_ms, self.start_ms + duration_ms


def infer_cross_stream_predecessors(
    intervals: list[RecordedTurnInterval],
) -> dict[ReplayTurnKey, tuple[ReplayTurnKey, ...]]:
    """Infer the recorded completion frontier each request must join.

    Per-stream ordering remains owned by normal conversation replay. For every
    other stream, a request depends on that stream's latest request known to
    have completed by its recorded start. Overlapping intervals create no edge.
    This represents transitive overlap precisely: a long request may overlap
    several sequential requests on another stream without forcing those later
    requests into one simultaneously launched connected component.

    Exact boundary touches are ordered. Equal starts are unordered, including
    zero-width intervals. Missing or non-finite starts add no cross-stream edge;
    missing, negative, or non-finite durations are deterministic zero-width
    intervals, matching the Weka loader's request-end fallback.
    """
    by_stream: dict[str, list[tuple[RecordedTurnInterval, float, float]]] = {}
    for interval in intervals:
        normalized = interval.normalized_interval
        if normalized is None:
            continue
        start_ms, end_ms = normalized
        by_stream.setdefault(interval.stream_id, []).append(
            (interval, start_ms, end_ms)
        )

    dependencies: dict[ReplayTurnKey, tuple[ReplayTurnKey, ...]] = {}
    for target in intervals:
        target_interval = target.normalized_interval
        if target_interval is None:
            dependencies[target.key] = ()
            continue
        target_start_ms, _ = target_interval
        frontier: list[tuple[RecordedTurnInterval, float, float]] = []
        for stream_id, candidates in by_stream.items():
            if stream_id == target.stream_id:
                continue
            completed = [
                candidate
                for candidate in candidates
                if candidate[1] < target_start_ms and candidate[2] <= target_start_ms
            ]
            if not completed:
                continue
            latest = max(
                completed,
                key=lambda candidate: (
                    candidate[2],
                    candidate[1],
                    candidate[0].key,
                ),
            )
            frontier.append(latest)
        predecessors = [
            candidate[0].key
            for candidate in frontier
            if not any(
                candidate[1] < later[1] and candidate[2] <= later[1]
                for later in frontier
                if later is not candidate
            )
        ]
        dependencies[target.key] = tuple(sorted(predecessors))
    return dependencies


@dataclass(slots=True)
class _PendingDispatch:
    issue: Callable[[], Awaitable[bool]]
    on_refused: Callable[[], Awaitable[None]] | None


@dataclass(slots=True)
class _RootBarrierState:
    completed: set[ReplayTurnKey]
    pending: dict[ReplayTurnKey, _PendingDispatch]


class ReplayBarrierCoordinator:
    """Release requests only after their recorded frontier has completed."""

    def __init__(self, dataset_metadata: DatasetMetadata) -> None:
        self._predecessors: dict[ReplayTurnKey, tuple[ReplayTurnKey, ...]] = {}
        for conversation in dataset_metadata.conversations:
            for turn_index, turn in enumerate(conversation.turns):
                key = ReplayTurnKey(conversation.conversation_id, turn_index)
                self._predecessors[key] = tuple(
                    ReplayTurnKey(ref.conversation_id, ref.turn_index)
                    for ref in turn.replay_predecessors
                )
        self._roots: dict[str, _RootBarrierState] = {}
        self._dispatch_tasks: set[asyncio.Task] = set()
        self._active = False

    def activate(self) -> None:
        """Enable barriers after baseline cache priming completes."""
        if self._active:
            return
        self._active = True
        widths = Counter(
            len(predecessors)
            for predecessors in self._predecessors.values()
            if predecessors
        )
        _logger.info(
            "Replay interval barriers active: %d requests, %d gated turns, "
            "join-widths=%s",
            len(self._predecessors),
            sum(widths.values()),
            dict(sorted(widths.items())),
        )

    async def submit(
        self,
        turn: TurnToSend,
        issue: Callable[[], Awaitable[bool]],
        *,
        on_refused: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        """Issue now when ready, otherwise retain one deferred dispatch."""
        if not self._active:
            return await issue()
        root_id = turn.effective_root_correlation_id
        state = self._roots.setdefault(
            root_id, _RootBarrierState(completed=set(), pending={})
        )
        key = ReplayTurnKey(turn.conversation_id, turn.turn_index)
        if self._ready(state, key):
            return await issue()
        if key in state.pending:
            raise RuntimeError(
                f"Duplicate deferred replay dispatch for root={root_id!r}, turn={key!r}"
            )
        state.pending[key] = _PendingDispatch(issue=issue, on_refused=on_refused)
        return True

    def complete(self, credit: Credit) -> None:
        """Record any terminal request outcome and release newly ready work."""
        if not self._active:
            return
        root_id = credit.effective_root_correlation_id
        state = self._roots.setdefault(
            root_id, _RootBarrierState(completed=set(), pending={})
        )
        state.completed.add(ReplayTurnKey(credit.conversation_id, credit.turn_index))
        ready = [key for key in state.pending if self._ready(state, key)]
        for key in sorted(ready):
            pending = state.pending.pop(key)
            task = asyncio.create_task(self._dispatch_pending(pending))
            self._dispatch_tasks.add(task)
            task.add_done_callback(self._dispatch_tasks.discard)

    def close_root(self, root_id: str) -> None:
        """Discard completed runtime state when a recycled tree drains."""
        self._roots.pop(root_id, None)

    def seed_completed_prefixes(
        self,
        root_id: str,
        boundaries: tuple[ReplayResumeBoundary, ...],
    ) -> None:
        """Seed exact pre-resume history before any turn can be submitted."""
        state = self._roots.setdefault(
            root_id, _RootBarrierState(completed=set(), pending={})
        )
        if state.pending:
            raise RuntimeError(
                f"Cannot seed replay history after dispatch for root={root_id!r}"
            )
        for boundary in boundaries:
            if boundary.next_turn_index < 0:
                raise ValueError(
                    "Replay resume boundary must have a non-negative turn index"
                )
            state.completed.update(
                ReplayTurnKey(boundary.conversation_id, turn_index)
                for turn_index in range(boundary.next_turn_index)
            )

    def completed_prefixes(self, root_id: str) -> tuple[ReplayResumeBoundary, ...]:
        """Return the contiguous completed prefix of every replay stream."""
        state = self._roots.get(root_id)
        if state is None:
            return ()
        next_turn_by_conversation: dict[str, int] = {}
        for key in state.completed:
            next_turn_by_conversation[key.conversation_id] = max(
                next_turn_by_conversation.get(key.conversation_id, 0),
                key.turn_index + 1,
            )
        for conversation_id, next_turn_index in next_turn_by_conversation.items():
            if any(
                ReplayTurnKey(conversation_id, turn_index) not in state.completed
                for turn_index in range(next_turn_index)
            ):
                raise RuntimeError(
                    "Replay completion history is not a contiguous stream prefix: "
                    f"root={root_id!r}, conversation={conversation_id!r}"
                )
        return tuple(
            ReplayResumeBoundary(conversation_id, next_turn_index)
            for conversation_id, next_turn_index in sorted(
                next_turn_by_conversation.items()
            )
        )

    async def cancel_pending(self, *, notify_refused: bool) -> None:
        """Cancel retained dispatches during phase teardown."""
        callbacks = []
        for state in self._roots.values():
            if notify_refused:
                callbacks.extend(
                    pending.on_refused
                    for pending in state.pending.values()
                    if pending.on_refused is not None
                )
            state.pending.clear()
        for task in self._dispatch_tasks:
            task.cancel()
        self._dispatch_tasks.clear()
        for callback in callbacks:
            await callback()

    def _ready(self, state: _RootBarrierState, key: ReplayTurnKey) -> bool:
        return all(
            predecessor in state.completed
            for predecessor in self._predecessors.get(key, ())
        )

    @staticmethod
    async def _dispatch_pending(pending: _PendingDispatch) -> None:
        issued = await pending.issue()
        if not issued and pending.on_refused is not None:
            await pending.on_refused()


class ReplayIssueGate:
    """Small CreditIssuer adapter around a replay barrier coordinator."""

    def __init__(self, coordinator: ReplayBarrierCoordinator | None) -> None:
        self._coordinator = coordinator
        self._child_refused: Callable[[str], Awaitable[None]] | None = None
        self._credit_issued: Callable[[Credit], Awaitable[None]] | None = None

    @property
    def enabled(self) -> bool:
        return self._coordinator is not None

    def set_child_refused(self, callback: Callable[[str], Awaitable[None]]) -> None:
        self._child_refused = callback

    def set_credit_issued(self, callback: Callable[[Credit], Awaitable[None]]) -> None:
        self._credit_issued = callback

    async def submit(
        self,
        turn: TurnToSend,
        issue: Callable[[], Awaitable[bool]],
        *,
        child_refusal_cleanup: bool = False,
    ) -> bool:
        if self._coordinator is None:
            return await issue()
        on_refused = None
        if child_refusal_cleanup and self._child_refused is not None:

            async def on_refused() -> None:
                await self._child_refused(turn.x_correlation_id)

        return await self._coordinator.submit(turn, issue, on_refused=on_refused)

    def activate(self) -> None:
        if self._coordinator is not None:
            self._coordinator.activate()

    def complete(self, credit: Credit) -> None:
        if self._coordinator is not None:
            self._coordinator.complete(credit)

    def close_root(self, root_correlation_id: str) -> None:
        if self._coordinator is not None:
            self._coordinator.close_root(root_correlation_id)

    def seed_completed_prefixes(
        self,
        root_correlation_id: str,
        boundaries: tuple[ReplayResumeBoundary, ...],
    ) -> None:
        if self._coordinator is not None:
            self._coordinator.seed_completed_prefixes(root_correlation_id, boundaries)

    def completed_prefixes(
        self, root_correlation_id: str
    ) -> tuple[ReplayResumeBoundary, ...]:
        if self._coordinator is None:
            return ()
        return self._coordinator.completed_prefixes(root_correlation_id)

    async def cancel(self, *, notify_refused: bool) -> None:
        if self._coordinator is not None:
            await self._coordinator.cancel_pending(notify_refused=notify_refused)

    async def observe_issued(self, credit: Credit) -> None:
        if self._credit_issued is not None:
            await self._credit_issued(credit)
