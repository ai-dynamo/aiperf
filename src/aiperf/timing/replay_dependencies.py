# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorded interval-order dependencies for agentic replay."""

from __future__ import annotations

import asyncio
import math
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import ConversationBranchMode, ReplayDependencyEvent
from aiperf.common.models.dataset_models import ReplayTurnReference

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.common.models import (
        ConversationBranchInfo,
        ConversationMetadata,
        DatasetMetadata,
        TurnMetadata,
    )
    from aiperf.credit.structs import Credit, TurnToSend

_logger = AIPerfLogger(__name__)
_MAX_TIMESTAMP_NS = 2**63 - 1


@dataclass(frozen=True, slots=True, order=True)
class ReplayTurnKey:
    """Stable dataset identity for one replayed request."""

    conversation_id: str
    """Template conversation ID this request belongs to."""
    turn_index: int
    """Zero-based turn position within the conversation."""


@dataclass(frozen=True, slots=True, order=True)
class ReplayResumeBoundary:
    """Completed prefix of one replay stream at a phase boundary."""

    conversation_id: str
    """Template conversation ID of the replay stream."""
    next_turn_index: int
    """Index of the first turn not yet completed at the phase boundary."""


@dataclass(frozen=True, slots=True)
class RecordedTurnInterval:
    """One request interval on a logical replay stream."""

    key: ReplayTurnKey
    """Dataset identity (conversation + turn) of this interval."""
    stream_id: str
    """Logical stream this interval belongs to (root or subagent chain)."""
    start_ms: float | None
    """Recorded wall-clock start offset in ms; None when unknown."""
    api_time_ms: float | None
    """Recorded server processing duration in ms; None when unknown."""

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
    """A dispatch held back until its recorded predecessors complete."""

    turn: TurnToSend
    """The turn queued for dispatch once its barrier clears."""
    issue: Callable[[], Awaitable[bool]]
    """Coroutine factory that issues the credit; returns True on acceptance."""
    on_refused: Callable[[], Awaitable[None]] | None
    """Optional callback run when the dispatch is refused or cancelled."""


@dataclass(slots=True)
class _RootBarrierState:
    """Per-tree completion frontier and the dispatches waiting on it."""

    completed: set[ReplayTurnKey]
    """Keys of requests on this tree that have recorded completion."""
    pending: dict[ReplayTurnKey, _PendingDispatch]
    """Dispatches keyed by request, waiting on their predecessors to complete."""
    dispatch_events: dict[ReplayTurnKey, int] = field(default_factory=dict)
    completion_events: dict[ReplayTurnKey, int] = field(default_factory=dict)
    root_start_perf_ns: int | None = None


class ReplayBarrierCoordinator:
    """Release requests only after their recorded frontier has completed."""

    def __init__(
        self,
        dataset_metadata: DatasetMetadata,
        *,
        strict_finite: bool = False,
        scheduler: LoopScheduler | None = None,
        fail_finite: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._predecessors: dict[ReplayTurnKey, tuple[ReplayTurnKey, ...]] = {}
        self._dependencies: dict[ReplayTurnKey, tuple[ReplayTurnReference, ...]] = {}
        self._floor_ns: dict[ReplayTurnKey, int] = {}
        self._strict_finite = strict_finite
        self._scheduler = scheduler
        if strict_finite and scheduler is None:
            raise ValueError("Finite replay barrier requires an absolute scheduler")
        self._fail_finite = fail_finite or self._raise_finite
        self._closed_roots: set[str] = set()
        self._credits_by_id: dict[int, Credit] = {}
        self._scheduled_finite: set[tuple[str, ReplayTurnKey]] = set()
        for conversation in dataset_metadata.conversations:
            for turn_index, turn in enumerate(conversation.turns):
                key = ReplayTurnKey(conversation.conversation_id, turn_index)
                self._dependencies[key] = tuple(turn.replay_predecessors)
                self._predecessors[key] = tuple(
                    ReplayTurnKey(ref.conversation_id, ref.turn_index)
                    for ref in turn.replay_predecessors
                )
        if strict_finite:
            self._build_finite_graph(dataset_metadata)
        self._roots: dict[str, _RootBarrierState] = {}
        self._dispatch_tasks: set[asyncio.Task] = set()
        self._finite_dispatch_lateness_count = 0
        self._finite_dispatch_lateness_total_ns = 0
        self._finite_dispatch_lateness_max_ns = 0
        self._finite_clock_spread_max_ns = 0
        self._active = False
        self._releases_paused = False

    @staticmethod
    def _raise_finite(error: BaseException) -> None:
        raise error

    def _build_finite_graph(self, dataset_metadata: DatasetMetadata) -> None:
        conversations = {
            conversation.conversation_id: conversation
            for conversation in dataset_metadata.conversations
        }
        if len(conversations) != len(dataset_metadata.conversations):
            raise RuntimeError("Finite replay conversation IDs must be unique")
        roots = self._resolve_conversation_roots(conversations)
        root_timestamps = self._validate_root_timestamps(conversations, roots)
        self._add_spawn_dependencies(dataset_metadata, conversations, roots)
        self._predecessors = {
            key: tuple(
                ReplayTurnKey(reference.conversation_id, reference.turn_index)
                for reference in references
            )
            for key, references in self._dependencies.items()
        }
        graph = self._build_turn_dependency_graph(conversations, roots, root_timestamps)
        self._validate_dependency_graph(graph)

    @staticmethod
    def _resolve_conversation_roots(
        conversations: dict[str, ConversationMetadata],
    ) -> dict[str, str]:
        def root_id(conversation_id: str) -> str:
            seen: set[str] = set()
            current = conversations[conversation_id]
            while current.parent_conversation_id is not None:
                if current.conversation_id in seen:
                    raise RuntimeError("Finite replay conversation parent cycle")
                seen.add(current.conversation_id)
                parent = conversations.get(current.parent_conversation_id)
                if parent is None:
                    raise RuntimeError(
                        f"Finite replay parent {current.parent_conversation_id!r} is missing"
                    )
                current = parent
            return current.conversation_id

        return {
            conversation_id: root_id(conversation_id)
            for conversation_id in conversations
        }

    @staticmethod
    def _validate_root_timestamps(
        conversations: dict[str, ConversationMetadata], roots: dict[str, str]
    ) -> dict[str, float]:
        root_timestamps: dict[str, float] = {}
        for conversation_id, conversation in conversations.items():
            if roots[conversation_id] != conversation_id:
                continue
            if not conversation.turns:
                raise RuntimeError(
                    f"Finite replay root {conversation_id!r} has no turns"
                )
            timestamp = conversation.turns[0].timestamp_ms
            if not isinstance(timestamp, int | float) or isinstance(timestamp, bool):
                raise RuntimeError(
                    f"Finite replay root {conversation_id!r} turn 0 has no timestamp_ms"
                )
            if not math.isfinite(float(timestamp)):
                raise RuntimeError(
                    f"Finite replay root {conversation_id!r} turn 0 timestamp_ms is not finite"
                )
            root_timestamps[conversation_id] = float(timestamp)
        return root_timestamps

    def _add_spawn_dependencies(
        self,
        dataset_metadata: DatasetMetadata,
        conversations: dict[str, ConversationMetadata],
        roots: dict[str, str],
    ) -> None:
        for parent in dataset_metadata.conversations:
            if any(
                getattr(branch, "dispatch_timing", "post") == "pre"
                for branch in parent.branches
            ):
                raise RuntimeError(
                    f"Finite replay does not support pre-session branches in "
                    f"conversation {parent.conversation_id!r}"
                )
            branches_by_id = {branch.branch_id: branch for branch in parent.branches}
            self._add_parent_spawn_dependencies(
                parent, branches_by_id, conversations, roots
            )

    def _add_parent_spawn_dependencies(
        self,
        parent: ConversationMetadata,
        branches_by_id: dict[str, ConversationBranchInfo],
        conversations: dict[str, ConversationMetadata],
        roots: dict[str, str],
    ) -> None:
        for parent_turn_index, parent_turn in enumerate(parent.turns):
            parent_start_value = parent_turn.timestamp_ms
            if (
                not isinstance(parent_start_value, int | float)
                or isinstance(parent_start_value, bool)
                or not math.isfinite(float(parent_start_value))
            ):
                raise RuntimeError(
                    f"Finite replay turn {parent.conversation_id!r}/{parent_turn_index} "
                    "has no finite timestamp_ms"
                )
            parent_start_ms = float(parent_start_value)
            for branch_id in parent_turn.branch_ids:
                branch = branches_by_id.get(branch_id)
                if branch is None:
                    raise RuntimeError(
                        f"Finite replay turn {parent.conversation_id!r}/{parent_turn_index} "
                        f"references missing branch {branch_id!r}"
                    )
                if branch.mode != ConversationBranchMode.SPAWN:
                    continue
                if parent_turn.no_request:
                    raise RuntimeError(
                        f"Finite replay SPAWN parent "
                        f"{parent.conversation_id!r}/{parent_turn_index} "
                        "does not issue an HTTP request"
                    )
                for child_id in branch.child_conversation_ids:
                    self._add_spawn_child_dependency(
                        parent=parent,
                        parent_turn_index=parent_turn_index,
                        parent_start_ms=parent_start_ms,
                        branch_id=branch_id,
                        child_id=child_id,
                        conversations=conversations,
                        roots=roots,
                    )

    def _add_spawn_child_dependency(
        self,
        *,
        parent: ConversationMetadata,
        parent_turn_index: int,
        parent_start_ms: float,
        branch_id: str,
        child_id: str,
        conversations: dict[str, ConversationMetadata],
        roots: dict[str, str],
    ) -> None:
        child = conversations.get(child_id)
        if child is None or not child.turns:
            raise RuntimeError(
                f"Finite replay SPAWN branch {branch_id!r} references "
                f"missing child conversation {child_id!r}"
            )
        if roots[child_id] != roots[parent.conversation_id]:
            raise RuntimeError(
                f"Finite replay SPAWN branch {branch_id!r} crosses root traces"
            )
        child_start_value = child.turns[0].timestamp_ms
        if (
            not isinstance(child_start_value, int | float)
            or isinstance(child_start_value, bool)
            or not math.isfinite(float(child_start_value))
        ):
            raise RuntimeError(
                f"Finite replay SPAWN child {child_id!r} turn 0 "
                "has no finite timestamp_ms"
            )
        delay_ms = float(child_start_value) - parent_start_ms
        delay_ns_value = delay_ms * 1_000_000
        if (
            not math.isfinite(delay_ms)
            or not math.isfinite(delay_ns_value)
            or delay_ms < 0
            or delay_ns_value > _MAX_TIMESTAMP_NS
        ):
            raise RuntimeError(
                f"Finite replay SPAWN child {child_id!r} starts before "
                f"its declaring turn {parent.conversation_id!r}/{parent_turn_index}"
            )
        child_key = ReplayTurnKey(child_id, 0)
        reference = ReplayTurnReference(
            conversation_id=parent.conversation_id,
            turn_index=parent_turn_index,
            event=ReplayDependencyEvent.DISPATCH,
            delay_ns=int(delay_ns_value),
        )
        dependencies = list(self._dependencies[child_key])
        if reference not in dependencies:
            dependencies.append(reference)
            self._dependencies[child_key] = tuple(dependencies)

    def _build_turn_dependency_graph(
        self,
        conversations: dict[str, ConversationMetadata],
        roots: dict[str, str],
        root_timestamps: dict[str, float],
    ) -> dict[ReplayTurnKey, set[ReplayTurnKey]]:
        graph: dict[ReplayTurnKey, set[ReplayTurnKey]] = {}
        for conversation_id, conversation in conversations.items():
            root_timestamp = root_timestamps[roots[conversation_id]]
            for turn_index, turn in enumerate(conversation.turns):
                key = ReplayTurnKey(conversation_id, turn_index)
                timestamp = self._validate_turn_floor(
                    key=key,
                    turn=turn,
                    turn_index=turn_index,
                    conversation_id=conversation_id,
                    roots=roots,
                    root_timestamp=root_timestamp,
                )
                graph[key] = self._validate_turn_dependencies(
                    key=key,
                    conversation_id=conversation_id,
                    timestamp=timestamp,
                    conversations=conversations,
                    roots=roots,
                )
        return graph

    def _validate_turn_floor(
        self,
        *,
        key: ReplayTurnKey,
        turn: TurnMetadata,
        turn_index: int,
        conversation_id: str,
        roots: dict[str, str],
        root_timestamp: float,
    ) -> float:
        timestamp_value = turn.timestamp_ms
        if not isinstance(timestamp_value, int | float) or isinstance(
            timestamp_value, bool
        ):
            raise RuntimeError(f"Finite replay turn {key!r} has no timestamp_ms")
        timestamp = float(timestamp_value)
        if not math.isfinite(timestamp):
            raise RuntimeError(f"Finite replay turn {key!r} timestamp is not finite")
        floor_ms = timestamp - root_timestamp
        floor_scaled = floor_ms * 1_000_000
        if (
            not math.isfinite(floor_ms)
            or not math.isfinite(floor_scaled)
            or floor_scaled > _MAX_TIMESTAMP_NS
        ):
            raise RuntimeError(f"Finite replay turn {key!r} has an invalid root floor")
        floor_ns = int(floor_scaled)
        if floor_ns < 0:
            raise RuntimeError(f"Finite replay turn {key!r} has an invalid root floor")
        if turn_index == 0 and turn.no_request:
            raise RuntimeError(
                f"Finite replay root {conversation_id!r} turn 0 must issue an HTTP request"
            )
        if (
            conversation_id == roots[conversation_id]
            and turn_index == 0
            and self._dependencies[key]
        ):
            raise RuntimeError(
                f"Finite replay root {conversation_id!r} turn 0 cannot have predecessors"
            )
        self._floor_ns[key] = floor_ns
        return timestamp

    def _validate_turn_dependencies(
        self,
        *,
        key: ReplayTurnKey,
        conversation_id: str,
        timestamp: float,
        conversations: dict[str, ConversationMetadata],
        roots: dict[str, str],
    ) -> set[ReplayTurnKey]:
        predecessors: set[ReplayTurnKey] = set()
        for reference in self._dependencies[key]:
            predecessor = ReplayTurnKey(reference.conversation_id, reference.turn_index)
            predecessor_conversation = conversations.get(reference.conversation_id)
            if predecessor_conversation is None or predecessor.turn_index >= len(
                predecessor_conversation.turns
            ):
                raise RuntimeError(
                    f"Finite replay dependency {predecessor!r} for {key!r} is missing"
                )
            predecessor_turn = predecessor_conversation.turns[predecessor.turn_index]
            predecessor_timestamp = predecessor_turn.timestamp_ms
            if (
                not isinstance(predecessor_timestamp, int | float)
                or isinstance(predecessor_timestamp, bool)
                or not math.isfinite(float(predecessor_timestamp))
            ):
                raise RuntimeError(
                    f"Finite replay dependency {predecessor!r} has no finite timestamp_ms"
                )
            if float(predecessor_timestamp) > timestamp:
                raise RuntimeError(
                    f"Finite replay dependency {predecessor!r} starts after {key!r}"
                )
            if predecessor_turn.no_request:
                raise RuntimeError(
                    f"Finite replay dependency {predecessor!r} for {key!r} "
                    "references a turn without a transport event"
                )
            if roots[reference.conversation_id] != roots[conversation_id]:
                raise RuntimeError(
                    f"Finite replay dependency {predecessor!r} crosses root traces"
                )
            if predecessor == key:
                raise RuntimeError(f"Finite replay turn {key!r} depends on itself")
            predecessors.add(predecessor)
        return predecessors

    @staticmethod
    def _validate_dependency_graph(
        graph: dict[ReplayTurnKey, set[ReplayTurnKey]],
    ) -> None:
        visiting: set[ReplayTurnKey] = set()
        visited: set[ReplayTurnKey] = set()

        def visit(key: ReplayTurnKey) -> None:
            if key in visiting:
                raise RuntimeError(f"Finite replay dependency cycle at {key!r}")
            if key in visited:
                return
            visiting.add(key)
            for predecessor in graph[key]:
                visit(predecessor)
            visiting.remove(key)
            visited.add(key)

        for key in graph:
            visit(key)

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

    def activate_finite(self) -> None:
        self._active = True

    def pause_releases(self) -> None:
        """Retain newly ready dispatches for an explicit phase handoff."""
        self._releases_paused = True

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
        if self._strict_finite:
            if turn.agent_depth == 0 and turn.turn_index == 0:
                return await issue()
            if key in state.pending:
                raise RuntimeError(
                    f"Duplicate finite replay dispatch for root={root_id!r}, turn={key!r}"
                )
            state.pending[key] = _PendingDispatch(turn, issue, on_refused)
            self._schedule_finite_if_ready(root_id, state, key)
            return True
        if self._ready(state, key) and not self._releases_paused:
            return await issue()
        if key in state.pending:
            raise RuntimeError(
                f"Duplicate deferred replay dispatch for root={root_id!r}, turn={key!r}"
            )
        state.pending[key] = _PendingDispatch(
            turn=turn, issue=issue, on_refused=on_refused
        )
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
        if self._releases_paused:
            return
        ready = [key for key in state.pending if self._ready(state, key)]
        for key in sorted(ready):
            pending = state.pending.pop(key)
            task = asyncio.create_task(self._dispatch_pending(pending))
            self._dispatch_tasks.add(task)
            task.add_done_callback(self._dispatch_tasks.discard)

    def close_root(self, root_id: str) -> None:
        """Discard completed runtime state when a recycled tree drains."""
        self._roots.pop(root_id, None)
        if self._strict_finite:
            self._closed_roots.add(root_id)
            self._credits_by_id = {
                credit_id: credit
                for credit_id, credit in self._credits_by_id.items()
                if credit.effective_root_correlation_id != root_id
            }

    def register_credit(self, credit: Credit) -> None:
        if not self._strict_finite or not credit.finite_replay:
            return
        previous = self._credits_by_id.get(credit.id)
        if previous is not None and previous != credit:
            raise RuntimeError(f"Finite replay credit ID {credit.id} was reused")
        self._credits_by_id[credit.id] = credit

    def unregister_credit(self, credit: Credit) -> None:
        if self._credits_by_id.get(credit.id) == credit:
            self._credits_by_id.pop(credit.id, None)

    def credit_for_id(self, credit_id: int) -> Credit | None:
        return self._credits_by_id.get(credit_id)

    def record_dispatch(
        self, credit: Credit, perf_ns: int, clock_spread_ns: int
    ) -> None:
        self._validate_transport_event("transport start", perf_ns, clock_spread_ns)
        root_id = credit.effective_root_correlation_id
        if root_id in self._closed_roots:
            _logger.warning("Dropping late finite dispatch for closed root %s", root_id)
            return
        state = self._roots.setdefault(root_id, _RootBarrierState(set(), {}))
        key = ReplayTurnKey(credit.conversation_id, credit.turn_index)
        if self._remember_first(state.dispatch_events, key, perf_ns):
            self._finite_clock_spread_max_ns = max(
                self._finite_clock_spread_max_ns, clock_spread_ns
            )
            if credit.agent_depth == 0 and credit.turn_index == 0:
                state.root_start_perf_ns = perf_ns
            self._release_finite_ready(root_id, state)

    def record_completion(
        self,
        credit: Credit,
        eof_perf_ns: int | None,
        *,
        clock_spread_ns: int | None,
        failed: bool,
    ) -> None:
        root_id = credit.effective_root_correlation_id
        if root_id in self._closed_roots:
            _logger.warning(
                "Dropping late finite completion for closed root %s", root_id
            )
            return
        state = self._roots.setdefault(root_id, _RootBarrierState(set(), {}))
        key = ReplayTurnKey(credit.conversation_id, credit.turn_index)
        if not credit.no_request and key not in state.dispatch_events:
            raise RuntimeError(f"Finite replay transport start missing for {key!r}")
        if failed:
            raise RuntimeError(f"Finite replay terminal request failed for {key!r}")
        if eof_perf_ns is None:
            if not credit.no_request:
                raise RuntimeError(f"Finite replay response EOF missing for {key!r}")
            state.completed.add(key)
            return
        self._validate_transport_event("response EOF", eof_perf_ns, clock_spread_ns)
        assert clock_spread_ns is not None
        dispatch_perf_ns = state.dispatch_events.get(key)
        if dispatch_perf_ns is not None and eof_perf_ns < dispatch_perf_ns:
            raise RuntimeError(
                f"Finite replay response EOF precedes transport start for {key!r}"
            )
        state.completed.add(key)
        if self._remember_first(state.completion_events, key, eof_perf_ns):
            self._finite_clock_spread_max_ns = max(
                self._finite_clock_spread_max_ns, clock_spread_ns
            )
            self._release_finite_ready(root_id, state)

    def finite_diagnostics(self) -> tuple[int, int, int, int]:
        """Return dispatch lateness count, total, max, and max clock spread."""
        return (
            self._finite_dispatch_lateness_count,
            self._finite_dispatch_lateness_total_ns,
            self._finite_dispatch_lateness_max_ns,
            self._finite_clock_spread_max_ns,
        )

    def has_pending_finite_work(self) -> bool:
        return bool(
            any(state.pending for state in self._roots.values())
            or self._scheduled_finite
            or self._dispatch_tasks
        )

    def fail_finite(self, error: BaseException) -> None:
        self._fail_finite(error)

    @staticmethod
    def _validate_transport_event(
        label: str, perf_ns: int, clock_spread_ns: int | None
    ) -> None:
        if (
            not isinstance(perf_ns, int)
            or isinstance(perf_ns, bool)
            or not 0 < perf_ns <= _MAX_TIMESTAMP_NS
        ):
            raise RuntimeError(f"Finite replay {label} has an invalid timestamp")
        if (
            not isinstance(clock_spread_ns, int)
            or isinstance(clock_spread_ns, bool)
            or not 0 <= clock_spread_ns <= _MAX_TIMESTAMP_NS
        ):
            raise RuntimeError(f"Finite replay {label} has an invalid clock spread")

    @staticmethod
    def _remember_first(
        events: dict[ReplayTurnKey, int], key: ReplayTurnKey, perf_ns: int
    ) -> bool:
        previous = events.get(key)
        if previous is None:
            events[key] = perf_ns
            return True
        if previous != perf_ns:
            _logger.warning(
                "Conflicting finite replay event for %r; retaining first timestamp %d",
                key,
                previous,
            )
        return False

    def _finite_deadline(
        self, state: _RootBarrierState, key: ReplayTurnKey
    ) -> int | None:
        if state.root_start_perf_ns is None:
            return None
        root_deadline = state.root_start_perf_ns + self._floor_ns[key]
        if root_deadline > _MAX_TIMESTAMP_NS:
            raise RuntimeError(f"Finite replay deadline overflows for {key!r}")
        deadlines = [root_deadline]
        for reference in self._dependencies[key]:
            predecessor = ReplayTurnKey(reference.conversation_id, reference.turn_index)
            events = (
                state.dispatch_events
                if reference.event == ReplayDependencyEvent.DISPATCH
                else state.completion_events
            )
            event_ns = events.get(predecessor)
            if event_ns is None:
                return None
            dependency_deadline = event_ns + reference.delay_ns
            if dependency_deadline > _MAX_TIMESTAMP_NS:
                raise RuntimeError(
                    f"Finite replay dependency deadline overflows for {key!r}"
                )
            deadlines.append(dependency_deadline)
        return max(deadlines)

    def _schedule_finite_if_ready(
        self, root_id: str, state: _RootBarrierState, key: ReplayTurnKey
    ) -> None:
        if self._releases_paused:
            return
        pending = state.pending.get(key)
        if pending is None:
            return
        due_ns = self._finite_deadline(state, key)
        if due_ns is None:
            return
        scheduled_key = (root_id, key)
        if scheduled_key in self._scheduled_finite:
            return
        self._scheduled_finite.add(scheduled_key)
        task = self._scheduler.schedule_at_perf_ns(
            due_ns, self._dispatch_finite_pending(root_id, key, pending)
        )
        if isinstance(task, asyncio.Task):
            self._dispatch_tasks.add(task)
            task.add_done_callback(self._dispatch_tasks.discard)

    def _release_finite_ready(self, root_id: str, state: _RootBarrierState) -> None:
        for key in tuple(state.pending):
            self._schedule_finite_if_ready(root_id, state, key)

    async def _dispatch_finite_pending(
        self, root_id: str, key: ReplayTurnKey, pending: _PendingDispatch
    ) -> None:
        self._scheduled_finite.discard((root_id, key))
        state = self._roots.get(root_id)
        if state is None or state.pending.get(key) is not pending:
            return
        deadline_ns = self._finite_deadline(state, key)
        if deadline_ns is None:
            self._fail_finite(
                RuntimeError(f"Finite replay dispatch deadline missing for {key!r}")
            )
            return
        now_ns = time.perf_counter_ns()
        if now_ns < deadline_ns:
            self._schedule_finite_if_ready(root_id, state, key)
            return
        state.pending.pop(key)
        lateness_ns = now_ns - deadline_ns
        self._finite_dispatch_lateness_count += 1
        self._finite_dispatch_lateness_total_ns += lateness_ns
        self._finite_dispatch_lateness_max_ns = max(
            self._finite_dispatch_lateness_max_ns, lateness_ns
        )
        try:
            issued = await pending.issue()
        except Exception as exc:
            self._fail_finite(exc)
            if pending.on_refused is not None:
                try:
                    await pending.on_refused()
                except Exception as cleanup_error:
                    self._fail_finite(cleanup_error)
            return
        if not issued and pending.on_refused is not None:
            try:
                await pending.on_refused()
            except Exception as exc:
                self._fail_finite(exc)

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

    def pending_turns(self, root_id: str) -> tuple[TurnToSend, ...]:
        """Return barrier-retained turns that have not gone on wire yet."""
        state = self._roots.get(root_id)
        if state is None:
            return ()
        return tuple(pending.turn for key, pending in sorted(state.pending.items()))

    def pending_turns_by_root(self) -> dict[str, tuple[TurnToSend, ...]]:
        """Return all barrier-retained turns grouped by runtime root id."""
        return {
            root_id: tuple(
                pending.turn for key, pending in sorted(state.pending.items())
            )
            for root_id, state in self._roots.items()
            if state.pending
        }

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
        try:
            issued = await pending.issue()
        except Exception:
            # This runs detached in a task whose only done-callback discards it,
            # so a raise here would be swallowed ("Task exception was never
            # retrieved") and any parent join waiting on this stream would hang
            # until the drain timeout. Treat an issue failure as a refusal so
            # on_refused cleanup runs and the phase can fail fast.
            _logger.exception(
                "Barrier-released replay dispatch failed for %r", pending.turn
            )
            issued = False
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

    def pause_releases(self) -> None:
        """Retain ready barrier work instead of issuing it immediately."""
        if self._coordinator is not None:
            self._coordinator.pause_releases()

    async def submit(
        self,
        turn: TurnToSend,
        issue: Callable[[], Awaitable[bool]],
        *,
        child_refusal_cleanup: bool = False,
        on_refused: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        if self._coordinator is None:
            return await issue()
        if (
            on_refused is None
            and child_refusal_cleanup
            and self._child_refused is not None
        ):

            async def on_refused() -> None:
                await self._child_refused(turn.x_correlation_id)

        return await self._coordinator.submit(turn, issue, on_refused=on_refused)

    def activate(self) -> None:
        if self._coordinator is not None:
            self._coordinator.activate()

    def complete(self, credit: Credit) -> None:
        if self._coordinator is not None:
            self._coordinator.complete(credit)

    def register_credit(self, credit: Credit) -> None:
        if self._coordinator is not None:
            self._coordinator.register_credit(credit)

    def unregister_credit(self, credit: Credit) -> None:
        if self._coordinator is not None:
            self._coordinator.unregister_credit(credit)

    def credit_for_id(self, credit_id: int) -> Credit | None:
        if self._coordinator is None:
            return None
        return self._coordinator.credit_for_id(credit_id)

    def record_dispatch(
        self, credit: Credit, perf_ns: int, clock_spread_ns: int
    ) -> None:
        if self._coordinator is not None:
            self._coordinator.record_dispatch(credit, perf_ns, clock_spread_ns)

    def record_completion(
        self,
        credit: Credit,
        eof_perf_ns: int | None,
        *,
        clock_spread_ns: int | None,
        failed: bool,
    ) -> None:
        if self._coordinator is not None:
            self._coordinator.record_completion(
                credit,
                eof_perf_ns,
                clock_spread_ns=clock_spread_ns,
                failed=failed,
            )

    def has_pending_finite_work(self) -> bool:
        return bool(self._coordinator and self._coordinator.has_pending_finite_work())

    def finite_diagnostics(self) -> tuple[int, int, int, int]:
        if self._coordinator is None:
            return 0, 0, 0, 0
        return self._coordinator.finite_diagnostics()

    def fail_finite(self, error: BaseException) -> None:
        if self._coordinator is not None:
            self._coordinator.fail_finite(error)

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

    def pending_turns(self, root_correlation_id: str) -> tuple[TurnToSend, ...]:
        if self._coordinator is None:
            return ()
        return self._coordinator.pending_turns(root_correlation_id)

    def pending_turns_by_root(self) -> dict[str, tuple[TurnToSend, ...]]:
        if self._coordinator is None:
            return {}
        return self._coordinator.pending_turns_by_root()

    async def cancel(self, *, notify_refused: bool) -> None:
        if self._coordinator is not None:
            await self._coordinator.cancel_pending(notify_refused=notify_refused)

    async def observe_issued(self, credit: Credit) -> None:
        if self._credit_issued is not None and not credit.finite_replay:
            await self._credit_issued(credit)
