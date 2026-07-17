# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-trace state container, sentinel exception, and node-result shape for the
async-dataflow executor.

Three small types co-located to avoid circular imports between the executor,
the channel store, and the node-dispatch table.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.dataset.graph.models import TraceRecord
    from aiperf.graph.channel_store import VersionedChannelStore


class _NodeExpectedExit(BaseException):
    """Signals a _fire task exiting cleanly without writing successors.

    Base class for clean-exit signals a node can raise to terminate without
    propagating to successors. Inherits BaseException (not Exception) so the
    asyncio.TaskGroup cascade does not treat it as a programming error.

    The sole concrete raiser today is the ``_NodeOverflowTerminate`` subclass
    (defined in ``aiperf.graph.credit_dispatch_adapter``), which the credit
    path raises on a context-overflow response so the executor terminates the
    whole trajectory cleanly.
    """


@dataclass(slots=True, frozen=True)
class Write:
    """A single channel write produced by a node's _execute return.

    Frozen so accidental mutation between produce and consume is impossible.
    """

    channel: str
    value: object


@dataclass(slots=True)
class NodeExecutionResult:
    """The return value of every node-kind's _execute.

    `writes` carries the channel writes the node produced (possibly empty).
    """

    writes: list[Write] = field(default_factory=list)


@dataclass(slots=True)
class _TraceContext:
    """Per-trace mutable state passed into every _fire call.

    Owned by `TraceExecutor.run` for the duration of a single trace; never
    shared across traces. Mutable fields default to empty collections.
    """

    trace: TraceRecord
    store: VersionedChannelStore
    tg: asyncio.TaskGroup | None = None
    scheduled_node_ids: set[str] = field(default_factory=set)
    tasks_by_node_id: dict[str, asyncio.Task] = field(default_factory=dict)
    # Set True when any LlmNode in this trace returned a context-overflow error
    # (early termination). The overflowed node exits cleanly via
    # ``_NodeOverflowTerminate``; this flag lets the executor treat the resulting
    # downstream ``ChannelOrphanedError`` cascade (successors awaiting the
    # never-written output channel) as a CLEAN trajectory stop rather than a trace
    # error, so the rest of the trace's turns do not dispatch and the instance is
    # not counted as ``errored_traces``.
    overflow_terminated: bool = False
    # Wall-clock-us at which each node's `_fire` reached its `finally` block,
    # i.e. the moment it actually finished executing (success, expected exit,
    # or race-cancel). Read by successor `_apply_firing_delay` to anchor
    # incoming-edge `delay_after_predecessor_us` gates on the predecessor's
    # ACTUAL finish time.
    # No lock needed: writes happen in the predecessor's `_fire` `finally`
    # before successor scheduling (`_schedule(succ_id, ...)` runs AFTER
    # `mark_producer_done`, which runs in the same `finally`), so the
    # successor's `await_inputs` is guaranteed to see the write.
    node_finish_wall_us: dict[str, float] = field(default_factory=dict)
    # Wall-clock-us at which each node's firing gate cleared and it proceeded
    # to execute (its dispatch instant). Written in `_prepare_node_inputs`
    # immediately after `_apply_firing_delay` returns and BEFORE start-anchored
    # successors are scheduled on the same loop iteration, so a successor's
    # `_compute_firing_gate_us` read is guaranteed to see the write (same
    # single-loop happens-before argument as `node_finish_wall_us`).
    node_dispatch_wall_us: dict[str, float] = field(default_factory=dict)
    # Wall-clock-us at which each node's first token was observed (FirstToken
    # event routed through the dispatch adapter's stamp closure). Written on
    # the same single loop that reads it in successor gate computation --
    # same happens-before argument as node_dispatch_wall_us. A node absent
    # from this map after its first-token event is set terminated without
    # streaming a first token (fallback gate applies).
    node_first_token_wall_us: dict[str, float] = field(default_factory=dict)
    # Per-node first-token latches. SET by the stamp closure (first token
    # observed) or by _finalize_node (terminal without one). Successors with
    # first-token-anchored incoming edges await these before computing their
    # firing gate.
    node_first_token_events: dict[str, asyncio.Event] = field(default_factory=dict)

    def first_token_event(self, node_id: str) -> asyncio.Event:
        """Lazy per-node first-token latch; single-loop access needs no lock."""
        return self.node_first_token_events.setdefault(node_id, asyncio.Event())
