# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TraceExecutor - async-dataflow per-trace executor for the graph package.

Nodes fire as soon as their input channels are ready. Per-trace
state lives on `_TraceContext`; the executor itself is shared across traces in
a phase (all graph-derived state is immutable post-`__init__`).

Per-kind ``_execute`` implementations live in
``aiperf.graph.dispatch``; see that package's
``__init__.py`` for the ownership map.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.clock import AIPerfClock, WallClock
from aiperf.common.environment import Environment
from aiperf.dataset.graph.models import (
    ChannelType,
    LlmNode,
    NodeUnion,
    ParsedGraph,
)
from aiperf.graph.channel_store import (
    VersionedChannelStore,
)
from aiperf.graph.channels import (
    producers_per_channel as compute_producers_per_channel,
)
from aiperf.graph.context import (
    NodeExecutionResult,
    _NodeExpectedExit,
    _TraceContext,
)
from aiperf.graph.scheduler import Scheduler

if TYPE_CHECKING:
    from aiperf.dataset.graph.models import (
        ChannelSpec,
        TraceRecord,
    )

__all__ = ["TraceExecutor", "TraceResult"]


def _default_store_factory(
    channel_specs: dict[str, ChannelSpec],
    initial: dict[str, Any],
    producers_per_channel: dict[str, int],
) -> VersionedChannelStore:
    return VersionedChannelStore(
        initial=initial,
        channel_specs=channel_specs,
        producers_per_channel=producers_per_channel,
    )


@dataclass(slots=True)
class TraceResult:
    """The return value of `TraceExecutor.run`.

    Carries the trace id and the post-run channel snapshot.
    """

    trace_id: str
    channels: dict[str, Any]


_logger = AIPerfLogger(__name__)


class TraceExecutor:
    """Async-dataflow trace executor. Shared across traces in a phase.

    All mutable per-trace state lives on `_TraceContext`; constructor-injected
    collaborators are graph-derived and immutable. Node-kind ``_execute``
    bodies are registered on the dispatch table by sibling modules under
    ``dispatch/`` without touching this class (each
    ``_execute.register(<NodeKind>)`` lives in its own dispatch module).
    """

    def __init__(
        self,
        parsed: ParsedGraph,
        *,
        credit_issuer: Any = None,
        scheduler: Scheduler | None = None,
        producers_per_channel: dict[str, int] | None = None,
        compress_edge_delays: bool = False,
        absolute_start_offsets: bool = False,
        clock: AIPerfClock | None = None,
    ) -> None:
        from aiperf.graph.dispatch import _import_dispatch_modules

        _import_dispatch_modules()
        self._parsed = parsed
        # Time source for the firing loop. Defaults to WallClock (reads
        # ``time.perf_counter_ns`` + ``asyncio.sleep`` -- behavior identical to
        # the prior direct ``time.*`` calls). A VirtualClock can be injected so
        # a driver pump fast-forwards sim time, letting a multi-hour recorded
        # trace replay in milliseconds while reproducing the exact firing
        # timeline.
        self._clock: AIPerfClock = clock if clock is not None else WallClock()
        # Anchor node-level ``min_start_delay_us`` to the INSTANCE run-start (the
        # t* / phase-start anchor) instead of each node's input-ready instant.
        # The PROFILING snapshot rewrite (``timing/snapshot_chop.py``,
        # adapter-agnostic) stamps every surviving frontier
        # turn's ``min_start_delay_us`` as its ABSOLUTE ``dispatch_offset_us`` from
        # t*.
        # Measuring it from input-ready double-counts the lead for any stream
        # whose first turn's inputs arrive late (a co-scoped subagent/worker
        # gated on its spawn), so those streams drift out of recorded-time order
        # and occupy lanes longer than recorded. Anchoring to the shared instance
        # run-start makes every stream's frontier turn fire at its absolute
        # recorded offset (clamped to input-readiness). Off by
        # default == relative semantics.
        self._absolute_start_offsets = absolute_start_offsets
        self._anchor_wall_us: float | None = None
        # Per-instance counterpart of the global ``AIPERF_GRAPH_IGNORE_EDGE_DELAYS``
        # env: when True this single executor collapses ALL incoming firing-edge
        # delays (zero-idle / burst pacing), independent of the env default. The
        # accelerated cache-pressure warmup sets this True ONLY on
        # the WARMUP-phase executors (knob-gated), so the live trajectories replay
        # with zero idle delay to drive the server's KV cache to pressure. Default
        # False == honor every captured edge delay exactly (byte-for-byte).
        self._compress_edge_delays = compress_edge_delays
        # `scheduler` and `producers_per_channel` are pure functions of the
        # immutable `parsed.graph`. When a
        # caller already memoized them per template_id, inject them to skip the
        # two O(graph) rescans on this per-bind instance. Both are read-only
        # after construction, so sharing them across instances is safe; only the
        # mutable `_credit_issuer` (monkeypatched per-run) stays per-instance.
        self._scheduler = scheduler or Scheduler(parsed.graph)
        self._credit_issuer = credit_issuer
        self._producers_per_channel = (
            producers_per_channel
            if producers_per_channel is not None
            else compute_producers_per_channel(parsed.graph)
        )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    async def run(
        self,
        trace: TraceRecord,
    ) -> TraceResult:
        """Drive one trace through the dataflow firing loop.

        Opens a per-trace `asyncio.TaskGroup`, schedules every entry node
        returned by `Scheduler.entry_nodes()`, and blocks until every `_fire`
        task finishes. The channel snapshot is captured once the frontier
        drive completes; a cancellation or unhandled trace error propagates
        out of `run` before a `TraceResult` is built.

        When `Environment.GRAPH.EXECUTOR_WATCHDOG_TIMEOUT` is set, the
        frontier drive is bounded by that wall-clock deadline: a firing loop
        wedged on an unsatisfiable channel input (a producer-accounting bug)
        raises `asyncio.TimeoutError` instead of hanging. Default `None`
        preserves the faithful, unbounded idle-gap replay bare/count/session
        runs rely on (see `GraphIRReplayStrategy` AgentX count-mode parity).
        """
        # Pin the instance t* origin once, on the first run, so every firing in
        # this executor shares one absolute-offset anchor (AgentX dispatches all
        # streams from t*).
        if self._absolute_start_offsets and self._anchor_wall_us is None:
            self._anchor_wall_us = self._loop_wall_us()

        store = _default_store_factory(
            self._parsed.graph.state,
            dict(trace.initial_state),
            self._producers_per_channel,
        )

        ctx = _TraceContext(trace=trace, store=store)

        watchdog_s = Environment.GRAPH.EXECUTOR_WATCHDOG_TIMEOUT
        if watchdog_s is not None:
            await asyncio.wait_for(self._drive_frontier(ctx), timeout=watchdog_s)
        else:
            await self._drive_frontier(ctx)

        return TraceResult(
            trace_id=trace.id,
            channels=store.snapshot(),
        )

    async def _drive_frontier(self, ctx: _TraceContext) -> None:
        """Open the per-trace TaskGroup and drive the readiness frontier.

        Schedules every entry node and blocks until every `_fire` finishes.
        """
        async with asyncio.TaskGroup() as tg:
            ctx.tg = tg
            for entry_id in self._scheduler.entry_nodes():
                self._schedule(entry_id, ctx)

    def _schedule(self, node_id: str, ctx: _TraceContext) -> None:
        """Schedule a node's `_fire` task on the per-trace TaskGroup.

        Dedup semantics combine §5.1 (AND-fan-in: silent skip when the same
        successor is reached via multiple predecessors) with §5.2 (cycle
        guard: re-entering an already-COMPLETED node is a `RuntimeError`).
        The disambiguator is `tasks_by_node_id[node_id].done()`:

        - In-flight task or no task entry yet -> legitimate concurrent
          scheduling from a fan-in predecessor; silently skip.
        - Completed task -> cycle; the loader's cycle check should have
          rejected this graph, so raise `RuntimeError`.
        """
        if node_id in ctx.scheduled_node_ids:
            existing = ctx.tasks_by_node_id.get(node_id)
            if existing is not None and existing.done():
                raise RuntimeError(
                    f"cycle detected: node {node_id!r} re-scheduled after "
                    "completing. The loader's cycle check should have "
                    "rejected this graph; a mixed-anchor fan-in (one "
                    "start-anchored plus one completion in-edge on the same "
                    "target, rejected at Scheduler construction) is another "
                    "likely cause."
                )
            # In-flight: AND-fan-in dedup, silent no-op.
            return

        ctx.scheduled_node_ids.add(node_id)
        assert ctx.tg is not None  # set inside run()'s TaskGroup
        task = ctx.tg.create_task(
            self._fire(node_id, ctx), name=f"fire:{ctx.trace.id}:{node_id}"
        )
        ctx.tasks_by_node_id[node_id] = task

    async def _prepare_node_inputs(
        self, node_id: str, node: NodeUnion, ctx: _TraceContext
    ) -> dict[str, Any]:
        requirements = node.inputs
        capture = await ctx.store.await_inputs(requirements)
        gate_seq = ctx.store.current_seq
        node_firable_wall_us = self._loop_wall_us()
        await self._apply_firing_delay(node_id, ctx, node_firable_wall_us)
        ctx.node_dispatch_wall_us[node_id] = self._loop_wall_us()
        for succ_id in self._scheduler.start_anchored_successors(node_id):
            self._schedule(succ_id, ctx)
        if isinstance(node, LlmNode):
            return ctx.store.snapshot_at_seq(gate_seq)
        return ctx.store.read(requirements, capture)

    # ------------------------------------------------------------------
    # _fire
    # ------------------------------------------------------------------
    async def _fire(self, node_id: str, ctx: _TraceContext) -> None:
        """Drive one node through the dataflow firing path."""
        node = self._parsed.graph.nodes[node_id]
        success = False
        wrote_channels: set[str] = set()
        try:
            result, wrote_channels = await self._run_node(node_id, node, ctx)
            success = result is not None
        finally:
            self._finalize_node(
                node_id,
                node,
                ctx,
                success=success,
                wrote_channels=wrote_channels,
            )
        if result is None:
            return
        self._schedule_successors(node_id, result, ctx)

    async def _run_node(
        self,
        node_id: str,
        node: NodeUnion,
        ctx: _TraceContext,
    ) -> tuple[NodeExecutionResult | None, set[str]]:
        from aiperf.graph.channel_store import ChannelOrphanedError
        from aiperf.graph.credit_dispatch_adapter import _NodeOverflowTerminate

        try:
            inputs = await self._prepare_node_inputs(node_id, node, ctx)
            result = await self._execute(node, inputs, ctx)
            wrote_channels = self._publish_writes(node_id, result, ctx)
            return result, wrote_channels
        except _NodeOverflowTerminate:
            # This node's response was a context-overflow. Flag the trace so
            # the downstream orphan cascade (successors awaiting this node's
            # never-written output) is also treated as a clean stop, and exit
            # cleanly WITHOUT scheduling successors -- the trajectory terminates
            # here, since later turns carry even more context and would only
            # overflow too.
            ctx.overflow_terminated = True
            return None, set()
        except _NodeExpectedExit:
            return None, set()
        except ChannelOrphanedError:
            # A successor of an overflow-terminated node will orphan on its
            # never-written input channel. That is the EXPECTED downstream
            # consequence of the early-termination, not a trace error: swallow it as
            # a clean exit so the whole trajectory stops without inflating
            # ``errored_traces``. Outside an overflow termination, re-raise so a
            # genuine producer-failure orphan still surfaces.
            if ctx.overflow_terminated:
                return None, set()
            raise
        except Exception as exc:
            return self._handle_node_exception(node_id, node, ctx=ctx, exc=exc)

    def _publish_writes(
        self, node_id: str, result: NodeExecutionResult, ctx: _TraceContext
    ) -> set[str]:
        wrote_channels: set[str] = set()
        for write in result.writes:
            ctx.store.write([write.channel], write.value, writer_node_id=node_id)
            wrote_channels.add(write.channel)
        return wrote_channels

    def _handle_node_exception(
        self,
        node_id: str,
        node: NodeUnion,
        *,
        ctx: _TraceContext,
        exc: Exception,
    ) -> tuple[NodeExecutionResult | None, set[str]]:
        from aiperf.graph.credit_dispatch_adapter import (
            CreditIssueRefusedError,
            GraphDispatchError,
            GraphStickinessError,
        )

        # Issuer refusal (duration / request-count caps, cancellation) and
        # broken stickiness (missing dynamic-pool value) are TRACE-STOPS,
        # never contained: sentinel-continuing past a refusal would churn
        # every remaining node without dispatching, and past a pool miss
        # would silently corrupt the workload's content dependencies.
        # Checked before the containment branch (both subclass
        # GraphDispatchError so the untyped check below would swallow them).
        if isinstance(exc, (CreditIssueRefusedError, GraphStickinessError)):
            raise exc

        # Mid-conversation resilience: a dispatch/request FAILURE (connection
        # reset, HTTP error, worker-cancel -> GraphDispatchError)
        # CONTAINS to this node instead of unwinding the whole per-trace
        # TaskGroup (which would lose every remaining turn + subagent of the
        # conversation). The failed request is ALREADY recorded as an error by
        # the RecordProcessor; the conversation continues to its next turn.
        # Returning a result (not raising) lets _finalize_node mark this node's
        # output channels producer-done so downstream await_inputs resolve, and
        # _schedule_successors fires the next turn. Keeping the instance alive
        # also preserves its adapter, so any still-in-flight returns still
        # resolve (no "no live adapter" drops). Faithful: downstream weka turns
        # splice the RECORDED @messages delta, not this node's live response, so
        # a failed turn does not corrupt downstream content. Structural /
        # programming errors (not GraphDispatchError) still raise.
        if isinstance(exc, GraphDispatchError):
            _logger.warning(
                lambda: (
                    f"node {node_id!r} dispatch failed; continuing "
                    f"conversation past the failed turn: {exc!r}"
                )
            )
            # Segment-store IRs ONLY: still PRODUCE this node's output channels
            # (a content-neutral sentinel) so downstream AND-fan-in readers
            # unblock instead of orphaning. These IRs wire each turn to read its
            # predecessors' ``{nid}_out`` channels purely as a completion GATE
            # (prompts come from the segment pool, not the channel), so an empty
            # sentinel lets the conversation continue past the failed turn
            # rather than cascade-erroring the whole trace. The sentinel is
            # TYPE-CORRECT per channel: ``[]`` for messages-typed channels (the
            # global ``snapshot_at_seq`` reduces EVERY channel and
            # ``add_messages`` rejects non-list values), ``None`` for the rest
            # (never read). Non-segment-store parses
            # keep the prior behavior (return empty -> downstream orphans ->
            # trace errors).
            if getattr(self._parsed, "segment_pool", None) is not None:
                wrote_channels: set[str] = set()
                for channel in node.write_channels:
                    spec = self._parsed.graph.state.get(channel)
                    sentinel = (
                        []
                        if spec is not None and spec.type is ChannelType.MESSAGES
                        else None
                    )
                    ctx.store.write([channel], sentinel, writer_node_id=node_id)
                    wrote_channels.add(channel)
                return NodeExecutionResult(), wrote_channels
            return NodeExecutionResult(), set()
        raise exc

    def _finalize_node(
        self,
        node_id: str,
        node: NodeUnion,
        ctx: _TraceContext,
        *,
        success: bool,
        wrote_channels: set[str],
    ) -> None:
        ctx.node_finish_wall_us[node_id] = self._loop_wall_us()
        # Latch the first-token event even when no first token was observed
        # (error/cancel/non-streaming terminal). Idempotent: the stamp closure
        # may have already set it. A successor with a first-token-anchored edge
        # awaits this latch; the absence of a `node_first_token_wall_us` entry
        # tells its gate computation to fall back to the dispatch anchor.
        ctx.first_token_event(node_id).set()
        for channel in node.write_channels:
            ctx.store.mark_producer_done(
                channel,
                success=success,
                wrote=channel in wrote_channels,
            )

    def _schedule_successors(
        self, node_id: str, result: NodeExecutionResult, ctx: _TraceContext
    ) -> None:
        for succ_id in self._scheduler.successors_after(node_id):
            self._schedule(succ_id, ctx)

    # ------------------------------------------------------------------
    # edge delay
    # ------------------------------------------------------------------
    async def _apply_firing_delay(
        self,
        node_id: str,
        ctx: _TraceContext,
        node_firable_wall_us: float,
    ) -> None:
        """Sleep until ``node_id``'s incoming-edge gate clears.

        Edge-gate semantics, anchored to the dataflow loop:

        - ``edge.delay_after_predecessor_us``: gate >= predecessor finish
          wall + delay. The predecessor's finish wall is recorded in its
          ``_fire`` ``finally`` block on
          ``ctx.node_finish_wall_us[edge.source]``. Because the successor
          is scheduled after the predecessor's ``mark_producer_done``
          (same ``finally``), the write happens-before the successor's
          ``_apply_firing_delay`` read on the single asyncio loop — no
          lock needed.
        - ``edge.min_start_delay_us``: gate >= ``node_firable_wall_us``
          (the moment ``await_inputs`` returned) + delay, measured from the
          moment the node became input-ready.
        - ``edge.delay_after_predecessor_start_us``: gate >= predecessor
          DISPATCH wall (``ctx.node_dispatch_wall_us[edge.source]``, stamped
          when its firing gate cleared) + delay; the successor was scheduled at
          that dispatch, so the stamp happens-before this read.
        - ``edge.delay_after_predecessor_first_token_us``: gate >= predecessor
          OBSERVED FIRST-TOKEN wall (``ctx.node_first_token_wall_us[edge.source]``,
          stamped by the dispatch adapter's first-token cb) + delay. Refines a
          start-anchored edge: this method first awaits the source's first-token
          latch, so the wall (or its absence -- a terminal without a first token)
          is settled before the gate is computed. An observed first token
          SUPERSEDES the dispatch anchor for that edge; a source that terminated
          without one has no wall entry, so the dispatch-anchor fallback applies.
        - Node-level ``min_start_delay_us``: gate >= ``node_firable_wall_us``
          + node delay.

        AND-fan-in: every incoming static edge contributes a gate; the runtime
        takes the ``max``.

        ``AIPERF_GRAPH_IGNORE_EDGE_DELAYS=1`` short-circuits ALL of the
        above uniformly — useful for "how fast on infinite hardware"
        A/B comparison. The per-instance ``compress_edge_delays`` flag
        (set by the accelerated cache-pressure warmup on
        WARMUP-phase executors only) does the same for a single executor.
        """
        if Environment.GRAPH.IGNORE_EDGE_DELAYS or self._compress_edge_delays:
            return
        # First-token-anchored edges gate on the source's OBSERVED first token,
        # so wait for the source's first-token latch (set by its stamp closure
        # OR by ``_finalize_node`` when it terminates without one) before
        # computing the gate. Placed AFTER the ignore/compress early return so
        # compressed replays skip the wait entirely -- their dispatch-time
        # scheduling already happened via ``start_anchored_successors``.
        for edge in self._scheduler.incoming_static_edges(node_id):
            if edge.delay_after_predecessor_first_token_us is not None:
                await ctx.first_token_event(edge.source).wait()
        gate_us = self._compute_firing_gate_us(node_id, ctx, node_firable_wall_us)
        if gate_us <= 0.0:
            return
        wait_us = gate_us - self._loop_wall_us()
        if wait_us <= 0:
            return
        # Replay the recorded firing gate FAITHFULLY (verbatim). The edge delays
        # are already on the build-plane per-gap-warped clock (the shared
        # ``ActiveIdleWarp`` in ``segment_ir/trie_content.py``, byte-faithful to
        # agentx's ``_IdleGapTimeWarp`` per-gap cap); there is
        # NO runtime clamp here. A faithfully-replayed trace can therefore span
        # its full recorded wall time -- the GraphIRReplayStrategy bounds phase
        # completion by the AgentX stop-condition model (duration / count), so a
        # long idle-gap replay finalizes on the stop condition rather than being
        # truncated mid-gap by an unfaithful per-node sleep cap.
        #
        # Sleeps on the injected clock (``WallClock`` -> ``asyncio.sleep``;
        # ``VirtualClock`` -> park until the driver pump advances sim time past
        # this gate). ``wait_us`` is microseconds; the clock takes ns.
        await self._clock.sleep_ns(int(wait_us * 1_000.0))

    def _compute_firing_gate_us(
        self,
        node_id: str,
        ctx: _TraceContext,
        node_firable_wall_us: float,
    ) -> float:
        """Return the max incoming-edge gate wall (us) for ``node_id``.

        AND-fan-in: every incoming static edge plus the node-level
        ``min_start_delay_us`` contributes a gate; the runtime takes the ``max``.
        See :meth:`_apply_firing_delay` for the edge-gate semantics.
        """
        gate_us = 0.0
        finishes = ctx.node_finish_wall_us
        dispatches = ctx.node_dispatch_wall_us
        first_tokens = ctx.node_first_token_wall_us
        for edge in self._scheduler.incoming_static_edges(node_id):
            if edge.delay_after_predecessor_first_token_us is not None:
                ft_us = first_tokens.get(edge.source)
                if ft_us is not None:
                    gate_us = max(
                        gate_us,
                        ft_us + edge.delay_after_predecessor_first_token_us,
                    )
                    # Observed first token supersedes the dispatch fallback for
                    # this edge. A source that terminated WITHOUT a first token
                    # has no wall entry, so fall through to the dispatch-anchor
                    # block below.
                    continue
            if edge.delay_after_predecessor_us is not None:
                finish_us = finishes.get(edge.source)
                if finish_us is not None:
                    gate_us = max(gate_us, finish_us + edge.delay_after_predecessor_us)
            if edge.min_start_delay_us is not None:
                gate_us = max(gate_us, node_firable_wall_us + edge.min_start_delay_us)
            if edge.delay_after_predecessor_start_us is not None:
                dispatch_us = dispatches.get(edge.source)
                if dispatch_us is not None:
                    gate_us = max(
                        gate_us,
                        dispatch_us + edge.delay_after_predecessor_start_us,
                    )
        node = self._parsed.graph.nodes.get(node_id)
        node_min_start = node.min_start_delay_us if node is not None else None
        if node_min_start is not None:
            # In absolute mode the snapshot rewrite stamped this as the firing's
            # ABSOLUTE dispatch_offset_us from t*; anchor it to the shared
            # instance run-start (t* origin) so a late-input co-scoped stream
            # fires at its recorded offset, not input-ready + offset (which
            # double-counts the lead and drifts it out of recorded-time order).
            # The wait is still clamped to input-readiness upstream (await_inputs
            # precedes this gate), so the node never fires before its data.
            anchor = (
                self._anchor_wall_us
                if self._absolute_start_offsets and self._anchor_wall_us is not None
                else node_firable_wall_us
            )
            gate_us = max(gate_us, anchor + node_min_start)
        return gate_us

    def _loop_wall_us(self) -> float:
        """Return the firing-loop clock reading in microseconds.

        Reads the injected ``AIPerfClock`` (``now_ns``) so the gate
        computation and any caller-side telemetry share ONE time source. With
        the default ``WallClock`` this is ``time.perf_counter_ns() / 1_000``
        (monotonic wall, behavior unchanged); with a ``VirtualClock`` it is
        sim time advanced by the driver pump.
        """
        return self._clock.now_ns() / 1_000.0

    # ------------------------------------------------------------------
    # dispatch table
    # ------------------------------------------------------------------
    # LlmNode's `_execute` body lives in `dispatch/llm.py`. The stub below is
    # a fallback raiser that fires only if the dispatch module failed to load;
    # any other node kind hits the singledispatch default (every live producer
    # lowers to LlmNode-only graphs).

    @singledispatchmethod
    async def _execute(
        self,
        node: Any,
        inputs: dict[str, Any],
        ctx: _TraceContext,
    ) -> NodeExecutionResult:
        raise NotImplementedError(
            f"no _execute registered for node kind {type(node).__name__!r}"
        )

    @_execute.register
    async def _execute_llm(
        self, node: LlmNode, inputs: dict[str, Any], ctx: _TraceContext
    ) -> NodeExecutionResult:
        raise NotImplementedError(
            "LlmNode dispatch was not loaded; ensure 'import "
            "aiperf.graph.dispatch' ran before "
            "TraceExecutor instantiation"
        )
