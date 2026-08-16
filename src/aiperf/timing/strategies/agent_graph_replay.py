# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentGraphReplayStrategy — the agent graph trace runner on the v1 credit pipeline.

This timing strategy drives the dataflow ``TraceExecutor`` once per
recorded trace and OWNS both completion and concurrency, rather than letting the
linear ``CreditCounter`` / session-slot arithmetic decide. A fan-out trace has
no linear acquire/release; graph credits flow through real issuance -> worker ->
records -> returns but BYPASS the session-slot lifecycle via
``CreditIssuer.issue_graph_credit``.

Ownership model
---------------
* **Completion.** Faithful to AgentX, the legacy branch orchestrator this
  strategy is benchmarked against (``aiperf.timing.phase.stop_conditions``): the
  phase finalizes on the STOP CONDITION (session count / request count /
  duration), NOT on every admitted trace fully draining its faithful idle-gap
  replay. ``execute_phase``'s ``finally`` always signals sending-complete, and a
  ``--benchmark-duration`` cancels in-flight executors when the budget elapses
  (see :meth:`AgentGraphReplayStrategy._run_traces_under_duration_budget`). We do
  NOT consult ``CreditCounter.is_final_turn`` (it cannot express a DAG).
* **Concurrency.** A fixed pool of ``max_concurrent_traces`` lanes (see
  :meth:`AgentGraphReplayStrategy._run_lanes` / ``_resolve_lane_count``) admits
  traces: each lane runs one trace instance at a time and recycles onto the next
  wrap template when the stop-condition gate still admits a new session. Within a
  trace the executor self-gates via its dataflow channels -- the lane pool is the
  only cross-trace bound (prevents parked-Future explosion).

Return routing
--------------
The strategy installs ONE graph-return observer on the ``CreditCallbackHandler``
(via ``register_observer``) that routes each ``(credit, error, cancelled)`` to
the owning per-trace ``CreditDispatchAdapter`` by ``credit.trace_id``. It fires
UNCONDITIONALLY (before the gated ``handle_credit_return`` path), so an in-flight
dispatch Future is always resolved even after the phase can no longer send.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import statistics
import uuid
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec
import orjson

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.clock import Clock, RealClock
from aiperf.common.constants import MICROS_PER_SECOND
from aiperf.common.environment import Environment
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.scenario.base import TrajectoryWarmupFailedError
from aiperf.dataset.graph.models import (
    START_NODE_ID,
    StaticEdge,
    ToolNode,
    resolve_trace_graph,
)
from aiperf.graph.credit_dispatch_adapter import (
    CreditDispatchAdapter,
    CreditIssueRefusedError,
)
from aiperf.graph.executor import TraceExecutor, TraceResult
from aiperf.graph.sandbox.protocols import ToolSandbox
from aiperf.graph.sandbox.provider import (
    DockerSandboxProvider,
    LocalSandboxProvider,
    SandboxProvider,
)
from aiperf.graph.tool_dispatch.sandbox_dispatcher import SandboxToolDispatcher
from aiperf.timing.agent_graph_trace_view import parsed_for_trace
from aiperf.timing.strategies.graph_trace_planner import GraphTracePlanner
from aiperf.timing.strategies.graph_warmup import (
    GraphWarmupKind,
    first_token_sources,
    rewrite_for_warmup,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from aiperf.common.enums import CacheBustTarget
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.credit.messages import FirstToken
    from aiperf.credit.structs import Credit
    from aiperf.dataset.graph.models import (
        GraphRecord,
        ParsedGraph,
        TraceRecord,
    )
    from aiperf.plugin.enums import DatasetSamplingStrategy
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.stop_conditions import StopConditionChecker

_logger = AIPerfLogger(__name__)

# Per-trace-instance tool sandbox workspaces, rooted in the run's artifact dir.
GRAPH_TOOL_WORKSPACE_DIRNAME = "graph-tool-workspaces"


def _write_tool_time_artifact(
    path: Path,
    *,
    durations: list[float],
    traces: int,
    backend: str,
) -> None:
    """Write per-run tool-time summary to a JSON artifact.

    Called from ``AgentGraphReplayStrategy.report_tool_execution`` so the
    headline measurement is available to analysis scripts without grepping
    logs. Only written when ``artifact_dir`` is set and tool nodes ran.

    The artifact name ``profile_export_graph_tool_time.json`` mirrors the
    ``profile_export_*`` family already written by the exporter manager.
    """
    if not durations:
        return
    sorted_d = sorted(durations)
    payload = {
        "command_count": len(durations),
        "trace_count": traces,
        "backend": backend,
        "total_s": sum(durations),
        "mean_s": statistics.mean(durations),
        "median_s": statistics.median(sorted_d),
        "max_s": sorted_d[-1],
        "durations_s": durations,
    }
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


_OSL_WARNING_THRESHOLD = 0.5  # observed < 50% of target triggers an OSL warning


def _compute_normalized_model_s(
    durations: list[float],
    ttft_s: list[float | None],
    target_osl: list[int | None],
    observed_osl: list[int | None],
) -> tuple[float | None, int]:
    """Return (normalized_model_s, low_osl_call_count).

    Matches Agent Trace Replay's per-call normalization:
        observed_decode_tokens = max(observed_osl - 1, 1)
        target_decode_tokens = max(target_osl - 1, 0)
        raw_generation = max(raw_duration - ttft, 0)
        normalized = raw_duration - raw_generation + (
            raw_generation / observed_decode_tokens * target_decode_tokens
        )

    Returns None for normalized_model_s when no call has both target and
    observed OSL plus TTFT. Calls with missing timing or OSL fall back to raw
    duration.
    """
    if not durations or any(
        len(series) != len(durations) for series in (ttft_s, target_osl, observed_osl)
    ):
        return None, 0
    normalized_total = 0.0
    has_any = False
    low_osl = 0
    for dur, ttft, tgt, obs in zip(
        durations, ttft_s, target_osl, observed_osl, strict=True
    ):
        if ttft is not None and tgt is not None and obs:
            has_any = True
            raw_generation_s = max(dur - ttft, 0.0)
            observed_decode_tokens = max(obs - 1, 1)
            target_decode_tokens = max(tgt - 1, 0)
            normalized_total += (
                dur
                - raw_generation_s
                + (raw_generation_s / observed_decode_tokens * target_decode_tokens)
            )
            if obs < _OSL_WARNING_THRESHOLD * tgt:
                low_osl += 1
        else:
            normalized_total += dur
    return (normalized_total if has_any else None), low_osl


def _write_trace_summary_artifact(
    path: Path,
    *,
    summaries: list[dict],
) -> None:
    """Write per-trace wall-time breakdown to a JSON artifact.

    Mirrors the Agent Trace Replay ``summary`` block:
    ``total_s / model_s / tool_s / model_time_fraction / tool_time_fraction /
    model_calls / tool_calls`` for each trace plus an aggregate section.
    Fractions are in [0, 1], matching Agent Trace Replay's ``model_time_fraction`` field.
    When OSL data is available, also includes ``normalized_model_s`` and
    ``total_osl_warnings`` in the aggregate block.
    """
    if not summaries:
        return
    agg_total = sum(s["total_s"] for s in summaries)
    agg_model = sum(s["model_s"] for s in summaries)
    agg_tool = sum(s["tool_s"] for s in summaries)
    agg_model_calls = sum(s["model_calls"] for s in summaries)
    agg_tool_calls = sum(s["tool_calls"] for s in summaries)
    norm_values = [
        s["normalized_model_s"]
        for s in summaries
        if s.get("normalized_model_s") is not None
    ]
    agg_normalized = sum(norm_values) if norm_values else None
    agg_osl_warnings = sum(s.get("low_osl_model_calls", 0) for s in summaries)
    payload = {
        "trace_count": len(summaries),
        "aggregate": {
            "total_s": agg_total,
            "model_s": agg_model,
            "tool_s": agg_tool,
            "model_time_fraction": agg_model / agg_total if agg_total > 0 else 0.0,
            "tool_time_fraction": agg_tool / agg_total if agg_total > 0 else 0.0,
            "model_calls": agg_model_calls,
            "tool_calls": agg_tool_calls,
            "normalized_model_s": agg_normalized,
            "total_osl_warnings": agg_osl_warnings,
        },
        "traces": summaries,
    }
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


# Every ``StaticEdge`` field the executor can turn into a firing gate
# (``TraceExecutor._compute_firing_gate_us``). Kept as ONE list so a new gate
# field cannot be added to the runtime without the replay-wait advisory seeing
# it -- the omission of ``min_start_delay_us`` here is exactly what let a
# START-rooted leading offset go unreported.
_EDGE_DELAY_FIELDS = (
    "min_start_delay_us",
    "delay_after_predecessor_us",
    "delay_after_predecessor_start_us",
    "delay_after_predecessor_first_token_us",
)

__all__ = ["AgentGraphReplayStrategy", "first_token_sources", "rewrite_for_warmup"]


def _leaf_credit_refusal(exc: BaseException) -> CreditIssueRefusedError | None:
    """Return the refusal iff ``exc`` is (a group of) ONLY issuer refusals.

    The per-trace executor TaskGroup wraps node failures in an
    ``ExceptionGroup``, so a healthy stop-gate refusal
    (``CreditIssueRefusedError``: request-count / duration cap reached, or run
    cancelled) may surface bare OR grouped -- and multiple concurrent node
    coroutines may each carry one. A mixed group (a refusal alongside a genuine
    error) is NOT a clean stop and returns ``None`` so the caller keeps the
    error path.
    """
    if isinstance(exc, CreditIssueRefusedError):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        matched, rest = exc.split(CreditIssueRefusedError)
        if matched is not None and rest is None:
            leaf: BaseException = matched.exceptions[0]
            while isinstance(leaf, BaseExceptionGroup):
                leaf = leaf.exceptions[0]
            return leaf  # type: ignore[return-value]
    return None


class AgentGraphReplayStrategy(AIPerfLoggerMixin):
    """Drive a ``TraceExecutor`` per agent graph trace over the v1 credit pipeline.

    Constructed per-phase by ``PhaseRunner._build_strategy`` with the standard
    timing-strategy kwargs PLUS the graph-only injection channel
    (``parsed_graph`` + ``register_observer``); see the module docstring for the
    ownership contract.
    """

    def __init__(
        self,
        *,
        config: CreditPhaseConfig,
        stop_checker: StopConditionChecker | None = None,
        credit_issuer: CreditIssuer,
        lifecycle: PhaseLifecycle | None = None,
        parsed_graph: ParsedGraph,
        register_observer: Callable[[Any], None],
        register_first_token_observer: Callable[[Any], None],
        unregister_observer: Callable[[Any], None],
        unregister_first_token_observer: Callable[[Any], None],
        max_concurrent_traces: int | None = None,
        dispatch_timeout_s: float | None = None,
        start_min_ratio: float = 0.0,
        start_max_ratio: float = 0.0,
        t_star_random_seed: int = 0,
        burst_phase_starts: bool = False,
        dataset_sampling_strategy: DatasetSamplingStrategy | None = None,
        allow_dataset_wrap: bool | None = None,
        cache_bust: CacheBustTarget | None = None,
        replay_speedup: float | None = None,
        open_loop_replay: bool = True,
        open_loop_strict: bool = False,
        graph_tool_image: str | None = None,
        graph_tool_persistent_session: bool = False,
        warmup_kind: GraphWarmupKind | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialize the graph trace runner.

        Args:
            config: Per-phase ``CreditPhaseConfig`` (concurrency bound + stop
                thresholds: ``expected_num_sessions`` / ``total_expected_requests``).
            stop_checker: Gates RECYCLE admission via ``can_send_dag_child_turn``
                (:meth:`_recycle_has_stop_condition`). The issuer honors the stop
                gate for the initial dispatch, but a recycle pass must re-ask
                before starting a new instance, so this is NOT unused -- an
                earlier docstring claimed it was, and dropping it would let every
                lane recycle unbounded past the stop condition. ``None`` admits
                every recycle.
            credit_issuer: The real ``CreditIssuer``; adapters call its
                ``issue_graph_credit`` (session-slot-bypassing) path.
            lifecycle: Phase ``PhaseLifecycle``; read for the duration stop
                condition (``time_left_in_seconds()``) bounding the dispatch.
            parsed_graph: The built ``ParsedGraph`` whose ``traces`` this phase
                replays (the R3 injection channel; see ``PhaseRunner``).
            register_observer: Callback installing the single graph-return
                observer on the shared ``CreditCallbackHandler``.
            register_first_token_observer: Callback installing the single
                graph-first-token observer on the shared ``CreditCallbackHandler``
                (post-TTFT anchoring).
            unregister_observer: Compare-and-clear detach for the graph-return
                observer (``CreditCallbackHandler.clear_graph_return_observer``):
                teardown passes its OWN observer and the handler clears the
                shared slot only if that observer is still installed. Required
                for seamless multi-phase runs where a stale phase's deferred
                teardown fires after the next phase installed its observer.
            unregister_first_token_observer: Same compare-and-clear detach for
                the first-token observer slot.
            max_concurrent_traces: Trace-admission bound; defaults to the phase
                ``concurrency`` else ``1`` (the plain aiperf default).
            dispatch_timeout_s: Per-dispatch deadlock guard (adapter default).
            start_min_ratio: Lower bound (fraction of duration) of the t* window.
                Default ``0.0``; together with ``start_max_ratio=0.0`` this
                selects full recorded replay (t*=0, no snapshot rewrite). The
                AgentX 0.0..1.0 window is scenario-applied, not a bare default.
            start_max_ratio: Upper bound of the t* window. Default ``0.0``.
            t_star_random_seed: Base seed for per-trace t* sampling (trace-salted).
            burst_phase_starts: AgentX ``--burst-phase-starts`` parity. False
                (default) = SPREAD: each trace's first firing keeps its recorded
                (warped, <=cap) leading offset from t*. True = BURST: the leading
                per-firing dispatch offset is zeroed so every trace's earliest
                profiling firing (and every warmup priming credit) fires at
                phase-time 0; relative inter-turn delays after the first are
                still honored. Governs ONLY the phase-start dispatch pattern.
            dataset_sampling_strategy: Resolved run-level dataset sampling
                strategy (from ``run.resolved.dataset_sampling_strategy`` via the
                CreditPhaseConfig). Consumed by the per-lane trace draw
                (``GraphTracePlanner.draw_index``): SHUFFLE/RANDOM remap the draw through a seeded
                permutation; SEQUENTIAL/None keep the byte-identical cursor.
                None for non-graph phases / until resolution derives it.
            allow_dataset_wrap: Resolved graph-plane dataset-wrap policy (from
                ``run.resolved.allow_dataset_wrap`` via the CreditPhaseConfig).
                Consumed by the setup-phase wrap-guard (over-subscription raises
                unless wrapping is allowed). None until derived.
            cache_bust: Resolved per-trace-instance cache-bust target (from
                ``endpoint.cache_bust`` via the CreditPhaseConfig). Consumed by
                the dispatch duplication report: when lanes recycle (factor > 1)
                with cache-bust OFF (``None`` / ``NONE``), clones collide on
                identical prefixes, so the report warns; ON it stays quiet.
            graph_tool_image: Resolved ``--graph-tool-image``. Selects the
                backend for REAL tool execution: ``None``/empty runs each
                trace's recorded commands in a local shell, a non-empty image
                runs them in a per-trace container from that image. Read only
                when the graph carries ``ToolNode`` steps, so a run without
                ``--graph-execute-tools`` never touches a sandbox.
        """
        super().__init__(logger_name="AgentGraphReplayTiming")
        # Time source for BOTH replay pacing knobs this strategy owns: the
        # open-loop recorded-start schedule and the duration budget. Defaults to
        # ``RealClock`` (``time.perf_counter_ns`` + ``asyncio.sleep`` -- behavior
        # identical to the prior raw ``asyncio`` reads). A ``SimClock`` can be
        # injected so a driver pump fast-forwards sim time, letting a multi-hour
        # recorded corpus replay in milliseconds -- the contract
        # ``aiperf.common.clock`` was written for and ``TraceExecutor`` already
        # honors. ONE source for both knobs on purpose: straddling the loop clock
        # and a virtual clock would let the duration budget expire in wall time
        # while the schedule advanced in sim time.
        self._clock: Clock = clock if clock is not None else RealClock()
        self._config = config
        self._credit_issuer = credit_issuer
        self._stop_checker = stop_checker
        self._parsed = parsed_graph
        self._register_observer = register_observer
        self._register_first_token_observer = register_first_token_observer
        self._unregister_observer = unregister_observer
        self._unregister_first_token_observer = unregister_first_token_observer
        self._dispatch_timeout_s = dispatch_timeout_s
        self._burst_phase_starts = bool(burst_phase_starts)
        # Resolved dataset-selection policy threaded from ``run.resolved`` via
        # the CreditPhaseConfig. Consumed by the setup-phase wrap-guard and the
        # per-lane sampling draw (``GraphTracePlanner.draw_index``).
        self._dataset_sampling_strategy = dataset_sampling_strategy
        self._allow_dataset_wrap = allow_dataset_wrap
        # Resolved cache-bust target threaded from ``endpoint.cache_bust`` via the
        # CreditPhaseConfig. Consumed by the dispatch duplication report below.
        self._cache_bust = cache_bust
        self._replay_speedup = replay_speedup or 1.0
        self._open_loop_replay = bool(open_loop_replay)
        self._open_loop_strict = bool(open_loop_strict)
        # Tool-execution wiring. Scanned ONCE here rather than per instance: a
        # graph either lowers ToolNodes (--graph-execute-tools) or it does not,
        # and the answer decides whether a run ever touches a sandbox at all.
        # Every graph is scanned, not just ``parsed.graph``, because a
        # multi-graph workload resolves each trace to its own record.
        self._graph_tool_image = (graph_tool_image or "").strip() or None
        self._graph_tool_persistent_session = graph_tool_persistent_session
        self._warmup_kind = warmup_kind
        self._has_tool_nodes = any(
            isinstance(node, ToolNode)
            for graph in (parsed_graph.graph, *parsed_graph.graphs.values())
            for node in graph.nodes.values()
        )
        # Per-trace-instance sandbox workspaces live under the run's artifacts so
        # they are discoverable after the run and cleaned up with it, instead of
        # accumulating in a temp dir nothing owns.
        self._tool_workspace_root = (
            (config.artifact_dir or Path.cwd()) / GRAPH_TOOL_WORKSPACE_DIRNAME
            if self._has_tool_nodes
            else None
        )
        self._sandbox_provider: SandboxProvider | None = self._build_sandbox_provider(
            parsed_graph
        )
        # Accumulated per-command tool wall time, the headline measurement this
        # mode exists to produce. Kept on the strategy because a TraceResult is
        # otherwise discarded at the end of each instance.
        self._tool_durations_s: list[float] = []
        self._tool_traces = 0
        # Per-trace breakdown records: {trace_id, total_s, model_s, tool_s,
        # model_calls, tool_calls}. Accumulated across all completed instances;
        # written to profile_export_graph_trace_summary.json at phase teardown.
        self._trace_summaries: list[dict] = []
        self._schedule_zero_unix_ms = self._find_schedule_zero(parsed_graph)
        self._schedule_anchor: float | None = None
        # Corpus selected once in ``setup_phase`` and reused by ``execute_phase``
        # so the seed-deterministic draw (and its re-anchor) happens exactly once.
        self._selected_traces: list[TraceRecord] | None = None
        # Duration stop condition (AgentX parity): ``time_left_in_seconds()``
        # bounds the dispatch; ``None`` with no ``--benchmark-duration``.
        self._lifecycle = lifecycle

        self._start_min_ratio = start_min_ratio
        self._start_max_ratio = start_max_ratio
        self._t_star_random_seed = t_star_random_seed
        self._planner = GraphTracePlanner(
            parsed=parsed_graph,
            start_min_ratio=start_min_ratio,
            start_max_ratio=start_max_ratio,
            t_star_random_seed=t_star_random_seed,
            dataset_sampling_strategy=dataset_sampling_strategy,
        )
        self._init_registry_state()
        self._init_warmup_state()

        # Only an EXPLICIT concurrency gates the timestamped path; the limit and
        # its provenance come from ONE evaluation so they cannot drift apart.
        self._max_concurrent, self._concurrency_is_explicit = self._resolve_concurrency(
            max_concurrent_traces
        )
        # Live lane-admission limit (LaneSettableProtocol): the concurrency
        # ramper drives it 1 -> _max_concurrent; without a ramp every lane is
        # admitted immediately. The event is swapped fresh on every raise so
        # parked lanes re-check without lost wakeups (single-threaded asyncio:
        # a waiter's check-then-await pair cannot be preempted by the setter).
        self._lane_limit = self._max_concurrent
        self._lane_limit_raised = asyncio.Event()
        # Slots currently held by timestamped traces. The lane path parks by
        # lane INDEX (a lane owns its index for the whole phase); the
        # timestamped path has no fixed lane per trace, so it borrows the SAME
        # limit + event as a slot pool: a trace holds a slot only while its
        # executor runs.
        self._active_trace_slots: set[int] = set()
        # FIFO of traces parked for a slot. One release hands off to exactly one
        # waiter (see _acquire_trace_slot for why this is not the shared event).
        self._slot_waiters: deque[asyncio.Future] = deque()
        self._completed_traces = 0
        # Traces whose run attempt FINISHED, successfully or not, admitted or
        # not. This is the failure denominator: _completed_traces is paired with
        # _admitted_traces for the in-flight diagnostic and so cannot count an
        # instance that failed before the gate.
        self._finished_traces = 0
        self._errored_traces = 0
        self._admitted_traces = 0
        # Total instances dispatched off the finite pool (lane starts + every
        # serial recycle) for the dispatch duplication report. An instance
        # attribute so the report fires on BOTH natural completion AND
        # duration-cancel (the ``_run_lanes`` local would be lost when the lane
        # future is cancelled by the duration budget).
        self._instances_started = 0
        # Loop-clock deadline for the ACTIVE duration-budgeted phase; None
        # outside one. The budget wrappers
        # cancel the lane fan-out task on timeout, but cancellation delivery
        # through a TaskGroup whose children complete constantly is unreliable
        # on Python 3.11 (the cancel can be lost mid-abort), so the lane
        # recycle loops ALSO check this deadline cooperatively -- the stage
        # then halts on time even when the cancel never lands.
        self._duration_deadline: float | None = None
        # Latched the first time the issuer's stop gate refuses a dispatch
        # (``--request-count`` / ``--benchmark-duration`` / cancellation). The
        # gate is monotonic, so a refusal means every later dispatch would be
        # refused too. Traces parked in ``_wait_for_recorded_start`` race their
        # sleep against this so a closed gate ENDS the phase instead of leaving
        # it to sit out the rest of the recorded timeline.
        self._admission_closed = asyncio.Event()
        # Tasks currently parked in ``_wait_for_recorded_start``, and the subset
        # this strategy has deliberately cancelled to release them. The second
        # set is what lets the park distinguish OUR cancel from a duration-cancel
        # or Ctrl+C, which must keep propagating.
        self._parked_on_recorded_start: set[asyncio.Task] = set()
        self._released_from_park: set[asyncio.Task] = set()

    def _init_registry_state(self) -> None:
        """Initialize the per-instance adapter registry and its background tasks.

        The t* source / plans / catalog this used to build alongside the
        registry now live on :class:`GraphTracePlanner`; what remains here is
        purely the return-routing bookkeeping the strategy owns.
        """
        # ``{instance_id: CreditDispatchAdapter}`` -- the de-mux registry the
        # single return observer routes by ``credit.trace_id`` (the per-recycle
        # instance id ``{template}::{nonce}``, e.g. ``t-1::3f2a...``), so two
        # concurrent instances of one
        # template never collide.
        self._adapters: dict[str, CreditDispatchAdapter] = {}
        # Instance ids whose parent run has finished; the adapter is popped here
        # (not earlier) so a mid-run ``on_drained`` callback never reaps it while
        # the executor is still firing. Serialized with ``_registry_lock`` so the
        # parent finally and a late ``on_drained`` callback never race the pop.
        self._parent_done: set[str] = set()
        # ``{template_trace_id: frozenset[node_id]}`` -- the sources of each
        # trace's first-token-anchored edges (post-TTFT anchoring), computed once
        # per template from its OWN projected graph and reused across lanes /
        # recycles (the set is lane/t*-independent: a t* chop only drops edges).
        self._first_token_sources_cache: dict[str, frozenset[str]] = {}
        # ``{template_trace_id: node_id -> (agent_depth, parent_node_id)}`` --
        # the dag_jsonl lowering's ``metadata["dag"]`` legacy-identity stamp,
        # read once per template from its OWN projected graph. ``None`` for
        # stamp-free producers (weka/dynamo) so the adapter's root-chain
        # fallback stays byte-identical.
        self._node_identity_cache: dict[
            str, dict[str, tuple[int, str | None]] | None
        ] = {}
        self._registry_lock = asyncio.Lock()
        # Background deferred-release tasks scheduled from the sync ``on_drained``
        # callback; tracked so none is garbage-collected mid-flight.
        self._release_tasks: set[asyncio.Task[None]] = set()
        # Background GraphTraceEnd sends scheduled from the sync adapter-reap
        # path; tracked so none is garbage-collected mid-flight.
        self._trace_end_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _find_schedule_zero(parsed: ParsedGraph) -> int | None:
        """Return the earliest preserved source timestamp in the corpus.

        This is deliberately NOT driven by ``CreditPhaseConfig``'s
        ``auto_offset_timestamps`` / ``fixed_schedule_start_offset``, which
        ``FixedScheduleStrategy.setup_phase`` uses to pick linear replay's
        schedule zero. Those two fields are populated from ``auto_offset`` /
        ``start_offset``, which exist ONLY on ``FixedSchedulePhase``, and a
        graph workload with a fixed-schedule phase is rejected outright by
        ``_reject_graph_incompatible_phases`` (the CLI also rejects the offset
        flags without ``--fixed-schedule``). On the graph path they can
        therefore only ever hold their not-applicable defaults, so reading
        them would import a placeholder, not user intent.

        The behavior here already IS linear's ``auto_offset=True`` branch.
        Linear's other branches do not transfer: ``recorded_start_unix_ms`` is
        absolute unix epoch milliseconds, so an anchor of ``0`` (or of a small
        ``start_offset``) would place every trace's target decades out:
        ``_wait_for_recorded_start`` computes
        ``(trace_start - schedule_zero) / 1000`` SECONDS, and an epoch-ms
        ``trace_start`` of ~1.7e12 over 1000 is ~1.7e9 s, i.e. ~54 years.

        ``_reanchor_schedule_zero`` is the only other writer of the anchor;
        with no configured anchor competing with it, no precedence rule is
        needed. Pinned by ``tests/unit/graph/test_graph_schedule_anchor.py``.
        """
        timestamps = [
            int(node.recorded_start_unix_ms)
            for graph in [parsed.graph, *parsed.graphs.values()]
            for node in graph.nodes.values()
            if node.recorded_start_unix_ms is not None
        ]
        return min(timestamps) if timestamps else None

    @staticmethod
    def _graph_recorded_start(parsed: ParsedGraph) -> int | None:
        """Return the earliest source timestamp in one trace graph."""
        timestamps = [
            int(node.recorded_start_unix_ms)
            for node in parsed.graph.nodes.values()
            if node.recorded_start_unix_ms is not None
        ]
        return min(timestamps) if timestamps else None

    def _scale_timing(self, parsed: ParsedGraph) -> ParsedGraph:
        """Apply replay speedup to every executor-visible graph delay."""
        if self._replay_speedup == 1.0:
            return parsed

        def scale(value: float | None) -> float | None:
            return None if value is None else value / self._replay_speedup

        graph = parsed.graph
        nodes = {
            node_id: msgspec.structs.replace(
                node,
                arrival_offset_us=(
                    None
                    if node.arrival_offset_us is None
                    else int(round(node.arrival_offset_us / self._replay_speedup))
                ),
                min_start_delay_us=scale(node.min_start_delay_us),
            )
            for node_id, node in graph.nodes.items()
        }
        edges = [
            msgspec.structs.replace(
                edge,
                min_start_delay_us=scale(edge.min_start_delay_us),
                delay_after_predecessor_us=scale(edge.delay_after_predecessor_us),
                delay_after_predecessor_start_us=scale(
                    edge.delay_after_predecessor_start_us
                ),
                delay_after_predecessor_first_token_us=scale(
                    edge.delay_after_predecessor_first_token_us
                ),
            )
            for edge in graph.edges
        ]
        return msgspec.structs.replace(
            parsed, graph=msgspec.structs.replace(graph, nodes=nodes, edges=edges)
        )

    def _strict_schedule_projection(self, parsed: ParsedGraph) -> ParsedGraph:
        """Replace graph dependencies with START-relative timestamp gates.

        Dynamo trie prompts are materialized from the segment store, so their
        channel inputs are bookkeeping edges rather than prompt dependencies.
        The projection drops those runtime gates while leaving the source graph
        unchanged in the parsed workload and sidecar.
        """
        graph = parsed.graph
        starts = [
            node.recorded_start_unix_ms
            for node in graph.nodes.values()
            if node.recorded_start_unix_ms is not None
        ]
        if not starts:
            return parsed
        trace_zero = min(starts)
        nodes = {
            node_id: msgspec.structs.replace(node, min_start_delay_us=None, inputs=[])
            for node_id, node in graph.nodes.items()
        }
        edges = [
            StaticEdge(
                source=START_NODE_ID,
                target=node_id,
                min_start_delay_us=max(
                    0.0,
                    (node.recorded_start_unix_ms - trace_zero)
                    * 1000.0
                    / self._replay_speedup,
                ),
            )
            for node_id, node in nodes.items()
            if node.recorded_start_unix_ms is not None
        ]
        return msgspec.structs.replace(
            parsed, graph=msgspec.structs.replace(graph, nodes=nodes, edges=edges)
        )

    def _recorded_start_target(self, parsed: ParsedGraph) -> float | None:
        """Loop-clock instant this trace is due, or ``None`` when it is not paced.

        ``None`` covers closed-loop replay, a corpus with no schedule zero, and
        a trace carrying no recorded start -- all of which dispatch immediately.
        Pins the shared ``_schedule_anchor`` on first use so every trace measures
        its offset from one origin.
        """
        if not self._open_loop_replay or self._schedule_zero_unix_ms is None:
            return None
        trace_start = self._graph_recorded_start(parsed)
        if trace_start is None:
            return None
        if self._schedule_anchor is None:
            self._schedule_anchor = self._clock.perf_ns() / 1e9
        return self._schedule_anchor + (
            (trace_start - self._schedule_zero_unix_ms) / 1000.0 / self._replay_speedup
        )

    async def _wait_for_recorded_start(self, parsed: ParsedGraph) -> bool:
        """Wait until a trace's recorded start reaches the replay timeline.

        Returns ``True`` when the recorded start arrived and the caller should
        dispatch, ``False`` when admission closed while parked here and the
        caller must abandon the instance without dispatching.

        The wait races the recorded-start sleep against ``_admission_closed``
        (set the first time the issuer's stop gate refuses a dispatch). Without
        that race a closed gate could not end the phase: the timestamped path
        creates EVERY trace's task before the first dispatch, so by the time a
        refusal happens every remaining trace is already parked here -- past any
        guard at the top of its coroutine. The phase would then sit out the rest
        of the recorded timeline having already stopped issuing, which reads as
        a hang (progress bar at 100%, no output, no advisory).

        Latching is sound because the gate is monotonic: ``--request-count``
        (``requests_sent``), ``--benchmark-duration``, and the cancellation flag
        only ever close. A refusal therefore guarantees every later dispatch
        would also be refused -- the same reasoning the lane path already uses
        to stop recycling (``_run_lanes``: "recycling again would only
        re-refuse").
        """
        target = self._recorded_start_target(parsed)
        if target is None:
            return True
        if self._admission_closed.is_set():
            return False
        # Park on a plain ``clock.sleep_ns`` and let ``_release_parked_traces``
        # interrupt it by cancelling this task.
        #
        # The alternative -- racing the sleep against an Event via
        # ``asyncio.wait`` -- is rejected because it must pre-create TWO tasks
        # per parked trace, and this path parks one task PER TRACE: a
        # 500k-trace corpus would allocate ~1M transient tasks to save a
        # cancellation handler.
        me = asyncio.current_task()
        if me is not None:
            self._parked_on_recorded_start.add(me)
        try:
            while True:
                remaining = target - self._clock.perf_ns() / 1e9
                if remaining <= 0:
                    return True
                await self._clock.sleep_ns(int(remaining * 1_000_000_000))
        except asyncio.CancelledError:
            # Swallow ONLY the cancel we issued for this exact task, and only
            # when ``uncancel`` shows no outer cancellation is still pending --
            # a duration-cancel or Ctrl+C racing the release must still unwind.
            if me is not None and me in self._released_from_park:
                self._released_from_park.discard(me)
                if me.uncancel() == 0:
                    return False
            raise
        finally:
            if me is not None:
                self._parked_on_recorded_start.discard(me)

    def _release_parked_traces(self) -> None:
        """Wake every trace parked on its recorded start so it can stop.

        Called once the issuer's stop gate latches closed. Each released task
        returns ``False`` from :meth:`_wait_for_recorded_start` and abandons its
        instance WITHOUT dispatching. Traces already past the park -- mid
        dispatch, with requests on the wire -- are deliberately untouched: their
        records are real and must be allowed to land, which is why this releases
        specific parked tasks rather than cancelling the phase TaskGroup the way
        the duration budget does.
        """
        for task in list(self._parked_on_recorded_start):
            self._released_from_park.add(task)
            task.cancel()

    def _init_warmup_state(self) -> None:
        """Initialize the warmup-failure abort state for this phase."""
        from aiperf.common.enums import CreditPhase

        phase = self._config.phase
        # Warmup-failure abort state (AgentX parity): a WARMUP phase that
        # recorded terminal request failures aborts the run before profiling
        # (see :meth:`report_warmup_failures`).
        self._is_warmup_phase = bool(phase == CreditPhase.WARMUP)
        if self._is_warmup_phase != (self._warmup_kind is not None):
            raise RuntimeError(
                f"warmup state mismatch: _is_warmup_phase={self._is_warmup_phase} "
                f"but warmup_kind={self._warmup_kind!r} -- this is a construction bug"
            )
        self._warmup_failure_count = 0
        self._warmup_failure_samples: list[str] = []
        # Human-readable samples for the traces counted in ``_errored_traces``,
        # capped like the warmup samples. Without these, ``report_trace_failures``
        # could report only a count and the operator would have to trawl
        # per-instance WARNINGs to learn what actually broke.
        self._errored_trace_samples: list[str] = []

    def _resolved_num_sessions(self) -> int | None:
        """Effective session TARGET for this phase (explicit, else derived corpus size).

        An explicit ``--num-conversations`` (``expected_num_sessions`` on the
        CreditPhaseConfig) always wins. Otherwise, for a BARE graph PROFILING run
        -- no explicit stop condition at all (``expected_num_sessions`` /
        ``total_expected_requests`` / ``expected_duration_sec`` and the lifecycle
        duration all unset; the bare-graph config case validated against the
        graph dataset in ``check_phase_dataset_compatibility``) -- the
        single-corpus-pass target is derived from the loaded corpus:
        ``N = len(self._parsed.traces)`` -- mirroring dag_jsonl's roots->sessions
        convention and giving progress reporting a concrete ``N``.

        SCOPE (deliberate): this value is a TARGET used ONLY for
        :meth:`_resolve_lane_count` lane clamping and as the reported
        expected-session count. It is NOT a recycle-enabling stop condition --
        :meth:`_recycle_has_stop_condition` and :meth:`_can_recycle` read the
        EXPLICIT ``expected_num_sessions`` directly, so a bare run stays in
        SINGLE-PASS mode (each trace once, no recycle, clean pass-0 plans).
        Routing this derived ``N`` into the recycle gate would flip a bare run
        into the bounded/recycle path and silently change its dispatch (fresh-start
        plan salting, divergent t* instants) -- which is why it is excluded there.

        Returns ``None`` (no derived target) when another explicit stop already
        bounds the run without a session count (``--request-count`` /
        ``--benchmark-duration``), and for WARMUP phases (their priming fan-out
        keeps its historical behavior).
        """
        cfg = self._config
        explicit = cfg.expected_num_sessions
        if explicit is not None and explicit > 0:
            return int(explicit)
        if self._is_warmup_phase:
            return None
        if (cfg.total_expected_requests or cfg.expected_duration_sec) or (
            self._lifecycle is not None
            and self._lifecycle.time_left_in_seconds() is not None
        ):
            return None
        traces = self._parsed.traces
        return len(traces) if traces else None

    def _resolve_lane_count(self, total: int, recycle_is_bounded: bool) -> int:
        """Resolve how many concurrent lanes this phase fans out.

        AgentX builds exactly ``concurrency`` lanes wrapping the corpus
        (``trajectory_source.py`` ``_target_size = concurrency``), so lane
        fan-out can EXCEED the corpus size and is sustained by recycle. We mirror
        that: the lane count is the phase ``concurrency`` (``_max_concurrent``),
        independent of corpus size, when a stop condition exists for the lanes to
        recycle toward.

        Two clamps keep semantics sane:
        * ``--num-conversations N`` (explicit ``expected_num_sessions``) caps the
          TOTAL distinct roots ever started, so starting more than ``N`` lanes is
          pointless -- clamp lanes to ``N``. (The per-lane recycle gate
          ``_can_recycle`` enforces the same ``N`` cap across recycles.) A BARE
          graph run derives the SAME ``N = len(traces)`` target
          (:meth:`_resolved_num_sessions`), so lanes clamp to the corpus size --
          but the bare run stays in SINGLE-PASS mode (``recycle_is_bounded`` is
          False; see :meth:`_recycle_has_stop_condition`), so each lane does one
          pass with clean pass-0 plans and no recycle.
        * The ``recycle_is_bounded=False`` clamp to the corpus size is the bare
          run's actual single-pass bound; the derived session target produces the
          same lane clamp, so both agree.
        """
        lanes = self._max_concurrent
        sessions = self._resolved_num_sessions()
        if sessions is not None and sessions > 0:
            lanes = min(lanes, int(sessions))
        if not recycle_is_bounded:
            lanes = min(lanes, total)
        return max(1, lanes)

    def set_lane_limit(self, limit: int) -> None:
        """Update the live lane-admission limit (``LaneSettableProtocol``).

        Driven by the concurrency ramper (``--concurrency-ramp-duration``):
        lanes park in :meth:`_wait_for_lane_admission` before their first
        instance and are released as the limit rises. Clamped to
        ``[1, _max_concurrent]``; lowering the limit mid-run does not stop
        already-admitted lanes (ramp semantics are raise-only, matching the
        session-slot ramp).
        """
        new_limit = max(1, min(int(limit), self._max_concurrent))
        if new_limit == self._lane_limit:
            return
        previous = self._lane_limit
        self._lane_limit = new_limit
        self._wake_lane_waiters()
        # A raise opens exactly ``new_limit - previous`` slots; hand each to one
        # FIFO waiter rather than waking the whole queue.
        if new_limit > previous:
            self._wake_slot_waiters(new_limit - previous)

    def _wake_lane_waiters(self) -> None:
        """Release every parked lane/slot waiter so it re-checks the limit.

        Swap-then-set: waiters parked on the OLD event wake and re-check; later
        waiters park on the fresh event, so a raise can never be lost.
        """
        released, self._lane_limit_raised = (
            self._lane_limit_raised,
            asyncio.Event(),
        )
        released.set()

    async def _wait_for_lane_admission(self, lane: int) -> None:
        """Park lane ``lane`` until the live lane limit admits it (index < limit)."""
        while lane >= self._lane_limit:
            await self._lane_limit_raised.wait()

    async def _acquire_trace_slot(self) -> int:
        """Park until a slot below the live lane limit is free, then hold it.

        Same gate as the lane path -- the limit is shared, so
        ``--concurrency-ramp-duration`` (``set_lane_limit``) paces admission here
        for free. It cannot call :meth:`_wait_for_lane_admission` directly: that
        parks on a FIXED index, while a timestamped trace must re-check the
        lowest free slot after every release (its index is not owned for the
        phase).

        Waiters queue on a FIFO of futures rather than on the shared broadcast
        event: the timestamped path opens ONE TASK PER TRACE, so waking every
        parked trace on each of N completions would cost O(N^2) wakeups plus
        re-scans in the latency-critical timing-manager loop. One release hands
        the slot to exactly one waiter, in arrival (recorded-start) order.
        """
        while True:
            slot = self._lowest_free_slot()
            if slot < self._lane_limit:
                self._active_trace_slots.add(slot)
                return slot
            waiter = asyncio.get_running_loop().create_future()
            self._slot_waiters.append(waiter)
            try:
                await waiter
            except asyncio.CancelledError:
                if waiter.done() and not waiter.cancelled():
                    # The handoff landed but we are going away: pass it on so
                    # the freed slot is not stranded.
                    self._wake_slot_waiters(1)
                else:
                    self._drop_slot_waiter(waiter)
                raise

    def _wake_slot_waiters(self, count: int) -> None:
        """Hand a freed/opened slot to up to ``count`` FIFO waiters."""
        while count > 0 and self._slot_waiters:
            waiter = self._slot_waiters.popleft()
            if waiter.done():
                continue
            waiter.set_result(None)
            count -= 1

    def _drop_slot_waiter(self, waiter: asyncio.Future) -> None:
        """Remove a cancelled waiter from the FIFO."""
        with contextlib.suppress(ValueError):
            self._slot_waiters.remove(waiter)

    def _lowest_free_slot(self) -> int:
        """Lowest slot index not currently held by a running trace."""
        slot = 0
        while slot in self._active_trace_slots:
            slot += 1
        return slot

    def _release_trace_slot(self, slot: int) -> None:
        """Free ``slot`` and hand it to the longest-waiting trace, if any."""
        self._active_trace_slots.discard(slot)
        self._wake_slot_waiters(1)

    def _resolve_concurrency(self, override: int | None) -> tuple[int, bool]:
        """Resolve the trace-admission bound AND whether it was chosen by the operator.

        Precedence: explicit ``override`` > phase ``concurrency`` (when set and
        positive) > ``1`` (the plain aiperf default). Both the limit and its
        provenance come out of this single evaluation so the two can never
        disagree.

        Provenance is read from the phase config's persisted
        ``concurrency_explicitly_set`` flag, NEVER from the value: the default
        profiling phase type is ``concurrency``, whose ``concurrency`` field
        defaults to a positive ``1``, so every bare run would otherwise look
        like an explicit ceiling of one and serialize the open-loop timestamped
        path to a single trace at a time.
        """
        if override is not None and override > 0:
            return override, True
        cfg_conc = self._config.concurrency
        explicit = bool(
            cfg_conc is not None
            and cfg_conc > 0
            and getattr(self._config, "concurrency_explicitly_set", False)
        )
        if cfg_conc is not None and cfg_conc > 0:
            return int(cfg_conc), explicit
        return 1, False

    def _guard_explicit_oversubscription(self, distinct: int) -> None:
        """Fail loud when concurrency EXPLICITLY over-subscribes a non-wrapping corpus.

        Turns the old silent clone-to-fill into a hard configuration error
        (#1106-adjacent). ``distinct`` is ``len(self._parsed.traces)`` -- the
        per-tree distinct-loaded-traces count. We compare against the RESOLVED
        phase ``concurrency`` (``self._max_concurrent``, the raw over-subscription
        signal), NOT the clamped ``_resolve_lane_count`` (which already folds the
        corpus size in and so could never trip). When concurrency exceeds the
        corpus AND wrapping is not allowed, there are too few distinct traces to
        fill the lanes without cloning the operator never asked for -- so raise.
        EXCEPTION: an explicit session budget (``--num-conversations`` ->
        ``expected_num_sessions``) that fits within the distinct corpus bounds
        total instances below any cloning need, so concurrency stays a mere
        ceiling and the guard stands down.

        An empty corpus (``distinct < 1``) is already fail-loud upstream in the
        adapters; we do not double-handle it. The default concurrency ``1`` never
        exceeds a non-empty corpus, so a plain run never raises.

        ``wrap_allowed`` mirrors ``GraphDispatchResolver``'s default so a
        direct-construction
        path (``_allow_dataset_wrap is None``) never spuriously raises: wrapping is
        allowed when explicitly set True, or -- unset -- when cache-bust is on.
        """
        if distinct < 1:
            return
        from aiperf.common.enums import CacheBustTarget

        wrap_allowed = (
            self._allow_dataset_wrap
            if self._allow_dataset_wrap is not None
            else (self._cache_bust != CacheBustTarget.NONE)
        )
        if self._max_concurrent <= distinct or wrap_allowed:
            return

        # An explicit session budget (--num-conversations) bounds TOTAL
        # instances: when it fits within the distinct corpus, no lane can
        # ever need a clone -- concurrency is only a ceiling on simultaneous
        # lanes, not a demand for that many traces -- so over-provisioned
        # concurrency is not over-subscription.
        sessions = self._config.expected_num_sessions
        if sessions is not None and sessions <= distinct:
            return

        # CAPPED phrasing when the corpus was bounded by a selection knob:
        # ``num_dataset_entries``/``max_context_length`` are threaded onto the
        # phase ``CreditPhaseConfig`` in ``TimingConfig.from_run`` (mirroring
        # allow_dataset_wrap/cache_bust), so a live run distinguishes a capped
        # corpus from one that simply has fewer distinct traces.
        capped_by = [
            flag
            for value, flag in (
                (self._config.num_dataset_entries, "--num-dataset-entries"),
                (self._config.max_context_length, "--max-context-length"),
            )
            if value
        ]
        if capped_by:
            shortfall = f"distinct loaded traces (capped to {distinct} by {'/'.join(capped_by)})"
        else:
            shortfall = (
                f"distinct loaded traces; only {distinct} distinct traces available"
            )

        # Reaching this raise always means wrapping is disallowed, so the note is
        # unconditional and neutral -- it must NOT attribute the disable to the
        # user: ``GraphDispatchResolver`` collapses an UNSET --allow-dataset-wrap to False
        # when cache-bust is off (the default), so ``self._allow_dataset_wrap is
        # False`` cannot distinguish user-explicit-false from resolver-derived.
        parts = [
            f"concurrency {self._max_concurrent} exceeds {distinct} {shortfall}.",
            "Dataset wrapping is disabled, so the corpus cannot fill the requested lanes.",
            f"Reduce --concurrency to <= {distinct}, or pass --allow-dataset-wrap "
            "to intentionally clone/recycle the corpus.",
        ]
        if self._cache_bust is None or self._cache_bust == CacheBustTarget.NONE:
            parts.append(
                "If you enable wrapping, also enable cache-bust (e.g. --cache-bust "
                "first_turn_prefix) so clones do not collide on identical prefixes."
            )
        from aiperf.common.exceptions import ConfigurationError

        raise ConfigurationError(" ".join(parts))

    async def setup_phase(self) -> None:
        """Guard against over-subscription, then install the graph observers.

        The wrap-guard runs FIRST and on THIS awaited path (not ``execute_phase``,
        which ``PhaseRunner`` launches fire-and-forget so a raise there is
        swallowed): ``PhaseRunner`` awaits ``setup_phase`` directly, so a
        ``ConfigurationError`` here propagates up through ``_run_strategy`` to the
        run's failure handler and fails the run loudly (the #1106-adjacent
        contract). It must therefore precede any observer install / dispatch.

        It is LANE-ONLY: it rejects concurrency that exceeds the distinct corpus
        because lanes would have to clone traces to fill themselves. The
        open-loop timestamped path (:meth:`_run_timestamped_traces`) runs each
        loaded trace exactly once and never recycles or clones, so extra
        concurrency there is a harmless unused ceiling, not over-subscription.

        Both observers de-multiplex to the owning per-trace adapter by
        ``trace_id``: the return observer resolves the parked dispatch Future,
        the first-token observer releases a successor gated on a node's observed
        first token (post-TTFT anchoring).
        """
        if not self._open_loop_replay:
            self._guard_explicit_oversubscription(len(self._parsed.traces))
        else:
            # Select HERE, not in ``execute_phase``: the selection is what the
            # timestamp validation must run against (a bound that excludes the
            # untimestamped traces is a legitimate run), and only this awaited
            # path can raise loudly. ``execute_phase`` reuses the result, so the
            # seed-deterministic draw and its re-anchor still happen exactly once.
            self._selected_traces = self._select_replay_corpus(
                list(self._parsed.traces)
            )
            self._validate_recorded_starts(self._selected_traces)
        if self._sandbox_provider is not None:
            await self._sandbox_provider.setup()
        self._register_observer(self._on_graph_return)
        self._register_first_token_observer(self._on_graph_first_token)

    def _record_warmup_failure(self, credit: Credit, error: str) -> None:
        """Account one TERMINAL warmup-phase request failure (AgentX parity).

        Called from :meth:`_on_graph_return` for a WARMUP-phase return carrying
        a non-None ``error`` that is NOT cancelled. Keeps the running count plus
        up to five human-readable samples for the abort message.

        Cancelled returns are deliberately EXCLUDED here, even when they carry
        error text: a cancellation surfacing at drain is self-inflicted teardown
        (duration-cancel / external cancel), not a server failure, so counting
        it would abort otherwise-healthy warmups.
        """
        self._warmup_failure_count += 1
        if len(self._warmup_failure_samples) < 5:
            self._warmup_failure_samples.append(
                f"{credit.trace_id}[node_ordinal={credit.node_ordinal}]: {error}"
            )

    def report_warmup_failures(self) -> None:
        """Raise if this WARMUP phase recorded terminal request failures.

        AgentX parity (``callback_handler.py`` warmup gate + runner report): a
        warmup that could not faithfully prime the server's KV cache must abort
        the run BEFORE profiling -- benchmark numbers from a degraded pool look
        valid and are not. Called by ``PhaseRunner._run_strategy`` after
        returning-complete (never on the cancelled early-return path). No-op for
        PROFILING phases and clean warmups.

        Reuses :class:`TrajectoryWarmupFailedError`; the recorded failure
        samples (``trace_id[node_ordinal]: error``) are passed as its trace
        descriptors so the raised message surfaces both the error text and the
        count. The exception derives its count from ``len(failed_trace_ids)``
        and the samples are capped at 5, so past the cap a synthetic trailing
        entry carries the TRUE total (the message must never under-report a
        degraded pool).
        """
        if not self._is_warmup_phase or self._warmup_failure_count == 0:
            return
        samples = list(self._warmup_failure_samples)
        overflow = self._warmup_failure_count - len(samples)
        if overflow > 0:
            samples.append(
                f"... and {overflow} more failure(s) "
                f"(total {self._warmup_failure_count})"
            )
        raise TrajectoryWarmupFailedError(samples)

    def report_trace_failures(self) -> None:
        """Surface traces whose executor unwound with an error.

        ``_errored_traces`` was previously incremented and never read: no
        exporter, no exit-code gate, no phase-failure check. A run in which
        EVERY trace failed therefore completed, exported, and exited 0, with the
        only evidence a per-instance WARNING buried in the log. Warmup has had
        an equivalent gate (:meth:`report_warmup_failures`) since it shipped;
        profiling had none.

        Two levels, because "some traces failed" and "nothing worked" are
        different events:

        * Any failures at all -> a single aggregated ERROR line with the count
          and up to five samples. Partial failure is normal on a large corpus
          and must NOT abort the run; the operator just has to be able to see it
          without trawling per-instance warnings.
        * Every completed trace failed -> raise, so the run cannot report
          success for a phase that produced no usable work at all (bad segment
          store, every dispatch refused, pool missing everywhere).

        Called by ``PhaseRunner`` after returning-complete, never on the
        cancelled path -- a duration-cancel legitimately unwinds live traces.
        """
        from aiperf.common.exceptions import InvalidStateError

        if self._errored_traces == 0:
            return
        phase = self._config.phase
        samples = list(self._errored_trace_samples)
        overflow = self._errored_traces - len(samples)
        if overflow > 0:
            samples.append(f"... and {overflow} more (total {self._errored_traces})")
        detail = "; ".join(samples)
        # Denominator = traces that FINISHED (a pre-admission lowering failure
        # is a finished attempt too), so the ratio can never exceed 1 and an
        # all-failed run trips the gate even when nothing reached the executor.
        finished = self._finished_traces
        if self._errored_traces >= finished:
            raise InvalidStateError(
                f"every graph trace failed in {phase}: {self._errored_traces}/"
                f"{finished} unwound with errors: {detail}"
            )
        self.error(
            lambda: (
                f"{self._errored_traces}/{finished} graph traces unwound with "
                f"errors in {phase}: {detail}"
            )
        )

    def _on_graph_return(
        self,
        credit: Credit,
        error: str | None,
        cancelled: bool,
        *,
        osl: int | None,
        request_latency_ns: int | None,
        ttft_ns: int | None,
    ) -> None:
        """Route one graph credit return to its owning per-trace adapter.

        The shared ``CreditCallbackHandler`` fires this UNCONDITIONALLY for every
        credit carrying a ``trace_id``. We look the adapter up by the credit's
        ``trace_id`` (the root trace id the adapter stamped on issue) and let it
        resolve / reject the parked dispatch Future. An unknown trace id is a
        graceful no-op (e.g. a late return after the trace already unwound).
        """
        if self._is_warmup_phase and error is not None and not cancelled:
            self._record_warmup_failure(credit, error)
        trace_id = credit.trace_id
        if trace_id is None:
            return
        adapter = self._adapters.get(trace_id)
        if adapter is None:
            node_ordinal = credit.node_ordinal
            _logger.warning(
                lambda: (
                    f"graph return for unknown instance_id={trace_id!r} "
                    f"node_ordinal={node_ordinal!r} dropped: no live adapter is "
                    "registered for this instance (a late return arrived after the "
                    "instance was cancelled/timed-out/errored and its adapter was "
                    "already reaped). The parked dispatch will not resolve from this "
                    "return."
                )
            )
            return
        adapter.resolve(
            credit,
            error,
            cancelled,
            osl=osl,
            request_latency_ns=request_latency_ns,
            ttft_ns=ttft_ns,
        )

    def _on_graph_first_token(self, first_token: FirstToken) -> None:
        """Route one graph credit's first-token event to its owning adapter.

        Mirrors :meth:`_on_graph_return`: the shared ``CreditCallbackHandler``
        fires this for every ``FirstToken`` carrying a ``trace_id`` (a graph
        credit). We look up the owning adapter by that instance id and hand it
        the emitting credit's ``(x_correlation_id, turn_index)`` so it can fire
        the successor's parked first-token callback (post-TTFT anchoring). An
        unknown / ``None`` trace id -- a late token after the trace unwound or
        its instance was reaped, or a non-graph fast-path token -- is a
        graceful no-op (the anchor simply falls back to its dispatch delay).
        """
        trace_id = first_token.trace_id
        if trace_id is None:
            return
        adapter = self._adapters.get(trace_id)
        if adapter is None:
            _logger.debug(
                lambda: (
                    f"graph first-token for unknown instance_id={trace_id!r} "
                    "dropped: no live adapter (successor falls back to dispatch delay)"
                )
            )
            return
        adapter.on_first_token(
            first_token.x_correlation_id,
            first_token.turn_index,
        )

    async def execute_phase(self) -> None:
        """Dispatch every admitted trace, owning completion AgentX-faithfully.

        Stop semantics mirror AgentX's stop-condition model
        (``aiperf.timing.phase.stop_conditions``), NOT a per-trace drain wait:
        ``--num-conversations N`` (``SessionCountStopCondition``) caps admitted
        traces (each recorded trace is one conversation/DAG); ``--request-count N``
        (``RequestCountStopCondition``) caps node dispatches via
        ``issue_graph_credit``'s gate (a refused issue is a clean per-trace stop);
        ``--benchmark-duration D`` (``DurationStopCondition``) caps WALL time via
        :meth:`_run_traces_under_duration_budget`. A recorded trace faithfully
        replays its recorded idle gaps, so a single trace can span its whole
        recorded duration -- the duration budget is what bounds that, exactly as
        AgentX's ``runner.py::_wait_for_sending_complete`` cancels in-flight
        scheduled turns when the duration elapses.

        Completion: a graph credit carries a ``trace_id`` so the linear
        ``CreditCounter`` never trips ``is_final_credit``; THIS strategy owns the
        signal. The ``finally`` ALWAYS freezes the sent count and sets
        ``all_credits_sent_event`` (drain / duration-cancel / error alike), so a
        never-returning executor can never wedge the phase with the event unset.
        The ``PhaseRunner`` then awaits ``all_credits_returned_event`` (set by the
        callback handler once every issued-and-not-cancelled credit returns).
        """
        # ``_selected_traces`` is set by ``setup_phase``; the None fallback is
        # reachable only from direct construction in tests, which deliberately
        # skips corpus bounding and recorded-start validation.
        traces = (
            self._selected_traces
            if self._selected_traces is not None
            else list(self._parsed.traces)
        )
        if not traces:
            self.debug("no traces to replay; phase complete")
            self._credit_issuer.mark_graph_sending_complete()
            self._credit_issuer.set_graph_all_returned_event()
            return

        self._advise_if_long_replay_waits_without_duration(traces)
        try:
            await self._run_traces_under_duration_budget(traces)
        finally:
            # ALWAYS runs (drain / duration-cancel / error): no further graph
            # credit will be issued, so freeze the authoritative per-node sent
            # count and signal sending-complete. If no credit was ever issued
            # (no-op corpus, or every trace cancelled pre-dispatch), also set the
            # returned event since no return will fire it.
            self._credit_issuer.mark_graph_sending_complete()
            if self._credit_issuer.graph_all_returned():
                self._credit_issuer.set_graph_all_returned_event()

    def _select_replay_corpus(self, traces: list[TraceRecord]) -> list[TraceRecord]:
        """Bound the replayed corpus to ``--num-conversations`` on the open-loop path.

        Open-loop timestamped replay enumerates each loaded trace exactly once,
        so without this bound ``--num-conversations`` was accepted and silently
        ignored: a 500-trace corpus ran 500 traces while the operator believed
        10. The bound is applied at SELECTION time through the dataset sampler
        (:meth:`~aiperf.timing.strategies.graph_trace_planner.GraphTracePlanner.select_corpus`),
        which takes the N EARLIEST traces by recorded start -- a contiguous
        slice of the capture, seed-deterministic, and free of the lexicographic
        bias a CORPUS-ORDER prefix would carry (the dynamo adapter id-sorts its
        traces, so corpus order is id order, not arrival order).

        The lane path is deliberately untouched: there an explicit
        ``expected_num_sessions`` already bounds the run as the recycle stop
        condition (:meth:`_can_recycle`), and re-bounding the template pool
        would change which templates lanes draw without changing the admitted
        count.
        """
        if not self._open_loop_replay:
            return traces
        limit = self._config.expected_num_sessions if self._config is not None else None
        selected = self._planner.select_corpus(traces, limit)
        if len(selected) == len(traces):
            return traces
        draw = (
            self._dataset_sampling_strategy.value
            if self._dataset_sampling_strategy is not None
            else "sequential (corpus order)"
        )
        self.info(
            lambda: f"--num-conversations {limit}: replaying {len(selected)} of "
            f"{len(traces)} loaded trace(s), drawn {draw}"
        )
        self._reanchor_schedule_zero(selected)
        return selected

    def _validate_recorded_starts(self, selected: list[TraceRecord]) -> None:
        """Refuse to start when a trace to be paced has no recorded start.

        ``_wait_for_recorded_start`` returns early -- no pacing -- for a trace
        whose graph carries no ``recorded_start_unix_ms``, so a PARTIALLY
        timestamped corpus front-loaded those traces as a burst at t=0 on top of
        an otherwise faithful replay, with nothing in the output saying so. This
        mirrors ``FixedScheduleStrategy.setup_phase``, which raises rather than
        replay part of a corpus unpaced.

        Scope, deliberately narrow:

        * Only the OPEN-LOOP path (``open_loop_replay``); closed-loop replay
          never consults recorded starts, so timestamps are not required there.
        * Only NON-WARMUP phases: ``_run_instance`` skips
          ``_wait_for_recorded_start`` entirely when ``_is_warmup_phase``, so a
          warmup never paces and cannot be silently un-paced.
        * Only the SELECTED corpus: ``--num-conversations`` bounding onto a
          fully timestamped subset is a legitimate run, and traces that are
          never replayed must not fail it.
        * Only a PARTIALLY timestamped corpus. ``recorded_start_unix_ms`` is
          stamped by exactly one producer (the dynamo trie lowering); a
          hand-authored corpus (not included in this release) has none anywhere
          and runs under this same default ``open_loop_replay=True`` paced by
          its AUTHORED EDGE DELAYS -- the documented full replay, not a
          silent front-load. So when there is no corpus-level schedule zero at
          all, there is no faithful timeline to burst on top of and nothing to
          refuse.
        """
        from aiperf.common.exceptions import ConfigurationError

        if (
            not self._open_loop_replay
            or self._is_warmup_phase
            or self._schedule_zero_unix_ms is None
        ):
            # A wholly untimestamped corpus cannot reach here in this release, so
            # there is nothing extra to refuse: ``graph_adapter`` registers only
            # ``dynamo_trace`` (plugins.yaml), whose lowering stamps
            # ``recorded_start_unix_ms`` on EVERY node from a required
            # ``event_time_unix_ms`` (trace_reader.py, trie_lowering.py). If a
            # future adapter can omit timestamps, note that ``--open-loop-strict``
            # would then silently degrade to dependency-gated replay --
            # ``_strict_schedule_projection`` declines to project a graph with no
            # timestamps -- and this is where that must be refused.
            return
        missing = [
            trace.id
            for trace in selected
            if self._graph_record_recorded_start(
                resolve_trace_graph(self._parsed, trace)
            )
            is None
        ]
        if not missing:
            return
        named = missing[:5]
        overflow = len(missing) - len(named)
        listed = ", ".join(named)
        if overflow > 0:
            listed += f", ... and {overflow} more (total {len(missing)})"
        scope = (
            "None of the selected traces are timestamped"
            if len(missing) == len(selected)
            else "This corpus is only partially timestamped"
        )
        raise ConfigurationError(
            f"{scope} and it cannot be replayed "
            f"faithfully: {len(missing)} of {len(selected)} selected trace(s) "
            f"carry no recorded start timestamp ({listed}) while the rest do. "
            f"The untimestamped traces cannot be paced, so they would all fire "
            f"at once at t=0 on top of the faithful replay of the timestamped "
            f"ones, inflating early load with nothing in the results saying so. "
            f"Fix the trace source so every trace carries a recorded start, or "
            f"restrict the corpus to timestamped traces."
        )

    def _reanchor_schedule_zero(self, selected: list[TraceRecord]) -> None:
        """Re-anchor the replay timeline on the SELECTED corpus minimum.

        ``_schedule_zero_unix_ms`` is computed in ``__init__`` over the FULL
        corpus. A bounded selection that excludes the earliest traces would then
        wait ``selection_min - corpus_min`` before its first request -- on an
        hour-long corpus, twenty minutes of dead air, long enough for a
        ``--benchmark-duration`` to expire having issued nothing.

        Re-anchoring shifts every selected trace by the SAME constant, so all
        relative offsets (and therefore inter-trace spacing) are preserved
        exactly; only the artifact of selection is removed. No-op when the
        selection's minimum already IS the corpus minimum, which is always the
        case for a single-graph corpus (every trace shares one graph).
        """
        if self._schedule_zero_unix_ms is None:
            return
        starts = [
            start
            for start in (
                self._graph_record_recorded_start(resolve_trace_graph(self._parsed, t))
                for t in selected
            )
            if start is not None
        ]
        if not starts:
            return
        selection_zero = min(starts)
        if selection_zero > self._schedule_zero_unix_ms:
            self._schedule_zero_unix_ms = selection_zero

    @staticmethod
    def _graph_record_recorded_start(graph: GraphRecord) -> int | None:
        """Earliest preserved source timestamp within one ``GraphRecord``."""
        timestamps = [
            int(node.recorded_start_unix_ms)
            for node in graph.nodes.values()
            if node.recorded_start_unix_ms is not None
        ]
        return min(timestamps) if timestamps else None

    def _has_benchmark_duration(self) -> bool:
        """True iff a ``--benchmark-duration`` wall budget bounds this phase.

        Both sources are checked, and NEITHER alone suffices. The phase config
        (``expected_duration_sec``) is the declared budget, but it is None on a
        phase whose duration is imposed only by the run-level lifecycle. The
        lifecycle (``time_left_in_seconds()``) reflects the live budget, but it
        also returns None BEFORE the phase starts (``started_at_perf_ns is
        None``, see ``PhaseLifecycle.time_left_in_seconds``) -- and this
        advisory can run in that window. Either being set means a duration
        bounds the phase.
        """
        if (
            self._lifecycle is not None
            and self._lifecycle.time_left_in_seconds() is not None
        ):
            return True
        return bool(self._config.expected_duration_sec)

    @staticmethod
    def _strict_global_gap_us(graphs: Iterable[GraphRecord]) -> float | None:
        """Largest gap (us) between consecutive recorded starts across ALL graphs.

        The ``--open-loop-strict`` analogue of an inter-turn delay. The strict
        projection gates every node at its own ``recorded_start_unix_ms -
        trace_zero`` offset, and ``--open-loop-strict`` implies
        ``open_loop_replay``, so each trace is ALSO held to its own recorded start
        (:meth:`_recorded_start_target`). The run is therefore one absolute
        timeline, and the stretch the phase sits silent through is the gap between
        consecutive recorded starts ANYWHERE in the admitted corpus.

        Pooling across graphs is what makes this correct, and the dynamo adapter
        emits ONE GRAPH PER TRACE, so the multi-graph shape is the normal case:

        * per-graph maxima OVER-report two traces that interleave (each has a wide
          internal gap, but the other trace is firing inside it, so the phase is
          never quiet that long);
        * per-graph maxima UNDER-report two internally dense traces recorded an
          hour apart (each looks 1s-tight while the run parks the full hour
          between them).

        Returns ``None`` when NOTHING carries a timestamp, which is precisely when
        :meth:`_strict_schedule_projection` declines to project and leaves the
        recorded edge delays in force -- the caller then scans those instead.
        """
        starts = sorted(
            node.recorded_start_unix_ms
            for graph in graphs
            for node in graph.nodes.values()
            if node.recorded_start_unix_ms is not None
        )
        if not starts:
            return None
        gaps_ms = (b - a for a, b in itertools.pairwise(starts))
        return max(gaps_ms, default=0.0) * 1000.0

    @staticmethod
    def _graph_max_delay_us(graph: GraphRecord, *, collapse_leading: bool) -> float:
        """Largest declared firing-gate delay (us) in one graph.

        ``collapse_leading`` mirrors ``--burst-phase-starts``: when set, the
        LEADING offsets are excluded, because
        :func:`~aiperf.graph.scheduler.collapse_leading_start_offsets` zeroes
        exactly those before the executor sees them -- a START-sourced
        ``min_start_delay_us``, and a node-level one on a node with no non-START
        predecessor. Counting them under burst would report a park of minutes on a
        run that fires immediately. Inter-turn delays are NOT collapsed by burst
        and stay in scope either way.

        Every gate the executor can park a node on:

        * node-level ``min_start_delay_us`` -- never stamped by ANY producer in
          this release (both real stampers,
          ``interval_order.build_interval_edges`` and
          ``snapshot_chop._chop_edges``, write ``StaticEdge``s), and there is no
          agent-graph authoring path either: ``graph_adapter`` registers only
          ``dynamo_trace``. Scanned anyway because it is a decodable schema
          field and a firing gate the executor honors, so a future producer
          cannot make it invisible here;
        * ``StaticEdge.min_start_delay_us`` -- the leading START-relative offset a
          gap-started chain roots at;
        * ``StaticEdge.delay_after_predecessor_us`` -- end-to-start;
        * ``StaticEdge.delay_after_predecessor_start_us`` -- dispatch-to-start;
        * ``StaticEdge.delay_after_predecessor_first_token_us`` -- kind-complete,
          though the first-token refinement is ``<=`` the start delay on the same
          edge, so it never raises the max in practice.

        The executor takes the MAX of whichever apply to a node
        (``TraceExecutor._compute_firing_gate_us``). ``0.0`` for a gate-free graph.
        """
        # Same predicate collapse_leading_start_offsets uses to decide which
        # node-level delays are leading anchors rather than mid-graph pacing.
        has_real_pred = {e.target for e in graph.edges if e.source != START_NODE_ID}
        max_us = 0.0
        for node_id, node in graph.nodes.items():
            if node.min_start_delay_us is None:
                continue
            if collapse_leading and node_id not in has_real_pred:
                continue
            max_us = max(max_us, float(node.min_start_delay_us))
        for edge in graph.edges:
            leading = edge.source == START_NODE_ID
            for name in _EDGE_DELAY_FIELDS:
                if collapse_leading and leading and name == "min_start_delay_us":
                    continue
                value = getattr(edge, name)
                if value is not None:
                    max_us = max(max_us, float(value))
        return max_us

    def _max_replay_wait_seconds(self, traces: list[TraceRecord]) -> float:
        """Largest RECORDED scheduled wait (seconds) across the ADMITTED graphs.

        Scoped to ``traces`` -- the corpus ``execute_phase`` actually runs, after
        ``setup_phase``'s selection/bounding -- resolved per trace through
        :func:`resolve_trace_graph` and deduplicated by identity (a single-graph
        workload has every trace pointing at the same ``GraphRecord``). Scanning
        the whole parse instead would report a wait from a trace that
        ``--num-dataset-entries`` excluded and that will never run.

        Under ``--open-loop-strict`` the executor does not park on the declared
        gates at all: :meth:`_strict_schedule_projection` discards every edge and
        re-roots each node at START on an absolute
        ``recorded_start_unix_ms - trace_zero`` offset, so the silent stretch is
        :meth:`_strict_consecutive_gap_us`. That branch falls back to
        :meth:`_graph_max_delay_us` for a graph carrying no timestamps -- exactly
        as the projection itself falls back to the unprojected graph.

        Returned in RECORDED seconds, before ``--replay-speedup`` divides them
        (:meth:`_scale_timing` rescales a per-trace copy, never ``self._parsed``).
        These waits are NOT necessarily idle time in the capture: an end-to-start
        delay spanning a concurrent long-running request is BUSY in the recording,
        which is exactly why the active-interval idle warp behind
        ``--trace-idle-gap-cap-seconds`` leaves it intact. Returns ``0.0`` for a
        wait-free corpus (the AgentX bare-CLI default, end-to-start delays OFF).
        """
        # Dedup by identity: a single-graph workload resolves every trace to the
        # SAME GraphRecord, so a per-trace scan would be quadratic for nothing.
        graphs = {
            id(g): g for g in (resolve_trace_graph(self._parsed, t) for t in traces)
        }
        if self._open_loop_strict:
            # ONE pooled timeline across every admitted graph, not a per-graph
            # max: concurrent traces fill each other's gaps, and the parks
            # BETWEEN traces belong to no single graph at all.
            strict_us = self._strict_global_gap_us(graphs.values())
            if strict_us is not None:
                return strict_us / MICROS_PER_SECOND
        max_us = 0.0
        for graph in graphs.values():
            max_us = max(
                max_us,
                self._graph_max_delay_us(
                    graph, collapse_leading=self._burst_phase_starts
                ),
            )
        return max_us / MICROS_PER_SECOND

    def _advise_if_long_replay_waits_without_duration(
        self, traces: list[TraceRecord]
    ) -> None:
        """Emit a once-per-phase advisory for a long-parking corpus with no duration.

        A faithful recorded-trace replay honors every recorded scheduled wait
        verbatim, so a count/session/bare run (no ``--benchmark-duration``) spans
        the slowest admitted trace's full recorded wall time -- exactly AgentX's
        count-mode (``is_final_credit`` fires only once the last turn is SENT,
        after every scheduled wait elapses; ``runner._wait_for_sending_complete``
        then waits on that with a ``None`` timeout). This is faithful, NOT a hang,
        but on a human-pace corpus the wall time is minutes-scale with no console
        output while parked. Advise the operator that ``--benchmark-duration``
        bounds it (cancels the still-parked nodes, keeps dispatched records) and
        that Ctrl+C exports the partial results gracefully.

        The threshold is compared against the EFFECTIVE wait -- the recorded max
        divided by ``--replay-speedup`` -- because that is the wall time the
        operator actually experiences, and it is what
        ``IDLE_GAP_NO_DURATION_WARN_SECONDS`` is denominated in. Comparing the raw
        recorded value instead made the advisory fire on sub-second real parks at a
        high speedup (a 30s recorded wait at ``--replay-speedup 60`` is a 0.5s
        park, not something to advise about).

        No-op when a duration is set, when the corpus has no waits, or when the
        effective max is below the threshold. Also a no-op for a WARMUP phase
        (warmup bursts only the boundary priming turns, see
        :func:`rewrite_for_warmup`, and never replays recorded waits) and under
        ``AIPERF_GRAPH_IGNORE_EDGE_DELAYS`` (the executor skips every gate, so
        nothing parks no matter what the corpus carries).
        """
        if self._is_warmup_phase:
            return
        if self._has_benchmark_duration():
            return
        if Environment.GRAPH.IGNORE_EDGE_DELAYS:
            # The executor short-circuits every firing gate before computing it
            # (``TraceExecutor._apply_firing_delay``), so the corpus can carry
            # arbitrarily long delays and still park for zero seconds. Advising
            # about a wait that cannot happen is the same false positive this
            # advisory already emitted by reporting unscaled delays.
            return
        recorded_s = self._max_replay_wait_seconds(traces)
        # ``_scale_timing`` divides every executor-visible delay by the speedup on
        # a per-trace copy, so the wall-clock park is the recorded wait scaled.
        effective_s = recorded_s / self._replay_speedup
        threshold_s = Environment.GRAPH.IDLE_GAP_NO_DURATION_WARN_SECONDS
        if effective_s < threshold_s:
            return
        recorded_note = (
            ""
            if self._replay_speedup == 1.0
            else f" ({recorded_s:.0f}s recorded, /{self._replay_speedup:g} speedup)"
        )
        self.notice(
            lambda: (
                f"replay corpus parks up to {effective_s:.0f}s{recorded_note} "
                "between turns and no --benchmark-duration is set: this phase "
                "replays every recorded delay faithfully, so its wall time spans "
                "the admitted corpus's recorded span with no console output "
                "while parked (under open-loop replay each trace is also held to "
                "its own recorded start, so that span is the whole corpus's, not "
                "just the slowest single trace's). This is the recorded "
                "predecessor-to-successor delay, which is NOT necessarily idle "
                "time in the capture -- a delay spanning a concurrent "
                "long-running request was BUSY in the recording, so "
                "--trace-idle-gap-cap-seconds deliberately leaves it intact (it "
                "collapses only stretches where the WHOLE trace was idle). Pass "
                "--benchmark-duration <seconds> to bound the run (it cancels the "
                "still-parked nodes and keeps the records dispatched so far), "
                "or press Ctrl+C to finalize + export the partial results now."
            )
        )

    async def _run_traces_under_duration_budget(
        self, traces: list[TraceRecord]
    ) -> None:
        """Run the lane fan-out + recycle loop under the phase duration budget.

        No ``--benchmark-duration`` (``time_left_in_seconds() is None``) -> await
        natural completion; every lane recycles fresh templates until the
        stop-condition gate (``--num-conversations`` / ``--request-count``) refuses
        a new session, matching AgentX's count/session-mode. Otherwise on timeout
        the lane-runner task is cancelled, cancelling the outer ``TaskGroup`` and
        every in-flight executor (parked Futures reject, coroutines unwind), so the
        phase finalizes on the ``DurationStopCondition`` exactly as AgentX's
        ``_wait_for_sending_complete`` duration path (Gap-4). Already-returned
        records are kept; only not-yet-fired idle-parked nodes drop.
        """
        timeout = (
            self._lifecycle.time_left_in_seconds()
            if self._lifecycle is not None
            else None
        )
        if timeout is not None:
            self._duration_deadline = self._clock.perf_ns() / 1e9 + timeout
        dispatch = asyncio.ensure_future(self._run_lanes(traces))
        try:
            try:
                await asyncio.wait_for(dispatch, timeout=timeout)
            except TimeoutError:
                # ``wait_for`` already requested cancellation of ``dispatch``
                # (AgentX cancel_all_pending parity); await its unwind so no
                # orphan survives.
                self.notice(
                    lambda: (
                        f"graph phase duration budget ({timeout:.1f}s) elapsed with "
                        f"{self._admitted_traces - self._completed_traces} instance(s) still "
                        "replaying recorded idle gaps; cancelling in-flight executors "
                        "(duration stop condition)"
                    )
                )
                try:
                    await dispatch
                except asyncio.CancelledError:
                    # ``wait_for``'s own cancellation of ``dispatch`` unwinds here
                    # and is absorbed (a healthy duration stop). An EXTERNAL cancel
                    # of THIS task (runner send-timeout, Ctrl+C) racing that unwind
                    # must NOT be swallowed: re-raise when the current task itself
                    # has a pending cancellation.
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
        finally:
            self._duration_deadline = None
            # Duplication report: fire on BOTH natural completion and the
            # duration-cancel path (the MOST recycle-heavy mode -- lanes recycle
            # until the timer). This is the SOLE ``_run_lanes`` caller and the
            # finally runs exactly once, so the report is emitted once per phase.
            self._report_dispatch_duplication(self._instances_started, len(traces))

    def _past_duration_deadline(self) -> bool:
        """True once the active stage's duration budget has elapsed.

        Cooperative twin of the budget wrappers' ``wait_for`` cancel: a cancel
        delivered into a TaskGroup whose children complete constantly can be
        lost on Python 3.11, leaving the recycle loop running unbounded. Lanes
        check this each iteration so the stage halts on time regardless of
        whether the cancel lands.
        """
        deadline = self._duration_deadline
        return deadline is not None and (self._clock.perf_ns() / 1e9 >= deadline)

    async def _run_lanes(self, traces: list[TraceRecord]) -> None:
        """Drive ``concurrency`` recycling lanes over the wrapped corpus.

        AgentX (``trajectory_source.py`` ``_target_size = concurrency``;
        the retired agentic-replay plane's recycle-on-root-final-turn): we build
        ``_resolve_lane_count`` lanes (``concurrency``, clamped only by an
        ``--num-conversations`` cap or a no-stop-condition single pass) -- lane
        ``i`` starts on ``traces[i % N]`` -- and each lane LOOPS, drawing the next
        wrap template (round-robin over
        the corpus) and re-dispatching it onto its freed slot until the
        stop-condition gate refuses a new session. So ``concurrency`` is SUSTAINED
        even when it exceeds the corpus size (the C2 + C1 fix), instead of
        decaying to ``N`` after one corpus pass.

        ``--num-conversations N`` caps the TOTAL number of distinct root sessions
        ever started (``SessionCountStopCondition.can_start_new_session`` -- the
        same gate AgentX recycle consults); ``--request-count N`` caps total
        node/LLM dispatches via ``issue_graph_credit``'s per-dispatch gate (a
        refused issue cleanly stops the in-flight instance, and the lane's next
        recycle attempt sees the gate closed and stops). With NEITHER cap set the
        lanes run a SINGLE corpus pass: each freed lane keeps drawing the next
        unclaimed template until every corpus position has been claimed exactly
        once (no unbounded recycle without a stop condition), so a bare
        ``--input-file <corpus>`` graph run covers the whole corpus and still
        terminates.

        Each lane task runs on ``phase_tg``; each lane swallows its own instance
        errors so one failed instance never aborts the phase.
        """
        if self._open_loop_replay:
            await self._run_timestamped_traces(traces)
            return
        recycle_is_bounded = self._recycle_has_stop_condition()
        lanes = self._resolve_lane_count(len(traces), recycle_is_bounded)
        # Pass-0 lane assignment mirrors AgentX ``_build_trajectories``: lane ``i``
        # (i == spawnable rank) takes the i-th SPAWNABLE trace in corpus order,
        # SKIPPING unspawnable ones, so a trace after an unspawnable one gets the
        # SAME shifted lane_index (= spawnable rank) AgentX gives it -- making the
        # lane-salted t* byte-identical across both engines. The resolution is
        # SEQUENTIAL because spawnability is tested at the lane-salted t* for the
        # candidate's TARGET rank (rank advances only on a spawnable hit), exactly
        # as AgentX seeds ``lane = len(trajectories)`` per draw.
        pass0_traces, corpus_cursor = self._resolve_pass0_lanes(traces, lanes)
        lanes = len(pass0_traces)
        # Recycle draws continue from the corpus position AFTER the last one the
        # pass-0 spawnable resolution consumed (AgentX's recycle reuses the SAME
        # sampler, so it resumes where ``_build_trajectories`` left off -- past the
        # skipped-unspawnable traces too), turn-0 full replay onto the freed slot.
        next_index = corpus_cursor

        # Duplication report: every lane start AND every serial recycle is
        # one more instance dispatched off the finite trace pool. Counting each
        # ``_run_instance`` call here (recycles included) gives the total the
        # duplication factor divides by ``distinct_loaded_traces``. Tracked on an
        # instance attribute (reset here, one ``_run_lanes`` call per phase) so
        # ``_run_traces_under_duration_budget`` can report it even when this lane
        # future is CANCELLED by the duration budget.
        self._instances_started = 0

        async with asyncio.TaskGroup() as phase_tg:
            for lane in range(lanes):

                async def _lane(lane: int = lane) -> None:
                    nonlocal next_index
                    # Lane-admission gate: under a concurrency ramp
                    # (--concurrency-ramp-duration) lanes above the live limit
                    # park here and are admitted as the ramper raises it --
                    # 1 -> concurrency over the ramp duration. No ramp => the
                    # limit already equals _max_concurrent and this returns
                    # immediately (the ramp's purpose is spreading load onto a
                    # cold server; duration-cancel cancels parked waiters
                    # cleanly).
                    await self._wait_for_lane_admission(lane)
                    trace: TraceRecord = pass0_traces[lane]
                    recycle_pass = 0
                    while True:
                        self._instances_started += 1
                        refused = await self._run_instance(trace, lane, recycle_pass)
                        # A refusal means the issuer's stop gate closed (request
                        # cap / cancel): recycling again would only re-refuse.
                        if refused or self._past_duration_deadline():
                            return
                        if recycle_is_bounded:
                            # Recycle while the stop-condition gate still admits
                            # a new session.
                            if not self._can_recycle():
                                return
                        elif next_index >= len(traces):
                            # No stop condition: single corpus pass -- stop once
                            # every corpus position has been claimed once.
                            return
                        template_index = next_index
                        next_index += 1
                        trace = traces[
                            self._planner.draw_index(template_index, len(traces))
                        ]
                        recycle_pass += 1

                phase_tg.create_task(_lane(), name=f"graph-lane:{lane}")

        if not recycle_is_bounded:
            self.info(
                f"no stop condition set: single corpus pass complete, covering "
                f"{self._admitted_traces} of {len(traces)} trace(s)"
            )
        # NOTE: the duplication report is emitted by the SOLE caller
        # (``_run_traces_under_duration_budget``) in a ``finally`` so it fires on
        # both natural completion and duration-cancel; do NOT report here.

    def _report_dispatch_duplication(
        self, total_instances_started: int, distinct_loaded_traces: int
    ) -> None:
        """Report the phase's dispatch duplication factor; warn only when unsafe.

        ``factor = total_instances_started / distinct_loaded_traces`` counts every
        lane start AND every serial recycle as an instance dispatched off the
        finite trace pool, so ``factor > 1`` means the same recorded traces were
        replayed more than once (lanes recycled to sustain concurrency / satisfy
        ``--request-count`` / ``--num-conversations``). This is a REPORT, never a
        failure. A WARNING fires ONLY when the duplication has no cache-bust
        antidote (``self._cache_bust`` is ``None`` / ``NONE``): identical prefixes
        across clones collide in the server KV cache and inflate cache-hit
        metrics. With cache-bust ON every instance mints a distinct marker, so the
        duplication is safe and the report stays quiet. Warmup phases are skipped
        (their boundary priming is meant to warm the cache; duplication there is
        expected), mirroring the idle-gap advisory's warmup carve-out.
        """
        if distinct_loaded_traces <= 0 or self._is_warmup_phase:
            return
        factor = total_instances_started / distinct_loaded_traces
        if factor <= 1.0:
            return
        from aiperf.common.enums import CacheBustTarget

        if self._cache_bust is not None and self._cache_bust != CacheBustTarget.NONE:
            return
        self.warning(
            lambda: (
                f"dispatch duplication factor {factor:.2f}x "
                f"({total_instances_started} instances started over "
                f"{distinct_loaded_traces} distinct loaded trace(s)): lanes "
                "recycled the finite trace pool and cache-bust is OFF, so cloned "
                "instances replay identical first-turn prefixes and collide in the "
                "inference server's KV cache -- inflating prefix-cache-hit metrics. "
                "Set --cache-bust first_turn_prefix to give each instance a "
                "distinct marker, or reduce --concurrency / raise "
                "--num-dataset-entries so the corpus covers the load without reuse."
            )
        )

    async def _run_timestamped_traces(self, traces: list[TraceRecord]) -> None:
        """Launch timestamped traces together so starts are independent.

        One task per trace: each waits out its recorded start independently, so
        a trace held at the admission gate slips only itself. An EXPLICIT
        ``--concurrency`` bounds how many trace executors run at once; with none
        set every trace runs as soon as its recorded start arrives.
        """
        self._instances_started = 0
        gate_admission = self._concurrency_is_explicit

        async def run_trace(trace: TraceRecord, index: int) -> None:
            self._instances_started += 1
            await self._run_instance(trace, index, 0, gate_admission=gate_admission)

        async with asyncio.TaskGroup() as phase_tg:
            for index, trace in enumerate(traces):
                phase_tg.create_task(
                    run_trace(trace, index), name=f"graph-timestamp:{index}"
                )

    def _resolve_pass0_lanes(
        self, traces: list[TraceRecord], lanes: int
    ) -> tuple[list[TraceRecord], int]:
        """Resolve the pass-0 lane->trace map (AgentX spawnable-rank assignment).

        Walks ``traces`` in corpus order assigning lane ``i`` (== spawnable rank)
        to the next SPAWNABLE trace, computing each candidate's spawnability at
        the lane-salted t* for its TARGET rank (so rank advances only on a hit --
        the sequential dependence AgentX's ``lane = len(trajectories)`` loop has).
        Unspawnable candidates are skipped (no lane), exactly mirroring
        ``TrajectorySource._build_trajectories`` skipping a ``None`` from
        ``_build_trajectory_for_lane`` (``_snapshot_for`` is None). Returns the
        per-lane trace list (length ``<= lanes``; shorter only when the corpus has
        too few spawnable traces) AND the corpus cursor one past the last consumed
        position, so recycle resumes the wrap there.

        The per-candidate corpus draw routes through
        :meth:`~aiperf.timing.strategies.graph_trace_planner.GraphTracePlanner.draw_index`, so
        ``--dataset-sampling-strategy`` selects WHICH template each rank serves;
        under the default ``sequential`` draw it is ``traces[cursor % n]``
        unchanged. With the default ``[0, 0]`` window (t*=0) every trace is
        spawnable (full replay always dispatches), so ``sequential`` collapses to
        the prior raw-position assignment (lane ``i`` -> ``traces[i]``)
        byte-for-byte.
        """
        pass0: list[TraceRecord] = []
        cursor = 0
        n = len(traces)
        # Bound the skip walk to one full corpus pass past the wrap of ``lanes``
        # so an all-unspawnable corpus can't spin (AgentX caps at
        # ``_target_size + 2 * pool_size``); we stop at ``lanes`` hits regardless.
        max_cursor = lanes + n
        while len(pass0) < lanes and cursor < max_cursor:
            trace = traces[self._planner.draw_index(cursor, n)]
            cursor += 1
            rank = len(pass0)
            if self._is_spawnable(trace, rank):
                pass0.append(trace)
        return pass0, cursor

    def _is_spawnable(self, trace: TraceRecord, lane_index: int) -> bool:
        """Whether ``trace`` yields a live dispatch at its lane-salted t* (rank).

        Mirrors AgentX ``TrajectorySource._build_timestamped_trajectory`` ->
        ``_snapshot_for`` returning ``None``: a trace is UNSPAWNABLE when, at the
        t* sampled for ``(trace, lane_index)``, there is no live, dispatchable
        stream -- AgentX's ``not states`` / ``not any(not waiting_on_children)``.
        The native equivalent over :func:`compute_snapshot` is "the snapshot has
        at least one PROFILED LLM firing": a profiled LLM firing is a turn at/after
        t* that survived the spawn-completion gate (AgentX's live, non-future
        stream), and an LLM turn is the only thing AgentX would dispatch (markers
        carry no request) -- so a snapshot with zero profiled LLM firings is one
        AgentX builds no sendable session for and skips.

        At t*=0 (the default ``[0, 0]`` window, or a zero-duration trace) every
        trace's full timeline is profiled, so this is always True -- the prior
        no-skip behavior, byte-for-byte.

        The lane-salted t* is computed directly
        (``GraphTracePlanner.lane_salted_t_star``) rather than through
        ``GraphTracePlanner.plan_for_lane`` so the spawnability scan never builds the
        per-lane catalog / node partition (only the t* + one snapshot are needed);
        the value is byte-identical to the plan's ``t_star_us`` because both call
        the same ``_sample_t_star`` math, so spawnability and the eventual
        dispatch plan agree.
        """
        t_star_us = self._planner.lane_salted_t_star(trace, lane_index)
        if t_star_us <= 0:
            return True
        from aiperf.dataset.graph.models import NodeKind
        from aiperf.graph.analysis import compute_snapshot

        # ``compute_snapshot`` resolves the trace's own graph internally
        # (``elaborate_trace`` -> ``resolve_trace_graph``), so passing the raw
        # ``self._parsed`` is correct here.
        snapshot = compute_snapshot(self._parsed, trace, t_star_us=int(t_star_us))
        return any(sf.firing.kind == NodeKind.LLM for sf in snapshot.profiled)

    def _recycle_has_stop_condition(self) -> bool:
        """True iff an EXPLICIT stop condition exists that recycle can saturate toward.

        Recycle runs ONLY when the user set ``--num-conversations``
        (``expected_num_sessions``) / ``--request-count``
        (``total_expected_requests``) / ``--benchmark-duration``. With none, the
        corpus is replayed once and the lanes stop (a bare ``--input-file <corpus>``
        must NOT recycle forever).

        The DERIVED single-pass session target
        (:meth:`_resolved_num_sessions` = ``len(traces)`` for a bare run) is
        DELIBERATELY EXCLUDED here: it is a lane-clamp + reported-target only, not
        a recycle-enabling stop condition. Routing it in would flip a bare run out
        of SINGLE-PASS mode into the bounded/recycle path -- serving recycle
        passes at t*=0 instead of the clean lane-salted pass-0 plans a
        cover-the-corpus-once pass must keep. So a bare run stays single-pass:
        each lane does one pass, no recycle.
        """
        cfg = self._config
        return bool(
            (
                cfg.expected_num_sessions
                or cfg.total_expected_requests
                or cfg.expected_duration_sec
            )
            or (
                self._lifecycle is not None
                and self._lifecycle.time_left_in_seconds() is not None
            )
        )

    def _can_recycle(self) -> bool:
        """Whether a freed lane may start a fresh root (AgentX recycle gate).

        Mirrors AgentX ``_dispatch_recycled_on_lane`` consulting
        ``stop_checker.can_start_new_session()``, but the v1 ``CreditCounter``
        does NOT bump ``sent_sessions`` for graph credits (they bypass the linear
        session arithmetic -- ``credit_counter.increment_sent``), so the
        ``--num-conversations`` cap must be counted by the STRATEGY: the number of
        distinct root sessions ever started is ``_admitted_traces`` (one per
        instance run), and the cap is ``expected_num_sessions``. The request-count
        and cancellation caps ARE expressed in the counter, so we also honor
        ``can_send_dag_child_turn`` (which includes ``RequestCountStopCondition``
        and the DAG cancellation gate but excludes the session gate). The duration
        cap is enforced by the outer ``wait_for`` (Gap-4 cancel-in-flight), not
        here, so an in-flight instance is never abandoned mid-replay by this gate.

        Reads the EXPLICIT ``expected_num_sessions`` only -- never the derived
        single-pass target -- so this gate is consulted solely on the bounded
        (explicit-stop) recycle path. A bare run never reaches here (it takes the
        single-pass branch in :meth:`_run_lanes`), so the derived ``N`` is not a
        recycle cap.
        """
        sessions = (
            self._config.expected_num_sessions if self._config is not None else None
        )
        if sessions is not None and sessions > 0 and self._admitted_traces >= sessions:
            return False
        checker = self._stop_checker
        if checker is None:
            return True
        return bool(checker.can_send_dag_child_turn())

    async def _run_instance(
        self,
        trace: TraceRecord,
        lane_index: int,
        recycle_pass: int,
        *,
        gate_admission: bool = False,
    ) -> bool:
        """Run ONE trace instance on ``lane_index`` (recycle pass ``recycle_pass``).

        Returns True iff the instance stopped cleanly at the issuer's stop gate.

        The instance id ``f"{trace.id}::{uuid4().hex}"`` is stamped
        on every credit's ``trace_id`` so the cache-bust marker ROTATES per recycle
        AND decorrelates concurrent lanes that wrapped onto the SAME template (C3 +
        AgentX's per-lane ``trajectory_index`` in the digest), while the build-time
        catalog + graph store mmap stay keyed by the base template id (the
        worker strips everything after the first ``::``). The id is GLOBALLY UNIQUE
        per instance, so two lanes wrapping the same template never collide on
        the return-routing registry. Pass 0 uses the lane-salted t* plan (C2);
        recycle passes (>0) run the full t*=0 replay -- AgentX recycle restarts a
        session from turn 0 with no snapshot/t* slice
        (the retired agentic-replay plane's recycled-lane dispatch).

        A per-instance adapter is registered under the instance id for return
        routing, bound to a ``TraceExecutor``, and awaited. An instance error is
        counted (``errored_traces``) and contained -- it does not propagate out of
        the lane's ``TaskGroup`` (which would cancel sibling lanes).

        Args:
            gate_admission: When True, the instance acquires a trace slot from
                the shared lane limit AFTER its recorded-start wait and holds it
                for the executor run; the ``finally`` releases it on every exit
                path (clean, refused, errored, cancelled) so a failed instance
                cannot leak a slot and stall the traces parked behind it. The
                open-loop timestamped path passes True only for an EXPLICIT
                ``--concurrency``; the lane path parks by lane index instead and
                leaves this False.
        """
        refused = False
        slot: int | None = None
        # Admitted/completed are a PAIR (credit_counter.py's in-flight report
        # subtracts them), so an instance that never reached the executor
        # increments neither.
        admitted = False
        # {template}::{nonce}: instance identity is template + fresh nonce
        # (uuid4 -- collision-proof across concurrent lanes, recycles, phases,
        # and runs). Lane / pass are diagnostics, logged below instead of
        # encoded in the id.
        instance_id = f"{trace.id}::{uuid.uuid4().hex}"
        self.debug(
            lambda t=trace.id, i=instance_id, ln=lane_index, k=recycle_pass: (
                f"instance {i} (template={t} lane={ln} pass={k})"
            )
        )
        plan = (
            self._planner.plan_for_lane(trace, lane_index)
            if recycle_pass == 0
            else None
        )
        parsed: ParsedGraph | None = None
        try:
            parsed, run_trace = self._planner.graph_at_t_star(
                trace,
                plan,
                warmup=self._warmup_kind,
                burst_phase_starts=self._burst_phase_starts,
            )
            if not self._is_warmup_phase:
                if not await self._wait_for_recorded_start(parsed):
                    # Admission closed while this instance was parked on its
                    # recorded start: the stop gate is shut for good, so this
                    # trace would only be refused. Report it as a clean stop
                    # (never a trace error) and leave before dispatching.
                    return True
                parsed = self._scale_timing(parsed)
                if self._open_loop_strict:
                    parsed = self._strict_schedule_projection(parsed)
            if gate_admission:
                # AFTER the recorded-start wait: the schedule is computed from a
                # fixed anchor and is immutable. A trace held here starts late
                # (execution slips) but its computed offset -- and every other
                # trace's -- is untouched.
                slot = await self._acquire_trace_slot()
            # Registered only once the instance is cleared to RUN, so
            # ``_admitted_traces`` (and the drain's in-flight report) counts
            # running traces and adapter allocation stays bounded by the live
            # limit instead of by corpus size.
            adapter = self._build_adapter(
                trace.id,
                instance_id,
                first_token_sources=self._first_token_sources_for(trace),
                node_identity=self._node_identity_for(trace),
            )
            self._adapters[instance_id] = adapter
            self._admitted_traces += 1
            admitted = True
            executor = TraceExecutor(
                parsed,
                credit_issuer=adapter,
                tool_dispatcher=self._build_tool_dispatcher(instance_id, trace),
                compress_edge_delays=False,
                # Every live stream's frontier turn must fire at its ABSOLUTE
                # offset from t*, not relative to when
                # its inputs arrive. Anchor node-level leading offsets to the
                # shared instance run-start so co-scoped subagent/worker streams
                # interleave in recorded-time order instead of
                # drifting behind their spawn-parent. Correct for every phase of
                # this strategy: WARMUP primes boundary turns at min_start_delay
                # 0 (anchor is a no-op) and t*=0 full replay anchors to the trace
                # start, which IS the run-start.
                absolute_start_offsets=True,
            )
            self._record_trace_timing(await executor.run(run_trace))
        except Exception as exc:
            refusal = _leaf_credit_refusal(exc)
            if refusal is not None:
                # Issuer refusal is a HEALTHY stop (request-count / duration cap
                # reached, or run cancelled), not a trace error: log quietly and
                # keep ``errored_traces`` untouched.
                refused = True
                # The gate is monotonic, so every later dispatch would refuse
                # too: release the traces parked on their recorded start
                # (open-loop path) instead of leaving them to sit out the rest
                # of the recorded timeline.
                self._admission_closed.set()
                self._release_parked_traces()
                self.debug(
                    lambda exc=refusal, iid=instance_id: (
                        f"graph instance {iid!r} "
                        f"stopped cleanly at the issuer's stop gate: {exc!r}"
                    )
                )
            else:
                self._errored_traces += 1
                if len(self._errored_trace_samples) < 5:
                    self._errored_trace_samples.append(f"{instance_id}: {exc!r}")
                self.warning(
                    lambda exc=exc, iid=instance_id: (
                        f"graph instance {iid!r} unwound with error: {exc!r}"
                    )
                )
        finally:
            if slot is not None:
                self._release_trace_slot(slot)
            self._finished_traces += 1
            if admitted:
                self._completed_traces += 1
            async with self._registry_lock:
                self._parent_done.add(instance_id)
                self._release_adapter_if_idle(instance_id)
        return refused

    def _build_tool_dispatcher(
        self, instance_id: str, trace: TraceRecord
    ) -> SandboxToolDispatcher | None:
        """Build this instance's tool dispatcher, or ``None`` for a tool-free graph.

        ONE dispatcher per trace instance because ``SandboxToolDispatcher``
        holds a single sandbox and refuses two traces at once. The sandbox
        itself is produced by the phase-level ``_sandbox_provider``, whose
        ``setup()`` already ran before timing began (images pre-pulled,
        pools warmed). Returns ``None`` when the graph carries no
        ``ToolNode``, so a plain replay never constructs a sandbox.
        """
        if not self._has_tool_nodes:
            return None

        assert self._sandbox_provider is not None

        def _factory(trace_id: str) -> ToolSandbox:
            # `trace_id` is deliberately unused: the workspace is keyed by
            # INSTANCE so two concurrent replays of the same template trace
            # cannot collide.
            del trace_id
            return self._sandbox_provider.make_sandbox(instance_id, trace)  # type: ignore[union-attr]

        return SandboxToolDispatcher(_factory)

    def _build_sandbox_provider(
        self, parsed_graph: ParsedGraph
    ) -> SandboxProvider | None:
        """Build the phase-level provider, or ``None`` for tool-free graphs.

        Collects the full set of unique Docker images from the phase's traces
        (per-trace override wins over global fallback) so ``DockerSandboxProvider``
        can pre-pull all of them in one ``setup()`` call before timing starts.
        Traces with no image (neither per-trace nor global) drive local execution
        and don't contribute to the image set. The provider also keeps SWE-Bench
        ``/testbed`` traces on per-trace containers so each starts from its image's
        pristine checkout rather than a pooled trace's mutated filesystem.
        """
        if not self._has_tool_nodes:
            return None

        assert self._tool_workspace_root is not None

        images: frozenset[str] = frozenset(
            filter(
                None,
                (
                    (trace.tool_sandbox.container if trace.tool_sandbox else None)
                    or self._graph_tool_image
                    for trace in parsed_graph.traces
                ),
            )
        )

        if images:
            non_pooled_images = frozenset(
                (trace.tool_sandbox.container or self._graph_tool_image)
                for trace in parsed_graph.traces
                if trace.tool_sandbox is not None
                and trace.tool_sandbox.cwd == "/testbed"
                and (trace.tool_sandbox.container or self._graph_tool_image) is not None
            )
            return DockerSandboxProvider(
                images=images,
                workspace_root=self._tool_workspace_root,
                global_image=self._graph_tool_image,
                persistent_session=self._graph_tool_persistent_session,
                pool_size=self._config.concurrency,
                non_pooled_images=non_pooled_images,
            )
        return LocalSandboxProvider(workspace_root=self._tool_workspace_root)

    def _record_trace_timing(self, result: TraceResult) -> None:
        """Fold one finished instance's timing data into the run totals.

        Records per-trace breakdown (model time, tool time, sandbox setup time,
        total wall time, call counts) into ``_trace_summaries`` for the JSON
        artifact, and accumulates the flat tool-duration list for the existing
        ``profile_export_graph_tool_time.json`` artifact.
        """
        model_s = sum(result.llm_durations_s)
        tool_s = sum(result.tool_durations_s)
        setup_s = result.sandbox_setup_s
        total_s = result.trace_wall_s
        model_calls = len(result.llm_durations_s)
        tool_calls = len(result.tool_durations_s)
        if len(result.llm_request_latency_s) == model_calls:
            normalization_durations_s = [
                request_latency_s
                if request_latency_s is not None
                else dispatch_duration_s
                for dispatch_duration_s, request_latency_s in zip(
                    result.llm_durations_s,
                    result.llm_request_latency_s,
                    strict=True,
                )
            ]
            normalized_model_s, low_osl_calls = _compute_normalized_model_s(
                normalization_durations_s,
                result.llm_ttft_s,
                result.llm_target_osl,
                result.llm_observed_osl,
            )
        else:
            normalized_model_s, low_osl_calls = None, 0
        self._trace_summaries.append(
            {
                "trace_id": result.trace_id,
                "total_s": total_s,
                "model_s": model_s,
                "tool_s": tool_s,
                "sandbox_setup_s": setup_s,
                "model_time_fraction": model_s / total_s if total_s > 0 else 0.0,
                "tool_time_fraction": tool_s / total_s if total_s > 0 else 0.0,
                "model_calls": model_calls,
                "tool_calls": tool_calls,
                "normalized_model_s": normalized_model_s,
                "low_osl_model_calls": low_osl_calls,
            }
        )
        if not result.tool_durations_s:
            return
        self._tool_durations_s.extend(result.tool_durations_s)
        self._tool_traces += 1
        count = len(result.tool_durations_s)
        self.debug(
            lambda tid=result.trace_id, n=count, t=tool_s: (
                f"trace {tid!r} executed {n} tool command(s) in {t:.3f}s"
            )
        )

    def report_tool_execution(self) -> None:
        """Log this phase's aggregate tool wall time and write a JSON artifact.

        Emitted at phase teardown. The aggregate is also written to
        ``profile_export_graph_tool_time.json`` in ``artifact_dir`` so the
        measurement survives outside the run log and is accessible to
        analysis scripts without grepping.

        Skipped for warmup phases to prevent a deferred warmup teardown from
        clobbering the profiling phase's artifact (same file path).
        """
        if self._is_warmup_phase or not self._tool_durations_s:
            return
        backend = (
            "docker:" + self._graph_tool_image if self._graph_tool_image else "local"
        )
        durations = sorted(self._tool_durations_s)
        total = sum(durations)
        mid = durations[len(durations) // 2]
        self.info(
            f"tool execution: {len(durations):,} commands across "
            f"{self._tool_traces:,} traces, {total:.3f}s total, "
            f"{total / len(durations):.3f}s mean, {mid:.3f}s median, "
            f"{durations[-1]:.3f}s max "
            f"(backend={backend})"
        )
        artifact_dir = getattr(self._config, "artifact_dir", None)
        if artifact_dir is not None:
            try:
                _write_tool_time_artifact(
                    Path(artifact_dir) / "profile_export_graph_tool_time.json",
                    durations=list(self._tool_durations_s),
                    traces=self._tool_traces,
                    backend=backend,
                )
            except Exception as exc:
                self.warning(
                    lambda exc=exc: f"failed to write tool-time artifact: {exc!r}"
                )

    def report_trace_summary(self) -> None:
        """Log per-trace wall-time breakdown and write a JSON artifact.

        Emitted at phase teardown alongside ``report_tool_execution``. The
        breakdown mirrors Agent Trace Replay's per-trace ``summary`` block:
        total / model / tool wall time, fractions, and call counts.
        Written to ``profile_export_graph_trace_summary.json``.

        Skipped for warmup phases: warmup burst-replays only boundary turns at
        zero edge delay, so its ``total_s`` ≈ ``model_s`` and the fractions are
        semantically meaningless. Skipping also prevents a deferred warmup
        teardown from clobbering the profiling phase's artifact.
        """
        if self._is_warmup_phase or not self._trace_summaries:
            return
        agg_total = sum(s["total_s"] for s in self._trace_summaries)
        agg_model = sum(s["model_s"] for s in self._trace_summaries)
        agg_tool = sum(s["tool_s"] for s in self._trace_summaries)
        n = len(self._trace_summaries)
        self.info(
            f"trace summary: {n} trace(s), "
            f"{agg_total:.3f}s total, "
            f"{agg_model:.3f}s model ({agg_model / agg_total * 100:.1f}%), "
            f"{agg_tool:.3f}s tool ({agg_tool / agg_total * 100:.1f}%)"
            if agg_total > 0
            else f"trace summary: {n} trace(s), 0.000s total"
        )
        artifact_dir = getattr(self._config, "artifact_dir", None)
        if artifact_dir is not None:
            try:
                _write_trace_summary_artifact(
                    Path(artifact_dir) / "profile_export_graph_trace_summary.json",
                    summaries=self._trace_summaries,
                )
            except Exception as exc:
                self.warning(
                    lambda exc=exc: f"failed to write trace-summary artifact: {exc!r}"
                )

    def _release_adapter_if_idle(self, instance_id: str) -> None:
        """Pop ``instance_id``'s adapter once its parent finished and it is idle.

        A mid-run drain of a still-running parent's dispatches must not reap
        the adapter (``instance_id`` not yet in ``_parent_done``); the parent
        finally and the last return's ``on_drained`` both funnel here so the
        pop happens exactly once, whichever lands second.

        Idempotent: a second call after the pop is a no-op. Callers MUST hold
        ``_registry_lock``.
        """
        if instance_id not in self._parent_done:
            return
        adapter = self._adapters.get(instance_id)
        if adapter is not None and adapter.inflight_count > 0:
            return
        self._adapters.pop(instance_id, None)
        self._parent_done.discard(instance_id)
        if adapter is not None:
            self._send_graph_trace_end(adapter)

    def _on_adapter_drained(self, adapter: CreditDispatchAdapter) -> None:
        """Adapter callback: its last in-flight dispatch just returned.

        Fires synchronously from ``adapter.resolve`` on the event loop. We cannot
        ``await`` the registry lock here, so schedule a lock-guarded retry of the
        deferred pop; the adapter's ``instance_id`` is its de-mux registry key.
        Drives the idle-pop for an instance whose parent finally already ran
        with returns still in flight. A spurious schedule (adapter already
        popped) is harmless.
        """

        async def _release(instance_id: str = adapter.instance_id) -> None:
            async with self._registry_lock:
                self._release_adapter_if_idle(instance_id)

        task = asyncio.get_running_loop().create_task(_release())
        self._release_tasks.add(task)
        task.add_done_callback(self._release_tasks.discard)

    def _first_token_sources_for(self, trace: TraceRecord) -> frozenset[str]:
        """Source node ids of ``trace``'s first-token-anchored edges (cached).

        Computed from the trace's OWN projected graph (``parsed_for_trace``) --
        the same per-trace topology the executor dispatches -- and cached per
        template trace id (lane/t*-independent). This matches the executed graph
        exactly for the default t*=0 replay; under a t*>0 snapshot chop the set
        is a safe superset (a chop only DROPS edges, so a source whose anchor
        edge was chopped no longer dispatches -- its stray ``first_token_event``
        flag is inert -- and no chopped graph can contain an anchor edge absent
        from the projection).
        """
        cached = self._first_token_sources_cache.get(trace.id)
        if cached is None:
            graph = parsed_for_trace(self._parsed, trace).graph
            cached = first_token_sources(graph)
            self._first_token_sources_cache[trace.id] = cached
        return cached

    def _node_identity_for(
        self, trace: TraceRecord
    ) -> dict[str, tuple[int, str | None]] | None:
        """``trace``'s ``node_id -> (agent_depth, parent_node_id)`` map (cached).

        Reads the dag_jsonl lowering's ``metadata["dag"]`` legacy-identity stamp
        off the trace's OWN projected graph (``parsed_for_trace``), cached per
        template trace id -- the identity is lane/t*-independent (a chop /
        warmup rewrite keeps surviving node ids, so a superset map is safe).
        Returns ``None`` when no node carries the stamp (weka/dynamo) so the
        adapter's root-chain fallback stays byte-identical.
        """
        if trace.id in self._node_identity_cache:
            return self._node_identity_cache[trace.id]
        graph = parsed_for_trace(self._parsed, trace).graph
        identity = {
            nid: (m["agent_depth"], m.get("parent_node"))
            for nid, node in graph.nodes.items()
            if (m := (node.metadata or {}).get("dag"))
        }
        result = identity or None
        self._node_identity_cache[trace.id] = result
        return result

    def _build_adapter(
        self,
        trace_id: str,
        instance_id: str,
        *,
        first_token_sources: frozenset[str] = frozenset(),
        node_identity: dict[str, tuple[int, str | None]] | None = None,
    ) -> CreditDispatchAdapter:
        """Construct a fresh per-instance ``CreditDispatchAdapter``.

        ``trace_id`` is the BASE template id keying the catalog + graph store mmap;
        ``instance_id`` (``{trace_id}::{nonce}``) is stamped on every
        credit's ``trace_id`` for marker rotation + return de-mux. Routing keys
        are per TRAJECTORY: the adapter lazily mints one
        ``{conversation_id}::{nonce}`` x_correlation_id per scope, so
        concurrent instances never share a key. ``first_token_sources`` names
        the nodes whose dispatch emits a ``FirstToken`` (post-TTFT anchoring).
        ``node_identity`` is the trace's dag legacy-identity map (agent_depth /
        parent corr stamping); None for stamp-free producers.
        """
        return CreditDispatchAdapter(
            credit_issuer=self._credit_issuer,
            catalog_context=self._planner.catalog,
            trace_id=trace_id,
            instance_id=instance_id,
            phase=self._config.phase,
            dispatch_timeout_s=self._dispatch_timeout_s,
            on_drained=self._on_adapter_drained,
            first_token_sources=first_token_sources,
            node_identity=node_identity,
        )

    async def handle_credit_return(
        self, credit: Credit, *, error: str | None = None
    ) -> None:
        """No-op for graph credits.

        Graph returns are resolved by the UNCONDITIONAL observer
        (``_on_graph_return`` -> ``adapter.resolve``) BEFORE this gated path is
        reached, so there is nothing to do here. We do NOT route graph returns
        through this method: it is skipped once ``can_send_any_turn()`` is False,
        which would strand already-issued dispatch Futures.
        """
        return

    def _send_graph_trace_end(self, adapter: CreditDispatchAdapter) -> None:
        """Fire-and-forget the trace's sticky-lifecycle close (GraphTraceEnd).

        Called from the adapter-reap points -- the successful idle-pop (all
        in-flight dispatches drained) and phase teardown for retained adapters
        -- NOT the per-instance finally, which can run while credits are still
        in flight (duration-end local cancel, dispatch timeout, watchdog, error
        unwind). ONE call per instance: graph sessions key on the instance
        trace_id, so the router closes the whole instance synchronously before
        forwarding to the worker; the chain is idempotent, and the worker pool
        has an LRU backstop for lost sends.
        """
        task = asyncio.create_task(
            self._credit_issuer.end_graph_trace(adapter.instance_id)
        )
        self._trace_end_tasks.add(task)
        task.add_done_callback(self._trace_end_tasks.discard)

    def _detach_observer(
        self,
        unregister: Callable[[Any], None],
        own_observer: Callable[..., None],
        label: str,
    ) -> None:
        """Best-effort detach of one shared observer slot at teardown.

        Compare-and-clear (``unregister(own_observer)``: the handler clears the
        slot only if OUR observer is still installed -- see
        ``CreditCallbackHandler.clear_graph_return_observer``). Exceptions are
        swallowed at debug: teardown must never mask the phase's own exit path.
        """
        try:
            unregister(own_observer)
        except Exception as exc:
            self.debug(lambda exc=exc: f"{label} observer detach failed: {exc!r}")

    async def teardown_phase(self) -> None:
        """Detach the graph-return observer after the phase finalizes.

        Invoked by ``PhaseRunner`` in its ``run()`` ``finally`` (see
        ``PhaseTeardownStrategyProtocol``); tests may also call it directly.
        Best-effort: a subsequent phase / cleanup must not dispatch into this
        torn-down strategy's adapter registry. Also reaps any de-mux entry retained
        for an instance not popped at its finally -- the phase is over, so no further
        return will arrive and the registry must not leak into the next phase --
        and closes the sticky lifecycle for every retained adapter.

        The SYNC parts (observer detach + registry clear) run FIRST so they
        complete even when the phase is being cancelled and the awaited sticky
        closes below get interrupted at their first suspension point.

        Detach is compare-and-clear, NOT unconditional: for seamless non-final
        phases this teardown is deferred to the background return-wait
        completion and can fire AFTER the next phase's ``setup_phase`` installed
        ITS observer on the same shared ``CreditCallbackHandler`` slot. Clearing
        unconditionally here would drop every subsequent graph return of the
        live phase, so the slot is cleared only if OUR observer is still the
        one installed.
        """
        # Before the sync teardown: the aggregate is the mode's headline
        # measurement, so it must be reported even if a later teardown step
        # raises.
        self.report_tool_execution()
        self.report_trace_summary()
        retained = list(self._adapters.values())
        self._adapters.clear()
        self._parent_done.clear()
        self._detach_observer(
            self._unregister_observer,
            self._on_graph_return,
            "graph-return",
        )
        self._detach_observer(
            self._unregister_first_token_observer,
            self._on_graph_first_token,
            "first-token",
        )
        for adapter in retained:
            try:
                await self._credit_issuer.end_graph_trace(adapter.instance_id)
            except Exception as exc:
                self.debug(lambda exc=exc: f"teardown trace-end failed: {exc!r}")
        if self._sandbox_provider is not None:
            with contextlib.suppress(Exception):
                await self._sandbox_provider.teardown()
