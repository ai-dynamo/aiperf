# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraphIRReplayStrategy — the weka IR trace runner on the v1 credit pipeline.

This timing strategy drives the dataflow ``TraceExecutor`` once per
weka trace and OWNS both completion and concurrency, rather than letting the
linear ``CreditCounter`` / session-slot arithmetic decide. A fan-out trace has
no linear acquire/release; graph credits flow through real issuance -> worker ->
records -> returns but BYPASS the session-slot lifecycle via
``CreditIssuer.issue_graph_credit``.

Ownership model
---------------
* **Completion.** AgentX-faithful (``aiperf.timing.phase.stop_conditions``): the
  phase finalizes on the STOP CONDITION (session count / request count /
  duration), NOT on every admitted trace fully draining its faithful idle-gap
  replay. ``execute_phase``'s ``finally`` always signals sending-complete, and a
  ``--benchmark-duration`` cancels in-flight executors when the budget elapses
  (see :meth:`GraphIRReplayStrategy._run_traces_under_duration_budget`). We do
  NOT consult ``CreditCounter.is_final_turn`` (it cannot express a DAG).
* **Concurrency.** A fixed pool of ``max_concurrent_traces`` lanes (see
  :meth:`GraphIRReplayStrategy._run_lanes` / ``_resolve_lane_count``) admits
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
import hashlib
import time
import uuid
from typing import TYPE_CHECKING, Any

import msgspec

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import MICROS_PER_SECOND
from aiperf.common.environment import Environment
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.scenario.base import TrajectoryWarmupFailedError
from aiperf.dataset.graph.graph_path_catalog import build_catalog_context
from aiperf.dataset.graph.models import START_NODE_ID, StaticEdge
from aiperf.graph.credit_dispatch_adapter import (
    CreditDispatchAdapter,
    CreditIssueRefusedError,
)
from aiperf.graph.executor import TraceExecutor
from aiperf.graph.scheduler import collapse_leading_start_offsets
from aiperf.timing.graph_ir_source import GraphIRConversationSource, GraphTrace
from aiperf.timing.graph_ir_trace_view import parsed_for_trace
from aiperf.timing.graph_warmup_handoff import (
    GraphWarmupHandoff,
    LaneHandoff,
)
from aiperf.timing.snapshot_chop import chop_trie_at_frontier, chop_trie_at_tstar

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiperf.common.enums import CacheBustTarget
    from aiperf.credit.messages import FirstToken
    from aiperf.credit.structs import Credit
    from aiperf.dataset.graph.models import (
        GraphRecord,
        LlmNode,
        ParsedGraph,
        TraceRecord,
    )
    from aiperf.plugin.enums import DatasetSamplingStrategy
    from aiperf.timing.graph_channel import GraphPhaseChannel

_logger = AIPerfLogger(__name__)

__all__ = ["GraphIRReplayStrategy", "first_token_sources", "rewrite_for_warmup"]


def first_token_sources(graph: GraphRecord) -> frozenset[str]:
    """Source node ids of every first-token-anchored ``StaticEdge`` in ``graph``.

    A ``StaticEdge`` carrying ``delay_after_predecessor_first_token_us`` anchors
    its successor to the SOURCE node's observed first token (post-TTFT anchoring,
    validator rule 55). That source must therefore emit a ``FirstToken`` --
    i.e. its issued turn carries ``first_token_event=True`` -- so the runtime can
    release the successor when the token arrives. Returns the (possibly empty)
    set of such source node ids; a gap-free / pre-anchoring graph yields the
    empty set (no node emits a first-token event).
    """
    return frozenset(
        edge.source
        for edge in getattr(graph, "edges", [])
        if isinstance(edge, StaticEdge)
        and edge.delay_after_predecessor_first_token_us is not None
    )


def _chain_key(node_id: str) -> str:
    """Return the per-session chain key a trie node id belongs to.

    Both live trie producers mint node ids as ``{chain_prefix}_{ordinal}`` --
    one linear chain per recorded session: the weka walk emits ``{trace_id}:{k}``
    for the root request list and ``r_1_0``/``r_1_1`` for the subagent spawned
    at index 1 (``adapters/weka/trie_build._walk``), and the dynamo lowering
    emits ``{session_id}:{k}`` per session
    (``adapters/dynamo/trie_lowering``). Stripping the final ``_``-delimited
    token therefore recovers the enclosing session chain (root chain + each
    subagent chain) -- the same per-session decomposition the trie build's
    recorded structure encodes. Chain identity must come from node ids because
    the sidecar-loaded timing plane strips ``metadata["trie"]`` contents (hash
    ids / segment paths), leaving ids + edges as the only runtime chain signal,
    and the interval-order edges are cross-chain ordering edges, not session
    boundaries. A node id with no ``_`` forms a defensive singleton chain.
    """
    prefix, sep, _ = node_id.rpartition("_")
    return prefix if sep else node_id


def _warmup_boundary_nodes(graph: GraphRecord, t_star_us: float) -> dict[str, LlmNode]:
    """Return ``{node_id: node}`` of each chain-live-at-``t*`` boundary turn.

    Chains are the per-session linear paths the trie node ids encode
    (:func:`_chain_key`), ordered by recorded arrival. A chain is LIVE when it
    has BOTH a node arriving before ``t*`` and a node arriving at/after ``t*``;
    its boundary is the LAST pre-``t*`` node. Chains with no pre-``t*`` node
    need no priming (profiling replays them from their own start); chains
    entirely pre-``t*`` are not live (nothing of them is profiled).
    """
    chains: dict[str, list[tuple[int, str]]] = {}
    for nid, node in graph.nodes.items():
        arrival = getattr(node, "arrival_offset_us", None) or 0
        chains.setdefault(_chain_key(nid), []).append((arrival, nid))
    boundary: dict[str, LlmNode] = {}
    for members in chains.values():
        members.sort()
        pre = [nid for arrival, nid in members if arrival < t_star_us]
        if pre and any(arrival >= t_star_us for arrival, _ in members):
            boundary[pre[-1]] = graph.nodes[pre[-1]]
    return boundary


def rewrite_for_warmup(parsed: ParsedGraph, t_star_us: float) -> ParsedGraph:
    """Rewrite ``parsed`` into the WARMUP boundary-priming graph at ``t*``.

    AgentX-parity contract (``timing.config._build_graph_auto_warmup_config``):
    warmup dispatches exactly ONE priming credit per chain LIVE at ``t*`` --
    the chain's boundary turn, the last node of that per-session chain whose
    recorded arrival precedes ``t*`` (:func:`_warmup_boundary_nodes`). Because
    trie prompts are cumulative along a chain, priming the boundary turn's
    prompt (at the worker-side warmup ``max_tokens`` cap, keyed off the
    ``"warmup"`` phase variant) warms the chain's whole prefix.

    The produced graph is FLAT: only the boundary nodes survive, each re-rooted
    from ``START`` with NO leading offset (warmup bursts every priming credit
    at phase start rather than replaying recorded gaps) and with fan-in
    ``inputs`` cleared (their predecessors are gone). Node identity, the trie
    envelope, and ``dispatch_overrides`` are preserved so the worker resolves
    the unmodified catalog ordinal and materializes the exact recorded prompt.
    ``t_star_us <= 0`` (full native replay, or a zero-duration trace) yields an
    EMPTY graph so the warmup phase finalizes immediately.
    """
    graph = parsed.graph
    boundary = _warmup_boundary_nodes(graph, t_star_us) if t_star_us > 0 else {}
    new_nodes = {
        nid: msgspec.structs.replace(node, inputs=[], min_start_delay_us=None)
        for nid, node in boundary.items()
    }
    new_edges = [StaticEdge(source=START_NODE_ID, target=nid) for nid in new_nodes]
    new_graph = msgspec.structs.replace(graph, nodes=new_nodes, edges=new_edges)
    return msgspec.structs.replace(parsed, graph=new_graph)


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


def _seed_for_draw_pass(base_seed: int, pass_index: int) -> int:
    """Derive a per-pass RNG seed for the shuffle draw (matches the t* salt).

    Mirrors :func:`aiperf.timing.graph_ir_source._seed_for_trace_lane`: SHA-256
    over ``f"{base_seed}:dataset-draw:{pass_index}"`` and take the low 8 bytes,
    so each recycle pass re-permutes under a distinct-yet-deterministic seed
    derived from the run's ``t_star_random_seed``. Same base seed + pass index
    always yields the same permutation (cross-run reproducibility), while
    different passes decorrelate -- the same order-independent SHA-256 derivation
    the conversation-plane samplers use via ``rng.derive``.
    """
    digest = hashlib.sha256(f"{base_seed}:dataset-draw:{pass_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class GraphIRReplayStrategy(AIPerfLoggerMixin):
    """Drive a ``TraceExecutor`` per weka trace over the v1 credit pipeline.

    Constructed per-phase by ``PhaseRunner._build_strategy`` with the standard
    timing-strategy kwargs PLUS the graph-only injection channel
    (``parsed_graph`` + ``register_observer``); see the module docstring for the
    ownership contract.
    """

    def __init__(
        self,
        *,
        config: Any = None,
        conversation_source: Any = None,
        scheduler: Any = None,
        stop_checker: Any = None,
        credit_issuer: Any,
        lifecycle: Any = None,
        parsed_graph: ParsedGraph,
        register_observer: Callable[[Any], None],
        register_first_token_observer: Callable[[Any], None] | None = None,
        unregister_observer: Callable[[Any], None] | None = None,
        unregister_first_token_observer: Callable[[Any], None] | None = None,
        max_concurrent_traces: int | None = None,
        dispatch_timeout_s: float | None = None,
        start_min_ratio: float = 0.0,
        start_max_ratio: float = 0.0,
        t_star_random_seed: int = 0,
        burst_phase_starts: bool = False,
        cache_pressure_duration_s: float | None = None,
        warmup_handoff: GraphWarmupHandoff | None = None,
        graph_channel: GraphPhaseChannel | None = None,
        dataset_sampling_strategy: DatasetSamplingStrategy | None = None,
        allow_dataset_wrap: bool | None = None,
        cache_bust: CacheBustTarget | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the graph trace runner.

        Args:
            config: Per-phase ``CreditPhaseConfig`` (concurrency bound + stop
                thresholds: ``expected_num_sessions`` / ``total_expected_requests``).
            conversation_source: Accepted for timing-strategy signature parity;
                unused by graph runs (``None``), which carry state on
                ``graph_channel`` instead.
            scheduler: Unused (loop scheduler); protocol parity.
            stop_checker: Unused here; the issuer honors the stop gate itself.
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
                (post-TTFT anchoring). ``None`` (default, e.g. unit harness)
                skips first-token routing entirely -- anchored edges then fall
                back to their dispatch-time delay.
            unregister_observer: Compare-and-clear detach for the graph-return
                observer (``CreditCallbackHandler.clear_graph_return_observer``):
                teardown passes its OWN observer and the handler clears the
                shared slot only if that observer is still installed. Required
                for seamless multi-phase runs where a stale phase's deferred
                teardown fires after the next phase installed its observer.
                ``None`` (unit harness) falls back to ``register_observer(None)``.
            unregister_first_token_observer: Same compare-and-clear detach for
                the first-token observer slot. ``None`` falls back to
                ``register_first_token_observer(None)``.
            max_concurrent_traces: Trace-admission bound; defaults to the phase
                ``concurrency`` else ``1`` (the plain aiperf default).
            dispatch_timeout_s: Per-dispatch deadlock guard (adapter default).
            start_min_ratio: Lower bound (fraction of duration) of the t* window.
                Default ``0.0``; together with ``start_max_ratio=0.0`` this
                selects full native replay (t*=0, no snapshot rewrite). The
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
            cache_pressure_duration_s: Extended (cache-pressure) warmup stage
                budget in seconds (``float | None``). Non-None on a WARMUP phase
                arms the pressure stage: after boundary priming drains, the
                post-t* graphs replay compressed for this many seconds, then
                drain into the profiling handoff. None (default) disables it,
                keeping warmup byte-identical.
            warmup_handoff: The consume-once ``GraphWarmupHandoff | None`` a prior
                extended warmup stashed on the shared channel, threaded into the
                first PROFILING graph phase so each lane resumes at its recorded
                execution frontier. None (default) means no prior extended warmup.
            graph_channel: The typed ``GraphPhaseChannel`` threaded from the
                orchestrator; a WARMUP phase stashes its ``warmup_handoff`` here
                at teardown for the next PROFILING phase to consume. None in unit
                harnesses that construct the strategy directly.
            dataset_sampling_strategy: Resolved run-level dataset sampling
                strategy (from ``run.resolved.dataset_sampling_strategy`` via the
                CreditPhaseConfig). Consumed by the per-lane trace draw
                (``_draw_index``): SHUFFLE/RANDOM remap the draw through a seeded
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
        """
        super().__init__(logger_name="GraphIRReplayTiming")
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
        self._graph_channel = graph_channel
        self._cache_pressure_duration_s = cache_pressure_duration_s
        self._warmup_handoff = warmup_handoff
        # Resolved dataset-selection policy threaded from ``run.resolved`` via
        # the CreditPhaseConfig. Consumed by the setup-phase wrap-guard and the
        # per-lane sampling draw (``_draw_index``).
        self._dataset_sampling_strategy = dataset_sampling_strategy
        self._allow_dataset_wrap = allow_dataset_wrap
        # Resolved cache-bust target threaded from ``endpoint.cache_bust`` via the
        # CreditPhaseConfig. Consumed by the dispatch duplication report below.
        self._cache_bust = cache_bust
        # Duration stop condition (AgentX parity): ``time_left_in_seconds()``
        # bounds the dispatch; ``None`` with no ``--benchmark-duration``.
        self._lifecycle = lifecycle

        self._start_min_ratio = start_min_ratio
        self._start_max_ratio = start_max_ratio
        self._t_star_random_seed = t_star_random_seed
        self._init_tstar_source(
            parsed_graph, start_min_ratio, start_max_ratio, t_star_random_seed
        )
        self._init_accelerated_warmup()

        self._max_concurrent = self._resolve_concurrency(max_concurrent_traces)
        # Live lane-admission limit (LaneSettableProtocol): the concurrency
        # ramper drives it 1 -> _max_concurrent; without a ramp every lane is
        # admitted immediately. The event is swapped fresh on every raise so
        # parked lanes re-check without lost wakeups (single-threaded asyncio:
        # a waiter's check-then-await pair cannot be preempted by the setter).
        self._lane_limit = self._max_concurrent
        self._lane_limit_raised = asyncio.Event()
        self._completed_traces = 0
        self._errored_traces = 0
        self._admitted_traces = 0
        # Total instances dispatched off the finite pool (lane starts + every
        # serial recycle) for the dispatch duplication report. An instance
        # attribute so the report fires on BOTH natural completion AND
        # duration-cancel (the ``_run_lanes`` local would be lost when the lane
        # future is cancelled by the duration budget).
        self._instances_started = 0
        # Loop-clock deadline for the ACTIVE duration-budgeted stage (phase
        # duration or pressure duration); None outside one. The budget wrappers
        # cancel the lane fan-out task on timeout, but cancellation delivery
        # through a TaskGroup whose children complete constantly is unreliable
        # on Python 3.11 (the cancel can be lost mid-abort), so the lane
        # recycle loops ALSO check this deadline cooperatively -- the stage
        # then halts on time even when the cancel never lands.
        self._duration_deadline: float | None = None
        # Per-(template_trace_id, lane_index) t* plan cache. AgentX seeds t* per
        # ``(trace_id, lane)`` so the same template recurring across lanes (or
        # recycled onto one lane) resumes at a DIFFERENT t*. Built lazily so a
        # large concurrency over a small corpus only plans the lanes it runs.
        self._lane_plans: dict[tuple[str, int], GraphTrace] = {}
        # Per-lane t* source cache (catalog/namespace map are lane-independent),
        # so a fan-out reuses one ``GraphIRConversationSource`` per lane instead
        # of rebuilding the corpus catalog on every per-trace plan.
        self._lane_sources: dict[int, GraphIRConversationSource] = {}
        # Per-(total, pass) shuffled trace-index permutation cache for the
        # ``--dataset-sampling-strategy`` draw (:meth:`_draw_index`). Built lazily
        # once per pass and reused for every draw in that pass so repeated draws
        # are cheap and consistent. A single event loop mutates it with no await
        # between read and write, so no lock is needed. Empty / unused under the
        # default SEQUENTIAL draw.
        self._draw_perm_cache: dict[tuple[int, int], list[int]] = {}

    def _init_tstar_source(
        self,
        parsed_graph: ParsedGraph,
        start_min_ratio: float,
        start_max_ratio: float,
        t_star_random_seed: int,
    ) -> None:
        """Build the t* snapshot source, per-trace t* plans, and node catalog.

        The t* source samples a per-trace snapshot instant and partitions nodes
        into warmup (history) / profiling. Default ratios [0, 0] => t*=0 => full
        native replay (identity rewrite), so the profiling path is byte-identical
        unless a caller supplies a positive window.
        """
        self._source = GraphIRConversationSource(
            parsed=parsed_graph,
            start_min_ratio=start_min_ratio,
            start_max_ratio=start_max_ratio,
            random_seed=t_star_random_seed,
        )
        # ``{trace_id: GraphTrace}`` -- the lane-0 per-trace t* plan (the default
        # single-pass disposition). Lanes > 0 and recycle passes resolve their own
        # lane-salted plan lazily via :meth:`_plan_for_lane`.
        self._plans = {gt.trace_id: gt for gt in self._source.iter_traces()}
        self._catalog = build_catalog_context(parsed_graph)
        # ``{instance_id: CreditDispatchAdapter}`` -- the de-mux registry the
        # single return observer routes by ``credit.trace_id`` (the per-recycle
        # instance id, e.g. ``t-1#0``), so two concurrent instances of one
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

    def _init_accelerated_warmup(self) -> None:
        """Decide whether this phase runs the accelerated cache-pressure warmup.

        Active ONLY when this is the WARMUP phase AND a cache-pressure
        duration is configured (``--agentic-cache-warmup-duration``): every
        WARMUP ``TraceExecutor`` then builds with ``compress_edge_delays=True``
        (zero-idle replay) to drive the server KV cache to pressure.
        PROFILING and the warmup output cap are untouched; without a duration
        warmup honors every captured edge delay.
        """
        from aiperf.common.enums import CreditPhase

        phase = getattr(self._config, "phase", None)
        self._accelerated_warmup = bool(
            phase == CreditPhase.WARMUP and self._cache_pressure_duration_s is not None
        )
        # Extended (cache-pressure) warmup stage state. The ledger records
        # every NON-CANCELLED warmup-phase return (priming + pressure) so the
        # teardown handoff can anchor residual delays; cancelled returns are
        # excluded (not executed -- profiling refires them). Keyed
        # {instance_id: {node_id: wall_us}} on the _wall_us() monotonic clock.
        self._pressure_enabled = bool(
            phase == CreditPhase.WARMUP and self._cache_pressure_duration_s is not None
        )
        self._pressure_active = False
        self._pressure_live: dict[int, tuple[str, str, float, int]] = {}
        # (template_id, lane) -> the warmup boundary-PRIMING instance id run on
        # that lane. Instance ids carry fresh nonces (no longer recomputable),
        # so the pressure handoff's pass-0 priming-wall merge looks the id up
        # here instead of reconstructing it.
        self._priming_instance_ids: dict[tuple[str, int], str] = {}
        self._return_walls: dict[str, dict[str, float]] = {}
        self._ordinal_to_node: dict[str, dict[int, str]] = {}
        # Next corpus draw index for the pressure recycle loop; stashed into the
        # handoff so profiling's bounded recycle continues from it. A single
        # event loop mutates it, so an instance attr has the same atomicity the
        # prior ``nonlocal`` closure had, and it survives past the lane loop for
        # the teardown stash.
        self._pressure_next_index = 0
        # Number of pressure lanes this warmup fanned out (== len(pass0_traces)
        # in _run_pressure_lanes). Stashed into the handoff so profiling can
        # fresh-start any drained-empty lane below this count instead of
        # re-running a t* resume the pressure stage already executed.
        self._pressure_lane_count = 0
        # Warmup-failure abort state (agentx parity): a WARMUP phase that
        # recorded terminal request failures aborts the run before profiling
        # (see :meth:`report_warmup_failures`). Gated on the phase, not on the
        # pressure stage, so plain boundary priming is covered too.
        self._is_warmup_phase = bool(phase == CreditPhase.WARMUP)
        self._warmup_failure_count = 0
        self._warmup_failure_samples: list[str] = []

    @property
    def accelerated_warmup(self) -> bool:
        """True when this phase runs the accelerated cache-pressure warmup."""
        return self._accelerated_warmup

    @property
    def completed_traces(self) -> int:
        """Number of traces whose executor run finished (success OR error)."""
        return self._completed_traces

    @property
    def errored_traces(self) -> int:
        """Number of traces whose executor run unwound with an exception."""
        return self._errored_traces

    @property
    def admitted_traces(self) -> int:
        """Number of traces admitted (run started) this phase."""
        return self._admitted_traces

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
        ``--benchmark-duration``), and for WARMUP phases (their priming /
        cache-pressure fan-out keeps its historical behavior).
        """
        explicit = getattr(self._config, "expected_num_sessions", None)
        if explicit is not None and explicit > 0:
            return int(explicit)
        if self._is_warmup_phase:
            return None
        if (
            getattr(self._config, "total_expected_requests", None)
            or getattr(self._config, "expected_duration_sec", None)
            or (
                self._lifecycle is not None
                and self._lifecycle.time_left_in_seconds() is not None
            )
        ):
            return None
        traces = getattr(self._parsed, "traces", None)
        return len(traces) if traces else None

    def _resolve_lane_count(self, total: int, recycle_is_bounded: bool) -> int:
        """Resolve how many concurrent lanes this phase fans out.

        AgentX builds exactly ``concurrency`` lanes wrapping the corpus
        (``trajectory_source.py:212`` ``_target_size = concurrency``), so lane
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
        self._lane_limit = new_limit
        # Swap-then-set: waiters parked on the OLD event wake and re-check;
        # later waiters park on the fresh event.
        released, self._lane_limit_raised = (
            self._lane_limit_raised,
            asyncio.Event(),
        )
        released.set()

    async def _wait_for_lane_admission(self, lane: int) -> None:
        """Park lane ``lane`` until the live lane limit admits it (index < limit)."""
        while lane >= self._lane_limit:
            await self._lane_limit_raised.wait()

    def _resolve_concurrency(self, override: int | None) -> int:
        """Resolve the trace-admission bound.

        Precedence: explicit ``override`` > phase ``concurrency`` (when set and
        positive) > ``1`` (the plain aiperf default).
        """
        if override is not None and override > 0:
            return override
        cfg_conc = getattr(self._config, "concurrency", None)
        if cfg_conc is not None and cfg_conc > 0:
            return int(cfg_conc)
        return 1

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
        sessions = getattr(self._config, "expected_num_sessions", None)
        if sessions is not None and sessions <= distinct:
            return

        # CAPPED phrasing when the corpus was bounded by a selection knob:
        # ``num_dataset_entries``/``max_context_length`` are threaded onto the
        # phase ``CreditPhaseConfig`` in ``TimingConfig.from_run`` (mirroring
        # allow_dataset_wrap/cache_bust), so a live run distinguishes a capped
        # corpus from one that simply has fewer distinct traces.
        capped_by = [
            flag
            for attr, flag in (
                ("num_dataset_entries", "--num-dataset-entries"),
                ("max_context_length", "--max-context-length"),
            )
            if getattr(self._config, attr, None)
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

    def _plan_for_lane(self, trace: Any, lane_index: int) -> GraphTrace | None:
        """Resolve the t* plan for ``trace`` on ``lane_index`` (AgentX lane salt).

        Lane ``0`` reuses the prebuilt ``_plans`` entry (byte-identical to the
        single-pass path). Higher lanes / recycle passes draw a DISTINCT
        lane-salted t* (``sha256(seed:trace_id:lane)``) so the same template
        recurring across lanes resumes at a different snapshot instant, exactly
        as AgentX's ``_build_trajectory_for_lane`` seeds per absolute lane. The
        per-lane source is cached by ``(trace_id, lane_index)`` so repeated
        recycle passes onto the same lane re-plan only once. With the default
        ``[0, 0]`` window every lane collapses to ``t*=0`` (identity), so lane
        fan-out adds no t* divergence on the working profiling path.
        """
        key = (trace.id, lane_index)
        cached = self._lane_plans.get(key)
        if cached is not None:
            return cached
        if lane_index == 0:
            plan = self._plans.get(trace.id)
        else:
            # Plan ONLY this trace on the lane-salted source. ``iter_traces`` would
            # plan EVERY corpus trace just to find one (O(lanes * traces) snapshot
            # elaborations across a fan-out), so call ``_plan_trace`` directly. The
            # per-lane source is cached by ``lane_index`` (catalog / namespace map
            # are lane-independent) so a fan-out reuses one source per lane.
            plan = self._lane_source(lane_index)._plan_trace(trace)
        if plan is not None:
            self._lane_plans[key] = plan
        return plan

    def _lane_source(self, lane_index: int) -> GraphIRConversationSource:
        """Return the (cached) lane-salted t* source for ``lane_index``.

        The source's catalog + namespace map are lane-independent, so one source
        per lane is reused across every trace planned on that lane (avoids the
        O(lanes * traces) catalog rebuild a per-call construction would incur).
        """
        source = self._lane_sources.get(lane_index)
        if source is None:
            source = GraphIRConversationSource(
                parsed=self._parsed,
                start_min_ratio=self._start_min_ratio,
                start_max_ratio=self._start_max_ratio,
                random_seed=self._t_star_random_seed,
                lane_index=lane_index,
            )
            self._lane_sources[lane_index] = source
        return source

    def _draw_index(self, x: int, total: int) -> int:
        """Remap a monotonic draw counter ``x`` to a trace index in ``[0, total)``.

        This is the single choke point every cross-trace draw in the lane
        fan-out / recycle loop routes through (:meth:`_resolve_pass0_lanes`, the
        index-safe pass-0 fallback, the profiling recycle draw, the pressure
        fresh-start + recycle draws), so ``--dataset-sampling-strategy`` governs
        WHICH template a freed lane serves without changing the draw counters.

        * ``sequential`` (or ``None``): return ``x % total`` -- byte-for-byte the
          historical cursor-with-wrap draw. Sequential must be unchanged.
        * ``shuffle``: map ``x`` to ``perm[pass][x % total]`` where
          ``pass = x // total``, drawing each pass's permutation from a
          pass-salted seed (:func:`_seed_for_draw_pass`). Each pass of ``total``
          draws covers every index exactly ONCE (without replacement), then a
          fresh seeded permutation begins -- the same music-shuffle contract the
          conversation-plane ``ShuffleSampler`` provides.
        * ``random``: coerced to ``shuffle`` (without-replacement) semantics.
          Each lane recycle here is a single corpus pass, so with-replacement
          ``random`` would duplicate/omit templates within a pass; coercing to
          shuffle keeps coverage exact. random == shuffle in this context.
        """
        if total <= 0:
            return 0
        if not self._draw_is_shuffled():
            return x % total
        pass_index, offset = divmod(x, total)
        return self._draw_permutation(pass_index, total)[offset]

    def _draw_is_shuffled(self) -> bool:
        """True iff the resolved sampling strategy permutes (shuffle / random).

        ``None`` and ``sequential`` take the byte-identical ``x % total`` draw;
        ``shuffle`` and ``random`` (coerced to without-replacement) permute.
        """
        strategy = self._dataset_sampling_strategy
        if strategy is None:
            return False
        from aiperf.plugin.enums import DatasetSamplingStrategy

        return strategy in (
            DatasetSamplingStrategy.SHUFFLE,
            DatasetSamplingStrategy.RANDOM,
        )

    def _draw_permutation(self, pass_index: int, total: int) -> list[int]:
        """Return the cached seeded permutation of ``range(total)`` for a pass.

        Built once per ``(total, pass_index)`` from a pass-salted numpy RNG
        (:func:`_seed_for_draw_pass` -> ``np.random.default_rng`` -> in-place
        Fisher-Yates ``shuffle``, matching ``ShuffleSampler``'s numpy shuffle),
        then reused for every draw in that pass so draws are cheap + consistent.
        """
        key = (total, pass_index)
        cached = self._draw_perm_cache.get(key)
        if cached is not None:
            return cached
        import numpy as np

        rng = np.random.default_rng(
            _seed_for_draw_pass(self._t_star_random_seed, pass_index)
        )
        perm = list(range(total))
        rng.shuffle(perm)
        self._draw_perm_cache[key] = perm
        return perm

    async def setup_phase(self) -> None:
        """Guard against over-subscription, then install the graph observers.

        The wrap-guard runs FIRST and on THIS awaited path (not ``execute_phase``,
        which ``PhaseRunner`` launches fire-and-forget so a raise there is
        swallowed): ``PhaseRunner`` awaits ``setup_phase`` directly, so a
        ``ConfigurationError`` here propagates up through ``_run_strategy`` to the
        run's failure handler and fails the run loudly (the #1106-adjacent
        contract). It must therefore precede any observer install / dispatch.

        Both observers de-multiplex to the owning per-trace adapter by
        ``trace_id``: the return observer resolves the parked dispatch Future,
        the first-token observer releases a successor gated on a node's observed
        first token (post-TTFT anchoring). The first-token observer is installed
        only when a registrar was wired (``None`` in the unit harness).
        """
        self._guard_explicit_oversubscription(len(self._parsed.traces))
        self._register_observer(self._on_graph_return)
        if self._register_first_token_observer is not None:
            self._register_first_token_observer(self._on_graph_first_token)

    @staticmethod
    def _wall_us() -> float:
        """Monotonic wall instant in microseconds (ledger + handoff clock)."""
        return time.perf_counter_ns() / 1_000.0

    def _record_return_wall(self, credit: Credit) -> None:
        """Ledger one warmup-phase return: ``{instance_id: {node_id: wall}}``.

        Runs BEFORE adapter de-mux so late returns after a drain-cancel (the
        adapter already reaped) still land in the ledger -- they were real
        server-side executions and must count as executed in the handoff.
        """
        instance_id = getattr(credit, "trace_id", None)
        ordinal = getattr(credit, "node_ordinal", None)
        if instance_id is None or ordinal is None:
            return
        template_id = instance_id.split("::", 1)[0]
        inverse = self._ordinal_to_node.get(template_id)
        if inverse is None:
            inverse = {
                o: nid for nid, o in self._catalog.catalog.get(template_id, {}).items()
            }
            self._ordinal_to_node[template_id] = inverse
        node_id = inverse.get(ordinal)
        if node_id is None:
            return
        self._return_walls.setdefault(instance_id, {})[node_id] = self._wall_us()

    def _record_warmup_failure(self, credit: Credit, error: str) -> None:
        """Account one TERMINAL warmup-phase request failure (agentx parity).

        Called from :meth:`_on_graph_return` for a WARMUP-phase return carrying
        a non-None ``error`` that is NOT cancelled. Keeps the running count plus
        up to five human-readable samples for the abort message.

        Cancelled returns are deliberately EXCLUDED here, even when they carry
        error text. AgentX counts a cancelled warmup credit as a failure because
        its warmup drain PAUSES the issuance gate and lets genuinely in-flight
        server requests finish -- a cancellation there means the server never
        answered. Our extended-warmup drain instead CANCELS the executor
        coroutines on the pressure duration timer (:meth:`_run_pressure_stage`),
        so a cancellation surfacing at drain is self-inflicted teardown, not a
        server failure; counting it would abort every healthy timed warmup.
        """
        self._warmup_failure_count += 1
        if len(self._warmup_failure_samples) < 5:
            self._warmup_failure_samples.append(
                f"{getattr(credit, 'trace_id', '?')}"
                f"[node_ordinal={getattr(credit, 'node_ordinal', '?')}]: {error}"
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

    def _on_graph_return(
        self, credit: Credit, error: str | None, cancelled: bool
    ) -> None:
        """Route one graph credit return to its owning per-trace adapter.

        The shared ``CreditCallbackHandler`` fires this UNCONDITIONALLY for every
        credit carrying a ``trace_id``. We look the adapter up by the credit's
        ``trace_id`` (the root trace id the adapter stamped on issue) and let it
        resolve / reject the parked dispatch Future. An unknown trace id is a
        graceful no-op (e.g. a late return after the trace already unwound).
        """
        if self._pressure_enabled and not cancelled:
            # Wire-cancelled turns are NOT executed: the grace-expiry drain
            # cancels in-flight requests, and the server may never have
            # completed them. Keeping them out of the ledger keeps them out of
            # executed_node_ids, so profiling refires them -- which also makes
            # a SUCCESSFUL cancel-drain (all credits returned-or-cancelled)
            # yield a VALID handoff instead of needing the completeness skip.
            self._record_return_wall(credit)
        if self._is_warmup_phase and error is not None and not cancelled:
            self._record_warmup_failure(credit, error)
        trace_id = getattr(credit, "trace_id", None)
        if trace_id is None:
            return
        adapter = self._adapters.get(trace_id)
        if adapter is None:
            node_ordinal = getattr(credit, "node_ordinal", None)
            # Routine at the extended-warmup drain (executors cancelled, wire
            # credits still landing -- already ledgered above); exceptional
            # everywhere else.
            log = _logger.debug if self._pressure_active else _logger.warning
            log(
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
        adapter.resolve(credit, error, cancelled)

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
        trace_id = getattr(first_token, "trace_id", None)
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
            getattr(first_token, "x_correlation_id", None),
            getattr(first_token, "turn_index", None),
        )

    async def execute_phase(self) -> None:
        """Dispatch every admitted trace, owning completion AgentX-faithfully.

        Stop semantics mirror AgentX's stop-condition model
        (``aiperf.timing.phase.stop_conditions``), NOT a per-trace drain wait:
        ``--num-conversations N`` (``SessionCountStopCondition``) caps admitted
        traces (each weka trace is one conversation/DAG); ``--request-count N``
        (``RequestCountStopCondition``) caps node dispatches via
        ``issue_graph_credit``'s gate (a refused issue is a clean per-trace stop);
        ``--benchmark-duration D`` (``DurationStopCondition``) caps WALL time via
        :meth:`_run_traces_under_duration_budget`. A weka trace faithfully
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
        traces = list(self._parsed.traces)
        if not traces:
            self.debug("no traces to replay; phase complete")
            self._credit_issuer.mark_graph_sending_complete()
            self._credit_issuer.set_graph_all_returned_event()
            return

        self._advise_if_idle_gap_corpus_without_duration()
        try:
            await self._run_traces_under_duration_budget(traces)
            if self._pressure_enabled:
                await self._run_pressure_stage(traces)
        finally:
            # ALWAYS runs (drain / duration-cancel / error): no further graph
            # credit will be issued, so freeze the authoritative per-node sent
            # count and signal sending-complete. If no credit was ever issued
            # (no-op corpus, or every trace cancelled pre-dispatch), also set the
            # returned event since no return will fire it.
            self._credit_issuer.mark_graph_sending_complete()
            if self._credit_issuer.graph_all_returned():
                self._credit_issuer.set_graph_all_returned_event()

    def _has_benchmark_duration(self) -> bool:
        """True iff a ``--benchmark-duration`` wall budget bounds this phase.

        The budget lives on the lifecycle (``time_left_in_seconds()`` is non-None
        only when ``expected_duration_sec`` is configured) or, equivalently, on the
        phase config (``expected_duration_sec``). Either source is authoritative;
        we check both so the advisory fires correctly under the unit harness (which
        may wire only one).
        """
        if (
            self._lifecycle is not None
            and self._lifecycle.time_left_in_seconds() is not None
        ):
            return True
        return bool(getattr(self._config, "expected_duration_sec", None))

    def _max_inter_turn_gap_seconds(self) -> float:
        """Largest recorded inter-turn idle gap (seconds) across every graph.

        Scans every node's ``min_start_delay_us`` and every ``StaticEdge``'s
        ``delay_after_predecessor_us`` (and the start-anchored
        ``delay_after_predecessor_start_us``) over ``ParsedGraph.graph`` and every
        entry in ``ParsedGraph.graphs`` (the multi-graph corpus). These are the
        per-gap-warped delays the executor replays VERBATIM -- both end-to-start
        (``delay_after_predecessor_us``) and dispatch-to-start
        (``delay_after_predecessor_start_us``); the max is the worst single
        stretch a faithful replay parks on. Returns ``0.0`` for
        a gap-free corpus (the AgentX bare-CLI default, end-to-start delays OFF).
        """
        max_us = 0.0
        graphs = [self._parsed.graph]
        graphs.extend(getattr(self._parsed, "graphs", {}).values())
        for graph in graphs:
            if graph is None:
                continue
            for node in getattr(graph, "nodes", {}).values():
                delay = getattr(node, "min_start_delay_us", None)
                if delay is not None:
                    max_us = max(max_us, float(delay))
            for edge in getattr(graph, "edges", []):
                delay = getattr(edge, "delay_after_predecessor_us", None)
                if delay is not None:
                    max_us = max(max_us, float(delay))
                start_delay = getattr(edge, "delay_after_predecessor_start_us", None)
                if start_delay is not None:
                    max_us = max(max_us, float(start_delay))
                # Kind-complete scan: the first-token refinement is <= the start
                # delay on the same edge, so this never raises the max in practice.
                first_token_delay = getattr(
                    edge, "delay_after_predecessor_first_token_us", None
                )
                if first_token_delay is not None:
                    max_us = max(max_us, float(first_token_delay))
        return max_us / MICROS_PER_SECOND

    def _advise_if_idle_gap_corpus_without_duration(self) -> None:
        """Emit a once-per-phase advisory for an idle-gap corpus with no duration.

        A faithful recorded-trace replay honors every recorded inter-turn idle gap verbatim,
        so a count/session/bare run (no ``--benchmark-duration``) spans the slowest
        admitted trace's full recorded wall time -- exactly AgentX's count-mode
        (``is_final_credit`` fires only once the last turn is SENT, after every
        scheduled idle gap elapses; ``runner._wait_for_sending_complete`` then waits
        on that with a ``None`` timeout). This is faithful, NOT a hang, but on a
        human-pace corpus the wall time is minutes-scale with no console output
        during the parked gaps. Advise the operator that ``--benchmark-duration``
        bounds it (cancels the still-parked idle nodes, keeps dispatched records)
        and that Ctrl+C exports the partial results gracefully. No-op when a
        duration is set, when the corpus is gap-free, or when its largest gap is
        below ``IDLE_GAP_NO_DURATION_WARN_SECONDS``. Also a no-op for a
        WARMUP phase: warmup bursts only the boundary priming turns (see
        :func:`rewrite_for_warmup`) and never replays recorded gaps.
        """
        if self._phase_variant() == "warmup":
            return
        if self._has_benchmark_duration():
            return
        max_gap_s = self._max_inter_turn_gap_seconds()
        threshold_s = Environment.GRAPH.IDLE_GAP_NO_DURATION_WARN_SECONDS
        if max_gap_s < threshold_s:
            return
        self.notice(
            lambda: (
                f"idle-gap corpus has a recorded inter-turn gap of up to "
                f"{max_gap_s:.0f}s and no --benchmark-duration is set: this phase "
                "replays every gap faithfully (AgentX count-mode parity), so its "
                "wall time spans the slowest admitted trace's full recorded "
                "duration with no console output during the parked gaps. Pass "
                "--benchmark-duration <seconds> to bound the run (it cancels the "
                "still-parked idle nodes and keeps the records dispatched so far), "
                "or press Ctrl+C to finalize + export the partial results now."
            )
        )

    async def _run_traces_under_duration_budget(self, traces: list[Any]) -> None:
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
            self._duration_deadline = asyncio.get_running_loop().time() + timeout
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
        return deadline is not None and (asyncio.get_running_loop().time() >= deadline)

    async def _run_lanes(self, traces: list[Any]) -> None:
        """Drive ``concurrency`` recycling lanes over the wrapped corpus.

        AgentX (``trajectory_source.py:212`` ``_target_size = concurrency``;
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
        ``--graph weka.json`` run covers the whole corpus and still terminates.

        Each lane task runs on ``phase_tg``; each lane swallows its own instance
        errors so one failed instance never aborts the phase.
        """
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
        if self._warmup_handoff is not None:
            # AgentX dispatches EVERY drained lane at the handoff; the session
            # cap gates only recycles (resumed streams are continuations, not
            # new sessions). Lanes beyond len(pass0_traces) get an index-safe
            # fallback template below -- their real template comes from the
            # handoff entry or the fresh-start cursor draw anyway. Applied AFTER
            # the re-shrink so an unspawnable-shrunk resolution can't undo it.
            lanes = max(lanes, self._warmup_handoff.pressure_lane_count)
        # Recycle draws continue from the corpus position AFTER the last one the
        # pass-0 spawnable resolution consumed (AgentX's recycle reuses the SAME
        # sampler, so it resumes where ``_build_trajectories`` left off -- past the
        # skipped-unspawnable traces too), turn-0 full replay onto the freed slot.
        # AgentX shared-sampler parity: bounded profiling recycles continue from
        # the pressure stage's last draw so freed lanes don't re-serve templates
        # pressure just replayed. Single-pass mode (no stop conditions) keeps its
        # own cursor -- its termination check (next_index >= len(traces)) encodes
        # cover-the-corpus-once, which a carried cursor would break.
        if self._warmup_handoff is not None and recycle_is_bounded:
            next_index = self._warmup_handoff.corpus_cursor
        else:
            next_index = corpus_cursor
        traces_by_id = {t.id: t for t in traces}

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
                    # immediately. Applies to handoff-resume lanes too (the
                    # ramp's purpose is spreading load onto a cold server;
                    # duration-cancel cancels parked waiters cleanly).
                    await self._wait_for_lane_admission(lane)
                    # Index-safe pass-0 template: a handoff can bump ``lanes``
                    # past ``len(pass0_traces)`` (every drained pressure lane is
                    # honored), so lanes beyond the spawnable-resolved set take a
                    # wrap-around fallback here and get their real template from
                    # the handoff entry or the fresh-start cursor draw below.
                    trace: Any = (
                        pass0_traces[lane]
                        if lane < len(pass0_traces)
                        else traces[self._draw_index(lane, len(traces))]
                    )
                    recycle_pass = 0
                    entry = self._handoff_for_lane(lane)
                    fresh_start = False
                    if entry is not None:
                        resumed = traces_by_id.get(entry.template_trace_id)
                        if resumed is not None:
                            trace = resumed
                        else:
                            self.warning(
                                f"handoff template {entry.template_trace_id!r} "
                                f"not in corpus; lane {lane} resumes its "
                                "normal pass-0 assignment"
                            )
                    elif (
                        self._warmup_handoff is not None
                        and recycle_is_bounded
                        and lane < self._warmup_handoff.pressure_lane_count
                    ):
                        # This pressure lane completed at drain: fresh-start it
                        # from the shared cursor (agentx empty-lane parity)
                        # instead of re-running a t* resume the pressure stage
                        # already executed against the server (which would be
                        # measured warm). Single-pass mode is excluded: a fresh
                        # draw would consume a corpus position and hole the
                        # cover-the-corpus-once contract.
                        template_index = next_index
                        next_index += 1
                        trace = traces[self._draw_index(template_index, len(traces))]
                        fresh_start = True
                    while True:
                        self._instances_started += 1
                        await self._run_instance(
                            trace, lane, recycle_pass, fresh_start=fresh_start
                        )
                        fresh_start = False
                        if self._past_duration_deadline():
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
                        trace = traces[self._draw_index(template_index, len(traces))]
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
        if distinct_loaded_traces <= 0 or self._phase_variant() == "warmup":
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

    async def _run_pressure_stage(self, traces: list[Any]) -> None:
        """Run the extended (cache-pressure) warmup stage for the configured duration.

        AgentX v1.0 ``_start_accelerated_warmup`` parity, dataflow-native: the
        post-t* remainder of every lane's template replays compressed (the
        WARMUP executors are built with ``compress_edge_delays=True`` via
        ``accelerated_warmup``; the worker's 1-token warmup cap keys off the
        unchanged ``"warmup"`` phase variant), and freed lanes recycle fresh
        templates at t*=0. The stage ends on the duration timer: in-flight
        executors are cancelled exactly like the ``--benchmark-duration`` stop
        (issued credits still return and land in the ledger; un-fired nodes
        drop), which is the drain the teardown handoff is built from.

        The pressure warmup phase is mode-owned: user warmup phases are
        superseded at config build (timing/config.py), so the stage always gets
        its full duration; the drain is bounded by min(duration,
        ``PRESSURE_DRAIN_GRACE_CAP``) with the stash completeness gate as
        the safety net.
        """
        assert self._cache_pressure_duration_s is not None
        duration = self._cache_pressure_duration_s
        self._pressure_active = True
        self.notice(
            f"WARMUP cache pressure: replaying post-t* graphs with zero idle "
            f"delay and max_tokens="
            f"{Environment.GRAPH.WARMUP_MAX_OUTPUT_TOKENS} for {duration:.1f}s"
        )
        self._duration_deadline = asyncio.get_running_loop().time() + duration
        dispatch = asyncio.ensure_future(self._run_pressure_lanes(traces))
        try:
            await asyncio.wait_for(dispatch, timeout=duration)
        except TimeoutError:
            self.notice(
                "WARMUP cache pressure duration reached; draining in-flight "
                "requests for the profiling handoff"
            )
            try:
                await dispatch
            except asyncio.CancelledError:
                # Absorb wait_for's own cancellation (healthy stage end) but
                # re-raise when THIS task is being cancelled externally
                # (runner send-timeout, Ctrl+C) -- same guard as
                # _run_traces_under_duration_budget.
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
        finally:
            self._duration_deadline = None

    async def _run_pressure_lanes(self, traces: list[Any]) -> None:
        """Drive the pressure lanes: same pass-0 assignment as priming, then recycle.

        Reuses ``_resolve_pass0_lanes`` (with the SAME bounded-recycle shape as
        the Stage-A priming fan-out) so lane i continues the SAME (template,
        lane-salted t*) chain its boundary priming just warmed -- a lane count
        wider than priming would run unprimed pass-0 chops and produce handoff
        entries with no priming walls to merge. Lanes recycle until the stage
        ends (fresh template at t*=0 per freed slot), matching agentx's
        recycle-forever pressure loop (``_handle_accelerated_warmup_return``
        -> ``_spawn_from_recycle_or_id``), with two lane-local gates so a
        closed issuer stop gate or a deterministically failing server cannot
        hot-spin adapter builds for the whole duration: a clean issuer refusal
        ends the lane, and consecutive instance errors back off exponentially
        (the sleep also yields the loop under the ``wait_for`` budget).
        """
        lane_count = self._resolve_lane_count(
            len(traces), self._recycle_has_stop_condition()
        )
        pass0_traces, corpus_cursor = self._resolve_pass0_lanes(traces, lane_count)
        self._pressure_next_index = corpus_cursor
        self._pressure_lane_count = len(pass0_traces)

        async with asyncio.TaskGroup() as pressure_tg:
            for lane in range(len(pass0_traces)):

                async def _lane(
                    lane: int = lane, start_trace: Any = pass0_traces[lane]
                ) -> None:
                    trace = start_trace
                    pressure_pass = 0
                    consecutive_errors = 0
                    while True:
                        plan = (
                            self._plan_for_lane(trace, lane)
                            if pressure_pass == 0
                            else None
                        )
                        # {template}::{nonce}: instance identity is template +
                        # fresh nonce (lane/pass are logged, not encoded).
                        instance_id = f"{trace.id}::{uuid.uuid4().hex}"
                        self.debug(
                            lambda t=trace.id,
                            i=instance_id,
                            ln=lane,
                            p=pressure_pass: (
                                f"pressure instance {i} (template={t} lane={ln} pass=p{p})"
                            )
                        )
                        # Set-before-await / pop-after: entry present == the
                        # instance is mid-flight, so a duration-cancel landing
                        # inside _run_instance leaves it for the handoff.
                        # Benign race: a cancel delivered exactly at the await
                        # RESUMPTION of a completed instance stashes it as
                        # live with all nodes executed -- the profiling chop
                        # is then empty and the lane recycles (harmless).
                        self._pressure_live[lane] = (
                            trace.id,
                            instance_id,
                            plan.t_star_us if plan is not None else 0.0,
                            pressure_pass,
                        )
                        errors_before = self._errored_traces
                        refused = await self._run_instance(
                            trace,
                            lane,
                            pressure_pass,
                            pressure=True,
                            instance_id=instance_id,
                        )
                        self._pressure_live.pop(lane, None)
                        if refused:
                            # Issuer stop gate closed (run cancelled): a fresh
                            # instance would refuse instantly, forever.
                            return
                        if self._past_duration_deadline():
                            return
                        if self._errored_traces > errors_before:
                            consecutive_errors += 1
                            await asyncio.sleep(min(0.25 * 2**consecutive_errors, 5.0))
                        else:
                            consecutive_errors = 0
                        template_index = self._pressure_next_index
                        self._pressure_next_index += 1
                        trace = traces[self._draw_index(template_index, len(traces))]
                        pressure_pass += 1

                pressure_tg.create_task(_lane(), name=f"graph-pressure-lane:{lane}")

    def _resolve_pass0_lanes(
        self, traces: list[Any], lanes: int
    ) -> tuple[list[Any], int]:
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

        The per-candidate corpus draw routes through :meth:`_draw_index`, so
        ``--dataset-sampling-strategy`` selects WHICH template each rank serves;
        under the default ``sequential`` draw it is ``traces[cursor % n]``
        unchanged. With the default ``[0, 0]`` window (t*=0) every trace is
        spawnable (full replay always dispatches), so ``sequential`` collapses to
        the prior raw-position assignment (lane ``i`` -> ``traces[i]``)
        byte-for-byte.
        """
        pass0: list[Any] = []
        cursor = 0
        n = len(traces)
        # Bound the skip walk to one full corpus pass past the wrap of ``lanes``
        # so an all-unspawnable corpus can't spin (AgentX caps at
        # ``_target_size + 2 * pool_size``); we stop at ``lanes`` hits regardless.
        max_cursor = lanes + n
        while len(pass0) < lanes and cursor < max_cursor:
            trace = traces[self._draw_index(cursor, n)]
            cursor += 1
            rank = len(pass0)
            if self._is_spawnable(trace, rank):
                pass0.append(trace)
        return pass0, cursor

    def _is_spawnable(self, trace: Any, lane_index: int) -> bool:
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

        The lane-salted t* is computed directly (``_lane_salted_t_star``) rather
        than through ``_plan_for_lane`` so the spawnability scan never builds the
        per-lane catalog / node partition (only the t* + one snapshot are needed);
        the value is byte-identical to the plan's ``t_star_us`` because both call
        the same ``_sample_t_star`` math, so spawnability and the eventual
        dispatch plan agree.
        """
        t_star_us = self._lane_salted_t_star(trace, lane_index)
        if t_star_us <= 0:
            return True
        from aiperf.dataset.graph.models import NodeKind
        from aiperf.graph.analysis import compute_snapshot

        # ``compute_snapshot`` resolves the trace's own graph internally
        # (``elaborate_trace`` -> ``resolve_trace_graph``), so passing the raw
        # ``self._parsed`` is correct here.
        snapshot = compute_snapshot(self._parsed, trace, t_star_us=int(t_star_us))
        return any(sf.firing.kind == NodeKind.LLM for sf in snapshot.profiled)

    def _lane_salted_t_star(self, trace: Any, lane_index: int) -> float:
        """Compute ``trace``'s lane-salted t* (us) WITHOUT building a t* source.

        Reuses the prebuilt lane-0 plan when ``lane_index == 0`` (the common
        single-pass case). For higher lanes it inlines ``_sample_t_star``'s seed +
        duration math (``_seed_for_trace_lane`` -> ``rng.uniform(lo, hi)`` over the
        ratio window) so the spawnability scan never constructs a
        ``GraphIRConversationSource`` (whose ``__init__`` rebuilds the whole-corpus
        node catalog -- an O(lanes * traces) cost across a fan-out the scan does
        not need). The value is byte-identical to ``_plan_for_lane``'s
        ``t_star_us`` because it is the SAME computation on the SAME seed.
        """
        if lane_index == 0:
            plan = self._plans.get(trace.id)
            return plan.t_star_us if plan is not None else 0.0
        import numpy as np

        from aiperf.graph.analysis import trace_duration_us
        from aiperf.timing.graph_ir_source import _seed_for_trace_lane

        # ``trace_duration_us`` resolves the trace's OWN graph internally
        # (``elaborate_trace`` -> ``resolve_trace_graph``), so passing the full
        # multi-graph parse matches ``GraphIRConversationSource._sample_t_star``
        # byte-for-byte (it too uses the source's raw ``self._parsed``).
        duration_us = trace_duration_us(self._parsed, trace)
        if duration_us <= 0:
            return 0.0
        lo = self._start_min_ratio * duration_us
        hi = self._start_max_ratio * duration_us
        if hi <= lo:
            return float(lo)
        rng = np.random.default_rng(
            _seed_for_trace_lane(self._t_star_random_seed, trace.id, lane_index)
        )
        return float(rng.uniform(lo, hi))

    def _recycle_has_stop_condition(self) -> bool:
        """True iff an EXPLICIT stop condition exists that recycle can saturate toward.

        Recycle runs ONLY when the user set ``--num-conversations``
        (``expected_num_sessions``) / ``--request-count``
        (``total_expected_requests``) / ``--benchmark-duration``. With none, the
        corpus is replayed once and the lanes stop (a bare ``--graph weka.json``
        must NOT recycle forever).

        The DERIVED single-pass session target
        (:meth:`_resolved_num_sessions` = ``len(traces)`` for a bare run) is
        DELIBERATELY EXCLUDED here: it is a lane-clamp + reported-target only, not
        a recycle-enabling stop condition. Routing it in would flip a bare run out
        of SINGLE-PASS mode into the bounded/recycle path -- firing the
        pressure-lane fresh-start gate and lane-salted recycle plans (``.f``/``.1``
        ids, divergent t* instants) instead of the clean pass-0 plans a
        cover-the-corpus-once pass must keep. So a bare run stays single-pass:
        each lane does one pass, no recycle, no fresh-start.
        """
        cfg = self._config
        return bool(
            getattr(cfg, "expected_num_sessions", None)
            or getattr(cfg, "total_expected_requests", None)
            or getattr(cfg, "expected_duration_sec", None)
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
        sessions = getattr(self._config, "expected_num_sessions", None)
        if sessions is not None and sessions > 0 and self._admitted_traces >= sessions:
            return False
        checker = self._stop_checker
        if checker is None:
            return True
        can_send = getattr(checker, "can_send_dag_child_turn", None)
        if callable(can_send):
            return bool(can_send())
        return True

    async def _run_instance(
        self,
        trace: Any,
        lane_index: int,
        recycle_pass: int,
        *,
        pressure: bool = False,
        fresh_start: bool = False,
        instance_id: str | None = None,
    ) -> bool:
        """Run ONE trace instance on ``lane_index`` (recycle pass ``recycle_pass``).

        Returns True iff the instance stopped cleanly at the issuer's stop gate --
        the pressure lane loop's stop signal.

        ``pressure=True`` marks an extended-warmup instance: ``.p{pass}`` id and
        the profiling-style graph (see ``_graph_at_t_star``).

        ``fresh_start=True`` marks a profiling lane that WAS a pressure lane but
        drained empty: it suppresses the pass-0 t* plan (a full identity t*=0
        replay of the cursor-drawn template, NOT a resume of a t* the pressure
        stage already executed against the server) and mints a ``.f{pass}`` id.
        The dedicated ``.f`` namespace exists because a plain ``.0`` id can
        COLLIDE with the same lane's warmup boundary-priming id when the cursor
        wraps a small corpus back onto the lane's own template; the cache-bust
        marker digests the full instance id, so a collision would silently warm
        a lane that exists to measure cold (agentx mints a fresh uuid + marker
        for its empty-lane fresh conversations). The worker strips everything
        after the first ``#``, so catalog / mmap keying is unaffected -- the same
        argument as ``.p{k}``.

        The instance id ``f"{trace.id}#{lane_index}.{recycle_pass}"`` is stamped
        on every credit's ``trace_id`` so the cache-bust marker ROTATES per recycle
        AND decorrelates concurrent lanes that wrapped onto the SAME template (C3 +
        AgentX's per-lane ``trajectory_index`` in the digest), while the build-time
        catalog + graph store mmap stay keyed by the base template id (the
        worker strips everything after the first ``#``). The id is GLOBALLY UNIQUE
        per (lane, pass), so two lanes wrapping the same template never collide on
        the return-routing registry. Pass 0 uses the lane-salted t* plan (C2);
        recycle passes (>0) run the full t*=0 replay -- AgentX recycle restarts a
        session from turn 0 with no snapshot/t* slice
        (the retired agentic-replay plane's recycled-lane dispatch).

        A per-instance adapter is registered under the instance id for return
        routing, bound to a ``TraceExecutor``, and awaited. An instance error is
        counted (``errored_traces``) and contained -- it does not propagate out of
        the lane's ``TaskGroup`` (which would cancel sibling lanes).
        """
        refused = False
        handoff_entry = (
            self._handoff_for_lane(lane_index)
            if recycle_pass == 0 and not pressure
            else None
        )
        if handoff_entry is not None and trace.id != handoff_entry.template_trace_id:
            # _run_lanes declined the template swap (handoff template not in
            # this corpus): fall back to the normal pass-0 path COHERENTLY --
            # applying another template's executed set here would chop the
            # wrong graph.
            handoff_entry = None
        if handoff_entry is not None:
            # Marker continuity: reuse the live pressure instance's id so the
            # per-instance cache-bust marker (digest of credit.trace_id) is
            # unchanged across the handoff and the KV the pressure stage built
            # at this id transfers. The pressure instance's adapter was reaped at
            # warmup teardown, so re-registering the id in this profiling
            # strategy's fresh registry cannot collide; profiling instances mint
            # fresh nonces, so no collision there either.
            instance_id = handoff_entry.instance_id
        elif instance_id is None:
            # {template}::{nonce}: instance identity is template + fresh nonce
            # (uuid4 -- collision-proof across concurrent lanes, recycles,
            # phases, and runs). Lane / pass / flavor are diagnostics, logged
            # below instead of encoded in the id. Callers that pre-mint (the
            # pressure lane loop stashes the id in _pressure_live BEFORE the
            # await) pass theirs in so credits and bookkeeping share ONE id.
            instance_id = f"{trace.id}::{uuid.uuid4().hex}"
            flavor = "p" if pressure else ("f" if fresh_start else "")
            self.debug(
                lambda t=trace.id,
                i=instance_id,
                ln=lane_index,
                k=recycle_pass,
                f=flavor: (f"instance {i} (template={t} lane={ln} pass={f}{k})")
            )
        plan = (
            self._plan_for_lane(trace, lane_index)
            if recycle_pass == 0 and handoff_entry is None and not fresh_start
            else None
        )
        adapter = self._build_adapter(
            trace.id,
            instance_id,
            first_token_sources=self._first_token_sources_for(trace),
            node_identity=self._node_identity_for(trace),
        )
        self._adapters[instance_id] = adapter
        self._admitted_traces += 1
        if not pressure and not fresh_start and recycle_pass == 0:
            # Boundary-priming instance for (template, lane): the pressure
            # handoff's pass-0 merge needs this id (no longer reconstructable).
            self._priming_instance_ids[(trace.id, lane_index)] = instance_id
        parsed: ParsedGraph | None = None
        try:
            if handoff_entry is not None:
                parsed, run_trace = self._graph_at_handoff(trace, handoff_entry)
            else:
                parsed, run_trace = self._graph_at_t_star(
                    trace, plan, pressure=pressure
                )
            executor = TraceExecutor(
                parsed,
                credit_issuer=adapter,
                compress_edge_delays=self._accelerated_warmup,
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
            await executor.run(run_trace)
        except Exception as exc:
            refusal = _leaf_credit_refusal(exc)
            if refusal is not None:
                # Issuer refusal is a HEALTHY stop (request-count / duration cap
                # reached, or run cancelled), not a trace error: log quietly and
                # keep ``errored_traces`` untouched. Signal the pressure lane loop
                # to end this lane -- a fresh instance would refuse identically.
                refused = True
                self.debug(
                    lambda exc=refusal, iid=instance_id: (
                        f"graph instance {iid!r} "
                        f"stopped cleanly at the issuer's stop gate: {exc!r}"
                    )
                )
            else:
                self._errored_traces += 1
                self.warning(
                    lambda exc=exc, iid=instance_id: (
                        f"graph instance {iid!r} unwound with error: {exc!r}"
                    )
                )
        finally:
            self._completed_traces += 1
            async with self._registry_lock:
                self._parent_done.add(instance_id)
                self._release_adapter_if_idle(instance_id)
        return refused

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

    @staticmethod
    def _is_trie_graph(parsed: ParsedGraph) -> bool:
        """True iff ``parsed`` is a trie graph (the flat LlmNode IR the weka and
        dynamo adapters emit).

        The trie builder stamps ``metadata["trie"]`` on every emitted ``LlmNode``.
        Detecting the trie marker on any top-level node confirms the trie path; a
        non-trie parse reaching ``t*>0`` is a lowering bug (raises in
        ``_graph_at_t_star``).
        """
        for node in parsed.graph.nodes.values():
            if "trie" in getattr(node, "metadata", {}):
                return True
        return False

    def _handoff_for_lane(self, lane_index: int) -> LaneHandoff | None:
        """This profiling lane's extended-warmup resume entry, if any.

        Only pass-0 profiling instances consume the handoff (the gate lives at
        the call sites); recycle passes always run fresh t*=0 templates,
        matching agentx's post-handoff recycle draws. ``self._warmup_handoff``
        is only ever non-None on a PROFILING strategy (the runner pops it there),
        so this returns None for every warmup/pressure phase with no extra check.
        """
        if self._warmup_handoff is None:
            return None
        return self._warmup_handoff.lanes.get(lane_index)

    def _graph_at_handoff(
        self, trace: TraceRecord, entry: LaneHandoff
    ) -> tuple[ParsedGraph, TraceRecord]:
        """Reconstruct the profiling resume graph at the warmup drain frontier.

        The frontier chop drops pre-t* history AND every node the pressure
        stage executed, re-rooting each chain's first not-yet-executed node
        with its residual delay -- see ``chop_trie_at_frontier``. The trie
        envelope keeps each node's FULL prompt prefix, so the worker
        materializes the exact resume prompt with no back-seeding.

        A fully-executed handoff template yields an EMPTY chop: the executor
        finalizes instantly and the lane proceeds to its normal recycle draw --
        no special case needed.
        """
        parsed = parsed_for_trace(self._parsed, trace)
        if (entry.executed_node_ids or entry.t_star_us > 0) and not self._is_trie_graph(
            parsed
        ):
            raise RuntimeError(
                f"trace {trace.id!r}: the extended-warmup handoff requires a "
                "trie-stamped graph (metadata['trie'] on every LlmNode); a "
                "non-trie parse reaching the handoff is a lowering bug"
            )
        assert self._warmup_handoff is not None
        cap_s = Environment.GRAPH.HANDOFF_RESIDUAL_CAP
        rewritten = chop_trie_at_frontier(
            parsed,
            t_star_us=entry.t_star_us,
            executed=entry.executed_node_ids,
            return_wall_us=entry.return_wall_us,
            drain_end_wall_us=self._warmup_handoff.drain_end_wall_us,
            residual_cap_us=cap_s * MICROS_PER_SECOND if cap_s is not None else None,
        )
        # --burst-phase-starts collapses each resumed lane's leading offset
        # to 0 at the profiling start, residuals included.
        if self._burst_phase_starts:
            rewritten = self._burst_collapse_leading_offsets(rewritten)
        return rewritten, trace

    def _graph_at_t_star(
        self, trace: TraceRecord, plan: GraphTrace | None, *, pressure: bool = False
    ) -> tuple[ParsedGraph, TraceRecord]:
        """Reconstruct the per-trace graph + trace at this instance's t* disposition.

        ``plan`` is the lane-resolved t* plan for this instance (lane-salted on
        pass 0, ``None`` == full t*=0 replay for recycle passes). Returns
        ``(parsed_to_run, trace_to_run)`` for the ``TraceExecutor``.

        PROFILING: ``t*==0`` (default full-replay window, or any recycle pass)
        => IDENTITY (byte-identical to the original). ``t*>0`` =>
        ``chop_trie_at_tstar`` (a frontier chop re-rooting each live chain at
        the ``t*`` frontier) for a trie graph. Surviving nodes keep their ids so
        the adapter resolves the unmodified catalog ordinal.

        WARMUP: :func:`rewrite_for_warmup` -- the flat boundary-priming graph
        (one boundary turn per chain live at t*, START-rooted, zero leading
        offsets). ``t*<=0`` yields an EMPTY warmup graph so the instance
        finalizes immediately (the ``timing.config`` auto-warmup contract).

        A non-trie graph at ``t*>0`` is a lowering bug (raises). Multi-graph
        workloads project onto each trace's OWN graph via ``parsed_for_trace``
        first (else a non-first trace runs the first file's topology).

        PRESSURE (extended warmup): ``pressure=True`` routes a WARMUP-phase
        instance through the PROFILING branches (t*>0 chop / t*=0 identity) so
        the post-t* remainder replays compressed under the warmup token cap --
        never ``rewrite_for_warmup``.
        """
        parsed = parsed_for_trace(self._parsed, trace)
        t_star_us = plan.t_star_us if plan is not None else 0
        is_warmup = self._phase_variant() == "warmup" and not pressure
        if t_star_us <= 0:
            if is_warmup:
                return rewrite_for_warmup(parsed, 0), trace
            return parsed, trace

        # Trie graphs (the flat LlmNode + StaticEdge recorded-trace IR) snapshot via a
        # simple frontier chop -- there are no reducers / spawn / await / subgraph
        # primitives to re-root, and each node's full pre-t* prompt prefix is
        # preserved (the worker materializes the exact resume prompt). The
        # executor anchors the re-rooted frontier offsets via the
        # ``absolute_start_offsets=True`` already set on every instance.
        if not self._is_trie_graph(parsed):
            raise RuntimeError(
                f"trace {trace.id!r}: t*={t_star_us} requires a trie-stamped graph "
                "(metadata['trie'] on every LlmNode); every live producer lowers "
                "onto the trie, so a non-trie parse reaching t*>0 is a lowering bug"
            )
        if is_warmup:
            return rewrite_for_warmup(parsed, t_star_us), trace
        rewritten = chop_trie_at_tstar(parsed, t_star_us)
        if self._burst_phase_starts:
            rewritten = self._burst_collapse_leading_offsets(rewritten)
        return rewritten, trace

    def _burst_collapse_leading_offsets(self, rewritten: ParsedGraph) -> ParsedGraph:
        """Collapse leading phase-start offsets (AgentX ``--burst-phase-starts``).

        AgentX burst collapses the phase START into a synchronized burst: every
        trace's earliest profiling resume fires at once, IGNORING the per-stream
        leading offset from t*. On a chopped trie graph that leading offset
        lives on each re-rooted node's START in-edge ``min_start_delay_us``
        (stamped by ``snapshot_chop._chop_edges``; the node-level field is never
        stamped by the trie producers but is collapsed too for hand-authored
        graphs) -- delegate to the pure
        :func:`aiperf.graph.scheduler.collapse_leading_start_offsets`. The
        inter-turn ``StaticEdge.delay_after_predecessor_us`` end-to-start gaps
        are UNTOUCHED -- burst governs only the phase start, not the faithful
        inter-turn pacing. Warmup builds its boundary graph offset-free
        (:func:`rewrite_for_warmup`), keeping spread/burst warmup identical.
        """
        return msgspec.structs.replace(
            rewritten, graph=collapse_leading_start_offsets(rewritten.graph)
        )

    def _first_token_sources_for(self, trace: Any) -> frozenset[str]:
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
        self, trace: Any
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
            catalog_context=self._catalog,
            trace_id=trace_id,
            instance_id=instance_id,
            phase_variant=self._phase_variant(),
            dispatch_timeout_s=self._dispatch_timeout_s,
            on_drained=self._on_adapter_drained,
            first_token_sources=first_token_sources,
            node_identity=node_identity,
        )

    def _phase_variant(self) -> str:
        """Graph phase-variant label for this phase.

        ``"warmup"`` for a WARMUP phase, ``"profiling"`` otherwise. Falls back to
        ``"profiling"`` when no config / phase is wired (e.g. unit harness).
        """
        from aiperf.common.enums import CreditPhase

        phase = getattr(self._config, "phase", None)
        if phase == CreditPhase.WARMUP:
            return "warmup"
        return "profiling"

    async def handle_credit_return(self, credit: Credit) -> None:
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
        end = getattr(self._credit_issuer, "end_graph_trace", None)
        if end is None:
            return

        task = asyncio.create_task(end(adapter.instance_id, adapter.phase_variant))
        self._trace_end_tasks.add(task)
        task.add_done_callback(self._trace_end_tasks.discard)

    def _detach_observer(
        self,
        unregister: Callable[[Any], None] | None,
        register: Callable[[Any], None] | None,
        own_observer: Callable[..., None],
        label: str,
    ) -> None:
        """Best-effort detach of one shared observer slot at teardown.

        Prefers the compare-and-clear channel (``unregister(own_observer)``:
        the handler clears the slot only if OUR observer is still installed --
        see ``CreditCallbackHandler.clear_graph_return_observer``). Falls back
        to the unconditional ``register(None)`` when no unregister
        channel was wired (unit harness). Exceptions are swallowed at debug:
        teardown must never mask the phase's own exit path.
        """
        try:
            if unregister is not None:
                unregister(own_observer)
            elif register is not None:
                register(None)
        except Exception as exc:
            self.debug(lambda exc=exc: f"{label} observer detach failed: {exc!r}")

    def _stash_pressure_handoff(self) -> None:
        """Convert the drained pressure state into the profiling handoff.

        Runs at teardown -- for a non-seamless WARMUP phase that is AFTER every
        issued credit returned (``PhaseRunner._run_strategy`` awaits
        returning-complete before ``run()``'s finally), so the ledger is
        complete and ``drain_end`` stamped here credits the whole drain wait
        toward each recorded gap (agentx ``finalize_phase``'s
        ``finalized_at_ns`` parity). Pass-0 pressure instances merge their
        lane's Stage-A boundary-priming walls so chains the pressure never
        advanced still anchor their residual on the priming return.

        Completeness gate: two runner paths reach teardown BEFORE every return
        landed -- an external cancel (``PhaseRunner._run_strategy``'s cancelled
        early-return, right after sending-complete) and a USER warmup phase with
        a finite
        ``--warmup-grace-period`` (grace-timeout force-complete). A handoff
        built from an incomplete ledger would mark server-executed nodes as
        not-executed and profiling would REFIRE them, so skip the stash
        entirely and let profiling start from the plain t* plans.
        """
        channel = self._graph_channel
        if channel is None:
            self.debug("pressure handoff skipped: no graph channel wired")
            return
        all_returned = getattr(self._credit_issuer, "graph_all_returned", None)
        if all_returned is not None and not all_returned():
            self.notice(
                "WARMUP cache pressure handoff skipped: not every warmup "
                "credit return landed (cancelled run or finite grace period); "
                "profiling will start from the plain t* plans"
            )
            return
        drain_end = self._wall_us()
        lanes: dict[int, LaneHandoff] = {}
        for lane, (
            template_id,
            instance_id,
            t_star_us,
            pressure_pass,
        ) in self._pressure_live.items():
            live_walls = self._return_walls.get(instance_id, {})
            anchor_walls = dict(live_walls)
            if pressure_pass == 0:
                # Merge the lane's boundary-PRIMING walls under the pressure
                # pass-0 anchor set. Ids carry nonces, so the priming id is
                # looked up from the record made at its _run_instance.
                priming_id = self._priming_instance_ids.get((template_id, lane))
                priming = self._return_walls.get(priming_id, {}) if priming_id else {}
                for node_id, wall in priming.items():
                    anchor_walls.setdefault(node_id, wall)
            lanes[lane] = LaneHandoff(
                template_trace_id=template_id,
                instance_id=instance_id,
                t_star_us=t_star_us,
                executed_node_ids=frozenset(live_walls),
                return_wall_us=anchor_walls,
            )
        channel.warmup_handoff = GraphWarmupHandoff(
            lanes=lanes,
            drain_end_wall_us=drain_end,
            corpus_cursor=self._pressure_next_index,
            pressure_lane_count=self._pressure_lane_count,
        )
        self.notice(
            f"WARMUP cache pressure handoff: persisted {len(lanes)} live "
            "lane(s) for the profiling resume"
        )

    async def teardown_phase(self) -> None:
        """Detach the graph-return observer after the phase finalizes.

        Invoked by ``PhaseRunner`` in its ``run()`` ``finally`` (see
        ``PhaseTeardownStrategyProtocol``); tests may also call it directly.
        Best-effort: a subsequent phase / cleanup must not dispatch into this
        torn-down strategy's adapter registry. Also reaps any de-mux entry retained
        for an instance not popped at its finally -- the phase is over, so no further
        return will arrive and the registry must not leak into the next phase --
        and closes the sticky lifecycle for every retained adapter.

        When the extended-warmup pressure stage ran, the FIRST action here (a
        sync prefix, before any awaited sticky close that a cancel could
        interrupt) is to build and stash the WARMUP -> PROFILING handoff from
        the drained pressure state.

        The SYNC parts (observer detach + registry clear) run FIRST so they
        complete even when the phase is being cancelled and the awaited sticky
        closes below get interrupted at their first suspension point.

        Detach is compare-and-clear, NOT unconditional: for seamless non-final
        phases this teardown is deferred to the background return-wait
        completion and can fire AFTER the next phase's ``setup_phase`` installed
        ITS observer on the same shared ``CreditCallbackHandler`` slot. Clearing
        unconditionally here would drop every subsequent graph return of the
        live phase, so the slot is cleared only if OUR observer is still the
        one installed (unit harnesses without the unregister channel keep the
        unconditional ``register(None)`` detach).
        """
        if self._pressure_active:
            self._stash_pressure_handoff()
        retained = list(self._adapters.values())
        self._adapters.clear()
        self._parent_done.clear()
        self._detach_observer(
            self._unregister_observer,
            self._register_observer,
            self._on_graph_return,
            "graph-return",
        )
        self._detach_observer(
            self._unregister_first_token_observer,
            self._register_first_token_observer,
            self._on_graph_first_token,
            "first-token",
        )
        end = getattr(self._credit_issuer, "end_graph_trace", None)
        if end is not None:
            for adapter in retained:
                try:
                    await end(adapter.instance_id, adapter.phase_variant)
                except Exception as exc:
                    self.debug(lambda exc=exc: f"teardown trace-end failed: {exc!r}")
