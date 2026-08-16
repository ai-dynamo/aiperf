# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CreditDispatchAdapter — the Future bridge from the dataflow executor to v1 credits.

The async-dataflow ``TraceExecutor`` calls exactly one method on its injected
``credit_issuer``::

    async def dispatch(self, node, request, ctx, **kwargs) -> Any

(see ``aiperf.graph.dispatch.llm``). The return value feeds ``node.output``
(the dispatch module substitutes a type-correct empty list for messages-typed
channels). No downstream node consumes an LLM output's value (prompts are
materialized worker-side from recorded envelopes; output channels gate fan-in
only), so a placeholder ``str`` is contractually correct.

This adapter turns that fire-and-forget executor call into an awaitable backed by
the v1 credit pipeline:

1. Resolve the fired node's ``node_ordinal`` from the build-time catalog.
2. Mint a collision-free correlation key (see ``_mint``).
3. Park an ``asyncio.Future`` under that key.
4. Issue a real ``TurnToSend`` (via ``CreditIssuer.issue_graph_credit``, which
   BYPASSES the linear session-slot lifecycle -- the strategy owns concurrency).
5. ``await`` the Future under a timeout guard; ``resolve`` (driven by the
   unconditional graph-return hook on ``CreditCallbackHandler``) sets the result
   on a normal return or rejects it on cancel / error.

Why the adapter owns its own context (NOT ``ctx``)
--------------------------------------------------
The executor passes a ``PlacementContext`` carrying ONLY ``parent_trace_id`` /
``parent_node_id`` -- a frozen ``slots=True`` dataclass.
It has no ``trace``, ``phase``, ``agent_depth``, or correlation fields,
and cannot be monkey-patched. The adapter is therefore constructed PER-RUN with
the per-trace identity it needs and derives node identity from
``request.node_id`` + ``ctx.parent_trace_id`` alone.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.scenario import is_context_overflow_response
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.dataset.graph.graph_path_catalog import (
    CatalogContext,
    node_ordinal_for,
)
from aiperf.graph.context import _NodeExpectedExit
from aiperf.graph.errors import GraphErrorCode, parse_graph_error
from aiperf.graph.ids import split_node_id

if TYPE_CHECKING:
    from aiperf.graph.placement import DispatchRequest, PlacementContext

_logger = AIPerfLogger(__name__)

__all__ = ["CreditDispatchAdapter", "GraphDispatchError", "_NodeOverflowTerminate"]

# Placeholder resolved to the dispatch caller and (for non-messages channels)
# written to ``node.output``. Validated: no downstream node consumes an LLM
# output channel's value (fan-in gating only), so the concrete value is
# irrelevant.
_PLACEHOLDER = ""


class GraphDispatchError(RuntimeError):
    """A graph dispatch failed (worker error OR worker-reported cancellation).

    A plain ``Exception`` subclass (NOT ``asyncio.CancelledError``) so the
    executor's per-trace TaskGroup unwinds it as a normal task failure rather
    than mistaking it for cooperative cancellation of the awaiting coroutine.
    """


class GraphStickinessError(GraphDispatchError):
    """A dynamic slot's pool value was missing on the routed worker (§6.3).

    Broken stickiness (worker death re-route) or dynamic-pool backstop
    eviction. Non-containable: the executor re-raises it as a trace error --
    the workload's content dependency is unsatisfiable, so continuing with
    omission would silently corrupt the trace shape.
    """


class CreditIssueRefusedError(GraphDispatchError):
    """The issuer refused to place the credit on the wire (stop gate / caps).

    Distinct from a post-issue dispatch failure: the executor's
    mid-conversation containment must NOT sentinel-continue past a refusal --
    at duration end every remaining node would otherwise churn through
    sentinel writes without ever dispatching. Refusal is a trace-stop.
    """


class _NodeOverflowTerminate(_NodeExpectedExit):
    """A node dispatch returned a context-overflow error (early termination).

    Subclasses ``_NodeExpectedExit`` so ``_run_node`` catches it as a clean exit
    (no successors, no trace error); the dedicated subclass lets tests
    distinguish overflow.
    """


class CreditDispatchAdapter:
    """Bridges one trace's executor dispatches onto the v1 credit pipeline.

    Constructed per-trace by ``AgentGraphReplayStrategy`` with that trace's
    identity. Thread-confined to the TimingManager event loop: ``dispatch`` is
    awaited from the executor's per-trace TaskGroup and ``resolve`` is invoked
    synchronously from the credit-return callback on the SAME loop, so the
    waiter dict needs no locking.
    """

    def __init__(
        self,
        *,
        credit_issuer: Any,
        catalog_context: CatalogContext,
        trace_id: str,
        instance_id: str | None = None,
        phase: CreditPhase = CreditPhase.PROFILING,
        parent_correlation_id: str | None = None,
        dispatch_timeout_s: float | None = None,
        on_drained: Callable[[CreditDispatchAdapter], None] | None = None,
        first_token_sources: frozenset[str] = frozenset(),
        node_identity: dict[str, tuple[int, str | None]] | None = None,
    ) -> None:
        """Initialize the per-trace dispatch adapter.

        Args:
            credit_issuer: The real ``CreditIssuer``; the adapter calls its
                ``issue_graph_credit`` graph path (session-slot-bypassing).
            catalog_context: Build-time ``{trace_id: {node_key: ordinal}}`` +
                namespace maps; resolves a fired node to its ``node_ordinal``.
            trace_id: This trace's BASE/template root id (e.g. ``"t-1"``). Keys
                BOTH the build-time catalog (node-ordinal resolution) AND the
                graph store mmap (the worker strips any ``#`` recycle suffix
                back to this base). ``ctx.parent_trace_id`` carries this base as
                the bare instance id.
            instance_id: The per-recycle INSTANCE id stamped on ``credit.trace_id``
                (``{template}::{nonce}``, e.g. ``"t-1::3f2a..."`` then a fresh
                ``"t-1::9c17..."`` after one recycle). Distinct
                per (lane, recycle pass) so the worker's cache-bust marker ROTATES
                AND the strategy's return observer routes by it without two
                concurrent instances of one template colliding. Defaults to
                ``trace_id`` for non-recycling callers (unit harness).
            phase: Credit phase stamped on every credit
                (``"profiling"`` / ``"warmup"``). Each variant runs on its own
                adapter instance, so their correlation ids are independent
                mints and never collide.
            parent_correlation_id: Parent session correlation id for children.
            dispatch_timeout_s: Per-dispatch deadlock guard; defaults to
                ``Environment.GRAPH.DISPATCH_TIMEOUT``.
            on_drained: Optional callback invoked (with ``self``) whenever a return
                empties the in-flight waiter set (``inflight_count`` -> 0); the
                strategy uses it to defer adapter teardown until it is truly idle
                (C1 guard).
            first_token_sources: Node ids whose issued turn must carry
                ``first_token_event=True`` -- the sources of this trace's
                first-token-anchored edges (post-TTFT anchoring). A dispatch of
                such a node emits a ``FirstToken`` so a successor gated on the
                node's observed first token can be released. Empty (default) =>
                no node emits a first-token event (pre-anchoring byte parity).
            node_identity: ``node_id -> (agent_depth, parent_node_id)`` legacy
                record identity map (dag_jsonl lowering's ``metadata["dag"]``
                stamp). A dispatch whose node hits the map carries the mapped
                depth and, when ``parent_node_id`` is set, the parent NODE's
                derived correlation id. ``None`` (weka/dynamo default) keeps
                behavior byte-identical: depth 0 and the constructor's
                ``parent_correlation_id`` on every credit.
        """
        self._issuer = credit_issuer
        self._catalog = catalog_context
        self._trace_id = trace_id
        # Instance id stamped on every credit's ``trace_id`` (marker rotation +
        # strategy return de-mux); falls back to the base id when no recycle suffix.
        self._instance_id = instance_id if instance_id is not None else trace_id
        self._phase = phase
        # scope -> trajectory-instance x_correlation_id, minted lazily per
        # scope: fresh per adapter (= per instance/recycle), stable across all
        # of that trajectory's turns within this instance.
        self._scope_corrs: dict[str, str] = {}
        self._parent_correlation_id = parent_correlation_id
        # scope -> recorded turn count, from the build-time catalog: the max
        # {scope}:{turn} coordinate per scope (+1), with unshaped bare ids
        # counting as root-trajectory turns by ordinal. Gives every credit a
        # REAL num_turns so ``is_final_turn`` carries the trajectory's
        # recorded session-final fact (session-routing modes key bind/close
        # and session-final semantics on it).
        scope_last_turn: dict[str, int] = {}
        for node_key, ordinal in catalog_context.catalog.get(trace_id, {}).items():
            shaped = split_node_id(node_key)
            scope, turn = shaped if shaped is not None else (trace_id, ordinal)
            if turn > scope_last_turn.get(scope, -1):
                scope_last_turn[scope] = turn
        self._scope_num_turns = {
            scope: last + 1 for scope, last in scope_last_turn.items()
        }
        self._timeout_s = (
            dispatch_timeout_s
            if dispatch_timeout_s is not None
            else Environment.GRAPH.DISPATCH_TIMEOUT
        )
        # Correlation key -> parked Future: (x_correlation_id, turn_index),
        # i.e. the scope's lazily-minted trajectory-INSTANCE corr plus the
        # node's own recorded turn coordinate. Uniqueness follows from the
        # executor firing each node at most once per instance run; a re-fire
        # trips the duplicate-waiter guard in ``dispatch`` instead of silently
        # sharing a waiter.
        self._waiters: dict[tuple[str, int], asyncio.Future[str]] = {}
        # Node ids whose dispatch stamps ``first_token_event=True`` (sources of
        # this trace's first-token-anchored edges); the successor gates on the
        # emitted ``FirstToken`` (post-TTFT anchoring).
        self._first_token_sources = first_token_sources
        # node_id -> (agent_depth, parent_node_id) legacy identity map (dag);
        # None => every dispatch is a root-chain firing (weka/dynamo).
        self._node_identity = node_identity
        # Correlation key -> zero-arg callback fired ONCE when a FirstToken for
        # that key arrives (``on_first_token``). Parked per-dispatch beside the
        # waiter and popped in the SAME resolve/finally cleanup so nothing leaks.
        self._first_token_cbs: dict[tuple[str, int], Callable[[], None]] = {}
        # Invoked whenever the in-flight waiter set drains to empty so the owner
        # can pop the adapter only once it is truly idle.
        self._on_drained = on_drained

    @property
    def inflight_count(self) -> int:
        """Number of dispatches currently awaiting a return (test/diagnostic)."""
        return len(self._waiters)

    @property
    def phase(self) -> CreditPhase:
        """Agent-graph phase variant label this adapter stamps on its credits."""
        return self._phase

    @property
    def instance_id(self) -> str:
        """The per-recycle instance id this adapter stamps on ``credit.trace_id``.

        The strategy's return observer routes credits back to the owning adapter
        by this id (NOT the base template id), so two concurrent recycle
        instances of one template never collide on the de-mux registry.
        """
        return self._instance_id

    async def dispatch(
        self,
        node: Any,
        request: DispatchRequest,
        ctx: PlacementContext,
        first_token_cb: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> tuple[str, int | None, float | None, float | None]:
        """Issue a graph credit for ``node`` and await its correlated return.

        Tolerates and ignores extra keyword arguments from the LLM dispatch
        path. Returns ``(placeholder, observed_osl, request_latency_s,
        ttft_s)`` on a normal return, where ``observed_osl`` is
        ``credit_return.output_sequence_length`` (None when the worker did not
        report it). Raises on cancel / error / timeout so the executor coroutine
        unwinds rather than hangs.

        ``first_token_cb`` (optional): a zero-arg callable parked under this
        dispatch's waiter key and invoked AT MOST ONCE by :meth:`on_first_token`
        when the emitting credit's ``FirstToken`` arrives -- the release hook a
        first-token-anchored successor registers (post-TTFT anchoring). It is
        popped in the SAME resolve/timeout/finally paths as the waiter, so a
        dispatch that resolves without a TTFT never leaks or late-fires it.
        """
        runtime_trace_id = ctx.parent_trace_id or self._trace_id
        node_id = request.node_id
        node_ordinal = self._resolve_ordinal(runtime_trace_id, node_id)

        x_corr, turn_index, num_turns = self._mint(node_id, node_ordinal)
        key = (x_corr, turn_index)
        if key in self._waiters:
            raise RuntimeError(
                f"duplicate in-flight dispatch for {key!r} (node {node_id!r}, "
                f"instance {self._instance_id!r}): the executor fires each "
                "node at most once per instance run"
            )
        conversation_id = self._conversation_identity(node_id)
        agent_depth, parent_correlation_id = self._dag_identity(node_id)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[tuple[str, int | None, float | None, float | None]] = (
            loop.create_future()
        )
        self._waiters[key] = fut
        if first_token_cb is not None:
            self._first_token_cbs[key] = first_token_cb

        turn = TurnToSend(
            conversation_id=conversation_id,
            x_correlation_id=x_corr,
            turn_index=turn_index,
            num_turns=num_turns,
            agent_depth=agent_depth,
            parent_correlation_id=parent_correlation_id,
            # The instance's ROOT trajectory corr: session-tree identity for
            # session-routing plugins (dynamo parent/tree grouping). Minted
            # lazily, so it is stable even before the root scope fires.
            root_correlation_id=self._corr_of(self._trace_id),
            trace_id=self._instance_id,
            node_ordinal=node_ordinal,
            first_token_event=node_id in self._first_token_sources,
        )
        try:
            issued = await self._issuer.issue_graph_credit(turn)
            if not issued:
                pending = self._waiters.pop(key, None)
                # No credit reached the wire => no FirstToken will arrive; drop
                # any parked first-token cb so it cannot late-fire on reuse.
                self._first_token_cbs.pop(key, None)
                if pending is not None and not pending.done():
                    pending.set_exception(
                        CreditIssueRefusedError(
                            "graph credit refused by issuer (stop/duration/"
                            "request-count cap reached or run cancelled); "
                            f"trace={self._trace_id!r} node_ordinal={node_ordinal!r}"
                            " -- no return will arrive, stopping this trace"
                        )
                    )
            return await asyncio.wait_for(fut, timeout=self._timeout_s)
        except (TimeoutError, asyncio.CancelledError):
            # Drop the orphaned waiter so a late return can't resolve a dead
            # Future, then re-raise so the executor coroutine unwinds.
            self._waiters.pop(key, None)
            self._first_token_cbs.pop(key, None)
            raise
        finally:
            # Defensive: a resolved/rejected Future may already be popped by
            # ``resolve``; ensure no waiter (or parked first-token cb) lingers.
            self._waiters.pop(key, None)
            self._first_token_cbs.pop(key, None)

    def resolve(
        self,
        credit: Credit,
        error: str | None,
        cancelled: bool,
        *,
        osl: int | None = None,
        request_latency_ns: int | None = None,
        ttft_ns: int | None = None,
    ) -> None:
        """Resolve (or reject) the Future parked for ``credit``'s correlation key.

        Driven by ``CreditCallbackHandler``'s unconditional graph-return hook.
        Unknown keys (already resolved, or never parked) are a graceful no-op.
        Fires ``on_drained`` whenever this return empties the in-flight waiter set,
        so the owner can defer de-mux teardown until the adapter is idle.
        """
        key = (credit.x_correlation_id, credit.turn_index)
        fut = self._waiters.pop(key, None)
        # The dispatch is settling; a first-token cb not already consumed by
        # ``on_first_token`` (error/cancel before TTFT, or non-streaming node)
        # is now moot -- drop it so it cannot late-fire.
        self._first_token_cbs.pop(key, None)
        try:
            if fut is None:
                _logger.debug(
                    lambda: (
                        f"graph return for unknown waiter key {key} "
                        f"(trace={credit.trace_id}); dropped"
                    )
                )
                return
            if fut.done():
                return
            if error is not None and is_context_overflow_response(body=error):
                # An overflowed node TERMINATES the trajectory early -- later
                # turns carry even more context and would only overflow too. The
                # clean exit suppresses downstream dispatch; the overflow record
                # still flows to the RecordProcessor metrics-skip path.
                _logger.info(
                    lambda: (
                        f"Terminating trajectory {self._instance_id!r} early at "
                        f"node_ordinal={credit.node_ordinal!r}: "
                        "context-overflow error from server"
                    )
                )
                fut.set_exception(
                    _NodeOverflowTerminate(f"context-overflow early-term: {error}")
                )
            elif parse_graph_error(error) is GraphErrorCode.POOL_MISSING:
                fut.set_exception(GraphStickinessError(error))
            elif error is not None:
                fut.set_exception(
                    GraphDispatchError(f"graph dispatch errored: {error}")
                )
            elif cancelled:
                fut.set_exception(
                    GraphDispatchError("graph dispatch cancelled by worker return")
                )
            else:
                fut.set_result(
                    (
                        _PLACEHOLDER,
                        osl,
                        request_latency_ns / 1e9
                        if request_latency_ns is not None
                        else None,
                        ttft_ns / 1e9 if ttft_ns is not None else None,
                    )
                )
        finally:
            if self._on_drained is not None and not self._waiters:
                self._on_drained(self)

    def on_first_token(
        self, x_correlation_id: str | None, turn_index: int | None
    ) -> None:
        """Fire the first-token callback parked for one dispatch (AT MOST ONCE).

        Driven by ``AgentGraphReplayStrategy._on_graph_first_token`` when the
        emitting graph credit's TTFT arrives (post-TTFT anchoring). Pops the
        callback registered under ``(x_correlation_id, turn_index)`` and invokes
        it, so a successor gated on this node's observed first token is released.
        Popping guarantees the single-fire contract: a second call for the same
        key (or a return that already dropped the cb) finds nothing. An unknown
        key -- a node that carried no first-token successor, or a ``None`` field
        on a non-graph fast-path token -- is a graceful no-op.
        """
        cb = self._first_token_cbs.pop((x_correlation_id, turn_index), None)
        if cb is not None:
            # The observer path runs as a fire-and-forget task, so an exception
            # here would surface only as an unretrieved-task warning. Log it and
            # move on: the cb is already popped, so the successor simply waits for
            # its finalize latch instead of the (lost) first-token release.
            try:
                cb()
            except Exception:
                _logger.exception(
                    lambda: (
                        "first-token callback raised for key "
                        f"{(x_correlation_id, turn_index)!r}"
                    )
                )

    # ------------------------------------------------------------------ helpers

    def _resolve_ordinal(self, runtime_trace_id: str, node_id: str) -> int | None:
        """Map a fired executor node to its build-time ``node_ordinal``.

        Every live producer lowers to a flat LlmNode graph, so the fired node's
        bare ``node_id`` IS its catalog key (the executor never descends into
        child scopes; ``runtime_trace_id`` is always the bare instance id).
        """
        ordinal = node_ordinal_for(self._catalog, self._trace_id, node_id)
        if ordinal is None:
            _logger.warning(
                lambda: (
                    f"no node_ordinal for trace={self._trace_id!r} "
                    f"key={node_id!r} (runtime_trace={runtime_trace_id!r}); "
                    f"credit will carry node_ordinal=None"
                )
            )
        return ordinal

    def _conversation_identity(self, node_id: str) -> str:
        """Return the ``conversation_id`` stamped on one fired node's credit.

        The trajectory TEMPLATE id (legacy semantics: stable across recycles,
        deliberately duplicated in exports) -- the trace id for root-scope
        nodes, ``{trace_id}::{scope}`` for child trajectories. Instance
        identity rides ``x_correlation_id``; depth/parent identity is per-NODE
        and lives in :meth:`_dag_identity`. Unshaped ids (no
        ``{scope}:{turn}`` coordinate) belong to the root trajectory, mirroring
        :meth:`_mint`.
        """
        shaped = split_node_id(node_id)
        return self._conversation_of(shaped[0] if shaped else self._trace_id)

    def _dag_identity(self, node_id: str) -> tuple[int, str | None]:
        """Return ``(agent_depth, parent_correlation_id)`` for one fired node.

        A ``node_identity`` hit (dag_jsonl lowering's ``metadata["dag"]`` stamp)
        yields the instance's depth; when the stamp names a triggering parent
        node, the parent's correlation id comes from the SAME per-scope
        :meth:`_corr_of` mint :meth:`_mint` uses (one lazily minted id per
        trajectory scope, no node id in it), so the child's
        ``parent_correlation_id`` equals the parent node's minted
        ``x_correlation_id`` exactly. No map / miss / parent-less stamp falls
        back to the legacy root-chain identity: depth from the stamp (or 0) and
        the constructor's ``parent_correlation_id`` (weka/dynamo byte-identical
        when the map is absent).
        """
        identity = (
            None if self._node_identity is None else self._node_identity.get(node_id)
        )
        if identity is None:
            return 0, self._parent_correlation_id
        agent_depth, parent_node_id = identity
        if parent_node_id is None:
            return agent_depth, self._parent_correlation_id
        shaped = split_node_id(parent_node_id)
        return agent_depth, self._corr_of(shaped[0] if shaped else self._trace_id)

    def _mint(self, node_id: str, node_ordinal: int | None) -> tuple[str, int, int]:
        """Mint ``(x_correlation_id, turn_index, num_turns)`` for one fire --
        legacy semantics.

        ``x_correlation_id`` is the node's TRAJECTORY-INSTANCE id: one fresh
        ``{conversation_id}::{uuid4().hex}`` minted lazily per scope per
        adapter instance (all turns of one trajectory share it, exactly like a
        linear session; recycles get a fresh adapter and therefore fresh
        corrs). ``turn_index`` is the node's own 0-based turn within its
        trajectory -- the ``{scope}:{turn}`` node id IS the legacy
        ``(conversation, turn_index)`` coordinate. Uniqueness of the waiter
        key follows from the executor firing each node at most once per
        instance run; a re-fire would collide loudly in ``dispatch`` rather
        than silently share a waiter.

        A node id WITHOUT the ``{scope}:{turn}`` shape (an author-chosen id
        like ``"plan"``; no producer emits these in this release) maps to the
        ROOT
        trajectory with the catalog ``node_ordinal`` as its turn: the whole
        graph rides one sticky session (worker-local dynamic-slot content
        stays reachable) and ordinals keep waiter keys unique per node. A
        pathological mix of a bare id and a root-scoped ``{trace}:{ordinal}``
        id could collide -- caught loudly by the duplicate-waiter guard.
        """
        shaped = split_node_id(node_id)
        if shaped is not None:
            scope, turn = shaped
        else:
            scope, turn = self._trace_id, node_ordinal or 0
        # Recorded trajectory turn count from the catalog; a runtime scope the
        # catalog does not know (defensive) reads as NON-final -- a wrong
        # session-close on the wire is worse than a missing one.
        num_turns = self._scope_num_turns.get(scope, turn + 2)
        return self._corr_of(scope), turn, max(num_turns, turn + 1)

    def _corr_of(self, scope: str) -> str:
        """The trajectory-instance correlation id for ``scope`` (lazy mint)."""
        corr = self._scope_corrs.get(scope)
        if corr is None:
            corr = f"{self._conversation_of(scope)}::{uuid.uuid4().hex}"
            self._scope_corrs[scope] = corr
        return corr

    def _conversation_of(self, scope: str) -> str:
        """Trajectory TEMPLATE id: the trace id for the root scope, else
        ``{trace_id}::{scope}`` (corpus-unique; recorded child scopes like
        weka ``agent_001`` recur across traces)."""
        if scope == self._trace_id:
            return scope
        return f"{self._trace_id}::{scope}"
