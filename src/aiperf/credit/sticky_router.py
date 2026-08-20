# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sticky credit router with fair load balancing.

Routes credits to workers: sticky routing for multi-turn sessions,
least-loaded selection for first turns. Lock-free via asyncio serialization.

Terminology:
    session: A unique execution of a conversation template, identified by
        x_correlation_id (UUID). All turns in a session route to the same worker.
    conversation_id: Template ID from the dataset (can be reused across sessions).

Includes:
- WorkerLoad: Worker load tracking for fair load balancing
- StickyCreditRouter: Main router class
"""

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CommAddress, CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task
from aiperf.common.mixins import CommunicationMixin
from aiperf.common.phase import PhaseRuntimeKey, phase_runtime_key
from aiperf.common.protocols import (
    StreamingPullClientProtocol,
    StreamingRouterClientProtocol,
)
from aiperf.config.comm import BaseZMQCommunicationConfig, ZMQDualBindConfig
from aiperf.credit.messages import (
    CancelCredits,
    CreditReturn,
    FirstToken,
    TimePing,
    TimePong,
    WorkerConnected,
    WorkerDispatchable,
    WorkerShutdown,
    WorkerToRouterMessage,
    WorkerUndispatchable,
)
from aiperf.credit.structs import Credit

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

# =============================================================================
# Data Models
# =============================================================================


@dataclass(slots=True)
class _StickyEntry:
    """Sticky-routing state for a root correlation id.

    Tracks which worker owns the session and a refcount so DAG children that
    pin themselves to the parent's worker can keep the entry alive past the
    parent's own final turn. ``parent_final_seen`` records whether the owning
    session has finished its final turn; the entry is popped only once both
    ``ref_count`` hits zero and that flag is set.
    """

    worker_id: str
    phase_key: PhaseRuntimeKey | None = None
    ref_count: int = 1
    parent_final_seen: bool = False
    root_key: str = ""
    """The correlation id this entry is filed under in ``_sticky_sessions``.

    Eviction is driven by whichever key the caller happened to hold, and for a
    DAG descendant that key is an *alias*. Popping by it alone left the root
    key pointing at a dead entry and left the root id in the owning worker's
    ``active_session_ids``, so ``active_sessions`` and that set drifted apart
    for the rest of the run. Recording the root at creation lets eviction
    remove the entry by its real key no matter which alias found it.
    """
    aliases: set[str] = field(default_factory=set)
    """Extra keys resolving to this entry (descendant correlation ids).

    A DAG credit looks its entry up by ``parent_correlation_id``, but only the
    session root ever owns an entry: a depth-1 child shares the root's. So a
    depth-2 grandchild, whose parent id is that depth-1 child, found nothing
    and fell through to least-loaded routing -- losing exactly the
    prefix-cache locality the refcount machinery exists to preserve. Aliasing
    each registered child's own id onto its ancestor's entry makes the lookup
    resolve at any depth; the aliases are dropped with the entry.
    """


@dataclass(slots=True)
class UnavailableSession:
    """Sticky routing state retained after its worker becomes unavailable.

    The same instance is indexed by the root routing key and every descendant
    alias so nested DAG children cannot silently migrate after losing the
    worker that held their inherited context.
    """

    worker_id: str
    reason: str
    phase_key: PhaseRuntimeKey | None = None
    pending_holders: int = 1
    owner_pending: bool = True
    root_key: str = ""
    aliases: set[str] = field(default_factory=set)


@dataclass(slots=True)
class WorkerLoad:
    """Worker load tracking for fair load balancing.

    Note on virtual_sent_credits vs total_sent_credits:
        - total_sent_credits: Actual count of credits sent (for metrics/debugging)
        - virtual_sent_credits: Used for fairness tie-breaking, initialized to
          average when worker joins mid-benchmark to prevent "thundering herd"
          where a new worker with 0 credits gets all requests.

    Note on active_sessions:
        - active_sessions and active_session_ids only represent the number of sticky sessions assigned
          to the worker, which inherently means that it only tracks sessions with MORE turns left. This is
          because sticky sessions are only created when more than 1 turn exists, and are removed when SENDING the final turn.
    """

    worker_id: str
    total_sent_credits: int = 0
    virtual_sent_credits: int = (
        0  # For fairness comparison (initialized to avg on join)
    )
    total_completed_credits: int = 0
    total_cancelled_credits: int = 0
    total_errors_reported: int = 0
    in_flight_credits: int = 0
    last_heartbeat_ns: int = 0
    """Wall-clock of the last service heartbeat observed for this worker; 0
    until one arrives. Published on its own timer by the worker service
    independently of request work, so it is the only signal here that separates
    a dead worker from a slow one. Drives ``evict_stale_workers``."""
    active_credit_ids: set[int] = field(default_factory=set)
    active_sessions: int = 0  # Sticky sessions assigned to this worker
    active_session_ids: set[str] = field(default_factory=set)
    last_sent_at_ns: int = (
        0  # For tie-breaking (guaranteed unique in single-threaded asyncio)
    )


# ==============================================================================
# Credit Router Protocol
# ==============================================================================


@runtime_checkable
class CreditRouterProtocol(Protocol):
    """Protocol for routing credits to workers.

    Decouples credit issuing strategies from routing implementation.
    Enables mocking for tests and alternative routing strategies.
    """

    async def send_credit(self, credit: Credit) -> None:
        """Send credit to worker via routing strategy.

        Args:
            credit: Credit to send to worker
        """
        ...

    async def wait_for_workers(self, timeout: float) -> None:
        """Wait for the router to observe at least one registered worker.

        Best-effort startup gate, not an absolute guarantee: a worker can
        unregister again after this returns. Raises on timeout.

        Args:
            timeout: Seconds to wait for the first worker before giving up.
        """
        ...

    async def cancel_all_credits(self) -> None:
        """Cancel all in-flight credits.

        Used during phase timeout or system shutdown.
        """
        ...

    def begin_phase(self, phase: CreditPhase, phase_index: int | None = None) -> None:
        """Reset cancellation state before this phase starts issuing."""
        ...

    def end_phase(self, phase: CreditPhase, phase_index: int | None = None) -> None:
        """Discard unavailable-session state after one phase fully drains."""
        ...

    def mark_credits_complete(self) -> None:
        """Mark that all credits have been issued and returned.

        Called by orchestrator when benchmark completes normally.
        Suppresses warnings about orphaned sessions during shutdown.
        """
        ...

    def set_return_callback(
        self,
        callback: Callable[[str, CreditReturn], Awaitable[None]],
    ) -> None:
        """Register callback for credit returns.

        Args:
            callback: Async function called when credit returns.
                     Signature: (worker_id: str, message: CreditReturn) -> None
        """
        ...

    def set_first_token_callback(
        self,
        callback: Callable[[FirstToken], Awaitable[None]],
    ) -> None:
        """Register callback for first token events (prefill concurrency release).

        Args:
            callback: Async function called when first token is received.
                     Signature: (message: FirstToken) -> None
        """
        ...


# =============================================================================
# Sticky Credit Router
# =============================================================================


class StickyCreditRouter(CommunicationMixin):
    """Routes credits to workers with sticky sessions and fair load balancing.

    All messages between the Worker and TimingManager service flow through the CreditRouter.

    IMPORTANT:
        - This class has been highly optimized for performance, as it is a hot path.
        - Please be careful when making changes to ensure performance is not degraded.
        - All operations are atomic because there are no await calls between reads and writes.
        - Methods are intentionally large/inlined to avoid function call overhead in the hot path.
        - The class is designed for single-threaded asyncio use only.

    Credit Routing:
        - First turn → least-loaded worker (creates sticky session).
        - Subsequent turns → same worker via sticky session lookup.
        - Final turn → cleanup sticky session.

    Load Balancing:
        - Least-loaded worker selection for new sessions using fair load balancing
            - Determined by the worker(s) with the fewest in-flight credits.
        - Tie-breaking for multiple workers in this order:
            - `active_sessions`: Prefer workers with fewer committed multi-turn sessions
            - `virtual_sent_credits`: Prefer workers with fewer historical credits (virtual to handle
                late-joining workers fairly - they start at average, not zero)
            - `last_sent_at_ns`: Prefer workers with oldest send time (LRU-like fairness)

    Credit Returns:
        - All CreditReturns and FirstTokens flow through the CreditRouter and
          are forwarded via callbacks that are directly awaited for responsiveness.

    Lock-free:
        - Ensure there are no await calls in critical paths.

    Hot path complexity:
        - sticky session lookup is O(1)
        - min load tracking/lookup is O(1)
        - load balancing for new sessions is O(k) where k = workers tied at min load
        - credit sent/returned tracking is O(1)

    Cold path complexity:
        - worker register/unregister is O(n) where n = number of workers
        - credit cancellation is O(n × k) where n = number of workers, k = average in-flight credits per worker
    """

    def _init_credit_channels(
        self, comm_config: BaseZMQCommunicationConfig | None
    ) -> None:
        """Bind the credit dispatch ROUTER and the dedicated credit-return PULL.

        Dispatch (Credit/CancelCredits) goes router->worker over CREDIT_ROUTER;
        CreditReturn/FirstToken fan in worker->router over a separate PUSH/PULL
        channel (CREDIT_RETURN), so neither socket is bidirectional. In dual-bind
        (k8s controller) mode each also binds its TCP address so remote worker
        pods can connect; controller-side services otherwise use IPC.
        """
        dual_bind = (
            isinstance(comm_config, ZMQDualBindConfig)
            and not comm_config.controller_host
        )

        dispatch_bind = (
            comm_config.credit_router_tcp_bind_address if dual_bind else None
        )
        if dispatch_bind:
            self.info(
                f"Dual-bind mode: credit router will also bind to {dispatch_bind}"
            )
        self._router_client: StreamingRouterClientProtocol = (
            self.comms.create_streaming_router_client(
                address=CommAddress.CREDIT_ROUTER,
                bind=True,
                additional_bind_address=dispatch_bind,
            )
        )
        self._router_client.register_receiver(self._handle_router_message)

        return_bind = (
            comm_config.credit_return_push_pull_tcp_bind_address if dual_bind else None
        )
        if return_bind:
            self.info(
                f"Dual-bind mode: credit return PULL will also bind to {return_bind}"
            )
        self._return_pull_client: StreamingPullClientProtocol = (
            self.comms.create_streaming_pull_client(
                CommAddress.CREDIT_RETURN,
                bind=True,
                additional_bind_address=return_bind,
            )
        )
        self._return_pull_client.register_receiver(self._handle_return_pull_message)

    def __init__(
        self,
        run: "BenchmarkRun",
        service_id: str,
        **kwargs,
    ) -> None:
        super().__init__(run=run, service_id=service_id, **kwargs)

        self._init_credit_channels(run.cfg.comm_config)

        self._on_return_callback: (
            Callable[[str, CreditReturn], Awaitable[None]] | None
        ) = None
        self._on_first_token_callback: (
            Callable[[FirstToken], Awaitable[None]] | None
        ) = None
        self._on_fatal_error: Callable[[BaseException], None] | None = None
        self._on_worker_count_changed: Callable[[int], None] | None = None

        # Sticky sessions: routing_key -> _StickyEntry
        # Routes all turns of a conversation (and DAG children pinned to it) to the
        # same worker. Required because workers cache UserSession state by
        # x_correlation_id. The routing key is ``parent_correlation_id or
        # x_correlation_id`` so FORK-mode children co-locate with their parent.
        self._sticky_sessions: dict[str, _StickyEntry] = {}
        self._unavailable_sessions: dict[str, UnavailableSession] = {}

        self._cancellation_pending: bool = False
        self._credits_complete: bool = False

        # Snapshot list for iteration - avoids dict.values() overhead in hot path.
        # Rebuilt on worker add/remove (rare) to keep routing fast (common).
        self._workers_cache: list[WorkerLoad] = []
        self._workers: dict[str, WorkerLoad] = {}
        # Load records of workers dropped by the staleness sweep, kept so a
        # worker that was wrongly presumed dead can be readmitted with its
        # in-flight credit accounting intact (WorkerDispatchable is sent exactly
        # once, so without this a false positive is permanent).
        self._evicted_workers: dict[str, WorkerLoad] = {}
        self._peak_worker_count: int = 0

        # Workers whose return path is up but which are not necessarily
        # dispatchable yet. Strict superset of ``_workers``: a worker appears
        # here on WorkerConnected and only enters ``_workers`` (the routing
        # pool) on WorkerDispatchable. Tracked so "connected but not
        # dispatchable" is observable rather than looking like an absent pod.
        self._connected_workers: set[str] = set()

        # Map load level -> set of worker_ids at that load (O(1) add/remove)
        self._workers_by_load: dict[int, set[str]] = defaultdict(set)
        # Keep track of the minimum load to avoid recalculating it on every credit sent O(1) vs O(n)
        self._min_load: int = 0

        # Set while >=1 worker is registered; lets wait_for_workers() gate a
        # phase on worker readiness (see that method for the race it closes).
        self._worker_available_event: asyncio.Event = asyncio.Event()

    # =============================================================================
    # Public Methods
    # =============================================================================

    def set_return_callback(
        self, callback: Callable[[str, CreditReturn], Awaitable[None]]
    ) -> None:
        """Set callback for credit returns (enables concurrency control)."""
        self._on_return_callback = callback

    def set_fatal_error_callback(
        self, callback: Callable[[BaseException], None]
    ) -> None:
        """Register a sink for fatal request-free control-node failures.

        Called (synchronously) when a detached virtual-return callback raises,
        so the failure can be recorded on the phase and surfaced instead of
        being logged and swallowed.
        """
        self._on_fatal_error = callback

    def set_worker_count_changed_callback(
        self, callback: Callable[[int], None]
    ) -> None:
        """Register a cold-path observer for dispatchable-worker membership."""
        self._on_worker_count_changed = callback

    def set_first_token_callback(
        self, callback: Callable[[FirstToken], Awaitable[None]]
    ) -> None:
        """Set callback for first token events (enables prefill concurrency release)."""
        self._on_first_token_callback = callback

    async def wait_for_workers(self, timeout: float) -> None:
        """Close the startup race where a phase issues its first credit before
        any worker has sent ``WorkerDispatchable`` (which makes ``send_credit``
        raise on empty workers). Called once per phase before the first credit.

        Best-effort startup gate, not an absolute postcondition: the last worker
        can unregister between this returning and the first ``send_credit``, so
        callers must not treat a non-empty pool as guaranteed afterwards.

        Args:
            timeout: Seconds to wait for the first worker before giving up.

        Raises:
            RuntimeError: If no worker registers within ``timeout`` seconds.
        """
        if self._workers:
            return
        try:
            await asyncio.wait_for(self._worker_available_event.wait(), timeout)
        except TimeoutError as exc:
            raise RuntimeError(
                f"No workers registered with the credit router within {timeout}s "
                "(tunable via AIPERF_SERVICE_START_TIMEOUT); cannot start credit issuance"
            ) from exc

    async def _fire_virtual_return(
        self, credit_return: CreditReturn, worker_id: str = ""
    ) -> None:
        """Deliver a synthesized CreditReturn to the return consumer.

        Runs as a detached task (scheduled via ``execute_async``). On failure the
        error is logged AND forwarded to the fatal-error sink so it surfaces to
        the phase (which re-raises it) instead of only reaching asyncio's default
        "Task exception was never retrieved" handler -- otherwise a failure on the
        spawn-dispatch path (``intercept``) would become an opaque phase-timeout
        hang with the graph silently stuck.
        """
        try:
            await self._on_return_callback(worker_id, credit_return)
        except Exception as e:
            self.exception(
                lambda: f"synthetic return callback failed for credit "
                f"{credit_return.credit.id} (x_correlation_id="
                f"{credit_return.credit.x_correlation_id})"
            )
            if self._on_fatal_error is not None:
                self._on_fatal_error(e)

    async def send_credit(self, credit: Credit) -> None:
        """Determine the worker based on sticky sessions or least-loaded and send the credit to the worker.

        This method:
        - Determines the worker based on sticky sessions or least-loaded
        - Updates the worker load and sticky sessions
        - Sends the credit to the worker
        """
        if not credit.x_correlation_id:
            raise RuntimeError("x_correlation_id must be set in Credit")

        if credit.no_request:
            credit_return = CreditReturn(
                credit=credit, cancelled=False, error=None, first_token_sent=False
            )
            # Virtual orchestrator credit: never goes to a worker. Synthesize the
            # return in-process and hand it to the same return consumer a worker
            # return would hit, so slot release + BranchOrchestrator.intercept
            # (spawn firing) run identically. No _track_credit_sent/_returned here:
            # this credit never touches per-worker load, so tracking it would
            # desync in_flight_credits and trip the return-underflow error.
            #
            # Schedule DECOUPLED (not awaited inline): if this send_credit is
            # reached from inside BranchOrchestrator.intercept (which holds
            # _parent_locks[corr]), awaiting the callback here would re-enter
            # on_credit_return -> intercept and can deadlock the non-reentrant
            # asyncio.Lock when correlations collide, and risks unbounded
            # synchronous recursion. Deferring to a later event-loop turn matches
            # the exact semantics of the ZMQ worker round-trip we are replacing.
            if self._on_return_callback is None:
                raise RuntimeError(
                    "return callback not set; cannot short-circuit no_request credit"
                )
            self.execute_async(self._fire_virtual_return(credit_return))
            return

        # DAG children pin to their parent's worker; otherwise pin to self.
        routing_key = credit.parent_correlation_id or credit.x_correlation_id
        is_dag_child = credit.parent_correlation_id is not None
        sticky_entry = self._sticky_sessions.get(routing_key)
        sticky_worker_id = sticky_entry.worker_id if sticky_entry is not None else None
        # Empty until a worker is actually lost, which never happens in a
        # healthy run. Test the dict for emptiness before hashing the routing
        # key: this is the per-credit dispatch path.
        unavailable_session = (
            self._unavailable_sessions.get(routing_key)
            if self._unavailable_sessions
            else None
        )

        if unavailable_session is not None and not credit.allow_worker_migration:
            if not is_dag_child:
                self._release_unavailable_owner(routing_key)
            if self._on_return_callback is None:
                raise RuntimeError(
                    "return callback not set; cannot fail unavailable sticky session"
                )
            self.execute_async(
                self._fire_virtual_return(
                    CreditReturn(
                        credit=credit,
                        cancelled=True,
                        first_token_sent=False,
                        error=unavailable_session.reason,
                    ),
                    unavailable_session.worker_id,
                )
            )
            return

        if not self._workers:
            raise RuntimeError("No workers available for routing")

        # Use existing sticky session if worker still valid
        if sticky_worker_id and sticky_worker_id in self._workers:
            worker_id = sticky_worker_id
        else:
            if self._unavailable_sessions and not is_dag_child:
                self._pop_unavailable(routing_key)
            # Least-loaded selection with O(k) tie-breaking where k = workers at min load.
            # Min load lookup is O(1) due to caching.
            least_loaded_workers = self._workers_by_load[self._min_load]
            if len(least_loaded_workers) == 1:
                # Pop the single worker directly. _track_credit_sent will add it to the new load level.
                worker_id = least_loaded_workers.pop()
            else:
                # Multiple workers at min load - find best via single-pass scan.
                # O(k) where k = workers at min load.
                #
                # Tie-breaking priority (lower wins):
                #   1. active_sessions: Fewer committed multi-turn sessions
                #   2. virtual_sent_credits: Fewer historical credits
                #   3. last_sent_at_ns: Oldest send time (LRU-like fairness)
                #
                # Both virtual_sent_credits and last_sent_at_ns are initialized to
                # non-zero values on worker registration to prevent thundering herd.
                # Manual loop is benchmarked faster than min() with lambdas.
                best_worker_id = None
                best_load_key = None
                for _worker_id in least_loaded_workers:
                    load = self._workers[_worker_id]
                    load_key = (
                        load.active_sessions,
                        load.virtual_sent_credits,
                        load.last_sent_at_ns,
                    )
                    if best_load_key is None or load_key < best_load_key:
                        best_load_key = load_key
                        best_worker_id = _worker_id

                worker_id = best_worker_id

            # Create or rebind the sticky entry for non-final turns; also create
            # it when the final turn declares DAG spawns so the orchestrator's
            # register_child_routing can find it.
            #
            # DAG branch-children (parent_correlation_id set — FORK or SPAWN)
            # must not auto-create when the parent's sticky entry is already
            # gone. The auto-create path would mint a fresh entry keyed by the
            # parent's id, bumping ``active_sessions`` with no path to evict it
            # (final-turn eviction is gated on parent_correlation_id is None;
            # release_child_routing only decrements an existing entry). That
            # leaks active_sessions and biases load balancing. When the parent
            # entry still exists, both FORK and SPAWN children co-locate via
            # the sticky hit above (routing_key = parent_correlation_id).
            # SPAWN differs only in refcount: the orchestrator does not call
            # register_child_routing for SPAWN. When the parent entry is gone,
            # children fall through to least-loaded without minting a leak.
            if not credit.is_final_turn or credit.has_forks:
                if sticky_entry is None and not is_dag_child:
                    sticky_entry = _StickyEntry(
                        worker_id=worker_id,
                        phase_key=phase_runtime_key(credit.phase, credit.phase_index),
                        root_key=routing_key,
                    )
                    self._sticky_sessions[routing_key] = sticky_entry
                    load = self._workers[worker_id]
                    load.active_sessions += 1
                    load.active_session_ids.add(routing_key)
                elif (
                    sticky_entry is not None
                    and sticky_entry.worker_id not in self._workers
                ):
                    sticky_entry.worker_id = worker_id
                    load = self._workers[worker_id]
                    load.active_sessions += 1
                    # Re-file under the root: routing_key is an alias when a
                    # DAG descendant is what triggered the rebind, and the
                    # eviction paths only ever discard the root.
                    load.active_session_ids.add(sticky_entry.root_key or routing_key)

        # Owning session's final turn: mark parent_final_seen and decrement the
        # reservation. DAG children never touch the parent entry (managed via
        # release_child_routing). If this turn has DAG spawns, leave the entry
        # in place so register_child_routing lands on the same _StickyEntry.
        if credit.is_final_turn and credit.parent_correlation_id is None:
            entry = sticky_entry or self._sticky_sessions.get(routing_key)
            if entry is not None:
                entry.parent_final_seen = True
                entry.ref_count -= 1
                if entry.ref_count <= 0 and not credit.has_forks:
                    self._pop_entry(routing_key, entry)
                    # Give the session back to the worker it was counted
                    # against, not to whoever this turn happened to route to.
                    # When the pinned worker died the two differ, and
                    # decrementing the new worker drives it to -1 -- and
                    # active_sessions leads the tie-break key, so that worker
                    # then wins every tie for the rest of the run.
                    load = self._workers.get(entry.worker_id)
                    if load is not None:
                        load.active_sessions -= 1
                        load.active_session_ids.discard(entry.root_key or routing_key)

        self._track_credit_sent(worker_id, credit.id)

        await self._router_client.send_to(worker_id, credit)

    async def cancel_all_credits(self) -> None:
        """Send cancellation requests to all workers with in-flight credits."""
        # Mark cancellation first, so we suppress warnings for workers that unregister with in-flight credits.
        self._cancellation_pending = True

        # Build up the map of worker_id to credit_ids snapshot to cancel in an atomic way
        # This works because there are no await calls in this loop, they are all done afterwards.
        to_cancel: dict[str, set[int]] = {}
        for worker_load in self._workers_cache:
            if worker_load.in_flight_credits > 0:
                if self.is_debug_enabled:
                    self.debug(
                        f"Worker {worker_load.worker_id} has {worker_load.in_flight_credits} in-flight credits to cancel: {worker_load.active_credit_ids}"
                    )
                # Make sure to use copy of the set to avoid race conditions.
                to_cancel[worker_load.worker_id] = worker_load.active_credit_ids.copy()

        total_cancelled_credits = 0
        for worker_id, credit_ids in to_cancel.items():
            if self.is_debug_enabled:
                self.debug(
                    f"Sending CancelCredits to worker {worker_id} for {len(credit_ids)} credits"
                )

            await self._router_client.send_to(
                worker_id,
                CancelCredits(credit_ids=credit_ids),
            )
            total_cancelled_credits += len(credit_ids)

        if total_cancelled_credits > 0:
            self.info(
                f"Sent cancellation requests for {total_cancelled_credits} in-flight credits across {len(to_cancel)} workers"
            )
        else:
            self.debug("No in-flight credits to cancel")

    def mark_credits_complete(self) -> None:
        """Mark credits complete - suppresses orphan warnings during shutdown."""
        self._credits_complete = True
        self._unavailable_sessions.clear()

    def begin_phase(self, phase: CreditPhase, phase_index: int | None = None) -> None:
        """Reset cancellation state without disturbing another draining phase.

        Seamless phases overlap: the next phase starts issuing while the prior
        phase may still dispatch DAG descendants. Unavailable records therefore
        remain until their owning phase fully drains and calls ``end_phase``.
        """
        self._cancellation_pending = False

    def end_phase(self, phase: CreditPhase, phase_index: int | None = None) -> None:
        """Discard unavailable-session records once this phase fully drains."""
        self._discard_unavailable_for_phase(phase_runtime_key(phase, phase_index))

    def register_child_routing(
        self, parent_correlation_id: str, child_correlation_id: str | None = None
    ) -> None:
        """Increment the sticky-routing refcount for a parent's entry.

        Called by ``BranchOrchestrator`` before dispatching each DAG child so
        the parent's sticky entry survives past its own final turn until every
        descendant child session has terminated. If the parent has no active
        sticky entry we log a warning and continue without raising - the
        child will route via least-loaded selection rather than co-locating
        with the parent's worker, losing prefix-cache locality but not
        breaking correctness.
        """
        entry = self._sticky_sessions.get(parent_correlation_id)
        if entry is not None:
            entry.ref_count += 1
            if child_correlation_id and child_correlation_id not in (
                self._sticky_sessions
            ):
                # Alias so this child's own descendants resolve to the same
                # entry instead of missing at depth >= 2.
                entry.aliases.add(child_correlation_id)
                self._sticky_sessions[child_correlation_id] = entry
        else:
            unavailable = self._unavailable_sessions.get(parent_correlation_id)
            if unavailable is not None:
                unavailable.pending_holders += 1
                if child_correlation_id:
                    unavailable.aliases.add(child_correlation_id)
                    self._unavailable_sessions[child_correlation_id] = unavailable
            self.warning(
                lambda: f"register_child_routing: parent "
                f"{parent_correlation_id!r} has no sticky entry; "
                f"child will not co-locate with parent's worker"
            )

    def release_child_routing(self, parent_correlation_id: str) -> None:
        """Decrement the sticky-routing refcount when a DAG child terminates.

        Called by ``BranchOrchestrator`` when a child session reaches a leaf
        or errors out. If the refcount reaches zero and the parent's own final
        turn has already been observed, the sticky entry is evicted.
        """
        entry = self._sticky_sessions.get(parent_correlation_id)
        if entry is None:
            self._release_unavailable_child(parent_correlation_id)
            return
        entry.ref_count -= 1
        if entry.ref_count <= 0 and entry.parent_final_seen:
            worker_id = entry.worker_id
            root_key = entry.root_key or parent_correlation_id
            self._pop_entry(parent_correlation_id, entry)
            load = self._workers.get(worker_id)
            if load is not None:
                load.active_sessions -= 1
                load.active_session_ids.discard(root_key)

    def evict_unclaimed_sticky(self, parent_correlation_id: str) -> None:
        """Force-pop a sticky entry retained for FORK children that never registered.

        Parent final turns with ``has_forks=True`` keep the sticky entry alive
        (``parent_final_seen=True``, ``ref_count`` already decremented) so
        ``register_child_routing`` can find it. When every child fails before
        that register (e.g. ``start_branch_child`` raises), nothing else
        decrements ``active_sessions``. Called from
        ``BranchOrchestrator._finalize_failed_dispatches``; safe no-op when
        children hold refs or the parent final has not been seen.
        """
        entry = self._sticky_sessions.get(parent_correlation_id)
        if entry is None or not entry.parent_final_seen or entry.ref_count > 0:
            return
        worker_id = entry.worker_id
        root_key = entry.root_key or parent_correlation_id
        self._pop_entry(parent_correlation_id, entry)
        load = self._workers.get(worker_id)
        if load is not None:
            load.active_sessions -= 1
            load.active_session_ids.discard(root_key)

    def _pop_entry(self, key: str, entry: _StickyEntry) -> None:
        """Drop an entry along with its root key and every alias pointing at it.

        Popping only the key the caller held would leave the other names for
        this entry behind, routing later credits to a worker that no longer
        owns the session and growing the dict for the life of the run. ``key``
        is often an alias: DAG eviction paths are handed a descendant's
        correlation id.
        """
        self._sticky_sessions.pop(key, None)
        for alias in (entry.root_key, *entry.aliases):
            if alias and self._sticky_sessions.get(alias) is entry:
                self._sticky_sessions.pop(alias, None)
        entry.aliases.clear()

    def _pop_unavailable(
        self, key: str, record: UnavailableSession | None = None
    ) -> None:
        """Drop one unavailable record under its root and every alias."""
        record = record or self._unavailable_sessions.get(key)
        if record is None:
            return
        self._unavailable_sessions.pop(key, None)
        for alias in (record.root_key, *record.aliases):
            if alias and self._unavailable_sessions.get(alias) is record:
                self._unavailable_sessions.pop(alias, None)
        record.aliases.clear()

    def _discard_unavailable_for_phase(self, phase_key: PhaseRuntimeKey) -> None:
        """Drop every unavailable record owned by one runtime phase instance."""
        records_by_identity = {
            id(record): record
            for record in self._unavailable_sessions.values()
            if record.phase_key == phase_key
        }
        for record in records_by_identity.values():
            self._pop_unavailable(record.root_key, record)

    def _release_unavailable_owner(self, routing_key: str) -> None:
        """Release the owning session's hold on an unavailable routing key."""
        record = self._unavailable_sessions.get(routing_key)
        if record is None or not record.owner_pending:
            return
        record.owner_pending = False
        self._decrement_unavailable_holders(routing_key, record)

    def _release_unavailable_child(self, routing_key: str) -> None:
        """Release one registered DAG child's unavailable-session hold."""
        record = self._unavailable_sessions.get(routing_key)
        if record is not None:
            self._decrement_unavailable_holders(routing_key, record)

    def _decrement_unavailable_holders(
        self, routing_key: str, record: UnavailableSession
    ) -> None:
        record.pending_holders -= 1
        if record.pending_holders <= 0:
            self._pop_unavailable(routing_key, record)

    # =============================================================================
    # Private Methods
    # =============================================================================

    async def _handle_return_pull_message(self, message: WorkerToRouterMessage) -> None:
        """Adapt the identity-less PULL fan-in to the shared handler.

        The PUSH/PULL return channel has no ZMQ envelope identity, so the worker
        id rides inside CreditReturn (FirstToken does not need it). Unpack it and
        delegate to the common handler.

        Ordering note: CreditReturn/FirstToken now arrive on this PULL channel while
        WorkerDispatchable/WorkerShutdown stay on the DEALER, so a worker's returns and its
        lifecycle messages are no longer mutually ordered (on the single bidirectional
        DEALER they were). That is safe because a worker only emits WorkerShutdown
        after all its returns have been sent, and the timing manager's phase /
        cancellation barrier drains outstanding returns before workers are torn down;
        a return therefore cannot legitimately land after its worker's unregister
        outside the teardown window, where ``_cancellation_pending`` /
        ``_credits_complete`` already suppress the ``_warn_missing_worker`` path.
        """
        worker_id = getattr(message, "worker_id", None) or ""
        await self._handle_router_message(worker_id, message)

    def _readmit_on_traffic(
        self, worker_id: str, message: WorkerToRouterMessage
    ) -> None:
        """Undo a staleness eviction before the message is dispatched.

        Traffic from a worker the staleness sweep dropped disproves the eviction
        outright, so the load record is restored before the message is processed
        and its credit accounting lands on the restored record. Departure
        messages are excluded: a worker that is leaving must stay evicted.

        This runs on every CreditReturn and every FirstToken, i.e. twice per
        request, so the empty-dict test comes first: with nothing evicted (the
        overwhelmingly common case, and the only case in a healthy single-node
        run) it costs one truth test instead of hashing the worker id.
        """
        if not self._evicted_workers:
            return
        if worker_id in self._evicted_workers and not isinstance(
            message, WorkerShutdown | WorkerUndispatchable
        ):
            self._readmit_worker(worker_id)

    async def _handle_router_message(
        self, worker_id: str, message: WorkerToRouterMessage
    ) -> None:
        """Handle CreditReturn, FirstToken, WorkerDispatchable, WorkerShutdown from workers."""
        self._readmit_on_traffic(worker_id, message)
        match message:
            case CreditReturn():
                self._track_credit_returned(
                    worker_id,
                    message.credit.id,
                    message.cancelled,
                    message.error is not None,
                )
                if self._on_return_callback:
                    # Await directly instead of execute_async - credit returns release
                    # concurrency slots, so delays here directly impact throughput.
                    await self._on_return_callback(worker_id, message)
            case FirstToken():
                if self._on_first_token_callback:
                    # Forward TTFT to orchestrator so it can release the prefill slot.
                    await self._on_first_token_callback(message)
            case TimePing():
                await self._handle_time_ping(worker_id, message)
            case WorkerConnected():
                # Connectivity is NOT dispatchability. The worker's return path
                # is up, but in Kubernetes its pod-local dataset may not exist
                # yet; registering here would route credits to a worker that
                # fails every one of them. Wait for WorkerDispatchable.
                self._connected_workers.add(worker_id)
            case WorkerDispatchable():
                self._connected_workers.add(worker_id)
                self._register_worker(worker_id)
                self._note_peak_workers()
            case WorkerUndispatchable():
                # Stays connected, just stops receiving new credits.
                self._evicted_workers.pop(worker_id, None)
                self._unregister_worker(worker_id)
            case WorkerShutdown():
                self._evicted_workers.pop(worker_id, None)
                self._connected_workers.discard(worker_id)
                self._unregister_worker(
                    worker_id,
                    session_loss_reason=(
                        "worker_unavailable: worker shut down before next turn"
                    ),
                )
            case _:
                self.warning(f"Unknown message type: {type(message).__name__}")

    async def _handle_time_ping(self, worker_id: str, message: TimePing) -> None:
        """Echo a TimePing back as a TimePong on the credit channel.

        Both fields are echoed verbatim so the worker computes RTT entirely
        against its own clock; the router's clock never enters the measurement,
        which is what makes the baseline immune to cross-machine skew. The
        reply rides the same ROUTER socket credits use, so the measured latency
        reflects the queuing real credits will see.

        Does not register the worker: a probing worker is not yet dispatchable,
        and adding it to the load table here would route credits to a worker
        still in startup.
        """
        await self._router_client.send_to(
            worker_id,
            TimePong(sequence=message.sequence, sent_at_ns=message.sent_at_ns),
        )

    def _register_worker(self, worker_id: str) -> None:
        """Register worker for routing, create WorkerLoad entry.

        Late-joining workers initialize:
        - virtual_sent_credits to average (prevents thundering herd on credits)
        - last_sent_at_ns to current time (prevents winning all timestamp tie-breaks)
        """
        if worker_id not in self._workers:
            # A re-announced worker that the staleness sweep dropped comes back
            # through its stashed load record so its in-flight credits stay
            # accounted for; only a genuinely new worker starts from scratch.
            if worker_id in self._evicted_workers:
                self._readmit_worker(worker_id)
                return
            # Initialize to averages to prevent thundering herd
            avg_virtual = 0
            if self._workers_cache:
                avg_virtual = sum(
                    w.virtual_sent_credits for w in self._workers_cache
                ) // len(self._workers_cache)

            self._workers[worker_id] = WorkerLoad(
                worker_id=worker_id,
                virtual_sent_credits=avg_virtual,
                last_sent_at_ns=time.perf_counter_ns(),
            )
            if self.is_trace_enabled:
                self.trace(
                    f"Worker registered: {worker_id} (total={len(self._workers)}, "
                    f"virtual_credits={avg_virtual})"
                )
            self._workers_cache = list(self._workers.values())
            # We know that new workers are load 0, and load 0 is the absolute minimum load,
            # so we can cheat and just set minimum load to 0 without recalculating.
            self._min_load = 0
            self._workers_by_load[0].add(worker_id)
            self._worker_available_event.set()
            self._notify_worker_count_changed()

    @background_task(
        interval=lambda self: Environment.WORKER.STALE_TIME,
        immediate=False,
    )
    async def _evict_stale_workers_task(self) -> None:
        """Periodically drop workers whose heartbeats have stopped.

        Runs at the same cadence as the staleness window, so a dead worker
        leaves routing within roughly two sweeps. Suppressed once credits are
        complete or a cancellation is in flight: workers legitimately stop
        talking then, and evicting during teardown would log noise about a
        normal shutdown.
        """
        if self._credits_complete or self._cancellation_pending:
            return
        self.evict_stale_workers(Environment.WORKER.STALE_TIME * 3)

    def _note_peak_workers(self) -> None:
        """Track the high-water mark of registered workers.

        The floor is measured against the peak rather than the configured
        count because workers register over time; comparing against the
        request would trip during a normal ramp-up.
        """
        self._peak_worker_count = max(
            getattr(self, "_peak_worker_count", 0), len(self._workers)
        )

    def check_worker_floor(self, min_fraction: float) -> str | None:
        """Return a reason string when too few workers remain, else None."""
        if min_fraction <= 0:
            return None
        peak = getattr(self, "_peak_worker_count", 0)
        if peak <= 0:
            return None
        alive = len(self._workers)
        floor = peak * min_fraction
        if alive >= floor:
            return None
        return (
            f"only {alive} of {peak} worker(s) remain dispatchable "
            f"({alive / peak:.0%} < {min_fraction:.0%} floor)"
        )

    def note_worker_heartbeat(self, worker_id: str) -> None:
        """Record a service heartbeat from this worker, readmitting if evicted.

        The worker service publishes heartbeats on its own timer, independently
        of whatever request it is currently running, so heartbeat freshness is
        the one signal available here that a 60-second decode does not look
        identical to a dead pod. Called by the TimingManager's heartbeat
        subscription.

        A heartbeat from a worker the sweep already dropped proves the eviction
        was wrong, so the worker is put back into routing.
        """
        if load := self._workers.get(worker_id):
            load.last_heartbeat_ns = time.time_ns()
        elif worker_id in self._evicted_workers:
            self._readmit_worker(worker_id)

    def _readmit_worker(self, worker_id: str) -> None:
        """Return a wrongly-evicted worker to the routing pool.

        Its previous ``WorkerLoad`` is restored rather than rebuilt so the
        credits it still holds stay accounted for: a fresh record would report
        ``in_flight_credits == 0`` and every late return would log a spurious
        underflow error. Session state is intentionally NOT restored -- those
        sticky sessions were already converted to ``UnavailableSession`` records
        at eviction time and must not be resurrected.
        """
        load = self._evicted_workers.pop(worker_id, None)
        if load is None or worker_id in self._workers:
            return
        now_ns = time.time_ns()
        load.last_heartbeat_ns = now_ns
        load.active_sessions = 0
        load.active_session_ids = set()
        self._workers[worker_id] = load
        self._connected_workers.add(worker_id)
        self._workers_cache = list(self._workers.values())
        self._workers_by_load[load.in_flight_credits].add(worker_id)
        self._min_load = min(w.in_flight_credits for w in self._workers_cache)
        self._worker_available_event.set()
        self._notify_worker_count_changed()
        self.warning(
            f"Worker {worker_id} spoke again after being evicted as stale; "
            f"returning it to routing with {load.in_flight_credits} credit(s) in flight"
        )

    def evict_stale_workers(self, stale_after_s: float) -> list[str]:
        """Drop workers whose heartbeats have stopped, and stop routing to them.

        A dead worker cannot announce its own death, so nothing else can do
        this. Until it is dropped it keeps winning selections and every credit
        sent to it is never returned, which starves the concurrency limiter --
        throughput degrades with nothing naming the cause.

        Staleness is measured against ``last_heartbeat_ns``, NOT credit-channel
        traffic. Workers emit nothing on the credit channel while a request is
        in flight, so keying off that evicted healthy workers running long
        requests: a reasoning model with a one-second TTFT and a minute of
        decode looks exactly as silent as a crashed pod, and with concurrency
        spread across the pool every busy worker was evicted in turn until
        routing had none left and the run hard-failed. Heartbeats are published
        on the worker's own timer regardless of request duration, so a slow
        worker keeps its clock fresh.

        Returns the evicted worker ids. ``stale_after_s <= 0`` disables the
        check; a worker whose heartbeat has never been observed (nothing wired
        up the subscription, or it has not sent its first one yet) is never
        evicted, so a missing liveness feed degrades to no eviction rather than
        to evicting everybody. Eviction is recoverable: a later heartbeat
        readmits the worker with its in-flight accounting intact.
        """
        if stale_after_s <= 0:
            return []
        cutoff_ns = time.time_ns() - int(stale_after_s * NANOS_PER_SECOND)
        stale = [
            wid
            for wid, load in self._workers.items()
            if load.last_heartbeat_ns and load.last_heartbeat_ns < cutoff_ns
        ]
        for worker_id in stale:
            load = self._workers[worker_id]
            in_flight = load.in_flight_credits
            self.warning(
                f"Worker {worker_id} has not heartbeat for over {stale_after_s:.0f}s "
                f"with {in_flight} credit(s) in flight; dropping it from routing"
            )
            self._evicted_workers[worker_id] = load
            self._connected_workers.discard(worker_id)
            self._unregister_worker(
                worker_id,
                session_loss_reason=(
                    "worker_unavailable: worker stopped responding before next turn"
                ),
            )
        return stale

    def _unregister_worker(
        self, worker_id: str, *, session_loss_reason: str | None = None
    ) -> None:
        """Unregister a worker and retain lost sticky-session state when needed."""
        if worker_load := self._workers.pop(worker_id, None):
            if worker_load.in_flight_credits > 0 and not self._cancellation_pending:
                self.warning(
                    f"Worker {worker_id} unregistered with {worker_load.in_flight_credits} in-flight credits"
                )
            if self.is_trace_enabled:
                self.trace(
                    f"Worker unregistered: {worker_id} (remaining={len(self._workers)})"
                )
            self._workers_by_load[worker_load.in_flight_credits].discard(worker_id)
            if not self._workers:
                self._worker_available_event.clear()

            # Remove all orphaned sticky sessions now and warn once up front
            orphaned_session_ids = worker_load.active_session_ids
            if orphaned_session_ids and not (
                self._cancellation_pending or self._credits_complete
            ):
                self.warning(
                    f"Worker {worker_id} unregistered with "
                    f"{len(orphaned_session_ids)} active sessions"
                )
            self._preserve_orphaned_sessions(
                worker_id, orphaned_session_ids, session_loss_reason
            )

        if not worker_load:
            # Warn but continue - may happen if shutdown message arrives before ready message.
            self.warning(
                f"Worker {worker_id} not found when unregistering. This should not happen."
            )

        self._workers_cache = list(self._workers.values())

        if not worker_load or (
            worker_load.in_flight_credits == self._min_load
            and len(self._workers_by_load[self._min_load]) == 0
        ):
            # Recalculate min_load if the removed worker was the last at the current minimum.
            if len(self._workers_cache) > 0:
                self._min_load = min(w.in_flight_credits for w in self._workers_cache)
            else:
                self._min_load = 0
        self._notify_worker_count_changed()

    def _notify_worker_count_changed(self) -> None:
        """Tell TimingManager that the dispatchable-worker count changed."""
        if callback := getattr(self, "_on_worker_count_changed", None):
            callback(len(self._workers))

    def _preserve_orphaned_sessions(
        self,
        worker_id: str,
        orphaned_session_ids: set[str],
        session_loss_reason: str | None,
    ) -> None:
        """Retain routing aliases for sessions that cannot safely migrate."""
        for x_correlation_id in orphaned_session_ids:
            orphaned = self._sticky_sessions.get(x_correlation_id)
            if orphaned is None:
                continue
            if session_loss_reason is not None:
                root_key = orphaned.root_key or x_correlation_id
                unavailable = UnavailableSession(
                    worker_id=worker_id,
                    reason=session_loss_reason,
                    phase_key=orphaned.phase_key,
                    pending_holders=orphaned.ref_count,
                    owner_pending=not orphaned.parent_final_seen,
                    root_key=root_key,
                    aliases=set(orphaned.aliases),
                )
                for key in (root_key, *unavailable.aliases):
                    self._unavailable_sessions[key] = unavailable
            self._pop_entry(x_correlation_id, orphaned)

    def _track_credit_sent(self, worker_id: str, credit_id: int) -> None:
        """Update worker load: increment in_flight_credits. Lock-free."""
        if worker_load := self._workers.get(worker_id):
            old_load = worker_load.in_flight_credits

            worker_load.total_sent_credits += 1
            worker_load.virtual_sent_credits += 1
            worker_load.in_flight_credits += 1
            worker_load.active_credit_ids.add(credit_id)
            worker_load.last_sent_at_ns = time.perf_counter_ns()

            new_load = worker_load.in_flight_credits
            # Keep the workers by load updated for faster load balancing.
            self._workers_by_load[old_load].discard(worker_id)
            self._workers_by_load[new_load].add(worker_id)

            if old_load == self._min_load and len(self._workers_by_load[old_load]) == 0:
                # We only send credits one at a time, so if this worker was the last at the minimum load,
                # it is safe to assume that the new minimum load is this worker's new load. Saving a recalculation.
                self._min_load = new_load

        else:
            self._warn_missing_worker(worker_id, "sent")

    def _track_credit_returned(
        self, worker_id: str, credit_id: int, cancelled: bool, error_reported: bool
    ) -> None:
        """Update worker load: decrement in_flight_credits. Lock-free."""
        if worker_load := self._workers.get(worker_id):
            worker_load.active_credit_ids.discard(credit_id)

            if cancelled:
                worker_load.total_cancelled_credits += 1
            else:
                worker_load.total_completed_credits += 1
            if error_reported:
                worker_load.total_errors_reported += 1

            old_load = worker_load.in_flight_credits
            if worker_load.in_flight_credits > 0:
                worker_load.in_flight_credits -= 1
                new_load = worker_load.in_flight_credits

                self._workers_by_load[old_load].discard(worker_id)
                self._workers_by_load[new_load].add(worker_id)
                if new_load < self._min_load:
                    self._min_load = new_load
            else:
                self.error(
                    f"Worker {worker_id} in_flight_credits already 0 when tracking returned credit {credit_id}"
                )
        else:
            self._warn_missing_worker(worker_id, "returned")

    def _warn_missing_worker(self, worker_id: str, credit_action: str) -> None:
        """Warn if worker is missing when tracking credit sent or returned."""
        if self._cancellation_pending:
            # Even during cancellation, the workers should still be registered, but if they are not it won't cause any issues.
            self.warning(
                f"Worker {worker_id} not found when tracking credit {credit_action} during cancellation."
            )
        else:
            self.error(
                f"Worker {worker_id} not found when tracking credit {credit_action}. This should not happen."
            )
