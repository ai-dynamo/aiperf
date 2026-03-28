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

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.config.zmq import ZMQDualBindConfig

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CommAddress
from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task
from aiperf.common.mixins import CommunicationMixin
from aiperf.common.protocols import StreamingRouterClientProtocol
from aiperf.common.utils import yield_to_event_loop
from aiperf.credit.messages import (
    CancelCredits,
    CreditReturn,
    FirstToken,
    InFlightReconciliation,
    InFlightReport,
    TimePing,
    TimePong,
    WorkerConnected,
    WorkerDispatchable,
    WorkerShutdown,
    WorkerToRouterMessage,
    WorkerUndispatchable,
)
from aiperf.credit.structs import Credit

# =============================================================================
# Data Models
# =============================================================================


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
    active_credit_ids: set[int] = field(default_factory=set)
    active_credits: dict[int, Credit] = field(default_factory=dict)
    active_sessions: int = 0  # Sticky sessions assigned to this worker
    active_session_ids: set[str] = field(default_factory=set)
    last_sent_at_ns: int = (
        0  # For tie-breaking (guaranteed unique in single-threaded asyncio)
    )


@dataclass(slots=True, frozen=True)
class UnavailableSession:
    """A session whose sticky worker became unavailable before the next turn."""

    worker_id: str
    reason: str


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

    async def cancel_all_credits(self) -> None:
        """Cancel all in-flight credits.

        Used during phase timeout or system shutdown.
        """
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

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str,
        **kwargs,
    ) -> None:
        super().__init__(run=run, service_id=service_id, **kwargs)

        # For dual-bind mode (Kubernetes), also bind to TCP for remote workers.
        # Controller services use IPC (fast, same-pod) but workers connect via TCP.
        # Only bind to TCP if we're in controller mode (controller_host not set).
        additional_bind_address: str | None = None
        additional_return_bind_address: str | None = None
        comm_config = self.run.resolved.comm_config or self.run.cfg.comm_config
        if (
            isinstance(comm_config, ZMQDualBindConfig)
            and not comm_config.controller_host
        ):
            additional_bind_address = comm_config.credit_router_tcp_bind_address
            additional_return_bind_address = (
                comm_config.credit_return_router_tcp_bind_address
            )
            self.info(
                f"Dual-bind mode: credit router will also bind to {additional_bind_address}"
            )
            self.info(
                f"Dual-bind mode: credit return router will also bind to {additional_return_bind_address}"
            )

        # Credit channel: Router -> Worker only (Credit, CancelCredits).
        # Send-only from the router's perspective. Workers connect DEALERs
        # to receive credits but never send on this channel.
        self._credit_router_client: StreamingRouterClientProtocol = (
            self.comms.create_streaming_router_client(
                address=CommAddress.CREDIT_ROUTER,
                bind=True,
                additional_bind_address=additional_bind_address,
            )
        )

        # Return channel: Worker -> Router (CreditReturn, FirstToken,
        # WorkerConnected, WorkerDispatchable, WorkerUndispatchable,
        # WorkerShutdown, TimePing). Router replies with CreditAck / TimePong.
        self._return_router_client: StreamingRouterClientProtocol = (
            self.comms.create_streaming_router_client(
                address=CommAddress.CREDIT_RETURN_ROUTER,
                bind=True,
                additional_bind_address=additional_return_bind_address,
            )
        )
        self._return_router_client.register_receiver(self._handle_return_router_message)

        self._on_return_callback: (
            Callable[[str, CreditReturn], Awaitable[None]] | None
        ) = None
        self._on_first_token_callback: (
            Callable[[FirstToken], Awaitable[None]] | None
        ) = None

        # Sticky sessions: x_correlation_id -> worker_id
        # Routes all turns of a conversation to the same worker. Required because
        # workers cache UserSession state by x_correlation_id.
        self._sticky_sessions: dict[str, str] = {}
        self._unavailable_sessions: dict[str, UnavailableSession] = {}

        self._cancellation_pending: bool = False
        self._credits_complete: bool = False

        # Snapshot list for iteration - avoids dict.values() overhead in hot path.
        # Rebuilt on dispatchable worker add/remove (rare) to keep routing fast (common).
        self._workers_cache: list[WorkerLoad] = []
        # Dispatchable workers only. Connected-but-undispatchable workers are tracked
        # separately in _connected_workers and excluded from routing structures.
        self._workers: dict[str, WorkerLoad] = {}
        self._connected_workers: set[str] = set()
        self._initializing_workers: set[str] = set()

        # Map load level -> set of worker_ids at that load (O(1) add/remove)
        self._workers_by_load: dict[int, set[str]] = defaultdict(set)
        # Keep track of the minimum load to avoid recalculating it on every credit sent O(1) vs O(n)
        self._min_load: int = 0

        # Reconciliation state: detect orphaned credits via two-consecutive-misses
        self._first_token_received: set[tuple[str, int]] = set()
        self._pending_reconciliation: dict[str, frozenset[int]] = {}
        self._missed_reconciliation_cycles: dict[str, int] = defaultdict(int)
        self._suspected_orphans: dict[str, set[int]] = defaultdict(set)
        self._detached_workers: dict[str, WorkerLoad] = {}
        self._detached_worker_deadlines_ns: dict[str, int] = {}
        self._detached_reclaim_tasks: dict[str, asyncio.Task[None]] = {}
        self._reclaimed_credit_ids: set[tuple[str, int]] = set()

    # =============================================================================
    # Public Methods
    # =============================================================================

    def set_return_callback(
        self, callback: Callable[[str, CreditReturn], Awaitable[None]]
    ) -> None:
        """Set callback for credit returns (enables concurrency control)."""
        self._on_return_callback = callback

    def set_first_token_callback(
        self, callback: Callable[[FirstToken], Awaitable[None]]
    ) -> None:
        """Set callback for first token events (enables prefill concurrency release)."""
        self._on_first_token_callback = callback

    async def send_credit(self, credit: Credit) -> None:
        """Determine the worker based on sticky sessions or least-loaded and send the credit to the worker.

        This method:
        - Determines the worker based on sticky sessions or least-loaded
        - Updates the worker load and sticky sessions
        - Sends the credit to the worker
        """
        if not credit.x_correlation_id:
            raise RuntimeError("x_correlation_id must be set in Credit")

        x_correlation_id = credit.x_correlation_id
        sticky_worker_id = self._sticky_sessions.get(x_correlation_id)
        unavailable_session = self._unavailable_sessions.get(x_correlation_id)

        if unavailable_session and not credit.allow_worker_migration:
            self._unavailable_sessions.pop(x_correlation_id, None)
            if self._on_return_callback:
                await self._on_return_callback(
                    unavailable_session.worker_id,
                    CreditReturn(
                        credit=credit,
                        cancelled=True,
                        first_token_sent=False,
                        error=unavailable_session.reason,
                    ),
                )
            return

        if not self._workers:
            raise RuntimeError("No dispatchable workers available for routing")

        # Use existing sticky session if worker still valid
        if sticky_worker_id and sticky_worker_id in self._workers:
            worker_id = sticky_worker_id
        else:
            worker_id = self._select_least_loaded_worker_id()
            self._unavailable_sessions.pop(x_correlation_id, None)

            # Only create sticky session if there are more turns coming. Single-turn
            # conversations don't need routing state since there's no next turn.
            if not credit.is_final_turn:
                self._sticky_sessions[x_correlation_id] = worker_id
                load = self._workers[worker_id]
                load.active_sessions += 1
                load.active_session_ids.add(x_correlation_id)

        # Cleanup on final turn - only decrement if session was actually tracked
        # (single-turn sessions never get added to _sticky_sessions)
        if credit.is_final_turn and self._sticky_sessions.pop(x_correlation_id, None):
            load = self._workers[worker_id]
            load.active_sessions -= 1
            load.active_session_ids.discard(x_correlation_id)

        self._track_credit_sent(worker_id, credit)

        await self._credit_router_client.send_to(worker_id, credit)

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
        for sent_count, (worker_id, credit_ids) in enumerate(
            to_cancel.items(), start=1
        ):
            if self.is_debug_enabled:
                self.debug(
                    f"Sending CancelCredits to worker {worker_id} for {len(credit_ids)} credits"
                )

            await self._credit_router_client.send_to(
                worker_id,
                CancelCredits(credit_ids=credit_ids),
            )
            total_cancelled_credits += len(credit_ids)
            if sent_count % 50 == 0:
                await yield_to_event_loop()

        if total_cancelled_credits > 0:
            self.info(
                f"Sent cancellation requests for {total_cancelled_credits} in-flight credits across {len(to_cancel)} workers"
            )
        else:
            self.debug("No in-flight credits to cancel")

    def mark_credits_complete(self) -> None:
        """Mark credits complete - suppresses orphan warnings during shutdown."""
        self._credits_complete = True

    # =============================================================================
    # Private Methods
    # =============================================================================

    async def _handle_return_router_message(
        self, worker_id: str, message: WorkerToRouterMessage
    ) -> None:
        """Handle all worker -> router messages on the return channel.

        All worker-initiated messages arrive here. TimePong is sent back
        on the credit channel so both channels are truly unidirectional.
        """
        match message:
            case CreditReturn():
                if (
                    self._credit_id_key(message.credit.phase, message.credit.id)
                    in self._reclaimed_credit_ids
                ):
                    self.debug(
                        lambda: (
                            f"Ignoring late CreditReturn for reclaimed credit "
                            f"{message.credit.id} from {worker_id}"
                        )
                    )
                    return

                was_detached = worker_id in self._detached_workers
                self._track_credit_returned(
                    worker_id,
                    message.credit.id,
                    message.cancelled,
                    message.error is not None,
                    phase=message.credit.phase,
                )
                if was_detached and (
                    message.credit.is_final_turn
                    or not message.credit.allow_worker_migration
                ):
                    self._unavailable_sessions.pop(
                        message.credit.x_correlation_id, None
                    )
                if self._on_return_callback:
                    callback_message = message
                    if was_detached:
                        callback_message = CreditReturn(
                            credit=message.credit,
                            cancelled=message.cancelled,
                            first_token_sent=message.first_token_sent,
                            error=message.error,
                            worker_detached=True,
                        )
                    # Await directly instead of execute_async - credit returns release
                    # concurrency slots, so delays here directly impact throughput.
                    await self._on_return_callback(worker_id, callback_message)
                self._cleanup_detached_worker_if_drained(worker_id)
            case FirstToken():
                if (
                    self._credit_id_key(message.phase, message.credit_id)
                    in self._reclaimed_credit_ids
                ):
                    self.debug(
                        lambda: (
                            f"Ignoring late FirstToken for reclaimed credit "
                            f"{message.credit_id} from {worker_id}"
                        )
                    )
                    return

                self._first_token_received.add(
                    self._credit_id_key(message.phase, message.credit_id)
                )
                if self._on_first_token_callback:
                    await self._on_first_token_callback(message)
            case InFlightReport():
                await self._handle_reconciliation_report(worker_id, message)
            case TimePing():
                self._initializing_workers.add(worker_id)
                # Reply on the credit channel so both channels stay unidirectional.
                # RTT measurement spans both sockets, which is fine for clock offset.
                await self._credit_router_client.send_to(
                    worker_id,
                    TimePong(sequence=message.sequence, sent_at_ns=message.sent_at_ns),
                )
            case WorkerConnected():
                self._connected_workers.add(worker_id)
            case WorkerDispatchable():
                if detached := self._detached_workers.get(worker_id):
                    self._register_worker(worker_id)
                    await self._reclaim_detached_worker_credits(
                        worker_id,
                        detached,
                        "worker_unavailable: replacement worker registered before detached credits drained",
                    )
                else:
                    self._register_worker(worker_id)
            case WorkerUndispatchable():
                if worker_id in self._workers:
                    self._unregister_worker(worker_id)
            case WorkerShutdown():
                self._connected_workers.discard(worker_id)
                if worker_id in self._workers:
                    worker_load = self._unregister_worker(
                        worker_id,
                        session_loss_reason=(
                            "worker_unavailable: worker shut down before next turn"
                        ),
                    )
                    self._detach_worker(worker_id, worker_load)
                elif worker_id in self._initializing_workers:
                    self._initializing_workers.discard(worker_id)
                    self.info(
                        f"Worker {worker_id} shut down before becoming dispatchable"
                    )
                elif worker_id in self._detached_workers:
                    self.debug(
                        lambda: f"Worker {worker_id} sent duplicate shutdown while detached"
                    )
                else:
                    self._unregister_worker(worker_id)
            case _:
                self.warning(f"Unknown message type: {type(message).__name__}")

    def _register_worker(self, worker_id: str) -> None:
        """Register worker for routing, create WorkerLoad entry.

        Late-joining workers initialize:
        - virtual_sent_credits to average (prevents thundering herd on credits)
        - last_sent_at_ns to current time (prevents winning all timestamp tie-breaks)
        """
        self._initializing_workers.discard(worker_id)
        if worker_id not in self._workers:
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

    def _unregister_worker(
        self,
        worker_id: str,
        *,
        session_loss_reason: str | None = None,
    ) -> WorkerLoad | None:
        """Unregister worker. Sticky sessions are cleared and reassigned on next access."""
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

            # Remove all orphaned sticky sessions now and warn once up front
            orphaned_session_ids = worker_load.active_session_ids.copy()
            if orphaned_session_ids and not (
                self._cancellation_pending or self._credits_complete
            ):
                self.warning(
                    f"Worker {worker_id} unregistered with {len(orphaned_session_ids)} active sessions"
                )
            for x_correlation_id in orphaned_session_ids:
                self._sticky_sessions.pop(x_correlation_id, None)
                if session_loss_reason is not None:
                    self._unavailable_sessions[x_correlation_id] = UnavailableSession(
                        worker_id=worker_id,
                        reason=session_loss_reason,
                    )
            if orphaned_session_ids:
                worker_load.active_sessions = 0
                worker_load.active_session_ids.clear()

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

        self._pending_reconciliation.pop(worker_id, None)
        self._missed_reconciliation_cycles.pop(worker_id, None)
        self._suspected_orphans.pop(worker_id, None)
        return worker_load

    def _track_credit_sent(self, worker_id: str, credit: Credit) -> None:
        """Update worker load: increment in_flight_credits. Lock-free."""
        self._reclaimed_credit_ids.discard(self._credit_id_key(credit.phase, credit.id))
        if worker_load := self._workers.get(worker_id):
            old_load = worker_load.in_flight_credits

            worker_load.total_sent_credits += 1
            worker_load.virtual_sent_credits += 1
            worker_load.in_flight_credits += 1
            worker_load.active_credit_ids.add(credit.id)
            worker_load.active_credits[credit.id] = credit
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
        self,
        worker_id: str,
        credit_id: int,
        cancelled: bool,
        error_reported: bool,
        *,
        phase: object = "profiling",
    ) -> None:
        """Update worker load: decrement in_flight_credits. Lock-free."""
        if worker_load := self._workers.get(worker_id):
            self._apply_credit_return(
                worker_id,
                worker_load,
                credit_id,
                cancelled,
                error_reported,
                phase=phase,
                update_load_index=True,
            )
        elif worker_load := self._detached_workers.get(worker_id):
            self._apply_credit_return(
                worker_id,
                worker_load,
                credit_id,
                cancelled,
                error_reported,
                phase=phase,
                update_load_index=False,
            )
        else:
            self._warn_missing_worker(worker_id, "returned")

    # =============================================================================
    # Reconciliation
    # =============================================================================

    @background_task(
        immediate=False,
        interval=Environment.TIMING.RECONCILIATION_INTERVAL,
    )
    async def _send_reconciliation(self) -> None:
        """Send InFlightReconciliation to each worker with in-flight credits.

        Skips workers that already have a pending reconciliation (prevents stacking).
        Runs periodically as a background task.
        """
        if self._credits_complete or self._cancellation_pending:
            await self._reclaim_expired_detached_workers()
            return

        sent_count = 0
        for worker_load in self._workers_cache:
            worker_id = worker_load.worker_id
            if worker_load.in_flight_credits == 0 and worker_load.active_sessions == 0:
                # Clear stale reconciliation state for idle workers
                self._pending_reconciliation.pop(worker_id, None)
                self._suspected_orphans.pop(worker_id, None)
                self._missed_reconciliation_cycles.pop(worker_id, None)
                continue
            if worker_id in self._pending_reconciliation:
                self._missed_reconciliation_cycles[worker_id] += 1
                if self._missed_reconciliation_cycles[worker_id] >= 2:
                    self.warning(
                        f"Worker {worker_id} missed 2 reconciliation cycles "
                        f"with {worker_load.in_flight_credits} in-flight credits "
                        f"and {worker_load.active_sessions} active sessions"
                    )
                    detached = self._unregister_worker(
                        worker_id,
                        session_loss_reason=(
                            "worker_unavailable: worker missed 2 reconciliation cycles before next turn"
                        ),
                    )
                    await self._reclaim_detached_worker_credits(
                        worker_id,
                        detached,
                        "worker_unavailable: worker missed 2 reconciliation cycles",
                    )
                continue

            credit_ids = frozenset(worker_load.active_credit_ids)
            self._pending_reconciliation[worker_id] = credit_ids
            await self._credit_router_client.send_to(
                worker_id,
                InFlightReconciliation(credit_ids=credit_ids),
            )
            sent_count += 1
            # ZMQ NOBLOCK sends complete synchronously when HWM is unlimited,
            # so the await above never actually yields. Yield periodically to
            # prevent starving the event loop during large worker counts.
            if sent_count % 50 == 0:
                await yield_to_event_loop()

        await self._reclaim_expired_detached_workers()

    async def _handle_reconciliation_report(
        self, worker_id: str, report: InFlightReport
    ) -> None:
        """Compare worker's reported in-flight credits against what we sent.

        Two-consecutive-misses: a credit must be missing from two consecutive
        reports before it is treated as orphaned. This eliminates false positives
        from messages in transit between channels.
        """
        sent_set = self._pending_reconciliation.pop(worker_id, None)
        if sent_set is None:
            self.warning(
                f"Received InFlightReport from {worker_id} with no pending reconciliation"
            )
            return

        self._missed_reconciliation_cycles.pop(worker_id, None)

        reported_set = report.credit_ids
        missing = sent_set - reported_set

        worker_suspects = self._suspected_orphans.get(worker_id)

        if not missing:
            # All clear — drop any prior suspicions for this worker
            if worker_suspects:
                self._suspected_orphans.pop(worker_id, None)
            return

        # Check which missing credits are confirmed (second consecutive miss)
        confirmed_orphans: set[int] = set()
        new_suspects: set[int] = set()
        for credit_id in missing:
            if worker_suspects and credit_id in worker_suspects:
                confirmed_orphans.add(credit_id)
            else:
                new_suspects.add(credit_id)

        # Update suspected set: only keep new suspects (confirmed ones get acted on)
        if new_suspects:
            self._suspected_orphans[worker_id] = new_suspects
        else:
            self._suspected_orphans.pop(worker_id, None)

        # Act on confirmed orphans
        for credit_id in confirmed_orphans:
            await self._handle_orphaned_credit(worker_id, credit_id)

    async def _handle_orphaned_credit(self, worker_id: str, credit_id: int) -> None:
        """Handle a confirmed orphaned credit by synthesizing a CreditReturn.

        The credit was missing from the worker's report for two consecutive
        reconciliation cycles, meaning it was either never received or its
        return was lost.
        """
        worker_load = self._workers.get(worker_id)
        if not worker_load:
            return

        credit = worker_load.active_credits.get(credit_id)
        if credit is None:
            # Already returned between report and handling
            return

        self.warning(
            f"Orphaned credit {credit_id} on worker {worker_id} "
            f"(missing for 2 consecutive reconciliation cycles)"
        )

        key = self._credit_id_key(credit.phase, credit.id)
        first_token_sent = key in self._first_token_received
        self._first_token_received.discard(key)
        self._reclaimed_credit_ids.add(self._credit_id_key(credit.phase, credit.id))

        self._track_credit_returned(
            worker_id,
            credit_id,
            cancelled=True,
            error_reported=True,
            phase=credit.phase,
        )
        if credit.is_final_turn or not credit.allow_worker_migration:
            self._unavailable_sessions.pop(credit.x_correlation_id, None)

        if self._on_return_callback:
            synthetic_return = CreditReturn(
                credit=credit,
                cancelled=True,
                first_token_sent=first_token_sent,
                error="worker_unavailable: missing from worker for 2 consecutive reconciliation cycles",
            )
            await self._on_return_callback(worker_id, synthetic_return)

    @staticmethod
    def _credit_id_key(phase: object, credit_id: int) -> tuple[str, int]:
        """Stable key for deduplicating reclaimed credits."""
        return (str(phase), credit_id)

    async def _reclaim_detached_worker_credits(
        self,
        worker_id: str,
        worker_load: WorkerLoad | None,
        reason: str,
    ) -> None:
        """Synthesize cancelled returns for credits stranded on an unavailable worker."""
        if not worker_load or not worker_load.active_credits:
            self._detached_workers.pop(worker_id, None)
            self._detached_worker_deadlines_ns.pop(worker_id, None)
            self._cancel_detached_reclaim_task(worker_id)
            return

        stranded_credits = list(worker_load.active_credits.values())
        self.warning(
            f"Reclaiming {len(stranded_credits)} in-flight credits from worker "
            f"{worker_id}: {reason}"
        )

        synthetic_returns: list[CreditReturn] = []
        for credit in stranded_credits:
            worker_load.active_credit_ids.discard(credit.id)
            worker_load.active_credits.pop(credit.id, None)
            if worker_load.in_flight_credits > 0:
                worker_load.in_flight_credits -= 1
            worker_load.total_cancelled_credits += 1
            worker_load.total_errors_reported += 1

            key = self._credit_id_key(credit.phase, credit.id)
            first_token_sent = key in self._first_token_received
            self._first_token_received.discard(key)
            self._reclaimed_credit_ids.add(key)
            if credit.is_final_turn or not credit.allow_worker_migration:
                self._unavailable_sessions.pop(credit.x_correlation_id, None)

            if self._on_return_callback:
                synthetic_returns.append(
                    CreditReturn(
                        credit=credit,
                        cancelled=True,
                        first_token_sent=first_token_sent,
                        error=reason,
                    )
                )

        self._detached_workers.pop(worker_id, None)
        self._detached_worker_deadlines_ns.pop(worker_id, None)
        self._cancel_detached_reclaim_task(worker_id)

        for synthetic_return in synthetic_returns:
            await self._on_return_callback(worker_id, synthetic_return)

    def _apply_credit_return(
        self,
        worker_id: str,
        worker_load: WorkerLoad,
        credit_id: int,
        cancelled: bool,
        error_reported: bool,
        *,
        phase: object = "profiling",
        update_load_index: bool,
    ) -> None:
        """Update worker bookkeeping for a returned credit."""
        worker_load.active_credit_ids.discard(credit_id)
        worker_load.active_credits.pop(credit_id, None)
        self._first_token_received.discard(self._credit_id_key(phase, credit_id))

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

            if update_load_index:
                self._workers_by_load[old_load].discard(worker_id)
                self._workers_by_load[new_load].add(worker_id)
                if new_load < self._min_load:
                    self._min_load = new_load

            if new_load == 0:
                self._pending_reconciliation.pop(worker_id, None)
                self._missed_reconciliation_cycles.pop(worker_id, None)
                self._suspected_orphans.pop(worker_id, None)
        else:
            self.error(
                f"Worker {worker_id} in_flight_credits already 0 when tracking returned credit {credit_id}"
            )

    def _detach_worker(self, worker_id: str, worker_load: WorkerLoad | None) -> None:
        """Hold detached workers briefly so late returns can drain cleanly."""
        if not worker_load or worker_load.in_flight_credits == 0:
            return

        grace_timeout_sec = Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT
        grace_timeout_ns = int(grace_timeout_sec * NANOS_PER_SECOND)
        self._detached_workers[worker_id] = worker_load
        self._detached_worker_deadlines_ns[worker_id] = (
            time.perf_counter_ns() + grace_timeout_ns
        )
        self._cancel_detached_reclaim_task(worker_id)
        self._detached_reclaim_tasks[worker_id] = self.execute_async(
            self._wait_and_reclaim_detached_worker(worker_id, grace_timeout_sec)
        )
        self.warning(
            f"Worker {worker_id} shut down with {worker_load.in_flight_credits} "
            f"in-flight credits; waiting {grace_timeout_sec:.1f}s "
            "for late returns before reclaiming"
        )

    def _cleanup_detached_worker_if_drained(self, worker_id: str) -> None:
        """Forget detached workers once all in-flight credits have drained."""
        worker_load = self._detached_workers.get(worker_id)
        if worker_load and worker_load.in_flight_credits == 0:
            self._detached_workers.pop(worker_id, None)
            self._detached_worker_deadlines_ns.pop(worker_id, None)
            self._cancel_detached_reclaim_task(worker_id)

    async def _wait_and_reclaim_detached_worker(
        self, worker_id: str, delay_sec: float
    ) -> None:
        """Reclaim a detached worker once its grace period expires."""
        await asyncio.sleep(delay_sec)
        worker_load = self._detached_workers.get(worker_id)
        await self._reclaim_detached_worker_credits(
            worker_id,
            worker_load,
            "worker_unavailable: worker shut down before returning in-flight credits",
        )

    def _cancel_detached_reclaim_task(self, worker_id: str) -> None:
        """Cancel any scheduled reclaim task for a detached worker."""
        task = self._detached_reclaim_tasks.pop(worker_id, None)
        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def _reclaim_expired_detached_workers(self) -> None:
        """Reclaim credits from detached workers whose grace window expired."""
        if not self._detached_worker_deadlines_ns:
            return

        now_ns = time.perf_counter_ns()
        for worker_id, deadline_ns in list(self._detached_worker_deadlines_ns.items()):
            if deadline_ns > now_ns:
                continue

            worker_load = self._detached_workers.get(worker_id)
            await self._reclaim_detached_worker_credits(
                worker_id,
                worker_load,
                "worker_unavailable: worker shut down before returning in-flight credits",
            )

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

    def _select_least_loaded_worker_id(self) -> str:
        """Select the least-loaded worker using the router's fairness tie-breakers."""
        least_loaded_workers = self._workers_by_load[self._min_load]
        if len(least_loaded_workers) == 1:
            return least_loaded_workers.pop()

        best_worker_id = None
        best_load_key = None
        for worker_id in least_loaded_workers:
            load = self._workers[worker_id]
            load_key = (
                load.active_sessions,
                load.virtual_sent_credits,
                load.last_sent_at_ns,
            )
            if best_load_key is None or load_key < best_load_key:
                best_load_key = load_key
                best_worker_id = worker_id

        return best_worker_id
