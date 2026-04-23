# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sticky credit router with fair load balancing.

Routes credits to workers: sticky routing for multi-turn sessions,
least-loaded selection for first turns. Lock-free via asyncio serialization.

Terminology:
    session: A unique execution of a conversation template, identified by
        x_correlation_id (UUID). All turns in a session route to the same worker.
    conversation_id: Template ID from the dataset (can be reused across sessions).

The router class lives here. Data models (``WorkerLoad``, ``UnavailableSession``,
``CreditRouterProtocol``) live in ``_router_types``; reconciliation and
detached-worker helpers live in ``_router_reconciliation``. Both are re-exported
below so existing imports like ``from aiperf.credit.sticky_router import
WorkerLoad`` keep working.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from aiperf.common.enums import CommAddress
from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task
from aiperf.common.mixins import CommunicationMixin
from aiperf.common.protocols import StreamingRouterClientProtocol
from aiperf.common.utils import yield_to_event_loop
from aiperf.config.zmq import ZMQDualBindConfig
from aiperf.credit._router_reconciliation import _ReconciliationMixin
from aiperf.credit._router_types import (
    CreditRouterProtocol,
    UnavailableSession,
    WorkerLoad,
)
from aiperf.credit._router_workers import _WorkersMixin
from aiperf.credit.messages import (
    CancelCredits,
    CreditReturn,
    FirstToken,
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

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun

__all__ = [
    "CreditRouterProtocol",
    "StickyCreditRouter",
    "UnavailableSession",
    "WorkerLoad",
]


class StickyCreditRouter(_WorkersMixin, _ReconciliationMixin, CommunicationMixin):
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
        self._init_router_clients()
        self._init_router_state()
        self._init_reconciliation_state()

    def _init_router_clients(self) -> None:
        """Build both router clients and register the return-channel receiver."""
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

    def _init_router_state(self) -> None:
        """Initialize routing tables, sticky-session maps, and load indexes."""
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

    def _init_reconciliation_state(self) -> None:
        """Initialize reconciliation, orphan, and detached-worker bookkeeping."""
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
    # Return-channel dispatch
    # =============================================================================

    async def _handle_return_router_message(
        self, worker_id: str, message: WorkerToRouterMessage
    ) -> None:
        """Dispatch all worker -> router messages on the return channel.

        TimePong is sent back on the credit channel so both channels stay
        truly unidirectional.
        """
        match message:
            case CreditReturn():
                await self._handle_credit_return(worker_id, message)
            case FirstToken():
                await self._handle_first_token(worker_id, message)
            case InFlightReport():
                await self._handle_reconciliation_report(worker_id, message)
            case TimePing():
                await self._handle_time_ping(worker_id, message)
            case WorkerConnected():
                self._connected_workers.add(worker_id)
            case WorkerDispatchable():
                await self._handle_worker_dispatchable(worker_id)
            case WorkerUndispatchable():
                if worker_id in self._workers:
                    self._unregister_worker(worker_id)
            case WorkerShutdown():
                self._handle_worker_shutdown(worker_id)
            case _:
                self.warning(f"Unknown message type: {type(message).__name__}")

    async def _handle_credit_return(
        self, worker_id: str, message: CreditReturn
    ) -> None:
        """Handle a CreditReturn from a worker, routing through the return callback."""
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
            message.credit.is_final_turn or not message.credit.allow_worker_migration
        ):
            self._unavailable_sessions.pop(message.credit.x_correlation_id, None)
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

    async def _handle_first_token(self, worker_id: str, message: FirstToken) -> None:
        """Handle a FirstToken event from a worker."""
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

    async def _handle_time_ping(self, worker_id: str, message: TimePing) -> None:
        """Reply to a TimePing on the credit channel so channels stay unidirectional."""
        self._initializing_workers.add(worker_id)
        # RTT measurement spans both sockets, which is fine for clock offset.
        await self._credit_router_client.send_to(
            worker_id,
            TimePong(sequence=message.sequence, sent_at_ns=message.sent_at_ns),
        )

    async def _handle_worker_dispatchable(self, worker_id: str) -> None:
        """Register a (possibly previously-detached) worker as dispatchable."""
        detached = self._detached_workers.get(worker_id)
        self._register_worker(worker_id)
        if detached:
            await self._reclaim_detached_worker_credits(
                worker_id,
                detached,
                "worker_unavailable: replacement worker registered before detached credits drained",
            )

    def _handle_worker_shutdown(self, worker_id: str) -> None:
        """Handle a WorkerShutdown by unregistering and possibly detaching the worker."""
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
            self.info(f"Worker {worker_id} shut down before becoming dispatchable")
        elif worker_id in self._detached_workers:
            self.debug(
                lambda: f"Worker {worker_id} sent duplicate shutdown while detached"
            )
        else:
            self._unregister_worker(worker_id)

    # =============================================================================
    # Reconciliation (background task + helpers defined on the mixin)
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
        await self._send_reconciliation_impl()

    @staticmethod
    def _credit_id_key(phase: object, credit_id: int) -> tuple[str, int]:
        """Stable key for deduplicating reclaimed credits."""
        return (str(phase), credit_id)
