# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconciliation + detached-worker helpers for the sticky credit router.

Split out of ``sticky_router.py`` to keep that module under the file-size
ergonomics limit. All state lives on ``StickyCreditRouter``; this mixin only
holds methods that operate on that state.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.environment import Environment
from aiperf.common.utils import yield_to_event_loop
from aiperf.credit.messages import (
    CreditReturn,
    InFlightReconciliation,
    InFlightReport,
)

if TYPE_CHECKING:
    from aiperf.credit._router_types import WorkerLoad


class _ReconciliationMixin:
    """Reconciliation + detached-worker behavior for the sticky router.

    The mixin assumes it is composed into a class that provides all the
    mutable router state (``_workers``, ``_detached_workers``, ...) plus
    the ``CommunicationMixin`` logger/async helpers. Splitting this out
    keeps the main file under the 500-line ergonomics limit without
    changing runtime behavior.
    """

    # Attribute types provided by the host class (StickyCreditRouter).
    if TYPE_CHECKING:
        _workers_cache: list[WorkerLoad]
        _workers: dict[str, WorkerLoad]
        _pending_reconciliation: dict[str, frozenset[int]]
        _missed_reconciliation_cycles: dict[str, int]
        _suspected_orphans: dict[str, set[int]]
        _detached_workers: dict[str, WorkerLoad]
        _detached_worker_deadlines_ns: dict[str, int]
        _detached_reclaim_tasks: dict[str, asyncio.Task[None]]
        _reclaimed_credit_ids: set[tuple[str, int]]
        _first_token_received: set[tuple[str, int]]
        _unavailable_sessions: dict[str, Any]
        _on_return_callback: Any
        _credits_complete: bool
        _cancellation_pending: bool
        _credit_router_client: Any

    async def _send_reconciliation_impl(self) -> None:
        """Send InFlightReconciliation to each worker with in-flight credits."""
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
                await self._handle_missed_reconciliation(worker_id, worker_load)
                continue

            credit_ids = frozenset(worker_load.active_credit_ids)
            self._pending_reconciliation[worker_id] = credit_ids
            await self._credit_router_client.send_to(  # type: ignore[attr-defined]
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

    async def _handle_missed_reconciliation(
        self, worker_id: str, worker_load: WorkerLoad
    ) -> None:
        """Escalate workers that have skipped 2+ consecutive reconciliation cycles."""
        self._missed_reconciliation_cycles[worker_id] += 1
        if self._missed_reconciliation_cycles[worker_id] < 2:
            return
        self.warning(  # type: ignore[attr-defined]
            f"Worker {worker_id} missed 2 reconciliation cycles "
            f"with {worker_load.in_flight_credits} in-flight credits "
            f"and {worker_load.active_sessions} active sessions"
        )
        detached = self._unregister_worker(  # type: ignore[attr-defined]
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
            self.warning(  # type: ignore[attr-defined]
                f"Received InFlightReport from {worker_id} with no pending reconciliation"
            )
            return

        self._missed_reconciliation_cycles.pop(worker_id, None)

        missing = sent_set - report.credit_ids
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

        self.warning(  # type: ignore[attr-defined]
            f"Orphaned credit {credit_id} on worker {worker_id} "
            f"(missing for 2 consecutive reconciliation cycles)"
        )

        key = self._credit_id_key(credit.phase, credit.id)  # type: ignore[attr-defined]
        first_token_sent = key in self._first_token_received
        self._first_token_received.discard(key)
        self._reclaimed_credit_ids.add(key)

        self._track_credit_returned(  # type: ignore[attr-defined]
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
        self.warning(  # type: ignore[attr-defined]
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

            key = self._credit_id_key(credit.phase, credit.id)  # type: ignore[attr-defined]
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
        self._detached_reclaim_tasks[worker_id] = self.execute_async(  # type: ignore[attr-defined]
            self._wait_and_reclaim_detached_worker(worker_id, grace_timeout_sec)
        )
        self.warning(  # type: ignore[attr-defined]
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
