# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-registration, load-tracking, and selection helpers for the sticky router.

Split out of ``sticky_router.py`` to keep that module under the file-size
ergonomics limit. State lives on ``StickyCreditRouter``; this mixin only
holds methods that operate on that state.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from aiperf.credit._router_types import UnavailableSession, WorkerLoad
from aiperf.credit.structs import Credit


class _WorkersMixin:
    """Worker registration, credit-load tracking, and least-loaded selection."""

    # Attribute types provided by the host class (StickyCreditRouter).
    if TYPE_CHECKING:
        _workers: dict[str, WorkerLoad]
        _workers_cache: list[WorkerLoad]
        _workers_by_load: dict[int, set[str]]
        _min_load: int
        _sticky_sessions: dict[str, str]
        _unavailable_sessions: dict[str, UnavailableSession]
        _initializing_workers: set[str]
        _detached_workers: dict[str, WorkerLoad]
        _cancellation_pending: bool
        _credits_complete: bool
        _pending_reconciliation: dict[str, frozenset[int]]
        _missed_reconciliation_cycles: dict[str, int]
        _suspected_orphans: dict[str, set[int]]
        _first_token_received: set[tuple[str, int]]
        _reclaimed_credit_ids: set[tuple[str, int]]
        is_trace_enabled: bool

        def trace(self, msg: Any) -> None: ...
        def warning(self, msg: Any) -> None: ...
        def error(self, msg: Any) -> None: ...

    def _register_worker(self, worker_id: str) -> None:
        """Register worker for routing, create WorkerLoad entry.

        Late-joining workers initialize:
        - virtual_sent_credits to average (prevents thundering herd on credits)
        - last_sent_at_ns to current time (prevents winning all timestamp tie-breaks)
        """
        self._initializing_workers.discard(worker_id)
        if worker_id in self._workers:
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

    def _unregister_worker(
        self,
        worker_id: str,
        *,
        session_loss_reason: str | None = None,
    ) -> WorkerLoad | None:
        """Unregister worker. Sticky sessions are cleared and reassigned on next access."""
        worker_load = self._workers.pop(worker_id, None)
        if worker_load is not None:
            self._drop_worker_from_load_index(worker_id, worker_load)
            self._orphan_worker_sessions(worker_id, worker_load, session_loss_reason)
        else:
            # Warn but continue - may happen if shutdown message arrives before ready message.
            self.warning(
                f"Worker {worker_id} not found when unregistering. This should not happen."
            )

        self._workers_cache = list(self._workers.values())
        self._maybe_recompute_min_load(worker_load)

        self._pending_reconciliation.pop(worker_id, None)
        self._missed_reconciliation_cycles.pop(worker_id, None)
        self._suspected_orphans.pop(worker_id, None)
        return worker_load

    def _drop_worker_from_load_index(
        self, worker_id: str, worker_load: WorkerLoad
    ) -> None:
        """Remove a worker from the by-load index and log the departure."""
        if worker_load.in_flight_credits > 0 and not self._cancellation_pending:
            self.warning(
                f"Worker {worker_id} unregistered with {worker_load.in_flight_credits} in-flight credits"
            )
        if self.is_trace_enabled:
            self.trace(
                f"Worker unregistered: {worker_id} (remaining={len(self._workers)})"
            )
        self._workers_by_load[worker_load.in_flight_credits].discard(worker_id)

    def _orphan_worker_sessions(
        self,
        worker_id: str,
        worker_load: WorkerLoad,
        session_loss_reason: str | None,
    ) -> None:
        """Drop sticky sessions owned by a departing worker and record failure reasons."""
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

    def _maybe_recompute_min_load(self, removed_load: WorkerLoad | None) -> None:
        """Refresh ``_min_load`` if the removed worker drained the current bucket."""
        needs_recalc = removed_load is None or (
            removed_load.in_flight_credits == self._min_load
            and len(self._workers_by_load[self._min_load]) == 0
        )
        if not needs_recalc:
            return
        if self._workers_cache:
            self._min_load = min(w.in_flight_credits for w in self._workers_cache)
        else:
            self._min_load = 0

    def _track_credit_sent(self, worker_id: str, credit: Credit) -> None:
        """Update worker load: increment in_flight_credits. Lock-free."""
        self._reclaimed_credit_ids.discard(self._credit_id_key(credit.phase, credit.id))  # type: ignore[attr-defined]
        worker_load = self._workers.get(worker_id)
        if worker_load is None:
            self._warn_missing_worker(worker_id, "sent")
            return

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
            # We only send credits one at a time, so if this worker was the last at the
            # minimum load, the new minimum equals this worker's new load.
            self._min_load = new_load

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
        self._first_token_received.discard(self._credit_id_key(phase, credit_id))  # type: ignore[attr-defined]

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

        return best_worker_id  # type: ignore[return-value]
