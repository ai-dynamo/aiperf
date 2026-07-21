# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locust-equivalent stop strategy: poll completed count, abandon on hit.

Locust doesn't stop issuing new requests the moment a target count is SENT.
Instead (see ``locust_user_num_requests.py``'s ``on_worker_report`` handler,
which fires every ``WORKER_REPORT_INTERVAL`` ~3s): it keeps spawning users
and firing requests continuously, and on a periodic poll checks whether
``environment.stats.total.num_requests`` (COMPLETED count) has reached the
target. The instant it has, ``runner.quit()`` kills every in-flight greenlet
immediately — no drain, no grace period, no waiting for stragglers. Requests
still in flight at that moment never fire ``events.request.fire()`` and are
silently dropped from the count.

``OvershootAbandonPoller`` reproduces that shape for a 'requests'-bound
AIPerf phase: while enabled, ``RequestCountStopCondition`` is bypassed (see
``stop_conditions.py``) so credit issuance is not gated by requests SENT at
all. This poller instead watches requests COMPLETED and, once the target is
reached, stops issuance and abandons all in-flight credits immediately
rather than draining them — matching Locust's behavior byte-for-byte instead
of AIPerf's own default (issue exactly N, wait for all N to return).

Two modes, both selected via ``overshoot_poll_interval_sec``:
  - N > 0 ("poll" mode): background loop wakes every N seconds and checks
    completed >= target, same shape as Locust's own ~3s worker-report
    interval. The cutoff can overshoot by however many requests complete
    between polls.
  - N == 0 ("exact" mode): no polling at all. ``PhaseProgressTracker``
    invokes a synchronous callback (registered here) the instant the
    credit that pushes completed to exactly ``target`` returns, so the
    abandon fires on the precise Nth completion rather than up to N
    seconds late. Useful when you want the tightest possible cutoff
    (AIPerf's own semantics, unlike Locust, can express this) rather than
    reproducing Locust's specific polling cadence.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.credit.sticky_router import CreditRouterProtocol
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.phase.credit_counter import CreditCounter
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker


class OvershootAbandonPoller(AIPerfLoggerMixin):
    """Background poller implementing the Locust-equivalent stop strategy.

    Owned by ``PhaseRunner``, started alongside the phase's progress-report
    task, and only constructed when ``config.overshoot_poll_interval_sec``
    is set. ``stop()`` cancels the poll loop early (e.g. the phase completed
    normally via another path before the poller ever fired).
    """

    def __init__(
        self,
        *,
        config: CreditPhaseConfig,
        lifecycle: PhaseLifecycle,
        counter: CreditCounter,
        progress: PhaseProgressTracker,
        credit_router: CreditRouterProtocol,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._lifecycle = lifecycle
        self._counter = counter
        self._progress = progress
        self._credit_router = credit_router
        self._task: asyncio.Task | None = None
        self._fired = False
        self._abandoned_credit_ids: set[int] = set()

    @property
    def abandoned_credit_ids(self) -> set[int]:
        """Credit IDs cancelled by the abandon cutoff (empty until fired)."""
        return self._abandoned_credit_ids

    @property
    def fired(self) -> bool:
        """True once this poller has actually triggered the abandon path.

        Lets ``PhaseRunner`` skip its normal grace/drain wait when the
        poller (not the usual sending-complete path) already finished the
        phase.
        """
        return self._fired

    def start(self) -> None:
        interval = self._config.overshoot_poll_interval_sec
        if interval is None:
            return
        if interval == 0:
            # Exact-cutoff mode: no background loop. Register a synchronous
            # callback that PhaseProgressTracker fires the instant
            # requests_completed reaches the target; we hop back onto the
            # event loop via create_task since _abandon_now needs to await
            # (progress_tracker.increment_returned itself must stay sync).
            self._progress.set_exact_overshoot_callback(self._on_exact_target_reached)
            return
        self._task = asyncio.ensure_future(self._poll_loop(interval))

    def _on_exact_target_reached(self) -> None:
        target = self._config.total_expected_requests
        completed = self._counter.requests_completed
        self._task = asyncio.ensure_future(self._abandon_now(completed, target))

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def _poll_loop(self, interval: float) -> None:
        target = self._config.total_expected_requests
        if target is None:
            self.warning(
                "OvershootAbandonPoller started with no total_expected_requests "
                "configured; this should have been caught by config validation. "
                "Poller is a no-op."
            )
            return
        try:
            while True:
                await asyncio.sleep(interval)
                if self._lifecycle.was_cancelled or self._lifecycle.is_complete:
                    return
                completed = self._counter.requests_completed
                self.debug(
                    lambda completed=completed: f"Overshoot poll: completed={completed}/{target}"
                )
                if completed >= target:
                    await self._abandon_now(completed, target)
                    return
        except asyncio.CancelledError:
            self.debug("Overshoot poller cancelled before firing")
            raise

    async def _abandon_now(self, completed: int, target: int) -> None:
        """Stop issuance and abandon all in-flight credits immediately.

        Mirrors ``runner.quit()``: no grace period, no drain wait. Any
        credit still in flight at this instant is cancelled and excluded
        from final stats the same way an externally-cancelled request is —
        it never contributes a completed/error record.
        """
        self.notice(
            f"Overshoot poll: completed ({completed}) reached target ({target}); "
            "stopping issuance and abandoning in-flight credits immediately "
            "(Locust-equivalent stop, no drain)."
        )
        self._fired = True

        if not self._lifecycle.is_sending_complete:
            self._lifecycle.mark_sending_complete(timeout_triggered=False)
            self._progress.freeze_sent_counts()
            self._progress.all_credits_sent_event.set()

        self._abandoned_credit_ids = await self._credit_router.cancel_all_credits()

        if not self._lifecycle.is_complete:
            self._lifecycle.mark_complete(grace_period_triggered=False)
            self._progress.freeze_completed_counts()
        self._progress.all_credits_returned_event.set()
