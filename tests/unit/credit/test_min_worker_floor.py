# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatchable-worker floor and membership-notification tests."""

import time
from collections import defaultdict
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.credit.sticky_router import StickyCreditRouter, WorkerLoad


@pytest.fixture(autouse=True)
def _quiet_logging() -> Iterator[None]:
    with (
        patch.object(StickyCreditRouter, "is_trace_enabled", False),
        patch.object(StickyCreditRouter, "is_debug_enabled", False),
    ):
        yield


def _router(alive: int, peak: int) -> StickyCreditRouter:
    r = StickyCreditRouter.__new__(StickyCreditRouter)
    r._workers = {}
    r._workers_by_load = defaultdict(set)
    r._workers_cache = []
    r._sticky_sessions = {}
    r._connected_workers = set()
    r._min_load = 0
    r._cancellation_pending = False
    r._credits_complete = False
    r._on_worker_lost = None
    r._stale_worker_strikes = {}
    r._last_stale_sweep_ns = None
    r._worker_available_event = MagicMock()
    r.warning = MagicMock()
    r.trace = MagicMock()
    r.debug = MagicMock()
    r.error = MagicMock()
    for i in range(alive):
        load = WorkerLoad(worker_id=f"w-{i}")
        r._workers[f"w-{i}"] = load
        r._workers_by_load[0].add(f"w-{i}")
    r._workers_cache = list(r._workers.values())
    r._peak_worker_count = peak
    return r


class TestWorkerFloor:
    def test_reports_a_breach_when_the_fleet_halves(self) -> None:
        router = _router(alive=4, peak=10)
        assert router.check_worker_floor(min_fraction=0.5) is not None

    def test_healthy_fleet_reports_nothing(self) -> None:
        router = _router(alive=9, peak=10)
        assert router.check_worker_floor(min_fraction=0.5) is None

    def test_exactly_at_the_floor_is_acceptable(self) -> None:
        router = _router(alive=5, peak=10)
        assert router.check_worker_floor(min_fraction=0.5) is None

    def test_disabled_by_default(self) -> None:
        router = _router(alive=1, peak=100)
        assert router.check_worker_floor(min_fraction=0.0) is None

    def test_message_names_the_numbers(self) -> None:
        router = _router(alive=2, peak=10)
        reason = router.check_worker_floor(min_fraction=0.5)
        assert "2" in reason and "10" in reason

    def test_peak_tracks_the_high_water_mark(self) -> None:
        router = _router(alive=0, peak=0)
        router._note_peak_workers()
        assert router._peak_worker_count == 0
        router._workers["w-a"] = WorkerLoad(worker_id="w-a")
        router._note_peak_workers()
        assert router._peak_worker_count == 1
        router._workers.clear()
        router._note_peak_workers()
        assert router._peak_worker_count == 1, "peak must not fall back"


class TestRouterDoesNotDecideTheBreach:
    @pytest.mark.asyncio
    async def test_eviction_uses_configured_stale_multiplier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sweep cutoff is STALE_TIME * ROUTER_STALE_EVICTION_MULTIPLIER.

        Eviction is two-strike confirmed, so the cutoff reaches the candidate
        scan on every sweep but only reaches ``_evict_worker_ids`` on the
        second consecutive sweep that still finds the worker stale.
        """
        from aiperf.common.environment import Environment

        router = _router(alive=1, peak=1)
        router._stale_worker_candidates = MagicMock(return_value=["w-0"])
        router._evict_worker_ids = MagicMock()
        monkeypatch.setattr(Environment.WORKER, "STALE_TIME", 7.0)
        monkeypatch.setattr(
            Environment.WORKER, "ROUTER_STALE_EVICTION_MULTIPLIER", 4.25
        )

        await router._evict_stale_workers_task()

        router._stale_worker_candidates.assert_called_once_with(29.75)
        router._evict_worker_ids.assert_not_called()

        await router._evict_stale_workers_task()

        router._evict_worker_ids.assert_called_once_with(["w-0"], 29.75)

    @pytest.mark.asyncio
    async def test_breach_is_left_for_timing_manager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Eviction is reported at WARNING; the floor breach is left for TimingManager (never ERROR)."""
        from aiperf.common.environment import Environment

        router = _router(alive=1, peak=10)
        monkeypatch.setattr(Environment.WORKER, "MIN_ALIVE_FRACTION", 0.5)
        stale_after_s = Environment.WORKER.STALE_TIME * 3
        router._workers["w-0"].last_heartbeat_ns = time.time_ns() - int(
            (stale_after_s + 5) * NANOS_PER_SECOND
        )

        # First sweep only suspects the worker (two-strike gate).
        await router._evict_stale_workers_task()
        assert "w-0" in router._workers, "first sweep must not evict on one strike"

        # Second consecutive sweep confirms and evicts it.
        await router._evict_stale_workers_task()
        assert "w-0" not in router._workers, (
            "stale worker must be evicted on the second consecutive sweep"
        )

        router.warning.assert_called()
        router.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_report_during_teardown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Workers legitimately stop reporting once credits are complete."""
        from aiperf.common.environment import Environment

        router = _router(alive=1, peak=10)
        router._credits_complete = True
        monkeypatch.setattr(Environment.WORKER, "MIN_ALIVE_FRACTION", 0.5)
        stale_after_s = Environment.WORKER.STALE_TIME * 3
        router._workers["w-0"].last_heartbeat_ns = time.time_ns() - int(
            (stale_after_s + 5) * NANOS_PER_SECOND
        )

        await router._evict_stale_workers_task()
        await router._evict_stale_workers_task()

        assert "w-0" in router._workers, (
            "eviction must be suppressed once credits are complete"
        )
        router.warning.assert_not_called()
        router.error.assert_not_called()


class TestWorkerMembershipNotification:
    def test_unregister_notifies_the_timing_manager_of_the_new_count(self) -> None:
        """The floor decision belongs to TimingManager, after router removal."""
        router = _router(alive=2, peak=2)
        on_worker_count_changed = MagicMock()

        router.set_worker_count_changed_callback(on_worker_count_changed)
        router._unregister_worker("w-1")

        on_worker_count_changed.assert_called_once_with(1)


class TestWorkerLostFiresOnlyForUnrecoverableState:
    """Pin the claim that lets ``_on_worker_lost`` bypass MIN_ALIVE_FRACTION.

    ``timing/manager.py`` treats a worker-loss callback as automatically fatal,
    skipping the fleet-floor check, and justifies that in a comment: the router
    only fires it when the departing worker owned in-flight credits or a sticky
    session, i.e. state that lived in that worker's memory and cannot be
    recovered by the survivors. That reasoning is load-bearing -- if the
    callback can fire for a worker that owned neither, an idle worker leaving a
    healthy fleet would abort the run -- yet it was only ever asserted in prose.
    These tests make it executable.
    """

    def _router_with_loss_callback(self) -> tuple[StickyCreditRouter, MagicMock]:
        router = _router(alive=2, peak=2)
        on_lost = MagicMock()
        router._on_worker_lost = on_lost
        router._terminally_lost_workers = set()
        router._on_worker_count_changed = None
        return router, on_lost

    def test_idle_worker_departure_does_not_report_a_loss(self) -> None:
        """No credits, no sessions: the survivors can carry the run."""
        router, on_lost = self._router_with_loss_callback()

        assert router._unregister_worker("w-1", reason="shutdown") is False
        on_lost.assert_not_called()

    def test_worker_with_in_flight_credits_reports_a_loss(self) -> None:
        router, on_lost = self._router_with_loss_callback()
        router._workers["w-1"].in_flight_credits = 3

        assert router._unregister_worker("w-1", reason="died") is True
        on_lost.assert_called_once()

    def test_worker_with_a_sticky_session_reports_a_loss(self) -> None:
        """Session state was worker-local, so no survivor can continue it."""
        router, on_lost = self._router_with_loss_callback()
        router._workers["w-1"].active_session_ids = {"conv-a"}

        assert router._unregister_worker("w-1", reason="died") is True
        on_lost.assert_called_once()

    def test_teardown_suppresses_the_loss_even_with_in_flight_credits(self) -> None:
        """Credits outstanding during teardown are expected, not a failure."""
        router, on_lost = self._router_with_loss_callback()
        router._workers["w-1"].in_flight_credits = 3
        router._credits_complete = True

        assert router._unregister_worker("w-1", reason="shutdown") is False
        on_lost.assert_not_called()

    def test_dropping_orphaned_sessions_does_not_erase_the_loss_signal(self) -> None:
        """The session set is read for the verdict AFTER it is handed off.

        ``_unregister_worker`` passes ``worker_load.active_session_ids`` to
        ``_drop_orphaned_sessions`` and only afterwards reads that same set to
        decide whether a loss occurred. They are the same object, so a cleanup
        step that emptied it in place would silently downgrade a real,
        unrecoverable loss to a clean departure.
        """
        router, on_lost = self._router_with_loss_callback()
        router._workers["w-1"].active_session_ids = {"conv-a", "conv-b"}

        assert router._unregister_worker("w-1", reason="died") is True
        on_lost.assert_called_once()
