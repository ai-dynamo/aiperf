# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatchable-worker floor and membership-notification tests."""

from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest

from aiperf.credit.sticky_router import StickyCreditRouter, WorkerLoad


@pytest.fixture(autouse=True)
def _quiet_logging():
    with (
        patch.object(StickyCreditRouter, "is_trace_enabled", False),
        patch.object(StickyCreditRouter, "is_debug_enabled", False),
    ):
        yield


def _router(alive: int, peak: int):
    r = StickyCreditRouter.__new__(StickyCreditRouter)
    r._workers = {}
    r._workers_by_load = defaultdict(set)
    r._workers_cache = []
    r._sticky_sessions = {}
    r._connected_workers = set()
    r._min_load = 0
    r._cancellation_pending = False
    r._credits_complete = False
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
    def test_reports_a_breach_when_the_fleet_halves(self):
        router = _router(alive=4, peak=10)
        assert router.check_worker_floor(min_fraction=0.5) is not None

    def test_healthy_fleet_reports_nothing(self):
        router = _router(alive=9, peak=10)
        assert router.check_worker_floor(min_fraction=0.5) is None

    def test_exactly_at_the_floor_is_acceptable(self):
        router = _router(alive=5, peak=10)
        assert router.check_worker_floor(min_fraction=0.5) is None

    def test_disabled_by_default(self):
        router = _router(alive=1, peak=100)
        assert router.check_worker_floor(min_fraction=0.0) is None

    def test_message_names_the_numbers(self):
        router = _router(alive=2, peak=10)
        reason = router.check_worker_floor(min_fraction=0.5)
        assert "2" in reason and "10" in reason

    def test_peak_tracks_the_high_water_mark(self):
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
    async def test_breach_is_left_for_timing_manager(self, monkeypatch):
        from aiperf.common.environment import Environment

        router = _router(alive=1, peak=10)
        monkeypatch.setattr(Environment.WORKER, "MIN_ALIVE_FRACTION", 0.5)

        await router._evict_stale_workers_task()
        await router._evict_stale_workers_task()

        router.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_report_during_teardown(self, monkeypatch):
        """Workers legitimately stop reporting once credits are complete."""
        from aiperf.common.environment import Environment

        router = _router(alive=0, peak=10)
        router._credits_complete = True
        monkeypatch.setattr(Environment.WORKER, "MIN_ALIVE_FRACTION", 0.5)

        await router._evict_stale_workers_task()
        router.error.assert_not_called()


class TestWorkerMembershipNotification:
    def test_unregister_notifies_the_timing_manager_of_the_new_count(self) -> None:
        """The floor decision belongs to TimingManager, after router removal."""
        router = _router(alive=2, peak=2)
        on_worker_count_changed = MagicMock()

        router.set_worker_count_changed_callback(on_worker_count_changed)
        router._unregister_worker("w-1")

        on_worker_count_changed.assert_called_once_with(1)
