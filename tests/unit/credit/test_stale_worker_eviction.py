# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A worker that stops answering must stop receiving credits.

Documented 2026-03-09 and still open: the sticky router only dropped a worker
on an explicit WorkerShutdown, so a pod that died without one kept being
selected. Every credit routed to it was never returned, which starves the
concurrency limiter -- throughput degrades silently with nothing in the logs
naming the cause.

A dead worker cannot report its own death, so detection has to be router-side.
It must NOT be based on credit-channel silence: a worker emits nothing between
its FirstToken and its CreditReturn, so a reasoning model with a minute of
decode is indistinguishable from a crashed pod, and evicting on that killed
healthy workers one at a time until routing had none left. Detection keys off
service heartbeats, which the worker publishes on its own timer regardless of
what request it is running, and eviction is recoverable.
"""

import time
from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import CreditReturn, WorkerShutdown
from aiperf.credit.sticky_router import StickyCreditRouter, WorkerLoad
from aiperf.credit.structs import Credit


@pytest.fixture(autouse=True)
def _quiet_logging():
    """The router's log-level properties need a real logger; stub them out."""
    with (
        patch.object(StickyCreditRouter, "is_trace_enabled", False),
        patch.object(StickyCreditRouter, "is_debug_enabled", False),
    ):
        yield


NS = 1_000_000_000


def _router(*, workers: dict[str, float], in_flight: int = 1, heartbeat: bool = True):
    """Build a router with the given worker_id -> seconds-since-last-heartbeat.

    Workers hold ``in_flight`` credits each. ``heartbeat=False`` models a worker
    whose heartbeat has never been observed.
    """
    router = StickyCreditRouter.__new__(StickyCreditRouter)
    now = time.time_ns()
    router._workers = {}
    router._evicted_workers = {}
    router._workers_by_load = defaultdict(set)
    router._sticky_sessions = {}
    router._unavailable_sessions = {}
    router._connected_workers = set()
    router._workers_cache = []
    router._min_load = 0
    router._cancellation_pending = False
    router._credits_complete = False
    router._on_return_callback = None
    router._worker_available_event = MagicMock()
    router.warning = MagicMock()
    router.error = MagicMock()
    router.trace = MagicMock()
    router.debug = MagicMock()
    for wid, age_s in workers.items():
        load = WorkerLoad(worker_id=wid)
        if heartbeat:
            load.last_heartbeat_ns = now - int(age_s * NS)
        load.in_flight_credits = in_flight
        router._workers[wid] = load
        router._workers_by_load.setdefault(in_flight, set()).add(wid)
        router._connected_workers.add(wid)
    router._workers_cache = list(router._workers.values())
    return router


class TestStaleWorkerEviction:
    def test_worker_that_stopped_heartbeating_is_evicted(self):
        router = _router(workers={"w-dead": 120.0})
        evicted = router.evict_stale_workers(stale_after_s=60.0)
        assert evicted == ["w-dead"]
        assert "w-dead" not in router._workers

    def test_busy_worker_silent_on_the_credit_channel_is_kept(self):
        """THE regression: one long request (reasoning model, ~1s TTFT then a
        minute of decode) means no credit-channel traffic at all, but the
        worker is alive and heartbeating. Evicting it orphaned its sticky
        sessions and, with concurrency spread across the pool, took out every
        busy worker in turn until ``send_credit`` raised "No workers available".
        """
        router = _router(workers={"w-busy": 0.0})
        # Not one credit-channel message in 10x the staleness window (the
        # router does not track that at all -- that is the point), but
        # heartbeats kept arriving.
        router.note_worker_heartbeat("w-busy")
        assert router.evict_stale_workers(stale_after_s=60.0) == []
        assert "w-busy" in router._workers

    def test_idle_worker_that_keeps_heartbeating_is_kept(self):
        router = _router(workers={"w-idle": 0.0}, in_flight=0)
        assert router.evict_stale_workers(stale_after_s=60.0) == []
        assert "w-idle" in router._workers

    def test_idle_worker_that_stopped_heartbeating_is_evicted(self):
        """A dead worker sitting at zero in-flight still wins selections and
        blackholes every credit routed to it; heartbeats catch it before it
        takes one."""
        router = _router(workers={"w-dead-idle": 120.0}, in_flight=0)
        assert router.evict_stale_workers(stale_after_s=60.0) == ["w-dead-idle"]

    def test_recently_seen_worker_is_kept(self):
        router = _router(workers={"w-live": 5.0})
        assert router.evict_stale_workers(stale_after_s=60.0) == []
        assert "w-live" in router._workers

    def test_only_the_stale_one_goes(self):
        router = _router(workers={"w-live": 1.0, "w-dead": 300.0})
        assert router.evict_stale_workers(stale_after_s=60.0) == ["w-dead"]
        assert set(router._workers) == {"w-live"}

    def test_eviction_is_announced(self):
        """Silent degradation is the failure being fixed; say it out loud."""
        router = _router(workers={"w-dead": 120.0})
        router.evict_stale_workers(stale_after_s=60.0)
        router.warning.assert_called()

    def test_worker_with_no_heartbeat_yet_is_not_evicted(self):
        """No liveness feed (nothing wired it up, or the first heartbeat has
        not landed) degrades to no eviction, never to evicting everybody."""
        router = _router(workers={"w-new": 9999.0}, heartbeat=False)
        assert router.evict_stale_workers(stale_after_s=60.0) == []

    def test_disabled_when_threshold_is_zero(self):
        router = _router(workers={"w-dead": 9999.0})
        assert router.evict_stale_workers(stale_after_s=0.0) == []

    def test_heartbeat_refreshes_the_clock(self):
        router = _router(workers={"w-1": 300.0})
        router.note_worker_heartbeat("w-1")
        assert router.evict_stale_workers(stale_after_s=60.0) == []

    @pytest.mark.asyncio
    async def test_credit_channel_traffic_alone_does_not_refresh_the_clock(self):
        """A CreditReturn is not proof of liveness for the staleness sweep --
        keeping the two separate is what stops a busy worker from looking
        alive only while it happens to be chatty. The router keeps no
        credit-channel clock at all, so handling a return leaves the heartbeat
        clock untouched and the sweep still fires."""
        router = _router(workers={"w-1": 300.0})
        before = router._workers["w-1"].last_heartbeat_ns
        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="c1",
            x_correlation_id="x1",
            turn_index=0,
            num_turns=1,
            issued_at_ns=0,
        )
        await router._handle_router_message(
            "w-1",
            CreditReturn(
                credit=credit, cancelled=False, error=None, first_token_sent=True
            ),
        )
        assert router._workers["w-1"].last_heartbeat_ns == before
        assert router.evict_stale_workers(stale_after_s=60.0) == ["w-1"]


class TestEvictionIsRecoverable:
    """``WorkerDispatchable`` is latched to send exactly once, so before this a
    wrongly-evicted worker was gone for the rest of the run."""

    def test_heartbeat_after_eviction_readmits_the_worker(self):
        router = _router(workers={"w-1": 120.0}, in_flight=2)
        assert router.evict_stale_workers(stale_after_s=60.0) == ["w-1"]
        router.note_worker_heartbeat("w-1")
        assert "w-1" in router._workers
        assert "w-1" not in router._evicted_workers
        # In-flight accounting survives, so the credits it still holds do not
        # log a spurious return underflow when they land.
        assert router._workers["w-1"].in_flight_credits == 2
        assert "w-1" in router._workers_by_load[2]

    @pytest.mark.asyncio
    async def test_credit_return_after_eviction_readmits_the_worker(self):
        router = _router(workers={"w-1": 120.0}, in_flight=1)
        router.evict_stale_workers(stale_after_s=60.0)
        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="c1",
            x_correlation_id="x1",
            turn_index=0,
            num_turns=1,
            issued_at_ns=0,
        )
        await router._handle_router_message(
            "w-1",
            CreditReturn(
                credit=credit, cancelled=False, error=None, first_token_sent=True
            ),
        )
        assert "w-1" in router._workers
        assert router._workers["w-1"].in_flight_credits == 0
        router.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_after_eviction_does_not_readmit(self):
        router = _router(workers={"w-1": 120.0})
        router.evict_stale_workers(stale_after_s=60.0)
        await router._handle_router_message("w-1", WorkerShutdown(worker_id="w-1"))
        assert "w-1" not in router._workers
        assert "w-1" not in router._evicted_workers
        # A trailing heartbeat from a worker that announced its own shutdown
        # must not bring it back.
        router.note_worker_heartbeat("w-1")
        assert "w-1" not in router._workers
