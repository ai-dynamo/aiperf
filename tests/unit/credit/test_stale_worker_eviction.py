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
what request it is running. Eviction is terminal when active worker-local work
would be lost.
"""

import time
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import CreditReturn, WorkerDispatchable, WorkerShutdown
from aiperf.credit.sticky_router import StickyCreditRouter, WorkerLoad, _StickyEntry
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

    Workers hold ``in_flight`` credits each. ``heartbeat=False`` models a load
    entry built without going through ``_register_worker`` (which seeds the
    clock), i.e. no liveness feed at all.
    """
    router = StickyCreditRouter.__new__(StickyCreditRouter)
    now = time.monotonic_ns()
    router._workers = {}
    router._workers_by_load = defaultdict(set)
    router._sticky_sessions = {}
    router._terminally_lost_workers = set()
    router._gracefully_shutdown_workers = set()
    router._connected_workers = set()
    router._workers_cache = []
    router._min_load = 0
    router._cancellation_pending = False
    router._credits_complete = False
    router._on_return_callback = None
    router._on_worker_lost = None
    router._on_worker_count_changed = None
    router._peak_worker_count = 0
    router._stale_worker_strikes = {}
    router._last_stale_sweep_ns = None
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
        """A load entry built outside ``_register_worker`` has no liveness feed
        at all, and that degrades to no eviction, never to evicting everybody.
        Registration itself seeds the clock -- see
        ``test_worker_that_died_before_its_first_heartbeat_is_evicted``."""
        router = _router(workers={"w-new": 9999.0}, heartbeat=False)
        assert router.evict_stale_workers(stale_after_s=60.0) == []

    def test_registration_seeds_the_staleness_clock(self):
        """``last_heartbeat_ns`` must be non-zero the instant a worker joins."""
        router = _router(workers={})
        router._register_worker("w-new")
        assert router._workers["w-new"].last_heartbeat_ns > 0

    def test_worker_that_died_before_its_first_heartbeat_is_evicted(self):
        """THE gap: a worker is dispatchable and taking credits from the moment
        it announces itself, but its first heartbeat lands one heartbeat
        interval later. While ``last_heartbeat_ns`` sat at 0 the eviction guard
        read that as "immortal" rather than "never seen", so a worker killed in
        that window was never dropped -- and it could not be rescued by an
        earlier heartbeat either, because ``note_worker_heartbeat`` discards
        heartbeats for workers not yet in the routing pool. Its credits never
        returned, and a ``--request-count`` run waits for those returns with no
        deadline, so the run hung with nothing naming the cause."""
        router = _router(workers={})
        long_dead = time.monotonic_ns() - int(300 * NS)
        with patch.object(time, "monotonic_ns", return_value=long_dead):
            router._register_worker("w-stillborn")

        assert router.evict_stale_workers(stale_after_s=60.0) == ["w-stillborn"]
        assert "w-stillborn" not in router._workers

    def test_heartbeat_before_registration_is_discarded(self):
        """Documents why seeding at registration is the only fix available:
        heartbeats that arrive while the worker is merely connected cannot
        pre-seed the clock."""
        router = _router(workers={})
        router.note_worker_heartbeat("w-not-yet-registered")
        assert "w-not-yet-registered" not in router._workers

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


class TestTerminalStaleEviction:
    """A stale heartbeat is terminal once the router has dropped its sessions.

    Re-admission cannot restore the worker-local session state that was lost
    with the sticky aliases, so it would turn a terminal loss into an unsafe
    continuation on an arbitrary worker.
    """

    def test_heartbeat_after_eviction_does_not_readmit_the_worker(self) -> None:
        router = _router(workers={"w-1": 120.0}, in_flight=2)

        assert router.evict_stale_workers(stale_after_s=60.0) == ["w-1"]
        router.note_worker_heartbeat("w-1")

        assert "w-1" not in router._workers
        assert not hasattr(router, "_evicted_workers")

    @pytest.mark.asyncio
    async def test_dispatchable_after_eviction_does_not_readmit_the_worker(
        self,
    ) -> None:
        router = _router(workers={"w-1": 120.0}, in_flight=2)
        router.evict_stale_workers(stale_after_s=60.0)

        await router._handle_router_message("w-1", WorkerDispatchable(worker_id="w-1"))

        assert "w-1" not in router._workers

    @pytest.mark.asyncio
    async def test_return_after_eviction_does_not_readmit_the_worker(self) -> None:
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

        assert "w-1" not in router._workers
        assert not hasattr(router, "_evicted_workers")

    def test_worker_loss_drops_root_and_descendant_aliases(self) -> None:
        router = _router(workers={"w-1": 120.0})
        entry = _StickyEntry(
            worker_id="w-1",
            root_key="root",
            aliases={"child", "grandchild"},
        )
        router._sticky_sessions.update(
            {"root": entry, "child": entry, "grandchild": entry}
        )
        router._workers["w-1"].active_session_ids.add("root")

        router.evict_stale_workers(stale_after_s=60.0)

        assert router._sticky_sessions == {}


class TestTerminalWorkerLossNotification:
    """A lost worker is unrecoverable for the sessions it held, so the router
    reports the loss exactly once and lets the run fail fast rather than
    quietly continuing with truncated sessions."""

    def test_stale_worker_notifies_terminal_loss_once(self) -> None:
        router = _router(workers={"w-dead": 120.0})
        lost = MagicMock()
        router.set_worker_lost_callback(lost)

        assert router.evict_stale_workers(stale_after_s=60.0) == ["w-dead"]
        assert router.evict_stale_workers(stale_after_s=60.0) == []
        lost.assert_called_once_with("worker_unavailable: worker stopped responding")

    @pytest.mark.asyncio
    async def test_worker_shutdown_drains_late_single_turn_return_without_failure(
        self,
    ) -> None:
        """The return PULL channel can lag the graceful shutdown DEALER message."""
        router = _router(workers={"w-1": 0.0})
        lost = MagicMock()
        returned = AsyncMock()
        router.set_worker_lost_callback(lost)
        router.set_return_callback(returned)
        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="c1",
            x_correlation_id="x1",
            turn_index=0,
            num_turns=1,
            issued_at_ns=0,
        )

        await router._handle_router_message("w-1", WorkerShutdown(worker_id="w-1"))
        await router._handle_router_message(
            "w-1",
            CreditReturn(
                credit=credit, cancelled=False, error=None, first_token_sent=True
            ),
        )

        lost.assert_not_called()
        returned.assert_awaited_once()
        router.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_worker_shutdown_with_sticky_session_notifies_terminal_loss(
        self,
    ) -> None:
        router = _router(workers={"w-1": 0.0})
        router._workers["w-1"].active_session_ids.add("x1")
        lost = MagicMock()
        router.set_worker_lost_callback(lost)

        await router._handle_router_message("w-1", WorkerShutdown(worker_id="w-1"))

        lost.assert_called_once_with("worker_unavailable: worker shut down")

    def test_stale_sweep_reports_one_terminal_loss_for_multiple_workers(self) -> None:
        router = _router(workers={"w-1": 120.0, "w-2": 120.0})
        lost = MagicMock()
        router.set_worker_lost_callback(lost)

        assert router.evict_stale_workers(stale_after_s=60.0) == ["w-1", "w-2"]

        assert router._workers == {}
        lost.assert_called_once_with("worker_unavailable: worker stopped responding")

    def test_raising_loss_callback_does_not_interrupt_stale_sweep(self) -> None:
        router = _router(workers={"w-1": 120.0, "w-2": 120.0})
        router.set_worker_lost_callback(MagicMock(side_effect=RuntimeError("boom")))

        assert router.evict_stale_workers(stale_after_s=60.0) == ["w-1", "w-2"]

        assert router._workers == {}
        router.error.assert_called_once()

    def test_idle_worker_loss_is_non_terminal(self) -> None:
        router = _router(workers={"w-idle": 120.0}, in_flight=0)
        lost = MagicMock()
        counted = MagicMock()
        router.set_worker_lost_callback(lost)
        router.set_worker_count_changed_callback(counted)

        assert router.evict_stale_workers(stale_after_s=60.0) == ["w-idle"]

        lost.assert_not_called()
        counted.assert_called_once_with(0)

    def test_cancellation_suppresses_loss_but_still_drops_the_worker(self) -> None:
        """Teardown-time removal is expected, not a failure."""
        router = _router(workers={"w-dead": 120.0})
        lost = MagicMock()
        router.set_worker_lost_callback(lost)
        router._cancellation_pending = True
        counted = MagicMock()
        router.set_worker_count_changed_callback(counted)

        assert router.evict_stale_workers(stale_after_s=60.0) == ["w-dead"]

        assert "w-dead" not in router._workers
        lost.assert_not_called()
        counted.assert_called_once_with(0)

    def test_completed_credits_suppress_loss(self) -> None:
        router = _router(workers={"w-dead": 120.0})
        lost = MagicMock()
        router.set_worker_lost_callback(lost)
        router._credits_complete = True

        assert router.evict_stale_workers(stale_after_s=60.0) == ["w-dead"]

        assert "w-dead" not in router._workers
        lost.assert_not_called()

    def test_unknown_worker_removal_reports_nothing(self) -> None:
        router = _router(workers={})
        lost = MagicMock()
        router.set_worker_lost_callback(lost)

        router._unregister_worker("never-registered")

        lost.assert_not_called()


class TestPeriodicSweepTwoStrikeConfirmation:
    """The background sweep (``_evict_stale_workers_task``) must not evict a
    worker on the same tick where it first crosses the cutoff -- that is
    exactly the race flagged in review: a worker whose event loop stalls and
    then recovers should get a second sweep's worth of grace before the
    router calls it dead, rather than being evicted unconditionally on the
    first sweep past ``STALE_TIME * 3``.
    """

    @pytest.mark.asyncio
    async def test_first_sweep_past_cutoff_only_suspects_the_worker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.environment import Environment

        monkeypatch.setattr(Environment.WORKER, "STALE_TIME", 10.0)
        router = _router(workers={"w-1": 120.0})

        await router._evict_stale_workers_task()

        assert "w-1" in router._workers
        assert router._stale_worker_strikes.get("w-1") == 1

    @pytest.mark.asyncio
    async def test_second_consecutive_sweep_past_cutoff_evicts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.environment import Environment

        monkeypatch.setattr(Environment.WORKER, "STALE_TIME", 10.0)
        router = _router(workers={"w-1": 120.0})
        lost = MagicMock()
        router.set_worker_lost_callback(lost)

        await router._evict_stale_workers_task()
        await router._evict_stale_workers_task()

        assert "w-1" not in router._workers
        lost.assert_called_once_with("worker_unavailable: worker stopped responding")

    @pytest.mark.asyncio
    async def test_heartbeat_between_sweeps_clears_the_strike(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.environment import Environment

        monkeypatch.setattr(Environment.WORKER, "STALE_TIME", 10.0)
        router = _router(workers={"w-1": 120.0})

        await router._evict_stale_workers_task()
        assert router._stale_worker_strikes.get("w-1") == 1

        router.note_worker_heartbeat("w-1")
        await router._evict_stale_workers_task()

        assert "w-1" in router._workers
        assert "w-1" not in router._stale_worker_strikes


class TestDelayedSweepIsNotEvidenceOfWorkerDeath:
    """A stall in the *router's own* event loop must not evict live workers.

    Staleness is silence measured by this loop. If the loop is starved (GC
    pause, CPU contention, a blocking call), every worker crosses the cutoff at
    the same instant even though all of them are alive and heartbeating -- a
    false mass eviction that is terminal for every worker holding in-flight
    credits. The controller-side watchdog already forgives its own delayed
    ticks (``BaseServiceManager._monitor_heartbeats_tick``); the router sweep
    must do the same.
    """

    @pytest.mark.asyncio
    async def test_delayed_sweep_does_not_evict_and_clears_strikes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.environment import Environment

        monkeypatch.setattr(Environment.WORKER, "STALE_TIME", 10.0)
        monkeypatch.setattr(Environment.WORKER, "ROUTER_STALE_SWEEP_DELAY_FACTOR", 2.0)
        router = _router(workers={"w-1": 120.0, "w-2": 120.0})
        lost = MagicMock()
        router.set_worker_lost_callback(lost)

        await router._evict_stale_workers_task()
        assert router._stale_worker_strikes.get("w-1") == 1

        # The next sweep lands 60s late (>> STALE_TIME * 2): the router's own
        # loop stalled, so the apparent staleness is the router's fault.
        router._last_stale_sweep_ns = time.monotonic_ns() - int(60 * NS)
        await router._evict_stale_workers_task()

        assert "w-1" in router._workers
        assert "w-2" in router._workers
        assert router._stale_worker_strikes == {}
        lost.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_time_sweep_after_a_forgiven_one_still_evicts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forgiveness costs a sweep, it does not disable eviction."""
        from aiperf.common.environment import Environment

        monkeypatch.setattr(Environment.WORKER, "STALE_TIME", 10.0)
        monkeypatch.setattr(Environment.WORKER, "ROUTER_STALE_SWEEP_DELAY_FACTOR", 2.0)
        router = _router(workers={"w-1": 120.0})

        router._last_stale_sweep_ns = time.monotonic_ns() - int(60 * NS)
        await router._evict_stale_workers_task()
        assert "w-1" in router._workers

        await router._evict_stale_workers_task()
        await router._evict_stale_workers_task()

        assert "w-1" not in router._workers

    @pytest.mark.asyncio
    async def test_suppressed_sweep_resets_the_delay_baseline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Teardown suppression is not a stalled event loop."""
        from aiperf.common.environment import Environment

        monkeypatch.setattr(Environment.WORKER, "STALE_TIME", 10.0)
        router = _router(workers={"w-1": 120.0})
        router._last_stale_sweep_ns = time.monotonic_ns() - int(60 * NS)
        router._cancellation_pending = True

        await router._evict_stale_workers_task()

        assert router._last_stale_sweep_ns is None
