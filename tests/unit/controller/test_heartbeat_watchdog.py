# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dead services must be detected, without false-positive batch expiry.

Heartbeats reach the registry, but nothing acted on staleness: a service that
stopped heartbeating was never failed, so waiters blocked until an outer
timeout fired. Restores the watchdog with the two protections its predecessor
earned in production, where a controller stall flagged 141 of 285 worker-group
managers dead in the same millisecond.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from aiperf.common.environment import Environment
from aiperf.controller.base_service_manager import BaseServiceManager


class _Manager(BaseServiceManager):
    """Concrete stand-in: the watchdog lives entirely on the base class."""

    async def _start_service_manager(self) -> None: ...
    async def _stop_service_manager(self) -> None: ...
    async def run_services(self, *a, **k) -> None: ...
    async def stop_service(self, *a, **k) -> None: ...
    async def run_service(self, *a, **k) -> None: ...
    async def shutdown_all_services(self, *a, **k):
        return []

    async def kill_all_services(self, *a, **k):
        return []

    async def wait_for_all_services_registration(self, *a, **k) -> None: ...
    async def wait_for_all_services_start(self, *a, **k) -> None: ...


@pytest.fixture
def manager(monkeypatch):
    mgr = _Manager.__new__(_Manager)
    mgr._suspected_stale = {}
    mgr._last_heartbeat_tick_ns = None
    mgr._heartbeat_monitoring_active = True
    mgr._shutdown_complete = False
    # Result-join eviction state. The watchdog records reaped services here and
    # drains them to the controller; with no hook installed the drain is a no-op.
    mgr._pending_reaped = {}
    mgr.on_service_reaped = None
    mgr._stop_requested_event = asyncio.Event()
    mgr.warning = MagicMock()
    mgr.debug = MagicMock()
    return mgr


def _stale(service_id: str):
    return MagicMock(service_id=service_id, service_type="worker")


@pytest.mark.asyncio
async def test_two_strikes_required_before_failing(manager, monkeypatch):
    """One stale tick is a suspicion; two consecutive is a death."""
    failed: list[str] = []
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.get_stale_services",
        lambda _t: [_stale("worker_1")],
    )
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.fail_service",
        lambda sid, _st: failed.append(sid),
    )

    await manager._monitor_heartbeats()
    assert failed == [], "failed on the first strike"
    assert manager._suspected_stale == {"worker_1": 1}

    await manager._monitor_heartbeats()
    assert failed == ["worker_1"]


@pytest.mark.asyncio
async def test_recovered_service_drops_its_strike(manager, monkeypatch):
    """A heartbeat between ticks clears the suspicion."""
    stale_now = [_stale("worker_1")]
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.get_stale_services",
        lambda _t: list(stale_now),
    )
    failed: list[str] = []
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.fail_service",
        lambda sid, _st: failed.append(sid),
    )

    await manager._monitor_heartbeats()
    assert manager._suspected_stale == {"worker_1": 1}
    stale_now.clear()
    await manager._monitor_heartbeats()
    assert manager._suspected_stale == {}
    assert failed == []


@pytest.mark.asyncio
async def test_delayed_tick_blames_nobody(manager, monkeypatch):
    """If the watchdog itself stalled, every service looks stale. Skip.

    This is the 141-of-285 incident: a controller stall, not 141 deaths.
    """
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.get_stale_services",
        lambda _t: [_stale(f"worker_{i}") for i in range(141)],
    )
    failed: list[str] = []
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.fail_service",
        lambda sid, _st: failed.append(sid),
    )

    manager._suspected_stale = {f"worker_{i}": 1 for i in range(141)}
    interval = Environment.SERVICE.HEARTBEAT_INTERVAL
    manager._last_heartbeat_tick_ns = time.time_ns() - int(interval * 5 * 1_000_000_000)

    await manager._monitor_heartbeats()

    assert failed == [], "a delayed watchdog tick killed services"
    assert manager._suspected_stale == {}
    manager.warning.assert_called()


@pytest.mark.asyncio
async def test_inactive_watchdog_resets_state(manager, monkeypatch):
    """Before activation (startup) nothing is judged, and state starts clean."""
    manager._heartbeat_monitoring_active = False
    manager._suspected_stale = {"worker_1": 1}
    manager._last_heartbeat_tick_ns = 123

    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.get_stale_services",
        lambda _t: [_stale("worker_1")],
    )
    failed: list[str] = []
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.fail_service",
        lambda sid, _st: failed.append(sid),
    )

    await manager._monitor_heartbeats()

    assert failed == []
    assert manager._suspected_stale == {}
    assert manager._last_heartbeat_tick_ns is None


@pytest.mark.asyncio
async def test_shutdown_suppresses_the_watchdog(manager, monkeypatch):
    """Services exiting during teardown are not failures."""
    manager._shutdown_complete = True
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.get_stale_services",
        lambda _t: [_stale("worker_1")],
    )
    failed: list[str] = []
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.fail_service",
        lambda sid, _st: failed.append(sid),
    )
    await manager._monitor_heartbeats()
    assert failed == []
