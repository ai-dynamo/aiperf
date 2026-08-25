# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dead services must be detected, without false-positive batch expiry.

Heartbeats reach the registry, but nothing acted on staleness: a service that
stopped heartbeating was never failed, so waiters blocked until an outer
timeout fired. Restores the watchdog with the three protections its predecessor
earned in production, where a controller stall flagged 141 of 285 worker-group
managers dead in the same millisecond.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from pytest import param

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
    # "worker" is required; anything else the tests use is optional.
    mgr.required_services = {"worker": 1}
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
    mgr.error = MagicMock()
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


@pytest.mark.asyncio
async def test_judge_stale_service_process_alive_clears_strike(manager, monkeypatch):
    """Ground truth outranks inference: a silent-but-alive service is not reaped.

    A record processor blocking its event loop past the stale threshold (a long
    ``to_thread`` summarize, a tokenizer load, a GC pause) used to be declared
    dead, evicted from the result-join barrier, and have its buffered JSONL
    dropped from the final metrics.
    """
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.get_stale_services",
        lambda _t: [_stale("worker_1")],
    )
    failed: list[str] = []
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.fail_service",
        lambda sid, _st: failed.append(sid),
    )
    manager.get_service_liveness = lambda _sid: True

    await manager._monitor_heartbeats()
    await manager._monitor_heartbeats()
    await manager._monitor_heartbeats()

    assert failed == []
    assert manager._suspected_stale == {}
    assert manager._pending_reaped == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "liveness",
    [
        param(None, id="liveness_unknown_kubernetes"),
        param(False, id="liveness_dead"),
    ],
)  # fmt: skip
async def test_judge_stale_service_not_alive_still_reaps(
    manager, monkeypatch, liveness
):
    """Unknown (Kubernetes) and confirmed-dead liveness both still fail-fast."""
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.get_stale_services",
        lambda _t: [_stale("worker_1")],
    )
    failed: list[str] = []
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.fail_service",
        lambda sid, _st: failed.append(sid),
    )
    manager.get_service_liveness = lambda _sid: liveness

    await manager._monitor_heartbeats()
    await manager._monitor_heartbeats()

    assert failed == ["worker_1"]


@pytest.mark.asyncio
async def test_judge_stale_service_optional_dropped_without_failure(
    manager, monkeypatch
):
    """An optional service going silent degrades the run; it does not fail it.

    ``gpu_telemetry_manager`` is started outside ``required_services``, so on a
    machine with no DCGM source it produced a spurious ERROR-level
    ``ServiceProcessDiedError`` on every local run. This mirrors the startup
    contract in ``_reap_dead_processes_during_registration``.
    """
    stale = MagicMock(service_id="gpu_1", service_type="gpu_telemetry_manager")
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.get_stale_services",
        lambda _t: [stale],
    )
    failed: list[str] = []
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.fail_service",
        lambda sid, _st: failed.append(sid),
    )
    unregistered: list[str] = []
    monkeypatch.setattr(
        "aiperf.controller.base_service_manager.ServiceRegistry.unregister",
        unregistered.append,
    )

    await manager._monitor_heartbeats()
    await manager._monitor_heartbeats()

    assert failed == [], "an optional service was recorded as a fatal failure"
    assert unregistered == ["gpu_1"]
    assert manager._suspected_stale == {}
    manager.error.assert_not_called()


def test_get_service_liveness_base_returns_unknown(manager):
    """The base manager has no process handle; Kubernetes must answer None."""
    assert BaseServiceManager.get_service_liveness(manager, "worker_1") is None
