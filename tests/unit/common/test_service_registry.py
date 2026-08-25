# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the process-wide service registry and its async waiting mixin."""

import pytest

from aiperf.common.enums import LifecycleState, ServiceRegistrationStatus
from aiperf.common.exceptions import (
    ServiceProcessDiedError,
    ServiceRegistrationTimeoutError,
)
from aiperf.common.service_registry import _ServiceRegistry
from aiperf.plugin.enums import ServiceType


@pytest.fixture
def registry() -> _ServiceRegistry:
    """A fresh registry instance, isolated from the module-level singleton."""
    return _ServiceRegistry()


def _register(
    registry: _ServiceRegistry,
    service_id: str,
    seen_ns: int = 1,
    service_type: ServiceType = ServiceType.WORKER,
    **kwargs,
) -> None:
    """Register a service with the required keyword-only arguments."""
    registry.register(
        service_id=service_id,
        service_type=service_type,
        first_seen_ns=seen_ns,
        state=LifecycleState.RUNNING,
        **kwargs,
    )


def test_registry_tracks_registered_services(registry: _ServiceRegistry) -> None:
    registry.expect_services({ServiceType.WORKER: 1})
    _register(registry, "worker-0")
    assert registry.is_registered("worker-0")
    assert registry.all_types_registered(ServiceType.WORKER)
    assert registry.all_registered()


@pytest.mark.asyncio
async def test_wait_for_all_raises_when_quorum_never_reached(
    registry: _ServiceRegistry,
) -> None:
    registry.expect_services({ServiceType.WORKER: 2})
    with pytest.raises(ServiceRegistrationTimeoutError) as excinfo:
        await registry.wait_for_all(timeout=0.1)
    assert excinfo.value.missing == {ServiceType.WORKER: 2}


@pytest.mark.asyncio
async def test_timeout_counts_span_every_expected_type(
    registry: _ServiceRegistry,
) -> None:
    """Fully-registered types must count toward registered/expected totals.

    Excluding them reports "0 of 1" for a run that is really 2 of 3, which
    points an operator at the wrong pod.
    """
    registry.expect_services({ServiceType.WORKER: 2, ServiceType.RECORD_PROCESSOR: 1})
    _register(registry, "worker-0")
    _register(registry, "worker-1")
    with pytest.raises(ServiceRegistrationTimeoutError) as excinfo:
        await registry.wait_for_all(timeout=0.1)
    assert excinfo.value.registered == 2
    assert excinfo.value.expected == 3
    assert "2 of 3" in str(excinfo.value)


@pytest.mark.asyncio
async def test_timeout_message_names_the_missing_service_ids(
    registry: _ServiceRegistry,
) -> None:
    registry.expect_service("worker-0", ServiceType.WORKER)
    registry.expect_service("ghost", ServiceType.WORKER)
    _register(registry, "worker-0")
    with pytest.raises(ServiceRegistrationTimeoutError) as excinfo:
        await registry.wait_for_ids(["worker-0", "ghost"], timeout=0.1)
    assert "ghost" in str(excinfo.value)


@pytest.mark.asyncio
async def test_wait_for_all_returns_once_quorum_reached(
    registry: _ServiceRegistry,
) -> None:
    registry.expect_services({ServiceType.WORKER: 2})
    _register(registry, "worker-0")
    _register(registry, "worker-1")
    await registry.wait_for_all(timeout=1.0)


@pytest.mark.asyncio
async def test_wait_for_type_returns_once_that_type_registers(
    registry: _ServiceRegistry,
) -> None:
    registry.expect_services({ServiceType.WORKER: 1, ServiceType.RECORD_PROCESSOR: 1})
    _register(registry, "worker-0")
    await registry.wait_for_type(ServiceType.WORKER, timeout=1.0)
    with pytest.raises(ServiceRegistrationTimeoutError):
        await registry.wait_for_type(ServiceType.RECORD_PROCESSOR, timeout=0.1)


@pytest.mark.asyncio
async def test_wait_for_ids_reports_the_missing_ids(
    registry: _ServiceRegistry,
) -> None:
    registry.expect_service("worker-0", ServiceType.WORKER)
    registry.expect_service("worker-1", ServiceType.WORKER)
    _register(registry, "worker-0")
    with pytest.raises(ServiceRegistrationTimeoutError):
        await registry.wait_for_ids(["worker-0", "worker-1"], timeout=0.1)
    _register(registry, "worker-1")
    await registry.wait_for_ids(["worker-0", "worker-1"], timeout=1.0)


@pytest.mark.asyncio
async def test_fail_service_wakes_waiters_with_process_died(
    registry: _ServiceRegistry,
) -> None:
    registry.expect_services({ServiceType.WORKER: 1})
    registry.fail_service("worker-0", ServiceType.WORKER)
    with pytest.raises(ServiceProcessDiedError):
        await registry.wait_for_all(timeout=1.0)


def test_register_is_idempotent_and_updates_last_seen(
    registry: _ServiceRegistry,
) -> None:
    registry.expect_services({ServiceType.WORKER: 1})
    _register(registry, "worker-0", seen_ns=1)
    _register(registry, "worker-0", seen_ns=99)
    assert registry.is_registered("worker-0")
    assert registry.get_service("worker-0").last_seen_ns == 99


def test_update_service_ignores_unknown_and_stale_updates(
    registry: _ServiceRegistry,
) -> None:
    registry.expect_services({ServiceType.WORKER: 1})
    _register(registry, "worker-0", seen_ns=10)
    registry.update_service("ghost", ServiceType.WORKER, 50, LifecycleState.RUNNING)
    assert registry.get_service("ghost") is None

    registry.update_service("worker-0", ServiceType.WORKER, 5, LifecycleState.STOPPING)
    assert registry.get_service("worker-0").last_seen_ns == 10

    registry.update_service("worker-0", ServiceType.WORKER, 50, LifecycleState.STOPPING)
    info = registry.get_service("worker-0")
    assert info.last_seen_ns == 50
    assert info.state == LifecycleState.STOPPING


def test_unregister_keeps_the_entry_but_clears_registration(
    registry: _ServiceRegistry,
) -> None:
    registry.expect_services({ServiceType.WORKER: 1})
    _register(registry, "worker-0")
    registry.unregister("worker-0")
    assert not registry.is_registered("worker-0")
    info = registry.get_service("worker-0")
    assert info.registration_status == ServiceRegistrationStatus.UNREGISTERED
    assert info.state == LifecycleState.STOPPED


def test_forget_removes_a_service_without_failing_it(
    registry: _ServiceRegistry,
) -> None:
    registry.expect_services({ServiceType.WORKER: 1})
    _register(registry, "worker-0")
    registry.forget("worker-0")
    assert not registry.is_registered("worker-0")
    assert registry.get_service("worker-0") is None


def test_get_services_by_pod_filters_on_pod_index(registry: _ServiceRegistry) -> None:
    registry.expect_services({ServiceType.WORKER: 2})
    _register(registry, "worker-0", pod_name="aiperf-worker-0", pod_index="0")
    _register(registry, "worker-1", pod_name="aiperf-worker-1", pod_index="1")
    pod_0 = registry.get_services_by_pod("0")
    assert [info.service_id for info in pod_0] == ["worker-0"]
    assert pod_0[0].pod_name == "aiperf-worker-0"


def test_get_stale_services_uses_the_heartbeat_threshold(
    registry: _ServiceRegistry,
) -> None:
    import time

    registry.expect_services({ServiceType.WORKER: 2})
    now_ns = time.time_ns()
    _register(registry, "fresh", seen_ns=now_ns)
    _register(registry, "stale", seen_ns=now_ns - 10_000_000_000)
    stale_ids = [info.service_id for info in registry.get_stale_services(5.0)]
    assert stale_ids == ["stale"]


def test_reset_clears_all_tracking(registry: _ServiceRegistry) -> None:
    registry.expect_services({ServiceType.WORKER: 1})
    _register(registry, "worker-0")
    registry.reset()
    assert not registry.is_registered("worker-0")
    assert registry.get_services() == []
    assert registry.expected_by_type == {}


def test_recoverable_failure_is_cleared_by_replacement_registration(
    registry: _ServiceRegistry,
) -> None:
    _register(registry, "worker_0_0", seen_ns=1)
    registry.fail_service("worker_0_0", ServiceType.WORKER, fatal=False)

    assert not registry.is_registered("worker_0_0")
    assert registry.get_dead_services() == {"worker_0_0": ServiceType.WORKER}
    registry._raise_on_failure()

    _register(registry, "worker_0_0", seen_ns=2)

    assert registry.is_registered("worker_0_0")
    assert registry.get_dead_services() == {}


def test_second_failure_after_replacement_resets_registration(
    registry: _ServiceRegistry,
) -> None:
    _register(registry, "worker_0_0", seen_ns=1)
    registry.fail_service("worker_0_0", ServiceType.WORKER)
    _register(registry, "worker_0_0", seen_ns=2)
    registry.fail_service("worker_0_0", ServiceType.WORKER)

    assert not registry.is_registered("worker_0_0")
    assert registry.get_service("worker_0_0").state == LifecycleState.FAILED


def test_escalate_dead_services_promotes_recoverable_failure(
    registry: _ServiceRegistry,
) -> None:
    registry.fail_service("worker_0_0", ServiceType.WORKER, fatal=False)

    registry.escalate_dead_services()

    with pytest.raises(ServiceProcessDiedError, match="worker_0_0"):
        registry._raise_on_failure()
