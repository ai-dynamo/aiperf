# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the process-wide service registry and its async waiting mixin."""

import asyncio
import inspect
import time
from unittest.mock import patch

import pytest

from aiperf.common.enums import LifecycleState, ServiceRegistrationStatus
from aiperf.common.environment import Environment
from aiperf.common.exceptions import (
    ServiceProcessDiedError,
    ServiceRegistrationTimeoutError,
)
from aiperf.common.service_registry import ServiceRegistry, _ServiceRegistry
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
    state: LifecycleState = LifecycleState.RUNNING,
    **kwargs,
) -> None:
    """Register a service with the required keyword-only arguments."""
    registry.register(
        service_id=service_id,
        service_type=service_type,
        first_seen_ns=seen_ns,
        state=state,
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
    _register(
        registry,
        "worker-0",
        seen_ns=99,
        state=LifecycleState.STOPPING,
    )
    assert registry.is_registered("worker-0")
    info = registry.get_service("worker-0")
    assert info.last_seen_ns == 99
    assert info.state == LifecycleState.STOPPING


def test_update_service_ignores_unknown_and_stale_updates(
    registry: _ServiceRegistry,
) -> None:
    registry.expect_services({ServiceType.WORKER: 1})
    _register(registry, "worker-0", seen_ns=10)
    registry.update_service(
        "ghost",
        last_seen_ns=50,
        state=LifecycleState.RUNNING,
        seq=1,
    )
    assert registry.get_service("ghost") is None

    registry.update_service(
        "worker-0",
        last_seen_ns=5,
        state=LifecycleState.STOPPING,
        seq=5,
    )
    info = registry.get_service("worker-0")
    assert info.last_seen_ns == 5
    assert info.state == LifecycleState.STOPPING

    # A seq that is not strictly greater than the last-applied one is dropped
    # whole, even though its last_seen_ns is larger.
    registry.update_service(
        "worker-0",
        last_seen_ns=50,
        state=LifecycleState.RUNNING,
        seq=3,
    )
    info = registry.get_service("worker-0")
    assert info.last_seen_ns == 5
    assert info.state == LifecycleState.STOPPING

    registry.update_service(
        "worker-0",
        last_seen_ns=50,
        state=LifecycleState.RUNNING,
        seq=6,
    )
    info = registry.get_service("worker-0")
    assert info.last_seen_ns == 50
    assert info.state == LifecycleState.RUNNING


def test_update_service_does_not_accept_a_service_type_it_never_reads(
    registry: _ServiceRegistry,
) -> None:
    """A parameter that is accepted and dropped reads like an enforced contract.

    ``update_service`` looks the canonical ``ServiceRunInfo`` up by
    ``service_id`` and rewrites only state/timestamp/seq; ``register`` is the
    sole owner of ``service_type`` and ``by_type`` bucketing. Accepting a
    ``service_type`` here implied a validation that never happened.
    """
    params = inspect.signature(registry.update_service).parameters
    assert "service_type" not in params


def test_update_service_wallclock_ordering_lets_late_stale_update_regress_state(
    registry: _ServiceRegistry,
) -> None:
    """Bug (fixed): ``last_seen_ns`` is stamped by the controller at receipt
    time, so it is monotone by construction and can never catch a message
    that is logically stale but arrives (and gets stamped) after a logically
    newer one -- e.g. a Heartbeat sent before a StatusUpdate gets stuck
    behind an HWM backlog and is delivered to the controller after the
    StatusUpdate. Ordering must be decided by the sender-stamped ``seq``,
    not by the receipt-time ``last_seen_ns``.
    """
    registry.expect_services({ServiceType.WORKER: 1})
    _register(registry, "worker-0", seen_ns=1)

    # The StatusUpdate (logically the newest signal, seq=5) reaches the
    # controller first and is stamped with the controller's receipt clock.
    registry.update_service(
        "worker-0",
        last_seen_ns=200,
        state=LifecycleState.STOPPING,
        seq=5,
    )
    # A Heartbeat that was actually SENT before the StatusUpdate (seq=4)
    # arrives late, but still receives a *later* wall-clock stamp because
    # time only moves forward at the controller.
    registry.update_service(
        "worker-0",
        last_seen_ns=300,
        state=LifecycleState.RUNNING,
        seq=4,
    )

    info = registry.get_service("worker-0")
    assert info.state == LifecycleState.STOPPING, (
        "a wall-clock-stamped-but-logically-stale update must not regress state"
    )


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
    now_ns = time.monotonic_ns()
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


@pytest.mark.asyncio
async def test_retracted_failure_does_not_latch_wait_for_type(
    registry: _ServiceRegistry,
) -> None:
    """A cleared failure must not leave wait_for_type returning instantly.

    ``fail_service(fatal=True)`` force-sets every registration event so blocked
    callers re-check and see the failure. When the failure is then retracted by
    a replacement registration, ``_disarm_stale_waiters`` has to clear those
    events again -- otherwise the next wait wakes on the stale ``set()`` and
    reports a registration timeout that never elapsed.
    """
    registry.expect_services({ServiceType.WORKER: 2})

    with pytest.raises(ServiceRegistrationTimeoutError):
        await registry.wait_for_type(ServiceType.WORKER, timeout=0.01)

    registry.fail_service("worker_0_1", ServiceType.WORKER, fatal=True)
    assert registry._type_events[ServiceType.WORKER].is_set()

    _register(registry, "worker_0_1", seen_ns=2)
    assert not registry._type_events[ServiceType.WORKER].is_set()

    started = time.perf_counter()
    with pytest.raises(ServiceRegistrationTimeoutError) as excinfo:
        await registry.wait_for_type(ServiceType.WORKER, timeout=0.05)
    elapsed = time.perf_counter() - started

    # Scheduler resolution can finish a nominal 50 ms wait just below the
    # requested boundary; an immediate wake would still be well below 45 ms.
    assert elapsed >= 0.045
    assert excinfo.value.timeout_sec == 0.05


@pytest.mark.asyncio
async def test_premature_wake_reports_elapsed_not_nominal_timeout(
    registry: _ServiceRegistry,
) -> None:
    """The after-waking branch must not claim a window that never elapsed."""
    registry.expect_services({ServiceType.WORKER: 2})
    event = registry._type_events.setdefault(ServiceType.WORKER, asyncio.Event())
    event.set()

    with pytest.raises(ServiceRegistrationTimeoutError) as excinfo:
        await registry.wait_for_type(ServiceType.WORKER, timeout=600.0)

    assert "after waking" in str(excinfo.value)
    assert excinfo.value.timeout_sec is not None
    assert excinfo.value.timeout_sec < 600.0


def test_get_all_registered_ids_returns_only_registered_services() -> None:
    ServiceRegistry.reset()
    ServiceRegistry.expect_service("pending-1", ServiceType.WORKER)
    ServiceRegistry.register(
        service_id="live-1",
        service_type=ServiceType.WORKER,
        first_seen_ns=1,
        state=LifecycleState.RUNNING,
    )
    assert ServiceRegistry.get_all_registered_ids() == {"live-1"}


def test_register_type_mismatch_correction_preserves_total_expected(
    registry: _ServiceRegistry,
) -> None:
    """A pre-expected service that registers under a different type must not
    drop _total_expected -- only expected_by_type should shift between types."""
    registry.expect_service("worker-0", ServiceType.WORKER)
    registry.expect_services({ServiceType.RECORD_PROCESSOR: 1})
    assert registry._total_expected == 2

    _register(registry, "worker-0", service_type=ServiceType.RECORD_PROCESSOR)

    assert registry._total_expected == 2
    assert sum(registry.expected_by_type.values()) == registry._total_expected


@pytest.mark.asyncio
async def test_progress_log_interval_is_read_at_wait_time(
    registry: _ServiceRegistry,
) -> None:
    """The registry is a module-level singleton constructed at import time.

    Binding the interval to a class attribute froze it before any test or
    subprocess could set ``AIPERF_SERVICE_REGISTRATION_PROGRESS_LOG_INTERVAL``,
    which is inconsistent with every other ``Environment`` read in this file.
    """
    registry.expect_services({ServiceType.WORKER: 1})
    with (
        patch.object(Environment.SERVICE, "REGISTRATION_PROGRESS_LOG_INTERVAL", 0.01),
        patch.object(registry, "_log_waiting_for") as log_waiting,
        pytest.raises(ServiceRegistrationTimeoutError),
    ):
        await registry.wait_for_all(timeout=0.2)

    # A frozen 5.0s default yields one wait_for that consumes the whole
    # timeout, so no progress line is ever emitted.
    assert log_waiting.call_count > 1
