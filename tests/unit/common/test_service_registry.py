# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from aiperf.common.enums import LifecycleState, ServiceRegistrationStatus
from aiperf.common.exceptions import (
    ServiceProcessDiedError,
    ServiceRegistrationTimeoutError,
)
from aiperf.common.service_registry import _ServiceRegistry
from aiperf.plugin.enums import ServiceType


@pytest.fixture
def registry():
    """Create a fresh _ServiceRegistry instance for testing."""
    return _ServiceRegistry()


def test_expect_services(registry):
    """Test setting expected services by type."""
    expected = {ServiceType.WORKER: 2, ServiceType.DATASET_MANAGER: 1}
    registry.expect_services(expected)
    assert registry.expected_by_type[ServiceType.WORKER] == 2
    assert registry.expected_by_type[ServiceType.DATASET_MANAGER] == 1


def test_expect_service(registry):
    """Test setting a single expected service by ID and type."""
    registry.expect_service("worker_001", ServiceType.WORKER)
    assert "worker_001" in registry.expected_ids
    assert registry.expected_by_type[ServiceType.WORKER] == 1


def test_expect_service_is_idempotent(registry):
    """Test that calling expect_service twice with the same ID doesn't double-count."""
    registry.expect_service("worker_001", ServiceType.WORKER)
    registry.expect_service("worker_001", ServiceType.WORKER)
    assert registry.expected_by_type[ServiceType.WORKER] == 1


def test_unexpect_service_reverses_expect(registry):
    """Test that unexpect_service undoes expect_service."""
    registry.expect_service("worker_001", ServiceType.WORKER)
    assert registry.expected_by_type[ServiceType.WORKER] == 1
    assert "worker_001" in registry.expected_ids

    registry.unexpect_service("worker_001", ServiceType.WORKER)
    assert registry.expected_by_type[ServiceType.WORKER] == 0
    assert "worker_001" not in registry.expected_ids


def test_unexpect_service_is_idempotent(registry):
    """Test that calling unexpect_service on an unknown ID is a no-op."""
    registry.unexpect_service("nonexistent", ServiceType.WORKER)
    assert ServiceType.WORKER not in registry.expected_by_type


@pytest.mark.asyncio
async def test_unexpect_service_wakes_waiters(registry):
    """Test that unexpecting a service can satisfy wait_for_all."""
    registry.expect_service("w1", ServiceType.WORKER)
    registry.expect_service("w2", ServiceType.WORKER)
    registry.register(
        service_id="w1",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )

    # wait_for_all should block — w2 is expected but not registered
    wait_task = asyncio.create_task(registry.wait_for_all())
    await asyncio.sleep(0.01)
    assert not wait_task.done()

    # Unexpecting w2 should satisfy the condition
    registry.unexpect_service("w2", ServiceType.WORKER)
    await wait_task


def test_expect_service_creates_entry(registry):
    """Test that expect_service creates an UNREGISTERED entry in services."""
    registry.expect_service("worker_001", ServiceType.WORKER)
    registry.expect_service("worker_002", ServiceType.WORKER)
    assert registry.expected_ids == {"worker_001", "worker_002"}
    assert "worker_001" in registry.services
    assert (
        registry.services["worker_001"].registration_status
        == ServiceRegistrationStatus.UNREGISTERED
    )
    assert registry.expected_by_type[ServiceType.WORKER] == 2


def test_register_service(registry):
    """Test registering a service."""
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    assert "worker_001" in registry.services
    assert (
        registry.services["worker_001"].registration_status
        == ServiceRegistrationStatus.REGISTERED
    )
    assert "worker_001" in registry.by_type[ServiceType.WORKER]


def test_register_is_idempotent(registry):
    """Test that re-registering an already-registered service updates state."""
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.STARTING,
    )
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=2000,
        state=LifecycleState.RUNNING,
    )
    info = registry.services["worker_001"]
    assert info.registration_status == ServiceRegistrationStatus.REGISTERED
    assert info.last_seen_ns == 2000
    assert info.state == LifecycleState.RUNNING


def test_register_idempotent_ignores_stale_timestamp(registry):
    """Test that re-registering with an older timestamp doesn't regress state."""
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=2000,
        state=LifecycleState.RUNNING,
    )
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.STARTING,
    )
    info = registry.services["worker_001"]
    assert info.last_seen_ns == 2000
    assert info.state == LifecycleState.RUNNING


def test_register_pre_expected_with_type_mismatch(registry):
    """Test that registering a pre-expected service with a different type updates correctly."""
    registry.expect_service("svc_001", ServiceType.WORKER)
    assert registry.services["svc_001"].service_type == ServiceType.WORKER
    assert registry.expected_by_type[ServiceType.WORKER] == 1

    registry.register(
        service_id="svc_001",
        service_type=ServiceType.DATASET_MANAGER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )

    # service_type should be updated to the registered type
    assert registry.services["svc_001"].service_type == ServiceType.DATASET_MANAGER
    # by_type should reflect the new type, not the old one
    assert "svc_001" in registry.by_type[ServiceType.DATASET_MANAGER]
    assert "svc_001" not in registry.by_type.get(ServiceType.WORKER, set())
    # expected_by_type should be adjusted: old type decremented, new type incremented
    assert registry.expected_by_type.get(ServiceType.WORKER, 0) == 0
    assert registry.expected_by_type[ServiceType.DATASET_MANAGER] == 1


@pytest.mark.asyncio
async def test_register_type_mismatch_does_not_hang_wait_for_all(registry):
    """Test that a type mismatch during registration doesn't permanently block wait_for_all."""
    registry.expect_service("svc_001", ServiceType.WORKER)

    wait_task = asyncio.create_task(registry.wait_for_all())
    await asyncio.sleep(0.01)
    assert not wait_task.done()

    # Service registers with a different type than expected
    registry.register(
        service_id="svc_001",
        service_type=ServiceType.DATASET_MANAGER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    # Should complete instead of hanging — expected_by_type was adjusted
    await wait_task


def test_unregister_service(registry):
    """Test unregistering a service."""
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    registry.unregister("worker_001")

    assert (
        registry.services["worker_001"].registration_status
        == ServiceRegistrationStatus.UNREGISTERED
    )
    assert registry.services["worker_001"].state == LifecycleState.STOPPED


def test_unregister_removes_from_by_type(registry):
    """Test that unregistering a service removes it from by_type."""
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    assert "worker_001" in registry.by_type[ServiceType.WORKER]

    registry.unregister("worker_001")
    assert "worker_001" not in registry.by_type[ServiceType.WORKER]


def test_forget_service(registry):
    """Test forgetting a service removes it entirely."""
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    registry.forget("worker_001")

    assert "worker_001" not in registry.services
    assert "worker_001" not in registry.by_type[ServiceType.WORKER]


def test_forget_decrements_expected_count(registry):
    """Test that forgetting a service decrements expected_by_type."""
    registry.expect_services({ServiceType.WORKER: 2})
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    registry.register(
        service_id="worker_002",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    assert registry.all_registered()

    # Forgetting one should decrement the expected count so we still satisfy
    registry.forget("worker_002")
    assert registry.expected_by_type[ServiceType.WORKER] == 1
    assert registry.all_registered()


def test_forget_does_not_underflow_expected_count(registry):
    """Test that forget doesn't decrement expected_by_type below zero."""
    # Register without prior expectation (count stays 0)
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    registry.forget("worker_001")
    # Should not create a phantom defaultdict entry
    assert ServiceType.WORKER not in registry.expected_by_type


@pytest.mark.asyncio
async def test_forget_wakes_up_waiters(registry):
    """Test that forgetting an unregistered expected service wakes up wait_for_all."""
    registry.expect_service("w1", ServiceType.WORKER)
    registry.expect_service("w2", ServiceType.WORKER)
    registry.register(
        service_id="w1",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )

    # wait_for_all should block — w2 is expected but not registered
    wait_task = asyncio.create_task(registry.wait_for_all())
    await asyncio.sleep(0.01)
    assert not wait_task.done()

    # Forgetting w2 should satisfy the condition and wake up the waiter
    registry.forget("w2")
    await wait_task


def test_expect_service_does_not_overwrite_registered(registry):
    """Test that expect_service skips IDs that are already registered."""
    registry.register(
        service_id="w1",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    # Calling expect_service with an already-registered ID should not overwrite it
    registry.expect_service("w1", ServiceType.WORKER)
    registry.expect_service("w2", ServiceType.WORKER)

    assert (
        registry.services["w1"].registration_status
        == ServiceRegistrationStatus.REGISTERED
    )
    assert registry.services["w1"].first_seen_ns == 1000
    assert (
        registry.services["w2"].registration_status
        == ServiceRegistrationStatus.UNREGISTERED
    )


def test_update_service_existing(registry):
    """Test updating an existing service."""
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    registry.update_service(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        last_seen_ns=2000,
        state=LifecycleState.STOPPING,
    )
    assert registry.services["worker_001"].last_seen_ns == 2000
    assert registry.services["worker_001"].state == LifecycleState.STOPPING


def test_update_service_ignores_stale(registry):
    """Test that stale updates are ignored."""
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=2000,
        state=LifecycleState.RUNNING,
    )
    registry.update_service(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        last_seen_ns=1000,
        state=LifecycleState.STOPPING,
    )
    # Should keep original state since 1000 < 2000
    assert registry.services["worker_001"].state == LifecycleState.RUNNING


def test_update_service_ignores_unknown(registry):
    """Test that updating an unknown service is silently ignored."""
    registry.update_service(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        last_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    assert "worker_001" not in registry.services


def test_update_service_does_not_promote_pre_expected(registry):
    """Test that update_service on a pre-expected service updates timestamps but stays UNREGISTERED.

    This prevents premature registration from heartbeats/status messages
    arriving before the formal RegisterServiceCommand.
    """
    registry.expect_service("worker_001", ServiceType.WORKER)
    assert (
        registry.services["worker_001"].registration_status
        == ServiceRegistrationStatus.UNREGISTERED
    )

    registry.update_service(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        last_seen_ns=1000,
        state=LifecycleState.STARTING,
    )

    info = registry.services["worker_001"]
    assert info.registration_status == ServiceRegistrationStatus.UNREGISTERED
    assert info.last_seen_ns == 1000
    assert info.state == LifecycleState.STARTING
    assert not registry.is_registered("worker_001")

    # Formal registration promotes to REGISTERED
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=2000,
        state=LifecycleState.RUNNING,
    )
    assert registry.is_registered("worker_001")
    assert info.state == LifecycleState.RUNNING


def test_reset(registry):
    """Test that reset clears all state."""
    registry.expect_services({ServiceType.WORKER: 2})
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    registry.reset()

    assert len(registry.services) == 0
    assert len(registry.expected_by_type) == 0
    assert len(registry.expected_ids) == 0
    assert len(registry.by_type) == 0


@pytest.mark.asyncio
async def test_wait_for_all_immediate(registry):
    """Test wait_for_all when services are already registered."""
    registry.expect_services({ServiceType.WORKER: 1})
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    # Should return immediately
    await registry.wait_for_all()


@pytest.mark.asyncio
async def test_wait_for_all_async(registry):
    """Test wait_for_all when services register later."""
    registry.expect_services({ServiceType.WORKER: 1})

    wait_task = asyncio.create_task(registry.wait_for_all())
    await asyncio.sleep(0.01)
    assert not wait_task.done()

    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    await wait_task


@pytest.mark.asyncio
async def test_wait_for_type(registry):
    """Test wait_for_type functionality."""
    registry.expect_services({ServiceType.WORKER: 2})

    wait_task = asyncio.create_task(registry.wait_for_type(ServiceType.WORKER))
    await asyncio.sleep(0.01)
    assert not wait_task.done()

    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    await asyncio.sleep(0.01)
    assert not wait_task.done()

    registry.register(
        service_id="worker_002",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    await wait_task


@pytest.mark.asyncio
async def test_wait_for_type_immediate(registry):
    """Test wait_for_type when condition is already met."""
    registry.expect_services({ServiceType.WORKER: 1})
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    # Should return immediately
    await registry.wait_for_type(ServiceType.WORKER)


@pytest.mark.asyncio
async def test_wait_for_ids(registry):
    """Test wait_for_ids functionality."""
    service_ids = ["worker_001", "manager_001"]

    wait_task = asyncio.create_task(registry.wait_for_ids(service_ids))
    await asyncio.sleep(0.01)
    assert not wait_task.done()

    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    await asyncio.sleep(0.01)
    assert not wait_task.done()

    registry.register(
        service_id="manager_001",
        service_type=ServiceType.DATASET_MANAGER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    await wait_task


def test_get_services(registry):
    """Test get_services method."""
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    registry.register(
        service_id="worker_002",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    registry.register(
        service_id="manager_001",
        service_type=ServiceType.DATASET_MANAGER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )

    workers = registry.get_services(ServiceType.WORKER)
    assert len(workers) == 2

    all_services = registry.get_services()
    assert len(all_services) == 3


def test_get_services_excludes_unregistered(registry):
    """Test that get_services excludes unregistered services in both paths."""
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    registry.register(
        service_id="worker_002",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    registry.unregister("worker_002")

    # By type: only registered workers
    workers = registry.get_services(ServiceType.WORKER)
    assert len(workers) == 1
    assert workers[0].service_id == "worker_001"

    # All services (no type filter): also only registered
    all_services = registry.get_services()
    assert len(all_services) == 1
    assert all_services[0].service_id == "worker_001"


def test_num_registered_does_not_create_phantom_by_type(registry):
    """Test that querying an unseen type doesn't pollute by_type defaultdict."""
    registry.expect_services({ServiceType.WORKER: 1})
    # Querying registration status for an unseen type should not create a phantom entry
    registry.all_types_registered(ServiceType.DATASET_MANAGER)
    assert ServiceType.DATASET_MANAGER not in registry.by_type

    # Same for get_services
    registry.get_services(ServiceType.TIMING_MANAGER)
    assert ServiceType.TIMING_MANAGER not in registry.by_type


def test_get_service(registry):
    """Test get_service method."""
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    retrieved = registry.get_service("worker_001")
    assert retrieved is not None
    assert retrieved.service_id == "worker_001"

    assert registry.get_service("nonexistent") is None


def test_is_registered(registry):
    """Test is_registered method."""
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    assert registry.is_registered("worker_001") is True
    assert registry.is_registered("nonexistent") is False


@pytest.mark.asyncio
async def test_multiple_waiters(registry):
    """Test multiple concurrent waiters for the same condition."""
    registry.expect_services({ServiceType.WORKER: 1})

    wait_task1 = asyncio.create_task(registry.wait_for_type(ServiceType.WORKER))
    wait_task2 = asyncio.create_task(registry.wait_for_type(ServiceType.WORKER))
    wait_task3 = asyncio.create_task(registry.wait_for_all())

    await asyncio.sleep(0.01)
    assert not wait_task1.done()
    assert not wait_task2.done()
    assert not wait_task3.done()

    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )

    await wait_task1
    await wait_task2
    await wait_task3


@pytest.mark.asyncio
async def test_complex_scenario(registry):
    """Test a complex scenario with multiple service types and IDs."""
    registry.expect_services(
        {
            ServiceType.WORKER: 2,
            ServiceType.DATASET_MANAGER: 1,
        }
    )
    registry.expect_service("special_service_001", ServiceType.TIMING_MANAGER)

    wait_all_task = asyncio.create_task(registry.wait_for_all())
    wait_workers_task = asyncio.create_task(registry.wait_for_type(ServiceType.WORKER))
    wait_manager_task = asyncio.create_task(
        registry.wait_for_type(ServiceType.DATASET_MANAGER)
    )
    wait_special_task = asyncio.create_task(
        registry.wait_for_ids(["special_service_001"])
    )

    await asyncio.sleep(0.01)
    assert not any(
        task.done()
        for task in [
            wait_all_task,
            wait_workers_task,
            wait_manager_task,
            wait_special_task,
        ]
    )

    # Register first worker
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    await asyncio.sleep(0.01)
    assert not any(
        task.done()
        for task in [
            wait_all_task,
            wait_workers_task,
            wait_manager_task,
            wait_special_task,
        ]
    )

    # Register second worker
    registry.register(
        service_id="worker_002",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    await asyncio.sleep(0.01)
    assert wait_workers_task.done()
    assert not any(
        task.done() for task in [wait_all_task, wait_manager_task, wait_special_task]
    )

    # Register dataset manager
    registry.register(
        service_id="dataset_manager_001",
        service_type=ServiceType.DATASET_MANAGER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    await asyncio.sleep(0.01)
    assert wait_manager_task.done()
    assert not any(task.done() for task in [wait_all_task, wait_special_task])

    # Register special service
    registry.register(
        service_id="special_service_001",
        service_type=ServiceType.TIMING_MANAGER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )

    await wait_all_task
    await wait_special_task


@pytest.mark.asyncio
@pytest.mark.looptime
async def test_wait_for_all_timeout(registry):
    """Test wait_for_all raises TimeoutError when services don't register."""
    registry.expect_services({ServiceType.WORKER: 1})

    with pytest.raises(ServiceRegistrationTimeoutError, match="worker") as exc_info:
        await registry.wait_for_all(timeout=1)
    assert exc_info.value.missing


# -- fail_service tests --


def test_fail_service_marks_state_as_failed(registry):
    """Test that fail_service sets service state to FAILED and unregisters it."""
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    registry.fail_service("worker_001", ServiceType.WORKER)

    info = registry.services["worker_001"]
    assert info.state == LifecycleState.FAILED
    assert info.registration_status == ServiceRegistrationStatus.UNREGISTERED
    assert "worker_001" not in registry.by_type.get(ServiceType.WORKER, set())


def test_fail_service_stores_error(registry):
    """Test that fail_service stores a ServiceProcessDiedError."""
    registry.fail_service("svc_1", ServiceType.WORKER)
    assert len(registry._failure_errors) == 1
    assert isinstance(registry._failure_errors[0], ServiceProcessDiedError)
    assert registry._failure_event is not None and registry._failure_event.is_set()


def test_fail_service_stores_all_errors(registry):
    """Test that all failures are stored."""
    registry.fail_service("svc_1", ServiceType.WORKER)
    registry.fail_service("svc_2", ServiceType.DATASET_MANAGER)
    assert len(registry._failure_errors) == 2
    assert registry._failure_errors[0].service_id == "svc_1"
    assert registry._failure_errors[1].service_id == "svc_2"


def test_fail_service_for_untracked_service(registry):
    """Test that fail_service works for services not in the services dict."""
    registry.fail_service("unknown_svc", ServiceType.WORKER)
    assert len(registry._failure_errors) == 1
    assert isinstance(registry._failure_errors[0], ServiceProcessDiedError)
    assert registry._failure_event is not None and registry._failure_event.is_set()


def test_fail_service_is_idempotent(registry):
    """Test that calling fail_service twice for the same service_id doesn't duplicate errors."""
    registry.register(
        service_id="svc_1",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    registry.fail_service("svc_1", ServiceType.WORKER)
    registry.fail_service("svc_1", ServiceType.WORKER)
    assert len(registry._failure_errors) == 1


@pytest.mark.asyncio
async def test_wait_for_all_raises_on_failure(registry):
    """Test that wait_for_all raises ServiceProcessDiedError when a service fails."""
    registry.expect_services({ServiceType.WORKER: 1})

    wait_task = asyncio.create_task(registry.wait_for_all())
    await asyncio.sleep(0.01)
    assert not wait_task.done()

    registry.fail_service("worker_001", ServiceType.WORKER)

    with pytest.raises(ServiceProcessDiedError):
        await wait_task


@pytest.mark.asyncio
async def test_wait_for_type_raises_on_failure(registry):
    """Test that wait_for_type raises ServiceProcessDiedError when a service fails."""
    registry.expect_services({ServiceType.WORKER: 1})

    wait_task = asyncio.create_task(registry.wait_for_type(ServiceType.WORKER))
    await asyncio.sleep(0.01)
    assert not wait_task.done()

    registry.fail_service("worker_001", ServiceType.WORKER)

    with pytest.raises(ServiceProcessDiedError):
        await wait_task


@pytest.mark.asyncio
async def test_wait_for_ids_raises_on_failure(registry):
    """Test that wait_for_ids raises ServiceProcessDiedError when a service fails."""
    wait_task = asyncio.create_task(registry.wait_for_ids(["svc_1"]))
    await asyncio.sleep(0.01)
    assert not wait_task.done()

    registry.fail_service("svc_1", ServiceType.WORKER)

    with pytest.raises(ServiceProcessDiedError):
        await wait_task


@pytest.mark.asyncio
async def test_wait_for_all_raises_immediately_if_already_failed(registry):
    """Test that wait_for_all raises immediately if a failure was already recorded."""
    registry.expect_services({ServiceType.WORKER: 1})
    registry.fail_service("worker_001", ServiceType.WORKER)

    with pytest.raises(ServiceProcessDiedError):
        await registry.wait_for_all()


@pytest.mark.asyncio
async def test_failure_wakes_multiple_waiters(registry):
    """Test that a failure wakes up all concurrent waiters."""
    registry.expect_services({ServiceType.WORKER: 2, ServiceType.DATASET_MANAGER: 1})

    task_all = asyncio.create_task(registry.wait_for_all())
    task_type = asyncio.create_task(registry.wait_for_type(ServiceType.WORKER))
    task_ids = asyncio.create_task(registry.wait_for_ids(["w1"]))
    await asyncio.sleep(0.01)

    registry.fail_service("w1", ServiceType.WORKER)

    for task in (task_all, task_type, task_ids):
        with pytest.raises(ServiceProcessDiedError):
            await task


@pytest.mark.asyncio
async def test_reset_clears_failure_state(registry):
    """Test that reset clears failure state so subsequent waits don't raise."""
    registry.fail_service("svc_1", ServiceType.WORKER)
    registry.reset()

    assert registry._failure_errors == []
    assert registry._failure_event is None

    # A new wait should not raise immediately
    registry.expect_services({ServiceType.WORKER: 1})
    registry.register(
        service_id="w1",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )
    await registry.wait_for_all()


# -- Timeout diagnostics tests --


@pytest.mark.asyncio
@pytest.mark.looptime
async def test_timeout_error_includes_missing_service_details(registry):
    """Test that timeout error message includes which services are missing."""
    registry.expect_services({ServiceType.WORKER: 2, ServiceType.DATASET_MANAGER: 1})
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=1000,
        state=LifecycleState.RUNNING,
    )

    with pytest.raises(ServiceRegistrationTimeoutError) as exc_info:
        await registry.wait_for_all(timeout=1)

    assert "worker: 1/2" in str(exc_info.value)
    assert "dataset_manager: 0/1" in str(exc_info.value)
    assert ServiceType.WORKER in exc_info.value.missing
    assert ServiceType.DATASET_MANAGER in exc_info.value.missing
    assert exc_info.value.missing[ServiceType.WORKER] == 1
    assert exc_info.value.missing[ServiceType.DATASET_MANAGER] == 1


@pytest.mark.asyncio
@pytest.mark.looptime
async def test_wait_for_type_timeout_includes_diagnostics(registry):
    """Test that wait_for_type timeout includes service type in error."""
    registry.expect_services({ServiceType.WORKER: 2})

    with pytest.raises(ServiceRegistrationTimeoutError, match="worker"):
        await registry.wait_for_type(ServiceType.WORKER, timeout=1)


@pytest.mark.asyncio
@pytest.mark.looptime
async def test_wait_for_ids_timeout_includes_missing_ids(registry):
    """Test that wait_for_ids timeout includes which IDs are missing."""
    with pytest.raises(ServiceRegistrationTimeoutError, match="svc_2"):
        await registry.wait_for_ids(["svc_1", "svc_2"], timeout=1)


@pytest.mark.asyncio
@pytest.mark.looptime
async def test_timeout_error_is_catchable_as_timeout_error(registry):
    """Test that ServiceRegistrationTimeoutError is catchable as TimeoutError."""
    registry.expect_services({ServiceType.WORKER: 1})

    with pytest.raises(TimeoutError):
        await registry.wait_for_all(timeout=1)


# -- Condition re-verification tests --


@pytest.mark.asyncio
async def test_raise_on_failure_raises_first_error(registry):
    """Test that _raise_on_failure raises the first stored error."""
    registry.expect_services({ServiceType.WORKER: 1})
    registry.fail_service("svc_1", ServiceType.WORKER)
    registry.fail_service("svc_2", ServiceType.DATASET_MANAGER)

    with pytest.raises(ServiceProcessDiedError) as exc_info:
        await registry.wait_for_all()

    assert exc_info.value.service_id == "svc_1"


def test_raise_on_failure_logs_all_failures(registry, caplog):
    """Test that _raise_on_failure logs all failures when multiple exist."""
    registry.fail_service("svc_1", ServiceType.WORKER)
    registry.fail_service("svc_2", ServiceType.DATASET_MANAGER)

    with pytest.raises(ServiceProcessDiedError):
        registry._raise_on_failure()

    assert "2 service(s) failed" in caplog.text
    assert "svc_1" in caplog.text
    assert "svc_2" in caplog.text


def test_raise_on_failure_skips_summary_log_for_single_failure(registry, caplog):
    """Test that a single failure doesn't produce a redundant summary log."""
    registry.fail_service("svc_1", ServiceType.WORKER)

    with pytest.raises(ServiceProcessDiedError):
        registry._raise_on_failure()

    assert "service(s) failed" not in caplog.text


# -- reset() wakes waiters tests --


@pytest.mark.asyncio
async def test_reset_wakes_pending_wait_for_all(registry):
    """Test that reset() unblocks a pending wait_for_all."""
    registry.expect_services({ServiceType.WORKER: 1})

    wait_task = asyncio.create_task(registry.wait_for_all())
    await asyncio.sleep(0.01)
    assert not wait_task.done()

    registry.reset()
    await asyncio.sleep(0.01)

    # After reset, the event fires and all_registered() is vacuously true
    # (no expected services remain), so the task completes normally
    assert wait_task.done()


@pytest.mark.asyncio
async def test_reset_wakes_pending_wait_for_type(registry):
    """Test that reset() unblocks a pending wait_for_type."""
    registry.expect_services({ServiceType.WORKER: 1})

    wait_task = asyncio.create_task(registry.wait_for_type(ServiceType.WORKER))
    await asyncio.sleep(0.01)
    assert not wait_task.done()

    registry.reset()
    await asyncio.sleep(0.01)

    assert wait_task.done()


# -- Staleness detection tests --


def test_get_stale_services_returns_stale(registry):
    """Test that get_stale_services detects services past the threshold."""
    import time

    old_ns = time.time_ns() - 20_000_000_000  # 20 seconds ago
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=old_ns,
        state=LifecycleState.RUNNING,
    )

    stale = registry.get_stale_services(threshold_sec=10.0)
    assert len(stale) == 1
    assert stale[0].service_id == "worker_001"


def test_get_stale_services_excludes_fresh(registry):
    """Test that get_stale_services excludes recently seen services."""
    import time

    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=time.time_ns(),
        state=LifecycleState.RUNNING,
    )

    stale = registry.get_stale_services(threshold_sec=10.0)
    assert len(stale) == 0


def test_get_stale_services_excludes_unregistered(registry):
    """Test that get_stale_services excludes unregistered services."""
    import time

    old_ns = time.time_ns() - 20_000_000_000
    registry.register(
        service_id="worker_001",
        service_type=ServiceType.WORKER,
        first_seen_ns=old_ns,
        state=LifecycleState.RUNNING,
    )
    registry.unregister("worker_001")

    stale = registry.get_stale_services(threshold_sec=10.0)
    assert len(stale) == 0


def test_get_stale_services_excludes_no_last_seen(registry):
    """Test that services with no last_seen_ns are not considered stale."""
    registry.expect_service("worker_001", ServiceType.WORKER)

    stale = registry.get_stale_services(threshold_sec=10.0)
    assert len(stale) == 0
