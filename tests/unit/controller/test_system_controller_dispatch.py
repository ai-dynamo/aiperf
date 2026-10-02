# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-channel dispatch on the SystemController.

Every behavior asserted here was previously carried by the
``@on_command(REGISTER_SERVICE)`` / ``@on_message(HEARTBEAT|STATUS)`` pub-sub
handlers; the transport changed, the semantics must not.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from aiperf.common.control_structs import (
    Command,
    CommandAck,
    CommandErr,
    CommandUnhandled,
    Heartbeat,
    Registration,
    RegistrationAck,
    ReRegisterRequest,
    StatusUpdate,
)
from aiperf.common.enums import (
    CommandType,
    LifecycleState,
    ServiceRegistrationStatus,
    SystemState,
    make_result_producer_capability,
    parse_result_producer_capability,
)
from aiperf.common.exceptions import LifecycleOperationError
from aiperf.common.hooks import AIPerfHook
from aiperf.common.models import ErrorDetails, ServiceRunInfo
from aiperf.common.models.error_models import ExitErrorInfo
from aiperf.common.service_registry import ServiceRegistry
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType


def _registration(
    sid: str = "svc-1",
    *,
    rid: str = "r-1",
    stype: ServiceType = ServiceType.WORKER,
    state: LifecycleState = LifecycleState.RUNNING,
    pod_name: str | None = None,
    capabilities: tuple[str, ...] = (),
) -> Registration:
    return Registration(
        sid=sid,
        rid=rid,
        stype=str(stype),
        state=str(state),
        pod_name=pod_name,
        capabilities=capabilities,
    )


@pytest.fixture
def controller(system_controller: SystemController) -> SystemController:
    """The shared controller fixture with empty, real service-manager maps."""
    system_controller.service_manager.service_id_map = {}
    system_controller.service_manager.service_map = {}
    return system_controller


@pytest.mark.asyncio
async def test_registration_returns_ack_with_matching_rid(
    controller: SystemController,
) -> None:
    ack = await controller._handle_control_message("svc-1", _registration())
    assert ack == RegistrationAck(rid="r-1")
    assert ServiceRegistry.is_registered("svc-1")
    assert "svc-1" in controller.service_manager.service_id_map
    assert [
        info.service_id
        for info in controller.service_manager.service_map[ServiceType.WORKER]
    ] == ["svc-1"]


@pytest.mark.asyncio
async def test_registration_joins_capabilities_into_result_barrier(
    controller: SystemController,
) -> None:
    """Behavior 4: capabilities -> result-join barrier."""
    await controller._handle_control_message(
        "svc-1",
        _registration(
            stype=ServiceType.GPU_TELEMETRY_MANAGER,
            capabilities=("result_producer:telemetry",),
        ),
    )
    assert "telemetry" in controller._result_join_coordinator.pending_domains


@pytest.mark.asyncio
async def test_registration_ignores_unprefixed_capability(
    controller: SystemController,
) -> None:
    """A capability that is not a result-producer tag must not join the barrier."""
    await controller._handle_control_message(
        "svc-1", _registration(capabilities=("telemetry",))
    )
    assert controller._result_join_coordinator.pending_domains == ()


@pytest.mark.asyncio
async def test_registration_clears_reaped_service_id(
    controller: SystemController,
) -> None:
    """Behavior 2: a service alive again must rejoin command fan-out."""
    controller._reaped_service_ids.add("svc-1")
    await controller._handle_control_message("svc-1", _registration())
    assert "svc-1" not in controller._reaped_service_ids


@pytest.mark.asyncio
async def test_registration_rekeys_service_map_on_service_type_change(
    controller: SystemController,
) -> None:
    """Behavior 3: a changed service type must not leave a stale peer entry."""
    await controller._handle_control_message(
        "svc-1", _registration(stype=ServiceType.WORKER)
    )
    await controller._handle_control_message(
        "svc-1", _registration(rid="r-2", stype=ServiceType.RECORD_PROCESSOR)
    )

    assert controller.service_manager.service_map[ServiceType.WORKER] == []
    assert [
        info.service_id
        for info in controller.service_manager.service_map[ServiceType.RECORD_PROCESSOR]
    ] == ["svc-1"]
    assert controller.service_manager.service_id_map[
        "svc-1"
    ] is ServiceRegistry.get_service("svc-1")


@pytest.mark.asyncio
async def test_registration_raises_when_registry_loses_the_registration(
    controller: SystemController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behavior 6: silently continuing here would publish a half-registered service."""
    monkeypatch.setattr(ServiceRegistry, "get_service", lambda _sid: None)
    with pytest.raises(RuntimeError, match="lost registration"):
        await controller._handle_control_message("svc-1", _registration())


@pytest.mark.asyncio
async def test_failed_worker_group_reregistration_triggers_configure(
    controller: SystemController,
) -> None:
    """Behavior 1 (FAILED branch) + behavior 5 + behavior 7."""
    controller._system_state = SystemState.PROFILING
    controller._configure_replacement_worker_group = AsyncMock()
    message = _registration(
        sid="wgm-0", stype=ServiceType.WORKER_GROUP_MANAGER, pod_name="pod-a"
    )
    await controller._handle_control_message("wgm-0", message)
    ServiceRegistry.fail_service("wgm-0", ServiceType.WORKER_GROUP_MANAGER, fatal=False)

    await controller._handle_control_message("wgm-0", message)
    await asyncio.sleep(0)

    assert "wgm-0" in controller._replacement_configuring_ids
    controller._configure_replacement_worker_group.assert_called_once_with("wgm-0")


@pytest.mark.asyncio
async def test_worker_group_reregistration_from_new_pod_triggers_configure(
    controller: SystemController,
) -> None:
    """Behavior 1 (changed pod_name branch)."""
    controller._system_state = SystemState.PROFILING
    controller._configure_replacement_worker_group = AsyncMock()
    await controller._handle_control_message(
        "wgm-0",
        _registration(
            sid="wgm-0", stype=ServiceType.WORKER_GROUP_MANAGER, pod_name="pod-a"
        ),
    )
    await controller._handle_control_message(
        "wgm-0",
        _registration(
            sid="wgm-0",
            rid="r-2",
            stype=ServiceType.WORKER_GROUP_MANAGER,
            pod_name="pod-b",
        ),
    )
    await asyncio.sleep(0)

    controller._configure_replacement_worker_group.assert_called_once_with("wgm-0")


@pytest.mark.asyncio
async def test_replacement_configure_is_skipped_while_initializing(
    controller: SystemController,
) -> None:
    """Behavior 7: a replacement seen before startup finishes is configured by
    the normal bulk-configure pass, not by the per-service replacement path."""
    controller._system_state = SystemState.INITIALIZING
    controller._configure_replacement_worker_group = AsyncMock()
    await controller._handle_control_message(
        "wgm-0",
        _registration(
            sid="wgm-0", stype=ServiceType.WORKER_GROUP_MANAGER, pod_name="pod-a"
        ),
    )
    await controller._handle_control_message(
        "wgm-0",
        _registration(
            sid="wgm-0",
            rid="r-2",
            stype=ServiceType.WORKER_GROUP_MANAGER,
            pod_name="pod-b",
        ),
    )
    await asyncio.sleep(0)

    controller._configure_replacement_worker_group.assert_not_called()
    assert "wgm-0" not in controller._replacement_configuring_ids


@pytest.mark.asyncio
async def test_replacement_configure_is_not_started_twice(
    controller: SystemController,
) -> None:
    """Behavior 5: the in-flight guard survives a second replacement signal."""
    controller._system_state = SystemState.PROFILING
    controller._configure_replacement_worker_group = AsyncMock()
    for index, pod in enumerate(("pod-a", "pod-b", "pod-c")):
        await controller._handle_control_message(
            "wgm-0",
            _registration(
                sid="wgm-0",
                rid=f"r-{index}",
                stype=ServiceType.WORKER_GROUP_MANAGER,
                pod_name=pod,
            ),
        )
    await asyncio.sleep(0)

    controller._configure_replacement_worker_group.assert_called_once_with("wgm-0")


@pytest.mark.asyncio
async def test_heartbeat_stamps_last_seen_from_controller_clock(
    controller: SystemController,
) -> None:
    """Invariant I1: the wire carries no timestamp; the controller stamps.

    The stamp comes from the monotonic ``liveness_clock_ns``, the same clock
    ``get_stale_services`` ages it against.
    """
    await controller._handle_control_message("svc-1", _registration())
    before = time.monotonic_ns()
    result = await controller._handle_control_message(
        "svc-1",
        Heartbeat(
            sid="svc-1",
            stype=str(ServiceType.WORKER),
            state=str(LifecycleState.RUNNING),
        ),
    )
    after = time.monotonic_ns()

    assert result is None, "heartbeat is fire-and-forget"
    info = ServiceRegistry.get_service("svc-1")
    assert before <= info.last_seen_ns <= after
    assert not any(
        field in Heartbeat.__struct_fields__
        for field in ("ts", "request_ns", "sent_ns")
    ), "Heartbeat must not carry a sender-stamped timestamp"


@pytest.mark.asyncio
async def test_heartbeat_from_unknown_service_is_ignored(
    controller: SystemController,
) -> None:
    result = await controller._handle_control_message(
        "ghost",
        Heartbeat(
            sid="ghost",
            stype=str(ServiceType.WORKER),
            state=str(LifecycleState.RUNNING),
        ),
    )
    assert result is None
    assert ServiceRegistry.get_service("ghost") is None


@pytest.mark.asyncio
async def test_heartbeat_from_unknown_service_nudges_it_to_reregister(
    controller: SystemController,
) -> None:
    """Bug A: a controller ROUTER restart empties the ServiceRegistry while a
    surviving service keeps heartbeating. Without a nudge back to the sender,
    the service is orphaned forever -- it never learns it must re-register.
    """
    await controller._handle_control_message(
        "ghost",
        Heartbeat(
            sid="ghost",
            stype=str(ServiceType.WORKER),
            state=str(LifecycleState.RUNNING),
        ),
    )
    await asyncio.sleep(0)
    controller.control_router.send_to.assert_awaited()
    identity, struct = controller.control_router.send_to.call_args.args
    assert identity == "ghost"
    assert isinstance(struct, ReRegisterRequest)
    assert struct.sid == "ghost"


@pytest.mark.asyncio
async def test_status_update_from_unknown_service_nudges_it_to_reregister(
    controller: SystemController,
) -> None:
    await controller._handle_control_message(
        "ghost",
        StatusUpdate(
            sid="ghost",
            stype=str(ServiceType.WORKER),
            state=str(LifecycleState.RUNNING),
        ),
    )
    await asyncio.sleep(0)
    controller.control_router.send_to.assert_awaited()
    identity, struct = controller.control_router.send_to.call_args.args
    assert identity == "ghost"
    assert isinstance(struct, ReRegisterRequest)


@pytest.mark.asyncio
async def test_status_update_records_new_state(controller: SystemController) -> None:
    await controller._handle_control_message("svc-1", _registration())
    result = await controller._handle_control_message(
        "svc-1",
        StatusUpdate(
            sid="svc-1",
            stype=str(ServiceType.WORKER),
            state=str(LifecycleState.STOPPING),
        ),
    )
    assert result is None
    assert ServiceRegistry.get_service("svc-1").state == LifecycleState.STOPPING


@pytest.mark.asyncio
async def test_status_update_does_not_write_through_service_id_map(
    controller: SystemController,
) -> None:
    """Invariant I2: the registry is the sole writer of state/last_seen_ns.

    ``service_id_map`` holds the very ``ServiceRunInfo`` the registry owns, so a
    handler that wrote to it directly would pre-satisfy the ordering guard in
    ``update_service`` and let a stale update through.
    """
    await controller._handle_control_message("svc-1", _registration())
    info = controller.service_manager.service_id_map["svc-1"]
    assert info is ServiceRegistry.get_service("svc-1")

    await controller._handle_control_message(
        "svc-1",
        StatusUpdate(
            sid="svc-1",
            stype=str(ServiceType.WORKER),
            state=str(LifecycleState.STOPPING),
        ),
    )
    newest_ns = info.last_seen_ns

    # An update the transport delivered late must be dropped whole: its seq
    # (0, the StatusUpdate default) is not strictly greater than the seq
    # already applied above.
    ServiceRegistry.update_service(
        "svc-1",
        last_seen_ns=newest_ns - 1,
        state=LifecycleState.RUNNING,
        seq=0,
    )
    assert info.state == LifecycleState.STOPPING
    assert info.last_seen_ns == newest_ns


@pytest.mark.asyncio
async def test_status_update_from_unknown_service_is_ignored(
    controller: SystemController,
) -> None:
    result = await controller._handle_control_message(
        "ghost",
        StatusUpdate(
            sid="ghost",
            stype=str(ServiceType.WORKER),
            state=str(LifecycleState.RUNNING),
        ),
    )
    assert result is None
    assert ServiceRegistry.get_service("ghost") is None


def test_result_producer_capability_prefix_is_pinned() -> None:
    """The capability wire format is stringly-typed and unenforced.

    ``_on_registration`` is its first consumer, so pin the exact prefix: a
    silent change would drop producers out of the shutdown barrier with no
    error anywhere.
    """
    assert make_result_producer_capability("telemetry") == "result_producer:telemetry"
    assert parse_result_producer_capability("result_producer:telemetry") == "telemetry"
    assert parse_result_producer_capability("telemetry") is None
    assert parse_result_producer_capability("result_producer:") is None


def test_is_replacement_worker_group_registration_ignores_other_types(
    controller: SystemController,
) -> None:
    """Only worker-group managers get the replacement treatment."""
    prior = ServiceRunInfo(
        service_id="svc-1",
        service_type=ServiceType.WORKER,
        registration_status=ServiceRegistrationStatus.REGISTERED,
        first_seen_ns=1,
        last_seen_ns=1,
        state=LifecycleState.FAILED,
    )
    assert not controller._is_replacement_worker_group_registration(
        _registration(stype=ServiceType.WORKER, pod_name="pod-b"), prior, True
    )


def test_control_router_is_not_lifecycle_managed_by_comms(benchmark_run) -> None:
    """Invariant: comms must not own the control ROUTER.

    ``FakeCommunication`` never attaches child lifecycles at all, so only a
    real ZMQ communication layer can catch a regression in this wiring.
    """
    from types import SimpleNamespace

    from aiperf.zmq.zmq_comms import ZMQIPCCommunication

    comms = ZMQIPCCommunication(config=benchmark_run.comm_config)
    stub = SimpleNamespace(comms=comms, run=benchmark_run)
    SystemController._init_control_router(stub)

    assert stub.control_router is not None
    assert stub.control_router not in comms._children


@pytest.mark.asyncio
async def test_send_control_command_to_all_preserves_input_order(controller) -> None:
    """Invariant I9: two callers zip(service_ids, responses, strict=True)."""
    order = ["c", "a", "b"]

    async def fake_request_to(identity, struct, timeout):
        if identity == "c":
            await asyncio.sleep(0.05)  # finishes last, must still come first
        return CommandAck(cid=struct.cid, cmd=struct.cmd, sid=identity)

    controller.control_router.request_to = fake_request_to
    responses = await controller._send_control_command_to_all(
        CommandType.FINALIZE_ARTIFACTS, order, timeout=1.0
    )
    assert [r.sid for r in responses] == order


@pytest.mark.asyncio
async def test_send_control_command_to_all_reports_timeout_as_error_details(
    controller,
) -> None:
    async def fake_request_to(identity, struct, timeout):
        raise TimeoutError

    controller.control_router.request_to = fake_request_to
    responses = await controller._send_control_command_to_all(
        CommandType.PROFILE_START, ["a"], timeout=0.01
    )
    assert isinstance(responses[0], ErrorDetails)


@pytest.mark.asyncio
async def test_control_command_arm_dispatches_to_on_command_hooks(controller) -> None:
    """The Command arm must reach the hooks, not log-and-drop.

    A stub that returns None short-circuits the real GET_POD_STATES handler,
    which needs a populated worker cache.
    """
    calls: list[Command] = []

    async def handler(message: Command) -> None:
        calls.append(message)

    for hook in controller.get_hooks(AIPerfHook.ON_COMMAND):
        if CommandType.GET_POD_STATES in (hook.resolve_params(controller) or ()):
            hook.func = handler
            break
    else:
        pytest.fail("controller has no GET_POD_STATES @on_command hook")

    response = await controller._handle_control_message(
        "svc-1", Command(cid="c-1", cmd=CommandType.GET_POD_STATES)
    )
    assert [c.cid for c in calls] == ["c-1"]
    assert response == CommandAck(
        cid="c-1", cmd=CommandType.GET_POD_STATES, sid=controller.service_id
    )


@pytest.mark.asyncio
async def test_dispatch_control_command_unmatched_returns_command_unhandled(
    controller,
) -> None:
    """Invariant I7: no handler is a failure, not an ack."""
    response = await controller._handle_control_message(
        "svc-1", Command(cid="c-1", cmd="a_command_with_no_handler")
    )
    assert isinstance(response, CommandUnhandled)
    assert response.cid == "c-1"
    assert response.cmd == "a_command_with_no_handler"
    assert response.sid == controller.service_id


@pytest.mark.asyncio
async def test_dispatch_control_command_raising_hook_returns_command_err(
    controller,
) -> None:
    async def handler(message: Command) -> None:
        raise ValueError("boom")

    for hook in controller.get_hooks(AIPerfHook.ON_COMMAND):
        if CommandType.GET_POD_STATES in (hook.resolve_params(controller) or ()):
            hook.func = handler
            break

    response = await controller._handle_control_message(
        "svc-1", Command(cid="c-1", cmd=CommandType.GET_POD_STATES)
    )
    assert isinstance(response, CommandErr)
    assert response.cmd == CommandType.GET_POD_STATES
    assert response.error == "boom"
    assert "ValueError" in response.traceback


@pytest.mark.asyncio
async def test_finalize_artifacts_logs_unhandled_local_processor(controller) -> None:
    """Local artifact ACK failures do not discard already-flushed results."""
    controller.service_manager.service_id_map = {
        "rp-1": ServiceRunInfo(
            service_id="rp-1",
            service_type=ServiceType.RECORD_PROCESSOR,
            registration_status=ServiceRegistrationStatus.REGISTERED,
            first_seen_ns=1,
            last_seen_ns=1,
            state=LifecycleState.RUNNING,
        )
    }
    controller._exit_errors = []

    async def fake_request_to(identity, struct, timeout):
        return CommandUnhandled(cid=struct.cid, cmd=struct.cmd, sid=identity)

    controller.control_router.request_to = fake_request_to
    await controller._handle_finalize_artifacts_command(
        Command(cid="c-1", cmd=CommandType.FINALIZE_ARTIFACTS)
    )
    assert controller._exit_errors == []


@pytest.mark.asyncio
async def test_finalize_artifacts_logs_unpopulated_command_locally(
    controller,
) -> None:
    """A malformed local ACK is logged without failing already-flushed results."""
    controller.service_manager.service_id_map = {
        "rp-1": ServiceRunInfo(
            service_id="rp-1",
            service_type=ServiceType.RECORD_PROCESSOR,
            registration_status=ServiceRegistrationStatus.REGISTERED,
            first_seen_ns=1,
            last_seen_ns=1,
            state=LifecycleState.RUNNING,
        )
    }
    controller._exit_errors = []

    async def fake_request_to(identity, struct, timeout):
        return CommandAck(cid=struct.cid, sid=identity)  # cmd left at ""

    controller.control_router.request_to = fake_request_to
    await controller._handle_finalize_artifacts_command(
        Command(cid="c-1", cmd=CommandType.FINALIZE_ARTIFACTS)
    )
    assert controller._exit_errors == []


@pytest.mark.asyncio
async def test_finalize_artifacts_accepts_matching_ack(controller) -> None:
    """Invariant I8: the ack identity check passes on cmd + sid."""
    controller.service_manager.service_id_map = {
        "rp-1": ServiceRunInfo(
            service_id="rp-1",
            service_type=ServiceType.RECORD_PROCESSOR,
            registration_status=ServiceRegistrationStatus.REGISTERED,
            first_seen_ns=1,
            last_seen_ns=1,
            state=LifecycleState.RUNNING,
        )
    }
    controller._exit_errors = []

    async def fake_request_to(identity, struct, timeout):
        return CommandAck(cid=struct.cid, cmd=struct.cmd, sid=identity)

    controller.control_router.request_to = fake_request_to
    await controller._handle_finalize_artifacts_command(
        Command(cid="c-1", cmd=CommandType.FINALIZE_ARTIFACTS)
    )
    assert controller._exit_errors == []


@pytest.mark.asyncio
async def test_finalize_artifacts_excludes_reaped_processors(controller) -> None:
    """Invariant I3: reaped services stay out of the fan-out target list."""
    controller.service_manager.service_id_map = {
        sid: ServiceRunInfo(
            service_id=sid,
            service_type=ServiceType.RECORD_PROCESSOR,
            registration_status=ServiceRegistrationStatus.REGISTERED,
            first_seen_ns=1,
            last_seen_ns=1,
            state=LifecycleState.RUNNING,
        )
        for sid in ("rp-1", "rp-2")
    }
    controller._reaped_service_ids = {"rp-2"}
    controller._exit_errors = []
    targeted: list[str] = []

    async def fake_request_to(identity, struct, timeout):
        targeted.append(identity)
        return CommandAck(cid=struct.cid, cmd=struct.cmd, sid=identity)

    controller.control_router.request_to = fake_request_to
    await controller._handle_finalize_artifacts_command(
        Command(cid="c-1", cmd=CommandType.FINALIZE_ARTIFACTS)
    )
    assert targeted == ["rp-1"]


def test_parse_responses_for_errors_ignores_preexisting_exit_errors(
    controller,
) -> None:
    """Invariant I4: only THIS batch's errors raise.

    ``_exit_errors`` may already hold an unrelated optional-producer failure.
    """
    controller._exit_errors = [
        ExitErrorInfo(
            error_details=ErrorDetails(message="unrelated earlier failure"),
            operation="telemetry",
            service_id="tm-1",
        )
    ]
    controller._parse_responses_for_errors(
        [CommandAck(cid="c-1", cmd=CommandType.PROFILE_START, sid="a")], "Start"
    )
    assert len(controller._exit_errors) == 1


def test_parse_responses_for_errors_raises_on_command_err(controller) -> None:
    controller._exit_errors = []
    with pytest.raises(LifecycleOperationError):
        controller._parse_responses_for_errors(
            [
                CommandErr(
                    cid="c-1", cmd=CommandType.PROFILE_START, sid="a", error="nope"
                )
            ],
            "Start",
        )
    assert controller._exit_errors[0].error_details.message == "nope"
    assert controller._exit_errors[0].service_id == "a"


@pytest.mark.asyncio
async def test_fail_fast_does_not_abort_on_an_unhandled_command(controller) -> None:
    """PROFILE_CONFIGURE fans out at every service; several have no handler.

    RecordsManager has no PROFILE_CONFIGURE hook, so aborting on CommandUnhandled
    abandoned the fan-out at the first such service and cancelled the rest --
    the TimingManager never configured and PROFILE_START then died with
    "No phase orchestrator configured".
    """
    reached: list[str] = []

    async def fake_request_to(identity, struct, timeout):
        reached.append(identity)
        if identity == "records-manager":
            return CommandUnhandled(cid=struct.cid, cmd=struct.cmd, sid=identity)
        return CommandAck(cid=struct.cid, cmd=struct.cmd, sid=identity)

    controller.control_router.request_to = fake_request_to
    responses = await controller._send_control_command_to_all_fail_fast(
        CommandType.PROFILE_CONFIGURE,
        ["records-manager", "timing-manager", "dataset-manager"],
        timeout=1.0,
    )
    assert sorted(reached) == ["dataset-manager", "records-manager", "timing-manager"]
    assert len(responses) == 3


@pytest.mark.asyncio
async def test_fail_fast_still_aborts_on_a_command_error(controller) -> None:
    async def fake_request_to(identity, struct, timeout):
        if identity == "a":
            return CommandErr(cid=struct.cid, cmd=struct.cmd, sid=identity, error="no")
        await asyncio.sleep(5)  # must be cancelled, never awaited to completion
        return CommandAck(cid=struct.cid, cmd=struct.cmd, sid=identity)

    controller.control_router.request_to = fake_request_to
    responses = await controller._send_control_command_to_all_fail_fast(
        CommandType.PROFILE_CONFIGURE, ["a", "b"], timeout=10.0
    )
    assert len(responses) == 1
    assert isinstance(responses[0], CommandErr)
