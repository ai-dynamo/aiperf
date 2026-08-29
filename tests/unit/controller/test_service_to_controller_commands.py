# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service -> controller commands riding the DEALER/ROUTER control channel.

Three writers moved off the event bus: WorkerManager's SPAWN_WORKERS,
RecordsManager's FINALIZE_ARTIFACTS, and RecordsManager's PROFILE_COMPLETE.
The last one is not addressed to the controller at all -- the ROUTER is the
only path between two non-controller services, so the controller re-fans it.
Reader and writer of each command must agree on the payload shape or the run
fails at startup (SPAWN_WORKERS) or silently exports an empty metrics window
(PROFILE_COMPLETE).
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
import zmq
from pytest import param

from aiperf.common.control_structs import (
    Command,
    CommandAck,
    CommandErr,
    CommandOk,
    CommandUnhandled,
    encode_command_payload,
)
from aiperf.common.enums import CommandType, CreditPhase, LifecycleState
from aiperf.common.models import ErrorDetails, PhaseRecordsStats
from aiperf.common.service_registry import ServiceRegistry
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType
from aiperf.records.records_manager import RecordsManager
from aiperf.server_metrics.manager import ServerMetricsManager


def _controller(
    services: dict[str, ServiceType],
    reaped: set[str] | None = None,
    unregistered: set[str] | None = None,
    required: tuple[ServiceType, ...] | None = None,
) -> MagicMock:
    """A SystemController stub carrying only what the two handlers read.

    Services are put in *both* places the relays consult: the service manager's
    ``service_id_map`` and the process-wide ``ServiceRegistry``. Registering in
    only the former would misrepresent a live service, since the relays
    intersect the two. ``unregistered`` models the optional-service reaper,
    which calls ``ServiceRegistry.unregister`` but leaves ``service_id_map``
    untouched.
    """
    controller = MagicMock(spec=SystemController)
    controller.service_manager = SimpleNamespace(
        service_id_map={
            sid: SimpleNamespace(service_type=stype) for sid, stype in services.items()
        },
        required_services=dict.fromkeys(required or (), 1),
        run_service=AsyncMock(),
    )
    for sid, stype in services.items():
        ServiceRegistry.register(
            sid, stype, first_seen_ns=1, state=LifecycleState.RUNNING
        )
    for sid in unregistered or ():
        ServiceRegistry.unregister(sid)
    controller._reaped_service_ids = reaped or set()
    controller._PROFILE_COMPLETE_RELAY_TYPES = (
        SystemController._PROFILE_COMPLETE_RELAY_TYPES
    )
    controller._PROFILE_CANCEL_RELAY_TYPES = (
        SystemController._PROFILE_CANCEL_RELAY_TYPES
    )

    # _send_control_command_to_all zips its result against the target list, so
    # the stub must answer once per target.
    async def _fan_out(cmd, service_ids, **_kwargs):
        return [CommandAck(cid="c-1", cmd=cmd, sid=sid) for sid in service_ids]

    controller._send_control_command_to_all = AsyncMock(side_effect=_fan_out)
    controller.debug = MagicMock()
    controller.warning = MagicMock()
    # Bind the real severity helper rather than leaving it a mock, so relay
    # tests assert the log severity that actually reaches an operator instead
    # of merely that a decision function was called.
    controller._log_relay_transport_error = partial(
        SystemController._log_relay_transport_error, controller
    )
    return controller


class TestSpawnWorkersReader:
    """The controller decodes what WorkerManager now encodes."""

    @pytest.mark.asyncio
    async def test_spawn_workers_reads_num_workers_from_the_orjson_payload(
        self,
    ) -> None:
        controller = _controller({})
        controller.scale_record_processors_with_workers = False

        await SystemController._handle_spawn_workers_command(
            controller,
            Command(
                cid="c-1",
                cmd=CommandType.SPAWN_WORKERS,
                payload=orjson.dumps({"num_workers": 12}),
            ),
        )

        controller.service_manager.run_service.assert_awaited_once_with(
            ServiceType.WORKER, 12
        )

    @pytest.mark.asyncio
    async def test_spawn_workers_scales_record_processors_off_the_payload(self) -> None:
        controller = _controller({})
        controller.scale_record_processors_with_workers = True

        await SystemController._handle_spawn_workers_command(
            controller,
            Command(
                cid="c-1",
                cmd=CommandType.SPAWN_WORKERS,
                payload=orjson.dumps({"num_workers": 16}),
            ),
        )

        spawned = {
            call.args[0]: call.args[1]
            for call in controller.service_manager.run_service.await_args_list
        }
        assert spawned[ServiceType.WORKER] == 16
        assert spawned[ServiceType.RECORD_PROCESSOR] >= 1


class TestProfileCompleteRelay:
    """The controller is the only bridge between two non-controller services."""

    @pytest.mark.asyncio
    async def test_relay_targets_every_profile_complete_handler_type(self) -> None:
        controller = _controller(
            {
                "gpu-1": ServiceType.GPU_TELEMETRY_MANAGER,
                "netlat-1": ServiceType.NETWORK_LATENCY_MANAGER,
                "rp-1": ServiceType.RECORD_PROCESSOR,
                "sm-1": ServiceType.SERVER_METRICS_MANAGER,
                "records-1": ServiceType.RECORDS_MANAGER,
                "timing-1": ServiceType.TIMING_MANAGER,
            }
        )

        await SystemController._handle_profile_complete_relay(
            controller, Command(cid="c-1", cmd=CommandType.PROFILE_COMPLETE)
        )

        cmd, service_ids = controller._send_control_command_to_all.await_args.args
        assert cmd == CommandType.PROFILE_COMPLETE
        assert sorted(service_ids) == ["gpu-1", "netlat-1", "rp-1", "sm-1"]

    @pytest.mark.asyncio
    async def test_relay_excludes_reaped_services(self) -> None:
        controller = _controller(
            {
                "gpu-1": ServiceType.GPU_TELEMETRY_MANAGER,
                "sm-1": ServiceType.SERVER_METRICS_MANAGER,
            },
            reaped={"gpu-1"},
        )

        await SystemController._handle_profile_complete_relay(
            controller, Command(cid="c-1", cmd=CommandType.PROFILE_COMPLETE)
        )

        _, service_ids = controller._send_control_command_to_all.await_args.args
        assert service_ids == ["sm-1"]

    @pytest.mark.asyncio
    async def test_relay_excludes_a_service_the_reaper_unregistered(self) -> None:
        """A dropped optional service must not be addressed by the relay.

        The optional-service heartbeat reaper calls
        ``ServiceRegistry.unregister`` and ``record_reaped_service``, but a GPU
        telemetry manager is not a result producer, so
        ``SystemController._on_service_reaped`` returns before adding it to
        ``_reaped_service_ids``. It therefore stays in ``service_id_map`` with
        the right service_type and the ``reaped`` filter does not catch it.

        Relaying to it makes ROUTER_MANDATORY raise EHOSTUNREACH and logs a
        ``Host unreachable`` warning on every run on a box without a DCGM
        exporter -- user-facing noise pointing at a non-problem.
        """
        controller = _controller(
            {
                "gpu-1": ServiceType.GPU_TELEMETRY_MANAGER,
                "sm-1": ServiceType.SERVER_METRICS_MANAGER,
            },
            unregistered={"gpu-1"},
        )

        await SystemController._handle_profile_complete_relay(
            controller, Command(cid="c-1", cmd=CommandType.PROFILE_COMPLETE)
        )

        _, service_ids = controller._send_control_command_to_all.await_args.args
        assert service_ids == ["sm-1"]

    @pytest.mark.asyncio
    async def test_relay_skips_the_fan_out_when_every_target_is_unregistered(
        self,
    ) -> None:
        """No addressable peer means no fan-out at all, not an empty-list send."""
        controller = _controller(
            {"gpu-1": ServiceType.GPU_TELEMETRY_MANAGER},
            unregistered={"gpu-1"},
        )

        await SystemController._handle_profile_complete_relay(
            controller, Command(cid="c-1", cmd=CommandType.PROFILE_COMPLETE)
        )

        controller._send_control_command_to_all.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_relay_forwards_the_window_payload_verbatim(self) -> None:
        """Dropping the payload collapses every downstream export window."""
        controller = _controller({"sm-1": ServiceType.SERVER_METRICS_MANAGER})
        payload = orjson.dumps({"start_ns": 10, "end_ns": 20})

        await SystemController._handle_profile_complete_relay(
            controller,
            Command(cid="c-1", cmd=CommandType.PROFILE_COMPLETE, payload=payload),
        )

        assert (
            controller._send_control_command_to_all.await_args.kwargs["payload"]
            == payload
        )

    @pytest.mark.asyncio
    async def test_relay_is_a_noop_without_live_targets(self) -> None:
        controller = _controller({"records-1": ServiceType.RECORDS_MANAGER})

        await SystemController._handle_profile_complete_relay(
            controller, Command(cid="c-1", cmd=CommandType.PROFILE_COMPLETE)
        )

        controller._send_control_command_to_all.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_relay_warns_on_a_failed_peer_without_raising(self) -> None:
        """One failed peer must not fail the caller's completion path."""
        controller = _controller({"sm-1": ServiceType.SERVER_METRICS_MANAGER})
        controller._send_control_command_to_all = AsyncMock(
            return_value=[
                CommandErr(
                    cid="c-1",
                    cmd=CommandType.PROFILE_COMPLETE,
                    sid="sm-1",
                    error="scrape blew up",
                )
            ]
        )

        await SystemController._handle_profile_complete_relay(
            controller, Command(cid="c-1", cmd=CommandType.PROFILE_COMPLETE)
        )

        assert any(
            "scrape blew up" in str(call) for call in controller.warning.call_args_list
        )

    @pytest.mark.asyncio
    async def test_relay_warns_on_an_unhandled_peer(self) -> None:
        """A peer that lost its hook must not report as a successful flush.

        CommandUnhandled is ack-shaped, so swallowing it turns "this record
        processor never flushed its writers" into silence -- the exact failure
        the struct was added to make visible.
        """
        controller = _controller({"rp-1": ServiceType.RECORD_PROCESSOR})
        controller._send_control_command_to_all = AsyncMock(
            return_value=[
                CommandUnhandled(
                    cid="c-1", cmd=CommandType.PROFILE_COMPLETE, sid="rp-1"
                )
            ]
        )

        await SystemController._handle_profile_complete_relay(
            controller, Command(cid="c-1", cmd=CommandType.PROFILE_COMPLETE)
        )

        warnings = [str(call) for call in controller.warning.call_args_list]
        assert any("rp-1" in w and "unhandled" in w for w in warnings), warnings

    @pytest.mark.asyncio
    async def test_relay_warns_on_an_unexpected_response_shape(self) -> None:
        """An unrecognised shape must not pass as success.

        The loop enumerates the shapes it knows; CommandAck is the only one that
        means "flushed". Letting anything else fall through would report a peer
        as successful without ever inspecting it -- the same silent-success
        failure mode as an unhandled command, which is how a missing relay
        target went unnoticed once already.
        """
        controller = _controller({"rp-1": ServiceType.RECORD_PROCESSOR})
        controller._send_control_command_to_all = AsyncMock(
            return_value=[
                CommandOk(cid="c-1", cmd=CommandType.PROFILE_COMPLETE, sid="rp-1")
            ]
        )

        await SystemController._handle_profile_complete_relay(
            controller, Command(cid="c-1", cmd=CommandType.PROFILE_COMPLETE)
        )

        warnings = [str(call) for call in controller.warning.call_args_list]
        assert any(
            "rp-1" in w and "unexpected response shape" in w for w in warnings
        ), warnings

    @pytest.mark.asyncio
    async def test_relay_stays_quiet_on_a_plain_ack(self) -> None:
        """The success path must not warn, or the unhandled warning means nothing."""
        controller = _controller({"rp-1": ServiceType.RECORD_PROCESSOR})

        await SystemController._handle_profile_complete_relay(
            controller, Command(cid="c-1", cmd=CommandType.PROFILE_COMPLETE)
        )

        controller.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_relay_warns_on_a_timed_out_peer(self) -> None:
        controller = _controller({"sm-1": ServiceType.SERVER_METRICS_MANAGER})
        controller._send_control_command_to_all = AsyncMock(
            return_value=[ErrorDetails(type="TimeoutError", message="too slow")]
        )

        await SystemController._handle_profile_complete_relay(
            controller, Command(cid="c-1", cmd=CommandType.PROFILE_COMPLETE)
        )

        assert any(
            "too slow" in str(call) for call in controller.warning.call_args_list
        )


def _records_manager() -> MagicMock:
    manager = MagicMock(spec=RecordsManager)
    manager.send_command_to_controller = AsyncMock()
    manager.debug = MagicMock()
    manager.warning = MagicMock()
    manager.info = MagicMock()
    return manager


class TestRecordsManagerFinalizeArtifacts:
    """The record-processor durability barrier now rides the DEALER."""

    @pytest.mark.asyncio
    async def test_finalize_sends_finalize_artifacts_to_the_controller(self) -> None:
        manager = _records_manager()
        manager.send_command_to_controller.return_value = CommandAck(
            cid="c-1", cmd=CommandType.FINALIZE_ARTIFACTS, sid="ctl"
        )

        await RecordsManager._finalize_record_processor_artifacts(manager)

        cmd = manager.send_command_to_controller.await_args.args[0]
        assert cmd == CommandType.FINALIZE_ARTIFACTS
        payload = orjson.loads(
            manager.send_command_to_controller.await_args.kwargs["payload"]
        )
        assert payload["request_ns"] > 0

    @pytest.mark.asyncio
    async def test_finalize_raises_on_a_mismatched_cmd(self) -> None:
        """An ack for a different command is not this barrier's acknowledgement."""
        manager = _records_manager()
        manager.send_command_to_controller.return_value = CommandAck(
            cid="c-1", cmd=CommandType.PROFILE_COMPLETE, sid="ctl"
        )

        with pytest.raises(RuntimeError, match="Unexpected"):
            await RecordsManager._finalize_record_processor_artifacts(manager)

    @pytest.mark.asyncio
    async def test_finalize_raises_on_command_err(self) -> None:
        manager = _records_manager()
        manager.send_command_to_controller.return_value = CommandErr(
            cid="c-1", cmd=CommandType.FINALIZE_ARTIFACTS, sid="ctl", error="disk full"
        )

        with pytest.raises(RuntimeError, match="disk full"):
            await RecordsManager._finalize_record_processor_artifacts(manager)

    @pytest.mark.asyncio
    async def test_finalize_raises_on_timeout(self) -> None:
        manager = _records_manager()
        manager.send_command_to_controller.side_effect = TimeoutError("no answer")

        with pytest.raises(RuntimeError, match="timed out"):
            await RecordsManager._finalize_record_processor_artifacts(manager)


class TestRecordsManagerProfileComplete:
    """The window payload must survive the trip to the relay."""

    @staticmethod
    def _manager_with_window() -> MagicMock:
        manager = _records_manager()
        manager.service_id = "records-manager-1"
        manager._records_tracker = MagicMock()
        manager._records_tracker.create_aggregate_stats_for_phase.side_effect = [
            SimpleNamespace(start_ns=1, requests_end_ns=2),  # profiling
            SimpleNamespace(start_ns=3, requests_end_ns=4),  # warmup
        ]
        manager._records_tracker.create_stats_for_phase.return_value = (
            PhaseRecordsStats(phase=CreditPhase.WARMUP)
        )
        manager.publish = AsyncMock()
        manager._process_results = AsyncMock()
        return manager

    @pytest.mark.asyncio
    async def test_profile_complete_payload_carries_all_four_window_fields(
        self,
    ) -> None:
        manager = self._manager_with_window()
        manager.send_command_to_controller.return_value = CommandAck(
            cid="c-1", cmd=CommandType.PROFILE_COMPLETE, sid="ctl"
        )

        await RecordsManager._finalize_and_process_results_impl(
            manager, phase=CreditPhase.WARMUP, cancelled=False
        )

        assert (
            manager.send_command_to_controller.await_args.args[0]
            == CommandType.PROFILE_COMPLETE
        )
        payload = orjson.loads(
            manager.send_command_to_controller.await_args.kwargs["payload"]
        )
        assert payload == {
            "start_ns": 1,
            "end_ns": 2,
            "warmup_start_ns": 3,
            "warmup_end_ns": 4,
        }
        manager._process_results.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_profile_complete_timeout_still_processes_results(self) -> None:
        """Skipping _process_results would strand the controller's export."""
        manager = self._manager_with_window()
        manager.send_command_to_controller.side_effect = TimeoutError("relay stalled")

        await RecordsManager._finalize_and_process_results_impl(
            manager, phase=CreditPhase.WARMUP, cancelled=False
        )

        manager.send_command_to_controller.assert_awaited_once()
        manager._process_results.assert_awaited_once()


class TestServerMetricsProfileCompleteReader:
    """The server-metrics reader decodes what RecordsManager now encodes."""

    @staticmethod
    def _manager() -> MagicMock:
        import asyncio

        manager = MagicMock(spec=ServerMetricsManager)
        manager._profile_complete_lock = asyncio.Lock()
        manager._result_published = False
        manager._capture_profile_complete_scrape = AsyncMock()
        manager._stop_all_collectors = AsyncMock()
        manager._publish_server_metrics_result = AsyncMock()
        manager.debug = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_reader_decodes_the_window_payload(self) -> None:
        manager = self._manager()

        await ServerMetricsManager._handle_profile_complete_command(
            manager,
            Command(
                cid="c-1",
                cmd=CommandType.PROFILE_COMPLETE,
                payload=orjson.dumps(
                    {
                        "start_ns": 1,
                        "end_ns": 2,
                        "warmup_start_ns": 3,
                        "warmup_end_ns": 4,
                    }
                ),
            ),
        )

        manager._capture_profile_complete_scrape.assert_awaited_once_with(2)
        assert manager._publish_server_metrics_result.await_args.kwargs == {
            "start_ns": 1,
            "end_ns": 2,
            "warmup_start_ns": 3,
            "warmup_end_ns": 4,
        }

    @pytest.mark.asyncio
    async def test_reader_defaults_every_missing_window_field_to_none(self) -> None:
        """The Pydantic predecessor's fields were all ``int | None``."""
        manager = self._manager()

        await ServerMetricsManager._handle_profile_complete_command(
            manager, Command(cid="c-1", cmd=CommandType.PROFILE_COMPLETE)
        )

        manager._capture_profile_complete_scrape.assert_awaited_once_with(None)
        assert manager._publish_server_metrics_result.await_args.kwargs == {
            "start_ns": None,
            "end_ns": None,
            "warmup_start_ns": None,
            "warmup_end_ns": None,
        }


class TestProfileCancelRelay:
    """Service-originated aborts reach their peers only through the controller.

    RecordsManager's --failed-request-threshold abort and TimingManager's
    warmup / worker-loss aborts used to broadcast a profile-cancel command on the
    pub bus. Nothing subscribes to command messages any more, so without this
    relay both aborts reach nobody and the run hangs instead of terminating.
    """

    @pytest.mark.asyncio
    async def test_relay_targets_every_profile_cancel_handler_type(self) -> None:
        controller = _controller(
            {
                "gpu-1": ServiceType.GPU_TELEMETRY_MANAGER,
                "netlat-1": ServiceType.NETWORK_LATENCY_MANAGER,
                "rp-1": ServiceType.RECORD_PROCESSOR,
                "sm-1": ServiceType.SERVER_METRICS_MANAGER,
                "records-1": ServiceType.RECORDS_MANAGER,
                "timing-1": ServiceType.TIMING_MANAGER,
            }
        )

        await SystemController._relay_profile_cancel(controller, "", b"")

        cmd, service_ids = controller._send_control_command_to_all.await_args.args
        assert cmd == CommandType.PROFILE_CANCEL
        # RECORD_PROCESSOR has no @on_command(PROFILE_CANCEL) hook.
        assert sorted(service_ids) == [
            "gpu-1",
            "netlat-1",
            "records-1",
            "sm-1",
            "timing-1",
        ]

    @pytest.mark.asyncio
    async def test_relay_excludes_a_service_the_reaper_unregistered(self) -> None:
        """Kept symmetric with the PROFILE_COMPLETE relay.

        Both relays address peers by type out of ``service_id_map``, so both
        need the same liveness intersection. Fixing only one would leave the
        abort path emitting the EHOSTUNREACH warning the completion path no
        longer emits.
        """
        controller = _controller(
            {
                "gpu-1": ServiceType.GPU_TELEMETRY_MANAGER,
                "timing-1": ServiceType.TIMING_MANAGER,
            },
            unregistered={"gpu-1"},
        )

        await SystemController._relay_profile_cancel(controller, "", b"")

        _, service_ids = controller._send_control_command_to_all.await_args.args
        assert service_ids == ["timing-1"]

    @pytest.mark.asyncio
    async def test_relay_excludes_the_originator(self) -> None:
        """Pub/sub never delivered a broadcast back to its sender.

        Both callers run their own cancel handler locally; re-entering it over
        the wire would double-finalize the records manager.
        """
        controller = _controller(
            {
                "records-1": ServiceType.RECORDS_MANAGER,
                "timing-1": ServiceType.TIMING_MANAGER,
            }
        )

        await SystemController._relay_profile_cancel(controller, "records-1", b"")

        _, service_ids = controller._send_control_command_to_all.await_args.args
        assert service_ids == ["timing-1"]

    @pytest.mark.asyncio
    async def test_relay_excludes_reaped_services(self) -> None:
        controller = _controller(
            {
                "gpu-1": ServiceType.GPU_TELEMETRY_MANAGER,
                "sm-1": ServiceType.SERVER_METRICS_MANAGER,
            },
            reaped={"gpu-1"},
        )

        await SystemController._relay_profile_cancel(controller, "", b"")

        _, service_ids = controller._send_control_command_to_all.await_args.args
        assert service_ids == ["sm-1"]

    @pytest.mark.asyncio
    async def test_relay_is_a_noop_without_live_targets(self) -> None:
        controller = _controller({"rp-1": ServiceType.RECORD_PROCESSOR})

        await SystemController._relay_profile_cancel(controller, "", b"")

        controller._send_control_command_to_all.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_relay_warns_on_a_failed_peer_without_raising(self) -> None:
        controller = _controller({"sm-1": ServiceType.SERVER_METRICS_MANAGER})
        controller._send_control_command_to_all = AsyncMock(
            return_value=[
                CommandErr(
                    cid="c-1",
                    cmd=CommandType.PROFILE_CANCEL,
                    sid="sm-1",
                    error="cancel blew up",
                )
            ]
        )

        await SystemController._relay_profile_cancel(controller, "", b"")

        assert any(
            "cancel blew up" in str(call) for call in controller.warning.call_args_list
        )

    @pytest.mark.asyncio
    async def test_handler_detaches_the_fan_out_and_reads_the_origin(self) -> None:
        """The handler must answer at pub/sub latency.

        Awaiting the fan-out inline would make TimingManager's abort wait on
        RecordsManager's full result processing before cancelling its own
        orchestrator -- the publish it replaces returned immediately.
        """
        controller = _controller({"sm-1": ServiceType.SERVER_METRICS_MANAGER})
        payload = orjson.dumps({"origin_service_id": "records-1"})
        controller.execute_async = MagicMock()

        await SystemController._handle_profile_cancel_relay(
            controller,
            Command(cid="c-1", cmd=CommandType.PROFILE_CANCEL, payload=payload),
        )

        controller._send_control_command_to_all.assert_not_awaited()
        controller.execute_async.assert_called_once()
        controller._relay_profile_cancel.assert_called_once_with("records-1", payload)


@pytest.mark.asyncio
async def test_get_pod_states_handler_result_encodes_as_orjson() -> None:
    """The API decodes ``CommandOk.payload`` with orjson.loads.

    A preservation test: the handler already returned ``model_dump(mode="json")``
    and still does, so this passes both before and after. It pins the encoding
    contract the rewritten ``pod_state_rpc`` remote branch now depends on.
    """
    controller = MagicMock(spec=SystemController)
    controller.get_pod_state_snapshot = MagicMock(
        return_value=MagicMock(model_dump=MagicMock(return_value={"pod_states": {}}))
    )

    result = await SystemController._handle_get_pod_states_command(
        controller, Command(cid="c-1", cmd=CommandType.GET_POD_STATES)
    )

    encoded = encode_command_payload(result)
    assert orjson.loads(encoded) == {"pod_states": {}}
    assert isinstance(
        CommandOk(cid="c-1", cmd=CommandType.GET_POD_STATES, payload=encoded).payload,
        bytes,
    )


class TestRelayTransportErrorSeverity:
    """A departed optional peer is expected; everything else stays a warning.

    The relay builds its target list before the heartbeat watchdog can confirm a
    service stale -- measured at ~0.9s ahead in a live run -- so it legitimately
    addresses a peer whose ZMQ socket is already gone. That is unavoidable (peer
    death is only observable at send time) and must not be logged as a problem.
    But the discrimination has to stay narrow, which is what these cases pin.
    """

    @staticmethod
    def _error(errno: int | None) -> ErrorDetails:
        """A transport failure as `command_error_details` would produce it."""
        return ErrorDetails(
            code=errno, type="ZMQError", message=f"ZMQError(errno={errno})"
        )

    @pytest.mark.parametrize(
        "errno",
        [param(zmq.EHOSTUNREACH, id="ehostunreach"), param(zmq.ENOTCONN, id="enotconn")],
    )  # fmt: skip
    def test_peer_gone_from_an_optional_service_is_debug(self, errno: int) -> None:
        controller = _controller(
            {"gpu-1": ServiceType.GPU_TELEMETRY_MANAGER},
            required=(ServiceType.RECORDS_MANAGER,),
        )

        SystemController._log_relay_transport_error(
            controller, CommandType.PROFILE_COMPLETE, "gpu-1", self._error(errno)
        )

        controller.warning.assert_not_called()
        controller.debug.assert_called_once()

    def test_peer_gone_from_a_required_service_is_still_a_warning(self) -> None:
        """The half that catches an over-broad "just silence it" refactor.

        A required service that became unreachable mid-relay is exactly what
        someone needs to see in a log; muting it to kill the optional-service
        noise would discard real signal.
        """
        controller = _controller(
            {"records-1": ServiceType.RECORDS_MANAGER},
            required=(ServiceType.RECORDS_MANAGER,),
        )

        SystemController._log_relay_transport_error(
            controller,
            CommandType.PROFILE_COMPLETE,
            "records-1",
            self._error(zmq.EHOSTUNREACH),
        )

        controller.warning.assert_called_once()

    def test_non_peer_gone_failure_from_an_optional_service_is_a_warning(self) -> None:
        """A live optional service that faults or times out is a real fault."""
        controller = _controller(
            {"gpu-1": ServiceType.GPU_TELEMETRY_MANAGER},
            required=(ServiceType.RECORDS_MANAGER,),
        )

        SystemController._log_relay_transport_error(
            controller,
            CommandType.PROFILE_COMPLETE,
            "gpu-1",
            ErrorDetails(type="TimeoutError", message="Command timed out"),
        )

        controller.warning.assert_called_once()

    def test_unknown_service_id_is_a_warning(self) -> None:
        """Not in service_id_map means we cannot prove it was optional."""
        controller = _controller({}, required=(ServiceType.RECORDS_MANAGER,))

        SystemController._log_relay_transport_error(
            controller,
            CommandType.PROFILE_COMPLETE,
            "ghost-1",
            self._error(zmq.EHOSTUNREACH),
        )

        controller.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_the_relay_end_to_end_mutes_a_departed_optional_peer(self) -> None:
        """End to end: the exact log line the smoke run was emitting.

        Asserted through the relay rather than on the helper alone, so a future
        refactor that stops routing ErrorDetails through the severity decision
        brings the warning back and fails here.
        """
        controller = _controller(
            {"gpu-1": ServiceType.GPU_TELEMETRY_MANAGER},
            required=(ServiceType.RECORDS_MANAGER,),
        )
        error = self._error(zmq.EHOSTUNREACH)

        async def _fan_out(cmd, service_ids, **_kwargs):
            return [error for _ in service_ids]

        controller._send_control_command_to_all = AsyncMock(side_effect=_fan_out)

        await SystemController._handle_profile_complete_relay(
            controller, Command(cid="c-1", cmd=CommandType.PROFILE_COMPLETE)
        )

        controller.warning.assert_not_called()
