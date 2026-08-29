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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.common.control_structs import Command, CommandAck, CommandErr, CommandOk
from aiperf.common.enums import CommandType, CreditPhase
from aiperf.common.models import ErrorDetails, PhaseRecordsStats
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType
from aiperf.records.records_manager import RecordsManager
from aiperf.server_metrics.manager import ServerMetricsManager


def _controller(
    services: dict[str, ServiceType], reaped: set[str] | None = None
) -> MagicMock:
    """A SystemController stub carrying only what the two handlers read."""
    controller = MagicMock(spec=SystemController)
    controller.service_manager = SimpleNamespace(
        service_id_map={
            sid: SimpleNamespace(service_type=stype) for sid, stype in services.items()
        },
        run_service=AsyncMock(),
    )
    controller._reaped_service_ids = reaped or set()
    controller._PROFILE_COMPLETE_RELAY_TYPES = (
        SystemController._PROFILE_COMPLETE_RELAY_TYPES
    )

    # _send_control_command_to_all zips its result against the target list, so
    # the stub must answer once per target.
    async def _fan_out(cmd, service_ids, **_kwargs):
        return [CommandAck(cid="c-1", cmd=cmd, sid=sid) for sid in service_ids]

    controller._send_control_command_to_all = AsyncMock(side_effect=_fan_out)
    controller.debug = MagicMock()
    controller.warning = MagicMock()
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

    encoded = SystemController._encode_command_payload(result)
    assert orjson.loads(encoded) == {"pod_states": {}}
    assert isinstance(
        CommandOk(cid="c-1", cmd=CommandType.GET_POD_STATES, payload=encoded).payload,
        bytes,
    )
