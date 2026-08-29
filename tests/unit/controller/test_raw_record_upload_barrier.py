# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes RAW export uses service acknowledgements, not filename counts."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.control_structs import Command, CommandAck, CommandErr
from aiperf.common.enums import CommandType, ExportLevel
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType


@pytest.fixture
def controller():
    ctrl = SystemController.__new__(SystemController)
    ctrl.run = MagicMock()
    ctrl.run.cfg.artifacts.export_level = ExportLevel.RAW
    ctrl._is_kubernetes = MagicMock(return_value=True)
    ctrl._k8s_topology = MagicMock(num_worker_pods=2)
    ctrl.service_id = "system_controller"
    ctrl._exit_errors = []
    ctrl._export_failed = False
    ctrl._raw_artifacts_finalized = False
    ctrl._raw_artifacts_finalize_succeeded = False
    ctrl._reaped_service_ids = set()
    ctrl.info = MagicMock()
    ctrl.warning = MagicMock()
    ctrl.error = MagicMock()
    return ctrl


def _ack(cmd: str, service_id: str) -> CommandAck:
    return CommandAck(cid="c-1", cmd=cmd, sid=service_id)


class TestRawRecordBarrier:
    @pytest.mark.asyncio
    async def test_waits_for_exact_registered_worker_groups(self, controller):
        registered = [
            MagicMock(service_id="worker_group_manager_1"),
            MagicMock(service_id="worker_group_manager_0"),
        ]

        async def respond(cmd, service_ids, payload=b"", timeout=0.0):
            assert cmd == CommandType.FINALIZE_ARTIFACTS
            assert service_ids == ["worker_group_manager_0", "worker_group_manager_1"]
            assert timeout > 0
            return [_ack(cmd, service_id) for service_id in service_ids]

        controller._send_control_command_to_all = AsyncMock(side_effect=respond)
        with patch(
            "aiperf.controller.system_controller.ServiceRegistry.get_services",
            return_value=registered,
        ):
            await controller._finalize_kubernetes_raw_artifacts()

        assert controller._exit_errors == []
        assert controller._export_failed is False

    @pytest.mark.asyncio
    async def test_missing_worker_group_fails_without_broadcast(self, controller):
        controller._send_control_command_to_all = AsyncMock()
        with patch(
            "aiperf.controller.system_controller.ServiceRegistry.get_services",
            return_value=[MagicMock(service_id="worker_group_manager_0")],
        ):
            await controller._finalize_kubernetes_raw_artifacts()

        controller._send_control_command_to_all.assert_not_awaited()
        assert controller._export_failed is True
        assert controller._exit_errors[0].operation == "finalize_raw_artifacts"

    @pytest.mark.asyncio
    async def test_worker_group_failure_blocks_export(self, controller):
        registered = [
            MagicMock(service_id="worker_group_manager_0"),
            MagicMock(service_id="worker_group_manager_1"),
        ]

        async def respond(cmd, service_ids, payload=b"", timeout=0.0):
            del timeout
            return [
                _ack(cmd, service_ids[0]),
                CommandErr(
                    cid="c-1", cmd=cmd, sid=service_ids[1], error="upload failed"
                ),
            ]

        controller._send_control_command_to_all = AsyncMock(side_effect=respond)
        with patch(
            "aiperf.controller.system_controller.ServiceRegistry.get_services",
            return_value=registered,
        ):
            await controller._finalize_kubernetes_raw_artifacts()

        assert controller._export_failed is True
        assert controller._exit_errors[0].service_id == "worker_group_manager_1"
        assert controller._exit_errors[0].error_details.message == "upload failed"

    @pytest.mark.asyncio
    async def test_non_kubernetes_run_is_unchanged(self, controller):
        controller._is_kubernetes.return_value = False
        controller._send_control_command_to_all = AsyncMock()

        await controller._finalize_kubernetes_raw_artifacts()

        controller._send_control_command_to_all.assert_not_awaited()
        assert controller._exit_errors == []
        assert CommandType.FINALIZE_ARTIFACTS == "finalize_artifacts"


class TestRecordProcessorArtifactBarrier:
    @pytest.fixture
    def local_controller(self):
        ctrl = SystemController.__new__(SystemController)
        ctrl.run = MagicMock()
        ctrl._is_kubernetes = MagicMock(return_value=False)
        ctrl.service_id = "system_controller"
        ctrl.service_manager = MagicMock()
        ctrl.service_manager.service_id_map = {
            "record_processor_1": MagicMock(service_type=ServiceType.RECORD_PROCESSOR),
            "records_manager": MagicMock(service_type=ServiceType.RECORDS_MANAGER),
            "record_processor_0": MagicMock(service_type=ServiceType.RECORD_PROCESSOR),
        }
        ctrl._reaped_service_ids = set()
        ctrl._exit_errors = []
        ctrl._raw_artifacts_finalized = False
        ctrl._raw_artifacts_finalize_succeeded = False
        ctrl.error = MagicMock()
        return ctrl

    @pytest.mark.asyncio
    async def test_waits_for_exact_registered_record_processors(self, local_controller):
        async def respond(cmd, service_ids, payload=b"", timeout=0.0):
            assert cmd == CommandType.FINALIZE_ARTIFACTS
            assert service_ids == ["record_processor_0", "record_processor_1"]
            assert timeout > 0
            return [_ack(cmd, service_id) for service_id in service_ids]

        local_controller._send_control_command_to_all = AsyncMock(side_effect=respond)

        await local_controller._handle_finalize_artifacts_command(
            Command(cid="c-1", cmd=CommandType.FINALIZE_ARTIFACTS)
        )

        local_controller._send_control_command_to_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_record_processor_failure_degrades_without_raising(
        self, local_controller
    ):
        """A missing ack must not discard an otherwise complete result set.

        PROFILE_COMPLETE already drove the same _finalize_local_artifacts
        on every processor moments earlier, so locally this barrier contributes
        the acknowledgement rather than the flush. The failure still lands in
        _exit_errors and still forces a non-zero exit; it just no longer aborts
        the export. Kubernetes keeps failing closed -- see TestRawRecordBarrier,
        where raw records genuinely have not been uploaded yet.
        """

        async def respond(cmd, service_ids, payload=b"", timeout=0.0):
            del timeout
            return [
                _ack(cmd, service_ids[0]),
                CommandErr(cid="c-1", cmd=cmd, sid=service_ids[1], error="disk full"),
            ]

        local_controller._send_control_command_to_all = AsyncMock(side_effect=respond)
        local_controller.warning = MagicMock()

        await local_controller._handle_finalize_artifacts_command(
            Command(cid="c-1", cmd=CommandType.FINALIZE_ARTIFACTS)
        )

        assert [e.operation for e in local_controller._exit_errors] == [
            "finalize_artifacts"
        ]
        assert local_controller._exit_errors[0].error_details.message == "disk full"

    @pytest.mark.asyncio
    async def test_no_live_record_processors_degrades_without_raising(
        self, local_controller
    ):
        """Local processors auto-scale to one below eight workers.

        A single reap therefore empties the target list, and raising here threw
        away a complete, exportable run over a peer that was already gone.
        """
        local_controller.service_manager.service_id_map = {}
        local_controller._send_control_command_to_all = AsyncMock()

        await local_controller._handle_finalize_artifacts_command(
            Command(cid="c-1", cmd=CommandType.FINALIZE_ARTIFACTS)
        )

        local_controller._send_control_command_to_all.assert_not_awaited()
        assert [e.operation for e in local_controller._exit_errors] == [
            "finalize_artifacts"
        ]
