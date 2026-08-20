# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes RAW export uses service acknowledgements, not filename counts."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import CommandType, ExportLevel
from aiperf.common.messages import (
    CommandAcknowledgedResponse,
    CommandErrorResponse,
    FinalizeArtifactsCommand,
)
from aiperf.common.models import ErrorDetails
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
    ctrl._reaped_service_ids = set()
    ctrl.info = MagicMock()
    ctrl.warning = MagicMock()
    ctrl.error = MagicMock()
    return ctrl


def _ack(command: FinalizeArtifactsCommand, service_id: str):
    return CommandAcknowledgedResponse.from_command_message(command, service_id)


class TestRawRecordBarrier:
    @pytest.mark.asyncio
    async def test_waits_for_exact_registered_worker_groups(self, controller):
        registered = [
            MagicMock(service_id="worker_group_manager_1"),
            MagicMock(service_id="worker_group_manager_0"),
        ]

        async def respond(command, service_ids, timeout):
            assert isinstance(command, FinalizeArtifactsCommand)
            assert command.target_service_type == ServiceType.WORKER_GROUP_MANAGER
            assert service_ids == ["worker_group_manager_0", "worker_group_manager_1"]
            assert timeout > 0
            return [_ack(command, service_id) for service_id in service_ids]

        controller.send_command_and_wait_for_all_responses = AsyncMock(
            side_effect=respond
        )
        with patch(
            "aiperf.controller.system_controller.ServiceRegistry.get_services",
            return_value=registered,
        ):
            await controller._finalize_kubernetes_raw_artifacts()

        assert controller._exit_errors == []
        assert controller._export_failed is False

    @pytest.mark.asyncio
    async def test_missing_worker_group_fails_without_broadcast(self, controller):
        controller.send_command_and_wait_for_all_responses = AsyncMock()
        with patch(
            "aiperf.controller.system_controller.ServiceRegistry.get_services",
            return_value=[MagicMock(service_id="worker_group_manager_0")],
        ):
            await controller._finalize_kubernetes_raw_artifacts()

        controller.send_command_and_wait_for_all_responses.assert_not_awaited()
        assert controller._export_failed is True
        assert controller._exit_errors[0].operation == "finalize_raw_artifacts"

    @pytest.mark.asyncio
    async def test_worker_group_failure_blocks_export(self, controller):
        registered = [
            MagicMock(service_id="worker_group_manager_0"),
            MagicMock(service_id="worker_group_manager_1"),
        ]

        async def respond(command, service_ids, timeout):
            del timeout
            return [
                _ack(command, service_ids[0]),
                CommandErrorResponse.from_command_message(
                    command,
                    service_ids[1],
                    ErrorDetails(message="upload failed"),
                ),
            ]

        controller.send_command_and_wait_for_all_responses = AsyncMock(
            side_effect=respond
        )
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
        controller.send_command_and_wait_for_all_responses = AsyncMock()

        await controller._finalize_kubernetes_raw_artifacts()

        controller.send_command_and_wait_for_all_responses.assert_not_awaited()
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
        ctrl.error = MagicMock()
        return ctrl

    @pytest.mark.asyncio
    async def test_waits_for_exact_registered_record_processors(self, local_controller):
        async def respond(command, service_ids, timeout):
            assert isinstance(command, FinalizeArtifactsCommand)
            assert command.target_service_type == ServiceType.RECORD_PROCESSOR
            assert service_ids == ["record_processor_0", "record_processor_1"]
            assert timeout > 0
            return [_ack(command, service_id) for service_id in service_ids]

        local_controller.send_command_and_wait_for_all_responses = AsyncMock(
            side_effect=respond
        )

        await local_controller._handle_finalize_artifacts_command(
            FinalizeArtifactsCommand(
                service_id="records_manager",
                target_service_type=ServiceType.SYSTEM_CONTROLLER,
            )
        )

        local_controller.send_command_and_wait_for_all_responses.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_record_processor_failure_fails_barrier(self, local_controller):
        async def respond(command, service_ids, timeout):
            del timeout
            return [
                _ack(command, service_ids[0]),
                CommandErrorResponse.from_command_message(
                    command,
                    service_ids[1],
                    ErrorDetails(message="disk full"),
                ),
            ]

        local_controller.send_command_and_wait_for_all_responses = AsyncMock(
            side_effect=respond
        )

        with pytest.raises(ExceptionGroup, match="Failed to finalize 1"):
            await local_controller._handle_finalize_artifacts_command(
                FinalizeArtifactsCommand(service_id="records_manager")
            )
