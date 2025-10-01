# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import ServiceRegistrationStatus, ServiceType
from aiperf.common.exceptions import AIPerfError
from aiperf.common.models.service_models import ServiceRunInfo
from aiperf.controller.multiprocess_service_manager import (
    MultiProcessServiceManager,
)


class TestAsyncSubprocessRunInfo:
    """Test AsyncSubprocessRunInfo Pydantic model."""

    def test_async_subprocess_run_info_creation(self, create_subprocess_info):
        """Test creating AsyncSubprocessRunInfo with required fields."""
        info = create_subprocess_info(
            service_type=ServiceType.DATASET_MANAGER,
            service_id="test_service_123",
        )

        assert info.service_type == ServiceType.DATASET_MANAGER
        assert info.service_id == "test_service_123"
        assert info.process is None
        assert info.user_config_file is None
        assert info.service_config_file is None

    def test_async_subprocess_run_info_with_all_fields(
        self, create_subprocess_info, mock_subprocess_process
    ):
        """Test creating AsyncSubprocessRunInfo with all fields."""
        user_config_file = Path("/tmp/user_config.json")
        service_config_file = Path("/tmp/service_config.json")

        info = create_subprocess_info(
            process=mock_subprocess_process,
            service_type=ServiceType.TIMING_MANAGER,
            service_id="test_service_456",
            user_config_file=user_config_file,
            service_config_file=service_config_file,
        )

        assert info.process == mock_subprocess_process
        assert info.service_type == ServiceType.TIMING_MANAGER
        assert info.service_id == "test_service_456"
        assert info.user_config_file == user_config_file
        assert info.service_config_file == service_config_file


class TestMultiProcessServiceManager:
    """Test MultiProcessServiceManager async subprocess management."""

    @pytest.fixture
    def service_manager(
        self, service_config, user_config
    ) -> MultiProcessServiceManager:
        """Create a MultiProcessServiceManager instance for testing."""
        return MultiProcessServiceManager(
            required_services={
                ServiceType.DATASET_MANAGER: 1,
                ServiceType.TIMING_MANAGER: 1,
            },
            service_config=service_config,
            user_config=user_config,
        )

    def test_service_manager_initialization(self, service_manager):
        """Test proper initialization of MultiProcessServiceManager."""
        assert isinstance(service_manager.subprocess_info_map, dict)
        assert len(service_manager.subprocess_info_map) == 0
        assert service_manager.subprocess_map_lock is not None

    @pytest.mark.asyncio
    async def test_remove_subprocess_info(
        self, service_manager, create_subprocess_info
    ):
        """Test _remove_subprocess_info method."""
        service_id = "test_service_123"
        info = create_subprocess_info(service_id=service_id)

        service_manager.subprocess_info_map[service_id] = info
        assert service_id in service_manager.subprocess_info_map

        await service_manager._remove_subprocess_info(info)
        assert service_id not in service_manager.subprocess_info_map

    @pytest.mark.asyncio
    async def test_wait_for_subprocess_already_terminated(
        self, service_manager, create_subprocess_info
    ):
        """Test _wait_for_subprocess when process is already terminated."""
        mock_process = MagicMock()
        mock_process.returncode = 0

        info = create_subprocess_info(process=mock_process)

        await service_manager._wait_for_subprocess(info)
        mock_process.terminate.assert_not_called()

    @pytest.mark.asyncio
    async def test_wait_for_subprocess_graceful_termination(
        self, service_manager, create_subprocess_info, mock_subprocess_process
    ):
        """Test _wait_for_subprocess with graceful termination."""
        info = create_subprocess_info(process=mock_subprocess_process)

        await service_manager._wait_for_subprocess(info)

        mock_subprocess_process.terminate.assert_called_once()
        mock_subprocess_process.wait.assert_called()

    @pytest.mark.asyncio
    async def test_wait_for_subprocess_with_config_file_cleanup(
        self, service_manager, create_subprocess_info, mock_subprocess_process, tmp_path
    ):
        """Test _wait_for_subprocess cleans up config files."""
        user_config_path = tmp_path / "user_config.json"
        service_config_path = tmp_path / "service_config.json"
        user_config_path.touch()
        service_config_path.touch()

        info = create_subprocess_info(
            process=mock_subprocess_process,
            user_config_file=user_config_path,
            service_config_file=service_config_path,
        )

        assert user_config_path.exists()
        assert service_config_path.exists()

        await service_manager._wait_for_subprocess(info)

        assert not user_config_path.exists()
        assert not service_config_path.exists()

    @pytest.mark.asyncio
    async def test_shutdown_all_services_empty_map(self, service_manager):
        """Test shutdown_all_services with empty subprocess map."""
        results = await service_manager.shutdown_all_services()
        assert results == []

    @pytest.mark.asyncio
    async def test_kill_all_services(
        self, service_manager, create_subprocess_info, mock_subprocess_process
    ):
        """Test kill_all_services method."""
        service_id = "test_service"
        info = create_subprocess_info(
            process=mock_subprocess_process,
            service_id=service_id,
        )
        service_manager.subprocess_info_map[service_id] = info

        await service_manager.kill_all_services()

        mock_subprocess_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_service_by_type(
        self, service_manager, create_subprocess_info, mock_subprocess_process
    ):
        """Test stop_service method filtering by service type."""
        dataset_info = create_subprocess_info(
            process=mock_subprocess_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="dataset_service",
        )
        timing_info = create_subprocess_info(
            process=MagicMock(),
            service_type=ServiceType.TIMING_MANAGER,
            service_id="timing_service",
        )

        service_manager.subprocess_info_map["dataset_service"] = dataset_info
        service_manager.subprocess_info_map["timing_service"] = timing_info

        results = await service_manager.stop_service(ServiceType.DATASET_MANAGER)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_wait_for_all_services_registration_timeout(
        self, service_manager, create_subprocess_info
    ):
        """Test wait_for_all_services_registration timeout behavior."""
        service_info = ServiceRunInfo(
            service_type=ServiceType.DATASET_MANAGER,
            service_id="test_service",
            registration_status=ServiceRegistrationStatus.UNREGISTERED,
            required=True,
        )
        service_manager.service_id_map["test_service"] = service_info

        subprocess_info = create_subprocess_info(
            process=MagicMock(),
            service_id="test_service",
        )
        service_manager.subprocess_info_map["test_service"] = subprocess_info

        with pytest.raises(
            AIPerfError, match="Some services failed to register within timeout"
        ):
            await service_manager.wait_for_all_services_registration(
                stop_event=asyncio.Event(),
                timeout_seconds=0.1,
            )

    @pytest.mark.asyncio
    async def test_wait_for_all_services_start_not_implemented(self, service_manager):
        """Test that wait_for_all_services_start logs not implemented warning."""
        await service_manager.wait_for_all_services_start(
            stop_event=asyncio.Event(),
            timeout_seconds=1.0,
        )

    @pytest.mark.asyncio
    @patch(
        "aiperf.controller.multiprocess_service_manager.asyncio.create_subprocess_exec"
    )
    @patch("aiperf.controller.multiprocess_service_manager.tempfile.NamedTemporaryFile")
    async def test_run_service_replica_success(
        self,
        mock_tempfile,
        mock_create_subprocess,
        service_manager,
        mock_subprocess_process,
    ):
        """Test successful _run_service_replica execution."""
        # Mock temporary file creation
        mock_user_file = MagicMock()
        mock_user_file.name = "/tmp/user_config_test.json"
        mock_user_file.__enter__ = MagicMock(return_value=mock_user_file)
        mock_user_file.__exit__ = MagicMock(return_value=None)

        mock_service_file = MagicMock()
        mock_service_file.name = "/tmp/service_config_test.json"
        mock_service_file.__enter__ = MagicMock(return_value=mock_service_file)
        mock_service_file.__exit__ = MagicMock(return_value=None)

        mock_tempfile.side_effect = [mock_user_file, mock_service_file]
        mock_create_subprocess.return_value = mock_subprocess_process

        async def mock_watch_subprocess(*args, **kwargs):
            pass

        async def mock_handle_output(*args, **kwargs):
            pass

        def mock_execute_async(coro):
            coro.close()

        with (
            patch.object(
                service_manager, "execute_async", side_effect=mock_execute_async
            ) as mock_execute,
            patch.object(
                service_manager, "_watch_subprocess", side_effect=mock_watch_subprocess
            ) as mock_watch,
            patch.object(
                service_manager,
                "_handle_subprocess_output",
                side_effect=mock_handle_output,
            ) as mock_handle,
        ):
            await service_manager._run_service_replica(
                service_type=ServiceType.DATASET_MANAGER,
                service_id="test_service",
                user_config_json='{"test": "user"}',
                service_config_json='{"test": "service"}',
                env={"PYTHONPATH": "/test"},
                current_dir=Path("/test"),
            )

        mock_create_subprocess.assert_called_once()
        args = mock_create_subprocess.call_args[0]
        assert args[0] == "aiperf"
        assert args[1] == "service"
        assert args[2] == ServiceType.DATASET_MANAGER

        assert mock_execute.call_count == 2
        mock_watch.assert_called_once()
        mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_service_multiple_replicas(self, service_manager):
        """Test run_service with multiple replicas."""

        async def mock_run_replica(*args, **kwargs):
            pass

        async def mock_gather(*args, **kwargs):
            return []

        with (
            patch.object(
                service_manager, "_run_service_replica", side_effect=mock_run_replica
            ) as mock_run_replica_patch,
            patch(
                "aiperf.controller.multiprocess_service_manager.asyncio.gather",
                side_effect=mock_gather,
            ) as mock_gather_patch,
        ):
            await service_manager.run_service(
                service_type=ServiceType.DATASET_MANAGER, num_replicas=3
            )

        assert mock_run_replica_patch.call_count == 3
        mock_gather_patch.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_subprocess_output_stream_handling(self, service_manager):
        """Test _handle_subprocess_output stream reading."""
        mock_process = MagicMock()
        mock_stdout = AsyncMock()
        mock_stderr = AsyncMock()

        mock_stdout.read = AsyncMock(return_value=b"")
        mock_stderr.read = AsyncMock(return_value=b"")

        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.pid = 12345

        await service_manager._handle_subprocess_output(mock_process, "test_service")

        mock_stdout.read.assert_called()
        mock_stderr.read.assert_called()

    @pytest.mark.asyncio
    async def test_watch_subprocess_completion(
        self, service_manager, create_subprocess_info, mock_subprocess_process
    ):
        """Test _watch_subprocess when subprocess completes."""
        info = create_subprocess_info(
            process=mock_subprocess_process,
            service_id="test_service",
        )

        mock_subprocess_process.wait.return_value = 0
        mock_subprocess_process.returncode = 0

        with patch.object(
            service_manager, "publish", new_callable=AsyncMock
        ) as mock_publish:
            service_manager.stop_requested = True
            await service_manager._watch_subprocess(info)

        mock_subprocess_process.wait.assert_called_once()
        mock_publish.assert_not_called()
