# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import ServiceRegistrationStatus, ServiceType
from aiperf.common.exceptions import AIPerfError
from aiperf.common.models.service_models import ServiceRunInfo
from aiperf.controller.multiprocess_service_manager import (
    AsyncSubprocessRunInfo,
    MultiProcessServiceManager,
)


class TestAsyncSubprocessRunInfo:
    """Test AsyncSubprocessRunInfo Pydantic model."""

    def test_async_subprocess_run_info_creation(self):
        """Test creating AsyncSubprocessRunInfo with required fields."""
        service_type = ServiceType.DATASET_MANAGER
        service_id = "test_service_123"

        info = AsyncSubprocessRunInfo(
            service_type=service_type,
            service_id=service_id,
        )

        assert info.service_type == service_type
        assert info.service_id == service_id
        assert info.process is None
        assert info.user_config_file is None
        assert info.service_config_file is None

    def test_async_subprocess_run_info_with_all_fields(self):
        """Test creating AsyncSubprocessRunInfo with all fields."""
        mock_process = MagicMock()
        user_config_file = Path("/tmp/user_config.json")
        service_config_file = Path("/tmp/service_config.json")

        info = AsyncSubprocessRunInfo.model_construct(
            process=mock_process,
            service_type=ServiceType.TIMING_MANAGER,
            service_id="test_service_456",
            user_config_file=user_config_file,
            service_config_file=service_config_file,
        )

        assert info.process == mock_process
        assert info.service_type == ServiceType.TIMING_MANAGER
        assert info.service_id == "test_service_456"
        assert info.user_config_file == user_config_file
        assert info.service_config_file == service_config_file


class TestMultiProcessServiceManager:
    """Test MultiProcessServiceManager async subprocess management."""

    @pytest.fixture
    def mock_subprocess_process(self) -> MagicMock:
        """Create a mock asyncio subprocess process."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.returncode = None
        mock_process.wait = AsyncMock(return_value=0)
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        return mock_process

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
    async def test_remove_subprocess_info(self, service_manager):
        """Test _remove_subprocess_info method."""
        service_id = "test_service_123"
        info = AsyncSubprocessRunInfo(
            service_type=ServiceType.DATASET_MANAGER,
            service_id=service_id,
        )

        # Add the info to the map
        service_manager.subprocess_info_map[service_id] = info
        assert service_id in service_manager.subprocess_info_map

        # Remove it
        await service_manager._remove_subprocess_info(info)
        assert service_id not in service_manager.subprocess_info_map

    @pytest.mark.asyncio
    async def test_wait_for_subprocess_already_terminated(self, service_manager):
        """Test _wait_for_subprocess when process is already terminated."""
        mock_process = MagicMock()
        mock_process.returncode = 0  # Already terminated

        info = AsyncSubprocessRunInfo.model_construct(
            process=mock_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="test_service",
        )

        # Should return early without calling terminate
        await service_manager._wait_for_subprocess(info)
        mock_process.terminate.assert_not_called()

    @pytest.mark.asyncio
    async def test_wait_for_subprocess_graceful_termination(
        self, service_manager, mock_subprocess_process
    ):
        """Test _wait_for_subprocess with graceful termination."""
        info = AsyncSubprocessRunInfo.model_construct(
            process=mock_subprocess_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="test_service",
        )

        await service_manager._wait_for_subprocess(info)

        mock_subprocess_process.terminate.assert_called_once()
        mock_subprocess_process.wait.assert_called()

    @pytest.mark.asyncio
    async def test_wait_for_subprocess_with_config_file_cleanup(
        self, service_manager, mock_subprocess_process
    ):
        """Test _wait_for_subprocess cleans up config files."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False) as user_config_file:
            user_config_path = Path(user_config_file.name)
        with tempfile.NamedTemporaryFile(delete=False) as service_config_file:
            service_config_path = Path(service_config_file.name)

        info = AsyncSubprocessRunInfo.model_construct(
            process=mock_subprocess_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="test_service",
            user_config_file=user_config_path,
            service_config_file=service_config_path,
        )

        assert user_config_path.exists()
        assert service_config_path.exists()

        await service_manager._wait_for_subprocess(info)

        # Files should be cleaned up
        assert not user_config_path.exists()
        assert not service_config_path.exists()

    @pytest.mark.asyncio
    async def test_shutdown_all_services_empty_map(self, service_manager):
        """Test shutdown_all_services with empty subprocess map."""
        results = await service_manager.shutdown_all_services()
        assert results == []

    @pytest.mark.asyncio
    async def test_kill_all_services(self, service_manager, mock_subprocess_process):
        """Test kill_all_services method."""
        service_id = "test_service"
        info = AsyncSubprocessRunInfo.model_construct(
            process=mock_subprocess_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id=service_id,
        )
        service_manager.subprocess_info_map[service_id] = info

        await service_manager.kill_all_services()

        mock_subprocess_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_service_by_type(self, service_manager, mock_subprocess_process):
        """Test stop_service method filtering by service type."""
        dataset_service_id = "dataset_service"
        timing_service_id = "timing_service"

        dataset_info = AsyncSubprocessRunInfo.model_construct(
            process=mock_subprocess_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id=dataset_service_id,
        )
        timing_info = AsyncSubprocessRunInfo.model_construct(
            process=MagicMock(),
            service_type=ServiceType.TIMING_MANAGER,
            service_id=timing_service_id,
        )

        service_manager.subprocess_info_map[dataset_service_id] = dataset_info
        service_manager.subprocess_info_map[timing_service_id] = timing_info

        # Stop only dataset services
        results = await service_manager.stop_service(ServiceType.DATASET_MANAGER)

        # Should only affect the dataset service
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_wait_for_all_services_registration_timeout(self, service_manager):
        """Test wait_for_all_services_registration timeout behavior."""
        stop_event = asyncio.Event()

        # Create a service info that's not registered
        service_info = ServiceRunInfo(
            service_type=ServiceType.DATASET_MANAGER,
            service_id="test_service",
            registration_status=ServiceRegistrationStatus.UNREGISTERED,
            required=True,
        )
        service_manager.service_id_map["test_service"] = service_info

        # Add subprocess info
        subprocess_info = AsyncSubprocessRunInfo.model_construct(
            process=MagicMock(),
            service_type=ServiceType.DATASET_MANAGER,
            service_id="test_service",
        )
        service_manager.subprocess_info_map["test_service"] = subprocess_info

        with pytest.raises(
            AIPerfError, match="Some services failed to register within timeout"
        ):
            await service_manager.wait_for_all_services_registration(
                stop_event=stop_event,
                timeout_seconds=0.1,  # Very short timeout
            )

    @pytest.mark.asyncio
    async def test_wait_for_all_services_start_not_implemented(self, service_manager):
        """Test that wait_for_all_services_start logs not implemented warning."""
        stop_event = asyncio.Event()

        # Should complete without error but log a warning
        await service_manager.wait_for_all_services_start(
            stop_event=stop_event,
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

        # Create async mock functions that can be called without creating unawaited coroutines
        async def mock_watch_subprocess(*args, **kwargs):
            pass

        async def mock_handle_output(*args, **kwargs):
            pass

        # Track calls to execute_async
        execute_async_calls = []

        def mock_execute_async(coro):
            execute_async_calls.append(coro)
            # Close the coroutine to prevent warning
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

        # Verify subprocess was created with correct arguments
        mock_create_subprocess.assert_called_once()
        args = mock_create_subprocess.call_args[0]
        assert args[0] == "aiperf"
        assert args[1] == "service"
        assert args[2] == ServiceType.DATASET_MANAGER

        # Verify async tasks were started
        assert (
            mock_execute.call_count == 2
        )  # _watch_subprocess and _handle_subprocess_output

        # Verify the async methods were called (but not awaited due to execute_async)
        mock_watch.assert_called_once()
        mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_service_multiple_replicas(self, service_manager):
        """Test run_service with multiple replicas."""

        # Create async mock functions
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

        # Should create 3 replicas
        assert mock_run_replica_patch.call_count == 3
        mock_gather_patch.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_subprocess_output_stream_handling(self, service_manager):
        """Test _handle_subprocess_output stream reading."""
        mock_process = MagicMock()
        mock_stdout = AsyncMock()
        mock_stderr = AsyncMock()

        # Mock stream.read to return empty data (EOF)
        mock_stdout.read = AsyncMock(return_value=b"")
        mock_stderr.read = AsyncMock(return_value=b"")

        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.pid = 12345

        await service_manager._handle_subprocess_output(mock_process, "test_service")

        # Verify both streams were read
        mock_stdout.read.assert_called()
        mock_stderr.read.assert_called()

    @pytest.mark.asyncio
    async def test_watch_subprocess_completion(
        self, service_manager, mock_subprocess_process
    ):
        """Test _watch_subprocess when subprocess completes."""
        service_id = "test_service"
        info = AsyncSubprocessRunInfo.model_construct(
            process=mock_subprocess_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id=service_id,
        )

        # Mock the process wait to complete successfully
        mock_subprocess_process.wait.return_value = 0
        mock_subprocess_process.returncode = 0

        with patch.object(
            service_manager, "publish", new_callable=AsyncMock
        ) as mock_publish:
            service_manager.stop_requested = True  # Prevent failure message
            await service_manager._watch_subprocess(info)

        mock_subprocess_process.wait.assert_called_once()
        # Should not publish failure message when stop_requested is True
        mock_publish.assert_not_called()
