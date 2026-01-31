# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.config import UserConfig
from aiperf.common.environment import Environment
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.messages import (
    CommandMessage,
    DatasetConfiguredNotification,
    ProfileCancelCommand,
    ProfileConfigureCommand,
)
from aiperf.common.models import DatasetMetadata, MemoryMapClientMetadata
from aiperf.plugin.enums import TimingMode
from aiperf.timing.manager import TimingManager
from tests.unit.timing.conftest import make_dataset_with_schedule


@pytest.fixture
def user_config() -> UserConfig:
    mock_endpoint = MagicMock()
    mock_endpoint.urls = ["http://localhost:8000"]
    mock_endpoint.url_selection_strategy = "round_robin"
    return UserConfig.model_construct(
        endpoint=mock_endpoint, _timing_mode=TimingMode.REQUEST_RATE
    )


@pytest.fixture
def create_manager(service_config):
    def _create(cfg: UserConfig) -> TimingManager:
        return TimingManager(
            service_config=service_config,
            user_config=cfg,
            service_id="test-timing-manager",
        )

    return _create


@pytest.fixture
def configured_manager(create_manager, user_config):
    async def async_noop(*args, **kwargs):
        return None

    mgr = create_manager(user_config)
    mgr._phase_orchestrator = MagicMock()
    mgr._phase_orchestrator.start = MagicMock(side_effect=async_noop)
    mgr._phase_orchestrator.stop = MagicMock(side_effect=async_noop)
    mgr._phase_orchestrator.cancel = MagicMock(side_effect=async_noop)
    mgr.initialized_event.set()
    return mgr


@pytest.fixture
def mock_metadata() -> DatasetMetadata:
    return make_dataset_with_schedule(
        schedule=[(0, "conv1"), (100, "conv2"), (200, "conv3")]
    )


def _create_mock_endpoint() -> MagicMock:
    """Create a mock endpoint with required URL attributes."""
    mock_endpoint = MagicMock()
    mock_endpoint.urls = ["http://localhost:8000"]
    mock_endpoint.url_selection_strategy = "round_robin"
    return mock_endpoint


class TestTimingManagerDatasetConfiguration:
    @pytest.mark.parametrize(
        "timing_mode", [TimingMode.FIXED_SCHEDULE, TimingMode.REQUEST_RATE]
    )
    @pytest.mark.asyncio
    async def test_profile_configure_waits_for_dataset_notification(
        self, create_manager, mock_metadata, timing_mode
    ) -> None:
        cfg = UserConfig.model_construct(
            endpoint=_create_mock_endpoint(), _timing_mode=timing_mode
        )
        mgr = create_manager(cfg)
        mock_engine = MagicMock()
        mock_engine.initialize = lambda *a, **kw: asyncio.sleep(0)

        with patch(
            "aiperf.timing.manager.PhaseOrchestrator", return_value=mock_engine
        ) as mock_orch:
            task = asyncio.create_task(
                mgr._profile_configure_command(
                    ProfileConfigureCommand.model_construct(
                        service_id="test-system-controller", config={}
                    )
                )
            )
            await asyncio.sleep(0.2)
            await mgr._on_dataset_configured_notification(
                DatasetConfiguredNotification(
                    service_id="test-dataset-manager",
                    metadata=mock_metadata,
                    client_metadata=MemoryMapClientMetadata(
                        data_file_path=Path("/tmp/test_data.mmap"),
                        index_file_path=Path("/tmp/test_index.mmap"),
                        conversation_count=3,
                        total_size_bytes=1024,
                    ),
                )
            )
            await task
            assert mgr._dataset_metadata == mock_metadata
            assert mock_orch.call_args.kwargs["dataset_metadata"] == mock_metadata

    @pytest.mark.asyncio
    async def test_dataset_configuration_timeout(self, create_manager) -> None:
        cfg = UserConfig.model_construct(
            endpoint=_create_mock_endpoint(), _timing_mode=TimingMode.FIXED_SCHEDULE
        )
        mgr = create_manager(cfg)
        with (
            patch.object(Environment.DATASET, "CONFIGURATION_TIMEOUT", 0.1),
            pytest.raises(asyncio.TimeoutError),
        ):
            await mgr._profile_configure_command(
                ProfileConfigureCommand.model_construct(
                    service_id="test-system-controller", config={}
                )
            )

    @pytest.mark.asyncio
    async def test_dataset_notification_before_configure(
        self, create_manager, mock_metadata
    ) -> None:
        cfg = UserConfig.model_construct(
            endpoint=_create_mock_endpoint(), _timing_mode=TimingMode.FIXED_SCHEDULE
        )
        mgr = create_manager(cfg)
        await mgr._on_dataset_configured_notification(
            DatasetConfiguredNotification(
                service_id="test-dataset-manager",
                metadata=mock_metadata,
                client_metadata=MemoryMapClientMetadata(
                    data_file_path=Path("/tmp/test_data.mmap"),
                    index_file_path=Path("/tmp/test_index.mmap"),
                    conversation_count=3,
                    total_size_bytes=1024,
                ),
            )
        )
        assert mgr._dataset_metadata == mock_metadata

        mock_engine = MagicMock()
        mock_engine.initialize = lambda *a, **kw: asyncio.sleep(0)
        with patch(
            "aiperf.timing.manager.PhaseOrchestrator", return_value=mock_engine
        ) as mock_orch:
            await mgr._profile_configure_command(
                ProfileConfigureCommand.model_construct(
                    service_id="test-system-controller", config={}
                )
            )
            assert mock_orch.call_args.kwargs["dataset_metadata"] == mock_metadata


class TestTimingManagerCancelCommand:
    @pytest.mark.asyncio
    async def test_cancel_calls_orchestrator_cancel(self, configured_manager) -> None:
        await configured_manager._handle_profile_cancel_command(
            ProfileCancelCommand.model_construct(service_id="test-controller")
        )
        configured_manager._phase_orchestrator.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_without_orchestrator_is_safe(
        self, create_manager, user_config
    ) -> None:
        mgr = create_manager(user_config)
        await mgr._handle_profile_cancel_command(
            ProfileCancelCommand.model_construct(service_id="test-controller")
        )

    @pytest.mark.asyncio
    async def test_cancel_can_be_called_multiple_times(
        self, configured_manager
    ) -> None:
        cmd = ProfileCancelCommand.model_construct(service_id="test-controller")
        await configured_manager._handle_profile_cancel_command(cmd)
        await configured_manager._handle_profile_cancel_command(cmd)
        assert configured_manager._phase_orchestrator.cancel.call_count == 2


class TestTimingManagerStartProfilingAndInitialization:
    @pytest.mark.asyncio
    async def test_start_profiling_without_orchestrator_raises(
        self, create_manager, user_config
    ) -> None:
        mgr = create_manager(user_config)
        with pytest.raises(InvalidStateError, match="No phase orchestrator configured"):
            await mgr._on_start_profiling(
                CommandMessage.model_construct(service_id="test-controller")
            )

    @pytest.mark.asyncio
    async def test_start_profiling_calls_orchestrator_start(
        self, create_manager, user_config
    ) -> None:
        mgr = create_manager(user_config)
        mock_orchestrator = MagicMock()
        start_called = asyncio.Event()

        async def mock_start():
            start_called.set()

        mock_orchestrator.start = mock_start
        mgr._phase_orchestrator = mock_orchestrator

        await mgr._on_start_profiling(
            CommandMessage.model_construct(service_id="test-controller")
        )
        await asyncio.sleep(0.05)  # Allow execute_async to run
        assert start_called.is_set()

    @pytest.mark.asyncio
    async def test_configure_raises_when_event_set_but_no_metadata(
        self, create_manager, user_config
    ) -> None:
        mgr = create_manager(user_config)
        mgr._dataset_configured_event.set()
        with pytest.raises(
            InvalidStateError, match="Dataset metadata is not available"
        ):
            await mgr._profile_configure_command(
                ProfileConfigureCommand.model_construct(
                    service_id="test-controller", config={}
                )
            )

    def test_creates_timing_config_from_user_config(
        self, create_manager, user_config
    ) -> None:
        mgr = create_manager(user_config)
        assert mgr.config.phase_configs[0].timing_mode == TimingMode.REQUEST_RATE

    def test_creates_phase_publisher_and_sticky_router(
        self, create_manager, user_config
    ) -> None:
        mgr = create_manager(user_config)
        assert mgr.phase_publisher is not None and mgr.sticky_router is not None

    def test_no_orchestrator_and_event_not_set_initially(
        self, create_manager, user_config
    ) -> None:
        mgr = create_manager(user_config)
        assert (
            mgr._phase_orchestrator is None
            and not mgr._dataset_configured_event.is_set()
        )
