# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType
from aiperf.common.environment import Environment
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.models import DatasetMetadata, MemoryMapClientMetadata
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.plugin.enums import TimingMode
from aiperf.timing.manager import TimingManager
from tests.unit.timing.conftest import make_dataset_with_schedule

_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    datasets={
        "default": {
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
)


def _make_config(phases: dict | None = None) -> AIPerfConfig:
    """Create an AIPerfConfig for timing manager tests."""
    if phases is None:
        phases = {
            "profiling": {
                "type": "poisson",
                "rate": 10.0,
                "requests": 100,
                "concurrency": 10,
            }
        }
    return AIPerfConfig(**_BASE, phases=phases)


@pytest.fixture
def config() -> AIPerfConfig:
    return _make_config()


def _make_run(config: AIPerfConfig) -> BenchmarkRun:
    return BenchmarkRun(benchmark_id="test", cfg=config, artifact_dir=Path("/tmp/test"))


@pytest.fixture
def create_manager(config, skip_service_registration):
    def _create(cfg: AIPerfConfig | None = None) -> TimingManager:
        return TimingManager(
            run=_make_run(cfg or config),
            service_id="test-timing-manager",
        )

    return _create


@pytest.fixture
def configured_manager(create_manager, config):
    async def async_noop(*args, **kwargs):
        return None

    mgr = create_manager(config)
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


class TestTimingManagerDatasetConfiguration:
    @pytest.mark.parametrize("phase_type", ["fixed_schedule", "poisson"])
    @pytest.mark.asyncio
    async def test_profile_configure_waits_for_dataset_notification(
        self, create_manager, mock_metadata, phase_type
    ) -> None:
        phases = {"profiling": {"type": phase_type, "requests": 100}}
        if phase_type != "fixed_schedule":
            phases["profiling"]["rate"] = 10.0
        cfg = _make_config(phases=phases)
        mgr = create_manager(cfg)
        mock_engine = MagicMock()
        mock_engine.initialize = lambda *a, **kw: asyncio.sleep(0)

        with patch(
            "aiperf.timing.manager.PhaseOrchestrator", return_value=mock_engine
        ) as mock_orch:
            task = asyncio.create_task(
                mgr._profile_configure_command(
                    Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
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
                    benchmark_generation="test-bench",
                    dataset_generation="test-bench:dataset",
                )
            )
            await task
            assert mgr._dataset_metadata == mock_metadata
            assert mock_orch.call_args.kwargs["dataset_metadata"] == mock_metadata

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_dataset_configuration_timeout(self, create_manager) -> None:
        cfg = _make_config(
            phases={"profiling": {"type": "fixed_schedule", "requests": 100}}
        )
        mgr = create_manager(cfg)
        with (
            patch.object(Environment.DATASET, "CONFIGURATION_TIMEOUT", 1),
            pytest.raises(asyncio.TimeoutError),
        ):
            await mgr._profile_configure_command(
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )

    @pytest.mark.asyncio
    async def test_dataset_notification_before_configure(
        self, create_manager, mock_metadata
    ) -> None:
        cfg = _make_config(
            phases={"profiling": {"type": "fixed_schedule", "requests": 100}}
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
                benchmark_generation="test-bench",
                dataset_generation="test-bench:dataset",
            )
        )
        assert mgr._dataset_metadata == mock_metadata

        mock_engine = MagicMock()
        mock_engine.initialize = lambda *a, **kw: asyncio.sleep(0)
        with patch(
            "aiperf.timing.manager.PhaseOrchestrator", return_value=mock_engine
        ) as mock_orch:
            await mgr._profile_configure_command(
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )
            assert mock_orch.call_args.kwargs["dataset_metadata"] == mock_metadata


class TestTimingManagerCancelCommand:
    @pytest.mark.asyncio
    async def test_cancel_calls_orchestrator_cancel(self, configured_manager) -> None:
        await configured_manager._handle_profile_cancel_command(
            Command(cid="test", cmd=CommandType.PROFILE_CANCEL)
        )
        configured_manager._phase_orchestrator.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_without_orchestrator_is_safe(
        self, create_manager, config
    ) -> None:
        mgr = create_manager(config)
        await mgr._handle_profile_cancel_command(
            Command(cid="test", cmd=CommandType.PROFILE_CANCEL)
        )

    @pytest.mark.asyncio
    async def test_cancel_can_be_called_multiple_times(
        self, configured_manager
    ) -> None:
        cmd = Command(cid="test", cmd=CommandType.PROFILE_CANCEL)
        await configured_manager._handle_profile_cancel_command(cmd)
        await configured_manager._handle_profile_cancel_command(cmd)
        assert configured_manager._phase_orchestrator.cancel.call_count == 2


class TestTimingManagerStartProfilingAndInitialization:
    @pytest.mark.asyncio
    async def test_start_profiling_without_orchestrator_raises(
        self, create_manager, config
    ) -> None:
        mgr = create_manager(config)
        with pytest.raises(InvalidStateError, match="No phase orchestrator configured"):
            await mgr._on_start_profiling(
                Command(cid="test", cmd=CommandType.PROFILE_START)
            )

    @pytest.mark.asyncio
    async def test_start_profiling_calls_orchestrator_start(
        self, create_manager, config
    ) -> None:
        mgr = create_manager(config)
        mock_orchestrator = MagicMock()
        start_called = asyncio.Event()

        async def mock_start():
            start_called.set()

        mock_orchestrator.start = mock_start
        mgr._phase_orchestrator = mock_orchestrator

        await mgr._on_start_profiling(
            Command(cid="test", cmd=CommandType.PROFILE_START)
        )
        await asyncio.sleep(0.05)  # Allow execute_async to run
        assert start_called.is_set()

    @pytest.mark.asyncio
    async def test_configure_raises_when_event_set_but_no_metadata(
        self, create_manager, config
    ) -> None:
        mgr = create_manager(config)
        mgr._dataset_configured_event.set()
        with pytest.raises(
            InvalidStateError, match="Dataset metadata is not available"
        ):
            await mgr._profile_configure_command(
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )

    def test_creates_timing_config_from_config(self, create_manager, config) -> None:
        mgr = create_manager(config)
        assert mgr.timing_config.phase_configs[0].timing_mode == TimingMode.REQUEST_RATE

    def test_creates_phase_publisher_and_sticky_router(
        self, create_manager, config
    ) -> None:
        mgr = create_manager(config)
        assert mgr.phase_publisher is not None and mgr.sticky_router is not None

    def test_no_orchestrator_and_event_not_set_initially(
        self, create_manager, config
    ) -> None:
        mgr = create_manager(config)
        assert (
            mgr._phase_orchestrator is None
            and not mgr._dataset_configured_event.is_set()
        )
