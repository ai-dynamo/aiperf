# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the dataset-configured rebroadcast loop.

Covers ``DatasetManager._start_dataset_rebroadcast`` and
``_rebroadcast_dataset_configured``: the Kubernetes-only gate around starting
the task, and the ``stop_requested`` early exit inside the loop. Previously
untested (no hits for "rebroadcast" anywhere under tests/unit/dataset/).
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.common.enums import MemoryMapFormat
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.models.dataset_models import DatasetMetadata, MemoryMapClientMetadata
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin.enums import DatasetSamplingStrategy


def _notification() -> DatasetConfiguredNotification:
    client_metadata = MemoryMapClientMetadata(
        format=MemoryMapFormat.PAYLOAD_BYTES,
        data_file_path=Path("/aiperf/datasets/aiperf_mmap_bench-7f2a/dataset.dat"),
        index_file_path=Path("/aiperf/datasets/aiperf_mmap_bench-7f2a/index.bin"),
        conversation_count=4,
        total_size_bytes=1024,
    )
    return DatasetConfiguredNotification(
        service_id="dataset_manager",
        metadata=DatasetMetadata(sampling_strategy=DatasetSamplingStrategy.RANDOM),
        client_metadata=client_metadata,
    )


@pytest.fixture
def dataset_manager(empty_dataset_manager: DatasetManager) -> DatasetManager:
    """empty_dataset_manager with publish mocked so we can assert on rebroadcasts."""
    empty_dataset_manager.publish = AsyncMock()
    return empty_dataset_manager


class TestStartDatasetRebroadcastGating:
    """`_start_dataset_rebroadcast` only starts a task in Kubernetes mode."""

    @pytest.mark.asyncio
    async def test_start_dataset_rebroadcast_kubernetes_run_starts_and_republishes(
        self, dataset_manager: DatasetManager
    ) -> None:
        notification = _notification()
        # deadline = perf_counter() [0.0] + REBROADCAST_WINDOW (default 120.0)
        # -> deadline is 120.0. Two loop iterations then exit: iter1 (< deadline),
        # iter2 (< deadline), iter3 (>= deadline, exits).
        with (
            patch.object(dataset_manager, "_is_kubernetes_run", return_value=True),
            patch(
                "aiperf.dataset.dataset_manager.time.perf_counter",
                side_effect=[0.0, 1.0, 2.0, 200.0],
            ),
        ):
            dataset_manager._start_dataset_rebroadcast(notification)
            assert dataset_manager._dataset_rebroadcast_task is not None
            await dataset_manager._dataset_rebroadcast_task

        assert dataset_manager.publish.await_count == 2
        for call in dataset_manager.publish.await_args_list:
            assert call.args[0] is notification

    @pytest.mark.asyncio
    async def test_start_dataset_rebroadcast_local_run_does_not_rebroadcast(
        self, dataset_manager: DatasetManager
    ) -> None:
        notification = _notification()
        with patch.object(dataset_manager, "_is_kubernetes_run", return_value=False):
            dataset_manager._start_dataset_rebroadcast(notification)

        assert dataset_manager._dataset_rebroadcast_task is None
        dataset_manager.publish.assert_not_awaited()


class TestRebroadcastDatasetConfiguredStopRequested:
    """The loop must exit promptly once `stop_requested` is set, without publishing again."""

    @pytest.mark.asyncio
    async def test_rebroadcast_dataset_configured_stop_requested_exits_without_publishing(
        self, dataset_manager: DatasetManager
    ) -> None:
        notification = _notification()
        dataset_manager.stop_requested = True
        # deadline = 0.0 + 120.0 = 120.0. One loop iteration is needed:
        # a single < deadline check before the stop_requested check short-circuits.
        with patch(
            "aiperf.dataset.dataset_manager.time.perf_counter",
            side_effect=[0.0, 1.0, 200.0],
        ):
            await dataset_manager._rebroadcast_dataset_configured(notification)

        dataset_manager.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rebroadcast_dataset_configured_stop_requested_mid_loop_stops_promptly(
        self, dataset_manager: DatasetManager
    ) -> None:
        """Stop requested after the first republish must prevent a second one."""
        notification = _notification()

        async def publish_then_stop(_msg):
            dataset_manager.stop_requested = True

        dataset_manager.publish.side_effect = publish_then_stop
        # deadline = 0.0 + 120.0 = 120.0. iter1 (< deadline, publishes and sets
        # stop), iter2 (< deadline) would publish again if the stop check were
        # skipped.
        with patch(
            "aiperf.dataset.dataset_manager.time.perf_counter",
            side_effect=[0.0, 1.0, 2.0, 200.0],
        ):
            await dataset_manager._rebroadcast_dataset_configured(notification)

        assert dataset_manager.publish.await_count == 1
