# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from aiperf.api.routers.dataset import (
    DatasetRouter,
    DatasetStateResponse,
    get_dataset_state,
)
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.models import DatasetMetadata, MemoryMapClientMetadata
from aiperf.plugin.enums import DatasetSamplingStrategy


class _FakeDatasetComponent:
    def __init__(self) -> None:
        self._dataset_configured = asyncio.Event()
        self._dataset_client_metadata = MemoryMapClientMetadata(
            data_file_path=Path("/tmp/dataset.dat"),
            index_file_path=Path("/tmp/index.dat"),
            conversation_count=4,
            total_size_bytes=1024,
        )
        self._benchmark_generation = "gen-1"
        self._dataset_generation = "data-1"
        self.info = Mock()
        self._dataset_configured.set()

    @property
    def dataset_configured(self) -> asyncio.Event:
        return self._dataset_configured

    @property
    def dataset_client_metadata(self) -> MemoryMapClientMetadata:
        return self._dataset_client_metadata

    @property
    def benchmark_generation(self) -> str:
        return self._benchmark_generation

    @property
    def dataset_generation(self) -> str:
        return self._dataset_generation


class TestDatasetRouter:
    @pytest.mark.asyncio
    async def test_on_dataset_configured_stores_generations(self) -> None:
        router = object.__new__(DatasetRouter)
        router._dataset_client_metadata = None
        router._benchmark_generation = None
        router._dataset_generation = None
        router._dataset_configured = asyncio.Event()
        router.info = Mock()
        router.warning = Mock()

        await router._on_dataset_configured(
            DatasetConfiguredNotification(
                service_id="dataset-manager",
                metadata=DatasetMetadata(
                    conversations=[],
                    sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
                ),
                client_metadata=MemoryMapClientMetadata(
                    data_file_path=Path("/tmp/dataset.dat"),
                    index_file_path=Path("/tmp/index.dat"),
                    conversation_count=4,
                    total_size_bytes=1024,
                ),
                benchmark_generation="gen-1",
                dataset_generation="data-1",
            )
        )

        assert router.benchmark_generation == "gen-1"
        assert router.dataset_generation == "data-1"
        assert router.dataset_configured.is_set()

    @pytest.mark.asyncio
    async def test_get_dataset_state_returns_versioned_snapshot(self) -> None:
        response = await get_dataset_state(_FakeDatasetComponent())

        assert isinstance(response, DatasetStateResponse)
        assert response.ready is True
        assert response.benchmark_generation == "gen-1"
        assert response.dataset_generation == "data-1"
        assert response.data_url == "/api/dataset/data"
        assert response.index_url == "/api/dataset/index"
