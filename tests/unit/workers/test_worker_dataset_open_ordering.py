# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""When a Kubernetes worker is allowed to open the dataset mmap.

Regression: the worker opened the mmap the instant
DatasetConfiguredNotification arrived. In Kubernetes those paths name files on
the *controller* pod; the worker pod's own copy is still being downloaded by
its WorkerGroupManager, which starts on that same notification. Every worker
died with "Data file not found" and the run failed at 'Configure Profiling'.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import MemoryMapFormat
from aiperf.common.messages import (
    DatasetConfiguredNotification,
    DatasetDownloadedNotification,
)
from aiperf.common.models.dataset_models import DatasetMetadata, MemoryMapClientMetadata
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.workers.worker import Worker


def _metadata(tag: str) -> MemoryMapClientMetadata:
    return MemoryMapClientMetadata(
        format=MemoryMapFormat.CONVERSATION,
        data_file_path=Path(f"/{tag}/dataset.dat"),
        index_file_path=Path(f"/{tag}/index.dat"),
        conversation_count=1,
        total_size_bytes=8,
    )


def _configured() -> DatasetConfiguredNotification:
    return DatasetConfiguredNotification(
        service_id="dataset_manager",
        metadata=DatasetMetadata(sampling_strategy=DatasetSamplingStrategy.RANDOM),
        client_metadata=_metadata("controller"),
    )


class _FakeWorker:
    """Minimal stand-in carrying only the state the two handlers touch."""

    def __init__(self, *, is_kubernetes: bool, pod_index: str | None) -> None:
        self._is_kubernetes = is_kubernetes
        self._pod_index = pod_index
        self._pending_dataset_metadata = None
        self.session_manager = AsyncMock()
        self.opened: list[MemoryMapClientMetadata] = []
        self.debug = lambda *a, **k: None
        self.warning = lambda *a, **k: None

    async def _open_dataset_client(self, client_metadata, mark_ready=True) -> None:
        self.opened.append(client_metadata)


@pytest.mark.asyncio
async def test_kubernetes_worker_defers_open_until_its_pod_downloaded() -> None:
    worker = _FakeWorker(is_kubernetes=True, pod_index="0")

    await Worker._on_dataset_configured(worker, _configured())
    assert worker.opened == [], "must not open the controller's paths"

    await Worker._on_dataset_downloaded(
        worker,
        DatasetDownloadedNotification(
            service_id="wgm", client_metadata=_metadata("pod"), pod_index="0"
        ),
    )
    assert [m.data_file_path for m in worker.opened] == [Path("/pod/dataset.dat")]


@pytest.mark.asyncio
async def test_worker_ignores_another_pods_download() -> None:
    """Sibling pods' emptyDirs are invisible to this container."""
    worker = _FakeWorker(is_kubernetes=True, pod_index="0")
    await Worker._on_dataset_configured(worker, _configured())

    await Worker._on_dataset_downloaded(
        worker,
        DatasetDownloadedNotification(
            service_id="wgm", client_metadata=_metadata("pod1"), pod_index="1"
        ),
    )
    assert worker.opened == []


@pytest.mark.asyncio
async def test_worker_ignores_a_failed_download() -> None:
    worker = _FakeWorker(is_kubernetes=True, pod_index="0")
    await Worker._on_dataset_configured(worker, _configured())

    await Worker._on_dataset_downloaded(
        worker,
        DatasetDownloadedNotification(
            service_id="wgm",
            client_metadata=_metadata("placeholder"),
            pod_index="0",
            success=False,
            error_message="HTTP 404",
        ),
    )
    assert worker.opened == []


@pytest.mark.asyncio
async def test_non_kubernetes_worker_opens_immediately() -> None:
    """The multiprocessing path must keep opening on the configured notification."""
    worker = _FakeWorker(is_kubernetes=False, pod_index=None)

    await Worker._on_dataset_configured(worker, _configured())
    assert [m.data_file_path for m in worker.opened] == [
        Path("/controller/dataset.dat")
    ]
