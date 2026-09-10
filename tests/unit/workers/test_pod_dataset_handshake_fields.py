# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the group-local dataset wire contract."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pytest import param

from aiperf.common.enums import ConversationContextMode, MemoryMapFormat
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.models import DatasetMetadata, MemoryMapClientMetadata
from aiperf.common.pod_lifecycle_structs import (
    GroupDatasetReady,
    GroupDatasetStateSnapshot,
)
from aiperf.config import BenchmarkRun
from aiperf.plugin.enums import DatasetSamplingStrategy, ServiceRunType
from aiperf.workers.worker import Worker
from aiperf.workers.worker_pod_helpers import (
    build_pod_dataset_ready,
    build_pod_dataset_snapshot,
    run_dataset_download,
)


def _client_metadata(tmp_path: Path, fmt: MemoryMapFormat) -> MemoryMapClientMetadata:
    return MemoryMapClientMetadata(
        format=fmt,
        data_file_path=tmp_path / "dataset.dat",
        index_file_path=tmp_path / "index.dat",
        conversation_count=3,
        total_size_bytes=99,
    )


@pytest.mark.parametrize(
    "fmt",
    [
        param(MemoryMapFormat.PAYLOAD_BYTES, id="payload_bytes"),
        param(MemoryMapFormat.CONVERSATION, id="conversation"),
    ],
)  # fmt: skip
def test_build_pod_dataset_ready_preserves_mmap_format(
    tmp_path: Path, fmt: MemoryMapFormat
) -> None:
    ready = build_pod_dataset_ready(
        service_id="wgm-1",
        pod_index="0",
        client_metadata=_client_metadata(tmp_path, fmt),
        success=True,
    )

    assert ready.mmap_format == fmt


@pytest.mark.parametrize(
    "fmt",
    [
        param(MemoryMapFormat.PAYLOAD_BYTES, id="payload_bytes"),
        param(MemoryMapFormat.CONVERSATION, id="conversation"),
    ],
)  # fmt: skip
def test_build_pod_dataset_snapshot_preserves_mmap_format(
    tmp_path: Path, fmt: MemoryMapFormat
) -> None:
    snapshot = build_pod_dataset_snapshot(
        rid="rid-1",
        service_id="wgm-1",
        pod_index="0",
        benchmark_generation="g1",
        dataset_generation="d1",
        dataset_metadata=None,
        client_metadata=_client_metadata(tmp_path, fmt),
        dataset_downloaded=True,
    )

    assert snapshot.mmap_format == fmt


def test_build_pod_dataset_snapshot_without_client_metadata_defaults() -> None:
    snapshot = build_pod_dataset_snapshot(
        rid="rid-1",
        service_id="wgm-1",
        pod_index="0",
        benchmark_generation=None,
        dataset_generation=None,
        dataset_metadata=None,
        client_metadata=None,
        dataset_downloaded=False,
    )

    assert snapshot.mmap_format == MemoryMapFormat.CONVERSATION


def test_build_pod_dataset_ready_carries_default_context_mode(
    tmp_path: Path,
) -> None:
    ready = build_pod_dataset_ready(
        service_id="wgm-1",
        pod_index="0",
        client_metadata=_client_metadata(tmp_path, MemoryMapFormat.CONVERSATION),
        success=True,
        default_context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
    )

    assert ready.default_context_mode == ConversationContextMode.DELTAS_WITH_RESPONSES


@pytest.mark.asyncio
async def test_run_dataset_download_preserves_mmap_format(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.dat"
    data_path.write_bytes(b"x" * 17)
    index_path = tmp_path / "index.dat"
    index_path.write_bytes(b"y")
    message = DatasetConfiguredNotification(
        service_id="dataset-manager",
        metadata=DatasetMetadata(
            conversations=[],
            sampling_strategy=DatasetSamplingStrategy.RANDOM,
        ),
        client_metadata=_client_metadata(tmp_path, MemoryMapFormat.PAYLOAD_BYTES),
    )
    captured: dict[str, object] = {}

    async def _download() -> tuple[Path, Path]:
        return data_path, index_path

    async def _notify(**kwargs: object) -> None:
        captured.update(kwargs)

    async def _publish_summary() -> None: ...

    class _Logger:
        def info(self, msg: str) -> None: ...
        def debug(self, msg: str) -> None: ...
        def warning(self, msg: str) -> None: ...
        def exception(self, msg: str) -> None: ...

    result = await run_dataset_download(
        run=None,
        message=message,
        download_fn=_download,
        notify_fn=_notify,
        publish_summary_fn=_publish_summary,
        logger=_Logger(),
    )

    assert result.format == MemoryMapFormat.PAYLOAD_BYTES
    client_metadata = captured["client_metadata"]
    assert isinstance(client_metadata, MemoryMapClientMetadata)
    assert client_metadata.format == MemoryMapFormat.PAYLOAD_BYTES


def _k8s_worker(run: BenchmarkRun) -> Worker:
    run.cfg.runtime.service_run_type = ServiceRunType.KUBERNETES
    worker = Worker(
        run=run,
        service_id="k8s-worker",
    )
    worker._pod_index = "0"
    worker.credit_dealer_client.send = AsyncMock()
    worker._open_dataset_client = AsyncMock()
    worker.session_manager.set_default_context_mode = Mock()
    return worker


def _dataset_ready(run: BenchmarkRun, **overrides: object) -> GroupDatasetReady:
    benchmark_id = run.benchmark_id
    values = {
        "service_id": "worker-pod-manager",
        "data_file_path": f"/aiperf/datasets/aiperf_mmap_{benchmark_id}/dataset.dat",
        "index_file_path": f"/aiperf/datasets/aiperf_mmap_{benchmark_id}/index.dat",
        "conversation_count": 4,
        "total_size_bytes": 1024,
        "pod_index": "0",
        "success": True,
    }
    values.update(overrides)
    return GroupDatasetReady(**values)


@pytest.mark.asyncio
async def test_on_dataset_ready_applies_pushed_default_context_mode(
    benchmark_run: BenchmarkRun,
) -> None:
    worker = _k8s_worker(benchmark_run)
    worker._query_pod_dataset_state = AsyncMock(return_value=None)

    await worker._on_pod_lifecycle_message(
        _dataset_ready(
            benchmark_run,
            mmap_format=MemoryMapFormat.PAYLOAD_BYTES,
            default_context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
        )
    )

    client_metadata = worker._open_dataset_client.await_args.args[0]
    assert client_metadata.format == MemoryMapFormat.PAYLOAD_BYTES
    worker.session_manager.set_default_context_mode.assert_called_once_with(
        ConversationContextMode.DELTAS_WITH_RESPONSES
    )
    worker._query_pod_dataset_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_dataset_ready_falls_back_to_snapshot_context_mode(
    benchmark_run: BenchmarkRun,
) -> None:
    worker = _k8s_worker(benchmark_run)
    worker._query_pod_dataset_state = AsyncMock(
        return_value=GroupDatasetStateSnapshot(
            rid="rid-1",
            service_id="worker-pod-manager",
            default_context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
            ready=True,
        )
    )

    await worker._on_pod_lifecycle_message(_dataset_ready(benchmark_run))

    worker._query_pod_dataset_state.assert_awaited_once()
    worker.session_manager.set_default_context_mode.assert_called_once_with(
        ConversationContextMode.DELTAS_WITH_RESPONSES
    )


@pytest.mark.asyncio
async def test_on_dataset_ready_no_context_mode_anywhere_leaves_default(
    benchmark_run: BenchmarkRun,
) -> None:
    worker = _k8s_worker(benchmark_run)
    worker._query_pod_dataset_state = AsyncMock(return_value=None)

    await worker._on_pod_lifecycle_message(_dataset_ready(benchmark_run))

    worker.session_manager.set_default_context_mode.assert_not_called()
    assert worker._worker_ready_event.is_set()
