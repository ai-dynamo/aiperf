# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""When a Kubernetes worker is allowed to open the dataset mmap.

Regression: the worker opened the mmap the instant
DatasetConfiguredNotification arrived. In Kubernetes those paths name files on
the *controller* pod; the worker pod's own copy is still being downloaded by
its WorkerGroupManager, which starts on that same notification. Every worker
died with "Data file not found" and the run failed at 'Configure Profiling'.
"""

import asyncio
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import MemoryMapFormat, WorkerStartupState
from aiperf.common.messages import (
    DatasetConfigStatusRequest,
    DatasetConfigStatusResponse,
    DatasetConfiguredNotification,
    DatasetDownloadedNotification,
)
from aiperf.common.models.dataset_models import DatasetMetadata, MemoryMapClientMetadata
from aiperf.credit.messages import WorkerConnected, WorkerDispatchable
from aiperf.plugin.enums import DatasetSamplingStrategy, PluginType
from aiperf.workers.worker import Worker
from tests.harness import mock_plugin


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
        self._worker_ready_lock = asyncio.Lock()
        self._dataset_configured_event = asyncio.Event()
        self._mark_worker_ready_locked = AsyncMock()
        self.session_manager = AsyncMock()
        self.opened: list[MemoryMapClientMetadata] = []
        self.debug = lambda *a, **k: None
        self.warning = lambda *a, **k: None

    async def _open_dataset_client(self, client_metadata, mark_ready=True) -> None:
        self.opened.append(client_metadata)
        self._dataset_configured_event.set()


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


@pytest.mark.asyncio
async def test_download_before_config_warning_names_the_real_recovery_path() -> None:
    """The out-of-order warning must not promise a recovery that cannot happen.

    ``_on_dataset_downloaded`` only runs in Kubernetes mode, and in that mode
    ``_on_dataset_configured`` records the metadata and returns without opening
    anything. The recovery is the group dataset-state poll
    (``_retry_group_dataset_state_until_ready``), which asks this pod's
    WorkerGroupManager directly.
    """
    worker = _FakeWorker(is_kubernetes=True, pod_index="0")
    warnings: list[str] = []
    worker.warning = warnings.append  # type: ignore[method-assign]

    # No DatasetConfiguredNotification seen yet.
    await Worker._on_dataset_downloaded(
        worker,
        DatasetDownloadedNotification(
            service_id="wgm", client_metadata=_metadata("pod"), pod_index="0"
        ),
    )

    assert worker.opened == []
    assert len(warnings) == 1
    message = warnings[0]
    assert "_retry_group_dataset_state_until_ready" in message
    assert "the configuration will open the client" not in message


@pytest.fixture
def local_worker(benchmark_run) -> Worker:
    """Use real readiness state with stubbed startup transports."""
    worker = Worker(run=benchmark_run, service_id="late-worker")
    worker.publish = AsyncMock()
    worker.credit_dealer_client.send = AsyncMock()
    worker.conversation_request_client.request = AsyncMock()
    worker._await_return_channel_ready = AsyncMock()
    worker.execute_async = MagicMock()
    return worker


@pytest.fixture
def local_store() -> Iterator[type]:
    """Track dataset initialization through the real plugin factory."""

    class Store:
        initialize = AsyncMock()

        def __init__(self, client_metadata: MemoryMapClientMetadata) -> None:
            pass

    with mock_plugin(
        PluginType.DATASET_CLIENT_STORE, _metadata("local").client_type, Store
    ):
        yield Store


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["configured", "pending", "timeout"])  # fmt: skip
async def test_local_worker_catches_up_without_blocking_startup(
    local_worker: Worker, local_store: type, status: str
) -> None:
    """A late worker recovers without another broadcast, including after a timeout."""
    notification = _configured()
    request = local_worker.conversation_request_client.request
    configured = DatasetConfigStatusResponse(
        service_id="dataset_manager", notification=notification
    )
    responses: list[DatasetConfigStatusResponse | TimeoutError] = [configured]
    if status == "pending":
        responses.insert(0, DatasetConfigStatusResponse(service_id="dataset_manager"))
    elif status == "timeout":
        responses.insert(0, TimeoutError())
    request.side_effect = responses
    await local_worker._send_worker_ready_message()
    assert local_worker._startup_state == WorkerStartupState.WAITING_FOR_DATASET
    assert not local_worker._worker_ready_event.is_set()
    request.assert_not_awaited()

    await local_worker.execute_async.call_args.args[0]
    assert request.await_count == (1 if status == "configured" else 2)
    assert isinstance(request.call_args.args[0], DatasetConfigStatusRequest)

    local_store.initialize.assert_awaited_once()
    assert local_worker._dataset_configured_event.is_set()
    assert local_worker._worker_ready_event.is_set()
    assert local_worker._startup_state == WorkerStartupState.READY
    assert [
        type(call.args[0])
        for call in local_worker.credit_dealer_client.send.await_args_list
    ] == [WorkerConnected, WorkerDispatchable]


@pytest.mark.asyncio
async def test_local_catch_up_stops_retrying_when_worker_stops(
    local_worker: Worker, local_store: type
) -> None:
    async def stop_during_request(*args: object, **kwargs: object) -> None:
        local_worker.stop_requested = True
        raise TimeoutError

    request = local_worker.conversation_request_client.request
    request.side_effect = stop_during_request
    await local_worker._send_worker_ready_message()
    await local_worker.execute_async.call_args.args[0]

    request.assert_awaited_once()
    local_store.initialize.assert_not_awaited()
    assert not local_worker._worker_ready_event.is_set()


@pytest.mark.asyncio
async def test_open_dataset_client_marks_worker_ready_by_default(
    local_worker: Worker, local_store: type
) -> None:
    await local_worker._open_dataset_client(_metadata("local"))

    local_store.initialize.assert_awaited_once()
    assert local_worker._dataset_configured_event.is_set()
    assert local_worker._worker_ready_event.is_set()
    local_worker.credit_dealer_client.send.assert_awaited_once_with(
        WorkerDispatchable(worker_id=local_worker.service_id)
    )


@pytest.mark.asyncio
async def test_local_catch_up_logs_dataset_open_failure(
    local_worker: Worker, local_store: type, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed background open stays non-dispatchable and reports the cause."""
    local_worker.conversation_request_client.request.return_value = (
        DatasetConfigStatusResponse(
            service_id="dataset_manager", notification=_configured()
        )
    )
    local_store.initialize.side_effect = RuntimeError("dataset open failed")

    await local_worker._send_worker_ready_message()
    await local_worker.execute_async.call_args.args[0]

    assert not local_worker._dataset_configured_event.is_set()
    assert not local_worker._worker_ready_event.is_set()
    assert (
        "Error in background task catch_up_dataset: dataset open failed" in caplog.text
    )
    local_worker.credit_dealer_client.send.assert_awaited_once_with(
        WorkerConnected(worker_id=local_worker.service_id)
    )


@pytest.mark.asyncio
async def test_concurrent_dataset_notifications_open_local_client_once(
    local_worker: Worker,
    local_store: type,
) -> None:
    """A catch-up response racing the broadcast must not replace an open client."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def initialize() -> None:
        entered.set()
        await release.wait()

    local_store.initialize.side_effect = initialize
    notification = _configured()
    first = asyncio.create_task(local_worker._on_dataset_configured(notification))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert not local_worker._dataset_configured_event.is_set()
        assert not local_worker._worker_ready_event.is_set()
        local_worker.credit_dealer_client.send.assert_not_awaited()
        second = asyncio.create_task(local_worker._on_dataset_configured(notification))
    finally:
        release.set()
        await first
    await second

    local_store.initialize.assert_awaited_once()
    assert local_worker._dataset_configured_event.is_set()
    assert local_worker._worker_ready_event.is_set()
    local_worker.credit_dealer_client.send.assert_awaited_once_with(
        WorkerDispatchable(worker_id=local_worker.service_id)
    )


@pytest.mark.asyncio
async def test_local_profile_configure_waits_for_dispatchable(
    local_worker: Worker,
) -> None:
    """Acking configure between dataset open and WorkerDispatchable strands the worker."""
    local_worker.event_loop_monitor = MagicMock()
    local_worker._dataset_configured_event.set()
    configure = asyncio.create_task(
        local_worker._on_profile_configure_command(MagicMock())
    )
    await asyncio.sleep(0)
    assert not configure.done()

    await local_worker._mark_worker_ready()
    await asyncio.wait_for(configure, timeout=1)
    local_worker.event_loop_monitor.start.assert_called_once()
