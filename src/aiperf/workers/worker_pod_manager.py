# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WorkerPodManager service for Kubernetes worker pods.

This module provides the shared worker-pod infrastructure service. It downloads
the dataset once per pod, runs the local raw-inference proxy, coordinates raw
record uploads, and reports pod capacity to the controller while workers and
record processors run as sibling containers in the same pod.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import zlib
from pathlib import Path

import aiofiles
import aiohttp
import zstandard

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.control_structs import Command, Registration
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    MessageType,
    WorkerStartupState,
    WorkerStatus,
)
from aiperf.common.environment import Environment
from aiperf.common.hooks import (
    background_task,
    on_command,
    on_init,
    on_message,
    on_start,
    on_stop,
)
from aiperf.common.messages import (
    DatasetConfiguredNotification,
    WorkerHealthMessage,
    WorkerStartupStateMessage,
)
from aiperf.common.messages.worker_messages import WorkerStatusSummaryMessage
from aiperf.common.models import (
    MemoryMapClientMetadata,
    ProcessHealth,
    WorkerTaskStats,
)
from aiperf.common.pod_lifecycle_structs import (
    PeerToPodManagerMessage,
    PodDatasetReady,
    PodPeerAck,
    PodPeerHello,
    PodPeerShutdown,
    PodWorkerHealth,
    PodWorkerStartupState,
)
from aiperf.common.protocols import StreamingRouterClientProtocol
from aiperf.config import BenchmarkRun
from aiperf.config.defaults import OutputDefaults
from aiperf.controller.proxy_manager import ProxyManager
from aiperf.plugin.enums import ServiceType
from aiperf.transports.aiohttp_client import create_tcp_connector
from aiperf.workers.worker_manager import WorkerStatusInfo


class WorkerPodManager(BaseComponentService):
    """Coordinates shared worker-pod infrastructure for sibling service containers.

    This service is the main process in a worker pod container. It:
    1. Downloads the dataset once from the control-plane (via HTTP API)
    2. Runs the pod-local raw-inference proxy over the shared IPC volume
    3. Reports worker/record-processor capacity to the control-plane
    4. Republishes dataset download notifications for late-starting workers
    5. Uploads raw record files after sibling record-processor containers flush them

    Architecture:
        Worker Pod (multi-container)
        ┌─────────────────────────────────────────────────────────────┐
        │ WorkerPodManager (main process)                             │
        │   - Downloads dataset once from control-plane               │
        │   - Serves as the pod-local raw-inference proxy host        │
        │                                                             │
        │  Worker containers: worker-0, worker-1, ...                 │
        │  RecordProcessor containers: record-processor-0, ...        │
        │                                                             │
        │  Shared volumes: /aiperf/ipc, /aiperf/datasets, /results    │
        └─────────────────────────────────────────────────────────────┘

    Configuration:
        - workers_per_pod: Number of worker service containers per pod
        - record_processors_per_pod: Number of record processor containers per pod
    """

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        self._pod_index = os.environ.get("AIPERF_POD_INDEX")

        super().__init__(
            run=run,
            service_id=service_id,
            **kwargs,
        )

        cfg = self.run.cfg

        # Configuration for workers per pod
        self.workers_per_pod = (
            cfg.runtime.workers_per_pod or Environment.WORKER.DEFAULT_WORKERS_PER_POD
        )

        # Configuration for record processors per pod
        # Default: 1 RP for every 4 workers, minimum 1.
        # The Kubernetes path should set record_processors_per_pod explicitly.
        if cfg.runtime.record_processors_per_pod is not None:
            self.record_processors_per_pod = cfg.runtime.record_processors_per_pod
        else:
            self.record_processors_per_pod = max(
                1, self.workers_per_pod // Environment.RECORD.PROCESSOR_SCALE_FACTOR
            )

        # Track worker health/startup state across sibling worker containers.
        self.worker_health: dict[str, WorkerStatusInfo] = {}
        self._pod_peer_identities: dict[str, str] = {}
        self._pod_peer_types: dict[str, str] = {}
        self._record_processors_shutdown: set[str] = set()

        self.pod_lifecycle_router: StreamingRouterClientProtocol = (
            self.comms.create_streaming_router_client(
                address=CommAddress.POD_LIFECYCLE,
                bind=True,
                decode_type=PeerToPodManagerMessage,
            )
        )
        self.pod_lifecycle_router.register_receiver(self._on_pod_lifecycle_message)

        # Dataset download state
        self._dataset_downloaded = False
        self._dataset_download_event = asyncio.Event()
        self._dataset_client_metadata: MemoryMapClientMetadata | None = None
        self._dataset_download_task: asyncio.Task[None] | None = None
        self._tokenizer_prefetch_task: asyncio.Task[None] | None = None
        self._stopping = False

        self._proxy_manager = ProxyManager(
            run=self.run,
            enable_raw_inference=True,
        )

        self.info(
            f"WorkerPodManager configured for {self.workers_per_pod} worker container(s) "
            f"and {self.record_processors_per_pod} record processor container(s)"
        )

    def _make_registration(self) -> Registration:
        """Build a Registration with pod capacity info.

        Extends the base registration (which includes pod_name/pod_index)
        with num_workers and num_record_processors so the controller knows
        how many child services to expect from this pod.
        """
        import uuid

        return Registration(
            sid=self.service_id,
            rid=uuid.uuid4().hex,
            stype=str(self.service_type),
            state=str(self.state),
            pod_name=os.environ.get("HOSTNAME"),
            pod_index=self._pod_index,
            num_workers=self.workers_per_pod,
            num_record_processors=self.record_processors_per_pod,
        )

    @on_init
    async def _initialize_proxy(self) -> None:
        """Initialize and start the local raw inference proxy.

        Workers and record processors in this pod communicate through a local
        push/pull proxy instead of routing through the controller.
        """
        await self._proxy_manager.initialize_and_start()

    @on_start
    async def _start_worker_pod_manager(self) -> None:
        """Start the WorkerPodManager.

        Worker and record-processor containers start independently and register
        with the controller on their own. Tokenizer prefetch is kicked off in
        the background so it does not delay pod-manager registration.
        """
        self.info("WorkerPodManager starting...")

        # Warm the tokenizer cache opportunistically without blocking startup.
        # Each K8s pod is a separate machine — the controller's cache warming
        # only covers the controller pod, so worker pods must cache independently.
        if (
            self._tokenizer_prefetch_task is None
            or self._tokenizer_prefetch_task.done()
        ):
            self._tokenizer_prefetch_task = self.execute_async(
                self._prefetch_tokenizers()
            )
        self.debug("Tokenizer prefetch started in background")
        self.debug("Waiting for dataset configuration...")

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(
        self, message: DatasetConfiguredNotification
    ) -> None:
        """Handle dataset configuration notification.

        Downloads the dataset from control-plane so workers can access it via mmap.
        After download, notifies sibling workers directly over the pod lifecycle channel.
        """
        if self._dataset_downloaded:
            self.debug(
                "Dataset already downloaded, re-sending pod lifecycle dataset-ready for late subscribers"
            )
            if self._dataset_client_metadata is not None:
                await self._notify_registered_workers_of_dataset(
                    client_metadata=self._dataset_client_metadata,
                    success=True,
                )
            return

        if self._dataset_download_task is not None:
            self.debug(
                "Dataset download already in progress, waiting for existing task"
            )
            await self._dataset_download_task
            return

        self.info("Received dataset configuration, downloading dataset...")
        self._dataset_download_task = self.execute_async(
            self._download_and_publish_dataset(message)
        )
        try:
            await self._dataset_download_task
        finally:
            self._dataset_download_task = None

    async def _on_pod_lifecycle_message(
        self, identity: str, message: PeerToPodManagerMessage
    ) -> PodPeerAck | None:
        """Handle pod-local lifecycle updates from sibling workers/processors."""
        match message:
            case PodPeerHello():
                self._pod_peer_identities[message.service_id] = identity
                self._pod_peer_types[message.service_id] = message.service_type
                if message.service_type == str(ServiceType.RECORD_PROCESSOR):
                    self._record_processors_shutdown.discard(message.service_id)
                if (
                    message.service_type == str(ServiceType.WORKER)
                    and self._dataset_client_metadata is not None
                ):
                    await self.pod_lifecycle_router.send_to(
                        identity,
                        self._build_pod_dataset_ready(
                            client_metadata=self._dataset_client_metadata,
                            success=self._dataset_downloaded,
                        ),
                    )
                return PodPeerAck(service_id=self.service_id)
            case PodPeerShutdown():
                self._pod_peer_types[message.service_id] = message.service_type
                if message.service_type == str(ServiceType.RECORD_PROCESSOR):
                    self._record_processors_shutdown.add(message.service_id)
                return None
            case PodWorkerHealth():
                info = self._get_or_create_worker_info(message.service_id)
                self._update_worker_status(
                    info,
                    self._worker_health_message_from_struct(message),
                )
                return None
            case PodWorkerStartupState():
                info = self._get_or_create_worker_info(message.service_id)
                info.startup_state = WorkerStartupState(message.startup_state)
                info.startup_state_updated_ns = message.request_ns
                await self._publish_worker_summary()
                return None

    def _build_pod_dataset_ready(
        self,
        *,
        client_metadata: MemoryMapClientMetadata,
        success: bool,
        error_message: str | None = None,
    ) -> PodDatasetReady:
        """Build the pod-local dataset-ready notification."""
        return PodDatasetReady(
            service_id=self.service_id,
            data_file_path=str(client_metadata.data_file_path),
            index_file_path=str(client_metadata.index_file_path),
            conversation_count=client_metadata.conversation_count,
            total_size_bytes=client_metadata.total_size_bytes,
            pod_index=self._pod_index,
            success=success,
            error_message=error_message,
        )

    async def _notify_registered_workers_of_dataset(
        self,
        *,
        client_metadata: MemoryMapClientMetadata,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        """Push dataset availability directly to registered sibling workers."""
        worker_identities = [
            self._pod_peer_identities[service_id]
            for service_id, service_type in self._pod_peer_types.items()
            if service_type == str(ServiceType.WORKER)
            and service_id in self._pod_peer_identities
        ]
        if not worker_identities:
            return
        message = self._build_pod_dataset_ready(
            client_metadata=client_metadata,
            success=success,
            error_message=error_message,
        )
        await asyncio.gather(
            *(
                self.pod_lifecycle_router.send_to(identity, message)
                for identity in worker_identities
            )
        )

    def _worker_health_message_from_struct(
        self, message: PodWorkerHealth
    ) -> WorkerHealthMessage:
        """Convert pod-local worker health struct into the existing model."""
        return WorkerHealthMessage(
            service_id=message.service_id,
            health=ProcessHealth(
                pid=message.pid,
                create_time=message.create_time,
                uptime=message.uptime,
                cpu_usage=message.cpu_usage,
                memory_usage=message.memory_usage,
                pss_memory=message.pss_memory,
                io_counters=message.io_counters,
                cpu_times=message.cpu_times,
                num_ctx_switches=message.num_ctx_switches,
                num_threads=message.num_threads,
            ),
            task_stats=WorkerTaskStats(
                total=message.task_total,
                failed=message.task_failed,
                completed=message.task_completed,
            ),
        )

    async def _download_and_publish_dataset(
        self, message: DatasetConfiguredNotification
    ) -> None:
        """Download the dataset once and publish the local client metadata."""
        cfg = self.run.cfg
        try:
            data_path, index_path = await self._download_dataset()

            # Get file sizes for notification
            data_size = data_path.stat().st_size
            conversation_count = len(message.metadata.conversations)

            self.info(
                f"Dataset download complete, notifying workers: "
                f"{conversation_count} conversations, {data_size} bytes"
            )

            # Notify workers that dataset is ready with client metadata
            client_metadata = MemoryMapClientMetadata(
                data_file_path=data_path,
                index_file_path=index_path,
                conversation_count=conversation_count,
                total_size_bytes=data_size,
            )
            await self._notify_registered_workers_of_dataset(
                client_metadata=client_metadata,
                success=True,
            )

            # Mark downloaded only after successful direct notification so a retry
            # can re-attempt if delivery fails
            self._dataset_client_metadata = client_metadata
            self._dataset_downloaded = True
            self._dataset_download_event.set()

        except Exception as e:
            self.exception(f"Failed to download dataset: {e!r}")
            # Notify workers of failure with placeholder paths
            mmap_base = Environment.DATASET.MMAP_BASE_PATH or Path(
                tempfile.gettempdir()
            )
            benchmark_id = cfg.benchmark_id
            local_dir = mmap_base / f"aiperf_mmap_{benchmark_id}"
            client_metadata = MemoryMapClientMetadata(
                data_file_path=local_dir / "dataset.dat",
                index_file_path=local_dir / "index.dat",
                conversation_count=0,
                total_size_bytes=0,
            )
            await self._notify_registered_workers_of_dataset(
                client_metadata=client_metadata,
                success=False,
                error_message=str(e),
            )
            raise

    async def _download_dataset(self) -> tuple[Path, Path]:
        """Download the dataset from the control-plane API with retry.

        The dataset is downloaded once and saved to local storage (emptyDir volume).
        Workers will then mmap this local file for fast access. Retries with
        exponential backoff on transient network failures.

        The API serves:
        - GET /api/dataset/data → dataset.dat (serialized conversations)
        - GET /api/dataset/index → index.dat (byte offset index)

        Returns:
            Tuple of (data_path, index_path) where files were saved.

        Raises:
            RuntimeError: If download fails after all retries or dataset_api_base_url not configured.
        """
        cfg = self.run.cfg
        if not cfg.runtime.dataset_api_base_url:
            raise RuntimeError(
                "No dataset_api_base_url configured. "
                "WorkerPodManager requires this to download the dataset."
            )

        base_url = cfg.runtime.dataset_api_base_url.rstrip("/")
        self.info(f"Downloading dataset from {base_url}")

        # Determine local storage path for dataset files
        # Use MMAP_BASE_PATH if set (Kubernetes emptyDir), otherwise temp directory
        mmap_base = Environment.DATASET.MMAP_BASE_PATH or Path(tempfile.gettempdir())
        benchmark_id = cfg.benchmark_id
        local_dir = mmap_base / f"aiperf_mmap_{benchmark_id}"
        local_dir.mkdir(parents=True, exist_ok=True)

        data_path = local_dir / "dataset.dat"
        index_path = local_dir / "index.dat"

        self.info(f"Saving dataset to {local_dir}")

        max_retries = Environment.DATASET.DOWNLOAD_MAX_RETRIES
        retry_delay = Environment.DATASET.DOWNLOAD_RETRY_DELAY
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                connector = create_tcp_connector()
                async with aiohttp.ClientSession(connector=connector) as session:
                    await asyncio.gather(
                        self._download_file(session, f"{base_url}/data", data_path),
                        self._download_file(session, f"{base_url}/index", index_path),
                    )

                self.info(
                    f"Dataset download complete: data={data_path.stat().st_size} bytes, "
                    f"index={index_path.stat().st_size} bytes"
                )
                return data_path, index_path

            except (aiohttp.ClientError, RuntimeError) as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_delay * (2**attempt)
                    self.warning(
                        f"Dataset download attempt {attempt + 1}/{max_retries + 1} failed: {e!r}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"Dataset download failed after {max_retries + 1} attempts"
        ) from last_error

    async def _download_file(
        self, session: aiohttp.ClientSession, url: str, dest_path: Path
    ) -> None:
        """Download a file from HTTP to local path with compression support.

        Requests compressed transfer using Accept-Encoding header. The server
        may respond with zstd or gzip compression. aiohttp auto-decompresses
        gzip; zstd is handled manually.

        Args:
            session: aiohttp client session
            url: URL to download from
            dest_path: Local path to save to

        Raises:
            RuntimeError: If download fails
        """
        self.debug(f"Downloading {url} -> {dest_path}")

        # Request best available compression
        headers = {"Accept-Encoding": "zstd, gzip"}

        try:
            # Disable auto_decompress so we can handle zstd manually
            # (aiohttp doesn't have native zstd support)
            async with session.get(
                url, headers=headers, auto_decompress=False
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to download {url}: HTTP {response.status}"
                    )

                content_encoding = response.headers.get("Content-Encoding", "").lower()
                self.debug(f"Response encoding: {content_encoding or 'none'}")

                await self._download_response(response, dest_path, content_encoding)

            self.debug(f"Downloaded {dest_path.stat().st_size} bytes to {dest_path}")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to download {url}: {e}") from e

    async def _download_response(
        self,
        response: aiohttp.ClientResponse,
        dest_path: Path,
        content_encoding: str,
    ) -> None:
        """Download response to file, streaming decompression if needed."""
        if content_encoding == "zstd":
            dctx = zstandard.ZstdDecompressor()
            decompressor = dctx.decompressobj()
        elif content_encoding == "gzip":
            decompressor = zlib.decompressobj(wbits=31)
        else:
            decompressor = None

        async with aiofiles.open(dest_path, "wb") as f:
            async for chunk in response.content.iter_chunked(
                Environment.COMPRESSION.CHUNK_SIZE
            ):
                if decompressor is not None:
                    chunk = decompressor.decompress(chunk)
                if chunk:
                    await f.write(chunk)
            # zlib decompressobj has flush(); zstandard decompressobj does not
            if decompressor is not None and hasattr(decompressor, "flush"):
                remaining = decompressor.flush()
                if remaining:
                    await f.write(remaining)

    async def _prefetch_tokenizers(self) -> None:
        """Warm the shared HF tokenizer cache so sibling containers load from disk.

        Runs ``validate_tokenizer_early`` in a thread to avoid blocking the
        event loop. The resolved names are stored on the resolved config so
        downstream services can reuse the warmed cache from the shared ``/tmp``
        volume instead of stampeding the network.

        Skipped when ``use_server_token_count`` is True because worker pods
        only need the tokenizer for counting response tokens. The controller
        pod still caches it for synthetic dataset generation. Downloading the
        tokenizer here would delay service startup and risk ZMQ connection
        timeouts with the controller.
        """
        if self.run.cfg.endpoint.use_server_token_count:
            self.debug("Tokenizer prefetch skipped (using server token counts)")
            return

        from aiperf.common.aiperf_logger import AIPerfLogger
        from aiperf.common.tokenizer_validator import validate_tokenizer_early

        logger = AIPerfLogger(f"{__name__}.tokenizer_prefetch")
        resolved = await asyncio.to_thread(
            validate_tokenizer_early, self.run.cfg, logger
        )
        if resolved:
            self.run.resolved.tokenizer_names = resolved
            self.info(f"Tokenizer cache warmed: {len(resolved)} model(s)")
        else:
            self.debug("Tokenizer prefetch skipped (not required)")

    def _get_or_create_worker_info(self, worker_id: str) -> WorkerStatusInfo:
        info = self.worker_health.get(worker_id)
        if info is None:
            info = WorkerStatusInfo(worker_id=worker_id)
            self.worker_health[worker_id] = info
        return info

    def _update_worker_status(
        self, info: WorkerStatusInfo, message: WorkerHealthMessage
    ) -> None:
        """Check the status of a worker."""
        info.last_update_ns = time.time_ns()
        if message.task_stats.failed > info.task_stats.failed:
            info.last_error_ns = time.time_ns()
            info.status = WorkerStatus.ERROR
        elif (time.time_ns() - (info.last_error_ns or 0)) / NANOS_PER_SECOND < Environment.WORKER.ERROR_RECOVERY_TIME:  # fmt: skip
            info.status = WorkerStatus.ERROR
        elif message.health.cpu_usage > Environment.WORKER.HIGH_LOAD_CPU_USAGE:
            info.last_high_load_ns = time.time_ns()
            self.warning(
                f"CPU usage for {message.service_id} is {round(message.health.cpu_usage)}%. AIPerf results may be inaccurate."
            )
            info.status = WorkerStatus.HIGH_LOAD
        elif (time.time_ns() - (info.last_high_load_ns or 0)) / NANOS_PER_SECOND < Environment.WORKER.HIGH_LOAD_RECOVERY_TIME:  # fmt: skip
            info.status = WorkerStatus.HIGH_LOAD
        elif message.task_stats.total == 0 or message.task_stats.in_progress == 0:
            info.status = WorkerStatus.IDLE
        else:
            info.status = WorkerStatus.HEALTHY

        info.health = message.health
        info.task_stats = message.task_stats

        agg = info.health_aggregates
        agg.memory_usage.update(message.health.memory_usage)
        agg.cpu_usage.update(message.health.cpu_usage)
        agg.num_threads.update(message.health.num_threads)
        if message.health.num_ctx_switches:
            agg.voluntary_ctx_switches.update(message.health.num_ctx_switches[0])
            agg.involuntary_ctx_switches.update(message.health.num_ctx_switches[1])
        if message.health.io_counters:
            agg.io_read_bytes.update(message.health.io_counters[4])
            agg.io_write_bytes.update(message.health.io_counters[5])
        if message.health.cpu_times:
            agg.cpu_time_user.update(message.health.cpu_times[0])
            agg.cpu_time_system.update(message.health.cpu_times[1])
            agg.cpu_time_iowait.update(message.health.cpu_times[2])

    @background_task(immediate=False, interval=Environment.WORKER.CHECK_INTERVAL)
    async def _worker_status_loop(self) -> None:
        """Check the status of all workers."""
        for info in self.worker_health.values():
            last_activity_ns = max(
                info.last_update_ns or 0,
                info.startup_state_updated_ns or 0,
            )
            if last_activity_ns == 0:
                continue
            if (time.time_ns() - last_activity_ns) / NANOS_PER_SECOND > Environment.WORKER.STALE_TIME:  # fmt: skip
                info.status = WorkerStatus.STALE

    @background_task(
        immediate=False, interval=Environment.WORKER.STATUS_SUMMARY_INTERVAL
    )
    async def _worker_summary_loop(self) -> None:
        """Generate a summary of the worker status."""
        await self._publish_worker_summary()

    async def _publish_worker_summary(self) -> None:
        """Publish the current worker status and startup-state summary."""
        summary = WorkerStatusSummaryMessage(
            service_id=self.service_id,
            worker_statuses={
                worker_id: info.status for worker_id, info in self.worker_health.items()
            },
            worker_startup_states={
                worker_id: info.startup_state
                for worker_id, info in self.worker_health.items()
                if info.startup_state is not None
            },
        )
        await self.publish(summary)

    @on_command(CommandType.REPORT_WORKER_STATUS_SUMMARY)
    async def _on_report_worker_status_summary(self, message: Command) -> None:
        """Publish an immediate worker status summary on controller request."""
        await self._publish_worker_summary()

    @on_message(MessageType.WORKER_HEALTH)
    async def _on_worker_health(self, message: WorkerHealthMessage) -> None:
        """Track worker health from sibling workers and derive a summary status."""
        info = self._get_or_create_worker_info(message.service_id)
        self._update_worker_status(info, message)

    @on_message(MessageType.WORKER_STARTUP_STATE)
    async def _on_worker_startup_state(
        self, message: WorkerStartupStateMessage
    ) -> None:
        info = self._get_or_create_worker_info(message.service_id)
        info.startup_state = message.startup_state
        info.startup_state_updated_ns = message.request_ns
        await self._publish_worker_summary()

    @on_stop
    async def _stop_worker_pod_manager(self) -> None:
        """Stop pod-local infrastructure, then upload raw records to controller."""
        self._stopping = True
        await self._wait_for_record_processor_shutdowns()
        await self._wait_for_raw_record_files()
        await self._proxy_manager.stop()
        await self._upload_raw_records()

    async def _wait_for_record_processor_shutdowns(self) -> None:
        """Wait for sibling record processors to announce a clean local shutdown."""
        if self.record_processors_per_pod <= 0:
            return
        deadline = (
            asyncio.get_running_loop().time()
            + Environment.SERVICE.RAW_RECORD_UPLOAD_TIMEOUT
        )
        while asyncio.get_running_loop().time() < deadline:
            if len(self._record_processors_shutdown) >= self.record_processors_per_pod:
                return
            await asyncio.sleep(0.2)
        self.warning(
            "Timed out waiting for record processors to report local shutdown: "
            f"expected {self.record_processors_per_pod}, got {len(self._record_processors_shutdown)}"
        )

    async def _wait_for_raw_record_files(self) -> None:
        """Wait for sibling record-processor containers to flush raw files locally.

        In the multi-container worker pod, record processors now own their
        lifecycle. Before uploading shared raw-record files to the controller,
        wait for the expected files to appear and stop changing size.
        """
        from aiperf.common.enums import ExportLevel

        cfg = self.run.cfg
        if cfg.output.export_level != ExportLevel.RAW:
            return

        raw_records_dir = (
            cfg.output.artifact_directory / OutputDefaults.RAW_RECORDS_FOLDER
        )
        expected_files = max(1, self.record_processors_per_pod)
        deadline = (
            asyncio.get_running_loop().time()
            + Environment.SERVICE.RAW_RECORD_UPLOAD_TIMEOUT
        )
        last_snapshot: tuple[tuple[str, int], ...] | None = None
        stable_reads = 0

        while asyncio.get_running_loop().time() < deadline:
            files = (
                sorted(raw_records_dir.glob("raw_records_*.jsonl"))
                if raw_records_dir.exists()
                else []
            )
            snapshot = tuple((path.name, path.stat().st_size) for path in files)
            if len(files) >= expected_files and snapshot == last_snapshot:
                stable_reads += 1
                if stable_reads >= 2:
                    return
            else:
                stable_reads = 0
            last_snapshot = snapshot
            await asyncio.sleep(0.5)

        actual_files = len(last_snapshot or ())
        self.warning(
            "Timed out waiting for raw record files to stabilize before upload: "
            f"expected at least {expected_files}, found {actual_files}"
        )

    async def _upload_raw_records(self) -> None:
        """Upload raw record files to the controller API for aggregation.

        After sibling record-processor containers flush their raw record JSONL
        files to the shared results volume, upload them to the controller API so
        the RawRecordAggregator can find and aggregate them.
        """
        from aiperf.common.enums import ExportLevel

        cfg = self.run.cfg
        if cfg.output.export_level != ExportLevel.RAW:
            return

        raw_records_dir = (
            cfg.output.artifact_directory / OutputDefaults.RAW_RECORDS_FOLDER
        )
        if not raw_records_dir.exists():
            self.debug("No raw_records directory found, skipping upload")
            return

        raw_files = list(raw_records_dir.glob("raw_records_*.jsonl"))
        if not raw_files:
            self.debug("No raw record files found, skipping upload")
            return

        upload_base_url = self._get_upload_base_url()
        if not upload_base_url:
            self.warning("Cannot determine controller API URL for raw record upload")
            return

        self.info(f"Uploading {len(raw_files)} raw record file(s) to controller API")

        connector = create_tcp_connector()
        async with aiohttp.ClientSession(connector=connector) as session:
            for file_path in raw_files:
                await self._upload_file(session, upload_base_url, file_path)

    def _get_upload_base_url(self) -> str | None:
        """Derive the results upload URL from the dataset API URL."""
        base_url = self.run.cfg.runtime.dataset_api_base_url
        if not base_url:
            return None
        # dataset_api_base_url is http://{host}:{port}/api/dataset
        # We need http://{host}:{port}/api/results/upload
        api_base = base_url.rsplit("/api/dataset", 1)[0]
        return f"{api_base}/api/results/upload"

    async def _upload_file(
        self, session: aiohttp.ClientSession, upload_base_url: str, file_path: Path
    ) -> None:
        """Upload a single raw record file to the controller API."""
        url = f"{upload_base_url}/{file_path.name}"
        try:
            file_size = file_path.stat().st_size
            file_bytes = await asyncio.to_thread(file_path.read_bytes)
            data = aiohttp.FormData()
            data.add_field(
                "file",
                file_bytes,
                filename=file_path.name,
                content_type="application/x-ndjson",
            )
            async with session.post(
                url, data=data, timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status == 201:
                    self.info(
                        f"Uploaded raw record file: {file_path.name} "
                        f"({file_size:,} bytes)"
                    )
                else:
                    body = await resp.text()
                    self.warning(
                        f"Failed to upload {file_path.name}: "
                        f"HTTP {resp.status} - {body}"
                    )
        except Exception as e:
            self.warning(f"Error uploading {file_path.name}: {e!r}")
