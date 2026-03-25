# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WorkerPodManager service for Kubernetes worker pods.

This module provides a service that manages multiple workers and record processors
within a single Kubernetes pod. It downloads the dataset once and spawns workers
as subprocesses, reducing network overhead and simplifying pod management.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
import zlib
from pathlib import Path

import aiofiles
import aiohttp
import zstandard

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.control_structs import Registration
from aiperf.common.enums import MessageType, WorkerStartupState, WorkerStatus
from aiperf.common.environment import Environment
from aiperf.common.error_queue import ErrorCollector
from aiperf.common.hooks import (
    background_task,
    on_init,
    on_message,
    on_start,
    on_stop,
)
from aiperf.common.logging import get_global_log_queue
from aiperf.common.messages import (
    DatasetConfiguredNotification,
    DatasetDownloadedNotification,
    WorkerHealthMessage,
    WorkerStartupStateMessage,
)
from aiperf.common.models import MemoryMapClientMetadata
from aiperf.common.models.progress_models import WorkerStats
from aiperf.common.subprocess_manager import SubprocessManager
from aiperf.config import BenchmarkRun
from aiperf.controller.proxy_manager import ProxyManager
from aiperf.plugin.enums import ServiceType
from aiperf.transports.aiohttp_client import create_tcp_connector


class WorkerPodManager(BaseComponentService):
    """Manages multiple workers and record processors within a single Kubernetes pod.

    This service is the main process in a worker pod container. It:
    1. Downloads the dataset once from the control-plane (via HTTP API)
    2. Spawns N workers as subprocesses
    3. Spawns M record processors as subprocesses (default: N/4)
    4. Monitors subprocess health and reports to control-plane
    5. Tolerates subprocess crashes (continues with remaining workers)

    Architecture:
        Worker Pod (single container)
        ┌─────────────────────────────────────────────────────────────┐
        │ WorkerPodManager (main process)                             │
        │   - Downloads dataset once from control-plane               │
        │   - Spawns N workers + M record processors as subprocesses  │
        │                                                             │
        │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
        │   │Worker 0 │ │Worker 1 │ │Worker 2 │ │Worker 3 │          │
        │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘          │
        │        └─────┬─────┴─────┬─────┴─────┬─────┘               │
        │              │    IPC    │           │                      │
        │         ┌────▼───────────▼───────────▼────┐                │
        │         │       RecordProcessor 0          │── ZMQ TCP ──► │
        │         └─────────────────────────────────┘                │
        └─────────────────────────────────────────────────────────────┘

    Configuration:
        - workers_per_pod: Number of worker subprocesses (configurable)
        - record_processors_per_pod: Number of record processor subprocesses
          (default: max(1, workers_per_pod / 4))
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

        # Initialize subprocess error collector and manager for spawning workers/RPs
        self._error_collector = ErrorCollector(
            logger=self, exit_errors=self._exit_errors
        )
        self._subprocess_manager = SubprocessManager(
            run=run,
            log_queue=get_global_log_queue(),
            error_queue=self._error_collector.error_queue,
            logger=self,
        )

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

        # Track worker health from subprocesses
        self.worker_health: dict[str, WorkerStats] = {}

        # Dataset download state
        self._dataset_downloaded = False
        self._dataset_download_event = asyncio.Event()
        self._dataset_client_metadata: MemoryMapClientMetadata | None = None
        self._stopping = False

        self._proxy_manager = ProxyManager(
            run=self.run,
            enable_raw_inference=True,
        )

        self.info(
            f"WorkerPodManager configured: {self.workers_per_pod} workers, "
            f"{self.record_processors_per_pod} record processors"
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

        Warms the HF tokenizer cache, then spawns worker and record processor
        subprocesses so they can register with WorkerManager. Workers will wait
        for the dataset to be ready before processing requests.
        """
        self.info("WorkerPodManager starting...")

        # Warm the tokenizer cache on this pod before spawning subprocesses.
        # Each K8s pod is a separate machine — the controller's cache warming
        # only covers the controller pod, so worker pods must cache independently.
        await self._prefetch_tokenizers()

        self.info("Spawning subprocesses...")
        await self._spawn_subprocesses()
        self.debug("Subprocesses spawned, waiting for dataset configuration...")

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(
        self, message: DatasetConfiguredNotification
    ) -> None:
        """Handle dataset configuration notification.

        Downloads the dataset from control-plane so workers can access it via mmap.
        After download, publishes DatasetDownloadedNotification with file paths.
        """
        if self._dataset_downloaded:
            self.debug(
                "Dataset already downloaded, re-publishing notification for late subscribers"
            )
            if self._dataset_client_metadata is not None:
                await self.publish(
                    DatasetDownloadedNotification(
                        service_id=self.service_id,
                        client_metadata=self._dataset_client_metadata,
                        success=True,
                    )
                )
            return

        self.info("Received dataset configuration, downloading dataset...")

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
            await self.publish(
                DatasetDownloadedNotification(
                    service_id=self.service_id,
                    client_metadata=client_metadata,
                    success=True,
                )
            )

            # Mark downloaded only after successful publish so a retry
            # can re-attempt if publish fails
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
            await self.publish(
                DatasetDownloadedNotification(
                    service_id=self.service_id,
                    client_metadata=client_metadata,
                    success=False,
                    error_message=str(e),
                )
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
                    await self._download_file(session, f"{base_url}/data", data_path)
                    await self._download_file(session, f"{base_url}/index", index_path)

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
        """Warm the HF tokenizer cache so subprocesses load from disk only.

        Runs ``validate_tokenizer_early`` in a thread to avoid blocking the
        event loop. The resolved names are stored on the resolved config so
        subprocesses inherit them. After this, ``_enable_hf_offline_mode``
        (already called by bootstrap) ensures subprocesses never hit the network.

        Skipped when ``use_server_token_count`` is True because worker pods
        only need the tokenizer for counting response tokens. The controller
        pod still caches it for synthetic dataset generation. Downloading the
        tokenizer here would delay subprocess spawn and risk ZMQ connection
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

    async def _spawn_subprocesses(self) -> None:
        """Spawn worker and record processor subprocesses."""
        self.info(
            f"Spawning {self.workers_per_pod} workers and "
            f"{self.record_processors_per_pod} record processors"
        )

        pod_id = self._pod_index or uuid.uuid4().hex[:8]

        # Spawn record processors first so they're ready when workers start
        for i in range(self.record_processors_per_pod):
            service_id = f"record_processor_{pod_id}_{i}"
            await self._subprocess_manager.spawn_service(
                service_type=ServiceType.RECORD_PROCESSOR,
                service_id=service_id,
                replicable=True,
            )

        # Spawn workers
        for i in range(self.workers_per_pod):
            service_id = f"worker_{pod_id}_{i}"
            await self._subprocess_manager.spawn_service(
                service_type=ServiceType.WORKER,
                service_id=service_id,
                replicable=True,
            )

        self.info(
            f"Spawned {len(self._subprocess_manager.subprocesses)} subprocesses "
            f"({self.workers_per_pod} workers, {self.record_processors_per_pod} RPs)"
        )

    @on_message(MessageType.WORKER_HEALTH)
    async def _on_worker_health(self, message: WorkerHealthMessage) -> None:
        """Track worker health from subprocesses.

        This allows WorkerPodManager to report aggregate health to the control-plane.
        """
        worker_id = message.service_id
        existing = self.worker_health.get(worker_id)
        self.worker_health[worker_id] = WorkerStats(
            worker_id=worker_id,
            health=message.health,
            task_stats=message.task_stats,
            status=WorkerStatus.HEALTHY,
            startup_state=existing.startup_state if existing else None,
            startup_state_updated_ns=(
                existing.startup_state_updated_ns if existing else None
            ),
            last_update_ns=message.request_ns,
        )

    @on_message(MessageType.WORKER_STARTUP_STATE)
    async def _on_worker_startup_state(
        self, message: WorkerStartupStateMessage
    ) -> None:
        worker_id = message.service_id
        existing = self.worker_health.get(worker_id)
        if existing is None:
            existing = WorkerStats(worker_id=worker_id)
            self.worker_health[worker_id] = existing
        existing.startup_state = message.startup_state
        existing.startup_state_updated_ns = message.request_ns

    @background_task(interval=Environment.WORKER.HEALTH_CHECK_INTERVAL)
    async def _monitor_subprocesses(self) -> None:
        """Monitor subprocess health and handle crashes.

        Crashed subprocesses are logged but not restarted (tolerant mode).
        The pod continues operating with remaining workers.
        """
        if self._stopping:
            return

        dead_processes = self._subprocess_manager.check_alive()
        for info in dead_processes:
            exit_code = info.process.exitcode if info.process else None
            startup_state = None
            if info.service_type == ServiceType.WORKER:
                startup_state = self.worker_health.get(info.service_id, None)
                startup_state = (
                    startup_state.startup_state if startup_state is not None else None
                )
            message = (
                f"Subprocess {info.service_type} ({info.service_id}) exited "
                f"with code {exit_code}"
            )
            if (
                info.service_type == ServiceType.WORKER
                and startup_state is not None
                and startup_state != WorkerStartupState.READY
            ):
                message += (
                    f" before becoming ready (last startup state: {startup_state})"
                )
            self.warning(message)
            # Remove from tracking (tolerant mode - don't restart)
            self._subprocess_manager.remove(info)

            # Check if all workers are gone
            remaining_workers = self._subprocess_manager.get_by_type(ServiceType.WORKER)
            if not remaining_workers:
                self.error("All workers have exited, pod cannot continue")
                await self.stop()
                return

    @on_stop
    async def _stop_worker_pod_manager(self) -> None:
        """Stop all subprocesses gracefully, then upload raw records to controller."""
        self._stopping = True
        subprocess_count = len(self._subprocess_manager.subprocesses)
        self.info(f"Stopping {subprocess_count} subprocesses...")

        await self._subprocess_manager.stop_all()

        self._subprocess_manager.clear()

        self._error_collector.drain_into()

        self.info("All subprocesses stopped")

        await self._proxy_manager.stop()
        await self._upload_raw_records()

    async def _upload_raw_records(self) -> None:
        """Upload raw record files to the controller API for aggregation.

        After subprocesses stop, RecordProcessors have flushed their raw record
        JSONL files to the local artifact directory. This uploads them to the
        controller's API so the RawRecordAggregator can find and aggregate them.
        """
        from aiperf.common.enums import ExportLevel
        from aiperf.config.defaults import OutputDefaults

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
