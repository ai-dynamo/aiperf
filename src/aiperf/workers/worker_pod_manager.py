# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WorkerGroupManager service for Kubernetes worker pods.

This module provides the shared worker-pod infrastructure service. It downloads
the dataset once per pod, runs the local raw-inference proxy, coordinates raw
record uploads, and reports pod capacity to the controller while workers and
record processors run as sibling containers in the same pod.
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
from aiperf.common.control_structs import Command, Registration
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    MessageType,
    WorkerStartupState,
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
from aiperf.common.messages.worker_messages import WorkerPodStateMessage
from aiperf.common.models import (
    MemoryMapClientMetadata,
    ProcessHealth,
    WorkerTaskStats,
)
from aiperf.common.pod_lifecycle_structs import (
    GroupDatasetReady,
    GroupDatasetStateQuery,
    GroupDatasetStateSnapshot,
    GroupPeerAck,
    GroupPeerCommand,
    GroupPeerCommandAck,
    GroupPeerHello,
    GroupPeerShutdown,
    GroupWorkerHealth,
    GroupWorkerStartupState,
    PeerToGroupManagerMessage,
)
from aiperf.common.protocols import StreamingRouterClientProtocol
from aiperf.common.subprocess_manager import SubprocessInfo, SubprocessManager
from aiperf.config import BenchmarkRun
from aiperf.config.defaults import OutputDefaults
from aiperf.controller.proxy_manager import ProxyManager
from aiperf.plugin.enums import ServiceType
from aiperf.transports.aiohttp_client import create_tcp_connector
from aiperf.workers.group_runtime import GroupRuntimeAdapter, GroupRuntimeRegistration
from aiperf.workers.worker_group_state import (
    WorkerStatusInfo,
    build_worker_status_summary,
    mark_stale_workers,
    update_worker_status,
)


class WorkerGroupManagerBase(BaseComponentService):
    """Coordinates shared worker-pod infrastructure for sibling service containers.

    This service is the main process in a worker pod container. It:
    1. Downloads the dataset once from the control-plane (via HTTP API)
    2. Runs the group-local raw-inference proxy over the shared IPC volume
    3. Owns the pod's only controller-facing lifecycle connection
    4. Configures and shuts down group-local workers and record processors
    5. Republishes dataset download notifications for late-starting workers
    6. Uploads raw record files after sibling record-processor containers flush them

    Architecture:
        Worker Pod (multi-container)
        ┌─────────────────────────────────────────────────────────────┐
        │ WorkerGroupManager (main process)                             │
        │   - Downloads dataset once from control-plane               │
        │   - Serves as the group-local raw-inference proxy host        │
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

    @property
    def service_type(self) -> str:
        """Expose the Kubernetes worker group-manager service identity."""
        return str(ServiceType.WORKER_GROUP_MANAGER)

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        runtime_adapter: GroupRuntimeAdapter | None = None,
        **kwargs,
    ) -> None:
        self._pod_index = os.environ.get("AIPERF_POD_INDEX")
        self._runtime_adapter = runtime_adapter
        self._runtime_registration: GroupRuntimeRegistration | None = (
            runtime_adapter.build_registration()
            if runtime_adapter is not None
            else None
        )

        super().__init__(
            run=run,
            service_id=service_id,
            **kwargs,
        )

        cfg = self.run.cfg

        if self._runtime_registration is not None:
            self.workers_per_pod = self._runtime_registration.declared_workers
            self.record_processors_per_pod = (
                self._runtime_registration.declared_record_processors
            )
        else:
            # Configuration for workers per pod
            self.workers_per_pod = (
                cfg.runtime.workers_per_pod
                or Environment.WORKER.DEFAULT_WORKERS_PER_POD
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
        self._configure_started = False

        self.pod_lifecycle_router: StreamingRouterClientProtocol = (
            self.comms.create_streaming_router_client(
                address=CommAddress.GROUP_LIFECYCLE,
                bind=True,
                decode_type=PeerToGroupManagerMessage,
            )
        )
        self.pod_lifecycle_router.register_receiver(self._on_pod_lifecycle_message)

        # Dataset download state
        self._dataset_downloaded = False
        self._dataset_download_event = asyncio.Event()
        self._dataset_client_metadata: MemoryMapClientMetadata | None = None
        self._dataset_metadata = None
        self._benchmark_generation: str | None = None
        self._dataset_generation: str | None = None
        self._dataset_download_task: asyncio.Task[None] | None = None
        self._tokenizer_prefetch_task: asyncio.Task[None] | None = None
        self._stopping = False

        self._proxy_manager = ProxyManager(
            run=self.run,
            enable_raw_inference=True,
        )
        self._local_subprocess_manager: SubprocessManager | None = None
        if self._runtime_registration is not None:
            self._local_subprocess_manager = SubprocessManager(
                run=self.run,
                logger=self,
            )
            self._local_subprocess_manager._local_worker_group_manager = SubprocessInfo(
                service_type=ServiceType.WORKER_GROUP_MANAGER,
                service_id=self.service_id,
                launch_adapter=self._runtime_adapter,
            )

        self.info(
            f"WorkerGroupManager configured for {self.workers_per_pod} worker container(s) "
            f"and {self.record_processors_per_pod} record processor container(s)"
        )

    def _make_registration(self) -> Registration:
        """Build a Registration with pod capacity info.

        Extends the base registration (which includes pod_name/pod_index)
        with num_workers and num_record_processors so the controller knows
        how many child services to expect from this pod.
        """
        import uuid

        registration = self._runtime_registration
        return Registration(
            sid=self.service_id,
            rid=uuid.uuid4().hex,
            stype=str(self.service_type),
            state=str(self.state),
            pod_name=os.environ.get("HOSTNAME"),
            pod_index=self._pod_index,
            num_workers=(
                registration.declared_workers
                if registration is not None
                else self.workers_per_pod
            ),
            num_record_processors=(
                registration.declared_record_processors
                if registration is not None
                else self.record_processors_per_pod
            ),
        )

    @on_init
    async def _initialize_proxy(self) -> None:
        """Initialize and start the local raw inference proxy.

        Workers and record processors in this pod communicate through a local
        push/pull proxy instead of routing through the controller.
        """
        await self._proxy_manager.initialize_and_start()

    @on_start
    async def _start_worker_group_manager(self) -> None:
        """Start the WorkerGroupManager.

        Worker and record-processor containers start independently and register
        with the group-local lifecycle router. Tokenizer prefetch is kicked off in
        the background so it does not delay pod-manager registration.
        """
        self.info("WorkerGroupManager starting...")

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
        if self._local_subprocess_manager is not None:
            await self._start_local_peers()
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
        self._dataset_metadata = message.metadata
        self._benchmark_generation = message.benchmark_generation
        self._dataset_generation = message.dataset_generation
        await self._publish_worker_summary()

        if self._dataset_downloaded:
            self.debug(
                "Dataset already downloaded; late workers should query group-local current state"
            )
            return

        if self._dataset_download_task is not None:
            self.debug(
                "Dataset download already in progress, waiting for existing task"
            )
            await self._dataset_download_task
            return

        # Take the local fast-path when a runtime adapter is attached or
        # when running in the fake in-process component-integration mode,
        # where dataset files already live on the same filesystem and can
        # be attached directly from the notification.
        fake_mode = os.environ.get("AIPERF_FAKE_IN_PROCESS_MODE") == "1"
        local_fast_path = self._runtime_registration is not None or fake_mode
        if local_fast_path:
            self.info("Received dataset configuration, attaching local dataset state")
            self._dataset_client_metadata = message.client_metadata
            self._dataset_downloaded = True
            self._dataset_download_event.set()
            await self._notify_registered_workers_of_dataset(
                client_metadata=message.client_metadata,
                success=True,
            )
            await self._publish_worker_summary()
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
        self, identity: str, message: PeerToGroupManagerMessage
    ) -> GroupPeerAck | None:
        """Handle group-local lifecycle updates from sibling workers/processors."""
        match message:
            case GroupPeerHello():
                self._pod_peer_identities[message.service_id] = identity
                self._pod_peer_types[message.service_id] = message.service_type
                if message.service_type == str(ServiceType.RECORD_PROCESSOR):
                    self._record_processors_shutdown.discard(message.service_id)
                return GroupPeerAck(rid=message.rid, service_id=self.service_id)
            case GroupPeerShutdown():
                self._pod_peer_types[message.service_id] = message.service_type
                if message.service_type == str(ServiceType.RECORD_PROCESSOR):
                    self._record_processors_shutdown.add(message.service_id)
                return None
            case GroupWorkerHealth():
                info = self._get_or_create_worker_info(message.service_id)
                self._update_worker_status(
                    info,
                    self._worker_health_message_from_struct(message),
                )
                return None
            case GroupWorkerStartupState():
                info = self._get_or_create_worker_info(message.service_id)
                info.startup_state = WorkerStartupState(message.startup_state)
                info.startup_state_updated_ns = message.request_ns
                await self._publish_worker_summary()
                return None
            case GroupDatasetStateQuery():
                return self._build_pod_dataset_snapshot(message.rid)
            case GroupPeerCommandAck():
                return message

    def _expected_peer_counts(self) -> dict[str, int]:
        """Return the required group-local peer counts by service type."""
        return {
            str(ServiceType.WORKER): self.workers_per_pod,
            str(ServiceType.RECORD_PROCESSOR): self.record_processors_per_pod,
        }

    def _registered_peer_counts(self) -> dict[str, int]:
        """Return the currently registered group-local peer counts by service type."""
        counts: dict[str, int] = {}
        for service_type in self._pod_peer_types.values():
            counts[service_type] = counts.get(service_type, 0) + 1
        return counts

    async def _wait_for_expected_peers(self) -> None:
        """Wait for the full group-local worker and record-processor set to register."""
        deadline = (
            asyncio.get_running_loop().time()
            + Environment.SERVICE.PROFILE_CONFIGURE_TIMEOUT
        )
        expected = self._expected_peer_counts()
        while asyncio.get_running_loop().time() < deadline:
            counts = self._registered_peer_counts()
            if all(
                counts.get(service_type, 0) >= expected_count
                for service_type, expected_count in expected.items()
            ):
                return
            await asyncio.sleep(0.2)
        counts = self._registered_peer_counts()
        raise TimeoutError(
            "Timed out waiting for group-local peers to register: "
            f"expected={expected}, registered={counts}"
        )

    async def _send_pod_command(self, service_id: str, command: CommandType) -> None:
        """Send a group-local lifecycle command and wait for its ack."""
        identity = self._pod_peer_identities[service_id]
        response = await self.pod_lifecycle_router.request_to(
            identity,
            GroupPeerCommand(
                cid=uuid.uuid4().hex,
                service_id=self.service_id,
                command=str(command),
            ),
            timeout=Environment.SERVICE.PROFILE_CONFIGURE_TIMEOUT,
        )
        if not isinstance(response, GroupPeerCommandAck):
            raise TypeError(
                f"Unexpected group-local response from {service_id}: {type(response).__name__}"
            )

    async def _wait_for_local_startup_convergence(self) -> None:
        """Wait for the group-local worker and record-processor set to converge."""
        await self._wait_for_expected_peers()
        await self._dataset_download_event.wait()

    async def _configure_local_peers(self) -> None:
        """Fan out PROFILE_CONFIGURE to group-local workers and record processors."""
        peer_ids = list(self._pod_peer_identities)
        if not peer_ids:
            return
        await asyncio.gather(
            *(
                self._send_pod_command(service_id, CommandType.PROFILE_CONFIGURE)
                for service_id in peer_ids
            )
        )

    async def _start_local_peers(self) -> None:
        """Spawn local worker and record-processor subprocesses under this group."""
        if self._local_subprocess_manager is None:
            return
        await self._local_subprocess_manager.spawn_services(
            ServiceType.WORKER,
            self.workers_per_pod,
        )
        await self._local_subprocess_manager.spawn_services(
            ServiceType.RECORD_PROCESSOR,
            self.record_processors_per_pod,
        )

    async def _shutdown_local_peers(self) -> None:
        """Ask group-local workers and record processors to shut down.

        When this WGM's lifecycle has already recorded an exit error (i.e. an
        on_init/on_start hook failed and we are being torn down by ``_fail``),
        send ``ABORT`` instead of ``SHUTDOWN``. Peers that receive ABORT will
        ``os._exit(1)`` so kubelet restarts the containers — otherwise they
        would exit 0 and leave the pod stuck at 1/13 Ready with only a
        restarted WGM.
        """
        peer_ids = list(self._pod_peer_identities)
        if not peer_ids:
            return
        command = CommandType.ABORT if self._exit_errors else CommandType.SHUTDOWN
        results = await asyncio.gather(
            *(self._send_pod_command(service_id, command) for service_id in peer_ids),
            return_exceptions=True,
        )
        for service_id, result in zip(peer_ids, results, strict=False):
            if isinstance(result, Exception):
                self.warning(
                    f"Failed to send {command} to group-local peer "
                    f"{service_id}: {result!r}"
                )

    def _build_pod_dataset_ready(
        self,
        *,
        client_metadata: MemoryMapClientMetadata,
        success: bool,
        error_message: str | None = None,
    ) -> GroupDatasetReady:
        """Build the group-local dataset-ready notification."""
        return GroupDatasetReady(
            service_id=self.service_id,
            data_file_path=str(client_metadata.data_file_path),
            index_file_path=str(client_metadata.index_file_path),
            conversation_count=client_metadata.conversation_count,
            total_size_bytes=client_metadata.total_size_bytes,
            pod_index=self._pod_index,
            success=success,
            error_message=error_message,
        )

    def _build_pod_dataset_snapshot(self, rid: str) -> GroupDatasetStateSnapshot:
        """Build a queryable current-state dataset snapshot for sibling workers."""
        metadata = self._dataset_client_metadata
        return GroupDatasetStateSnapshot(
            rid=rid,
            service_id=self.service_id,
            benchmark_generation=self._benchmark_generation,
            dataset_generation=self._dataset_generation,
            default_context_mode=(
                self._dataset_metadata.default_context_mode
                if self._dataset_metadata is not None
                else None
            ),
            data_file_path=str(metadata.data_file_path)
            if metadata is not None
            else None,
            index_file_path=(
                str(metadata.index_file_path) if metadata is not None else None
            ),
            conversation_count=metadata.conversation_count
            if metadata is not None
            else 0,
            total_size_bytes=metadata.total_size_bytes if metadata is not None else 0,
            pod_index=self._pod_index,
            ready=self._dataset_downloaded and metadata is not None,
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
        self, message: GroupWorkerHealth
    ) -> WorkerHealthMessage:
        """Convert group-local worker health struct into the existing model."""
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
            await self._publish_worker_summary()

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
            await self._publish_worker_summary()
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
                "WorkerGroupManager requires this to download the dataset."
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
        update_worker_status(info, message, warning=self.warning)

    @background_task(immediate=False, interval=Environment.WORKER.CHECK_INTERVAL)
    async def _worker_status_loop(self) -> None:
        """Check the status of all workers."""
        mark_stale_workers(self.worker_health)

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _on_profile_configure(self, message: Command) -> None:
        """Wait for group-local startup convergence before profiling."""
        if self._configure_started:
            return
        self._configure_started = True
        # When no runtime adapter is attached, this WGM owns no group-local
        # peers (in-process fake service manager): workers and record
        # processors run as independent services addressed directly by the
        # SystemController, so there is nothing to wait for or fan out to.
        if self._runtime_registration is None:
            await self._publish_worker_summary()
            return
        await self._wait_for_local_startup_convergence()
        await self._configure_local_peers()
        await self._publish_worker_summary()

    @background_task(
        immediate=False, interval=Environment.WORKER.STATUS_SUMMARY_INTERVAL
    )
    async def _worker_summary_loop(self) -> None:
        """Generate a summary of the worker status."""
        await self._publish_worker_summary()

    async def _publish_worker_summary(self) -> None:
        """Publish worker-centric and pod-centric state snapshots."""
        summary = build_worker_status_summary(
            service_id=self.service_id,
            worker_infos=self.worker_health,
        )
        startup_states = summary.worker_startup_states
        ready_workers = sum(
            1 for state in startup_states.values() if state == WorkerStartupState.READY
        )
        router_connected_workers = sum(
            1
            for state in startup_states.values()
            if state
            in {
                WorkerStartupState.ROUTER_PROBING,
                WorkerStartupState.WAITING_FOR_DATASET,
                WorkerStartupState.READY,
            }
        )
        ready_record_processors = sum(
            1
            for service_type in self._pod_peer_types.values()
            if service_type == str(ServiceType.RECORD_PROCESSOR)
        )
        pod_state = (
            "ready"
            if ready_workers >= 1 and ready_record_processors >= 1
            else "starting"
        )
        pod_summary = WorkerPodStateMessage(
            service_id=self.service_id,
            pod_index=self._pod_index or "",
            benchmark_generation=self._benchmark_generation,
            dataset_generation=self._dataset_generation,
            declared_workers=self.workers_per_pod,
            declared_record_processors=self.record_processors_per_pod,
            router_connected_workers=router_connected_workers,
            dispatchable_workers=ready_workers,
            ready_workers=ready_workers,
            ready_record_processors=ready_record_processors,
            degraded_workers=max(0, self.workers_per_pod - ready_workers),
            degraded_record_processors=max(
                0, self.record_processors_per_pod - ready_record_processors
            ),
            pod_state=pod_state,
            admission_state=("dispatchable" if ready_workers >= 1 else "admitting"),
        )
        await self.publish(summary)
        await self.publish(pod_summary)

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
    async def _stop_worker_group_manager(self) -> None:
        """Stop group-local infrastructure, then upload raw records to controller."""
        self._stopping = True
        # In the fake in-process component-integration mode this WGM never
        # owned group-local peers, so skip shutdown coordination that would
        # otherwise wait forever for peers that never registered.
        if os.environ.get("AIPERF_FAKE_IN_PROCESS_MODE") == "1":
            await self._proxy_manager.stop()
            return
        await self._shutdown_local_peers()
        await self._wait_for_record_processor_shutdowns()
        if self._local_subprocess_manager is not None:
            await self._local_subprocess_manager.stop_all()
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
