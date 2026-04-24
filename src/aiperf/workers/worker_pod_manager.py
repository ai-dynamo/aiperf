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
import uuid
from pathlib import Path

import aiohttp

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
from aiperf.common.models import MemoryMapClientMetadata
from aiperf.common.pod_lifecycle_structs import (
    GroupDatasetStateQuery,
    GroupDatasetStateSnapshot,
    GroupPeerAck,
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
from aiperf.controller.proxy_manager import ProxyManager
from aiperf.plugin.enums import ServiceType
from aiperf.workers import worker_pod_dataset_download as _dataset_dl
from aiperf.workers.group_runtime import GroupRuntimeAdapter, GroupRuntimeRegistration
from aiperf.workers.worker_group_state import (
    WorkerStatusInfo,
    build_worker_status_summary,
    mark_stale_workers,
    update_worker_status,
)
from aiperf.workers.worker_pod_dataset_download import download_dataset
from aiperf.workers.worker_pod_helpers import (
    build_pod_dataset_snapshot,
    build_pod_summary,
    configure_local_peers,
    notify_registered_workers_of_dataset,
    prefetch_tokenizers,
    run_dataset_download,
    shutdown_local_peers,
    wait_for_expected_peers,
    wait_for_record_processor_shutdowns,
    worker_health_message_from_struct,
)
from aiperf.workers.worker_pod_upload import (
    upload_raw_records,
    wait_for_raw_record_files,
)


class WorkerGroupManagerBase(BaseComponentService):
    """Coordinates shared worker-pod infrastructure for sibling service containers.

    The main process in a worker pod container; downloads the dataset once,
    runs the group-local raw-inference proxy, owns the pod's controller-facing
    lifecycle connection, configures/shuts down group-local workers and record
    processors, republishes dataset notifications for late workers, and uploads
    raw record files after record-processor containers flush them.
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
        super().__init__(run=run, service_id=service_id, **kwargs)
        self._resolve_pod_capacity()
        self._init_pod_state()
        self.pod_lifecycle_router: StreamingRouterClientProtocol = (
            self.comms.create_streaming_router_client(
                address=CommAddress.GROUP_LIFECYCLE,
                bind=True,
                decode_type=PeerToGroupManagerMessage,
            )
        )
        self.pod_lifecycle_router.register_receiver(self._on_pod_lifecycle_message)
        self._proxy_manager = ProxyManager(run=self.run, enable_raw_inference=True)
        self._local_subprocess_manager: SubprocessManager | None = None
        if self._runtime_registration is not None:
            self._local_subprocess_manager = SubprocessManager(
                run=self.run, logger=self
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

    def _resolve_pod_capacity(self) -> None:
        """Set workers_per_pod / record_processors_per_pod from registration or config."""
        cfg = self.run.cfg
        registration = self._runtime_registration
        if registration is not None:
            self.workers_per_pod = registration.declared_workers
            self.record_processors_per_pod = registration.declared_record_processors
            return

        self.workers_per_pod = (
            cfg.runtime.workers_per_pod or Environment.WORKER.DEFAULT_WORKERS_PER_POD
        )
        # Default: 1 RP for every 4 workers, minimum 1.
        # The Kubernetes path should set record_processors_per_pod explicitly.
        if cfg.runtime.record_processors_per_pod is not None:
            self.record_processors_per_pod = cfg.runtime.record_processors_per_pod
        else:
            self.record_processors_per_pod = max(
                1, self.workers_per_pod // Environment.RECORD.PROCESSOR_SCALE_FACTOR
            )

    def _init_pod_state(self) -> None:
        """Initialize per-pod worker/peer and dataset-download bookkeeping."""
        self.worker_health: dict[str, WorkerStatusInfo] = {}
        self._pod_peer_identities: dict[str, str] = {}
        self._pod_peer_types: dict[str, str] = {}
        self._record_processors_shutdown: set[str] = set()
        self._configure_started = False
        self._dataset_downloaded = False
        self._dataset_download_event = asyncio.Event()
        self._dataset_client_metadata: MemoryMapClientMetadata | None = None
        self._dataset_metadata = None
        self._benchmark_generation: str | None = None
        self._dataset_generation: str | None = None
        self._dataset_download_task: asyncio.Task[None] | None = None
        self._tokenizer_prefetch_task: asyncio.Task[None] | None = None
        self._stopping = False

    def _make_registration(self) -> Registration:
        """Build a Registration extending the base with pod capacity info."""
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
        """Initialize and start the local raw inference proxy."""
        await self._proxy_manager.initialize_and_start()

    @on_start
    async def _start_worker_group_manager(self) -> None:
        """Start the WorkerGroupManager."""
        self.info("WorkerGroupManager starting...")
        # Each K8s pod is a separate machine — worker pods must cache
        # tokenizers independently. Kick off opportunistically so startup is
        # not blocked.
        if (
            self._tokenizer_prefetch_task is None
            or self._tokenizer_prefetch_task.done()
        ):
            self._tokenizer_prefetch_task = self.execute_async(
                self._prefetch_tokenizers()
            )
        if self._local_subprocess_manager is not None:
            await self._local_subprocess_manager.spawn_services(
                ServiceType.WORKER, self.workers_per_pod
            )
            await self._local_subprocess_manager.spawn_services(
                ServiceType.RECORD_PROCESSOR, self.record_processors_per_pod
            )
        self.debug("Waiting for dataset configuration...")

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(
        self, message: DatasetConfiguredNotification
    ) -> None:
        """Handle dataset configuration notification.

        Downloads the dataset from control-plane so workers can mmap it, then
        notifies sibling workers directly over the pod lifecycle channel.
        """
        self._dataset_metadata = message.metadata
        self._benchmark_generation = message.benchmark_generation
        self._dataset_generation = message.dataset_generation
        await self._publish_worker_summary()

        if self._dataset_downloaded:
            self.debug("Dataset already downloaded; late workers query current state")
            return

        if self._dataset_download_task is not None:
            self.debug("Dataset download in progress, waiting for existing task")
            await self._dataset_download_task
            return

        # Local fast-path: runtime adapter or fake in-process mode — files
        # already on the local filesystem, skip HTTP download.
        fake_mode = os.environ.get("AIPERF_FAKE_IN_PROCESS_MODE") == "1"
        if self._runtime_registration is not None or fake_mode:
            self.info("Received dataset configuration, attaching local dataset state")
            self._dataset_client_metadata = message.client_metadata
            self._dataset_downloaded = True
            self._dataset_download_event.set()
            await self._notify_registered_workers_of_dataset(
                client_metadata=message.client_metadata, success=True
            )
            await self._publish_worker_summary()
            return

        self.info("Received dataset configuration, downloading dataset...")
        self._dataset_download_task = self.execute_async(
            self._run_dataset_download(message)
        )
        try:
            await self._dataset_download_task
        finally:
            self._dataset_download_task = None

    async def _run_dataset_download(
        self, message: DatasetConfiguredNotification
    ) -> None:
        """Download the dataset and update local dataset state on success."""
        client_metadata = await run_dataset_download(
            run=self.run,
            message=message,
            download_fn=self._download_dataset,
            notify_fn=self._notify_registered_workers_of_dataset,
            publish_summary_fn=self._publish_worker_summary,
            logger=self,
        )
        # Mark downloaded only after successful direct notification so a retry
        # can re-attempt if delivery fails
        self._dataset_client_metadata = client_metadata
        self._dataset_downloaded = True
        self._dataset_download_event.set()
        await self._publish_worker_summary()

    # --- Thin wrappers: tests patch; subclasses may override. ---
    async def _download_dataset(self) -> tuple[Path, Path]:
        return await download_dataset(self.run, self, download_file=self._download_file)

    async def _download_file(
        self, session: aiohttp.ClientSession, url: str, dest_path: Path
    ) -> None:
        await _dataset_dl._download_file(session, url, dest_path, self)

    async def _prefetch_tokenizers(self) -> None:
        await prefetch_tokenizers(self.run, self)

    async def _wait_for_raw_record_files(self) -> None:
        await wait_for_raw_record_files(self.run, self.record_processors_per_pod, self)

    async def _upload_raw_records(self) -> None:
        await upload_raw_records(self.run, self)

    async def _wait_for_record_processor_shutdowns(self) -> None:
        await wait_for_record_processor_shutdowns(
            record_processors_per_pod=self.record_processors_per_pod,
            shutdown_set=self._record_processors_shutdown,
            logger=self,
        )

    async def _notify_registered_workers_of_dataset(
        self,
        *,
        client_metadata: MemoryMapClientMetadata,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        await notify_registered_workers_of_dataset(
            router=self.pod_lifecycle_router,
            service_id=self.service_id,
            pod_index=self._pod_index,
            peer_identities=self._pod_peer_identities,
            peer_types=self._pod_peer_types,
            client_metadata=client_metadata,
            success=success,
            error_message=error_message,
        )

    def _build_pod_dataset_snapshot(self, rid: str) -> GroupDatasetStateSnapshot:
        return build_pod_dataset_snapshot(
            rid=rid,
            service_id=self.service_id,
            pod_index=self._pod_index,
            benchmark_generation=self._benchmark_generation,
            dataset_generation=self._dataset_generation,
            dataset_metadata=self._dataset_metadata,
            client_metadata=self._dataset_client_metadata,
            dataset_downloaded=self._dataset_downloaded,
        )

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
                update_worker_status(
                    info,
                    worker_health_message_from_struct(message),
                    warning=self.warning,
                )
                return None
            case GroupWorkerStartupState():
                info = self._get_or_create_worker_info(message.service_id)
                info.startup_state = WorkerStartupState(message.startup_state)
                info.startup_state_updated_ns = message.request_ns
                await self._publish_worker_summary()
                return None
            case GroupDatasetStateQuery():
                return build_pod_dataset_snapshot(
                    rid=message.rid,
                    service_id=self.service_id,
                    pod_index=self._pod_index,
                    benchmark_generation=self._benchmark_generation,
                    dataset_generation=self._dataset_generation,
                    dataset_metadata=self._dataset_metadata,
                    client_metadata=self._dataset_client_metadata,
                    dataset_downloaded=self._dataset_downloaded,
                )
            case GroupPeerCommandAck():
                return message

    def _get_or_create_worker_info(self, worker_id: str) -> WorkerStatusInfo:
        info = self.worker_health.get(worker_id)
        if info is None:
            info = WorkerStatusInfo(worker_id=worker_id)
            self.worker_health[worker_id] = info
        return info

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
        # In-process fake mode: no group-local peers to wait for or fan out to.
        if self._runtime_registration is None:
            await self._publish_worker_summary()
            return
        await wait_for_expected_peers(
            workers_per_pod=self.workers_per_pod,
            record_processors_per_pod=self.record_processors_per_pod,
            peer_types=self._pod_peer_types,
        )
        await self._dataset_download_event.wait()
        await configure_local_peers(
            router=self.pod_lifecycle_router,
            sender_service_id=self.service_id,
            peer_identities=self._pod_peer_identities,
        )
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
        pod_summary = build_pod_summary(
            service_id=self.service_id,
            pod_index=self._pod_index,
            benchmark_generation=self._benchmark_generation,
            dataset_generation=self._dataset_generation,
            workers_per_pod=self.workers_per_pod,
            record_processors_per_pod=self.record_processors_per_pod,
            worker_startup_states=summary.worker_startup_states,
            peer_types=self._pod_peer_types,
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
        update_worker_status(info, message, warning=self.warning)

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
        # Fake in-process mode never owns group-local peers; skip coordination.
        if os.environ.get("AIPERF_FAKE_IN_PROCESS_MODE") == "1":
            await self._proxy_manager.stop()
            return
        # ABORT (not SHUTDOWN) when this WGM failed so peers exit non-zero
        # and kubelet restarts the whole pod — otherwise pod stalls at 1/13 Ready.
        command = CommandType.ABORT if self._exit_errors else CommandType.SHUTDOWN
        await shutdown_local_peers(
            router=self.pod_lifecycle_router,
            sender_service_id=self.service_id,
            peer_identities=self._pod_peer_identities,
            command=command,
            logger=self,
        )
        await self._wait_for_record_processor_shutdowns()
        if self._local_subprocess_manager is not None:
            await self._local_subprocess_manager.stop_all()
        await self._wait_for_raw_record_files()
        await self._proxy_manager.stop()
        await self._upload_raw_records()
