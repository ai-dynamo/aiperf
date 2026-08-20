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
from pathlib import Path

import aiohttp

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.control_structs import Command, Registration
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    ExportLevel,
    MessageType,
    WorkerStartupState,
)
from aiperf.common.environment import Environment
from aiperf.common.error_queue import ErrorCollector, cleanup_global_error_queue
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
    DatasetDownloadedNotification,
    FinalizeArtifactsCommand,
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
    GroupTokenizerReady,
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
    build_worker_group_stats,
    command_record_processor_peers_strict,
    configure_local_peers,
    exact_record_processor_peer_ids,
    notify_registered_workers_of_dataset,
    run_dataset_download,
    shutdown_local_peers,
    wait_for_exact_record_processor_shutdowns,
    wait_for_expected_peers,
    wait_for_record_processor_shutdowns,
    worker_health_message_from_struct,
)
from aiperf.workers.worker_pod_tokenizer_download import download_tokenizer
from aiperf.workers.worker_pod_upload import upload_raw_records


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
        self._proxy_manager = ProxyManager(
            run=self.run,
            enable_event_bus=False,
            enable_dataset_manager=False,
            enable_raw_inference=True,
        )
        self._local_subprocess_manager: SubprocessManager | None = None
        self._error_collector: ErrorCollector | None = None
        if self._runtime_registration is not None:
            # Worker / RecordProcessor containers are separate processes: their
            # accumulated failures reach this manager only over the backchannel.
            self._error_collector = ErrorCollector(
                logger=self, exit_errors=self._exit_errors
            )
            self._local_subprocess_manager = SubprocessManager(
                run=self.run,
                logger=self,
                error_queue=self._error_collector.error_queue,
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
        self._latest_tokenizer_ready: GroupTokenizerReady | None = None
        self._stopping = False
        self._artifact_finalization_lock = asyncio.Lock()
        self._artifacts_finalized = False
        self._artifact_finalization_failed = False
        self._local_children_stopped = False
        self._errors_drained = False

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
        """Fetch tokenizer bundles from the operator API and notify peers.

        Mirrors the dataset-download path: each unique tokenizer (resolved from
        ``cfg.tokenizer.name`` or, when absent, the model list) is pulled as a
        tar+zstd bundle into ``MMAP_BASE_PATH/aiperf_tokenizers/{benchmark_id}``,
        then a ``GroupTokenizerReady`` is fanned out to registered worker peers.
        On failure the same struct is published with ``success=False`` and the
        exception is re-raised so the WGM lifecycle fails the pod.
        """
        self.info("Tokenizer prefetch task starting")
        api_base_full = self.run.cfg.runtime.dataset_api_base_url
        if not api_base_full:
            # No operator API available (in-process / component-integration mode,
            # or any non-k8s run). There is nothing to prefetch from — workers load
            # their tokenizers directly — so emit an empty ready (the same signal
            # used for the no-tokenizers case) instead of failing the WGM lifecycle.
            # In production k8s the operator always templates this URL, so this
            # branch never fires there.
            self.info(
                "No dataset_api_base_url; skipping tokenizer prefetch "
                "(workers load tokenizers directly)"
            )
            await self._publish_group_message(
                GroupTokenizerReady(service_id=self.service_id, bundles={})
            )
            return
        # ``runtime.dataset_api_base_url`` ends in ``/api/dataset``; strip
        # that suffix so ``download_tokenizer`` can append
        # ``/api/tokenizer/{name}/bundle`` to the same host:port without
        # producing a ``/api/dataset/api/tokenizer/...`` URL.
        api_base = api_base_full.rsplit("/api/dataset", 1)[0]
        names = self._unique_tokenizer_names()
        self.info(f"Tokenizers to fetch: {names}")
        if not names:
            # Nothing to fetch (server token counting, tiktoken-only, etc.).
            # Still emit ready so downstream waiters do not hang.
            await self._publish_group_message(
                GroupTokenizerReady(service_id=self.service_id, bundles={})
            )
            return
        dest_root = self._tokenizer_dest_root()
        dest_root.mkdir(parents=True, exist_ok=True)
        # Tokenizer downloads kick off at WGM startup before the controller
        # pod's api container has finished booting and warming the shared HF
        # cache. Use a generous retry budget so the natural api-startup
        # window doesn't fail the pull.
        tokenizer_max_retries = max(20, Environment.DATASET.DOWNLOAD_MAX_RETRIES)
        try:
            results = await asyncio.gather(
                *(
                    download_tokenizer(
                        api_base_url=api_base,
                        name=name,
                        dest_root=dest_root,
                        max_retries=tokenizer_max_retries,
                        logger=self,
                    )
                    for name in names
                )
            )
        except Exception as exc:
            self.exception(f"Tokenizer prefetch failed: {exc!r}")
            await self._publish_group_message(
                GroupTokenizerReady(
                    service_id=self.service_id,
                    bundles={},
                    success=False,
                    error_message=str(exc),
                )
            )
            raise
        bundles = {name: str(path) for name, path in zip(names, results, strict=True)}
        self.info(f"Tokenizer prefetch complete; bundles: {bundles}")
        await self._publish_group_message(
            GroupTokenizerReady(service_id=self.service_id, bundles=bundles)
        )

    def _unique_tokenizer_names(self) -> list[str]:
        """Return the unique tokenizer names this pod must serve.

        Mirrors ``validate_tokenizer_early`` resolution: explicit
        ``cfg.tokenizer.name`` wins; otherwise fall back to the configured
        model names. Returns an empty list when neither is set.

        Excludes the local-only tokenizer names (``builtin`` and the
        tiktoken encodings) — those are constructed in-process by every
        consumer via the special-case in ``Tokenizer.from_pretrained`` and
        do not need a controller-side bundle.
        """
        from aiperf.common.tokenizer import (
            BUILTIN_TOKENIZER_NAME,
            TIKTOKEN_ENCODING_NAMES,
        )

        seen: dict[str, None] = {}
        cfg = self.run.cfg
        tokenizer_cfg = getattr(cfg, "tokenizer", None)
        if tokenizer_cfg is not None and getattr(tokenizer_cfg, "name", None):
            seen.setdefault(tokenizer_cfg.name, None)
        else:
            for model_name in cfg.get_model_names():
                seen.setdefault(model_name, None)
        return [
            n
            for n in seen
            if n != BUILTIN_TOKENIZER_NAME and n not in TIKTOKEN_ENCODING_NAMES
        ]

    def _tokenizer_dest_root(self) -> Path:
        """Resolve the on-disk root for tokenizer bundles for this benchmark."""
        base = Environment.DATASET.MMAP_BASE_PATH or Path(tempfile.gettempdir())
        return base / f"aiperf_tokenizers/{self.run.benchmark_id}"

    async def _publish_group_message(self, message: GroupTokenizerReady) -> None:
        """Fan out a tokenizer-ready struct to RecordProcessor sibling peers.

        Only ``ServiceType.RECORD_PROCESSOR`` consumes the bundle paths —
        in-process workers never tokenize. The latest message is cached on
        ``self._latest_tokenizer_ready`` so that any RP that registers later
        (after this call returned) gets a replay from
        ``_on_pod_lifecycle_message`` when it sends ``GroupPeerHello``.
        Without that replay, the publish-before-peer-hello race silently
        drops the ready signal (ZMQ ROUTER does not queue for unknown
        identities) and the RP hangs forever on ``_tokenizer_ready.wait()``.
        """
        self._latest_tokenizer_ready = message
        target_type = str(ServiceType.RECORD_PROCESSOR)
        peer_identities = [
            self._pod_peer_identities[sid]
            for sid, stype in self._pod_peer_types.items()
            if stype == target_type and sid in self._pod_peer_identities
        ]
        if not peer_identities:
            return
        await asyncio.gather(
            *(
                self.pod_lifecycle_router.send_to(identity, message)
                for identity in peer_identities
            )
        )

    async def _upload_raw_records(self) -> None:
        await upload_raw_records(self.run, self)

    async def _wait_for_record_processor_shutdowns(self) -> None:
        await wait_for_record_processor_shutdowns(
            record_processors_per_pod=self.record_processors_per_pod,
            shutdown_set=self._record_processors_shutdown,
            logger=self,
        )

    async def _stop_local_children_and_drain_errors(self) -> None:
        """Stop manager-owned children and surface their failures once."""
        if (
            self._local_subprocess_manager is not None
            and not self._local_children_stopped
        ):
            await self._local_subprocess_manager.stop_all()
            self._local_children_stopped = True
        if self._error_collector is not None and not self._errors_drained:
            self._error_collector.drain_into()
            await cleanup_global_error_queue()
            self._errors_drained = True

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
            default_context_mode=(
                self._dataset_metadata.default_context_mode
                if self._dataset_metadata is not None
                else None
            ),
        )
        # The group-local router only reaches peers that have said hello on it.
        # Broadcast the same fact on the message bus, which every sibling
        # worker container is already subscribed to, so a worker learns the
        # pod-local files exist without depending on peer registration.
        await self.publish(
            DatasetDownloadedNotification(
                service_id=self.service_id,
                client_metadata=client_metadata,
                pod_index=self._pod_index,
                success=success,
                error_message=error_message,
            )
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
                    # Replay the latest tokenizer-ready message to recover from
                    # the publish-before-hello race. send_to is fire-and-forget
                    # on a known identity so this is safe to await inline.
                    if self._latest_tokenizer_ready is not None:
                        await self.pod_lifecycle_router.send_to(
                            identity, self._latest_tokenizer_ready
                        )
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
        try:
            await asyncio.wait_for(
                self._dataset_download_event.wait(),
                timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
            )
        except TimeoutError as e:
            raise RuntimeError(
                f"Dataset download did not complete within "
                f"{Environment.DATASET.CONFIGURATION_TIMEOUT}s; aborting "
                f"profile configuration for pod {self.service_id}. The dataset "
                f"download likely failed (see prior _run_dataset_download logs)."
            ) from e
        failures = await configure_local_peers(
            router=self.pod_lifecycle_router,
            sender_service_id=self.service_id,
            peer_identities=self._pod_peer_identities,
        )
        # Publish before raising: the controller's view of this pod is more
        # useful when configure went partially wrong, not less.
        await self._publish_worker_summary()
        if failures:
            detail = ", ".join(f"{sid}: {err!r}" for sid, err in failures)
            raise RuntimeError(
                f"PROFILE_CONFIGURE failed for {len(failures)} group-local "
                f"peer(s) of pod {self.service_id} -- {detail}"
            )

    @background_task(
        immediate=False, interval=Environment.WORKER.STATUS_SUMMARY_INTERVAL
    )
    async def _worker_summary_loop(self) -> None:
        """Generate a summary of the worker status."""
        await self._publish_worker_summary()

    async def _publish_worker_summary(self) -> None:
        """Publish worker-group, worker-centric, and pod-centric state snapshots."""
        sid, infos = self.service_id, self.worker_health
        summary = build_worker_status_summary(service_id=sid, worker_infos=infos)
        group_stats = build_worker_group_stats(
            service_id=sid, declared_workers=self.workers_per_pod, worker_infos=infos
        )
        pod_summary = build_pod_summary(
            service_id=sid,
            pod_index=self._pod_index,
            benchmark_generation=self._benchmark_generation,
            dataset_generation=self._dataset_generation,
            workers_per_pod=self.workers_per_pod,
            record_processors_per_pod=self.record_processors_per_pod,
            worker_startup_states=summary.worker_startup_states,
            peer_types=self._pod_peer_types,
        )
        for message in (group_stats, summary, pod_summary):
            await self.publish(message)

    @on_command(CommandType.REPORT_WORKER_STATUS_SUMMARY)
    async def _on_report_worker_status_summary(self, message: Command) -> None:
        """Publish an immediate worker status summary on controller request."""
        await self._publish_worker_summary()

    @on_command(CommandType.FINALIZE_ARTIFACTS)
    async def _on_finalize_artifacts(
        self,
        message: FinalizeArtifactsCommand,  # noqa: ARG002
    ) -> None:
        """Acknowledge only after every local RAW shard is durable upstream."""
        await self._finalize_raw_artifacts()

    async def _finalize_raw_artifacts(self) -> None:
        """Flush exact record processors, stop them, and upload final shards."""
        async with self._artifact_finalization_lock:
            if self._artifacts_finalized:
                return
            if self._artifact_finalization_failed:
                raise RuntimeError(
                    f"RAW artifact finalization previously failed for {self.service_id}"
                )

            try:
                record_processor_ids = exact_record_processor_peer_ids(
                    peer_identities=self._pod_peer_identities,
                    peer_types=self._pod_peer_types,
                    expected_count=self.record_processors_per_pod,
                )
                await command_record_processor_peers_strict(
                    router=self.pod_lifecycle_router,
                    sender_service_id=self.service_id,
                    peer_identities=self._pod_peer_identities,
                    peer_ids=record_processor_ids,
                    command=CommandType.FINALIZE_ARTIFACTS,
                )
                await command_record_processor_peers_strict(
                    router=self.pod_lifecycle_router,
                    sender_service_id=self.service_id,
                    peer_identities=self._pod_peer_identities,
                    peer_ids=record_processor_ids,
                    command=CommandType.SHUTDOWN,
                )
                await wait_for_exact_record_processor_shutdowns(
                    expected_service_ids=set(record_processor_ids),
                    shutdown_set=self._record_processors_shutdown,
                )
                await self._stop_local_children_and_drain_errors()
                if self._exit_errors:
                    raise RuntimeError(
                        f"Local child failures prevent RAW artifact publication "
                        f"for {self.service_id}: {len(self._exit_errors)} error(s)"
                    )
                await self._proxy_manager.stop()
                await self._upload_raw_records()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._artifact_finalization_failed = True
                raise
            self._artifacts_finalized = True

    @on_message(MessageType.WORKER_HEALTH)
    async def _on_worker_health(self, message: WorkerHealthMessage) -> None:
        """Track health for workers in THIS pod only.

        Same cluster-wide broadcast hazard as ``_on_worker_startup_state``
        below. Without the filter every WorkerGroupManager adopts every worker
        in the cluster, so ``build_worker_group_stats`` sums task_stats and
        memory_usage across the whole cluster for each pod (N-fold inflation
        for N pods), and ``worst_status`` lets a single errored worker anywhere
        mark every group ERROR.

        Messages without a ``pod_index`` predate the field or come from
        non-Kubernetes runs, where one group owns every worker anyway -- accept
        those so local and component-integration modes are unaffected.
        """
        if (
            message.pod_index is not None
            and self._pod_index is not None
            and message.pod_index != self._pod_index
        ):
            return
        info = self._get_or_create_worker_info(message.service_id)
        update_worker_status(info, message, warning=self.warning)

    @on_message(MessageType.WORKER_STARTUP_STATE)
    async def _on_worker_startup_state(
        self, message: WorkerStartupStateMessage
    ) -> None:
        """Track startup state for workers in THIS pod only.

        This is a cluster-wide broadcast topic. Without the pod filter every
        WorkerGroupManager adopts every worker in the cluster, so each pod
        reports the full cluster roster and the controller's aggregate
        over-counts ``ready``/``dispatchable``/``router_connected`` by the
        number of worker pods (observed: ready=8 against a declared total of
        4 on a 2-pod run).

        Messages without a ``pod_index`` predate the field or come from
        non-Kubernetes runs, where a single group owns every worker anyway --
        accept those so local and component-integration modes are unaffected.
        """
        if (
            message.pod_index is not None
            and self._pod_index is not None
            and message.pod_index != self._pod_index
        ):
            return
        info = self._get_or_create_worker_info(message.service_id)
        info.startup_state = message.startup_state
        info.startup_state_updated_ns = message.request_ns
        await self._publish_worker_summary()

    @on_stop
    async def _stop_worker_group_manager(self) -> None:
        """Stop group-local infrastructure without publishing partial artifacts."""
        self._stopping = True
        # Fake in-process mode never owns group-local peers; skip coordination.
        if os.environ.get("AIPERF_FAKE_IN_PROCESS_MODE") == "1":
            await self._proxy_manager.stop()
            return

        if self._artifacts_finalized or self._artifact_finalization_failed:
            await self._stop_local_children_and_drain_errors()
            await self._proxy_manager.stop()
            return

        # ABORT (not SHUTDOWN) when this WGM failed so peers exit non-zero
        # and kubelet restarts the whole pod — otherwise pod stalls at 1/13 Ready.
        if (
            not self._exit_errors
            and self.run.cfg.artifacts.export_level == ExportLevel.RAW
        ):
            await self._finalize_raw_artifacts()
            return

        command = CommandType.ABORT if self._exit_errors else CommandType.SHUTDOWN
        await shutdown_local_peers(
            router=self.pod_lifecycle_router,
            sender_service_id=self.service_id,
            peer_identities=self._pod_peer_identities,
            command=command,
            logger=self,
        )
        await self._wait_for_record_processor_shutdowns()
        await self._stop_local_children_and_drain_errors()
        await self._proxy_manager.stop()
