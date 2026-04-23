# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helper functions for the WorkerGroupManager.

Extracted from ``worker_pod_manager`` to keep that module within the
ergonomics file-size limit. These helpers build pod-lifecycle structs, convert
group-local health messages, fan out peer commands, and publish status
summaries. They take only the state they need and have no side effects other
than the explicit router/publish calls they receive.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Protocol

from aiperf.common.enums import CommandType, WorkerStartupState
from aiperf.common.environment import Environment
from aiperf.common.messages import (
    DatasetConfiguredNotification,
    WorkerHealthMessage,
)
from aiperf.common.messages.worker_messages import WorkerPodStateMessage
from aiperf.common.models import (
    MemoryMapClientMetadata,
    ProcessHealth,
    WorkerTaskStats,
)
from aiperf.common.pod_lifecycle_structs import (
    GroupDatasetReady,
    GroupDatasetStateSnapshot,
    GroupPeerCommand,
    GroupPeerCommandAck,
    GroupWorkerHealth,
)
from aiperf.common.protocols import StreamingRouterClientProtocol
from aiperf.plugin.enums import ServiceType
from aiperf.workers.worker_pod_dataset_download import (
    placeholder_local_paths,
)

if TYPE_CHECKING:
    from aiperf.common.models.dataset_models import DatasetMetadata


class _PodLogger(Protocol):
    """Structural protocol matching logging methods used on BaseComponentService."""

    def info(self, msg: str) -> None: ...
    def debug(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def exception(self, msg: str) -> None: ...


def worker_health_message_from_struct(
    message: GroupWorkerHealth,
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


def build_pod_dataset_ready(
    *,
    service_id: str,
    pod_index: str | None,
    client_metadata: MemoryMapClientMetadata,
    success: bool,
    error_message: str | None = None,
) -> GroupDatasetReady:
    """Build the group-local dataset-ready notification."""
    return GroupDatasetReady(
        service_id=service_id,
        data_file_path=str(client_metadata.data_file_path),
        index_file_path=str(client_metadata.index_file_path),
        conversation_count=client_metadata.conversation_count,
        total_size_bytes=client_metadata.total_size_bytes,
        pod_index=pod_index,
        success=success,
        error_message=error_message,
    )


def build_pod_dataset_snapshot(
    *,
    rid: str,
    service_id: str,
    pod_index: str | None,
    benchmark_generation: str | None,
    dataset_generation: str | None,
    dataset_metadata: DatasetMetadata | None,
    client_metadata: MemoryMapClientMetadata | None,
    dataset_downloaded: bool,
) -> GroupDatasetStateSnapshot:
    """Build a queryable current-state dataset snapshot for sibling workers."""
    return GroupDatasetStateSnapshot(
        rid=rid,
        service_id=service_id,
        benchmark_generation=benchmark_generation,
        dataset_generation=dataset_generation,
        default_context_mode=(
            dataset_metadata.default_context_mode
            if dataset_metadata is not None
            else None
        ),
        data_file_path=str(client_metadata.data_file_path)
        if client_metadata is not None
        else None,
        index_file_path=(
            str(client_metadata.index_file_path)
            if client_metadata is not None
            else None
        ),
        conversation_count=client_metadata.conversation_count
        if client_metadata is not None
        else 0,
        total_size_bytes=client_metadata.total_size_bytes
        if client_metadata is not None
        else 0,
        pod_index=pod_index,
        ready=dataset_downloaded and client_metadata is not None,
    )


async def notify_registered_workers_of_dataset(
    *,
    router: StreamingRouterClientProtocol,
    service_id: str,
    pod_index: str | None,
    peer_identities: dict[str, str],
    peer_types: dict[str, str],
    client_metadata: MemoryMapClientMetadata,
    success: bool,
    error_message: str | None = None,
) -> None:
    """Push dataset availability directly to registered sibling workers."""
    worker_identities = [
        peer_identities[sid]
        for sid, stype in peer_types.items()
        if stype == str(ServiceType.WORKER) and sid in peer_identities
    ]
    if not worker_identities:
        return
    message = build_pod_dataset_ready(
        service_id=service_id,
        pod_index=pod_index,
        client_metadata=client_metadata,
        success=success,
        error_message=error_message,
    )
    await asyncio.gather(
        *(router.send_to(identity, message) for identity in worker_identities)
    )


def _expected_peer_counts(
    workers_per_pod: int, record_processors_per_pod: int
) -> dict[str, int]:
    return {
        str(ServiceType.WORKER): workers_per_pod,
        str(ServiceType.RECORD_PROCESSOR): record_processors_per_pod,
    }


def _registered_peer_counts(peer_types: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for service_type in peer_types.values():
        counts[service_type] = counts.get(service_type, 0) + 1
    return counts


async def wait_for_expected_peers(
    *,
    workers_per_pod: int,
    record_processors_per_pod: int,
    peer_types: dict[str, str],
) -> None:
    """Wait for the full group-local worker and record-processor set to register."""
    deadline = (
        asyncio.get_running_loop().time()
        + Environment.SERVICE.PROFILE_CONFIGURE_TIMEOUT
    )
    expected = _expected_peer_counts(workers_per_pod, record_processors_per_pod)
    while asyncio.get_running_loop().time() < deadline:
        counts = _registered_peer_counts(peer_types)
        if all(
            counts.get(service_type, 0) >= expected_count
            for service_type, expected_count in expected.items()
        ):
            return
        await asyncio.sleep(0.2)
    counts = _registered_peer_counts(peer_types)
    raise TimeoutError(
        "Timed out waiting for group-local peers to register: "
        f"expected={expected}, registered={counts}"
    )


async def _send_pod_command(
    *,
    router: StreamingRouterClientProtocol,
    service_id: str,
    sender_service_id: str,
    identity: str,
    command: CommandType,
) -> None:
    response = await router.request_to(
        identity,
        GroupPeerCommand(
            cid=uuid.uuid4().hex,
            service_id=sender_service_id,
            command=str(command),
        ),
        timeout=Environment.SERVICE.PROFILE_CONFIGURE_TIMEOUT,
    )
    if not isinstance(response, GroupPeerCommandAck):
        raise TypeError(
            f"Unexpected group-local response from {service_id}: {type(response).__name__}"
        )


async def configure_local_peers(
    *,
    router: StreamingRouterClientProtocol,
    sender_service_id: str,
    peer_identities: dict[str, str],
) -> None:
    """Fan out PROFILE_CONFIGURE to every registered group-local peer."""
    peer_ids = list(peer_identities)
    if not peer_ids:
        return
    await asyncio.gather(
        *(
            _send_pod_command(
                router=router,
                service_id=sid,
                sender_service_id=sender_service_id,
                identity=peer_identities[sid],
                command=CommandType.PROFILE_CONFIGURE,
            )
            for sid in peer_ids
        )
    )


async def shutdown_local_peers(
    *,
    router: StreamingRouterClientProtocol,
    sender_service_id: str,
    peer_identities: dict[str, str],
    command: CommandType,
    logger: _PodLogger,
) -> None:
    """Fan out a shutdown/abort command to every registered group-local peer.

    Failures are logged as warnings but do not propagate so peer loss does not
    stall the overall pod teardown sequence.
    """
    peer_ids = list(peer_identities)
    if not peer_ids:
        return
    results = await asyncio.gather(
        *(
            _send_pod_command(
                router=router,
                service_id=sid,
                sender_service_id=sender_service_id,
                identity=peer_identities[sid],
                command=command,
            )
            for sid in peer_ids
        ),
        return_exceptions=True,
    )
    for sid, result in zip(peer_ids, results, strict=False):
        if isinstance(result, Exception):
            logger.warning(
                f"Failed to send {command} to group-local peer {sid}: {result!r}"
            )


def build_pod_summary(
    *,
    service_id: str,
    pod_index: str | None,
    benchmark_generation: str | None,
    dataset_generation: str | None,
    workers_per_pod: int,
    record_processors_per_pod: int,
    worker_startup_states: dict[str, WorkerStartupState],
    peer_types: dict[str, str],
) -> WorkerPodStateMessage:
    """Build the pod-level status message from current worker + peer state."""
    ready_workers = sum(
        1
        for state in worker_startup_states.values()
        if state == WorkerStartupState.READY
    )
    router_connected_workers = sum(
        1
        for state in worker_startup_states.values()
        if state
        in {
            WorkerStartupState.ROUTER_PROBING,
            WorkerStartupState.WAITING_FOR_DATASET,
            WorkerStartupState.READY,
        }
    )
    ready_record_processors = sum(
        1
        for service_type in peer_types.values()
        if service_type == str(ServiceType.RECORD_PROCESSOR)
    )
    pod_state = (
        "ready" if ready_workers >= 1 and ready_record_processors >= 1 else "starting"
    )
    return WorkerPodStateMessage(
        service_id=service_id,
        pod_index=pod_index or "",
        benchmark_generation=benchmark_generation,
        dataset_generation=dataset_generation,
        declared_workers=workers_per_pod,
        declared_record_processors=record_processors_per_pod,
        router_connected_workers=router_connected_workers,
        dispatchable_workers=ready_workers,
        ready_workers=ready_workers,
        ready_record_processors=ready_record_processors,
        degraded_workers=max(0, workers_per_pod - ready_workers),
        degraded_record_processors=max(
            0, record_processors_per_pod - ready_record_processors
        ),
        pod_state=pod_state,
        admission_state=("dispatchable" if ready_workers >= 1 else "admitting"),
    )


async def prefetch_tokenizers(run, logger: _PodLogger) -> None:
    """Warm the shared HF tokenizer cache so sibling containers load from disk.

    Runs ``validate_tokenizer_early`` in a thread to avoid blocking the event
    loop. Skipped when ``use_server_token_count`` is True because worker pods
    only need the tokenizer for counting response tokens.
    """
    if run.cfg.endpoint.use_server_token_count:
        logger.debug("Tokenizer prefetch skipped (using server token counts)")
        return

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.tokenizer_validator import validate_tokenizer_early

    prefetch_logger = AIPerfLogger(f"{__name__}.tokenizer_prefetch")
    resolved = await asyncio.to_thread(
        validate_tokenizer_early, run.cfg, prefetch_logger
    )
    if resolved:
        run.resolved.tokenizer_names = resolved
        logger.info(f"Tokenizer cache warmed: {len(resolved)} model(s)")
    else:
        logger.debug("Tokenizer prefetch skipped (not required)")


async def wait_for_record_processor_shutdowns(
    *,
    record_processors_per_pod: int,
    shutdown_set: set[str],
    logger: _PodLogger,
) -> None:
    """Wait for sibling record processors to announce a clean local shutdown."""
    if record_processors_per_pod <= 0:
        return
    deadline = (
        asyncio.get_running_loop().time()
        + Environment.SERVICE.RAW_RECORD_UPLOAD_TIMEOUT
    )
    while asyncio.get_running_loop().time() < deadline:
        if len(shutdown_set) >= record_processors_per_pod:
            return
        await asyncio.sleep(0.2)
    logger.warning(
        "Timed out waiting for record processors to report local shutdown: "
        f"expected {record_processors_per_pod}, got {len(shutdown_set)}"
    )


async def run_dataset_download(
    *,
    run,
    message: DatasetConfiguredNotification,
    download_fn,
    notify_fn,
    publish_summary_fn,
    logger: _PodLogger,
) -> MemoryMapClientMetadata:
    """Coordinate the dataset download lifecycle and return client metadata.

    ``download_fn`` returns ``(data_path, index_path)`` so tests can patch the
    instance method. ``notify_fn`` must accept ``client_metadata`` / ``success``
    / optional ``error_message`` kwargs (matching the bound
    ``_notify_registered_workers_of_dataset``). On failure, notifies workers
    with placeholder paths, publishes the summary, and re-raises.
    """
    try:
        data_path, index_path = await download_fn()
        conversation_count = len(message.metadata.conversations)
        data_size = data_path.stat().st_size
        logger.info(
            f"Dataset download complete, notifying workers: "
            f"{conversation_count} conversations, {data_size} bytes"
        )
        client_metadata = MemoryMapClientMetadata(
            data_file_path=data_path,
            index_file_path=index_path,
            conversation_count=conversation_count,
            total_size_bytes=data_size,
        )
        await notify_fn(client_metadata=client_metadata, success=True)
        return client_metadata
    except Exception as e:
        logger.exception(f"Failed to download dataset: {e!r}")
        fail_data, fail_index = placeholder_local_paths(run)
        await notify_fn(
            client_metadata=MemoryMapClientMetadata(
                data_file_path=fail_data,
                index_file_path=fail_index,
                conversation_count=0,
                total_size_bytes=0,
            ),
            success=False,
            error_message=str(e),
        )
        await publish_summary_fn()
        raise
