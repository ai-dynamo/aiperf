# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native msgspec structs for group-local lifecycle traffic.

This channel is owned by WorkerGroupManager and carries group-scoped
worker/record-processor lifecycle coordination. Kubernetes uses the same wire
contract inside each worker pod, while local mode uses the same contract for
its worker group.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, TypeAlias

from msgspec import Struct

from aiperf.common.enums import ConversationContextMode

if TYPE_CHECKING:
    from aiperf.common.protocols import StreamingDealerClientProtocol


class GroupPeerHello(Struct, frozen=True, kw_only=True, tag_field="t", tag="hello"):
    """Initial registration of a group-local peer with WorkerGroupManager."""

    rid: str
    """Request id used to match the corresponding GroupPeerAck so the peer
    can retry delivery until the WGM confirms registration."""

    service_id: str
    """Unique service identifier for the registering child peer."""

    service_type: str
    """Child service type reported during group-local registration."""

    pod_index: str | None = None
    """Kubernetes pod index for the peer when the group maps to a worker pod."""


class GroupPeerAck(Struct, frozen=True, kw_only=True, tag_field="t", tag="ack"):
    """Acknowledgement of group-local peer registration."""

    rid: str
    """Echo of the GroupPeerHello.rid this ack responds to."""

    service_id: str
    """Service identifier of the peer whose registration was accepted."""


class GroupPeerShutdown(Struct, frozen=True, kw_only=True, tag_field="t", tag="bye"):
    """Graceful shutdown notification from a group-local peer."""

    service_id: str
    """Unique service identifier for the shutting-down child peer."""

    service_type: str
    """Child service type that is shutting down."""


class GroupWorkerHealth(Struct, frozen=True, kw_only=True, tag_field="t", tag="wh"):
    """Worker health snapshot delivered directly to WorkerGroupManager."""

    service_id: str
    """Unique service identifier for the worker reporting health."""

    pid: int | None = None
    """Process ID of the worker process, if available."""

    create_time: float
    """Worker process creation time in epoch seconds."""

    uptime: float
    """Worker uptime in seconds."""

    cpu_usage: float
    """Current CPU utilization percentage for the worker process."""

    memory_usage: int
    """Resident memory usage in bytes."""

    pss_memory: int | None = None
    """Proportional set size memory in bytes, when available."""

    io_counters: tuple[int, int, int, int, int, int] | None = None
    """Process IO counters snapshot, when available."""

    cpu_times: tuple[float, float, float] | None = None
    """Process CPU time counters snapshot, when available."""

    num_ctx_switches: tuple[int, int] | None = None
    """Voluntary and involuntary context switch counts, when available."""

    num_threads: int | None = None
    """Current number of process threads, when available."""

    task_total: int
    """Total number of tasks observed by the worker."""

    task_failed: int
    """Number of failed tasks observed by the worker."""

    task_completed: int
    """Number of completed tasks observed by the worker."""


class GroupWorkerStartupState(
    Struct, frozen=True, kw_only=True, tag_field="t", tag="ws"
):
    """Worker startup lifecycle transition delivered to WorkerGroupManager."""

    service_id: str
    """Unique service identifier for the worker reporting startup state."""

    startup_state: str
    """Current startup lifecycle state for the worker."""

    request_ns: int
    """Request timestamp associated with this startup transition."""


class GroupDatasetReady(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="t",
    tag="dataset",
):
    """Group-local dataset availability notification from WorkerGroupManager."""

    service_id: str
    """Service identifier for the group manager publishing dataset readiness."""

    data_file_path: str
    """Local path to the mmap-ready dataset data file."""

    index_file_path: str
    """Local path to the mmap-ready dataset index file."""

    conversation_count: int
    """Number of conversations contained in the dataset snapshot."""

    total_size_bytes: int
    """Total dataset size in bytes across local files."""

    pod_index: str | None = None
    """Kubernetes pod index for this dataset snapshot when applicable."""

    success: bool = True
    """Whether dataset acquisition completed successfully."""

    error_message: str | None = None
    """Dataset acquisition error message when success is false."""


class GroupDatasetStateQuery(
    Struct, frozen=True, kw_only=True, tag_field="t", tag="dq"
):
    """Request the current group-local dataset state from WorkerGroupManager."""

    rid: str
    """Request identifier used to correlate the snapshot response."""

    service_id: str
    """Service identifier of the child requesting dataset state."""


class GroupDatasetStateSnapshot(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="t",
    tag="ds",
):
    """Current group-local dataset snapshot returned by WorkerGroupManager."""

    rid: str
    """Request identifier copied from the triggering dataset-state query."""

    service_id: str
    """Service identifier for the group manager returning the snapshot."""

    benchmark_generation: str | None = None
    """Benchmark generation currently active for this group snapshot."""

    dataset_generation: str | None = None
    """Dataset generation currently active for this group snapshot."""

    default_context_mode: ConversationContextMode | None = None
    """Default conversation context mode for workers consuming this dataset."""

    data_file_path: str | None = None
    """Local path to the mmap-ready dataset data file, if available."""

    index_file_path: str | None = None
    """Local path to the mmap-ready dataset index file, if available."""

    conversation_count: int = 0
    """Number of conversations present in the current dataset snapshot."""

    total_size_bytes: int = 0
    """Total local dataset size in bytes for the current snapshot."""

    pod_index: str | None = None
    """Kubernetes pod index for this dataset snapshot when applicable."""

    ready: bool = False
    """Whether the group dataset is ready for child startup and dispatch gating."""

    error_message: str | None = None
    """Dataset acquisition error, if the current snapshot is not ready."""


class GroupPeerCommand(Struct, frozen=True, kw_only=True, tag_field="t", tag="cmd"):
    """Group-local lifecycle command sent from WorkerGroupManager to a child peer."""

    cid: str
    """Command identifier used to correlate acknowledgements."""

    service_id: str
    """Target child service identifier for this lifecycle command."""

    command: str
    """Lifecycle command name to execute group-locally."""


class GroupPeerCommandAck(
    Struct, frozen=True, kw_only=True, tag_field="t", tag="cmd_ack"
):
    """Acknowledgement for a group-local lifecycle command."""

    cid: str
    """Command identifier being acknowledged."""

    service_id: str
    """Child service identifier that processed the command."""


PeerToGroupManagerMessage: TypeAlias = (
    GroupPeerHello
    | GroupPeerShutdown
    | GroupWorkerHealth
    | GroupWorkerStartupState
    | GroupDatasetStateQuery
    | GroupPeerCommandAck
)

GroupManagerToPeerMessage: TypeAlias = (
    GroupPeerAck | GroupDatasetReady | GroupDatasetStateSnapshot | GroupPeerCommand
)


async def _send_group_peer_hello_with_retry(
    dealer_client: StreamingDealerClientProtocol,
    *,
    service_id: str,
    service_type: str,
    pod_index: str | None,
    logger: Any,
    attempt_timeout: float = 2.0,
    total_timeout: float = 120.0,
) -> None:
    """Send GroupPeerHello and retry until the WorkerGroupManager acks.

    In Kubernetes, sibling containers in a worker pod start concurrently,
    so a worker's first hello can be dropped if WGM's ROUTER isn't bound
    yet. Retry with a bounded deadline so the pod converges instead of
    hanging the controller's configure wait.
    """
    deadline = asyncio.get_running_loop().time() + total_timeout
    attempt = 0
    while True:
        attempt += 1
        hello = GroupPeerHello(
            rid=uuid.uuid4().hex,
            service_id=service_id,
            service_type=service_type,
            pod_index=pod_index,
        )
        try:
            ack = await dealer_client.request(hello, timeout=attempt_timeout)
        except asyncio.TimeoutError:
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"GroupPeerHello for {service_id} never acked after "
                    f"{attempt} attempt(s)"
                ) from None
            logger.debug(
                lambda a=attempt: f"GroupPeerHello attempt {a} timed out, retrying"
            )
            continue
        # Any non-TimeoutError reply is treated as "delivered"; the WGM's
        # ROUTER doesn't surface delivery-failed signals, so a successful
        # send+reply means the hello landed. Log unexpected payloads for
        # visibility but do not spin — spinning breaks fake-comm unit tests
        # that stub request() to return None.
        if isinstance(ack, GroupPeerAck) and ack.rid != hello.rid:
            logger.warning(
                f"GroupPeerAck rid mismatch for {service_id}: "
                f"expected {hello.rid}, got {ack.rid}"
            )
        return
