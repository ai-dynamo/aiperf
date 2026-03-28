# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native msgspec structs for pod-local lifecycle traffic.

This channel is owned by WorkerPodManager and is only used for pod-scoped
worker/record-processor lifecycle coordination in Kubernetes mode.
"""

from typing import TypeAlias

from msgspec import Struct


class PodPeerHello(Struct, frozen=True, kw_only=True, tag_field="t", tag="hello"):
    """Initial registration of a pod-local peer with WorkerPodManager."""

    service_id: str
    service_type: str
    pod_index: str | None = None


class PodPeerAck(Struct, frozen=True, kw_only=True, tag_field="t", tag="ack"):
    """Acknowledgement of pod-local peer registration."""

    service_id: str


class PodPeerShutdown(Struct, frozen=True, kw_only=True, tag_field="t", tag="bye"):
    """Graceful shutdown notification from a pod-local peer."""

    service_id: str
    service_type: str


class PodWorkerHealth(Struct, frozen=True, kw_only=True, tag_field="t", tag="wh"):
    """Worker health snapshot delivered directly to WorkerPodManager."""

    service_id: str
    pid: int | None = None
    create_time: float
    uptime: float
    cpu_usage: float
    memory_usage: int
    pss_memory: int | None = None
    io_counters: tuple[int, int, int, int, int, int] | None = None
    cpu_times: tuple[float, float, float] | None = None
    num_ctx_switches: tuple[int, int] | None = None
    num_threads: int | None = None
    task_total: int
    task_failed: int
    task_completed: int


class PodWorkerStartupState(Struct, frozen=True, kw_only=True, tag_field="t", tag="ws"):
    """Worker startup lifecycle transition delivered to WorkerPodManager."""

    service_id: str
    startup_state: str
    request_ns: int


class PodDatasetReady(Struct, frozen=True, kw_only=True, tag_field="t", tag="dataset"):
    """Pod-local dataset availability notification from WorkerPodManager."""

    service_id: str
    data_file_path: str
    index_file_path: str
    conversation_count: int
    total_size_bytes: int
    pod_index: str | None = None
    success: bool = True
    error_message: str | None = None


PeerToPodManagerMessage: TypeAlias = (
    PodPeerHello | PodPeerShutdown | PodWorkerHealth | PodWorkerStartupState
)

PodManagerToPeerMessage: TypeAlias = PodPeerAck | PodDatasetReady
