# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate worker-pod status and K8s topology types for the SystemController."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from aiperf.common.messages import WorkerPodStateMessage
from aiperf.common.models import AIPerfBaseModel


class AggregateWorkerStatus(AIPerfBaseModel):
    """Controller-authored aggregate worker-pod status snapshot."""

    ready: int = Field(default=0, description="Dispatch-ready worker count.")
    total: int = Field(default=0, description="Declared worker count.")
    dispatchable: int = Field(
        default=0,
        description="Workers eligible to receive credits.",
    )
    router_connected: int = Field(
        default=0,
        description="Workers connected to the credit router.",
    )
    ready_record_processors: int = Field(
        default=0,
        description="Record processors currently available across worker pods.",
    )
    declared_record_processors: int = Field(
        default=0,
        description="Declared record-processor count across worker pods.",
    )
    ready_pods: int = Field(
        default=0,
        description="Pods with usable worker capacity.",
    )
    total_pods: int = Field(
        default=0,
        description="Total worker pods seen by the controller.",
    )
    degraded_pods: int = Field(
        default=0,
        description="Pods that are usable but degraded.",
    )


@dataclass(frozen=True, slots=True)
class K8sServiceTopology:
    """Expected Kubernetes worker-pod topology derived from runtime config."""

    num_worker_pods: int
    """Number of Kubernetes worker pods to deploy."""

    workers_per_pod: int
    """Number of worker processes per pod."""

    record_processors_per_pod: int
    """Number of record processor processes per pod."""

    total_workers: int
    """Total worker count across all pods."""

    total_record_processors: int
    """Total record processor count across all pods."""


def build_aggregate_worker_status(
    pod_states: dict[str, WorkerPodStateMessage],
) -> AggregateWorkerStatus:
    """Summarize worker-pod snapshots into controller aggregate status."""
    pods = list(pod_states.values())
    return AggregateWorkerStatus(
        ready=sum(pod.ready_workers for pod in pods),
        total=sum(pod.declared_workers for pod in pods),
        dispatchable=sum(pod.dispatchable_workers for pod in pods),
        router_connected=sum(pod.router_connected_workers for pod in pods),
        ready_record_processors=sum(pod.ready_record_processors for pod in pods),
        declared_record_processors=sum(pod.declared_record_processors for pod in pods),
        ready_pods=sum(
            1
            for pod in pods
            if pod.dispatchable_workers >= 1 and pod.ready_record_processors >= 1
        ),
        total_pods=len(pods),
        degraded_pods=sum(
            1
            for pod in pods
            if pod.dispatchable_workers >= 1
            and pod.ready_record_processors >= 1
            and (pod.degraded_workers > 0 or pod.degraded_record_processors > 0)
        ),
    )
