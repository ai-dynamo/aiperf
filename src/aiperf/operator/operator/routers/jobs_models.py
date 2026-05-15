# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic request/response models for the jobs API router."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from aiperf.common.models import AIPerfBaseModel
from aiperf.kubernetes.models import AIPerfJobInfo

# ``ActiveJobSummary`` is the API-layer name for the per-job summary returned
# by GET /api/v1/jobs and friends. The underlying flat display model
# (``AIPerfJobInfo``) is shared with the CLI; re-exporting under a
# response-shaped alias keeps router code grep-friendly without duplicating
# the schema.
ActiveJobSummary = AIPerfJobInfo


class JobPodSummary(AIPerfBaseModel):
    """Pod identity + lifecycle summary returned in JobDetailResponse.

    Distinct from ``aiperf.kubernetes.models.PodSummary`` (an aggregate
    ``ready/total/restarts`` snapshot of a JobSet): this model is per-pod and
    includes the pod name / phase.
    """

    name: str = Field(description="Pod name.")
    phase: str = Field(description="Pod phase (Running, Pending, Succeeded, ...).")
    ready: bool = Field(description="True iff at least one container is ready.")
    restarts: int = Field(description="Sum of restart counts across containers.")
    containers: list[str] = Field(
        default_factory=list,
        description=(
            "Container names declared on the pod spec, in spec order. Drives "
            "the per-container picker in the UI logs pane; the same names are "
            "valid values for the ``container=`` query parameter on the logs "
            "endpoint. Empty list when the pod has no spec (e.g. terminated)."
        ),
    )


class ActiveJobListResponse(AIPerfBaseModel):
    """Response for GET /api/v1/jobs: active AIPerfJob CRs in the cluster."""

    jobs: list[dict[str, Any]] = Field(description="List of AIPerfJob summaries.")


class JobDetailResponse(AIPerfBaseModel):
    """Response for GET /api/v1/jobs/{namespace}/{name}."""

    job: dict[str, Any] = Field(description="AIPerfJob summary.")
    status: dict[str, Any] = Field(
        description="Raw CR status (phases, conditions, liveMetrics)."
    )
    pods: list[JobPodSummary] = Field(description="Pod summaries for this job.")


class ClusterResponse(AIPerfBaseModel):
    """Response for GET /api/v1/cluster."""

    nodes: int = Field(description="Number of cluster nodes.")
    gpus: int = Field(description="Total allocatable nvidia.com/gpu resources.")
    gpus_used: int = Field(
        default=0,
        description=(
            "Sum of nvidia.com/gpu requests across Running and Pending pods "
            "in all namespaces. 0 if cluster-wide pod listing fails."
        ),
    )
    gpus_free: int = Field(
        default=0,
        description="Allocatable GPUs not currently requested. 0 if pod listing fails.",
    )
    utilization_percent: float = Field(
        default=0.0,
        description="100 * gpus_used / gpus, rounded to one decimal. 0 if no GPUs.",
    )
    gpu_nodes: int = Field(
        default=0, description="Number of nodes with at least one nvidia.com/gpu."
    )
    nodes_free: int = Field(
        default=0, description="GPU nodes with zero requested GPUs (fully available)."
    )
    nodes_partial: int = Field(
        default=0, description="GPU nodes with some but not all GPUs requested."
    )
    nodes_full: int = Field(
        default=0, description="GPU nodes with every GPU requested."
    )
    kubernetes_version: str = Field(description="Kubernetes server version.")
    cluster_name: str | None = Field(
        default=None,
        description="Optional human-readable cluster name (from "
        "AIPERF_OPERATOR_CLUSTER_NAME). The UI banner shows this in place "
        "of the bare Kubernetes version when set.",
    )


class CancelResponse(AIPerfBaseModel):
    """Response for POST /api/v1/jobs/{namespace}/{name}/cancel."""

    cancelled: bool = Field(description="Whether cancellation was requested.")


class EventInvolvedObject(AIPerfBaseModel):
    """Subset of ``V1ObjectReference`` returned in ``EventEntry.involved_object``."""

    kind: str | None = Field(
        default=None, description="Involved object kind (Pod, AIPerfJob, ...)."
    )
    name: str | None = Field(default=None, description="Involved object name.")
    namespace: str | None = Field(
        default=None, description="Involved object namespace."
    )


class EventSource(AIPerfBaseModel):
    """Subset of ``V1EventSource`` returned in ``EventEntry.source``."""

    component: str | None = Field(
        default=None, description="Emitting component (e.g. kubelet, kopf)."
    )
    host: str | None = Field(
        default=None, description="Node/host that emitted the event."
    )


class EventEntry(AIPerfBaseModel):
    """A single Kubernetes Event relevant to an AIPerfJob run."""

    type: str | None = Field(default=None, description="Event type: Normal or Warning.")
    reason: str | None = Field(
        default=None,
        description="Short CamelCase reason (e.g. Scheduled, FailedMount).",
    )
    message: str | None = Field(
        default=None, description="Human-readable event message."
    )
    source: EventSource = Field(
        default_factory=EventSource,
        description="Emitting component + host (may both be None for kopf-emitted events).",
    )
    involved_object: EventInvolvedObject = Field(
        default_factory=EventInvolvedObject,
        description="The resource the event is about (CR, pod, ...).",
    )
    first_timestamp: str | None = Field(
        default=None,
        description="ISO-8601 first-seen timestamp; may be None for server-side events (see event_time).",
    )
    last_timestamp: str | None = Field(
        default=None,
        description="ISO-8601 last-seen timestamp; ordering key for the response list.",
    )
    count: int | None = Field(
        default=None,
        description="Number of times the event has fired (de-duplicated series).",
    )


class JobEventsResponse(AIPerfBaseModel):
    """Response for GET /api/v1/jobs/{namespace}/{name}/events."""

    events: list[EventEntry] = Field(
        description="Events for the AIPerfJob CR and its owned pods, newest first, capped at 200."
    )


class CreateJobRequest(AIPerfBaseModel):
    """Request body for POST /api/v1/jobs: a full AIPerfJob manifest as a dict.

    Pass the same shape you'd submit via ``kubectl apply -f``. ``apiVersion``
    and ``kind`` may be omitted — the handler fills them in. ``namespace``
    defaults to ``default`` when absent from ``metadata``.
    """

    manifest: dict[str, Any] = Field(description="AIPerfJob manifest.")


class CreateJobResponse(AIPerfBaseModel):
    """Response for POST /api/v1/jobs."""

    namespace: str = Field(description="Namespace the CR was created in.")
    name: str = Field(description="CR name (from metadata.name).")
    uid: str | None = Field(default=None, description="K8s-assigned UID.")


class JobEpochSummary(AIPerfBaseModel):
    """One epoch entry in the job-history listing."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        populate_by_name=True,
        alias_generator=to_camel,
    )

    epoch: str = Field(description="Decimal-seconds epoch identifier.")
    is_latest: bool = Field(
        description="Whether this is the current latest epoch (per latest.txt)."
    )
    mtime_epoch: int = Field(description="UNIX seconds of the run dir's mtime.")
    file_count: int = Field(
        description="Number of files persisted under this epoch dir."
    )
    status: Literal["running", "succeeded", "failed", "cancelled", "unknown"] = Field(
        default="unknown",
        description=(
            "Normalized run status. 'running' for the live in-flight epoch; "
            "'succeeded'/'failed'/'cancelled' for terminal phases; "
            "'unknown' when the runs index hasn't ingested this epoch yet."
        ),
    )
    started_at: int | None = Field(
        default=None,
        description="UNIX seconds when this run started, or None if unknown.",
    )
    ended_at: int | None = Field(
        default=None,
        description="UNIX seconds when this run ended, or None if still running / unknown.",
    )


class JobEpochsResponse(AIPerfBaseModel):
    """Response for GET /api/v1/jobs/{namespace}/{name}/epochs."""

    epochs: list[JobEpochSummary] = Field(
        default_factory=list,
        description="Run epochs for this job, ascending; latest flagged via is_latest.",
    )
