# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes label, annotation, and container-name constants.

Defined in this dependency-free module so both manifest-generation code
(jobset.py, resources.py) and CLI code (cli_helpers.py) can import them
without circular deps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobSetLabels:
    """Label keys from the JobSet controller (jobset.sigs.k8s.io)."""

    POD_INDEX: str = "jobset.sigs.k8s.io/job-index"
    """Pod index within the job."""

    JOBSET_NAME: str = "jobset.sigs.k8s.io/jobset-name"
    """Owning JobSet resource name."""

    REPLICATED_JOB_NAME: str = "jobset.sigs.k8s.io/replicatedjob-name"
    """Replicated job name within the JobSet."""


@dataclass(frozen=True)
class Labels:
    """Label keys and values used to identify AIPerf resources."""

    APP_KEY: str = "app"
    """Standard app label key."""

    APP_VALUE: str = "aiperf"
    """Standard app label value."""

    JOB_ID: str = "aiperf.nvidia.com/job-id"
    """Unique benchmark job identifier."""

    NAME: str = "aiperf.nvidia.com/name"
    """Human-readable benchmark name."""

    PARENT: str = "aiperf.nvidia.com/parent"
    """Parent resource name for sweep runs."""

    SWEEP_RUN: str = "aiperf.nvidia.com/sweep-run"
    """Sweep run identifier."""

    VARIATION_INDEX: str = "aiperf.nvidia.com/variation-index"
    """Sweep variation index within a sweep."""

    TRIAL: str = "aiperf.nvidia.com/trial"
    """Trial number for repeated runs."""

    RUN_INDEX: str = "aiperf.nvidia.com/run-index"
    """Sequential run index within a sweep."""

    SELECTOR: str = "app=aiperf"
    """Label selector string for filtering AIPerf pods."""


@dataclass(frozen=True)
class Annotations:
    """Annotation keys used on AIPerf Kubernetes resources."""

    MODEL: str = "aiperf.nvidia.com/model"
    """Target model name for the benchmark."""

    ENDPOINT: str = "aiperf.nvidia.com/endpoint"
    """Target inference endpoint URL."""

    BENCHMARK_COMPLETE: str = "aiperf.nvidia.com/benchmark-complete"
    """Marks the benchmark as finished."""

    COMPLETION_CLAIMED: str = "aiperf.nvidia.com/completion-claimed"
    """Set by the operator when handle_completion begins for a job.

    Durable marker that survives operator pod restart so the completion
    branch is not re-entered if the previous run crashed before phase
    reached Completed."""


@dataclass(frozen=True)
class ProgressAnnotations:
    """Progress annotations patched onto the JobSet during benchmark execution.

    External tools can observe benchmark progress without connecting to the
    controller pod's API.
    """

    PHASE: str = "aiperf.nvidia.com/progress-phase"
    """Current benchmark phase name."""

    PERCENT: str = "aiperf.nvidia.com/progress-percent"
    """Completion percentage of the current phase."""

    REQUESTS: str = "aiperf.nvidia.com/progress-requests"
    """Completed and total request counts."""

    STATUS: str = "aiperf.nvidia.com/progress-status"
    """Human-readable status summary."""


@dataclass(frozen=True)
class Containers:
    """Container names used in pod specs and CLI commands."""

    CONTROL_PLANE: str = "control-plane"
    """SystemController and orchestration logic."""

    DATASET_MANAGER: str = "dataset-manager"
    """Dataset generation and memory-map serving."""

    TIMING_MANAGER: str = "timing-manager"
    """Request scheduling and timing coordination."""

    WORKER_MANAGER: str = "worker-manager"
    """Worker lifecycle management (deprecated name, kept for compat)."""

    RECORDS_MANAGER: str = "records-manager"
    """Metric record aggregation and storage."""

    API: str = "api"
    """HTTP API service for monitoring and data access."""

    GPU_TELEMETRY_MANAGER: str = "gpu-telemetry-manager"
    """DCGM GPU metrics collection."""

    SERVER_METRICS_MANAGER: str = "server-metrics-manager"
    """Prometheus server metrics scraping."""

    RESULTS_SIDECAR: str = "results-sidecar"
    """Lightweight sidecar serving exported result artifacts."""

    EVENT_BUS_PROXY: str = "event-bus-proxy"
    """Dedicated XPUB/XSUB event-bus proxy sidecar in the controller pod.
    Isolates pub/sub forwarding from the SystemController event loop so that
    hundreds of simultaneous RP/worker connections at startup don't starve
    the control plane's CPU."""

    WORKER_GROUP_MANAGER: str = "worker-group-manager"
    """Per-pod worker group lifecycle and dataset coordination."""


@dataclass(frozen=True)
class KueueLabels:
    """Label keys for Kueue queue integration."""

    QUEUE_NAME: str = "kueue.x-k8s.io/queue-name"
    """Kueue local queue name for job admission."""

    PRIORITY_CLASS: str = "kueue.x-k8s.io/priority-class"
    """Kueue priority class for scheduling priority."""


# Default namespace for benchmark jobs. All benchmark runs land here unless
# the user provides --namespace explicitly.
DEFAULT_BENCHMARK_NAMESPACE = "aiperf-benchmarks"

# CRD API coordinates for AIPerfJob live in aiperf.kubernetes.cr_refs.
# Re-export here for callers that historically imported them from this
# module; new code should import from aiperf.kubernetes.cr_refs directly.
from aiperf.kubernetes.cr_refs import (  # noqa: E402
    AIPERF_GROUP,
    AIPERF_PLURAL,
    AIPERF_VERSION,
)

ANNOTATION_PREFIX = AIPERF_GROUP

__all__ = [
    "AIPERF_GROUP",
    "AIPERF_PLURAL",
    "AIPERF_VERSION",
    "ANNOTATION_PREFIX",
]

__all__ = [
    "AIPERF_GROUP",
    "AIPERF_PLURAL",
    "AIPERF_VERSION",
    "ANNOTATION_PREFIX",
]
