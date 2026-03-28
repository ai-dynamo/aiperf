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

# CRD API coordinates for AIPerfJob
AIPERF_GROUP = "aiperf.nvidia.com"
AIPERF_VERSION = "v1alpha1"
AIPERF_PLURAL = "aiperfjobs"
ANNOTATION_PREFIX = AIPERF_GROUP
