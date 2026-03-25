# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Kubernetes environment configuration.

All settings can be configured via environment variables with the AIPERF_K8S_ prefix.
Resource settings per container type use AIPERF_K8S_{SERVICE}_{FIELD} naming.

Examples:
    AIPERF_K8S_CONTROLLER_POD_CPU=4000m
    AIPERF_K8S_WORKER_POD_MEMORY=4Gi
    AIPERF_K8S_HEALTH_INITIAL_DELAY_SECONDS=10
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "K8sEnvironment",
]


# ---------------------------------------------------------------------------
# Resource settings: one base class, instances created via _resource_settings()
# ---------------------------------------------------------------------------


class ResourceSettings(BaseSettings):
    """Container resource settings (CPU/memory).

    Guaranteed QoS: a single CPU/MEMORY value sets both request and limit,
    ensuring pods always get Guaranteed QoS class.
    """

    CPU: str = Field(description="CPU request and limit (Guaranteed QoS)")
    MEMORY: str = Field(description="Memory request and limit (Guaranteed QoS)")

    def to_k8s_resources(self) -> dict[str, dict[str, str]]:
        """Convert to Kubernetes resource spec (requests == limits)."""
        return {
            "requests": {"cpu": self.CPU, "memory": self.MEMORY},
            "limits": {"cpu": self.CPU, "memory": self.MEMORY},
        }


def _resource_settings(
    env_prefix: str,
    cpu: str,
    memory: str,
) -> ResourceSettings:
    """Create a ResourceSettings instance with the given env prefix and defaults.

    Each instance reads from AIPERF_K8S_{env_prefix}_{FIELD} environment
    variables, falling back to the provided defaults.
    """
    cls = type(
        f"_{env_prefix.rstrip('_')}Settings",
        (ResourceSettings,),
        {
            "__annotations__": {
                "CPU": str,
                "MEMORY": str,
            },
            "model_config": SettingsConfigDict(env_prefix=f"AIPERF_K8S_{env_prefix}"),
            "CPU": Field(
                default=cpu, description="CPU request and limit (Guaranteed QoS)"
            ),
            "MEMORY": Field(
                default=memory, description="Memory request and limit (Guaranteed QoS)"
            ),
        },
    )
    return cls()


# ---------------------------------------------------------------------------
# Non-resource settings
# ---------------------------------------------------------------------------


class _HealthProbeSettings(BaseSettings):
    """Health probe configuration for all containers."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_HEALTH_")

    INITIAL_DELAY_SECONDS: int = Field(
        default=5,
        ge=0,
        le=300,
        description="Seconds before starting probes after container starts",
    )
    PERIOD_SECONDS: int = Field(
        default=10,
        ge=1,
        le=300,
        description="Interval in seconds between probe checks",
    )
    TIMEOUT_SECONDS: int = Field(
        default=5, ge=1, le=60, description="Seconds before probe times out"
    )
    FAILURE_THRESHOLD: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Consecutive failures before container is restarted/marked unready",
    )
    SUCCESS_THRESHOLD: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Consecutive successes before container is marked healthy",
    )
    STARTUP_PERIOD_SECONDS: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Interval between startup probe checks",
    )
    STARTUP_FAILURE_THRESHOLD: int = Field(
        default=30,
        ge=1,
        le=120,
        description="Consecutive startup probe failures before pod is killed. "
        "Total startup time = STARTUP_PERIOD_SECONDS * STARTUP_FAILURE_THRESHOLD",
    )


class _ZMQSettings(BaseSettings):
    """ZMQ communication settings for Kubernetes deployments."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_ZMQ_")

    CONTROLLER_HOST: str | None = Field(
        default=None,
        description="Controller hostname for ZMQ dual-bind mode. "
        "Set on worker pods to connect via TCP to controller. "
        "When None, services use IPC (controller mode).",
    )
    IPC_PATH: str = Field(
        default="/aiperf/ipc", description="Path for IPC socket files in pods"
    )


class _PortSettings(BaseSettings):
    """Container port assignments."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_PORT_")

    # Controller pod ports
    SYSTEM_CONTROLLER_HEALTH: int = Field(
        default=8080, ge=1, le=65535, description="System controller health port"
    )
    WORKER_MANAGER_HEALTH: int = Field(
        default=8081, ge=1, le=65535, description="Worker manager health port"
    )
    TIMING_MANAGER_HEALTH: int = Field(
        default=8082, ge=1, le=65535, description="Timing manager health port"
    )
    DATASET_MANAGER_HEALTH: int = Field(
        default=8083, ge=1, le=65535, description="Dataset manager health port"
    )
    RECORDS_MANAGER_HEALTH: int = Field(
        default=8084, ge=1, le=65535, description="Records manager health port"
    )
    API_SERVICE: int = Field(
        default=9090, ge=1, le=65535, description="API service port"
    )
    RESULTS_SIDECAR: int = Field(
        default=9091,
        ge=1,
        le=65535,
        description="Results sidecar port for serving exported files after controller failure",
    )
    API_SERVICE_HEALTH: int = Field(
        default=8085, ge=1, le=65535, description="API service health port"
    )
    GPU_TELEMETRY_MANAGER_HEALTH: int = Field(
        default=8086, ge=1, le=65535, description="GPU telemetry manager health port"
    )
    SERVER_METRICS_MANAGER_HEALTH: int = Field(
        default=8087, ge=1, le=65535, description="Server metrics manager health port"
    )

    # Worker pod ports
    WORKER_HEALTH: int = Field(
        default=8080, ge=1, le=65535, description="Worker health port"
    )
    RECORD_PROCESSOR_HEALTH: int = Field(
        default=8081, ge=1, le=65535, description="Record processor health port"
    )


class _JobSetSettings(BaseSettings):
    """JobSet-level configuration."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_JOBSET_")

    TTL_SECONDS_AFTER_FINISHED: int | None = Field(
        default=300,
        ge=0,
        description="Seconds to keep JobSet after completion (None to disable)",
    )
    DIRECT_MODE_TTL_SECONDS: int = Field(
        default=28800,
        ge=0,
        description="TTL for operator-less (direct) deployments. Pods stay alive "
        "for manual results retrieval. Default 8 hours (28800s).",
    )
    CONTROLLER_BACKOFF_LIMIT: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Job backoff limit for controller (0 = no retries)",
    )
    WORKER_BACKOFF_LIMIT: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Job backoff limit for workers (allows retries for transient failures)",
    )
    CONFIG_MOUNT_PATH: str = Field(
        default="/etc/aiperf", description="Path to mount ConfigMap with configs"
    )
    DATASETS_PATH: str = Field(
        default="/aiperf/datasets",
        description="Shared path for dataset files (dataset-manager writes, API serves)",
    )


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------


class _K8sEnvironment(BaseSettings):
    """Root Kubernetes environment configuration.

    Loads configuration from environment variables with the AIPERF_K8S_ prefix.
    Resource settings per container type are created via _resource_settings()
    with service-specific env prefixes and defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_K8S_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # ---------------------------------------------------------------------------
    # Pod-level resource settings (user-facing).
    #
    # These are the container-level requests/limits applied to K8s manifests.
    # Guaranteed QoS: requests == limits (no throttling, dedicated resources).
    # Calibrated via scripts/measure_cpu_usage.py and scripts/calibrate_memory_estimates.py.
    #
    # Controller pod: single container running all control-plane services as subprocesses.
    #   Measured total: ~1.0 core CPU, ~0.6 GiB memory at typical workload.
    #   Default 3 cores / 2 GiB covers: timing_manager peak (1 core), dataset generation
    #   spike, records_manager scaling with request count, ZMQ proxies.
    #
    # Worker pod: single container running N workers + M record_processors + WPM.
    #   Measured per-worker: 131m CPU / 50-80 MiB at realistic server latency.
    #   Measured per-RP: 389m CPU / 200+ MiB (tokenizer-dependent).
    #   Default 3.3 cores / 6 GiB covers typical 10-worker pods with record-processing
    #   overhead. Increase via AIPERF_K8S_WORKER_POD_{CPU,MEMORY} for heavier workloads.
    # ---------------------------------------------------------------------------
    # Pod-level resource settings (user-facing).
    # Guaranteed QoS: requests == limits (no throttling, dedicated resources).
    # Calibrated via scripts/measure_cpu_usage.py and scripts/calibrate_memory_estimates.py.
    # ---------------------------------------------------------------------------
    # fmt: off
    CONTROLLER_POD: ResourceSettings = Field(default_factory=lambda: _resource_settings("CONTROLLER_POD_", "3000m", "2176Mi"), description="Controller pod container resources (all control-plane services)")
    RESULTS_SIDECAR: ResourceSettings = Field(default_factory=lambda: _resource_settings("RESULTS_SIDECAR_", "100m", "128Mi"), description="Results sidecar resources for serving exported files")
    WORKER_POD: ResourceSettings = Field(default_factory=lambda: _resource_settings("WORKER_POD_", "3350m", "6144Mi"), description="Worker pod container resources (workers + record processors + WPM)")
    # fmt: on
    RECORD_PROCESSOR_SCALE_FACTOR: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Kubernetes-only default scale factor for record processors per worker pod. "
        "Formula: 1 record processor for every X workers. Default: 1 record processor per worker.",
    )

    # Non-resource settings
    HEALTH: _HealthProbeSettings = Field(
        default_factory=_HealthProbeSettings,
        description="Health probe configuration",
    )
    PORTS: _PortSettings = Field(
        default_factory=_PortSettings,
        description="Container port assignments",
    )
    ZMQ: _ZMQSettings = Field(
        default_factory=_ZMQSettings,
        description="ZMQ communication settings",
    )
    JOBSET: _JobSetSettings = Field(
        default_factory=_JobSetSettings,
        description="JobSet-level configuration",
    )


# Global singleton instance
K8sEnvironment = _K8sEnvironment()
