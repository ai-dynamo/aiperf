# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Kubernetes environment configuration.

All settings can be configured via environment variables with the AIPERF_K8S_ prefix.
Resource settings per container type use AIPERF_K8S_{SERVICE}_{FIELD} naming.

Examples:
    AIPERF_K8S_SYSTEM_CONTROLLER_CPU=250m
    AIPERF_K8S_DATASET_MANAGER_MEMORY=512Mi
    AIPERF_K8S_WORKER_POD_MEMORY=8Gi
    AIPERF_K8S_HEALTH_INITIAL_DELAY_SECONDS=10

See also: ``aiperf.operator.environment.OperatorEnvironment`` (operator-process
tunables) and ``aiperf.common.environment.Environment`` (shared AIPerf runtime).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "CONTROLLER_OPTIONAL_RESOURCE_KEYS",
    "CONTROLLER_REQUIRED_RESOURCE_KEYS",
    "CONTROLLER_RESOURCE_KEYS",
    "K8sEnvironment",
]


CONTROLLER_REQUIRED_RESOURCE_KEYS = (
    "SYSTEM_CONTROLLER",
    "TIMING_MANAGER",
    "DATASET_MANAGER",
    "RECORDS_MANAGER",
    "API",
)
"""Controller containers that are always present in Kubernetes mode."""

CONTROLLER_OPTIONAL_RESOURCE_KEYS = (
    "GPU_TELEMETRY_MANAGER",
    "SERVER_METRICS_MANAGER",
    "EVENT_BUS_PROXY",
)
"""Controller containers that depend on benchmark config flags."""

CONTROLLER_RESOURCE_KEYS = (
    *CONTROLLER_REQUIRED_RESOURCE_KEYS,
    *CONTROLLER_OPTIONAL_RESOURCE_KEYS,
    "RESULTS_SIDECAR",
)
"""All controller-pod container resource settings, including the results sidecar."""


class ResourceSettings(BaseSettings):
    """Container resource settings (CPU/memory).

    Used by resource_mode to produce Kubernetes resource specs:
    - guaranteed: requests == limits (Guaranteed QoS)
    - burstable: requests only, no limits (Burstable QoS)
    - none: omits the resources block entirely
    """

    CPU: str = Field(description="CPU request (and limit in guaranteed mode)")
    MEMORY: str = Field(description="Memory request (and limit in guaranteed mode)")

    def to_k8s_resources(self, *, burstable: bool = False) -> dict[str, dict[str, str]]:
        """Convert to Kubernetes resource spec.

        Args:
            burstable: If True, emit requests only (no limits). Containers can
                burst beyond the request without being OOM-killed by cgroup.
        """
        resources: dict[str, dict[str, str]] = {
            "requests": {"cpu": self.CPU, "memory": self.MEMORY},
        }
        if not burstable:
            resources["limits"] = {"cpu": self.CPU, "memory": self.MEMORY}
        return resources


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


class _K8sZMQSettings(BaseSettings):
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
    EVENT_BUS_PROXY_HEALTH: int = Field(
        default=8088, ge=1, le=65535, description="Event-bus proxy sidecar health port"
    )
    EVENT_BUS_PROXY_PUB_FRONTEND: int = Field(
        default=5663,
        ge=1,
        le=65535,
        description="Event-bus XPUB/XSUB proxy publisher-frontend bind port "
        "(producers connect to this).",
    )
    EVENT_BUS_PROXY_SUB_BACKEND: int = Field(
        default=5664,
        ge=1,
        le=65535,
        description="Event-bus XPUB/XSUB proxy subscriber-backend bind port "
        "(subscribers connect to this).",
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
        default=20,
        ge=0,
        le=20,
        description="Job backoff limit for workers (allows retries for transient failures)",
    )
    WORKER_CONNECTION_PROBE_TIMEOUT: float = Field(
        default=60.0,
        ge=30.0,
        le=600.0,
        description="Seconds worker pods wait for the PUB/SUB connection probe to succeed. "
        "Overrides AIPERF_SERVICE_CONNECTION_PROBE_TIMEOUT for k8s worker containers only. "
        "Pods that cannot connect exit cleanly so Kubernetes restarts them with a "
        "fresh ZMQ context; WORKER_BACKOFF_LIMIT absorbs transient first-deploy flakes.",
    )
    CONFIG_MOUNT_PATH: str = Field(
        default="/etc/aiperf", description="Path to mount ConfigMap with configs"
    )
    DATASETS_PATH: str = Field(
        default="/aiperf/datasets",
        description="Shared path for dataset files (dataset-manager writes, API serves)",
    )
    SWEEP_AGGREGATE_INLINE_MAX_BYTES: int = Field(
        default=600_000,
        ge=10_000,
        le=900_000,
        description="Max encoded size of the AIPerfSweep aggregate bundle inlined "
        "into status.aggregate. K8s rejects CR patches over ~1 MiB with HTTP 413; "
        "if the bundle exceeds this cap, the sweep-controller drops `confidence` "
        "(the largest contributor on big sweeps) and relies on the disk-backed "
        "results sidecar to serve the full document. Default 600 KB leaves headroom "
        "for status fields and apiserver framing under the 1 MiB ceiling.",
    )
    KUEUE_DEFAULT_QUEUE_NAME: str = Field(
        default="",
        description="Operator-side default for Kueue gang-scheduling. When the "
        "AIPerfJob CR's spec.scheduling.queue_name is unset, the JobSet manifest "
        "falls back to this value. When non-empty, the JobSet gets the "
        "kueue.x-k8s.io/queue-name label, which Kueue's JobSet integration uses "
        "to admit the workload as a unit (gang-scheduling: controller + all "
        "worker pods admitted atomically, or none). Safe to leave unset on "
        "clusters without Kueue — the label is then never added. Set to e.g. "
        "'aiperf-lq' on clusters where Kueue is installed and a LocalQueue "
        "of that name exists in the benchmark namespace.",
    )
    KUEUE_DEFAULT_PRIORITY_CLASS: str = Field(
        default="",
        description="Operator-side default for Kueue WorkloadPriorityClass. "
        "Companion to KUEUE_DEFAULT_QUEUE_NAME. When unset, the JobSet gets "
        "no kueue.x-k8s.io/priority-class label and Kueue's default fairness "
        "applies.",
    )


class _PortForwardSettings(BaseSettings):
    """Tunables for ``aiperf.kubernetes.port_forward`` kubectl-based forwards."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_PORT_FORWARD_")

    TIMEOUT_SECONDS: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Total seconds to wait for kubectl port-forward to start "
        "and (optionally) for the API to respond.",
    )
    API_INITIAL_DELAY_SECONDS: float = Field(
        default=0.5,
        ge=0.0,
        le=10.0,
        description="Seconds to wait after the tunnel comes up before the "
        "first API health check.",
    )
    API_RETRY_DELAY_SECONDS: float = Field(
        default=2.0,
        ge=0.1,
        le=30.0,
        description="Seconds to back off between port-forward restart attempts "
        "while the API isn't ready.",
    )
    API_MAX_RETRIES: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Maximum number of port-forward restarts before giving up "
        "on the API readiness probe.",
    )
    PROCESS_CLEANUP_TIMEOUT_SECONDS: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Seconds to wait for graceful kubectl termination before "
        "escalating to SIGKILL.",
    )


class _ProgressStreamSettings(BaseSettings):
    """Tunables for ``aiperf.kubernetes.progress_stream`` WebSocket reconnects."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_PROGRESS_STREAM_")

    WS_INITIAL_BACKOFF_SECONDS: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial reconnect backoff after a WebSocket transport error.",
    )
    WS_MAX_BACKOFF_SECONDS: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Cap on the exponential reconnect backoff.",
    )
    WS_HEARTBEAT_SECONDS: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Seconds between aiohttp WebSocket heartbeats.",
    )


class _DiagnosisSettings(BaseSettings):
    """Thresholds for ``aiperf.kubernetes.benchmark_diagnosis`` heuristics."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_DIAGNOSIS_")

    STALLED_PENDING_THRESHOLD_SECONDS: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="Pending job is flagged as stalled after this many seconds.",
    )
    STALLED_RUNNING_THRESHOLD_SECONDS: float = Field(
        default=30.0,
        ge=1.0,
        le=3600.0,
        description="Running job with no throughput and no completed requests is "
        "flagged as stalled after this many seconds.",
    )
    HIGH_ERROR_RATE_THRESHOLD: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Error rate (0.0-1.0) above which a high-error-rate finding "
        "is reported.",
    )
    FAIL_ABOVE_ERROR_RATE: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Error rate (0.0-1.0) at or above which a finished benchmark "
        "is reported as Failed instead of Completed. Defaults to 1.0, so only a "
        "run in which every single request errored is failed outright; a run "
        "that merely errored heavily still completes and is flagged by the "
        "high-error-rate diagnosis. Lower it to enforce a stricter success bar.",
    )
    HIGH_LATENCY_P99_MULTIPLIER: float = Field(
        default=10.0,
        ge=1.0,
        le=1000.0,
        description="Multiplier on average latency above which p99 is flagged "
        "as a tail-latency outlier.",
    )


class _WatchdogSettings(BaseSettings):
    """Thresholds for ``aiperf.kubernetes.watchdog`` pod-health heuristics.

    These were plain keyword defaults on ``BenchmarkWatchdog.__init__`` with no
    environment binding, so a cluster with slow image pulls or an intentionally
    restart-tolerant workload had no way to raise them.
    """

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_WATCHDOG_")

    POLL_INTERVAL_SECONDS: float = Field(
        default=5.0,
        ge=0.0,
        le=300.0,
        description="Seconds between watchdog pod-state polls.",
    )
    STATUS_INTERVAL_SECONDS: float = Field(
        default=10.0,
        ge=0.0,
        le=3600.0,
        description="Seconds between watchdog status log lines.",
    )
    PENDING_THRESHOLD_SECONDS: float = Field(
        default=30.0,
        ge=1.0,
        le=3600.0,
        description=(
            "Seconds a pod startup blocker may remain stable before the CLI "
            "watchdog or operator raises a warning."
        ),
    )
    PENDING_CRITICAL_THRESHOLD_SECONDS: float = Field(
        default=90.0,
        ge=1.0,
        le=3600.0,
        description=(
            "Seconds a pod startup blocker may remain stable before escalation "
            "to critical. The operator fails only known non-recoverable image, "
            "configuration, crash-loop, or structural scheduling blockers; "
            "capacity-related scheduling remains retryable."
        ),
    )
    CRASHLOOP_RESTART_THRESHOLD: int = Field(
        default=2,
        ge=1,
        le=100,
        description=(
            "Container restart count at which a crash-loop warning is raised "
            "and the operator may treat a stable CrashLoopBackOff as terminal."
        ),
    )


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

    # Pod-level resource settings (user-facing).
    #
    # These are the container-level requests/limits applied to K8s manifests.
    # Guaranteed QoS: requests == limits (no throttling, dedicated resources).
    # Calibrated via ``tools/measure_cpu_usage.py`` and
    # ``tools/calibrate_memory_estimates.py``; cross-checked
    # against real-cluster RSS measurements (2026-04-30 ISL/OSL memory sweep).
    #
    # Controller pod: one container per control-plane service.
    #   Defaults are low requests for burstable QoS. They reserve enough for
    #   smoke/high-fanout startup while allowing real usage to burst above the
    #   request without cgroup limits. Increase via AIPERF_K8S_* for large runs.
    #
    # Worker pod: one worker-pod-manager plus one container per worker and
    # record processor, all sharing the WORKER_POD request budget.
    #   The 4Gi default holds the per-pod RSS we measured at up to 10K
    #   concurrency (1.8-3 GiB working-set per pod). Push higher with
    #   AIPERF_K8S_WORKER_POD_MEMORY for memory-heavy datasets or extreme
    #   concurrency.
    # fmt: off
    SYSTEM_CONTROLLER: ResourceSettings = Field(default_factory=lambda: _resource_settings("SYSTEM_CONTROLLER_", "75m", "192Mi"), description="SystemController container resources")
    SWEEP_CONTROLLER: ResourceSettings = Field(default_factory=lambda: _resource_settings("SWEEP_CONTROLLER_", "75m", "512Mi"), description="Sweep-controller container resources. Higher memory than SYSTEM_CONTROLLER because adaptive_search imports torch/BoTorch in this pod: measured RSS is 287 MiB after import and 350 MiB after the first GP fit, against SystemController's 192Mi default.")
    TIMING_MANAGER: ResourceSettings = Field(default_factory=lambda: _resource_settings("TIMING_MANAGER_", "50m", "192Mi"), description="TimingManager container resources")
    DATASET_MANAGER: ResourceSettings = Field(default_factory=lambda: _resource_settings("DATASET_MANAGER_", "50m", "256Mi"), description="DatasetManager container resources")
    RECORDS_MANAGER: ResourceSettings = Field(default_factory=lambda: _resource_settings("RECORDS_MANAGER_", "75m", "256Mi"), description="RecordsManager container resources")
    API: ResourceSettings = Field(default_factory=lambda: _resource_settings("API_", "75m", "256Mi"), description="API container resources")
    GPU_TELEMETRY_MANAGER: ResourceSettings = Field(default_factory=lambda: _resource_settings("GPU_TELEMETRY_MANAGER_", "25m", "192Mi"), description="GPU telemetry container resources")
    SERVER_METRICS_MANAGER: ResourceSettings = Field(default_factory=lambda: _resource_settings("SERVER_METRICS_MANAGER_", "25m", "192Mi"), description="Server metrics container resources")
    RESULTS_SIDECAR: ResourceSettings = Field(default_factory=lambda: _resource_settings("RESULTS_SIDECAR_", "25m", "192Mi"), description="Results sidecar resources for serving exported files")
    EVENT_BUS_PROXY: ResourceSettings = Field(default_factory=lambda: _resource_settings("EVENT_BUS_PROXY_", "50m", "64Mi"), description="Event-bus XPUB/XSUB proxy sidecar resources; isolates pub/sub socket I/O from control-plane")
    WORKER_POD: ResourceSettings = Field(default_factory=lambda: _resource_settings("WORKER_POD_", "150m", "4Gi"), description="Worker pod container resources (workers + record processors + WPM)")
    # fmt: on
    RECORD_PROCESSOR_CPU_REQUEST: str | None = Field(
        default=None,
        description="Optional per-record-processor CPU request override inside worker pods",
    )
    RECORD_PROCESSOR_SCALE_FACTOR: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Kubernetes-only default scale factor for record processors per worker pod. "
        "Formula: 1 record processor for every X workers. Default: 1 record processor per worker.",
    )

    EVENT_BUS_SIDECAR_ENABLED: bool = Field(
        default=True,
        description="Run the XPUB/XSUB event-bus proxy as a dedicated sidecar container "
        "in the controller pod rather than inside the control-plane (SystemController) "
        "container. Isolates pub/sub socket accept/forward from the control plane's "
        "event loop so large fan-ins (hundreds of simultaneous RP/worker connections) "
        "at startup don't starve the SystemController. Set to false to revert to the "
        "pre-sidecar behavior where SystemController owns the event-bus proxy.",
    )

    SHARE_PROCESS_NAMESPACE: bool = Field(
        default=False,
        description="When true, JobSet pods spawned by the operator set "
        "podSpec.shareProcessNamespace=true so all containers share a PID "
        "namespace. Enables cross-container `kubectl exec kill -9 <pid>` "
        "for chaos-testing workflows. Keep false in production; chaos "
        "fixtures flip it on via AIPERF_K8S_SHARE_PROCESS_NAMESPACE=true.",
    )

    CONTROLLER_HTTP_URL_OVERRIDE: str | None = Field(
        default=None,
        description="Chaos-test hook: when set, the operator's progress-client "
        "uses this base URL (scheme+host+port, e.g. "
        "http://toxiproxy.aiperf-chaos-toxiproxy.svc:20002) instead of the "
        "per-CR JobSet pod DNS + API_SERVICE port for controller HTTP calls. "
        "Production MUST leave unset — it collapses multi-job isolation "
        "because every CR funnels through the same URL. Chaos fixtures set "
        "it via AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE to steer traffic "
        "through toxiproxy for latency/blackhole injection.",
    )
    APISERVER_TLS_SERVER_NAME_OVERRIDE: str | None = Field(
        default=None,
        description="Chaos-test hook: when KUBERNETES_SERVICE_HOST points at an "
        "L4 proxy rather than kubernetes.default.svc, verify the apiserver "
        "certificate against this hostname while still dialing the proxy. "
        "Production MUST leave unset; C15 sets it to kubernetes.default.svc.",
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
    ZMQ: _K8sZMQSettings = Field(
        default_factory=_K8sZMQSettings,
        description="ZMQ communication settings",
    )
    JOBSET: _JobSetSettings = Field(
        default_factory=_JobSetSettings,
        description="JobSet-level configuration",
    )
    PORT_FORWARD: _PortForwardSettings = Field(
        default_factory=_PortForwardSettings,
        description="kubectl port-forward CLI tunables",
    )
    PROGRESS_STREAM: _ProgressStreamSettings = Field(
        default_factory=_ProgressStreamSettings,
        description="Progress-stream WebSocket reconnect tunables",
    )

    DIAGNOSIS: _DiagnosisSettings = Field(
        default_factory=_DiagnosisSettings,
        description="Benchmark-metric diagnosis thresholds",
    )

    WATCHDOG: _WatchdogSettings = Field(
        default_factory=_WatchdogSettings,
        description="Watchdog pod-health thresholds",
    )


# Global singleton instance
K8sEnvironment = _K8sEnvironment()
