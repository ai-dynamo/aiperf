# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator environment configuration.

All settings can be configured via environment variables with the AIPERF_OPERATOR_ prefix,
or AIPERF_ for shared settings (results dir, job timeout).

Examples:
    AIPERF_OPERATOR_MONITOR_INTERVAL=10.0
    AIPERF_RESULTS_DIR=/data
    AIPERF_JOB_TIMEOUT_SECONDS=3600

See also: ``aiperf.kubernetes.environment.K8sEnvironment`` (cluster defaults
baked into pod manifests) and ``aiperf.common.environment.Environment``
(shared AIPerf runtime).
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "OperatorEnvironment",
]


class _MonitorSettings(BaseSettings):
    """Timer settings for the kopf monitor handler."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_OPERATOR_MONITOR_")

    INTERVAL: float = Field(
        default=10.0,
        gt=0,
        le=3600,
        description="Seconds between progress checks",
    )
    INITIAL_DELAY: float = Field(
        default=5.0,
        ge=0,
        le=300,
        description="Seconds before first progress check after job creation",
    )


class _ResultsSettings(BaseSettings):
    """Results fetching and storage settings."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_RESULTS_")

    DIR: Path = Field(
        default=Path("/data"),
        description="Base directory for storing benchmark results (mounted PVC)",
    )
    MAX_RETRIES: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Max retries when fetching results from controller",
    )
    RETRY_DELAY: float = Field(
        default=2.0,
        ge=0,
        le=60,
        description="Seconds between result fetch retries",
    )
    TTL_DAYS: int = Field(
        default=30,
        ge=0,
        le=3650,
        description="Days to keep results before cleanup (0 = never clean)",
    )
    COMPRESS_ON_DISK: bool = Field(
        default=True,
        description="Store downloaded result files as zstd-compressed (.zst) on disk",
    )
    RETAIN_RUNS: int = Field(
        default=10,
        ge=1,
        le=10000,
        description="Max per-run result dirs to keep under <namespace>/<name>/ "
        "before retention trimming. Applied after every successful completion; "
        "the just-written epoch is always protected from deletion.",
    )
    RETAIN_DAYS: int = Field(
        default=0,
        ge=0,
        le=36500,
        description="Age-based retention cap in days. 0 disables age policy. "
        "A run is deleted only when BOTH this age cap AND RETAIN_RUNS "
        "agree the run is outside the keep window; protect_epoch still wins.",
    )
    TRANSIENT_FETCH_RETRY_BUDGET_SEC: float = Field(
        default=60.0,
        ge=0.0,
        le=600.0,
        description=(
            "Wall-clock budget (seconds, measured from the completion-claim "
            "annotation timestamp) within which a transient HTTP fetch failure "
            "is converted to a kopf.TemporaryError so the next monitor tick "
            "retries via the orphan-claim recovery path. Past this budget the "
            "operator gives up and marks the AIPerfJob Failed with the "
            "ResultsFetchFailed condition. WHY: sub-second benchmarks can "
            "race the controller's post-export shutdown — the marker has been "
            "written and key files exist on the controller PVC, but the "
            "operator's HTTP fetch hits a connection-refused or empty list "
            "as the controller container terminates. Set 0 to disable retries."
        ),
    )
    TRANSIENT_FETCH_RETRY_DELAY_SEC: float = Field(
        default=5.0,
        ge=0.5,
        le=60.0,
        description=(
            "Delay (seconds) passed to ``kopf.TemporaryError`` when retrying "
            "a transient results-fetch failure. Each retry runs through the "
            "orphan-claim recovery path on the next monitor tick."
        ),
    )


class _ProgressSettings(BaseSettings):
    """Operator progress-client retry settings.

    Used by ``aiperf.operator.progress_client.ProgressClient`` when polling the
    controller pod's HTTP progress API. Retries apply to transient failures
    (connection errors, retryable HTTP statuses); other errors propagate.
    """

    model_config = SettingsConfigDict(env_prefix="AIPERF_OPERATOR_PROGRESS_")

    MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Max retry attempts on transient progress-API failures.",
    )
    INITIAL_BACKOFF_SEC: float = Field(
        default=0.5,
        gt=0,
        le=60,
        description="Initial backoff (seconds) between progress-API retries.",
    )
    BACKOFF_MULTIPLIER: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Multiplicative backoff factor between progress-API retries.",
    )


class _SweepControllerSettings(BaseSettings):
    """Sweep-controller pod settings.

    Used by the sweep-controller pod (`aiperf.sweep_controller.k8s_executor`)
    when creating child AIPerfJob CRs.
    """

    model_config = SettingsConfigDict(env_prefix="AIPERF_SWEEP_CONTROLLER_")

    STALE_CHILD_DELETION_TIMEOUT_SECONDS: float = Field(
        default=60.0,
        gt=0,
        le=600,
        description="Max seconds the sweep-controller will wait for a same-named "
        "AIPerfJob from a prior sweep run to finish cascade-deletion before "
        "raising ChildNameConflictError. Hit when a user deletes and recreates "
        "a sweep with the same name while old children are still terminating.",
    )
    STALE_CHILD_POLL_INTERVAL_SECONDS: float = Field(
        default=2.0,
        gt=0,
        le=30,
        description="Poll interval (seconds) while waiting for a deleting same-named "
        "AIPerfJob to disappear. See STALE_CHILD_DELETION_TIMEOUT_SECONDS.",
    )
    CANCEL_GRACE_SECONDS: float = Field(
        default=120.0,
        gt=0,
        le=3600,
        description="Max seconds the sweep-controller will keep polling a child "
        "AIPerfJob for a terminal phase after requesting cancel before giving up "
        "and advancing the sweep. Bounds the post-cancel wait so a stuck child "
        "(stalled operator cancel path, wedged pod, repeatedly-failing JobSet "
        "delete) cannot wedge the whole sweep indefinitely.",
    )
    CHILD_MISSING_TIMEOUT_SECONDS: float = Field(
        default=300.0,
        gt=0,
        le=3600,
        description="Max seconds the sweep-controller will keep polling for a "
        "child AIPerfJob that has gone missing (404) before its terminal phase, "
        "with no cancel requested, before giving up and advancing the sweep. Hit "
        "when a user (or the kube garbage collector) deletes a child AIPerfJob "
        "out-of-band mid-run; without this bound the sequential sweep wedges "
        "forever on the deleted variation.",
    )


class _OperatorServiceSettings(BaseSettings):
    """Operator-service network identity.

    The operator Pod has three containers but only ONE FastAPI app: the
    ``results-server`` sidecar on ``resultsServer.port`` (8081 in the
    chart) hosts every ``/api/v1/*`` router (jobs, sweeps, results,
    config, admin, analytics, dashboard_proxy). The ``operator`` container
    on port 8080 runs kopf only — its sole HTTP surface is ``/healthz``
    plus Prometheus ``/metrics``. So there is no separate "sweeps API URL"
    and "results API URL" — there is one base URL for everything, pointing
    at the results-server.

    Used when the operator stamps absolute URLs onto CR status (e.g.
    ``AIPerfSweep.status.apiUrl``, ``AIPerfSweep.status.runsTruncated.fetchURL``)
    that external clients dereference to fetch results, and when in-pod
    consumers (e.g. the sweep-controller's empty-summary fallback) need
    the operator's API endpoint.
    """

    model_config = SettingsConfigDict(env_prefix="AIPERF_OPERATOR_")

    BASE_URL: str = Field(
        default="http://aiperf-operator.aiperf-system:8081",
        description="Base URL (no trailing slash) for the operator's HTTP API. "
        "All ``/api/v1/*`` routers — jobs, sweeps, results, config, admin, "
        "analytics, dashboard_proxy — are served by the ``results-server`` "
        "container on this port; the operator container exposes only "
        "``/healthz`` + ``/metrics`` on port 8080. Stamped onto "
        "``AIPerfSweep.status.apiUrl`` and ``AIPerfSweep.status.runsTruncated.fetchURL`` "
        "so external clients can fetch per-sweep summaries; also consumed by "
        "the sweep-controller's per-child summary fallback. Override via "
        "``AIPERF_OPERATOR_BASE_URL`` when the operator's Service+Namespace "
        "differ from the Helm chart defaults (e.g. a non-default "
        "``Release.Name`` or an alternate namespace).",
    )


class _DashboardSettings(BaseSettings):
    """Plotly Dashboard sidecar wiring (operator + results-server).

    The dashboard is an opt-in third container in the operator Pod;
    these settings let other containers locate it.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_DASHBOARD_",
    )

    PORT: int = Field(
        default=0,
        ge=0,
        le=65535,
        description="Pod-local HTTP port the dashboard sidecar listens on. "
        "0 means the sidecar is disabled / absent. results-server uses this "
        "to reverse-proxy /dashboard/*; the operator uses it to fire "
        "fire-and-forget refresh POSTs after a benchmark completion claim.",
    )
    PROXY_ENABLED: bool = Field(
        default=False,
        description="When true, results-server forwards /dashboard/* to the "
        "sidecar at localhost:PORT and the SPA shows the 'Plots ↗' top-nav "
        "entry. When false, /dashboard/* returns 503 and the link is hidden. "
        "Set independently from PORT so a misconfigured chart fails closed.",
    )


class _OperatorEnvironment(BaseSettings):
    """Root operator environment configuration.

    Loads from environment variables. Nested settings use their own prefixes.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    JOB_TIMEOUT_SECONDS: float = Field(
        default=0,
        ge=0,
        description="Job timeout in seconds (0 = no timeout)",
    )
    POD_RESTART_THRESHOLD: int = Field(
        default=3,
        ge=0,
        le=100,
        description="Pod restart count before emitting a warning event",
    )
    METRICS_PORT: int = Field(
        default=9090,
        ge=0,
        le=65535,
        description=(
            "Port for the Prometheus /metrics endpoint exposed by the kopf "
            "operator process. Set to 0 to disable. Scraped by ServiceMonitor."
        ),
    )
    ENDPOINT_CHECK_TIMEOUT: float = Field(
        default=10.0,
        gt=0,
        le=300,
        description="Seconds to wait for endpoint health check",
    )
    PREFLIGHT_TIMEOUT: float = Field(
        default=30.0,
        gt=0,
        le=120,
        description="Seconds to wait for all pre-flight checks to complete",
    )
    CONFIGMAP_PROPAGATION_DELAY_SECONDS: float = Field(
        default=10.0,
        ge=0,
        le=60,
        description="Seconds to wait after creating the benchmark ConfigMap before creating "
        "the JobSet. Allows kubelet caches on worker nodes to sync the ConfigMap before "
        "pods start mounting it, preventing FailedMount races on first deployment with "
        "a freshly pulled image.",
    )
    MUTATING_ROUTES_ENABLED: bool = Field(
        default=False,
        description="Enable results-server HTTP routes that mutate Kubernetes state. "
        "Defaults false so read-only results APIs remain exposed while POST routes "
        "fail closed unless an operator explicitly opts in.",
    )
    MUTATING_ROUTES_TOKEN: str = Field(
        default="",
        description="Bearer token required by enabled results-server mutating routes. "
        "Leave empty to fail closed even when MUTATING_ROUTES_ENABLED is true.",
    )

    CLUSTER_NAME: str = Field(
        default="",
        description="Optional human-readable cluster name surfaced in the UI top banner "
        "(e.g. 'dgx-prod', 'kind-aiperf'). When unset the banner falls back to the "
        "Kubernetes server version. Set via AIPERF_OPERATOR_CLUSTER_NAME on the "
        "operator deployment.",
    )

    MONITOR: _MonitorSettings = Field(
        default_factory=_MonitorSettings,
        description="Monitor timer settings",
    )
    RESULTS: _ResultsSettings = Field(
        default_factory=_ResultsSettings,
        description="Results fetching and storage settings",
    )
    PROGRESS: _ProgressSettings = Field(
        default_factory=_ProgressSettings,
        description="Progress-client retry settings (controller HTTP polling).",
    )
    SWEEP_CONTROLLER: _SweepControllerSettings = Field(
        default_factory=_SweepControllerSettings,
        description="Sweep-controller pod settings (child-create collision handling).",
    )
    SERVICE: _OperatorServiceSettings = Field(
        default_factory=_OperatorServiceSettings,
        description="Operator-service network identity (base URL stamped onto CR status).",
    )
    DASHBOARD: _DashboardSettings = Field(
        default_factory=_DashboardSettings,
        description="Plotly Dashboard sidecar wiring.",
    )


OperatorEnvironment = _OperatorEnvironment()
