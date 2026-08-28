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
from typing import Self

from pydantic import AliasChoices, Field, model_validator
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
    MISSING_JOBSET_SETTLE_DELAY_SECONDS: float = Field(
        default=2.0,
        ge=0,
        le=60,
        description="Seconds to wait before re-reading an AIPerfJob whose JobSet "
        "disappeared, allowing a concurrent completion status patch to settle.",
    )


class _ResultsSettings(BaseSettings):
    """Results fetching and storage settings."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_RESULTS_")

    DIR: Path = Field(
        default=Path("/data"),
        description="Base directory for storing benchmark results (mounted PVC)",
    )
    SERVER_PORT: int = Field(
        default=8081,
        ge=1,
        le=65535,
        description="Port exposed by the operator results-server sidecar.",
    )
    K8S_INIT_TIMEOUT_SEC: float = Field(
        default=10.0,
        gt=0,
        le=120,
        description=(
            "Seconds the results-server waits for its Kubernetes client to "
            "initialize at startup before giving up and serving PVC-only. The "
            "live-job endpoints need a cluster, but every results, sweeps, and "
            "artifact route reads the disk, so an unreachable apiserver must "
            "degrade the server rather than prevent it from starting."
        ),
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
    DOWNLOAD_TIMEOUT_SECONDS: float = Field(
        default=300.0,
        gt=0,
        le=3600,
        description="Total timeout in seconds for one controller result-file download.",
    )
    DOWNLOAD_MAX_CONCURRENCY: int = Field(
        default=5,
        ge=1,
        le=128,
        description="Maximum result files downloaded concurrently from one controller.",
    )
    RETRY_MAX_DELAY_SECONDS: float = Field(
        default=30.0,
        ge=0,
        le=600,
        description="Maximum backoff delay in seconds between result-fetch attempts.",
    )
    RETRY_BACKOFF_MULTIPLIER: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Multiplicative backoff factor between result-fetch attempts.",
    )
    CLEANUP_INTERVAL_SECONDS: float = Field(
        default=86400.0,
        gt=0,
        le=604800,
        description="Seconds between job and sweep result-retention passes.",
    )
    CLEANUP_INITIAL_DELAY_SECONDS: float = Field(
        default=3600.0,
        ge=0,
        le=604800,
        description="Seconds before the first per-job result-retention pass.",
    )
    CLEANUP_IDLE_SECONDS: float = Field(
        default=3600.0,
        ge=0,
        le=604800,
        description="Minimum idle seconds before a per-job result-retention timer runs.",
    )
    GZIP_MINIMUM_SIZE_BYTES: int = Field(
        default=500,
        ge=0,
        le=1048576,
        description="Minimum response size in bytes compressed by the results API.",
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
    PHASE_SETTLE_ATTEMPTS: int = Field(
        default=3,
        ge=0,
        le=20,
        description=(
            "How many times the completion handler re-samples controller "
            "progress while a phase reports its requests finished but its "
            "records still aggregating. Record aggregation trails the last "
            "request by a beat, so a single sample can leave status.phases "
            "showing isRecordsComplete=false on a run whose exports are "
            "complete. Set 0 to take exactly one sample."
        ),
    )
    PHASE_SETTLE_DELAY_SEC: float = Field(
        default=2.0,
        ge=0.1,
        le=30.0,
        description=(
            "Delay between the re-samples controlled by "
            "``PHASE_SETTLE_ATTEMPTS``. The total wait is a hard ceiling on "
            "how long completion is delayed for a cosmetic status mirror."
        ),
    )

    @model_validator(mode="after")
    def validate_retry_delay_ceiling(self) -> Self:
        """Keep exponential retry backoff from shrinking its initial delay."""
        if self.RETRY_MAX_DELAY_SECONDS < self.RETRY_DELAY:
            raise ValueError("RETRY_MAX_DELAY_SECONDS must be >= RETRY_DELAY")
        return self


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
    REQUEST_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        gt=0,
        le=300,
        description="Total timeout in seconds for an ordinary progress-API request.",
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

    CHILD_POLL_INTERVAL_SECONDS: float = Field(
        default=5.0,
        gt=0,
        le=300,
        description="Seconds between child AIPerfJob terminal-phase polls.",
    )
    CANCEL_POLL_INTERVAL_SECONDS: float = Field(
        default=10.0,
        gt=0,
        le=300,
        description="Seconds between parent AIPerfSweep cancellation-flag polls.",
    )
    RECOVERY_SUMMARY_CONCURRENCY: int = Field(
        default=8,
        ge=1,
        le=128,
        description="Maximum concurrent child-summary fetches during sweep recovery.",
    )
    OPERATOR_API_MAX_ATTEMPTS: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum attempts to fetch one child summary from the operator API.",
    )
    OPERATOR_API_REQUEST_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        gt=0,
        le=600,
        description="Total timeout in seconds for one operator-API summary request.",
    )
    OPERATOR_API_INITIAL_BACKOFF_SECONDS: float = Field(
        default=1.0,
        ge=0,
        le=60,
        description="Initial backoff seconds after a transient operator-API failure.",
    )
    OPERATOR_API_BACKOFF_MULTIPLIER: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Operator-API retry backoff multiplier.",
    )
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
    SUMMARY_RACE_REFRESH_ATTEMPTS: int = Field(
        default=15,
        ge=0,
        le=200,
        description="How many times the sweep-controller re-reads a terminal "
        "child AIPerfJob whose ``status.summary`` AND ``status.runEpoch`` are "
        "both still unset before giving up on its metrics. The operator's "
        "completion handler stamps both fields from a code path that is not "
        "atomic with the phase write, so a fast child (concurrency=1, few "
        "requests) routinely reaches Completed first. This window must cover "
        "the whole completion handler — results fetch + retries, disk "
        "recovery, JobSet delete, retention pass — because the operator-API "
        "fallback needs ``status.runEpoch`` and short-circuits without it. "
        "Exhausting the window collapses that variation's SLA bracket to "
        "``observed: null``, so err long: the loop exits the instant either "
        "field lands, and only a genuinely stuck completion pays the full "
        "wait. Set 0 to disable the settle loop entirely.",
    )
    SUMMARY_RACE_REFRESH_SECONDS: float = Field(
        default=2.0,
        gt=0,
        le=60,
        description="Delay between the child re-reads controlled by "
        "SUMMARY_RACE_REFRESH_ATTEMPTS. Attempts x this delay is the total "
        "grace granted to the operator's summary/runEpoch write.",
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

    @model_validator(mode="after")
    def validate_child_poll_deadlines(self) -> Self:
        """Require terminal polling to sample each bounded child wait at least once."""
        if self.CHILD_POLL_INTERVAL_SECONDS > self.CANCEL_GRACE_SECONDS:
            raise ValueError(
                "CHILD_POLL_INTERVAL_SECONDS must be <= CANCEL_GRACE_SECONDS"
            )
        if self.CHILD_POLL_INTERVAL_SECONDS > self.CHILD_MISSING_TIMEOUT_SECONDS:
            raise ValueError(
                "CHILD_POLL_INTERVAL_SECONDS must be <= CHILD_MISSING_TIMEOUT_SECONDS"
            )
        return self


class _ReconcileSettings(BaseSettings):
    """Retry delays for kopf reconciliation categories."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_OPERATOR_RECONCILE_")

    CONFLICT_RETRY_DELAY_SECONDS: float = Field(
        default=1.0,
        ge=0,
        le=300,
        description="Delay before rebasing status after an optimistic-write conflict.",
    )
    RUNS_CAS_MAX_ATTEMPTS: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum resourceVersion CAS attempts when appending status.runs.",
    )
    EVENT_RETRY_DELAY_SECONDS: float = Field(
        default=5.0,
        ge=0,
        le=300,
        description="Delay before retrying a watch-event read or status write.",
    )
    PERSISTENCE_RETRY_DELAY_SECONDS: float = Field(
        default=10.0,
        ge=0,
        le=300,
        description="Delay before retrying transient monitor or durable-state failures.",
    )
    STATE_RETRY_DELAY_SECONDS: float = Field(
        default=15.0,
        ge=0,
        le=300,
        description="Delay before retrying identity-fenced state reconciliation.",
    )
    CREATE_HARVEST_RETRY_DELAY_SECONDS: float = Field(
        default=30.0,
        ge=0,
        le=600,
        description="Delay before retrying resource creation or sweep-result harvest.",
    )
    TTL_DELETE_RETRY_DELAY_SECONDS: float = Field(
        default=60.0,
        ge=0,
        le=3600,
        description="Delay before retrying an expired AIPerfSweep deletion.",
    )


class _OperatorServiceSettings(BaseSettings):
    """Operator-service network identity.

    The operator Pod has three containers but only ONE FastAPI app: the
    ``results-server`` sidecar on ``resultsServer.port`` (8081 in the
    chart) hosts every ``/api/v1/*`` router (jobs, sweeps, results,
    config, admin, analytics, dashboard_proxy). The ``operator`` container
    on port 8080 runs kopf only — its sole HTTP surface there is
    ``/healthz``, with Prometheus ``/metrics`` on a separate server bound
    to ``METRICS_PORT`` (9090 in the chart). So there is no separate
    "sweeps API URL" and "results API URL" — one base URL, pointing
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
    PROXY_MAX_BODY_BYTES: int = Field(
        default=33_554_432,
        ge=1024,
        le=4_294_967_296,
        description="Maximum request-body size in bytes the results-server will "
        "buffer before forwarding it to the dashboard sidecar. Requests larger "
        "than this are rejected with HTTP 413 without contacting the sidecar. "
        "The proxy has to buffer the body to re-frame it for the upstream "
        "request, so an unbounded POST would be resident in the results-server "
        "sidecar's memory all at once. Dashboard traffic is SPA fetches and "
        "small refresh POSTs, so 32 MiB is far above any legitimate request; "
        "raise it via AIPERF_DASHBOARD_PROXY_MAX_BODY_BYTES if a future "
        "dashboard route accepts genuinely large uploads.",
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
    ENDPOINT_CHECK_DNS_RETRIES: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts when endpoint DNS resolution fails transiently",
    )
    ENDPOINT_CHECK_DNS_RETRY_DELAY: float = Field(
        default=5.0,
        ge=0.5,
        le=60.0,
        description="Seconds to wait between DNS retry attempts for endpoint health check",
    )
    PREFLIGHT_TIMEOUT: float = Field(
        default=30.0,
        gt=0,
        le=120,
        description="Seconds to wait for all pre-flight checks to complete",
    )
    CLIENT_CACHE_MAX_ENTRIES: int = Field(
        default=200,
        ge=1,
        le=100000,
        description="Upper bound on each process-wide kopf handler cache in "
        "``aiperf.operator.client_cache`` (cached ProgressClients, unset "
        "cancellation events, latched completion-claim timestamps). Eviction "
        "is FIFO/LRU per cache and is loss-tolerant by construction: a "
        "ProgressClient is re-created on demand, a SET cancellation flag is "
        "never evicted, and a claim timestamp falls back to the durable "
        "COMPLETION_CLAIMED annotation on the CR. Raise it on operators that "
        "reconcile more than this many AIPerfJobs concurrently.",
    )
    COMPLETION_CLAIM_TRUST_WINDOW_SECONDS: float = Field(
        default=900.0,
        gt=0,
        description="How long (seconds, measured from the claim timestamp) the "
        "``aiperf.nvidia.com/completion-claimed`` annotation may suppress the "
        "``spec.timeoutSeconds`` FAILED stamp and the 'JobSet not found' FAILED "
        "stamp in the monitor. The annotation lives on CR metadata, which any "
        "AIPerfJob editor can write, so trusting it without bound would let a "
        "forged or orphaned value disable terminal-phase enforcement forever. "
        "Deliberately NOT derived from ``spec.timeoutSeconds``: the window has "
        "to cover post-benchmark result draining (fetch + retries + retention), "
        "which is unrelated to — and routinely longer than — a short benchmark "
        "deadline, and a claim is only ever stamped after completion evidence. "
        "Crash-after-claim converges through orphan-claim recovery on the next "
        "monitor tick rather than through this window.",
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
        validation_alias=AliasChoices(
            "AIPERF_OPERATOR_MUTATING_ROUTES_ENABLED",
            "AIPERF_MUTATING_ROUTES_ENABLED",
        ),
        description="Enable results-server HTTP routes that mutate Kubernetes state. "
        "Defaults false so read-only results APIs remain exposed while POST routes "
        "fail closed unless an operator explicitly opts in.",
    )
    MUTATING_ROUTES_TOKEN: str = Field(
        default="",
        validation_alias=AliasChoices(
            "AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN",
            "AIPERF_MUTATING_ROUTES_TOKEN",
        ),
        description="Bearer token required by enabled results-server mutating routes. "
        "Leave empty to fail closed even when MUTATING_ROUTES_ENABLED is true.",
    )

    CLUSTER_NAME: str = Field(
        default="",
        description="Optional human-readable cluster name surfaced in the UI top banner "
        "(e.g. 'dgx-prod', 'kind-aiperf'). When unset the banner falls back to the "
        "Kubernetes server version. Set via AIPERF_CLUSTER_NAME on the "
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
        description="Sweep-controller pod settings (child lifecycle and API retries).",
    )
    RECONCILE: _ReconcileSettings = Field(
        default_factory=_ReconcileSettings,
        description="Kopf reconciliation retry delays grouped by operation semantics.",
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
