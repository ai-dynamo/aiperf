# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator deployment and AIPerfJob CR management for Kubernetes E2E tests."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import (
    JobSetStatus,
    KubectlClient,
    PodStatus,
    background_status,
)
from tests.kubernetes.helpers.log_streamer import PodLogStreamer
from tests.kubernetes.helpers.watchdog import BenchmarkWatchdog, make_watchdog_source

logger = AIPerfLogger(__name__)


AIPERF_CRD_NAMES: tuple[str, ...] = (
    "aiperfjobs.aiperf.nvidia.com",
    "aiperfsweeps.aiperf.nvidia.com",
)
"""All AIPerf CRDs the chart installs. Both must be Established before any
CR (AIPerfJob or AIPerfSweep) is applied — kubectl apply returns when the
apiserver accepts the CRD object, not when ``Established=True`` is set."""


async def wait_for_aiperf_crds_established(
    kubectl: KubectlClient, timeout: int = 60
) -> None:
    """Poll until every AIPerf CRD reports ``Established=True``.

    kubectl apply returns after the apiserver accepts the CRD object,
    not after ``Established`` is set. CRs created on the heels of apply
    can race the apiserver's CRD registration and fail with
    ``no matches for kind``. This helper closes that gap, covering both
    AIPerfJob and AIPerfSweep CRDs.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    pending = list(AIPERF_CRD_NAMES)
    while pending and asyncio.get_event_loop().time() < deadline:
        still_pending: list[str] = []
        for crd_name in pending:
            result = await kubectl.run(
                "get",
                "crd",
                crd_name,
                "-o",
                "jsonpath={.status.conditions[?(@.type=='Established')].status}",
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip() == "True":
                logger.info(f"CRD {crd_name} established")
                continue
            still_pending.append(crd_name)
        pending = still_pending
        if pending:
            await asyncio.sleep(1)
    if pending:
        raise TimeoutError(
            f"CRDs not Established within {timeout}s: {', '.join(pending)}"
        )


@dataclass
class AIPerfJobConfig:
    """Configuration for an AIPerfJob CR."""

    endpoint_url: str = "http://aiperf-mock-server.default.svc.cluster.local:8000/v1"
    """Inference server endpoint URL."""

    model_name: str = "mock-model"
    """Model name to benchmark."""

    endpoint_type: str = "chat"
    """Endpoint type (chat, completions, embeddings)."""

    concurrency: int = 5
    """Number of concurrent requests."""

    request_count: int | None = 50
    """Total requests to send, or None for duration-based."""

    warmup_request_count: int = 5
    """Number of warmup requests before measurement."""

    benchmark_duration: float | None = None
    """Duration in seconds for time-based benchmarks, or None for count-based."""

    tokenizer_name: str = "gpt2"
    """Tokenizer name for token counting."""

    image: str = "aiperf:local"
    """Container image for benchmark pods."""

    image_pull_policy: str = "Never"
    """Image pull policy for benchmark pods."""

    connections_per_worker: int | None = None
    """Override for connections per worker, or None for default."""

    queue_name: str | None = None
    """Kueue LocalQueue name for gang-scheduling, or None."""

    priority_class: str | None = None
    """Kubernetes PriorityClass name, or None."""

    num_conversations: int | None = None
    """Number of unique synthetic conversations (dataset entries). None defaults
    to ``max(request_count, 10)``. Must match the bare-side ``--num-conversations``
    for audit parity, since this caps total request count when smaller than
    ``request_count``.
    """

    random_seed: int | None = None
    """Global random seed for dataset/sampling determinism. Maps to
    ``BenchmarkConfig.random_seed`` so the operator-side run produces the same
    seeded prompts as a ``--random-seed`` bare invocation."""

    def to_flat_spec(self) -> dict[str, Any]:
        """Generate flat CRD spec (config v3 format, no userConfig wrapper).

        Emits ``phases:`` as an ordered array. The CRD's apiserver schema
        explicitly requires ``spec.benchmark.phases`` (see ``required:`` in
        ``deploy/helm/aiperf-operator/templates/crd-aiperfjob.yaml``); the ``profiling:``
        / ``warmup:`` top-level shorthand siblings are normalized into
        ``phases:`` only by an operator Pydantic before-validator that runs
        AFTER apiserver validation, so they cannot replace ``phases:`` from
        the client side. Order in the list IS execution order: warmup (when
        configured) precedes profiling.
        """
        profiling_phase: dict[str, Any] = {
            "name": "profiling",
            "type": "concurrency",
            "concurrency": self.concurrency,
        }
        if self.request_count is not None:
            profiling_phase["requests"] = self.request_count
        if self.benchmark_duration is not None:
            profiling_phase["duration"] = self.benchmark_duration
        if self.num_conversations is not None:
            # Local CLI's --num-conversations N maps to phases[].sessions=N,
            # which caps total requests in the phase. Without this, the phase
            # runs the full request_count regardless of dataset size, and the
            # audit's bare/operator counts diverge.
            profiling_phase["sessions"] = self.num_conversations

        phases: list[dict[str, Any]] = []
        if self.warmup_request_count:
            phases.append(
                {
                    "name": "warmup",
                    "kind": "warmup",
                    "type": "concurrency",
                    "concurrency": self.concurrency,
                    "requests": self.warmup_request_count,
                }
            )
        phases.append(profiling_phase)

        return {
            "models": {"items": [{"name": self.model_name}]},
            "endpoint": {"urls": [self.endpoint_url]},
            "datasets": [
                {
                    "name": "main",
                    "type": "synthetic",
                    "entries": (
                        self.num_conversations
                        if self.num_conversations is not None
                        else max(self.request_count or 100, 10)
                    ),
                    "prompts": {"isl": {"mean": 550}},
                },
            ],
            "phases": phases,
            "tokenizer": {"name": self.tokenizer_name},
            "runtime": {"ui": "none"},
        }

    def to_cr_manifest(self, name: str, namespace: str) -> str:
        """Generate AIPerfJob CR manifest (flat spec, no userConfig wrapper).

        Args:
            name: CR name.
            namespace: Namespace for the CR.

        Returns:
            YAML manifest string.
        """
        spec: dict[str, Any] = {
            "image": self.image,
            "imagePullPolicy": self.image_pull_policy,
            "benchmark": self.to_flat_spec(),
        }

        # randomSeed sits on the AIPerfConfig envelope (spec.randomSeed), not
        # inside the BenchmarkConfig body -- BenchmarkConfig explicitly excludes
        # the cross-variation fields (sweep, multi_run, variables, random_seed).
        # Emitting it under spec.benchmark trips the CRD's strict decoding.
        if self.random_seed is not None:
            spec["randomSeed"] = self.random_seed

        if self.connections_per_worker is not None:
            spec["connectionsPerWorker"] = self.connections_per_worker

        if self.queue_name is not None or self.priority_class is not None:
            scheduling: dict[str, str] = {}
            if self.queue_name is not None:
                scheduling["queueName"] = self.queue_name
            if self.priority_class is not None:
                scheduling["priorityClass"] = self.priority_class
            spec["scheduling"] = scheduling

        cr = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "metadata": {
                "name": name,
                "namespace": namespace,
            },
            "spec": spec,
        }

        return yaml.dump(cr, default_flow_style=False)


@dataclass
class AIPerfJobStatus:
    """Status of an AIPerfJob CR."""

    name: str
    """AIPerfJob CR name."""

    namespace: str
    """Namespace the CR belongs to."""

    phase: str | None = None
    """Top-level job phase (Pending, Running, Completed, Failed, Cancelled)."""

    current_phase: str | None = None
    """Current benchmark phase (warmup, profiling)."""

    job_id: str | None = None
    """Unique job identifier assigned by the operator."""

    jobset_name: str | None = None
    """Name of the associated JobSet resource."""

    error: str | None = None
    """Error message if the job failed."""

    workers: dict[str, int] | None = None
    """Worker counts with ready and total keys."""

    conditions: list[dict[str, Any]] = field(default_factory=list)
    """Kubernetes-style status conditions."""

    phases: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-phase status information."""

    results: dict[str, Any] | None = None
    """Benchmark results stored in the CR status."""

    results_path: str | None = None
    """Path where results are stored on the PVC."""

    live_metrics: dict[str, Any] | None = None
    """Live metrics snapshot from the running benchmark."""

    raw_status: dict[str, Any] = field(default_factory=dict)
    """Full raw status dict from the CR."""

    @property
    def is_pending(self) -> bool:
        """Check if job is pending."""
        return self.phase == "Pending"

    @property
    def is_running(self) -> bool:
        """Check if job is running."""
        return self.phase == "Running"

    @property
    def is_completed(self) -> bool:
        """Check if job completed successfully."""
        return self.phase == "Completed"

    @property
    def is_failed(self) -> bool:
        """Check if job failed."""
        return self.phase == "Failed"

    @property
    def is_cancelled(self) -> bool:
        """Check if job was cancelled."""
        return self.phase == "Cancelled"

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.phase in ("Completed", "Failed", "Cancelled")

    @property
    def workers_ready(self) -> int:
        """Get number of ready workers."""
        if self.workers:
            return self.workers.get("ready", 0)
        return 0

    @property
    def workers_total(self) -> int:
        """Get total number of workers."""
        if self.workers:
            return self.workers.get("total", 0)
        return 0

    def get_condition(self, condition_type: str) -> dict[str, Any] | None:
        """Get a specific condition by type."""
        for cond in self.conditions:
            if cond.get("type") == condition_type:
                return cond
        return None

    def is_condition_true(self, condition_type: str) -> bool:
        """Check if a condition is True."""
        cond = self.get_condition(condition_type)
        return cond is not None and cond.get("status") == "True"

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AIPerfJobStatus:
        """Create from kubectl JSON output."""
        metadata = data.get("metadata", {})
        status = data.get("status", {})

        return cls(
            name=metadata.get("name", ""),
            namespace=metadata.get("namespace", ""),
            phase=status.get("phase"),
            current_phase=status.get("currentPhase"),
            job_id=status.get("jobId"),
            jobset_name=status.get("jobSetName"),
            error=status.get("error"),
            workers=status.get("workers"),
            conditions=status.get("conditions", []),
            phases=status.get("phases", {}),
            results=status.get("results"),
            results_path=status.get("resultsPath"),
            live_metrics=status.get("liveMetrics"),
            raw_status=status,
        )


@dataclass
class OperatorJobResult:
    """Result of an operator-managed benchmark job."""

    namespace: str
    """Kubernetes namespace for this job."""

    job_name: str
    """AIPerfJob CR name."""

    config: AIPerfJobConfig
    """Configuration used for this job."""

    status: AIPerfJobStatus | None = None
    """Final AIPerfJob CR status, or None if not collected."""

    jobset_status: JobSetStatus | None = None
    """Final JobSet status, or None if not collected."""

    pods: list[PodStatus] = field(default_factory=list)
    """Pod statuses at collection time."""

    duration_seconds: float = 0.0
    """Total wall-clock duration in seconds."""

    success: bool = False
    """Whether the job completed successfully."""

    error_message: str | None = None
    """Error message if the job failed."""

    events: list[str] = field(default_factory=list)
    """Kubernetes events related to this job."""

    @property
    def controller_pod(self) -> PodStatus | None:
        """Get the controller pod."""
        for pod in self.pods:
            if "controller" in pod.name:
                return pod
        return None


class OperatorDeployer:
    """Manages operator deployment and AIPerfJob lifecycle."""

    OPERATOR_NAMESPACE = "aiperf-system"
    CRD_NAME = "aiperfjobs.aiperf.nvidia.com"

    def __init__(
        self,
        kubectl: KubectlClient,
        project_root: Path,
        operator_image: str = "aiperf:local",
        default_job_namespace: str = "default",
        share_process_namespace: bool = False,
        controller_http_url_override: str | None = None,
        apiserver_service_host_override: str | None = None,
        apiserver_service_port_override: str | None = None,
        apiserver_tls_server_name_override: str | None = None,
        image_pull_policy: str = "Never",
        image_pull_secret: str | None = None,
        operator_node_selector: dict[str, str] | None = None,
        disable_pvc: bool = False,
    ) -> None:
        """Initialize operator deployer.

        Args:
            kubectl: Kubectl client.
            project_root: Path to project root.
            operator_image: Operator image name.
            default_job_namespace: Namespace for spawned AIPerfJob resources.
            share_process_namespace: When True, configures the operator with
                ``AIPERF_K8S_SHARE_PROCESS_NAMESPACE=true`` so every JobSet pod
                it spawns sets ``spec.shareProcessNamespace=true``. Chaos
                scenarios flip this on to enable cross-container kubectl exec
                kills; keep off for normal e2e coverage.
            controller_http_url_override: When set, configures the operator
                with ``AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE=<url>`` so
                every controller HTTP call routes through this URL instead of
                per-CR JobSet pod DNS. Used by chaos scenario C16 to front
                operator -> controller traffic with toxiproxy. NEVER use in
                production-shaped tests; it collapses multi-job isolation.
            apiserver_service_host_override: Optional override for the in-cluster
                ``KUBERNETES_SERVICE_HOST`` env var. Used by chaos scenario C15
                to route operator -> apiserver traffic through toxiproxy.
            apiserver_service_port_override: Optional override for the in-cluster
                ``KUBERNETES_SERVICE_PORT`` env var paired with the host override.
            apiserver_tls_server_name_override: Optional TLS hostname used when
                verifying the apiserver certificate while dialing a proxy host.
            image_pull_policy: Kubernetes imagePullPolicy for the operator pod.
                Use ``"Never"`` for local Kind clusters (image pre-loaded via
                ``kind load``). Use ``"IfNotPresent"`` when targeting a real
                registry-backed cluster (e.g. an external DGX cluster).
            image_pull_secret: Optional name of a Kubernetes imagePullSecret in
                the operator namespace. Required when ``image_pull_policy`` is
                not ``"Never"`` and the registry requires authentication.
            operator_node_selector: Optional node selector labels for the operator
                pod itself. Use to pin the operator to CPU nodes on clusters where
                GPU nodes use storage classes (e.g. pd-balanced) that are
                incompatible with the operator's PVC, or to pin the operator to
                specific node pools.
            disable_pvc: When True, sets ``storage.enabled=false`` in the Helm
                chart so the operator uses an ``emptyDir`` instead of a
                PersistentVolumeClaim. Useful on external clusters where the
                default storage class cannot attach to the node the operator
                lands on, or on ephemeral test clusters without a StorageClass.
        """
        self.kubectl = kubectl
        self.project_root = project_root
        self.operator_image = operator_image
        self.default_job_namespace = default_job_namespace
        self.share_process_namespace = share_process_namespace
        self.controller_http_url_override = controller_http_url_override
        self.apiserver_service_host_override = apiserver_service_host_override
        self.apiserver_service_port_override = apiserver_service_port_override
        self.apiserver_tls_server_name_override = apiserver_tls_server_name_override
        self.image_pull_policy = image_pull_policy
        self.image_pull_secret = image_pull_secret
        self.operator_node_selector = operator_node_selector
        self.disable_pvc = disable_pvc
        self._deployed_jobs: list[OperatorJobResult] = []

    async def install_crd(self) -> None:
        """Install the AIPerfJob and AIPerfSweep CRDs by rendering them from the Helm chart."""
        chart_path = self.project_root / "deploy" / "helm" / "aiperf-operator"
        logger.info(
            f"Installing AIPerfJob and AIPerfSweep CRDs from chart {chart_path}"
        )

        result = subprocess.run(
            [
                "helm",
                "template",
                "aiperf-operator",
                str(chart_path),
                "--show-only",
                "templates/crd-aiperfjob.yaml",
                "--show-only",
                "templates/crd-aiperfsweep.yaml",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        await self.kubectl.apply(result.stdout)

        await self._wait_for_crd_established()

    async def _wait_for_crd_established(self, timeout: int = 60) -> None:
        """Wait for both AIPerf CRDs (AIPerfJob, AIPerfSweep) to be Established."""
        await wait_for_aiperf_crds_established(self.kubectl, timeout=timeout)

    async def is_operator_healthy(self) -> bool:
        """Return True if the operator Deployment + RBAC are fully present.

        Checks both the Deployment's ready replicas AND the cluster-scoped
        RBAC resources (ClusterRole + ClusterRoleBinding). A fresh helm
        release or manual cleanup can leave the Deployment ready while
        wiping RBAC out from under it, so we need the full picture.
        """
        dep = await self.kubectl.run(
            "get",
            "deployment",
            "aiperf-operator",
            "-n",
            self.OPERATOR_NAMESPACE,
            "-o",
            "jsonpath={.status.readyReplicas}",
            check=False,
        )
        if dep.returncode != 0:
            return False
        ready = dep.stdout.strip()
        if not (bool(ready) and ready != "0"):
            return False
        for resource in ("clusterrole", "clusterrolebinding"):
            result = await self.kubectl.run(
                "get", resource, "aiperf-operator", check=False
            )
            if result.returncode != 0:
                return False
        return True

    async def deploy_operator(self) -> None:
        """Deploy the operator to the cluster.

        Renders the Helm chart via `helm template` and applies it with kubectl.
        Handles idempotency: if an existing deployment has incompatible labels
        (e.g. from a helm install), it is deleted first. Any existing helm
        release is also uninstalled to avoid conflicts.
        """
        chart_path = self.project_root / "deploy" / "helm" / "aiperf-operator"
        logger.info(f"Deploying operator from chart {chart_path}")

        # Clean up any existing helm release that could conflict
        await self._cleanup_existing_operator()

        # Ensure the operator namespace exists
        await self.kubectl.run(
            "create",
            "namespace",
            self.OPERATOR_NAMESPACE,
            check=False,
        )

        # Strip Helm ownership annotations from the namespace to avoid
        # conflicts when re-deploying after a helm uninstall test.
        await self.kubectl.run(
            "annotate",
            "namespace",
            self.OPERATOR_NAMESPACE,
            "meta.helm.sh/release-name-",
            "meta.helm.sh/release-namespace-",
            check=False,
        )
        await self.kubectl.run(
            "label",
            "namespace",
            self.OPERATOR_NAMESPACE,
            "app.kubernetes.io/managed-by-",
            check=False,
        )

        # Render the Helm chart to a manifest
        if self.image_pull_secret:
            await self._ensure_pull_secret_in_operator_ns(self.image_pull_secret)

        helm_cmd = [
            "helm",
            "template",
            "aiperf-operator",
            str(chart_path),
            "-n",
            self.OPERATOR_NAMESPACE,
            "--set",
            f"image.repository={self.operator_image.rsplit(':', 1)[0]}",
            "--set",
            f"image.tag={self.operator_image.rsplit(':', 1)[-1]}",
            "--set",
            f"image.pullPolicy={self.image_pull_policy}",
        ]
        if self.image_pull_secret:
            helm_cmd += ["--set", f"imagePullSecrets[0].name={self.image_pull_secret}"]
        if self.disable_pvc:
            helm_cmd += ["--set", "storage.enabled=false"]
        for key, value in (self.operator_node_selector or {}).items():
            # Dots in label keys (e.g. "kubernetes.io/arch") must be escaped
            # for helm --set so they are not interpreted as nested keys.
            escaped_key = key.replace(".", "\\.")
            helm_cmd += ["--set", f"operator.nodeSelector.{escaped_key}={value}"]
        result = subprocess.run(
            helm_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        manifest = result.stdout

        await self.kubectl.apply(manifest)

        # kubectl apply returns when the apiserver accepts the CRD object,
        # not when Established=True is set. CRs (AIPerfJob/AIPerfSweep)
        # applied on the heels of deploy_operator can race CRD registration
        # and fail with 'no matches for kind'. Wait for both CRDs.
        await self._wait_for_crd_established()

        # Use defaults that match production; Kind node has enough memory
        env_pairs = [
            "AIPERF_K8S_WORKER_POD_CPU=3350m",
            "AIPERF_K8S_WORKER_POD_MEMORY=6144Mi",
            "AIPERF_K8S_SYSTEM_CONTROLLER_CPU=250m",
            "AIPERF_K8S_SYSTEM_CONTROLLER_MEMORY=256Mi",
            "AIPERF_K8S_TIMING_MANAGER_CPU=1000m",
            "AIPERF_K8S_TIMING_MANAGER_MEMORY=512Mi",
            "AIPERF_K8S_DATASET_MANAGER_CPU=500m",
            "AIPERF_K8S_DATASET_MANAGER_MEMORY=512Mi",
            "AIPERF_K8S_RECORDS_MANAGER_CPU=500m",
            "AIPERF_K8S_RECORDS_MANAGER_MEMORY=512Mi",
            "AIPERF_K8S_API_CPU=250m",
            "AIPERF_K8S_API_MEMORY=256Mi",
            "AIPERF_K8S_GPU_TELEMETRY_MANAGER_CPU=125m",
            "AIPERF_K8S_GPU_TELEMETRY_MANAGER_MEMORY=128Mi",
            "AIPERF_K8S_SERVER_METRICS_MANAGER_CPU=125m",
            "AIPERF_K8S_SERVER_METRICS_MANAGER_MEMORY=256Mi",
            "AIPERF_K8S_RESULTS_SIDECAR_CPU=100m",
            "AIPERF_K8S_RESULTS_SIDECAR_MEMORY=128Mi",
        ]
        if self.share_process_namespace:
            env_pairs.append("AIPERF_K8S_SHARE_PROCESS_NAMESPACE=true")
        if self.controller_http_url_override:
            env_pairs.append(
                f"AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE={self.controller_http_url_override}"
            )
        if self.apiserver_service_host_override:
            env_pairs.append(
                f"KUBERNETES_SERVICE_HOST={self.apiserver_service_host_override}"
            )
        if self.apiserver_service_port_override:
            env_pairs.append(
                f"KUBERNETES_SERVICE_PORT={self.apiserver_service_port_override}"
            )
        if self.apiserver_tls_server_name_override:
            env_pairs.append(
                "AIPERF_K8S_APISERVER_TLS_SERVER_NAME_OVERRIDE="
                f"{self.apiserver_tls_server_name_override}"
            )
        await self.kubectl.run(
            "set",
            "env",
            "deployment/aiperf-operator",
            *env_pairs,
            "-n",
            self.OPERATOR_NAMESPACE,
            check=True,
        )

        success = await self.kubectl.wait_for_rollout(
            "deployment",
            "aiperf-operator",
            namespace=self.OPERATOR_NAMESPACE,
            timeout=180,
        )

        if not success:
            logs = await self.kubectl.get_logs(
                "deployment/aiperf-operator",
                namespace=self.OPERATOR_NAMESPACE,
            )
            raise RuntimeError(f"Operator deployment failed. Logs:\n{logs}")

        logger.info("Operator deployed and ready")

    async def _ensure_pull_secret_in_operator_ns(self, secret_name: str) -> None:
        """Copy an imagePullSecret into the operator namespace if not already present.

        When targeting a real registry-backed cluster, the pull secret may
        exist only in a user-owned namespace. This searches all accessible
        namespaces for the secret and copies it to the operator namespace so
        the operator pod can pull its image.
        """
        import re

        existing = await self.kubectl.run(
            "get",
            "secret",
            secret_name,
            "-n",
            self.OPERATOR_NAMESPACE,
            check=False,
        )
        if existing.returncode == 0:
            logger.info(
                f"Pull secret {secret_name!r} already in {self.OPERATOR_NAMESPACE}"
            )
            return

        ns_list = await self.kubectl.run(
            "get",
            "namespaces",
            "-o",
            "jsonpath={.items[*].metadata.name}",
            check=False,
        )
        if ns_list.returncode != 0:
            logger.warning(
                f"Cannot list namespaces; pull secret {secret_name!r} may be missing "
                f"in {self.OPERATOR_NAMESPACE}"
            )
            return

        for ns in ns_list.stdout.strip().split():
            if ns == self.OPERATOR_NAMESPACE:
                continue
            fetch = await self.kubectl.run(
                "get", "secret", secret_name, "-n", ns, "-o", "yaml", check=False
            )
            if fetch.returncode != 0:
                continue
            # Strip per-resource metadata so kubectl apply creates a fresh copy.
            raw = fetch.stdout
            raw = re.sub(r"\n\s+namespace:.*", "", raw)
            raw = re.sub(r"\n\s+resourceVersion:.*", "", raw)
            raw = re.sub(r"\n\s+uid:.*", "", raw)
            raw = re.sub(r"\n\s+creationTimestamp:.*", "", raw)
            logger.info(
                f"Copying pull secret {secret_name!r} from {ns!r} → {self.OPERATOR_NAMESPACE!r}"
            )
            try:
                await self.kubectl.apply(raw, namespace=self.OPERATOR_NAMESPACE)
                return
            except RuntimeError as exc:
                logger.warning(f"Failed to copy pull secret from {ns!r}: {exc}")

        logger.warning(
            f"Pull secret {secret_name!r} not found in any namespace; image pull may fail"
        )

    async def _cleanup_existing_operator(self) -> None:
        """Remove any existing operator deployment that could conflict.

        Handles both helm-installed and directly-applied operators. This ensures
        idempotent deployment regardless of prior cluster state.
        """
        ctx_args = []
        if self.kubectl.context:
            ctx_args = ["--kube-context", self.kubectl.context]

        # Uninstall any helm release first
        helm_list = await self._run_cmd(
            "helm",
            *ctx_args,
            "list",
            "-n",
            self.OPERATOR_NAMESPACE,
            "-q",
            "--filter",
            "aiperf",
        )
        if helm_list.returncode == 0 and helm_list.stdout.strip():
            for release in helm_list.stdout.strip().split("\n"):
                release = release.strip()
                if release:
                    logger.info(f"Uninstalling existing helm release: {release}")
                    await self._run_cmd(
                        "helm",
                        *ctx_args,
                        "uninstall",
                        release,
                        "-n",
                        self.OPERATOR_NAMESPACE,
                    )
                    await asyncio.sleep(5)

        # Delete all resources in the operator namespace for a clean slate
        await self.kubectl.run(
            "delete",
            "all,sa,roles,rolebindings",
            "--all",
            "-n",
            self.OPERATOR_NAMESPACE,
            check=False,
        )
        # Force-delete PVCs (may have finalizers that block normal deletion)
        await self.kubectl.run(
            "delete",
            "pvc",
            "--all",
            "-n",
            self.OPERATOR_NAMESPACE,
            "--force",
            "--grace-period=0",
            check=False,
        )
        # Clean up cluster-scoped resources too
        for resource in ["clusterrole", "clusterrolebinding"]:
            await self.kubectl.run(
                "delete",
                resource,
                "aiperf-operator",
                check=False,
            )

        # Delete existing deployment if present (handles label selector conflicts)
        existing = await self.kubectl.run(
            "get",
            "deployment",
            "aiperf-operator",
            "-n",
            self.OPERATOR_NAMESPACE,
            "-o",
            "name",
            check=False,
        )
        if existing.returncode == 0 and existing.stdout.strip():
            logger.info("Deleting existing operator deployment for clean redeploy")
            await self.kubectl.delete(
                "deployment",
                "aiperf-operator",
                namespace=self.OPERATOR_NAMESPACE,
                ignore_not_found=True,
            )
            await asyncio.sleep(3)

    @staticmethod
    async def _run_cmd(*args: str) -> subprocess.CompletedProcess:
        """Run an arbitrary command and return CompletedProcess."""
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return subprocess.CompletedProcess(
            args,
            proc.returncode,
            stdout.decode() if stdout else "",
            stderr.decode() if stderr else "",
        )

    async def uninstall_operator(self, timeout: int = 90) -> None:
        """Uninstall the operator with a timeout to prevent fixture hangs."""
        logger.info("Uninstalling operator")
        try:
            await asyncio.wait_for(self._do_uninstall(), timeout=timeout)
        except TimeoutError:
            logger.warning(f"Operator uninstall timed out after {timeout}s")

    async def _do_uninstall(self) -> None:
        """Inner uninstall logic."""
        await self._cleanup_existing_operator()

        await self.kubectl.delete(
            "deployment",
            "aiperf-operator",
            namespace=self.OPERATOR_NAMESPACE,
            ignore_not_found=True,
        )
        await self.kubectl.delete(
            "namespace", self.OPERATOR_NAMESPACE, ignore_not_found=True
        )

    async def create_job(
        self,
        config: AIPerfJobConfig,
        name: str | None = None,
        namespace: str | None = None,
    ) -> OperatorJobResult:
        """Create an AIPerfJob CR.

        Args:
            config: Job configuration.
            name: Job name (auto-generated if not provided).
            namespace: Target namespace.

        Returns:
            OperatorJobResult with initial state.
        """
        import uuid

        if name is None:
            name = f"benchmark-{uuid.uuid4().hex[:8]}"
        if namespace is None:
            namespace = self.default_job_namespace

        manifest = config.to_cr_manifest(name, namespace)
        logger.info(f"Creating AIPerfJob {namespace}/{name}")

        await self.kubectl.apply(manifest)

        result = OperatorJobResult(
            namespace=namespace,
            job_name=name,
            config=config,
        )
        self._deployed_jobs.append(result)

        return result

    async def get_job_status(self, name: str, namespace: str) -> AIPerfJobStatus:
        """Get current status of an AIPerfJob.

        Args:
            name: Job name.
            namespace: Namespace.

        Returns:
            AIPerfJobStatus with current state.
        """
        data = await self.kubectl.get_json("aiperfjob", name, namespace=namespace)
        return AIPerfJobStatus.from_json(data)

    async def wait_for_job_completion(
        self,
        name: str,
        namespace: str,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> AIPerfJobStatus:
        """Wait for an AIPerfJob to reach terminal state.

        Args:
            name: Job name.
            namespace: Namespace.
            timeout: Timeout in seconds.
            poll_interval: Polling interval in seconds.

        Returns:
            Final job status.

        Raises:
            TimeoutError: If timeout exceeded.
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed > timeout:
                status = await self.get_job_status(name, namespace)
                raise TimeoutError(
                    f"Timeout waiting for AIPerfJob {name} completion. "
                    f"Current phase: {status.phase}"
                )

            status = await self.get_job_status(name, namespace)

            if status.is_terminal:
                logger.info(f"AIPerfJob {name} reached terminal state: {status.phase}")
                return status

            logger.info(
                f"AIPerfJob {name}: phase={status.phase}, "
                f"current_phase={status.current_phase}, "
                f"workers={status.workers_ready}/{status.workers_total}, "
                f"elapsed={elapsed:.0f}s"
            )
            await asyncio.sleep(poll_interval)

    async def wait_for_phase(
        self,
        name: str,
        namespace: str,
        target_phase: str,
        timeout: int = 120,
        poll_interval: int = 2,
    ) -> AIPerfJobStatus:
        """Wait for an AIPerfJob to reach a specific phase.

        Args:
            name: Job name.
            namespace: Namespace.
            target_phase: Target phase to wait for.
            timeout: Timeout in seconds.
            poll_interval: Polling interval in seconds.

        Returns:
            Job status when phase is reached.

        Raises:
            TimeoutError: If timeout exceeded.
            RuntimeError: If job fails before reaching target phase.
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed > timeout:
                status = await self.get_job_status(name, namespace)
                raise TimeoutError(
                    f"Timeout waiting for AIPerfJob {name} to reach {target_phase}. "
                    f"Current phase: {status.phase}"
                )

            status = await self.get_job_status(name, namespace)

            if status.phase == target_phase:
                return status

            if status.is_failed:
                raise RuntimeError(
                    f"AIPerfJob {name} failed before reaching {target_phase}: "
                    f"{status.error}"
                )

            await asyncio.sleep(poll_interval)

    async def cancel_job(self, name: str, namespace: str) -> None:
        """Cancel an AIPerfJob by setting spec.cancel=true.

        Args:
            name: Job name.
            namespace: Namespace.
        """
        logger.info(f"Cancelling AIPerfJob {namespace}/{name}")
        await self.kubectl.run(
            "patch",
            "aiperfjob",
            name,
            "--type=merge",
            "-p",
            '{"spec":{"cancel":true}}',
            namespace=namespace,
        )

    async def delete_job(self, name: str, namespace: str) -> None:
        """Delete an AIPerfJob CR.

        Args:
            name: Job name.
            namespace: Namespace.
        """
        logger.info(f"Deleting AIPerfJob {namespace}/{name}")
        await self.kubectl.delete("aiperfjob", name, namespace=namespace)

    async def run_job(
        self,
        config: AIPerfJobConfig,
        name: str | None = None,
        namespace: str | None = None,
        timeout: int = 300,
        stream_logs: bool = False,
    ) -> OperatorJobResult:
        """Create a job and wait for completion.

        Args:
            config: Job configuration.
            name: Job name (auto-generated if not provided).
            namespace: Target namespace.
            timeout: Timeout in seconds.
            stream_logs: If True, stream pod logs in the background.

        Returns:
            OperatorJobResult with final state.
        """
        start_time = asyncio.get_event_loop().time()

        result = await self.create_job(config, name, namespace)
        name = result.job_name
        namespace = result.namespace

        async with (
            make_watchdog_source(self.kubectl) as watchdog_source,
            BenchmarkWatchdog(
                watchdog_source,
                namespace,
                timeout=timeout,
                poll_interval=5.0,
                pending_threshold=30.0,
            ) as _watchdog,
            PodLogStreamer(self.kubectl, namespace, prefix="OPERATOR") as streamer,
            background_status(self.kubectl, namespace, label="OPERATOR", interval=15),
        ):
            if stream_logs:
                streamer.watch()

            try:
                status = await self.wait_for_job_completion(name, namespace, timeout)
                result.status = status
                result.success = status.is_completed

                if status.is_failed:
                    result.error_message = status.error

            except TimeoutError as e:
                result.success = False
                result.error_message = str(e)
                result.status = await self.get_job_status(name, namespace)

        if result.status and result.status.jobset_name:
            with contextlib.suppress(Exception):
                result.jobset_status = await self.kubectl.get_jobset(
                    result.status.jobset_name, namespace
                )

            result.pods = await self.kubectl.get_pods(namespace)

        with contextlib.suppress(Exception):
            result.events = await self._get_job_events(name, namespace)

        result.duration_seconds = asyncio.get_event_loop().time() - start_time

        return result

    async def _get_job_events(self, name: str, namespace: str) -> list[str]:
        """Get events related to an AIPerfJob."""
        output = await self.kubectl.get_events(namespace)
        lines = []
        for line in output.splitlines():
            if name in line or "aiperfjob" in line.lower():
                lines.append(line)
        return lines

    async def get_operator_logs(self, tail: int = 100) -> str:
        """Get operator logs.

        Args:
            tail: Number of lines to tail.

        Returns:
            Log content.
        """
        return await self.kubectl.get_logs(
            "deployment/aiperf-operator",
            namespace=self.OPERATOR_NAMESPACE,
            tail=tail,
        )

    async def cleanup_job(self, result: OperatorJobResult) -> None:
        """Clean up a job and its resources.

        Removes the AIPerfJob finalizer, the JobSet, and force-deletes any
        zombie benchmark pods so the next session starts on a clean cluster.
        """
        ns = result.namespace
        name = result.job_name
        try:
            await self.kubectl.run(
                "patch",
                "aiperfjob",
                name,
                "-n",
                ns,
                "--type=json",
                '-p=[{"op":"remove","path":"/metadata/finalizers"}]',
                check=False,
            )
            await self.kubectl.run(
                "delete",
                "aiperfjob",
                name,
                "-n",
                ns,
                "--ignore-not-found",
                check=False,
            )
            await self.kubectl.run(
                "delete",
                "jobsets",
                "--all",
                "-n",
                ns,
                "--ignore-not-found",
                check=False,
            )
            pods = await self.kubectl.run(
                "get",
                "pods",
                "-n",
                ns,
                "-l",
                "jobset.sigs.k8s.io/jobset-name",
                "-o",
                "jsonpath={.items[*].metadata.name}",
                check=False,
            )
            if pods.returncode == 0 and pods.stdout.strip():
                for pod in pods.stdout.strip().split():
                    await self.kubectl.run(
                        "patch",
                        "pod",
                        pod,
                        "-n",
                        ns,
                        "--type=json",
                        '-p=[{"op":"remove","path":"/metadata/finalizers"}]',
                        check=False,
                    )
                await self.kubectl.run(
                    "delete",
                    "pods",
                    "-l",
                    "jobset.sigs.k8s.io/jobset-name",
                    "-n",
                    ns,
                    "--force",
                    "--grace-period=0",
                    "--ignore-not-found",
                    check=False,
                )
        except Exception as e:
            logger.warning(f"Failed to delete job {name}: {e}")

    async def cleanup_all(self) -> None:
        """Clean up all deployed jobs in parallel."""
        if self._deployed_jobs:
            await asyncio.gather(*[self.cleanup_job(r) for r in self._deployed_jobs])
        self._deployed_jobs.clear()
