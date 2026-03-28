# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JobSet specification generation for Kubernetes deployments.

This module generates JobSet YAML for deploying AIPerf as a distributed
benchmark across multiple pods. All resource and port settings are configurable
via environment variables through K8sEnvironment.
"""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from aiperf.common.environment import Environment
from aiperf.common.models import AIPerfBaseModel
from aiperf.config.deployment import PodTemplateConfig, SchedulingConfig
from aiperf.kubernetes.constants import Containers, KueueLabels, Labels
from aiperf.kubernetes.enums import ImagePullPolicy, RestartPolicy
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.utils import parse_cpu, parse_memory_mib


@dataclass(frozen=True, slots=True)
class JobSetAPIConfig:
    """JobSet API configuration constants."""

    group: str = "jobset.x-k8s.io"
    version: str = "v1alpha2"
    plural: str = "jobsets"

    @property
    def api_version(self) -> str:
        """Get the full apiVersion string for manifests."""
        return f"{self.group}/{self.version}"


# Shared JobSet API constants
JOBSET_API = JobSetAPIConfig()

# Known-good fallback version for JobSet CRD installation
JOBSET_FALLBACK_VERSION = "v0.5.2"
JOBSET_GITHUB_REPO = "kubernetes-sigs/jobset"


def get_jobset_manifest_url(version: str | None = None) -> str:
    """Build the JobSet manifest URL for a given version.

    Args:
        version: JobSet release tag (e.g. "v0.5.2"). If None, uses the fallback.

    Returns:
        URL to the JobSet manifests.yaml for kubectl apply.
    """
    v = version or JOBSET_FALLBACK_VERSION
    return (
        f"https://github.com/{JOBSET_GITHUB_REPO}/releases/download/{v}/manifests.yaml"
    )


async def get_latest_jobset_version() -> str | None:
    """Query GitHub API for the latest JobSet release tag.

    Returns:
        Latest release tag (e.g. "v0.7.1"), or None if the lookup fails.
    """
    import aiohttp
    import orjson

    from aiperf.transports.aiohttp_client import create_tcp_connector

    url = f"https://api.github.com/repos/{JOBSET_GITHUB_REPO}/releases/latest"
    headers = {"Accept": "application/vnd.github+json"}
    try:
        connector = create_tcp_connector()
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5), connector=connector
            ) as session,
            session.get(url, headers=headers) as resp,
        ):
            data = orjson.loads(await resp.read())
            tag = data.get("tag_name")
            return tag if isinstance(tag, str) else None
    except (aiohttp.ClientError, orjson.JSONDecodeError, TimeoutError):
        return None


def get_jobset_install_hint(version: str | None = None) -> str:
    """Get a user-facing hint for installing JobSet CRD.

    Args:
        version: Specific version tag, or None for fallback.

    Returns:
        Formatted install command string.
    """
    url = get_jobset_manifest_url(version)
    return f"Install JobSet: kubectl apply --server-side -f {url}"


def controller_dns_name(jobset_name: str, namespace: str) -> str:
    """Build the controller pod DNS hostname for a JobSet.

    JobSet with enableDNSHostnames creates a headless service with the same name
    as the JobSet, and pods get DNS names like:
    {jobset-name}-{job-name}-{job-index}-{pod-index}.{jobset-name}.{namespace}.svc.cluster.local

    Since we have exactly 1 controller replica with 1 pod, indices are always 0-0.

    Args:
        jobset_name: The JobSet resource name.
        namespace: Kubernetes namespace.

    Returns:
        Fully qualified DNS hostname for the controller pod.
    """
    return f"{jobset_name}-controller-0-0.{jobset_name}.{namespace}.svc.cluster.local"


class ContainerSpec(AIPerfBaseModel):
    """Specification for a container within a pod."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    name: str = Field(description="Container name")
    image: str = Field(description="Container image")
    image_pull_policy: ImagePullPolicy | None = Field(
        default=None,
        description="Image pull policy (Always, Never, IfNotPresent). "
        "Defaults to Always for :latest tags, IfNotPresent otherwise.",
    )
    command: list[str] = Field(default_factory=list, description="Command to run")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: list[dict[str, Any]] = Field(
        default_factory=list, description="Environment variables"
    )
    resources: dict[str, dict[str, str]] | None = Field(
        default=None, description="Resource requests and limits"
    )
    volume_mounts: list[dict[str, Any]] = Field(
        default_factory=list, description="Volume mounts"
    )
    ports: list[dict[str, Any]] = Field(
        default_factory=list, description="Container ports"
    )
    startup_probe: dict[str, Any] | None = Field(
        default=None, description="Startup probe configuration"
    )
    liveness_probe: dict[str, Any] | None = Field(
        default=None, description="Liveness probe configuration"
    )
    readiness_probe: dict[str, Any] | None = Field(
        default=None, description="Readiness probe configuration"
    )
    security_context: dict[str, Any] | None = Field(
        default=None, description="Container security context"
    )

    def to_k8s_spec(self) -> dict[str, Any]:
        """Convert to Kubernetes container spec."""
        return self.model_dump(
            by_alias=True, exclude_unset=True, exclude_none=True, mode="json"
        )


class ReplicatedJobSpec(AIPerfBaseModel):
    """Specification for a replicated job within a JobSet."""

    name: str = Field(description="Replicated job name")
    replicas: int = Field(default=1, description="Number of replicas")
    containers: list[ContainerSpec] = Field(
        default_factory=list, description="Containers in the pod"
    )
    volumes: list[dict[str, Any]] = Field(
        default_factory=list, description="Pod volumes"
    )
    restart_policy: RestartPolicy = Field(
        default=RestartPolicy.ON_FAILURE, description="Pod restart policy"
    )
    backoff_limit: int = Field(default=0, description="Job backoff limit for retries")
    job_ttl_seconds: int | None = Field(
        default=None,
        description="TTL for the Job after completion. 0 = delete immediately.",
    )
    pod_template: PodTemplateConfig | None = Field(
        default=None, description="Pod template configuration"
    )
    job_id: str | None = Field(default=None, description="Job ID for pod labeling")
    extra_annotations: dict[str, str] = Field(
        default_factory=dict,
        description="Additional annotations to add to the pod template",
    )

    def to_k8s_spec(self) -> dict[str, Any]:
        """Convert to Kubernetes replicatedJob spec."""
        pod_spec: dict[str, Any] = {
            "restartPolicy": str(self.restart_policy),
            "containers": [c.to_k8s_spec() for c in self.containers],
            "volumes": self.volumes,
            # Pod-level security context
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 1000,
                "runAsGroup": 1000,
                "fsGroup": 1000,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
        }

        # Apply pod template customizations
        if self.pod_template:
            tmpl = self.pod_template
            if tmpl.node_selector:
                pod_spec["nodeSelector"] = tmpl.node_selector
            if tmpl.tolerations:
                pod_spec["tolerations"] = tmpl.tolerations
            if tmpl.image_pull_secrets:
                pod_spec["imagePullSecrets"] = [
                    {"name": name} for name in tmpl.image_pull_secrets
                ]
            if tmpl.service_account_name:
                pod_spec["serviceAccountName"] = tmpl.service_account_name

        # Build metadata with annotations and labels
        pod_metadata: dict[str, Any] = {}
        annotations: dict[str, str] = {}
        if self.pod_template and self.pod_template.annotations:
            annotations.update(self.pod_template.annotations)
        if self.extra_annotations:
            annotations.update(self.extra_annotations)
        if annotations:
            pod_metadata["annotations"] = annotations

        # Build pod labels: base AIPerf labels + custom labels
        pod_labels: dict[str, str] = {Labels.APP_KEY: Labels.APP_VALUE}
        if self.job_id:
            pod_labels[Labels.JOB_ID] = self.job_id
        if self.pod_template and self.pod_template.labels:
            pod_labels.update(self.pod_template.labels)
        pod_metadata["labels"] = pod_labels

        pod_template: dict[str, Any] = {"spec": pod_spec}
        if pod_metadata:
            pod_template["metadata"] = pod_metadata

        job_spec: dict[str, Any] = {
            "parallelism": 1,
            "completions": 1,
            "completionMode": "Indexed",
            "backoffLimit": self.backoff_limit,
            "template": pod_template,
        }
        if self.job_ttl_seconds is not None:
            job_spec["ttlSecondsAfterFinished"] = self.job_ttl_seconds

        return {
            "name": self.name,
            "replicas": self.replicas,
            "template": {"spec": job_spec},
        }


class JobSetSpec(AIPerfBaseModel):
    """Specification for a complete JobSet deployment.

    Resource settings, ports, and health probe configuration are loaded from
    K8sEnvironment and can be customized via AIPERF_K8S_* environment variables.
    """

    name: str = Field(description="JobSet name")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    job_id: str = Field(description="Unique benchmark job ID")
    image: str = Field(description="AIPerf container image")
    image_pull_policy: ImagePullPolicy | None = Field(
        default=None,
        description="Image pull policy for all containers (Always, Never, IfNotPresent). "
        "Set to 'Never' for local development with minikube.",
    )
    resource_mode: Literal["guaranteed", "burstable", "none"] = Field(
        default="guaranteed",
        description="CPU/memory resource mode for controller and worker pods. "
        "'guaranteed' emits requests==limits. "
        "'burstable' emits requests only (no limits). "
        "'none' omits the resources block.",
    )
    worker_replicas: int = Field(default=1, description="Number of worker pods")
    workers_per_pod: int | None = Field(
        default=None,
        description="Actual workers per pod (used for resource calculation). "
        "Defaults to Environment.WORKER.DEFAULT_WORKERS_PER_POD if not set.",
    )
    record_processors_per_pod: int | None = Field(
        default=None,
        description="Actual record processors per worker pod. "
        "Defaults to a Kubernetes scale factor derived from workers_per_pod.",
    )
    ttl_seconds: int | None = Field(
        default=None, description="TTL after finished (uses K8sEnvironment default)"
    )
    keep_failed_pods: bool = Field(
        default=False,
        description="Preserve failed JobSet pod attempts for debugging.",
    )

    # Pod template
    pod_template: PodTemplateConfig = Field(
        default_factory=PodTemplateConfig, description="Pod template configuration"
    )

    # Scheduling
    scheduling: SchedulingConfig = Field(
        default_factory=SchedulingConfig, description="Kueue scheduling configuration"
    )
    gpu_telemetry_enabled: bool = Field(
        default=True,
        description="Whether to include the GPU telemetry manager container.",
    )
    server_metrics_enabled: bool = Field(
        default=True,
        description="Whether to include the server metrics manager container.",
    )

    # Optional metadata for discovery
    name_label: str | None = Field(
        default=None, description="Human-readable name label for the JobSet"
    )
    extra_annotations: dict[str, str] = Field(
        default_factory=dict,
        description="Additional annotations for the JobSet metadata",
    )

    def _create_security_context(self) -> dict[str, Any]:
        """Create a security context for containers.

        Applies security best practices:
        - Run as non-root user
        - Drop all capabilities
        - Read-only root filesystem (writable emptyDir volumes for data/ipc/results)
        """
        ctx: dict[str, Any] = {
            "runAsNonRoot": True,
            "runAsUser": 1000,
            "runAsGroup": 1000,
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "capabilities": {"drop": ["ALL"]},
            "seccompProfile": {"type": "RuntimeDefault"},
        }
        overrides = self.pod_template.container_security_context
        if overrides:
            caps = overrides.get("capabilities")
            if isinstance(caps, dict):
                base_caps = dict(ctx.get("capabilities", {}))
                base_caps.update(caps)
                ctx["capabilities"] = base_caps
            for key, value in overrides.items():
                if key == "capabilities":
                    continue
                ctx[key] = value
        return ctx

    def _create_health_probe(self, port: int, path: str = "/healthz") -> dict[str, Any]:
        """Create a health probe configuration from K8sEnvironment settings."""
        health = K8sEnvironment.HEALTH
        return {
            "httpGet": {"path": path, "port": port},
            "initialDelaySeconds": health.INITIAL_DELAY_SECONDS,
            "periodSeconds": health.PERIOD_SECONDS,
            "timeoutSeconds": health.TIMEOUT_SECONDS,
            "failureThreshold": health.FAILURE_THRESHOLD,
            "successThreshold": health.SUCCESS_THRESHOLD,
        }

    def _resolve_pod_resources(
        self, settings_key: str
    ) -> dict[str, dict[str, str]] | None:
        """Resolve controller/worker pod resources for this JobSet.

        The default mode preserves the existing Guaranteed QoS behavior.
        The ``burstable`` mode sets requests only (no limits) so containers
        can burst beyond the reservation without being OOM-killed by cgroup.
        The ``none`` mode is an explicit escape hatch that omits CPU/memory
        requests and limits from the generated container specs.
        """
        if self.resource_mode == "none":
            return None
        return getattr(K8sEnvironment, settings_key).to_k8s_resources(
            burstable=self.resource_mode == "burstable"
        )

    def _resolve_workers_per_pod(self) -> int:
        """Resolve workers per pod for manifest generation."""
        return self.workers_per_pod or Environment.WORKER.DEFAULT_WORKERS_PER_POD

    def _resolve_record_processors_per_pod(self) -> int:
        """Resolve record processors per pod for manifest generation."""
        if self.record_processors_per_pod is not None:
            return self.record_processors_per_pod
        return max(
            1,
            self._resolve_workers_per_pod()
            // K8sEnvironment.RECORD_PROCESSOR_SCALE_FACTOR,
        )

    @staticmethod
    def _split_weighted_total(total: int, weights: list[int]) -> list[int]:
        """Split an integer total across weighted buckets.

        Uses a largest-remainder allocation so the sum is preserved exactly.
        """
        if not weights:
            return []
        if total <= 0:
            return [0] * len(weights)

        total_weight = sum(weights)
        raw_shares = [total * weight / total_weight for weight in weights]
        shares = [int(share) for share in raw_shares]
        remainder = total - sum(shares)

        ranked = sorted(
            range(len(weights)),
            key=lambda idx: raw_shares[idx] - shares[idx],
            reverse=True,
        )
        for idx in ranked[:remainder]:
            shares[idx] += 1

        return shares

    @staticmethod
    def _format_mcpu(mcpu: int) -> str:
        """Format millicores as a Kubernetes quantity."""
        if mcpu % 1000 == 0:
            return str(mcpu // 1000)
        return f"{mcpu}m"

    @staticmethod
    def _format_mib(mib: int) -> str:
        """Format MiB as a Kubernetes memory quantity."""
        return f"{mib}Mi"

    def _pod_template_env_value(self, name: str) -> str | None:
        """Return a string value from podTemplate env when present."""
        for item in self.pod_template.env:
            if (item or {}).get("name") != name:
                continue
            value = (item or {}).get("value")
            if isinstance(value, str):
                return value
        return None

    def _split_worker_pod_resources(
        self,
        worker_count: int,
        record_processor_count: int,
    ) -> list[dict[str, dict[str, str]] | None]:
        """Split the configured worker-pod budget across pod infrastructure and services.

        The external API remains pod-oriented (`WORKER_POD` is the total budget).
        Internally we divide that budget across the worker-pod-manager, workers,
        and record processors so the sum of container requests/limits matches the
        historical per-pod request.
        """
        worker_pod_resources = self._resolve_pod_resources("WORKER_POD")
        if worker_pod_resources is None:
            return [None] * (1 + worker_count + record_processor_count)

        total_mcpu = int(
            round(parse_cpu(worker_pod_resources["requests"]["cpu"]) * 1000)
        )
        total_mib = parse_memory_mib(worker_pod_resources["requests"]["memory"])

        # These weights reflect the measured relative cost noted in the K8s
        # environment comments: workers are lighter than record processors,
        # while the worker-pod-manager remains a small but non-zero share.
        cpu_weights = [100] + ([131] * worker_count) + ([389] * record_processor_count)
        memory_weights = (
            [128] + ([80] * worker_count) + ([256] * record_processor_count)
        )

        record_processor_cpu_request = (
            self._pod_template_env_value("AIPERF_K8S_RECORD_PROCESSOR_CPU_REQUEST")
            or K8sEnvironment.RECORD_PROCESSOR_CPU_REQUEST
        )
        if record_processor_cpu_request is None or record_processor_count == 0:
            cpu_shares = self._split_weighted_total(total_mcpu, cpu_weights)
        else:
            record_processor_mcpu = int(
                round(parse_cpu(record_processor_cpu_request) * 1000)
            )
            fixed_record_processor_total = (
                record_processor_mcpu * record_processor_count
            )
            remaining_mcpu = max(0, total_mcpu - fixed_record_processor_total)
            non_record_processor_weights = [100] + ([131] * worker_count)
            cpu_shares = (
                self._split_weighted_total(
                    remaining_mcpu,
                    non_record_processor_weights,
                )
                + [record_processor_mcpu] * record_processor_count
            )
        memory_shares = self._split_weighted_total(total_mib, memory_weights)

        burstable = self.resource_mode == "burstable"
        resources: list[dict[str, dict[str, str]]] = []
        for mcpu, mib in zip(cpu_shares, memory_shares, strict=True):
            entry: dict[str, dict[str, str]] = {
                "requests": {
                    "cpu": self._format_mcpu(mcpu),
                    "memory": self._format_mib(mib),
                },
            }
            if not burstable:
                entry["limits"] = {
                    "cpu": self._format_mcpu(mcpu),
                    "memory": self._format_mib(mib),
                }
            resources.append(entry)
        return resources

    def _allocate_worker_health_ports(
        self,
        worker_count: int,
        record_processor_count: int,
    ) -> tuple[int, list[int], list[int]]:
        """Allocate unique health ports for every container in a worker pod.

        Containers in a pod share a network namespace, so each service container
        needs its own port even though probes are scoped per container.
        """
        ports = K8sEnvironment.PORTS
        manager_port = ports.WORKER_HEALTH
        worker_ports = list(range(manager_port + 1, manager_port + 1 + worker_count))
        record_processor_start = max(
            ports.RECORD_PROCESSOR_HEALTH,
            manager_port + 1 + worker_count,
        )
        record_processor_ports = list(
            range(
                record_processor_start,
                record_processor_start + record_processor_count,
            )
        )

        allocated = [manager_port, *worker_ports, *record_processor_ports]
        if allocated and max(allocated) > 65535:
            raise ValueError(
                "Not enough port space to allocate unique worker-container health ports"
            )
        return manager_port, worker_ports, record_processor_ports

    def _create_startup_probe(
        self, port: int, path: str = "/healthz"
    ) -> dict[str, Any]:
        """Create a startup probe for slow-starting containers.

        Startup probes allow containers more time to initialize before
        liveness/readiness probes take over. Uses more lenient settings
        to accommodate initialization time.
        """
        health = K8sEnvironment.HEALTH
        return {
            "httpGet": {"path": path, "port": port},
            "initialDelaySeconds": 0,
            "periodSeconds": health.STARTUP_PERIOD_SECONDS,
            "timeoutSeconds": health.TIMEOUT_SECONDS,
            "failureThreshold": health.STARTUP_FAILURE_THRESHOLD,
        }

    def _create_env_vars(
        self,
        controller_host: str | None = None,
        include_pod_index: bool = True,
    ) -> list[dict[str, Any]]:
        """Create environment variables for a container."""
        jobset_config = K8sEnvironment.JOBSET
        datasets_path = jobset_config.DATASETS_PATH
        has_hf_home = any(
            (item or {}).get("name") == "HF_HOME" for item in self.pod_template.env
        )
        registration_timeout = max(Environment.SERVICE.REGISTRATION_TIMEOUT, 120.0)
        env: list[dict[str, Any]] = [
            # Shared dataset path: dataset-manager writes mmap files here,
            # API service serves them to workers via HTTP
            {"name": "AIPERF_DATASET_MMAP_BASE_PATH", "value": datasets_path},
            # Job ID and namespace for the benchmark
            {"name": "AIPERF_JOB_ID", "value": self.job_id},
            {"name": "AIPERF_NAMESPACE", "value": self.namespace},
            # Health server must bind to 0.0.0.0 so K8s probes can reach it via the pod IP
            {"name": "AIPERF_SERVICE_HEALTH_ENABLED", "value": "true"},
            {"name": "AIPERF_SERVICE_HEALTH_HOST", "value": "0.0.0.0"},
            # Keep registration timeout configurable instead of hard-coding a
            # separate K8s-only value, so startup tuning and failure detection
            # stay aligned with the active environment configuration.
            {
                "name": "AIPERF_SERVICE_REGISTRATION_TIMEOUT",
                "value": str(registration_timeout),
            },
        ]

        # HF cache must be writable (readOnlyRootFilesystem)
        if not has_hf_home:
            env.append({"name": "HF_HOME", "value": "/tmp/hf_home"})

        if include_pod_index:
            # Expose the JobSet job-index as a unique pod identifier for
            # worker-pod services. JOB_COMPLETION_INDEX is always 0 because
            # each replicated job has completions=1; the JobSet job-index
            # label is the true replica index.
            env.append(
                {
                    "name": "AIPERF_POD_INDEX",
                    "valueFrom": {
                        "fieldRef": {
                            "fieldPath": "metadata.labels['jobset.sigs.k8s.io/job-index']",
                        }
                    },
                }
            )

        if controller_host:
            env.append(
                {"name": "AIPERF_K8S_ZMQ_CONTROLLER_HOST", "value": controller_host}
            )

        # Add custom environment variables from pod template
        env.extend(self.pod_template.env)
        return env

    def _get_volume_mounts(self) -> list[dict[str, Any]]:
        """Get all volume mounts including config and IPC."""
        config_path = K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH
        ipc_path = K8sEnvironment.ZMQ.IPC_PATH
        datasets_path = K8sEnvironment.JOBSET.DATASETS_PATH
        mounts: list[dict[str, Any]] = [
            {"name": "config", "mountPath": config_path, "readOnly": True},
            {"name": "ipc", "mountPath": ipc_path},
            {"name": "results", "mountPath": "/results"},
            # Shared dataset volume: dataset-manager writes, API serves to workers
            {"name": "datasets", "mountPath": datasets_path},
            {"name": "tmp", "mountPath": "/tmp"},
        ]
        mounts.extend(self.pod_template.volume_mounts)
        return mounts

    def _create_container(
        self,
        name: str,
        service_type: str,
        health_port: int,
        resources: dict[str, dict[str, str]] | None,
        api_port: int | None = None,
        controller_host: str | None = None,
        service_id: str | None = None,
        extra_env: list[dict[str, Any]] | None = None,
        include_pod_index: bool = True,
        skip_readiness_probe: bool = False,
        skip_startup_probe: bool = False,
        skip_liveness_probe: bool = False,
    ) -> ContainerSpec:
        """Create a container spec with standard AIPerf configuration.

        Args:
            name: Container name.
            service_type: AIPerf service type for this container.
            health_port: Health check port.
            resources: Optional Kubernetes resource requests/limits.
            api_port: Optional API port for services that expose APIs.
            controller_host: Controller DNS for worker containers.
            service_id: Optional explicit AIPerf service_id for this container.
            extra_env: Additional environment variables for this container.
            include_pod_index: Whether to expose AIPERF_POD_INDEX into the container.
            skip_readiness_probe: If True, don't add a readiness probe.
            skip_startup_probe: If True, don't add a startup probe.
            skip_liveness_probe: If True, don't add a liveness probe.
        """
        jobset_config = K8sEnvironment.JOBSET
        run_file = f"{jobset_config.CONFIG_MOUNT_PATH}/run_config.json"
        args = [
            "service",
            "--type",
            service_type,
            "--health-port",
            str(health_port),
            "--benchmark-run",
            run_file,
        ]
        if service_id:
            args.extend(["--service-id", service_id])
        if api_port:
            args.extend(["--api-port", str(api_port)])

        ports: list[dict[str, Any]] = [{"containerPort": health_port, "name": "health"}]
        if api_port:
            ports.append({"containerPort": api_port, "name": "api"})

        env = self._create_env_vars(
            controller_host=controller_host,
            include_pod_index=include_pod_index,
        )
        if extra_env:
            env.extend(extra_env)

        # Configure probes - startup probe allows slow initialization,
        # then liveness/readiness take over for ongoing health monitoring.
        # The API service exposes /healthz and /readyz on its FastAPI port, not on
        # the separate service health port used by the other containers.
        probe_port = api_port if service_type == "api" and api_port else health_port
        startup_probe = (
            None if skip_startup_probe else self._create_startup_probe(probe_port)
        )
        liveness_probe = (
            None if skip_liveness_probe else self._create_health_probe(probe_port)
        )
        readiness_probe = (
            None
            if skip_readiness_probe
            else self._create_health_probe(probe_port, path="/readyz")
        )

        return ContainerSpec(
            name=name,
            image=self.image,
            image_pull_policy=self.image_pull_policy,
            command=["aiperf"],
            args=args,
            env=env,
            resources=resources,
            volume_mounts=self._get_volume_mounts(),
            ports=ports,
            startup_probe=startup_probe,
            liveness_probe=liveness_probe,
            readiness_probe=readiness_probe,
            security_context=self._create_security_context(),
        )

    def _create_controller_containers(self) -> list[ContainerSpec]:
        """Create one container per control-plane service in the controller pod.

        A small results sidecar shares /results and can continue serving
        exported artifacts if the main controller container terminates after
        export but before the operator downloads them.

        Workers and RecordProcessors are external worker-pod services managed
        by JobSet.
        """
        ports = K8sEnvironment.PORTS

        sidecar_resources = self._resolve_pod_resources("RESULTS_SIDECAR")

        results_sidecar = ContainerSpec(
            name=Containers.RESULTS_SIDECAR,
            image=self.image,
            image_pull_policy=self.image_pull_policy,
            command=["python", "-m", "aiperf.kubernetes.results_sidecar"],
            env=[
                {"name": "AIPERF_RESULTS_DIR", "value": "/results"},
                {
                    "name": "AIPERF_RESULTS_SIDECAR_PORT",
                    "value": str(ports.RESULTS_SIDECAR),
                },
            ],
            resources=sidecar_resources,
            volume_mounts=[
                {"name": "results", "mountPath": "/results", "readOnly": True},
                {"name": "tmp", "mountPath": "/tmp"},
            ],
            ports=[{"containerPort": ports.RESULTS_SIDECAR, "name": "results"}],
            startup_probe=self._create_startup_probe(ports.RESULTS_SIDECAR),
            liveness_probe=self._create_health_probe(ports.RESULTS_SIDECAR),
            readiness_probe=self._create_health_probe(ports.RESULTS_SIDECAR),
            security_context=self._create_security_context(),
        )

        containers = [
            self._create_container(
                name=Containers.CONTROL_PLANE,
                service_type="system_controller",
                health_port=ports.SYSTEM_CONTROLLER_HEALTH,
                resources=self._resolve_pod_resources("SYSTEM_CONTROLLER"),
                service_id="system_controller",
                include_pod_index=False,
                skip_readiness_probe=True,  # System controller manages its own lifecycle
                # Enable realtime metrics since we don't use DASHBOARD UI
                extra_env=[
                    {"name": "AIPERF_UI_REALTIME_METRICS_ENABLED", "value": "true"}
                ],
            ),
            self._create_container(
                name=Containers.DATASET_MANAGER,
                service_type="dataset_manager",
                health_port=ports.DATASET_MANAGER_HEALTH,
                resources=self._resolve_pod_resources("DATASET_MANAGER"),
                service_id="dataset_manager",
                include_pod_index=False,
            ),
            self._create_container(
                name=Containers.TIMING_MANAGER,
                service_type="timing_manager",
                health_port=ports.TIMING_MANAGER_HEALTH,
                resources=self._resolve_pod_resources("TIMING_MANAGER"),
                service_id="timing_manager",
                include_pod_index=False,
            ),
            self._create_container(
                name=Containers.RECORDS_MANAGER,
                service_type="records_manager",
                health_port=ports.RECORDS_MANAGER_HEALTH,
                resources=self._resolve_pod_resources("RECORDS_MANAGER"),
                service_id="records_manager",
                include_pod_index=False,
                skip_readiness_probe=True,
                skip_startup_probe=True,
                skip_liveness_probe=True,
            ),
            self._create_container(
                name=Containers.API,
                service_type="api",
                health_port=ports.API_SERVICE_HEALTH,
                resources=self._resolve_pod_resources("API"),
                api_port=ports.API_SERVICE,
                service_id="api",
                include_pod_index=False,
            ),
        ]

        if self.gpu_telemetry_enabled:
            containers.append(
                self._create_container(
                    name=Containers.GPU_TELEMETRY_MANAGER,
                    service_type="gpu_telemetry_manager",
                    health_port=ports.GPU_TELEMETRY_MANAGER_HEALTH,
                    resources=self._resolve_pod_resources("GPU_TELEMETRY_MANAGER"),
                    service_id="gpu_telemetry_manager",
                    include_pod_index=False,
                )
            )

        if self.server_metrics_enabled:
            containers.append(
                self._create_container(
                    name=Containers.SERVER_METRICS_MANAGER,
                    service_type="server_metrics_manager",
                    health_port=ports.SERVER_METRICS_MANAGER_HEALTH,
                    resources=self._resolve_pod_resources("SERVER_METRICS_MANAGER"),
                    service_id="server_metrics_manager",
                    include_pod_index=False,
                )
            )

        containers.append(results_sidecar)
        return containers

    def _create_worker_containers(self, controller_dns: str) -> list[ContainerSpec]:
        """Create worker-pod containers with one container per runtime service.

        The worker pod keeps a lightweight worker-pod-manager for shared pod
        infrastructure (dataset download once per pod, local raw-inference
        proxy, raw-record upload coordination), while each worker and record
        processor runs in its own container instead of a subprocess.
        """
        worker_count = self._resolve_workers_per_pod()
        record_processor_count = self._resolve_record_processors_per_pod()
        manager_port, worker_ports, record_processor_ports = (
            self._allocate_worker_health_ports(worker_count, record_processor_count)
        )
        resources = self._split_worker_pod_resources(
            worker_count, record_processor_count
        )

        containers: list[ContainerSpec] = [
            self._create_container(
                name=Containers.WORKER_POD_MANAGER,
                service_type="worker_pod_manager",
                health_port=manager_port,
                resources=resources[0],
                controller_host=controller_dns,
                skip_readiness_probe=True,
                skip_startup_probe=True,
                skip_liveness_probe=True,
            )
        ]

        for ordinal, health_port in enumerate(worker_ports):
            containers.append(
                self._create_container(
                    name=f"worker-{ordinal}",
                    service_type="worker",
                    service_id=f"worker_$(AIPERF_POD_INDEX)_{ordinal}",
                    health_port=health_port,
                    resources=resources[1 + ordinal],
                    controller_host=controller_dns,
                    skip_readiness_probe=True,
                    skip_startup_probe=True,
                    skip_liveness_probe=True,
                )
            )

        record_processor_offset = 1 + worker_count
        for ordinal, health_port in enumerate(record_processor_ports):
            containers.append(
                self._create_container(
                    name=f"record-processor-{ordinal}",
                    service_type="record_processor",
                    service_id=(f"record_processor_$(AIPERF_POD_INDEX)_{ordinal}"),
                    health_port=health_port,
                    resources=resources[record_processor_offset + ordinal],
                    controller_host=controller_dns,
                    skip_readiness_probe=True,
                    skip_startup_probe=True,
                    skip_liveness_probe=True,
                )
            )

        return containers

    def to_k8s_manifest(self) -> dict[str, Any]:
        """Generate the complete JobSet Kubernetes manifest."""
        controller_dns = controller_dns_name(self.name, self.namespace)
        jobset_config = K8sEnvironment.JOBSET

        # Common volumes
        volumes: list[dict[str, Any]] = [
            {"name": "config", "configMap": {"name": f"{self.name}-config"}},
            {"name": "ipc", "emptyDir": {}},
            {"name": "results", "emptyDir": {}},
            # Shared dataset volume for controller containers (dataset-manager creates, API serves)
            {"name": "datasets", "emptyDir": {}},
            {"name": "tmp", "emptyDir": {}},
        ]
        volumes.extend(self.pod_template.volumes)

        # Controller replicated job
        api_port = K8sEnvironment.PORTS.API_SERVICE
        controller_job = ReplicatedJobSpec(
            name="controller",
            replicas=1,
            containers=self._create_controller_containers(),
            volumes=volumes,
            restart_policy=RestartPolicy.NEVER,
            backoff_limit=jobset_config.CONTROLLER_BACKOFF_LIMIT,
            pod_template=self.pod_template,
            job_id=self.job_id,
            extra_annotations={
                "prometheus.io/scrape": "true",
                "prometheus.io/port": str(api_port),
                "prometheus.io/path": "/metrics",
            },
        )

        worker_job = ReplicatedJobSpec(
            name="workers",
            replicas=self.worker_replicas,
            containers=self._create_worker_containers(controller_dns),
            volumes=volumes,
            restart_policy=RestartPolicy.ON_FAILURE,
            backoff_limit=(
                0 if self.keep_failed_pods else jobset_config.WORKER_BACKOFF_LIMIT
            ),
            job_ttl_seconds=None if self.keep_failed_pods else self.ttl_seconds,
            pod_template=self.pod_template,
            job_id=self.job_id,
        )

        # Build JobSet manifest
        labels: dict[str, str] = {
            Labels.APP_KEY: Labels.APP_VALUE,
            Labels.JOB_ID: self.job_id,
        }
        if self.name_label:
            labels[Labels.NAME] = self.name_label
        if self.scheduling.queue_name:
            labels[KueueLabels.QUEUE_NAME] = self.scheduling.queue_name
        if self.scheduling.priority_class:
            labels[KueueLabels.PRIORITY_CLASS] = self.scheduling.priority_class

        metadata: dict[str, Any] = {
            "name": self.name,
            "namespace": self.namespace,
            "labels": labels,
        }
        if self.extra_annotations:
            metadata["annotations"] = self.extra_annotations

        manifest: dict[str, Any] = {
            "apiVersion": JOBSET_API.api_version,
            "kind": "JobSet",
            "metadata": metadata,
            "spec": {
                # Enable DNS hostnames for pod-to-pod communication
                # This creates a headless service with the same name as the JobSet,
                # allowing pods to have DNS names like:
                # {jobset-name}-{job-name}-{job-index}-{pod-index}.{jobset-name}.{namespace}.svc.cluster.local
                "network": {
                    "enableDNSHostnames": True,
                },
                "successPolicy": {
                    "operator": "All",
                    "targetReplicatedJobs": ["controller"],
                },
                "replicatedJobs": [
                    controller_job.to_k8s_spec(),
                    worker_job.to_k8s_spec(),
                ],
            },
        }

        # Kueue requires JobSets to start suspended; it unsuspends after admission
        if self.scheduling.queue_name:
            manifest["spec"]["suspend"] = True

        ttl = None
        if not self.keep_failed_pods:
            ttl = (
                self.ttl_seconds
                if self.ttl_seconds is not None
                else jobset_config.TTL_SECONDS_AFTER_FINISHED
            )
        if ttl is not None:
            manifest["spec"]["ttlSecondsAfterFinished"] = ttl

        return manifest
