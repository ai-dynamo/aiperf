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

from aiperf.common.models import AIPerfBaseModel
from aiperf.config.deployment import PodTemplateConfig, SchedulingConfig
from aiperf.kubernetes.constants import Containers, KueueLabels, Labels
from aiperf.kubernetes.enums import ImagePullPolicy, RestartPolicy
from aiperf.kubernetes.environment import K8sEnvironment


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
    resource_mode: Literal["guaranteed", "none"] = Field(
        default="guaranteed",
        description="CPU/memory resource mode for controller and worker pods. "
        "'guaranteed' emits requests==limits. 'none' omits the resources block.",
    )
    worker_replicas: int = Field(default=1, description="Number of worker pods")
    workers_per_pod: int | None = Field(
        default=None,
        description="Actual workers per pod (used for resource calculation). "
        "Defaults to Environment.WORKER.DEFAULT_WORKERS_PER_POD if not set.",
    )
    ttl_seconds: int | None = Field(
        default=None, description="TTL after finished (uses K8sEnvironment default)"
    )

    # Pod template
    pod_template: PodTemplateConfig = Field(
        default_factory=PodTemplateConfig, description="Pod template configuration"
    )

    # Scheduling
    scheduling: SchedulingConfig = Field(
        default_factory=SchedulingConfig, description="Kueue scheduling configuration"
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
        return {
            "runAsNonRoot": True,
            "runAsUser": 1000,
            "runAsGroup": 1000,
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "capabilities": {"drop": ["ALL"]},
            "seccompProfile": {"type": "RuntimeDefault"},
        }

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
        The ``none`` mode is an explicit escape hatch that omits CPU/memory
        requests and limits from the generated container specs.
        """
        if self.resource_mode == "none":
            return None
        return getattr(K8sEnvironment, settings_key).to_k8s_resources()

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
        self, controller_host: str | None = None
    ) -> list[dict[str, Any]]:
        """Create environment variables for a container."""
        jobset_config = K8sEnvironment.JOBSET
        datasets_path = jobset_config.DATASETS_PATH
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
            # K8s pods need longer registration timeout: controller pod startup
            # is slower due to ZMQ proxy setup, health server, and cross-pod networking
            {"name": "AIPERF_SERVICE_REGISTRATION_TIMEOUT", "value": "120"},
            # HF cache must be writable (readOnlyRootFilesystem)
            {"name": "HF_HOME", "value": "/tmp/hf_home"},
            # Expose the JobSet job-index as a unique pod identifier.
            # JOB_COMPLETION_INDEX is always 0 because each replicated job has
            # completions=1; the JobSet job-index label is the true replica index.
            {
                "name": "AIPERF_POD_INDEX",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.labels['jobset.sigs.k8s.io/job-index']",
                    }
                },
            },
        ]

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
        extra_env: list[dict[str, Any]] | None = None,
        skip_readiness_probe: bool = False,
    ) -> ContainerSpec:
        """Create a container spec with standard AIPerf configuration.

        Args:
            name: Container name.
            service_type: AIPerf service type(s), comma-separated.
            health_port: Health check port.
            resources: Optional Kubernetes resource requests/limits.
            api_port: Optional API port for services that expose APIs.
            controller_host: Controller DNS for worker containers.
            extra_env: Additional environment variables for this container.
            skip_readiness_probe: If True, don't add a readiness probe.
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
        if api_port:
            args.extend(["--api-port", str(api_port)])

        ports: list[dict[str, Any]] = [{"containerPort": health_port, "name": "health"}]
        if api_port:
            ports.append({"containerPort": api_port, "name": "api"})

        env = self._create_env_vars(controller_host=controller_host)
        if extra_env:
            env.extend(extra_env)

        # Configure probes - startup probe allows slow initialization,
        # then liveness/readiness take over for ongoing health monitoring
        startup_probe = self._create_startup_probe(health_port)
        liveness_probe = self._create_health_probe(health_port)
        readiness_probe = (
            None
            if skip_readiness_probe
            else self._create_health_probe(health_port, path="/readyz")
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
        """Create a single container for the control-plane pod.

        The control-plane runs as a single container where SystemController
        spawns all other control-plane services as subprocesses. This enables
        the control-plane to load, run, and fail as a single unit.

        Services spawned as subprocesses:
        - worker_manager, timing_manager, dataset_manager, records_manager
        - api, gpu_telemetry_manager, server_metrics_manager

        A small results sidecar shares /results and can continue serving
        exported artifacts if the main controller container terminates after
        export but before the operator downloads them.

        Workers and RecordProcessors are external pods managed by JobSet.
        """
        ports = K8sEnvironment.PORTS

        control_plane_resources = self._resolve_pod_resources("CONTROLLER_POD")
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

        return [
            self._create_container(
                name=Containers.CONTROL_PLANE,
                service_type="system_controller",
                health_port=ports.SYSTEM_CONTROLLER_HEALTH,
                resources=control_plane_resources,
                api_port=ports.API_SERVICE,  # API is spawned as subprocess but needs port exposed
                skip_readiness_probe=True,  # System controller manages its own lifecycle
                # Enable realtime metrics since we don't use DASHBOARD UI
                extra_env=[
                    {"name": "AIPERF_UI_REALTIME_METRICS_ENABLED", "value": "true"}
                ],
            ),
            results_sidecar,
        ]

    def _create_worker_containers(self, controller_dns: str) -> list[ContainerSpec]:
        """Create a single container for worker pods using WorkerPodManager.

        The WorkerPodManager runs as the main process and spawns multiple workers
        and record processors as subprocesses. This reduces network overhead by
        downloading the dataset once per pod and sharing it via mmap.
        """
        ports = K8sEnvironment.PORTS

        worker_pod_resources = self._resolve_pod_resources("WORKER_POD")

        return [
            self._create_container(
                name=Containers.WORKER_POD_MANAGER,
                service_type="worker_pod_manager",
                health_port=ports.WORKER_HEALTH,
                resources=worker_pod_resources,
                controller_host=controller_dns,
            ),
        ]

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

        # Worker replicated job — ttl=0 deletes worker Jobs+pods immediately
        # after they succeed, freeing cluster resources while the controller
        # pod continues serving results to the operator.
        worker_job = ReplicatedJobSpec(
            name="workers",
            replicas=self.worker_replicas,
            containers=self._create_worker_containers(controller_dns),
            volumes=volumes,
            restart_policy=RestartPolicy.ON_FAILURE,
            backoff_limit=jobset_config.WORKER_BACKOFF_LIMIT,
            job_ttl_seconds=0,
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

        # Add TTL (use instance value if set, otherwise use environment default)
        ttl = (
            self.ttl_seconds
            if self.ttl_seconds is not None
            else jobset_config.TTL_SECONDS_AFTER_FINISHED
        )
        if ttl is not None:
            manifest["spec"]["ttlSecondsAfterFinished"] = ttl

        return manifest
