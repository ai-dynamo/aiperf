# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JobSet specification generation for Kubernetes deployments.

This module generates JobSet YAML for deploying AIPerf as a distributed
benchmark across multiple pods. All resource and port settings are configurable
via environment variables through K8sEnvironment.
"""

from typing import Any, Literal

from pydantic import Field

from aiperf.common.environment import Environment
from aiperf.common.models import AIPerfBaseModel
from aiperf.config.deployment import PodTemplateConfig, SchedulingConfig
from aiperf.kubernetes.constants import AIPerfLabels, Containers, KueueLabels
from aiperf.kubernetes.cr_refs import JOBSET_API_VERSION
from aiperf.kubernetes.enums import ImagePullPolicy, RestartPolicy
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset_helpers import (
    build_container_args,
    build_container_ports,
    build_env_vars,
    build_health_probe,
    build_security_context,
    build_service_probes,
    build_shared_volumes,
    build_startup_probe,
    build_volume_mounts,
)
from aiperf.kubernetes.jobset_specs import AIPerfContainerSpec, AIPerfReplicatedJobSpec
from aiperf.kubernetes.utils import parse_cpu, parse_memory_mib

__all__ = [
    "AIPerfContainerSpec",
    "AIPerfJobSetSpec",
    "AIPerfReplicatedJobSpec",
    "controller_dns_name",
]


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


# ---------------------------------------------------------------------- pure resource/port allocation helpers


def split_weighted_total(total: int, weights: list[int]) -> list[int]:
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


def format_mcpu(mcpu: int) -> str:
    """Format millicores as a Kubernetes quantity."""
    if mcpu % 1000 == 0:
        return str(mcpu // 1000)
    return f"{mcpu}m"


def format_mib(mib: int) -> str:
    """Format MiB as a Kubernetes memory quantity."""
    return f"{mib}Mi"


def _compute_cpu_shares(
    total_mcpu: int,
    worker_count: int,
    record_processor_count: int,
    record_processor_cpu_request: str | None,
) -> list[int]:
    """Compute per-container CPU shares for a worker pod.

    When a fixed per-record-processor CPU request is configured, that value is
    pinned and the remaining budget is split across manager + workers.
    """
    cpu_weights = [100] + ([131] * worker_count) + ([389] * record_processor_count)
    if record_processor_cpu_request is None or record_processor_count == 0:
        return split_weighted_total(total_mcpu, cpu_weights)

    record_processor_mcpu = int(round(parse_cpu(record_processor_cpu_request) * 1000))
    fixed_total = record_processor_mcpu * record_processor_count
    remaining_mcpu = max(0, total_mcpu - fixed_total)
    non_record_weights = [100] + ([131] * worker_count)
    return (
        split_weighted_total(remaining_mcpu, non_record_weights)
        + [record_processor_mcpu] * record_processor_count
    )


def split_worker_pod_resources(
    worker_pod_resources: dict[str, dict[str, str]] | None,
    worker_count: int,
    record_processor_count: int,
    record_processor_cpu_request: str | None,
    *,
    burstable: bool,
) -> list[dict[str, dict[str, str]] | None]:
    """Split the worker-pod budget across manager/worker/record-processor containers.

    See :meth:`AIPerfJobSetSpec._split_worker_pod_resources` for the rationale:
    the external API remains pod-oriented (`WORKER_POD` is the total budget)
    and internally we divide across containers with measurement-derived weights.
    """
    total_containers = 1 + worker_count + record_processor_count
    if worker_pod_resources is None:
        return [None] * total_containers

    total_mcpu = int(round(parse_cpu(worker_pod_resources["requests"]["cpu"]) * 1000))
    total_mib = parse_memory_mib(worker_pod_resources["requests"]["memory"])

    # These weights reflect the measured relative cost noted in the K8s
    # environment comments: workers are lighter than record processors,
    # while the worker-pod-manager remains a small but non-zero share.
    memory_weights = [128] + ([80] * worker_count) + ([256] * record_processor_count)

    cpu_shares = _compute_cpu_shares(
        total_mcpu,
        worker_count,
        record_processor_count,
        record_processor_cpu_request,
    )
    memory_shares = split_weighted_total(total_mib, memory_weights)

    resources: list[dict[str, dict[str, str]]] = []
    for mcpu, mib in zip(cpu_shares, memory_shares, strict=True):
        entry: dict[str, dict[str, str]] = {
            "requests": {
                "cpu": format_mcpu(mcpu),
                "memory": format_mib(mib),
            },
        }
        if not burstable:
            entry["limits"] = {
                "cpu": format_mcpu(mcpu),
                "memory": format_mib(mib),
            }
        resources.append(entry)
    return resources


def allocate_worker_health_ports(
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
            f"Not enough port space to allocate unique worker-container health ports: "
            f"manager_port={manager_port}, worker_count={len(worker_ports)}, "
            f"record_processor_count={len(record_processor_ports)}, "
            f"max allocated port {max(allocated)} exceeds 65535. "
            f"Reduce --workers or lower base health port."
        )
    return manager_port, worker_ports, record_processor_ports


class AIPerfJobSetSpec(AIPerfBaseModel):
    """Specification for a complete JobSet deployment.

    Resource settings, ports, and health probe configuration are loaded from
    K8sEnvironment and can be customized via AIPERF_K8S_* environment variables.
    """

    name: str = Field(description="JobSet name")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    job_id: str = Field(description="Unique benchmark job ID")
    job_uid: str | None = Field(
        default=None,
        description="UID of the owning AIPerfJob resource for mutation fencing",
    )
    image: str = Field(description="AIPerf container image")
    image_pull_policy: ImagePullPolicy | None = Field(
        default=None,
        description="Image pull policy for all containers (Always, Never, IfNotPresent). "
        "Set to 'Never' for local development with minikube.",
    )
    resource_mode: Literal["guaranteed", "burstable", "none"] = Field(
        default="burstable",
        description="CPU/memory resource mode for controller and worker pods. "
        "'burstable' (default) emits requests only (no limits) so the controller "
        "can grow beyond the request during aggregation without being OOM-killed. "
        "'guaranteed' emits requests==limits. "
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

    def _resolved_queue_name(self) -> str | None:
        """Queue name from the CR, falling back to the operator-wide default."""
        return (
            self.scheduling.queue_name or K8sEnvironment.JOBSET.KUEUE_DEFAULT_QUEUE_NAME
        ) or None

    def _build_manifest_labels(self) -> dict[str, str]:
        """Build top-level JobSet labels (AIPerf, name, Kueue scheduling).

        Kueue queue-name and priority-class fall back to the operator-side
        defaults (`AIPERF_K8S_JOBSET_KUEUE_DEFAULT_QUEUE_NAME` /
        `_PRIORITY_CLASS`) when not set on the CR. This makes Kueue gang-
        scheduling default-on for clusters that have Kueue installed and a
        named LocalQueue, without forcing per-CR opt-in.
        """
        from aiperf.kubernetes.environment import K8sEnvironment

        labels: dict[str, str] = {
            AIPerfLabels.APP_KEY: AIPerfLabels.APP_VALUE,
            AIPerfLabels.JOB_ID: self.job_id,
        }
        if self.name_label:
            labels[AIPerfLabels.NAME] = self.name_label
        queue_name = self._resolved_queue_name()
        if queue_name:
            labels[KueueLabels.QUEUE_NAME] = queue_name
        priority_class = (
            self.scheduling.priority_class
            or K8sEnvironment.JOBSET.KUEUE_DEFAULT_PRIORITY_CLASS
        )
        if priority_class:
            labels[KueueLabels.PRIORITY_CLASS] = priority_class
        return labels

    def _resolve_manifest_ttl(self) -> int | None:
        """Resolve the top-level JobSet ttlSecondsAfterFinished value, if any."""
        if self.keep_failed_pods:
            return None
        if self.ttl_seconds is not None:
            return self.ttl_seconds
        return K8sEnvironment.JOBSET.TTL_SECONDS_AFTER_FINISHED

    # ------------------------------------------------------------------ pod-template fragments

    def _create_security_context(self) -> dict[str, Any]:
        """Container security context derived from this spec's pod template."""
        return build_security_context(self.pod_template)

    def _get_volume_mounts(self) -> list[dict[str, Any]]:
        """Volume mounts derived from this spec's pod template."""
        return build_volume_mounts(self.pod_template)

    # ------------------------------------------------------------------ resource resolution

    def _resolve_pod_resources(
        self, settings_key: str
    ) -> dict[str, dict[str, str]] | None:
        """Resolve controller/worker pod resources for this JobSet.

        The default ``burstable`` mode sets requests only (no limits) so
        containers can burst beyond the reservation without being OOM-killed by
        cgroup; ``guaranteed`` emits requests == limits for Guaranteed QoS.
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
        record_processor_cpu_request = (
            self._pod_template_env_value("AIPERF_K8S_RECORD_PROCESSOR_CPU_REQUEST")
            or K8sEnvironment.RECORD_PROCESSOR_CPU_REQUEST
        )
        return split_worker_pod_resources(
            self._resolve_pod_resources("WORKER_POD"),
            worker_count,
            record_processor_count,
            record_processor_cpu_request,
            burstable=self.resource_mode == "burstable",
        )

    # ------------------------------------------------------------------ env

    def _create_env_vars(
        self,
        controller_host: str | None = None,
        include_pod_index: bool = True,
        controller_pod: bool = False,
    ) -> list[dict[str, Any]]:
        """Create environment variables for a container."""
        return build_env_vars(
            job_id=self.job_id,
            job_uid=self.job_uid,
            namespace=self.namespace,
            pod_template=self.pod_template,
            controller_host=controller_host,
            include_pod_index=include_pod_index,
            controller_pod=controller_pod,
        )

    # ------------------------------------------------------------------ container factories

    def _create_container(
        self,
        name: str,
        service_type: str,
        health_port: int | None,
        resources: dict[str, dict[str, str]] | None,
        *,
        api_port: int | None = None,
        controller_host: str | None = None,
        service_id: str | None = None,
        extra_env: list[dict[str, Any]] | None = None,
        include_pod_index: bool = True,
        controller_pod: bool = False,
        skip_readiness_probe: bool = False,
        skip_startup_probe: bool = False,
        skip_liveness_probe: bool = False,
    ) -> AIPerfContainerSpec:
        """Create a container spec with standard AIPerf configuration."""
        args = build_container_args(service_type, health_port, api_port, service_id)
        ports = build_container_ports(health_port, api_port)

        env = self._create_env_vars(
            controller_host=controller_host,
            include_pod_index=include_pod_index,
            controller_pod=controller_pod,
        )
        if extra_env:
            env.extend(extra_env)

        # Probe the port the container actually serves on. A container that
        # runs FastAPI binds only its api_port; probing health_port instead
        # means the startup probe can never succeed, kubelet kills the
        # container, and the whole JobSet dies during startup. That is not
        # hypothetical -- it is how every 100k-250k-concurrency run failed on
        # the DGX cluster until this fallback was added, and the failure looks
        # like an unexplained startup timeout rather than a probe misconfig.
        probe_port = api_port or health_port
        startup_probe, liveness_probe, readiness_probe = build_service_probes(
            probe_port,
            skip_startup_probe=skip_startup_probe,
            skip_liveness_probe=skip_liveness_probe,
            skip_readiness_probe=skip_readiness_probe,
        )

        return AIPerfContainerSpec(
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

    def _create_event_bus_proxy_container(self) -> AIPerfContainerSpec:
        """Sidecar that runs the XPUB/XSUB event-bus proxy.

        Placed first in the controller pod's container list so the kubelet
        begins pulling and starting it before control-plane. The bind sockets
        come up in tens of milliseconds once the container starts — well
        inside the 90s client connection-probe timeout.

        Isolates pub/sub socket accept/forward from the SystemController
        event loop, so large fan-ins of record processors and workers at
        startup don't starve the control plane's CPU.
        """
        ports = K8sEnvironment.PORTS
        run_file = f"{K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH}/run_config.json"
        health_port = ports.EVENT_BUS_PROXY_HEALTH

        args = [
            "proxy",
            "--kind",
            "event_bus",
            "--benchmark-run",
            run_file,
            "--health-port",
            str(health_port),
        ]

        container_ports: list[dict[str, Any]] = [
            {"containerPort": health_port, "name": "health"},
            {
                "containerPort": ports.EVENT_BUS_PROXY_PUB_FRONTEND,
                "name": "pub-frontend",
            },
            {"containerPort": ports.EVENT_BUS_PROXY_SUB_BACKEND, "name": "sub-backend"},
        ]

        return AIPerfContainerSpec(
            name=Containers.EVENT_BUS_PROXY,
            image=self.image,
            image_pull_policy=self.image_pull_policy,
            command=["aiperf"],
            args=args,
            env=self._create_env_vars(include_pod_index=False, controller_pod=True),
            resources=self._resolve_pod_resources("EVENT_BUS_PROXY"),
            volume_mounts=self._get_volume_mounts(),
            ports=container_ports,
            startup_probe=build_startup_probe(health_port),
            liveness_probe=build_health_probe(health_port),
            readiness_probe=build_health_probe(health_port, path="/readyz"),
            security_context=self._create_security_context(),
        )

    def _create_results_sidecar(self) -> AIPerfContainerSpec:
        """Build the small results-serving sidecar container."""
        ports = K8sEnvironment.PORTS
        return AIPerfContainerSpec(
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
                {
                    "name": "AIPERF_RESULTS_SIDECAR_LOG_LEVEL",
                    "value": K8sEnvironment.RESULTS_SIDECAR_LOG_LEVEL,
                },
            ],
            resources=self._resolve_pod_resources("RESULTS_SIDECAR"),
            volume_mounts=[
                {"name": "results", "mountPath": "/results", "readOnly": True},
                {"name": "tmp", "mountPath": "/tmp"},
            ],
            ports=[{"containerPort": ports.RESULTS_SIDECAR, "name": "results"}],
            startup_probe=build_startup_probe(ports.RESULTS_SIDECAR),
            liveness_probe=build_health_probe(ports.RESULTS_SIDECAR),
            readiness_probe=build_health_probe(ports.RESULTS_SIDECAR),
            security_context=self._create_security_context(),
        )

    def _create_control_plane_containers(self) -> list[AIPerfContainerSpec]:
        """Build the five mandatory control-plane service containers."""
        ports = K8sEnvironment.PORTS
        return [
            self._create_container(
                name=Containers.CONTROL_PLANE,
                service_type="system_controller",
                health_port=ports.SYSTEM_CONTROLLER_HEALTH,
                resources=self._resolve_pod_resources("SYSTEM_CONTROLLER"),
                service_id="system_controller",
                include_pod_index=False,
                controller_pod=True,
                skip_readiness_probe=True,  # System controller manages its own lifecycle
                # Stamp the sidecar decision into the container that must honor
                # it: when the event-bus proxy runs as its own container, the
                # SystemController must NOT also bind the XPUB/XSUB addresses.
                extra_env=[
                    {
                        "name": "AIPERF_K8S_EVENT_BUS_SIDECAR_ENABLED",
                        "value": str(K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED).lower(),
                    }
                ],
            ),
            self._create_container(
                name=Containers.DATASET_MANAGER,
                service_type="dataset_manager",
                health_port=ports.DATASET_MANAGER_HEALTH,
                resources=self._resolve_pod_resources("DATASET_MANAGER"),
                service_id="dataset_manager",
                include_pod_index=False,
                controller_pod=True,
            ),
            self._create_container(
                name=Containers.TIMING_MANAGER,
                service_type="timing_manager",
                health_port=ports.TIMING_MANAGER_HEALTH,
                resources=self._resolve_pod_resources("TIMING_MANAGER"),
                service_id="timing_manager",
                include_pod_index=False,
                controller_pod=True,
            ),
            self._create_container(
                name=Containers.RECORDS_MANAGER,
                service_type="records_manager",
                health_port=ports.RECORDS_MANAGER_HEALTH,
                resources=self._resolve_pod_resources("RECORDS_MANAGER"),
                service_id="records_manager",
                include_pod_index=False,
                controller_pod=True,
                skip_readiness_probe=True,
                skip_startup_probe=True,
                skip_liveness_probe=True,
            ),
            self._create_container(
                name=Containers.API,
                service_type="api",
                # Must be explicit: containers in a pod share a network
                # namespace, and a health server with no --health-port falls
                # back to AIPERF_SERVICE_HEALTH_PORT (8080), racing the
                # control-plane container for that port. Whichever loses the
                # race dies on init.
                health_port=ports.API_SERVICE_HEALTH,
                resources=self._resolve_pod_resources("API"),
                api_port=K8sEnvironment.PORTS.API_SERVICE,
                service_id="api",
                include_pod_index=False,
                controller_pod=True,
            ),
        ]

    def _create_optional_manager_containers(self) -> list[AIPerfContainerSpec]:
        """Build GPU telemetry + server metrics containers when enabled."""
        ports = K8sEnvironment.PORTS
        managers: list[AIPerfContainerSpec] = []
        if self.gpu_telemetry_enabled:
            managers.append(
                self._create_container(
                    name=Containers.GPU_TELEMETRY_MANAGER,
                    service_type="gpu_telemetry_manager",
                    health_port=ports.GPU_TELEMETRY_MANAGER_HEALTH,
                    resources=self._resolve_pod_resources("GPU_TELEMETRY_MANAGER"),
                    service_id="gpu_telemetry_manager",
                    include_pod_index=False,
                    controller_pod=True,
                )
            )
        if self.server_metrics_enabled:
            managers.append(
                self._create_container(
                    name=Containers.SERVER_METRICS_MANAGER,
                    service_type="server_metrics_manager",
                    health_port=ports.SERVER_METRICS_MANAGER_HEALTH,
                    resources=self._resolve_pod_resources("SERVER_METRICS_MANAGER"),
                    service_id="server_metrics_manager",
                    include_pod_index=False,
                    controller_pod=True,
                )
            )
        return managers

    def create_controller_containers(self) -> list[AIPerfContainerSpec]:
        """Create one container per control-plane service in the controller pod.

        A small results sidecar shares /results and can continue serving
        exported artifacts if the main controller container terminates after
        export but before the operator downloads them.

        Workers and RecordProcessors are external worker-pod services managed
        by JobSet.
        """
        containers = self._create_control_plane_containers()
        containers.extend(self._create_optional_manager_containers())
        containers.append(self._create_results_sidecar())

        if K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED:
            # Prepend so the kubelet begins pulling/starting the proxy before
            # the control-plane container races to publish on it.
            containers.insert(0, self._create_event_bus_proxy_container())

        return containers

    def create_worker_containers(
        self, controller_dns: str
    ) -> list[AIPerfContainerSpec]:
        """Create worker-pod containers with one container per runtime service.

        The worker pod keeps a lightweight worker-group-manager for shared pod
        infrastructure (dataset download once per pod, local raw-inference
        proxy, raw-record upload coordination), while each worker and record
        processor runs in its own container instead of a subprocess.
        """
        worker_count = self._resolve_workers_per_pod()
        record_processor_count = self._resolve_record_processors_per_pod()
        manager_port, worker_ports, record_processor_ports = (
            allocate_worker_health_ports(worker_count, record_processor_count)
        )
        resources = self._split_worker_pod_resources(
            worker_count, record_processor_count
        )

        containers: list[AIPerfContainerSpec] = [
            self._create_container(
                name="worker-group-manager",
                service_type="worker_group_manager",
                service_id="worker_group_manager_$(AIPERF_POD_INDEX)",
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

    # ------------------------------------------------------------------ replicated-job assembly

    def build_controller_replicated_job(
        self, volumes: list[dict[str, Any]]
    ) -> AIPerfReplicatedJobSpec:
        """Build the controller replicatedJob spec with prometheus annotations."""
        jobset_config = K8sEnvironment.JOBSET
        api_port = K8sEnvironment.PORTS.API_SERVICE
        return AIPerfReplicatedJobSpec(
            name="controller",
            replicas=1,
            containers=self.create_controller_containers(),
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

    def build_worker_replicated_job(
        self, volumes: list[dict[str, Any]], controller_dns: str
    ) -> AIPerfReplicatedJobSpec:
        """Build the workers replicatedJob spec."""
        jobset_config = K8sEnvironment.JOBSET
        return AIPerfReplicatedJobSpec(
            name="workers",
            replicas=self.worker_replicas,
            containers=self.create_worker_containers(controller_dns),
            volumes=volumes,
            restart_policy=RestartPolicy.ON_FAILURE,
            backoff_limit=jobset_config.WORKER_BACKOFF_LIMIT,
            # Workers are ephemeral and must be garbage-collected immediately
            # once the benchmark ends; the outer JobSet carries the real TTL.
            job_ttl_seconds=None if self.keep_failed_pods else 0,
            pod_template=self.pod_template,
            job_id=self.job_id,
        )

    def to_k8s_manifest(self) -> dict[str, Any]:
        """Generate the complete JobSet Kubernetes manifest."""
        controller_dns = controller_dns_name(self.name, self.namespace)
        volumes = build_shared_volumes(self.name, self.pod_template)

        controller_job = self.build_controller_replicated_job(volumes)
        worker_job = self.build_worker_replicated_job(volumes, controller_dns)

        metadata: dict[str, Any] = {
            "name": self.name,
            "namespace": self.namespace,
            "labels": self._build_manifest_labels(),
        }
        if self.extra_annotations:
            metadata["annotations"] = self.extra_annotations

        manifest: dict[str, Any] = {
            "apiVersion": JOBSET_API_VERSION,
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

        # Kueue requires JobSets to start suspended; it unsuspends after
        # admission. This must use the same resolver as the queue label: it
        # keyed on scheduling.queue_name alone while the label also honored
        # the operator-wide env default, so an admin setting
        # AIPERF_K8S_JOBSET_KUEUE_DEFAULT_QUEUE_NAME plus a CR omitting
        # queueName produced a queue-labelled but UNsuspended JobSet, which
        # runs immediately and bypasses Kueue gang admission entirely.
        if self._resolved_queue_name():
            manifest["spec"]["suspend"] = True

        ttl = self._resolve_manifest_ttl()
        if ttl is not None:
            manifest["spec"]["ttlSecondsAfterFinished"] = ttl

        return manifest
