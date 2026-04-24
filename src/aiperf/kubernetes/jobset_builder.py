# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal builders for AIPerf JobSet manifest generation.

Separates container/pod/replicated-job construction from the public
:class:`aiperf.kubernetes.jobset.AIPerfJobSetSpec` configuration model so the
spec module stays small. Not part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common.environment import Environment
from aiperf.kubernetes.constants import Containers
from aiperf.kubernetes.enums import RestartPolicy
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset_helpers import (
    build_container_args,
    build_container_ports,
    build_env_vars,
    build_health_probe,
    build_security_context,
    build_service_probes,
    build_startup_probe,
    build_volume_mounts,
)
from aiperf.kubernetes.jobset_resources import (
    allocate_worker_health_ports,
    split_worker_pod_resources,
)
from aiperf.kubernetes.jobset_specs import AIPerfContainerSpec, AIPerfReplicatedJobSpec

if TYPE_CHECKING:
    from aiperf.kubernetes.jobset import AIPerfJobSetSpec


class _JobSetManifestBuilder:
    """Render container, pod, and replicated-job fragments for an AIPerfJobSetSpec.

    Instantiated per :meth:`AIPerfJobSetSpec.to_k8s_manifest` call. Keeps the
    spec as a reference; all state lives on the spec.
    """

    def __init__(self, spec: AIPerfJobSetSpec) -> None:
        self.spec = spec

    # ------------------------------------------------------------------ resource resolution

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
        if self.spec.resource_mode == "none":
            return None
        return getattr(K8sEnvironment, settings_key).to_k8s_resources(
            burstable=self.spec.resource_mode == "burstable"
        )

    def _resolve_workers_per_pod(self) -> int:
        """Resolve workers per pod for manifest generation."""
        return self.spec.workers_per_pod or Environment.WORKER.DEFAULT_WORKERS_PER_POD

    def _resolve_record_processors_per_pod(self) -> int:
        """Resolve record processors per pod for manifest generation."""
        if self.spec.record_processors_per_pod is not None:
            return self.spec.record_processors_per_pod
        return max(
            1,
            self._resolve_workers_per_pod()
            // K8sEnvironment.RECORD_PROCESSOR_SCALE_FACTOR,
        )

    def _pod_template_env_value(self, name: str) -> str | None:
        """Return a string value from podTemplate env when present."""
        for item in self.spec.pod_template.env:
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
            burstable=self.spec.resource_mode == "burstable",
        )

    # ------------------------------------------------------------------ env

    def _create_env_vars(
        self,
        controller_host: str | None = None,
        include_pod_index: bool = True,
    ) -> list[dict[str, Any]]:
        """Create environment variables for a container."""
        return build_env_vars(
            job_id=self.spec.job_id,
            namespace=self.spec.namespace,
            pod_template=self.spec.pod_template,
            controller_host=controller_host,
            include_pod_index=include_pod_index,
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
        )
        if extra_env:
            env.extend(extra_env)

        probe_port = api_port or health_port
        startup_probe, liveness_probe, readiness_probe = build_service_probes(
            probe_port,
            skip_startup_probe=skip_startup_probe,
            skip_liveness_probe=skip_liveness_probe,
            skip_readiness_probe=skip_readiness_probe,
        )

        return AIPerfContainerSpec(
            name=name,
            image=self.spec.image,
            image_pull_policy=self.spec.image_pull_policy,
            command=["aiperf"],
            args=args,
            env=env,
            resources=resources,
            volume_mounts=build_volume_mounts(self.spec.pod_template),
            ports=ports,
            startup_probe=startup_probe,
            liveness_probe=liveness_probe,
            readiness_probe=readiness_probe,
            security_context=build_security_context(self.spec.pod_template),
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
            {"containerPort": 5663, "name": "pub-frontend"},
            {"containerPort": 5664, "name": "sub-backend"},
        ]

        return AIPerfContainerSpec(
            name=Containers.EVENT_BUS_PROXY,
            image=self.spec.image,
            image_pull_policy=self.spec.image_pull_policy,
            command=["aiperf"],
            args=args,
            env=self._create_env_vars(include_pod_index=False),
            resources=self._resolve_pod_resources("EVENT_BUS_PROXY"),
            volume_mounts=build_volume_mounts(self.spec.pod_template),
            ports=container_ports,
            startup_probe=build_startup_probe(health_port),
            liveness_probe=build_health_probe(health_port),
            readiness_probe=build_health_probe(health_port, path="/readyz"),
            security_context=build_security_context(self.spec.pod_template),
        )

    def _create_results_sidecar(self) -> AIPerfContainerSpec:
        """Build the small results-serving sidecar container."""
        ports = K8sEnvironment.PORTS
        return AIPerfContainerSpec(
            name=Containers.RESULTS_SIDECAR,
            image=self.spec.image,
            image_pull_policy=self.spec.image_pull_policy,
            command=["python", "-m", "aiperf.kubernetes.results_sidecar"],
            env=[
                {"name": "AIPERF_RESULTS_DIR", "value": "/results"},
                {
                    "name": "AIPERF_RESULTS_SIDECAR_PORT",
                    "value": str(ports.RESULTS_SIDECAR),
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
            security_context=build_security_context(self.spec.pod_template),
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
                health_port=None,
                resources=self._resolve_pod_resources("API"),
                api_port=K8sEnvironment.PORTS.API_SERVICE,
                service_id="api",
                include_pod_index=False,
            ),
        ]

    def _create_optional_manager_containers(self) -> list[AIPerfContainerSpec]:
        """Build GPU telemetry + server metrics containers when enabled."""
        ports = K8sEnvironment.PORTS
        managers: list[AIPerfContainerSpec] = []
        if self.spec.gpu_telemetry_enabled:
            managers.append(
                self._create_container(
                    name=Containers.GPU_TELEMETRY_MANAGER,
                    service_type="gpu_telemetry_manager",
                    health_port=ports.GPU_TELEMETRY_MANAGER_HEALTH,
                    resources=self._resolve_pod_resources("GPU_TELEMETRY_MANAGER"),
                    service_id="gpu_telemetry_manager",
                    include_pod_index=False,
                )
            )
        if self.spec.server_metrics_enabled:
            managers.append(
                self._create_container(
                    name=Containers.SERVER_METRICS_MANAGER,
                    service_type="server_metrics_manager",
                    health_port=ports.SERVER_METRICS_MANAGER_HEALTH,
                    resources=self._resolve_pod_resources("SERVER_METRICS_MANAGER"),
                    service_id="server_metrics_manager",
                    include_pod_index=False,
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
            pod_template=self.spec.pod_template,
            job_id=self.spec.job_id,
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
            replicas=self.spec.worker_replicas,
            containers=self.create_worker_containers(controller_dns),
            volumes=volumes,
            restart_policy=RestartPolicy.ON_FAILURE,
            backoff_limit=jobset_config.WORKER_BACKOFF_LIMIT,
            # Workers are ephemeral and must be garbage-collected immediately
            # once the benchmark ends; the outer JobSet carries the real TTL.
            job_ttl_seconds=None if self.spec.keep_failed_pods else 0,
            pod_template=self.spec.pod_template,
            job_id=self.spec.job_id,
        )
