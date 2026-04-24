# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.jobset_builder.

Exercises the internal _JobSetManifestBuilder: its resolvers for workers/
record-processors per pod, container factories (control plane, worker,
event-bus proxy, results sidecar), and the assembled replicated-job specs.
"""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.config.deployment import PodTemplateConfig
from aiperf.kubernetes.constants import Containers
from aiperf.kubernetes.enums import RestartPolicy
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import AIPerfJobSetSpec
from aiperf.kubernetes.jobset_builder import _JobSetManifestBuilder


def _make_spec(**kwargs: Any) -> AIPerfJobSetSpec:
    """Build an AIPerfJobSetSpec with sensible defaults for builder tests."""
    defaults: dict[str, Any] = {
        "name": "bench",
        "namespace": "default",
        "job_id": "bench-abc",
        "image": "aiperf:latest",
    }
    defaults.update(kwargs)
    return AIPerfJobSetSpec(**defaults)


class TestResolveWorkersPerPod:
    """_resolve_workers_per_pod falls back to Environment.WORKER default."""

    def test_uses_spec_override(self) -> None:
        builder = _JobSetManifestBuilder(_make_spec(workers_per_pod=12))
        assert builder._resolve_workers_per_pod() == 12

    def test_falls_back_to_environment_default(self) -> None:
        """When workers_per_pod is None, the builder uses the shared Environment default."""
        builder = _JobSetManifestBuilder(_make_spec(workers_per_pod=None))
        assert (
            builder._resolve_workers_per_pod()
            == Environment.WORKER.DEFAULT_WORKERS_PER_POD
        )


class TestResolveRecordProcessorsPerPod:
    """Record processors default to workers_per_pod // scale_factor (min 1)."""

    def test_explicit_value_wins(self) -> None:
        builder = _JobSetManifestBuilder(
            _make_spec(workers_per_pod=10, record_processors_per_pod=4)
        )
        assert builder._resolve_record_processors_per_pod() == 4

    def test_derived_from_workers_and_scale_factor(self) -> None:
        """Default: max(1, workers_per_pod // RECORD_PROCESSOR_SCALE_FACTOR)."""
        scale = K8sEnvironment.RECORD_PROCESSOR_SCALE_FACTOR
        workers = scale * 3
        builder = _JobSetManifestBuilder(
            _make_spec(workers_per_pod=workers, record_processors_per_pod=None)
        )
        assert builder._resolve_record_processors_per_pod() == 3

    def test_minimum_is_one_even_for_tiny_worker_counts(self) -> None:
        """One worker should still produce at least one record processor."""
        builder = _JobSetManifestBuilder(
            _make_spec(workers_per_pod=1, record_processors_per_pod=None)
        )
        assert builder._resolve_record_processors_per_pod() == 1

    def test_zero_explicit_value_is_respected(self) -> None:
        """Setting record_processors_per_pod=0 must not be overridden by the min-1 floor."""
        builder = _JobSetManifestBuilder(
            _make_spec(workers_per_pod=10, record_processors_per_pod=0)
        )
        assert builder._resolve_record_processors_per_pod() == 0


class TestResolvePodResources:
    """_resolve_pod_resources honors the resource_mode selector."""

    def test_none_mode_returns_none(self) -> None:
        """resource_mode='none' is an escape hatch: emit no requests/limits."""
        builder = _JobSetManifestBuilder(_make_spec(resource_mode="none"))
        assert builder._resolve_pod_resources("SYSTEM_CONTROLLER") is None

    def test_guaranteed_mode_has_matching_requests_and_limits(self) -> None:
        builder = _JobSetManifestBuilder(_make_spec(resource_mode="guaranteed"))
        resources = builder._resolve_pod_resources("SYSTEM_CONTROLLER")
        assert resources is not None
        assert "requests" in resources
        assert "limits" in resources
        assert resources["requests"] == resources["limits"]

    def test_burstable_mode_omits_limits(self) -> None:
        builder = _JobSetManifestBuilder(_make_spec(resource_mode="burstable"))
        resources = builder._resolve_pod_resources("SYSTEM_CONTROLLER")
        assert resources is not None
        assert "requests" in resources
        assert "limits" not in resources


class TestPodTemplateEnvValue:
    """_pod_template_env_value reads string values off the pod template env list."""

    def test_reads_matching_value(self) -> None:
        spec = _make_spec(
            pod_template=PodTemplateConfig(env=[{"name": "MY_KEY", "value": "42"}])
        )
        builder = _JobSetManifestBuilder(spec)
        assert builder._pod_template_env_value("MY_KEY") == "42"

    def test_returns_none_when_absent(self) -> None:
        builder = _JobSetManifestBuilder(_make_spec())
        assert builder._pod_template_env_value("MISSING") is None

    def test_ignores_non_string_values_like_value_from(self) -> None:
        """valueFrom-style env entries have no 'value' string; treat as not-set."""
        spec = _make_spec(
            pod_template=PodTemplateConfig(
                env=[
                    {
                        "name": "FROM_SECRET",
                        "valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}},
                    }
                ]
            )
        )
        builder = _JobSetManifestBuilder(spec)
        assert builder._pod_template_env_value("FROM_SECRET") is None


class TestCreateEventBusProxyContainer:
    """Event-bus proxy sidecar exposes the three expected pub/sub ports."""

    def test_container_name_and_image(self) -> None:
        builder = _JobSetManifestBuilder(_make_spec())
        container = builder._create_event_bus_proxy_container()
        assert container.name == Containers.EVENT_BUS_PROXY
        assert container.image == "aiperf:latest"

    def test_ports_include_health_and_pubsub(self) -> None:
        """Health + pub-frontend (5663) + sub-backend (5664) ports must be declared."""
        builder = _JobSetManifestBuilder(_make_spec())
        container = builder._create_event_bus_proxy_container()
        port_names = {p["name"] for p in container.ports}
        assert port_names == {"health", "pub-frontend", "sub-backend"}

    def test_args_invoke_event_bus_proxy(self) -> None:
        """CLI args must select the event_bus proxy kind."""
        builder = _JobSetManifestBuilder(_make_spec())
        container = builder._create_event_bus_proxy_container()
        assert container.args[0] == "proxy"
        assert "event_bus" in container.args
        assert "--health-port" in container.args

    def test_env_omits_pod_index(self) -> None:
        """Controller sidecar is single-replica so AIPERF_POD_INDEX is not injected."""
        builder = _JobSetManifestBuilder(_make_spec())
        container = builder._create_event_bus_proxy_container()
        names = {item["name"] for item in container.env}
        assert "AIPERF_POD_INDEX" not in names


class TestCreateResultsSidecar:
    """Results sidecar exposes results read-only and has its own probes."""

    def test_mounts_results_readonly_only(self) -> None:
        """Results sidecar must mount /results read-only (no IPC/datasets)."""
        builder = _JobSetManifestBuilder(_make_spec())
        container = builder._create_results_sidecar()
        by_name = {m["name"]: m for m in container.volume_mounts}
        assert "results" in by_name
        assert by_name["results"]["readOnly"] is True
        assert "ipc" not in by_name
        assert "datasets" not in by_name

    def test_env_sets_results_dir(self) -> None:
        builder = _JobSetManifestBuilder(_make_spec())
        container = builder._create_results_sidecar()
        env = {item["name"]: item["value"] for item in container.env}
        assert env["AIPERF_RESULTS_DIR"] == "/results"
        assert env["AIPERF_RESULTS_SIDECAR_PORT"] == str(
            K8sEnvironment.PORTS.RESULTS_SIDECAR
        )

    def test_command_uses_python_module_entry(self) -> None:
        """Sidecar runs as a python -m invocation, not the aiperf CLI."""
        builder = _JobSetManifestBuilder(_make_spec())
        container = builder._create_results_sidecar()
        assert container.command == [
            "python",
            "-m",
            "aiperf.kubernetes.results_sidecar",
        ]


class TestCreateControlPlaneContainers:
    """Control-plane pod contains exactly the five mandatory services."""

    def test_five_mandatory_containers_present(self) -> None:
        builder = _JobSetManifestBuilder(_make_spec())
        containers = builder._create_control_plane_containers()
        names = [c.name for c in containers]
        assert names == [
            Containers.CONTROL_PLANE,
            Containers.DATASET_MANAGER,
            Containers.TIMING_MANAGER,
            Containers.RECORDS_MANAGER,
            Containers.API,
        ]

    def test_records_manager_has_no_probes(self) -> None:
        """Records manager opts out of all probes; it manages its own lifecycle."""
        builder = _JobSetManifestBuilder(_make_spec())
        records = next(
            c
            for c in builder._create_control_plane_containers()
            if c.name == Containers.RECORDS_MANAGER
        )
        assert records.startup_probe is None
        assert records.liveness_probe is None
        assert records.readiness_probe is None

    def test_api_container_declares_api_port(self) -> None:
        """API container is the only one that exposes the HTTP API port."""
        builder = _JobSetManifestBuilder(_make_spec())
        api = next(
            c
            for c in builder._create_control_plane_containers()
            if c.name == Containers.API
        )
        port_names = {p["name"] for p in api.ports}
        assert "api" in port_names

    def test_control_plane_realtime_metrics_env(self) -> None:
        """Control-plane container must enable realtime metrics explicitly."""
        builder = _JobSetManifestBuilder(_make_spec())
        cp = next(
            c
            for c in builder._create_control_plane_containers()
            if c.name == Containers.CONTROL_PLANE
        )
        env = {item["name"]: item["value"] for item in cp.env if "value" in item}
        assert env["AIPERF_UI_REALTIME_METRICS_ENABLED"] == "true"


class TestCreateOptionalManagerContainers:
    """GPU telemetry and server metrics managers gate on spec flags."""

    @pytest.mark.parametrize(
        "gpu_enabled,server_enabled,expected_names",
        [
            param(False, False, [], id="both-disabled"),
            param(
                True,
                False,
                [Containers.GPU_TELEMETRY_MANAGER],
                id="gpu-only",
            ),
            param(
                False,
                True,
                [Containers.SERVER_METRICS_MANAGER],
                id="server-only",
            ),
            param(
                True,
                True,
                [
                    Containers.GPU_TELEMETRY_MANAGER,
                    Containers.SERVER_METRICS_MANAGER,
                ],
                id="both",
            ),
        ],
    )  # fmt: skip
    def test_optional_managers_gated(
        self,
        gpu_enabled: bool,
        server_enabled: bool,
        expected_names: list[str],
    ) -> None:
        """GPU and server-metrics containers must only appear when enabled."""
        builder = _JobSetManifestBuilder(
            _make_spec(
                gpu_telemetry_enabled=gpu_enabled,
                server_metrics_enabled=server_enabled,
            )
        )
        containers = builder._create_optional_manager_containers()
        assert [c.name for c in containers] == expected_names


class TestCreateControllerContainers:
    """create_controller_containers assembles the full controller pod list."""

    def test_results_sidecar_always_last(self) -> None:
        """Results sidecar must be last so earlier containers declare probes before it."""
        builder = _JobSetManifestBuilder(
            _make_spec(gpu_telemetry_enabled=False, server_metrics_enabled=False)
        )
        containers = builder.create_controller_containers()
        assert containers[-1].name == Containers.RESULTS_SIDECAR

    def test_event_bus_proxy_prepended_when_enabled(self) -> None:
        """When event-bus sidecar is enabled, it must be the first container."""
        original = K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED
        K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED = True
        try:
            builder = _JobSetManifestBuilder(_make_spec())
            containers = builder.create_controller_containers()
            assert containers[0].name == Containers.EVENT_BUS_PROXY
        finally:
            K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED = original

    def test_event_bus_proxy_absent_when_disabled(self) -> None:
        original = K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED
        K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED = False
        try:
            builder = _JobSetManifestBuilder(_make_spec())
            containers = builder.create_controller_containers()
            names = [c.name for c in containers]
            assert Containers.EVENT_BUS_PROXY not in names
        finally:
            K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED = original


class TestCreateWorkerContainers:
    """Worker pod contains one manager + N workers + M record processors."""

    def test_container_counts_match_resolved_shape(self) -> None:
        """1 worker-group-manager + workers_per_pod workers + record processors."""
        spec = _make_spec(workers_per_pod=3, record_processors_per_pod=2)
        builder = _JobSetManifestBuilder(spec)
        containers = builder.create_worker_containers("controller.svc")
        # 1 manager + 3 workers + 2 record processors = 6 total
        assert len(containers) == 6
        assert containers[0].name == "worker-group-manager"
        worker_names = [
            c.name
            for c in containers
            if c.name.startswith("worker-") and c.name != "worker-group-manager"
        ]
        assert worker_names == ["worker-0", "worker-1", "worker-2"]
        rp_names = [
            c.name for c in containers if c.name.startswith("record-processor-")
        ]
        assert rp_names == ["record-processor-0", "record-processor-1"]

    def test_each_worker_has_unique_health_port(self) -> None:
        """Workers share a network namespace so health ports must not collide."""
        spec = _make_spec(workers_per_pod=4, record_processors_per_pod=2)
        builder = _JobSetManifestBuilder(spec)
        containers = builder.create_worker_containers("controller.svc")
        health_ports = []
        for c in containers:
            for p in c.ports:
                if p["name"] == "health":
                    health_ports.append(p["containerPort"])
        assert len(health_ports) == len(set(health_ports))

    def test_controller_host_injected_for_worker_containers(self) -> None:
        """Every worker container must see the controller DNS name via env."""
        spec = _make_spec(workers_per_pod=2, record_processors_per_pod=1)
        builder = _JobSetManifestBuilder(spec)
        containers = builder.create_worker_containers("controller.bench.svc")
        for c in containers:
            host_env = next(
                (e for e in c.env if e["name"] == "AIPERF_K8S_ZMQ_CONTROLLER_HOST"),
                None,
            )
            assert host_env is not None
            assert host_env["value"] == "controller.bench.svc"

    def test_all_worker_probes_skipped(self) -> None:
        """Worker containers skip every probe since the manager handles lifecycle."""
        spec = _make_spec(workers_per_pod=2, record_processors_per_pod=1)
        builder = _JobSetManifestBuilder(spec)
        for c in builder.create_worker_containers("controller.svc"):
            assert c.startup_probe is None
            assert c.liveness_probe is None
            assert c.readiness_probe is None

    def test_worker_service_ids_include_pod_index_placeholder(self) -> None:
        """Service IDs reference $(AIPERF_POD_INDEX) for kubelet substitution."""
        spec = _make_spec(workers_per_pod=2, record_processors_per_pod=1)
        builder = _JobSetManifestBuilder(spec)
        containers = builder.create_worker_containers("controller.svc")
        worker_0 = next(c for c in containers if c.name == "worker-0")
        service_id = worker_0.args[worker_0.args.index("--service-id") + 1]
        assert "$(AIPERF_POD_INDEX)" in service_id
        assert service_id.endswith("_0")


class TestBuildControllerReplicatedJob:
    """Controller replicated-job carries prometheus annotations and backoff limits."""

    def test_replicas_is_always_one(self) -> None:
        """Controller is singleton; replicas must be exactly 1."""
        builder = _JobSetManifestBuilder(_make_spec(worker_replicas=10))
        job = builder.build_controller_replicated_job(volumes=[])
        assert job.replicas == 1

    def test_restart_policy_is_never(self) -> None:
        """Controller uses Never so a crashed controller fails the job, not restarts in place."""
        builder = _JobSetManifestBuilder(_make_spec())
        job = builder.build_controller_replicated_job(volumes=[])
        assert job.restart_policy == RestartPolicy.NEVER

    def test_backoff_limit_from_environment(self) -> None:
        builder = _JobSetManifestBuilder(_make_spec())
        job = builder.build_controller_replicated_job(volumes=[])
        assert job.backoff_limit == K8sEnvironment.JOBSET.CONTROLLER_BACKOFF_LIMIT

    def test_prometheus_annotations_attached(self) -> None:
        """Prometheus scrape/port/path annotations must be on the controller pod."""
        builder = _JobSetManifestBuilder(_make_spec())
        job = builder.build_controller_replicated_job(volumes=[])
        ann = job.extra_annotations
        assert ann["prometheus.io/scrape"] == "true"
        assert ann["prometheus.io/port"] == str(K8sEnvironment.PORTS.API_SERVICE)
        assert ann["prometheus.io/path"] == "/metrics"

    def test_job_id_propagated_to_replicated_job(self) -> None:
        builder = _JobSetManifestBuilder(_make_spec(job_id="run-42"))
        job = builder.build_controller_replicated_job(volumes=[])
        assert job.job_id == "run-42"


class TestBuildWorkerReplicatedJob:
    """Worker replicated-job uses the spec's worker_replicas and on-failure restart."""

    def test_replicas_from_spec(self) -> None:
        builder = _JobSetManifestBuilder(_make_spec(worker_replicas=5))
        job = builder.build_worker_replicated_job(
            volumes=[], controller_dns="controller.bench.svc"
        )
        assert job.replicas == 5

    def test_restart_policy_is_on_failure(self) -> None:
        """Workers restart on failure so transient faults don't fail the whole job."""
        builder = _JobSetManifestBuilder(_make_spec())
        job = builder.build_worker_replicated_job(
            volumes=[], controller_dns="controller.svc"
        )
        assert job.restart_policy == RestartPolicy.ON_FAILURE

    def test_backoff_limit_from_environment(self) -> None:
        builder = _JobSetManifestBuilder(_make_spec())
        job = builder.build_worker_replicated_job(
            volumes=[], controller_dns="controller.svc"
        )
        assert job.backoff_limit == K8sEnvironment.JOBSET.WORKER_BACKOFF_LIMIT

    @pytest.mark.parametrize(
        "keep_failed_pods,expected_ttl",
        [
            param(True, None, id="keep-failed-preserves-worker-ttl-none"),
            param(False, 0, id="default-zeros-worker-ttl-for-immediate-gc"),
        ],
    )  # fmt: skip
    def test_worker_ttl_depends_on_keep_failed_pods(
        self, keep_failed_pods: bool, expected_ttl: int | None
    ) -> None:
        """When keep_failed_pods=False, workers must be garbage-collected immediately."""
        builder = _JobSetManifestBuilder(_make_spec(keep_failed_pods=keep_failed_pods))
        job = builder.build_worker_replicated_job(
            volumes=[], controller_dns="controller.svc"
        )
        assert job.job_ttl_seconds == expected_ttl

    def test_volumes_are_passed_through(self) -> None:
        """Input volumes must appear verbatim on the replicated-job spec."""
        volumes = [{"name": "custom", "emptyDir": {}}]
        builder = _JobSetManifestBuilder(_make_spec())
        job = builder.build_worker_replicated_job(
            volumes=volumes, controller_dns="controller.svc"
        )
        assert job.volumes == volumes


class TestSplitWorkerPodResourcesWiring:
    """Builder wraps split_worker_pod_resources with the correct pinned CPU request."""

    def test_uses_pod_template_override_when_present(self) -> None:
        """An env var on the pod template pins the record-processor CPU request."""
        spec = _make_spec(
            workers_per_pod=2,
            record_processors_per_pod=1,
            pod_template=PodTemplateConfig(
                env=[
                    {
                        "name": "AIPERF_K8S_RECORD_PROCESSOR_CPU_REQUEST",
                        "value": "750m",
                    }
                ]
            ),
        )
        builder = _JobSetManifestBuilder(spec)
        result = builder._split_worker_pod_resources(
            worker_count=2, record_processor_count=1
        )
        # Last entry is the record processor container
        assert result[-1] is not None
        assert result[-1]["requests"]["cpu"] == "750m"

    def test_returns_none_per_container_when_budget_absent(self) -> None:
        """resource_mode='none' propagates None through to each container slot."""
        spec = _make_spec(resource_mode="none")
        builder = _JobSetManifestBuilder(spec)
        result = builder._split_worker_pod_resources(
            worker_count=2, record_processor_count=1
        )
        assert result == [None, None, None, None]
