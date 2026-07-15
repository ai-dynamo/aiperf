# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for worker-pod and records-manager JobSet manifest wiring.

Focuses on:
- per-pod worker and record-processor fan-out in rendered Kubernetes manifests
- tokenizer cache and controller URL environment contracts across controller/worker pods
- worker-pod resource split boundaries and record-processor CPU pinning
- latency-sensitive service probe omissions and pod metadata preservation
- reserved AIPerf environment variables staying authoritative over podTemplate input

Out of scope: generic podTemplate passthrough and results-sidecar wiring, covered by
``tests/unit/operator/test_pod_template_adversarial.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.kubernetes.constants import AIPerfLabels, Containers
from aiperf.kubernetes.environment import K8sEnvironment, ResourceSettings
from aiperf.kubernetes.utils import parse_cpu
from aiperf.operator.handlers.create import _build_deployment
from tests.harness.operator import build_minimal_aiperfjob_spec

# =============================================================================
# Helpers
# =============================================================================


_RESERVED_WORKER_ENV_NAMES = frozenset(
    {
        "AIPERF_DATASET_MMAP_BASE_PATH",
        "AIPERF_JOB_ID",
        "AIPERF_NAMESPACE",
        "AIPERF_SERVICE_HEALTH_ENABLED",
        "AIPERF_SERVICE_HEALTH_HOST",
        "AIPERF_SERVICE_REGISTRATION_TIMEOUT",
        "AIPERF_POD_INDEX",
        "AIPERF_K8S_ZMQ_CONTROLLER_HOST",
        "HF_HOME",
    }
)


_RESERVED_CONTROLLER_ENV_NAMES = frozenset(
    {
        "AIPERF_DATASET_MMAP_BASE_PATH",
        "AIPERF_JOB_ID",
        "AIPERF_NAMESPACE",
        "AIPERF_SERVICE_HEALTH_ENABLED",
        "AIPERF_SERVICE_HEALTH_HOST",
        "AIPERF_SERVICE_REGISTRATION_TIMEOUT",
        "AIPERF_CONTROLLER_POD",
        "AIPERF_UI_REALTIME_METRICS_ENABLED",
        "HF_HOME",
    }
)


def _benchmark_with(**overrides: Any) -> dict[str, Any]:
    """Build a real AIPerfJob spec by mutating the canonical validated baseline."""
    return {**build_minimal_aiperfjob_spec(), **overrides}


def _benchmark_body_with(**overrides: Any) -> dict[str, Any]:
    """Build the nested ``spec.benchmark`` body with realistic model identifiers."""
    base = build_minimal_aiperfjob_spec()["benchmark"]
    return {
        **base,
        "models": ["meta-llama/Llama-3-8B-Instruct"],
        "endpoint": {
            "urls": ["http://llm-gateway.production.svc:8000/v1/chat/completions"]
        },
        **overrides,
    }


def _worker_manifest_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Render a JobSet manifest through the production AIPerfJob converter path."""
    deployment, _ = _build_deployment(
        spec=spec,
        name="latency-bench-7f2a",
        namespace="production-benchmarks",
        job_id="latency-bench-7f2a",
    )
    return deployment.get_jobset_spec().to_k8s_manifest()


def _replicated_job(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a named JobSet replicatedJob fragment."""
    for replicated_job in manifest["spec"]["replicatedJobs"]:
        if replicated_job["name"] == name:
            return replicated_job
    raise AssertionError(f"replicatedJob {name!r} not found")


def _pod_spec(manifest: dict[str, Any], replicated_job_name: str) -> dict[str, Any]:
    """Return the pod spec for a replicatedJob in a rendered JobSet manifest."""
    replicated_job = _replicated_job(manifest, replicated_job_name)
    return replicated_job["template"]["spec"]["template"]["spec"]


def _pod_metadata(manifest: dict[str, Any], replicated_job_name: str) -> dict[str, Any]:
    """Return the pod-template metadata for a replicatedJob."""
    replicated_job = _replicated_job(manifest, replicated_job_name)
    return replicated_job["template"]["spec"]["template"]["metadata"]


def _container(
    manifest: dict[str, Any], replicated_job_name: str, container_name: str
) -> dict[str, Any]:
    """Return a named container from a replicatedJob pod spec."""
    for container in _pod_spec(manifest, replicated_job_name)["containers"]:
        if container["name"] == container_name:
            return container
    raise AssertionError(
        f"container {container_name!r} not found in {replicated_job_name!r}"
    )


def _containers_by_name(
    manifest: dict[str, Any], replicated_job_name: str
) -> dict[str, dict[str, Any]]:
    """Index containers in a replicatedJob by Kubernetes container name."""
    return {
        container["name"]: container
        for container in _pod_spec(manifest, replicated_job_name)["containers"]
    }


def _env_entries(container: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Return all env entries with ``name`` so duplicate reserved keys are visible."""
    return [entry for entry in container.get("env", []) if entry["name"] == name]


def _env_by_name(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index EnvVar entries by name for cases where uniqueness is asserted separately."""
    return {entry["name"]: entry for entry in container.get("env", [])}


def _cpu_mcpu(container: dict[str, Any]) -> int:
    """Return a container CPU request as integer millicores."""
    cpu = container["resources"]["requests"]["cpu"]
    return int(round(parse_cpu(cpu) * 1000))


# =============================================================================
# Per-pod worker and record-processor fan-out
# =============================================================================


class TestWorkerManifestFanout:
    """Worker-pod manifests preserve per-pod worker and record-processor counts."""

    def test_worker_manifest_explicit_worker_and_record_processor_counts_render_exact_containers(
        self,
    ) -> None:
        spec = _benchmark_with(
            connectionsPerWorker=1,
            benchmark=_benchmark_body_with(
                runtime={"workers": 6, "workersPerPod": 3, "recordProcessorsPerPod": 2},
                phases={
                    "type": "concurrency",
                    "dataset": "main",
                    "requests": 24,
                    "concurrency": 6,
                },
            ),
        )

        manifest = _worker_manifest_from_spec(spec)
        containers = _containers_by_name(manifest, "workers")

        assert _replicated_job(manifest, "workers")["replicas"] == 2
        assert set(containers) == {
            Containers.WORKER_GROUP_MANAGER,
            "worker-0",
            "worker-1",
            "worker-2",
            "record-processor-0",
            "record-processor-1",
        }
        assert containers["worker-2"]["args"][-2:] == [
            "--service-id",
            "worker_$(AIPERF_POD_INDEX)_2",
        ]
        assert containers["record-processor-1"]["args"][-2:] == [
            "--service-id",
            "record_processor_$(AIPERF_POD_INDEX)_1",
        ]

    def test_worker_manifest_auto_record_processors_follow_per_pod_scale_factor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(K8sEnvironment, "RECORD_PROCESSOR_SCALE_FACTOR", 2)
        spec = _benchmark_with(
            connectionsPerWorker=1,
            benchmark=_benchmark_body_with(
                runtime={"workers": 8, "workersPerPod": 4},
                phases={
                    "type": "concurrency",
                    "dataset": "main",
                    "requests": 32,
                    "concurrency": 8,
                },
            ),
        )

        manifest = _worker_manifest_from_spec(spec)
        containers = _containers_by_name(manifest, "workers")

        assert _replicated_job(manifest, "workers")["replicas"] == 2
        assert [
            name for name in containers if name.startswith("record-processor-")
        ] == [
            "record-processor-0",
            "record-processor-1",
        ]

    def test_worker_manifest_health_ports_are_unique_with_dense_fanout(self) -> None:
        spec = _benchmark_with(
            connectionsPerWorker=1,
            benchmark=_benchmark_body_with(
                runtime={"workers": 5, "workersPerPod": 5, "recordProcessorsPerPod": 5},
            ),
        )

        manifest = _worker_manifest_from_spec(spec)
        containers = _containers_by_name(manifest, "workers")
        ports = [
            container["ports"][0]["containerPort"] for container in containers.values()
        ]

        assert len(ports) == len(set(ports))
        assert ports == sorted(ports)


# =============================================================================
# Tokenizer, controller URL, and reserved env trust boundaries
# =============================================================================


class TestWorkerManifestEnvironment:
    """Worker and controller env wiring survives hostile podTemplate input."""

    @pytest.mark.parametrize(
        "replicated_job_name,container_name,expected_controller_pod",
        [
            param("controller", Containers.API, "1", id="controller-api-online-tokenizer-cache"),
            param("workers", Containers.WORKER_GROUP_MANAGER, None, id="worker-manager-offline-bundle-path"),
            param("workers", "record-processor-0", None, id="record-processor-offline-bundle-path"),
        ],
    )  # fmt: skip
    def test_worker_manifest_tokenizer_env_uses_shared_cache_without_worker_controller_marker(
        self,
        replicated_job_name: str,
        container_name: str,
        expected_controller_pod: str | None,
    ) -> None:
        manifest = _worker_manifest_from_spec(_benchmark_with())
        env = _env_by_name(_container(manifest, replicated_job_name, container_name))

        assert env["HF_HOME"] == {"name": "HF_HOME", "value": "/aiperf/hf_home"}
        if expected_controller_pod is None:
            assert "AIPERF_CONTROLLER_POD" not in env
        else:
            assert env["AIPERF_CONTROLLER_POD"]["value"] == expected_controller_pod

    def test_worker_manifest_controller_dns_env_only_appears_on_worker_pod_containers(
        self,
    ) -> None:
        manifest = _worker_manifest_from_spec(_benchmark_with())
        expected = (
            "aiperf-latency-bench-7f2a-controller-0-0."
            "aiperf-latency-bench-7f2a.production-benchmarks.svc.cluster.local"
        )

        for container in _pod_spec(manifest, "workers")["containers"]:
            env = _env_by_name(container)
            assert env["AIPERF_K8S_ZMQ_CONTROLLER_HOST"] == {
                "name": "AIPERF_K8S_ZMQ_CONTROLLER_HOST",
                "value": expected,
            }
        for container in _pod_spec(manifest, "controller")["containers"]:
            env = _env_by_name(container)
            assert "AIPERF_K8S_ZMQ_CONTROLLER_HOST" not in env

    @pytest.mark.parametrize(
        "replicated_job_name,container_name,reserved_names",
        [
            param("workers", Containers.WORKER_GROUP_MANAGER, _RESERVED_WORKER_ENV_NAMES, id="worker-pod"),
            param("controller", Containers.RECORDS_MANAGER, _RESERVED_CONTROLLER_ENV_NAMES, id="controller-pod"),
        ],
    )  # fmt: skip
    def test_worker_manifest_reserved_aiperf_env_cannot_be_overwritten_by_pod_template_env(
        self,
        replicated_job_name: str,
        container_name: str,
        reserved_names: frozenset[str],
    ) -> None:
        hostile_env = [
            {"name": name, "value": "attacker-controlled"}
            for name in sorted(reserved_names)
        ]
        spec = _benchmark_with(podTemplate={"env": hostile_env})

        manifest = _worker_manifest_from_spec(spec)
        container = _container(manifest, replicated_job_name, container_name)
        env = _env_by_name(container)

        for name in reserved_names:
            assert len(_env_entries(container, name)) == 1
            assert env[name].get("value") != "attacker-controlled"


# =============================================================================
# Resources and latency-sensitive service settings
# =============================================================================


class TestWorkerManifestResourcesAndLatency:
    """Worker-pod resource splitting and probe choices match runtime contracts."""

    def test_worker_manifest_record_processor_cpu_request_override_pins_each_processor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            K8sEnvironment,
            "WORKER_POD",
            ResourceSettings(CPU="1000m", MEMORY="1000Mi"),
        )
        spec = _benchmark_with(
            podTemplate={
                "env": [
                    {"name": "AIPERF_K8S_RECORD_PROCESSOR_CPU_REQUEST", "value": "250m"}
                ]
            },
            benchmark=_benchmark_body_with(
                runtime={"workers": 4, "workersPerPod": 2, "recordProcessorsPerPod": 2},
            ),
        )

        manifest = _worker_manifest_from_spec(spec)
        containers = _containers_by_name(manifest, "workers")

        assert (
            containers["record-processor-0"]["resources"]["requests"]["cpu"] == "250m"
        )
        assert (
            containers["record-processor-1"]["resources"]["requests"]["cpu"] == "250m"
        )
        assert sum(_cpu_mcpu(container) for container in containers.values()) == 1000
        assert _cpu_mcpu(containers[Containers.WORKER_GROUP_MANAGER]) > 0
        assert _cpu_mcpu(containers["worker-0"]) > 0
        assert _cpu_mcpu(containers["worker-1"]) > 0

    def test_worker_manifest_resource_mode_none_omits_resources_from_worker_and_records_manager(
        self,
    ) -> None:
        manifest = _worker_manifest_from_spec(_benchmark_with(resourceMode="none"))

        for container in [
            *_pod_spec(manifest, "workers")["containers"],
            _container(manifest, "controller", Containers.RECORDS_MANAGER),
        ]:
            assert "resources" not in container

    @pytest.mark.parametrize(
        "replicated_job_name,container_name",
        [
            param("workers", Containers.WORKER_GROUP_MANAGER, id="worker-group-manager"),
            param("workers", "worker-0", id="worker"),
            param("workers", "record-processor-0", id="record-processor"),
            param("controller", Containers.RECORDS_MANAGER, id="records-manager"),
        ],
    )  # fmt: skip
    def test_worker_manifest_latency_sensitive_containers_omit_kubelet_probes(
        self, replicated_job_name: str, container_name: str
    ) -> None:
        manifest = _worker_manifest_from_spec(_benchmark_with())
        container = _container(manifest, replicated_job_name, container_name)

        assert "startupProbe" not in container
        assert "livenessProbe" not in container
        assert "readinessProbe" not in container


# =============================================================================
# Pod metadata
# =============================================================================


class TestWorkerManifestPodMetadata:
    """Pod labels and annotations stay compatible with operator discovery."""

    def test_worker_manifest_worker_pod_labels_keep_aiperf_identity_over_user_collisions(
        self,
    ) -> None:
        manifest = _worker_manifest_from_spec(
            _benchmark_with(
                podTemplate={
                    "labels": {
                        "app": "not-aiperf",
                        "aiperf.nvidia.com/job-id": "hijacked-job-id",
                        "team": "latency-platform",
                    }
                }
            )
        )

        labels = _pod_metadata(manifest, "workers")["labels"]

        assert labels["app"] == "aiperf"
        assert labels[AIPerfLabels.JOB_ID] == "latency-bench-7f2a"
        assert labels["team"] == "latency-platform"

    def test_worker_manifest_worker_pod_annotations_preserve_user_metadata_without_prometheus_scrape(
        self,
    ) -> None:
        manifest = _worker_manifest_from_spec(
            _benchmark_with(
                podTemplate={
                    "annotations": {
                        "dashboards.example.com/team": "latency-platform",
                        "prometheus.io/scrape": "false",
                    }
                }
            )
        )

        annotations = _pod_metadata(manifest, "workers")["annotations"]

        assert annotations == {
            "dashboards.example.com/team": "latency-platform",
            "prometheus.io/scrape": "false",
        }
