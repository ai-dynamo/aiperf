# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes pod-template and JobSet manifest construction.

Focuses on:
- controller, worker, and records-manager container env/resource contracts
- shareProcessNamespace propagation from CR and operator environment defaults
- results-sidecar port/env/base-URL wiring for AIPerfJob and AIPerfSweep pods
- owner metadata, reserved label protection, and structured image/env preservation
- volume/volumeMount consistency across generated controller and worker pods

Out of scope: kopf create-handler lifecycle/error transitions, covered by
``tests/unit/operator/handlers/test_create_adversarial.py`` and
``tests/unit/operator/test_sweep_handler_adversarial.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import pytest
from aiperf.operator.handlers.sweep import create as sweep_create
from pytest import param

from aiperf.kubernetes.constants import AIPerfLabels, Containers
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.operator.handlers.create import _build_deployment, _create_jobset
from tests.harness.operator import build_minimal_aiperfjob_spec

# =============================================================================
# Helpers
# =============================================================================


def _benchmark_spec_with(**overrides: Any) -> dict[str, Any]:
    """Build a real AIPerfJob spec by mutating the canonical validated baseline."""
    return {**build_minimal_aiperfjob_spec(), **overrides}


def _pod_template_spec(**pod_template_overrides: Any) -> dict[str, Any]:
    """Build an AIPerfJob spec with top-level ``podTemplate`` overrides."""
    return _benchmark_spec_with(
        image="registry.example.com/aiperf:manifest-adversarial",
        podTemplate={**pod_template_overrides},
    )


def _manifest_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Render a JobSet manifest through the production deployment converter path."""
    deployment, _ = _build_deployment(
        spec=spec,
        name="aiperf-bench-7f2a",
        namespace="production",
        job_id="aiperf-bench-7f2a",
    )
    return deployment.get_jobset_spec().to_k8s_manifest()


def _replicated_job(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a named JobSet replicatedJob fragment."""
    for replicated_job in manifest["spec"]["replicatedJobs"]:
        if replicated_job["name"] == name:
            return replicated_job
    raise AssertionError(f"replicatedJob {name!r} not found")


def _pod_spec(manifest: dict[str, Any], replicated_job_name: str) -> dict[str, Any]:
    """Return a replicatedJob pod spec from a rendered JobSet manifest."""
    replicated_job = _replicated_job(manifest, replicated_job_name)
    return replicated_job["template"]["spec"]["template"]["spec"]


def _pod_metadata(manifest: dict[str, Any], replicated_job_name: str) -> dict[str, Any]:
    """Return a replicatedJob pod-template metadata block."""
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


def _env_by_name(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index Kubernetes EnvVar entries by name while preserving valueFrom payloads."""
    return {entry["name"]: entry for entry in container.get("env", [])}


class TestEndpointCredentialTransport:
    """Endpoint secrets stay out of ConfigMaps and re-enter through Secret env."""

    def test_inline_api_key_without_secret_env_is_rejected(self) -> None:
        benchmark = build_minimal_aiperfjob_spec()["benchmark"]
        benchmark["endpoint"] = {
            **benchmark["endpoint"],
            "api_key": "plaintext-must-not-enter-configmap",
        }

        with pytest.raises(ValueError, match="AIPERF_INJECTED_API_KEY"):
            _build_deployment(
                spec=_benchmark_spec_with(benchmark=benchmark),
                name="credential-bench",
                namespace="production",
                job_id="credential-bench",
            )

    def test_secret_backed_openai_key_keeps_configmap_redacted(self) -> None:
        benchmark = build_minimal_aiperfjob_spec()["benchmark"]
        benchmark["endpoint"] = {
            **benchmark["endpoint"],
            "api_key": "plaintext-must-not-enter-configmap",
        }
        spec = _benchmark_spec_with(
            benchmark=benchmark,
            podTemplate={
                "env": [
                    {
                        "name": "OPENAI_API_KEY",
                        "valueFrom": {
                            "secretKeyRef": {"name": "llm-api", "key": "api-key"}
                        },
                    }
                ]
            },
        )

        deployment, _ = _build_deployment(
            spec=spec,
            name="credential-bench",
            namespace="production",
            job_id="credential-bench",
        )
        configmap = deployment.get_configmap_spec().to_k8s_manifest()
        manifest = deployment.get_jobset_spec().to_k8s_manifest()

        run_json = configmap["data"]["run_config.json"]
        assert "plaintext-must-not-enter-configmap" not in run_json
        assert "<redacted>" in run_json
        for replicated_job in manifest["spec"]["replicatedJobs"]:
            for container in _pod_spec(manifest, replicated_job["name"])["containers"]:
                if container["name"] == "results-sidecar":
                    continue
                assert _env_by_name(container)["OPENAI_API_KEY"]["valueFrom"] == {
                    "secretKeyRef": {"name": "llm-api", "key": "api-key"}
                }

    @pytest.mark.parametrize(
        ("endpoint_override", "required_env"),
        [
            param(
                {"headers": {"Authorization": "Bearer real-secret"}},
                "AIPERF_INJECTED_HEADERS",
                id="sensitive-header",
            ),
            param(
                {"urls": ["http://user:password@server/v1"]},
                "AIPERF_INJECTED_ENDPOINT_URLS",
                id="url-userinfo",
            ),
            param(
                {"urls": ["http://<redacted>@server/v1"]},
                "AIPERF_INJECTED_ENDPOINT_URLS",
                id="redacted-url-userinfo",
            ),
        ],
    )  # fmt: skip
    def test_other_credentials_require_matching_secret_env(
        self,
        endpoint_override: dict[str, Any],
        required_env: str,
    ) -> None:
        benchmark = build_minimal_aiperfjob_spec()["benchmark"]
        benchmark["endpoint"] = {**benchmark["endpoint"], **endpoint_override}

        with pytest.raises(ValueError, match=required_env):
            _build_deployment(
                spec=_benchmark_spec_with(benchmark=benchmark),
                name="credential-bench",
                namespace="production",
                job_id="credential-bench",
            )


class TestConfigMapIntentPreservation:
    """Authored defaults survive operator conversion without inventing omitted ones."""

    @pytest.mark.parametrize("explicit", [False, True])
    def test_default_valued_fields_preserve_authored_presence(
        self, explicit: bool
    ) -> None:
        import orjson

        from aiperf.config import BenchmarkRun

        benchmark = build_minimal_aiperfjob_spec()["benchmark"]
        if explicit:
            benchmark["endpoint"]["streaming"] = False
            benchmark["datasets"][0]["prompts"] = {"cacheBust": {"target": "none"}}
            benchmark["phases"]["trajectoryStartMinRatio"] = 0.0
            benchmark["artifacts"] = {"autoPlot": False}

        deployment, _ = _build_deployment(
            spec=_benchmark_spec_with(benchmark=benchmark),
            name="intent-bench",
            namespace="production",
            job_id="intent-bench",
        )
        run_payload = orjson.loads(
            deployment.get_configmap_spec().data["run_config.json"]
        )
        config_payload = run_payload["cfg"]

        assert ("streaming" in config_payload["endpoint"]) is explicit
        assert (
            "cache_bust" in config_payload["datasets"][0].get("prompts", {})
        ) is explicit
        assert ("trajectory_start_min_ratio" in config_payload["phases"][0]) is explicit
        assert ("auto_plot" in config_payload["artifacts"]) is explicit

        restored = BenchmarkRun.model_validate(run_payload)
        assert restored.cfg.endpoint._streaming_explicitly_set is explicit
        prompts = restored.cfg.datasets[0].prompts
        if explicit:
            assert prompts is not None
            assert prompts.cache_bust._target_explicitly_set is True
        else:
            assert prompts is None
        assert (
            restored.cfg.phases[0]._trajectory_start_min_ratio_explicitly_set
            is explicit
        )
        assert ("auto_plot" in restored.cfg.artifacts.model_fields_set) is explicit


@asynccontextmanager
async def _fake_k8s_client() -> AsyncIterator[MagicMock]:
    """Yield a mock ApiClient without opening a real Kubernetes connection."""
    yield MagicMock(name="ApiClient")


async def _capture_sweep_jobset_body(
    monkeypatch: pytest.MonkeyPatch,
    *,
    template_spec: dict[str, Any],
) -> dict[str, Any]:
    """Run the sweep JobSet builder and capture the body sent to the apiserver."""
    captured: dict[str, Any] = {}

    async def _capture_create(**kwargs: Any) -> None:
        captured["body"] = kwargs["body"]

    custom = MagicMock(
        create_namespaced_custom_object=AsyncMock(side_effect=_capture_create)
    )
    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client",
        lambda **_kwargs: _fake_k8s_client(),
        raising=True,
    )
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi",
        lambda _api: custom,
    )

    await sweep_create._create_sweep_controller_jobset(
        name="latency-grid",
        namespace="production",
        sweep_uid="uid-latency-grid",
        epoch="1714000000",
        template_spec=template_spec,
    )
    return captured["body"]


# =============================================================================
# Controller and worker container env/resource construction
# =============================================================================


class TestAIPerfJobContainerConstruction:
    """Generated containers carry the env and resources needed to boot services."""

    @pytest.mark.parametrize(
        "container_name,expected_service,expected_health_port",
        [
            param(Containers.CONTROL_PLANE, "system_controller", 8080, id="system-controller"),
            param(Containers.RECORDS_MANAGER, "records_manager", 8084, id="records-manager"),
            param(Containers.API, "api", 9090, id="api-service"),
        ],
    )  # fmt: skip
    def test_controller_container_service_args_env_and_resources_are_rendered(
        self, container_name: str, expected_service: str, expected_health_port: int
    ) -> None:
        manifest = _manifest_from_spec(_pod_template_spec())
        container = _container(manifest, "controller", container_name)
        env = _env_by_name(container)

        assert container["image"] == "registry.example.com/aiperf:manifest-adversarial"
        assert container["command"] == ["aiperf"]
        assert container["args"][:4] == [
            "service",
            "--type",
            expected_service,
            "--benchmark-run",
        ]
        if expected_health_port != 9090:
            health_index = container["args"].index("--health-port")
            assert container["args"][health_index + 1] == str(expected_health_port)
        assert env["AIPERF_CONTROLLER_POD"] == {
            "name": "AIPERF_CONTROLLER_POD",
            "value": "1",
        }
        assert env["AIPERF_UI_REALTIME_METRICS_ENABLED"]["value"] == "true"
        assert env["AIPERF_JOB_ID"]["value"] == "aiperf-bench-7f2a"
        assert "AIPERF_POD_INDEX" not in env
        assert container["resources"]["requests"]["cpu"]
        assert "limits" not in container["resources"]

    def test_worker_and_record_processor_containers_have_unique_ids_and_split_resources(
        self,
    ) -> None:
        spec = _benchmark_spec_with(
            image="registry.example.com/aiperf:workers",
            connectionsPerWorker=1,
            benchmark={
                **build_minimal_aiperfjob_spec()["benchmark"],
                "runtime": {"workersPerPod": 2, "recordProcessorsPerPod": 2},
                "phases": {
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 4,
                },
            },
        )

        manifest = _manifest_from_spec(spec)
        worker_pod = _pod_spec(manifest, "workers")
        containers = {
            container["name"]: container for container in worker_pod["containers"]
        }

        assert _replicated_job(manifest, "workers")["replicas"] == 2
        assert set(containers) == {
            "worker-group-manager",
            "worker-0",
            "worker-1",
            "record-processor-0",
            "record-processor-1",
        }
        assert containers["worker-0"]["args"][-2:] == [
            "--service-id",
            "worker_$(AIPERF_POD_INDEX)_0",
        ]
        assert containers["record-processor-1"]["args"][-2:] == [
            "--service-id",
            "record_processor_$(AIPERF_POD_INDEX)_1",
        ]
        for runtime_container in containers.values():
            env = _env_by_name(runtime_container)
            assert "AIPERF_POD_INDEX" in env
            assert env["AIPERF_K8S_ZMQ_CONTROLLER_HOST"]["value"].startswith(
                "aiperf-aiperf-bench-7f2a-controller-0-0."
            )
            assert runtime_container["resources"]["requests"]["cpu"]
            assert "limits" not in runtime_container["resources"]

    def test_guaranteed_resource_mode_sets_controller_and_worker_limits_equal_requests(
        self,
    ) -> None:
        spec = _benchmark_spec_with(
            resourceMode="guaranteed",
            benchmark={
                **build_minimal_aiperfjob_spec()["benchmark"],
                "runtime": {"workersPerPod": 1, "recordProcessorsPerPod": 1},
            },
        )
        manifest = _manifest_from_spec(spec)

        for replicated_job_name in ("controller", "workers"):
            for container in _pod_spec(manifest, replicated_job_name)["containers"]:
                resources = container["resources"]
                assert resources["limits"] == resources["requests"]


# =============================================================================
# Pod-template propagation, structured values, and labels
# =============================================================================


class TestPodTemplateAdversarial:
    """User podTemplate fields survive the trust-boundary conversion intact."""

    def test_share_process_namespace_true_propagates_to_controller_and_worker_pods(
        self,
    ) -> None:
        manifest = _manifest_from_spec(_pod_template_spec(shareProcessNamespace=True))

        assert _pod_spec(manifest, "controller")["shareProcessNamespace"] is True
        assert _pod_spec(manifest, "workers")["shareProcessNamespace"] is True

    def test_share_process_namespace_env_default_only_fills_missing_cr_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(K8sEnvironment, "SHARE_PROCESS_NAMESPACE", True)
        manifest = _manifest_from_spec(_benchmark_spec_with())

        assert _pod_spec(manifest, "controller")["shareProcessNamespace"] is True
        assert _pod_spec(manifest, "workers")["shareProcessNamespace"] is True

    def test_image_and_env_metacharacters_remain_structured_not_shell_joined(
        self,
    ) -> None:
        spec = _pod_template_spec(
            env=[
                {
                    "name": "HTTP_PROXY",
                    "value": "http://proxy.internal:8080/a b?token=$(danger);rm -rf /",
                },
                {
                    "name": "AIPERF_EXTRA_CONFIG",
                    "valueFrom": {
                        "secretKeyRef": {"name": "bench-secret", "key": "payload.json"}
                    },
                },
            ]
        )
        spec["image"] = "registry.example.com/aiperf:quote space;still-a-tag"
        manifest = _manifest_from_spec(spec)
        control_plane = _container(manifest, "controller", Containers.CONTROL_PLANE)
        worker = _container(manifest, "workers", "worker-0")
        env = _env_by_name(control_plane)

        assert (
            control_plane["image"]
            == "registry.example.com/aiperf:quote space;still-a-tag"
        )
        assert worker["image"] == "registry.example.com/aiperf:quote space;still-a-tag"
        assert control_plane["command"] == ["aiperf"]
        assert (
            env["HTTP_PROXY"]["value"]
            == "http://proxy.internal:8080/a b?token=$(danger);rm -rf /"
        )
        assert env["AIPERF_EXTRA_CONFIG"]["valueFrom"] == {
            "secretKeyRef": {"name": "bench-secret", "key": "payload.json"}
        }

    def test_reserved_aiperf_pod_labels_cannot_be_overwritten_by_pod_template_labels(
        self,
    ) -> None:
        manifest = _manifest_from_spec(
            _pod_template_spec(
                labels={
                    "app": "not-aiperf",
                    "aiperf.nvidia.com/job-id": "hijacked-job-id",
                    "owner": "ml-platform",
                }
            )
        )

        for replicated_job_name in ("controller", "workers"):
            labels = _pod_metadata(manifest, replicated_job_name)["labels"]
            assert labels["app"] == "aiperf"
            assert labels[AIPerfLabels.JOB_ID] == "aiperf-bench-7f2a"
            assert labels["owner"] == "ml-platform"

    def test_controller_pod_annotations_keep_prometheus_defaults_over_user_collisions(
        self,
    ) -> None:
        manifest = _manifest_from_spec(
            _pod_template_spec(
                annotations={
                    "prometheus.io/scrape": "false",
                    "prometheus.io/port": "1",
                    "dashboards.example.com/team": "inference-platform",
                }
            )
        )

        annotations = _pod_metadata(manifest, "controller")["annotations"]
        assert annotations["prometheus.io/scrape"] == "true"
        assert annotations["prometheus.io/port"] == "9090"
        assert annotations["prometheus.io/path"] == "/metrics"
        assert annotations["dashboards.example.com/team"] == "inference-platform"


# =============================================================================
# Results sidecars and owner metadata
# =============================================================================


class TestResultsSidecarAndOwnerMetadata:
    """Sidecar endpoints and owner metadata stay harvestable after failures."""

    def test_aiperfjob_results_sidecar_port_env_and_mount_contract_is_consistent(
        self,
    ) -> None:
        manifest = _manifest_from_spec(_pod_template_spec())
        sidecar = _container(manifest, "controller", Containers.RESULTS_SIDECAR)
        env = _env_by_name(sidecar)

        assert sidecar["command"] == [
            "python",
            "-m",
            "aiperf.kubernetes.results_sidecar",
        ]
        assert env["AIPERF_RESULTS_DIR"]["value"] == "/results"
        assert env["AIPERF_RESULTS_SIDECAR_PORT"]["value"] == str(
            K8sEnvironment.PORTS.RESULTS_SIDECAR
        )
        assert sidecar["ports"] == [
            {"containerPort": K8sEnvironment.PORTS.RESULTS_SIDECAR, "name": "results"}
        ]
        assert {mount["name"]: mount for mount in sidecar["volumeMounts"]} == {
            "results": {"name": "results", "mountPath": "/results", "readOnly": True},
            "tmp": {"name": "tmp", "mountPath": "/tmp"},
        }

    @pytest.mark.asyncio
    async def test_create_jobset_appends_aiperfjob_owner_reference_without_dropping_labels(
        self,
    ) -> None:
        captured: dict[str, Any] = {}

        async def _capture_custom_object(**kwargs: Any) -> None:
            captured["body"] = kwargs["body"]

        deployment, _ = _build_deployment(
            spec=_pod_template_spec(labels={"owner": "platform-benchmarks"}),
            name="aiperf-bench-7f2a",
            namespace="production",
            job_id="aiperf-bench-7f2a",
        )
        owner_ref = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "name": "aiperf-bench-7f2a",
            "uid": "uid-aiperf-bench-7f2a",
            "controller": True,
            "blockOwnerDeletion": True,
        }

        with mock_patch(
            "aiperf.operator.handlers.create.create_idempotent_custom_object",
            new=AsyncMock(side_effect=_capture_custom_object),
        ):
            jobset_name = await _create_jobset(
                MagicMock(name="ApiClient"), deployment, "production", owner_ref
            )

        assert jobset_name == "aiperf-aiperf-bench-7f2a"
        assert captured["body"]["metadata"]["ownerReferences"] == [owner_ref]
        assert (
            captured["body"]["metadata"]["labels"][AIPerfLabels.JOB_ID]
            == "aiperf-bench-7f2a"
        )

    @pytest.mark.asyncio
    async def test_sweep_controller_jobset_results_sidecar_and_operator_base_url_are_rendered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment

        monkeypatch.setattr(
            OperatorEnvironment.SERVICE,
            "BASE_URL",
            "https://operator-results.production.example:9443",
        )
        body = await _capture_sweep_jobset_body(
            monkeypatch,
            template_spec={
                "image": "registry.example.com/aiperf:sweep-controller",
                "podTemplate": {
                    "env": [
                        {"name": "HTTP_PROXY", "value": "http://proxy.internal:8080"},
                        {"name": "AIPERF_SWEEP_NAME", "value": "attacker"},
                    ],
                },
            },
        )
        pod_spec = body["spec"]["replicatedJobs"][0]["template"]["spec"]["template"][
            "spec"
        ]
        containers = {
            container["name"]: container for container in pod_spec["containers"]
        }
        controller_env = _env_by_name(containers["sweep-controller"])
        sidecar_env = _env_by_name(containers[Containers.RESULTS_SIDECAR])

        assert (
            controller_env["AIPERF_OPERATOR_BASE_URL"]["value"]
            == "https://operator-results.production.example:9443"
        )
        assert controller_env["AIPERF_SWEEP_NAME"]["value"] == "latency-grid"
        assert controller_env["HTTP_PROXY"]["value"] == "http://proxy.internal:8080"
        assert sidecar_env["AIPERF_RESULTS_SIDECAR_PORT"]["value"] == str(
            K8sEnvironment.PORTS.RESULTS_SIDECAR
        )
        assert containers[Containers.RESULTS_SIDECAR]["ports"] == [
            {"containerPort": K8sEnvironment.PORTS.RESULTS_SIDECAR, "name": "results"}
        ]
        assert body["metadata"]["ownerReferences"][0] == {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfSweep",
            "name": "latency-grid",
            "uid": "uid-latency-grid",
            "controller": True,
            "blockOwnerDeletion": True,
        }


# =============================================================================
# Volume and volumeMount consistency
# =============================================================================


class TestVolumeMountConsistency:
    """Every generated mount references a volume available in that pod spec."""

    @pytest.mark.parametrize(
        "replicated_job_name",
        [
            "controller",
            "workers",
        ],
    )  # fmt: skip
    def test_all_container_volume_mounts_reference_declared_pod_volumes(
        self, replicated_job_name: str
    ) -> None:
        manifest = _manifest_from_spec(
            _pod_template_spec(
                volumes=[{"name": "hf-token", "secret": {"secretName": "hf-token"}}],
                volumeMounts=[
                    {"name": "hf-token", "mountPath": "/var/run/secrets/hf-token"}
                ],
            )
        )
        pod_spec = _pod_spec(manifest, replicated_job_name)
        volume_names = {volume["name"] for volume in pod_spec["volumes"]}

        for container in pod_spec["containers"]:
            for mount in container.get("volumeMounts", []):
                assert mount["name"] in volume_names

    def test_core_runtime_mounts_are_present_on_controller_and_worker_containers(
        self,
    ) -> None:
        manifest = _manifest_from_spec(_pod_template_spec())

        for replicated_job_name in ("controller", "workers"):
            for container in _pod_spec(manifest, replicated_job_name)["containers"]:
                mount_names = {mount["name"] for mount in container["volumeMounts"]}
                if container["name"] == Containers.RESULTS_SIDECAR:
                    assert mount_names == {"results", "tmp"}
                else:
                    assert {
                        "config",
                        "ipc",
                        "results",
                        "datasets",
                        "tokenizer-cache",
                        "tmp",
                    }.issubset(mount_names)
