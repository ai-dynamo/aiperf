# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes JobSet spec generation.

Focuses on:
- reserved AIPerf labels and controller annotations staying authoritative
- JobSet ownerReferences boundaries, service DNS names, and replicatedJob names
- container args/env list-shape safety for shell-like user input
- results sidecar port/probe contract used by artifact harvesting
- podTemplate metadata merge behavior and invalid resource-shape rejection

Out of scope: live Kubernetes API submission and kopf create-handler retries; see
``tests/unit/operator/test_sweep_handler_adversarial.py`` for handler-level
state-machine and patch-shape regressions.
"""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.config.deployment import PodTemplateConfig
from aiperf.kubernetes.constants import AIPerfLabels, Containers
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import AIPerfJobSetSpec, controller_dns_name
from aiperf.kubernetes.jobset_helpers import CELL_CONTROLLER_PORT
from aiperf.kubernetes.jobset_specs import AIPerfContainerSpec

# ============================================================
# Helpers
# ============================================================


def _jobset_spec(**overrides: Any) -> AIPerfJobSetSpec:
    """Build a real JobSet spec with realistic benchmark identifiers."""
    baseline: dict[str, Any] = {
        "name": "aiperf-bench-7f2a",
        "namespace": "aiperf-benchmarks",
        "job_id": "aiperf-bench-7f2a",
        "image": "nvcr.io/nvidia/aiperf:custom-test-tag",
        "worker_replicas": 2,
        "workers_per_pod": 2,
        "record_processors_per_pod": 1,
    }
    baseline.update(overrides)
    return AIPerfJobSetSpec(**baseline)


def _manifest(**overrides: Any) -> dict[str, Any]:
    """Render the JobSet manifest from a real Pydantic spec."""
    return _jobset_spec(**overrides).to_k8s_manifest()


def _replicated_job(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one rendered replicatedJob by JobSet-level name."""
    return next(
        job for job in manifest["spec"]["replicatedJobs"] if job["name"] == name
    )


def _pod_metadata(replicated_job: dict[str, Any]) -> dict[str, Any]:
    """Return the nested PodTemplate metadata for a replicatedJob."""
    return replicated_job["template"]["spec"]["template"]["metadata"]


def _pod_spec(replicated_job: dict[str, Any]) -> dict[str, Any]:
    """Return the nested PodSpec for a replicatedJob."""
    return replicated_job["template"]["spec"]["template"]["spec"]


def _container(replicated_job: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one rendered container by name from a replicatedJob."""
    return next(
        container
        for container in _pod_spec(replicated_job)["containers"]
        if container["name"] == name
    )


def _env_by_name(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a container env list by variable name."""
    return {item["name"]: item for item in container.get("env", [])}


# ============================================================
# Reserved metadata and ownership boundaries
# ============================================================


class TestJobSetSpecReservedMetadata:
    """Reserved labels/annotations must beat user-provided podTemplate metadata."""

    def test_pod_labels_user_collision_preserves_aiperf_reserved_values(self) -> None:
        template = PodTemplateConfig(
            labels={
                AIPerfLabels.APP_KEY: "rogue-app",
                AIPerfLabels.JOB_ID: "rogue-benchmark",
                "team.nvidia.com/owner": "perf-infra",
            }
        )

        manifest = _manifest(pod_template=template)

        for replicated_job_name in ("controller", "cells"):
            labels = _pod_metadata(_replicated_job(manifest, replicated_job_name))[
                "labels"
            ]
            assert labels[AIPerfLabels.APP_KEY] == AIPerfLabels.APP_VALUE
            assert labels[AIPerfLabels.JOB_ID] == "aiperf-bench-7f2a"
            assert labels["team.nvidia.com/owner"] == "perf-infra"

    def test_cellular_controller_pod_carries_user_annotations_without_prometheus_override(
        self,
    ) -> None:
        """The cellular controller pod is a native runner + results sidecar, not the
        mesh control-plane API. It exposes no scrape endpoint, so unlike the retired
        mesh controller it injects no prometheus.io/* annotations -- user-supplied
        podTemplate annotations pass through verbatim."""
        template = PodTemplateConfig(
            annotations={
                "prometheus.io/scrape": "false",
                "prometheus.io/port": "1",
                "prometheus.io/path": "/do-not-scrape",
                "team.nvidia.com/owner": "perf-infra",
            }
        )

        manifest = _manifest(pod_template=template)
        annotations = _pod_metadata(_replicated_job(manifest, "controller"))[
            "annotations"
        ]

        # No forced prometheus override: the user's values survive unchanged.
        assert annotations["prometheus.io/scrape"] == "false"
        assert annotations["prometheus.io/port"] == "1"
        assert annotations["prometheus.io/path"] == "/do-not-scrape"
        assert annotations["team.nvidia.com/owner"] == "perf-infra"

    def test_to_k8s_manifest_does_not_forge_owner_references(self) -> None:
        """OwnerReferences are injected by the kopf create handler with the live CR UID."""
        manifest = _manifest(
            extra_annotations={"aiperf.nvidia.com/requested-by": "ci-smoke-7f2a"}
        )

        assert "ownerReferences" not in manifest["metadata"]
        assert manifest["metadata"]["annotations"] == {
            "aiperf.nvidia.com/requested-by": "ci-smoke-7f2a"
        }


# ============================================================
# Names and JobSet topology
# ============================================================


class TestJobSetSpecNamesAndTopology:
    """Service names and replicatedJob names are a cross-container wire contract."""

    def test_replicated_job_names_match_success_policy_target(self) -> None:
        manifest = _manifest()

        replicated_job_names = [
            job["name"] for job in manifest["spec"]["replicatedJobs"]
        ]
        assert replicated_job_names == ["controller", "cells"]
        assert manifest["spec"]["successPolicy"] == {
            "operator": "All",
            "targetReplicatedJobs": ["controller"],
        }

    @pytest.mark.parametrize(
        "jobset_name,namespace,expected",
        [
            (
                "aiperf-bench-7f2a",
                "aiperf-benchmarks",
                "aiperf-bench-7f2a-controller-0-0."
                "aiperf-bench-7f2a.aiperf-benchmarks.svc.cluster.local",
            ),
            param(
                "latency-sweep-v03",
                "perf-canary",
                "latency-sweep-v03-controller-0-0."
                "latency-sweep-v03.perf-canary.svc.cluster.local",
                id="sweep-style-name",
            ),
        ],
    )  # fmt: skip
    def test_controller_dns_name_uses_jobset_headless_service_shape(
        self, jobset_name: str, namespace: str, expected: str
    ) -> None:
        assert controller_dns_name(jobset_name, namespace) == expected

    def test_cell_controller_addr_env_matches_jobset_dns_contract(self) -> None:
        """Each cell dials the controller via AIPERF_CELL_CONTROLLER_ADDR, which is the
        controller pod's stable JobSet headless-service DNS name plus the cell
        transport port (tcp://<controller-dns>:9500)."""
        manifest = _manifest(name="aiperf-llama3-8b", namespace="perf-canary")
        cell = _container(_replicated_job(manifest, "cells"), "cell")

        env = _env_by_name(cell)
        assert env["AIPERF_CELL_CONTROLLER_ADDR"]["value"] == (
            f"tcp://aiperf-llama3-8b-controller-0-0."
            f"aiperf-llama3-8b.perf-canary.svc.cluster.local:{CELL_CONTROLLER_PORT}"
        )
        assert manifest["spec"]["network"] == {"enableDNSHostnames": True}


# ============================================================
# Container args/env trust boundaries
# ============================================================


class TestJobSetSpecContainerArgsAndEnv:
    """Container args/env are lists of data, not shell-rendered command strings."""

    def test_service_args_remain_tokenized_when_env_contains_shell_metacharacters(
        self,
    ) -> None:
        template = PodTemplateConfig(
            env=[
                {
                    "name": "AIPERF_OPERATOR_NOTE",
                    "value": "--type api; touch /tmp/aiperf-owned $(whoami)",
                }
            ]
        )

        manifest = _manifest(pod_template=template)
        cell = _container(_replicated_job(manifest, "cells"), "cell")

        assert cell["command"] == ["aiperf"]
        assert cell["args"] == [
            "cell",
            "--config",
            f"{K8sEnvironment.JOBSET.CONFIG_MOUNT_PATH}/config.yaml",
        ]
        assert _env_by_name(cell)["AIPERF_OPERATOR_NOTE"]["value"] == (
            "--type api; touch /tmp/aiperf-owned $(whoami)"
        )

    def test_reserved_cr_identity_env_wins_over_pod_template_entry(
        self,
    ) -> None:
        """The controller pod's CR-identity env (AIPERF_JOB_ID / AIPERF_NAMESPACE, which
        the runner uses to patch the owning AIPerfJob .status) is appended after any
        user podTemplate env. Kubernetes resolves duplicate env names last-wins, so the
        authoritative identity always takes effect regardless of a rogue override."""
        template = PodTemplateConfig(
            env=[
                {"name": "AIPERF_JOB_ID", "value": "rogue-benchmark"},
                {"name": "AIPERF_NAMESPACE", "value": "rogue-namespace"},
            ]
        )

        manifest = _manifest(pod_template=template)
        controller = _container(
            _replicated_job(manifest, "controller"), Containers.CELL_CONTROLLER
        )
        env_entries = controller["env"]
        env_names = [entry["name"] for entry in env_entries]

        # The authoritative CR-identity entry is emitted after the user's, so it is
        # the last occurrence (kubelet last-wins) and resolves to the real identity.
        job_id_indices = [i for i, n in enumerate(env_names) if n == "AIPERF_JOB_ID"]
        ns_indices = [i for i, n in enumerate(env_names) if n == "AIPERF_NAMESPACE"]
        assert env_entries[job_id_indices[-1]]["value"] == "aiperf-bench-7f2a"
        assert env_entries[ns_indices[-1]]["value"] == "aiperf-benchmarks"
        env = _env_by_name(controller)
        assert env["AIPERF_JOB_ID"]["value"] == "aiperf-bench-7f2a"
        assert env["AIPERF_NAMESPACE"]["value"] == "aiperf-benchmarks"

    def test_cell_id_env_uses_field_ref_not_shell_substitution(self) -> None:
        """A cell's CELL_ID (its budget-partition index) is sourced from the JobSet
        job-index label via a downward-API fieldRef, never a shell substitution."""
        manifest = _manifest()
        cell = _container(_replicated_job(manifest, "cells"), "cell")

        cell_id = _env_by_name(cell)["AIPERF_CELL_ID"]
        assert cell_id == {
            "name": "AIPERF_CELL_ID",
            "valueFrom": {
                "fieldRef": {
                    "fieldPath": "metadata.labels['jobset.sigs.k8s.io/job-index']",
                }
            },
        }


# ============================================================
# Results sidecar contract
# ============================================================


class TestJobSetSpecResultsSidecarContract:
    """Results sidecar service/port settings must stay in lock-step."""

    def test_results_sidecar_port_env_and_probes_share_single_port_constant(
        self,
    ) -> None:
        manifest = _manifest()
        sidecar = _container(
            _replicated_job(manifest, "controller"), Containers.RESULTS_SIDECAR
        )
        expected_port = K8sEnvironment.PORTS.RESULTS_SIDECAR

        assert sidecar["command"] == ["aiperf", "results-sidecar"]
        assert sidecar["ports"] == [{"containerPort": expected_port, "name": "results"}]
        assert _env_by_name(sidecar)["AIPERF_RESULTS_SIDECAR_PORT"]["value"] == str(
            expected_port
        )
        assert sidecar["startupProbe"]["httpGet"] == {
            "path": "/healthz",
            "port": expected_port,
        }
        assert sidecar["livenessProbe"]["httpGet"]["port"] == expected_port
        assert sidecar["readinessProbe"]["httpGet"]["port"] == expected_port

    def test_results_sidecar_mounts_only_results_and_tmp_volumes(self) -> None:
        manifest = _manifest()
        sidecar = _container(
            _replicated_job(manifest, "controller"), Containers.RESULTS_SIDECAR
        )

        assert sidecar["volumeMounts"] == [
            {"name": "results", "mountPath": "/results", "readOnly": True},
            {"name": "tmp", "mountPath": "/tmp"},
        ]
        assert _env_by_name(sidecar)["AIPERF_RESULTS_DIR"]["value"] == "/results"


# ============================================================
# PodTemplate merge and invalid shapes
# ============================================================


class TestJobSetSpecPodTemplateMergeAndInvalidShapes:
    """PodTemplateConfig is a trust boundary for arbitrary K8s-native fragments."""

    def test_pod_template_extra_pod_spec_overrides_typed_security_context_last(
        self,
    ) -> None:
        template = PodTemplateConfig(
            pod_security_context={"fsGroup": 2000},
            extra_pod_spec={
                "securityContext": {
                    "runAsNonRoot": True,
                    "runAsUser": 3000,
                    "runAsGroup": 3000,
                    "fsGroup": 3000,
                    "seccompProfile": {"type": "RuntimeDefault"},
                }
            },
        )

        manifest = _manifest(pod_template=template)
        pod_spec = _pod_spec(_replicated_job(manifest, "controller"))

        assert pod_spec["securityContext"] == {
            "runAsNonRoot": True,
            "runAsUser": 3000,
            "runAsGroup": 3000,
            "fsGroup": 3000,
            "seccompProfile": {"type": "RuntimeDefault"},
        }

    @pytest.mark.parametrize(
        "resources",
        [
            param(["cpu=500m"], id="list-when-mapping"),
            param({"requests": ["cpu=500m"]}, id="nested-list-when-mapping"),
            param({"requests": {"cpu": ["500m"]}}, id="list-when-quantity-string"),
        ],
    )  # fmt: skip
    def test_container_resources_malformed_shape_raises_with_field_name(
        self, resources: object
    ) -> None:
        with pytest.raises(ValueError, match="resources"):
            AIPerfContainerSpec(
                name="worker-0",
                image="nvcr.io/nvidia/aiperf:custom-test-tag",
                resources=resources,
            )

    def test_pod_template_negative_termination_grace_rejected_with_field_name(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="termination_grace_period_seconds"):
            PodTemplateConfig(termination_grace_period_seconds=-1)
