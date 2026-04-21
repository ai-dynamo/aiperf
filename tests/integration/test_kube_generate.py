# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for `aiperf kube generate` command.

These tests invoke the CLI directly and verify the YAML output without
needing a Kubernetes cluster.
"""

from __future__ import annotations

import subprocess

import pytest
import yaml


def _run_generate(*extra_args: str) -> subprocess.CompletedProcess:
    """Run `aiperf kube generate` with common flags and return result."""
    cmd = [
        "uv",
        "run",
        "aiperf",
        "kube",
        "generate",
        "--model",
        "test-model",
        "--url",
        "http://localhost:8000",
        "--image",
        "aiperf:latest",
        *extra_args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


class TestGenerateRequiresMode:
    """Verify --operator or --no-operator is required."""

    def test_no_flag_exits_with_error(self) -> None:
        result = _run_generate()
        assert result.returncode != 0
        assert "--operator" in result.stderr or "--no-operator" in result.stderr

    def test_both_flags_exits_with_error(self) -> None:
        result = _run_generate("--operator", "--no-operator")
        assert result.returncode != 0
        assert "Cannot use both" in result.stderr


class TestGenerateOperatorMode:
    """Verify --operator outputs an AIPerfJob CR."""

    def test_outputs_aiperfjob_cr(self) -> None:
        result = _run_generate("--operator")
        assert result.returncode == 0

        doc = yaml.safe_load(result.stdout)
        assert doc["apiVersion"] == "aiperf.nvidia.com/v1alpha1"
        assert doc["kind"] == "AIPerfJob"
        assert "spec" in doc

    def test_cr_has_required_spec_fields(self) -> None:
        result = _run_generate("--operator")
        doc = yaml.safe_load(result.stdout)
        spec = doc["spec"]

        # AIPerfConfig fields live under spec.benchmark; deployment fields
        # (image, podTemplate, scheduling) live at spec top level.
        assert "benchmark" in spec
        assert "models" in spec["benchmark"]
        assert "endpoint" in spec["benchmark"]
        assert "image" in spec
        assert spec["image"] == "aiperf:latest"

    def test_cr_has_metadata(self) -> None:
        result = _run_generate("--operator", "--name", "my-bench")
        doc = yaml.safe_load(result.stdout)

        assert doc["metadata"]["name"] == "my-bench"
        assert doc["metadata"]["namespace"] == "aiperf-benchmarks"

    def test_cr_custom_namespace(self) -> None:
        result = _run_generate("--operator", "--namespace", "custom-ns")
        doc = yaml.safe_load(result.stdout)

        assert doc["metadata"]["namespace"] == "custom-ns"


class TestGenerateNoOperatorMode:
    """Verify --no-operator outputs raw K8s manifests."""

    @pytest.fixture
    def manifests(self) -> list[dict]:
        result = _run_generate("--no-operator")
        assert result.returncode == 0
        return list(yaml.safe_load_all(result.stdout))

    def test_outputs_multiple_documents(self, manifests: list[dict]) -> None:
        assert len(manifests) >= 4

    def test_contains_role(self, manifests: list[dict]) -> None:
        kinds = [m["kind"] for m in manifests]
        assert "Role" in kinds

    def test_contains_rolebinding(self, manifests: list[dict]) -> None:
        kinds = [m["kind"] for m in manifests]
        assert "RoleBinding" in kinds

    def test_contains_configmap(self, manifests: list[dict]) -> None:
        kinds = [m["kind"] for m in manifests]
        assert "ConfigMap" in kinds

    def test_contains_jobset(self, manifests: list[dict]) -> None:
        kinds = [m["kind"] for m in manifests]
        assert "JobSet" in kinds

    def test_no_aiperfjob_cr(self, manifests: list[dict]) -> None:
        kinds = [m["kind"] for m in manifests]
        assert "AIPerfJob" not in kinds

    def test_jobset_has_controller_and_workers(self, manifests: list[dict]) -> None:
        jobset = next(m for m in manifests if m["kind"] == "JobSet")
        rj_names = [rj["name"] for rj in jobset["spec"]["replicatedJobs"]]
        assert "controller" in rj_names
        assert "workers" in rj_names

    def test_worker_job_has_ttl_zero(self, manifests: list[dict]) -> None:
        jobset = next(m for m in manifests if m["kind"] == "JobSet")
        worker_rj = next(
            rj for rj in jobset["spec"]["replicatedJobs"] if rj["name"] == "workers"
        )
        ttl = worker_rj["template"]["spec"]["ttlSecondsAfterFinished"]
        assert ttl == 0

    def test_jobset_ttl_at_least_one_hour(self, manifests: list[dict]) -> None:
        jobset = next(m for m in manifests if m["kind"] == "JobSet")
        ttl = jobset["spec"]["ttlSecondsAfterFinished"]
        assert ttl >= 28800

    def test_configmap_has_run_config(self, manifests: list[dict]) -> None:
        cm = next(m for m in manifests if m["kind"] == "ConfigMap")
        assert "run_config.json" in cm["data"]

    def test_all_resources_in_same_namespace(self, manifests: list[dict]) -> None:
        namespaced = [
            m
            for m in manifests
            if m["kind"] not in ("Namespace", "ClusterRole", "ClusterRoleBinding")
        ]
        namespaces = {m["metadata"]["namespace"] for m in namespaced}
        assert len(namespaces) == 1

    def test_custom_namespace(self) -> None:
        result = _run_generate("--no-operator", "--namespace", "my-ns")
        assert result.returncode == 0
        docs = list(yaml.safe_load_all(result.stdout))
        jobset = next(m for m in docs if m["kind"] == "JobSet")
        assert jobset["metadata"]["namespace"] == "my-ns"

    def test_custom_name(self) -> None:
        result = _run_generate("--no-operator", "--name", "my-bench")
        assert result.returncode == 0
        docs = list(yaml.safe_load_all(result.stdout))
        jobset = next(m for m in docs if m["kind"] == "JobSet")
        assert jobset["metadata"]["name"] == "aiperf-my-bench"

    def test_jobset_has_dns_hostnames(self, manifests: list[dict]) -> None:
        jobset = next(m for m in manifests if m["kind"] == "JobSet")
        assert jobset["spec"]["network"]["enableDNSHostnames"] is True

    def test_security_context_non_root(self, manifests: list[dict]) -> None:
        jobset = next(m for m in manifests if m["kind"] == "JobSet")
        controller_rj = next(
            rj for rj in jobset["spec"]["replicatedJobs"] if rj["name"] == "controller"
        )
        pod_spec = controller_rj["template"]["spec"]["template"]["spec"]
        assert pod_spec["securityContext"]["runAsNonRoot"] is True
        assert pod_spec["securityContext"]["runAsUser"] == 1000
