# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial Helm rendering tests for the AIPerf operator chart.

Focuses on chart values that silently break production operator installs when
rendered incorrectly:
- results-server port propagation into Service, Deployment probes, and
  ``AIPERF_OPERATOR_BASE_URL``.
- metrics port disabled-at-zero behavior across all scrape surfaces.
- chart-default namespace drift against ``DEFAULT_OPERATOR_NAMESPACE``.
- share-process-namespace env plumbing used by chaos tests.
- resource override preservation for operator and results-server containers.
- schema rejection for typo, type-confusion, and out-of-range values.

Out of scope: live Helm install/upgrade behavior; see
``tests/kubernetes/test_helm.py`` for cluster-backed coverage.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pytest import param

from aiperf.kubernetes.constants import DEFAULT_OPERATOR_NAMESPACE

CHART_PATH = Path(__file__).parents[3] / "deploy" / "helm" / "aiperf-operator"
PROJECT_ROOT = Path(__file__).parents[3]


def _helm_available() -> bool:
    return shutil.which("helm") is not None


def _helm_template(
    *extra: str,
    namespace: str = DEFAULT_OPERATOR_NAMESPACE,
    release: str = "aiperf-operator",
) -> list[dict]:
    """Render the chart with Prometheus APIs present and return YAML documents."""
    cmd = [
        "helm",
        "template",
        release,
        str(CHART_PATH),
        "-n",
        namespace,
        "--api-versions",
        "monitoring.coreos.com/v1",
        *extra,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed for release {release!r} in namespace {namespace!r}: "
            f"{result.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _helm_template_failure(*extra: str) -> subprocess.CompletedProcess[str]:
    """Run ``helm template`` expecting values.schema.json to reject the input."""
    cmd = [
        "helm",
        "template",
        "aiperf-operator",
        str(CHART_PATH),
        "-n",
        DEFAULT_OPERATOR_NAMESPACE,
        *extra,
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
        timeout=120,
    )


def _find(docs: list[dict], kind: str, name: str) -> dict:
    """Return the rendered Kubernetes object identified by kind and name."""
    for doc in docs:
        if doc.get("kind") == kind and doc.get("metadata", {}).get("name") == name:
            return doc
    available = [(doc.get("kind"), doc.get("metadata", {}).get("name")) for doc in docs]
    raise AssertionError(f"{kind}/{name} not found in chart output. Got: {available}")


def _operator_container(docs: list[dict]) -> dict:
    deploy = _find(docs, "Deployment", "aiperf-operator")
    return next(
        container
        for container in deploy["spec"]["template"]["spec"]["containers"]
        if container["name"] == "operator"
    )


def _results_server_container(docs: list[dict]) -> dict:
    deploy = _find(docs, "Deployment", "aiperf-operator")
    return next(
        container
        for container in deploy["spec"]["template"]["spec"]["containers"]
        if container["name"] == "results-server"
    )


def _env_by_name(container: dict) -> dict[str, str]:
    return {env["name"]: env["value"] for env in container.get("env", [])}


def _port_by_name(container: dict) -> dict[str, int]:
    return {port["name"]: port["containerPort"] for port in container.get("ports", [])}


# ============================================================
# resultsServer.port propagates to every HTTP surface
# ============================================================


@pytest.mark.skipif(not _helm_available(), reason="helm CLI not installed")
class TestResultsServerPortAdversarial:
    """The FastAPI sidecar port must be one value across all rendered objects."""

    @pytest.mark.parametrize(
        "port",
        [
            1,
            8081,
            param(65535, id="max-valid-port"),
        ],
    )  # fmt: skip
    def test_results_server_port_valid_boundary_propagates_everywhere(
        self, port: int
    ) -> None:
        docs = _helm_template("--set", f"resultsServer.port={port}")

        service = _find(docs, "Service", "aiperf-operator")
        service_ports = {item["name"]: item for item in service["spec"]["ports"]}
        assert service_ports["results"]["port"] == port
        assert service_ports["results"]["targetPort"] == "results"

        results_server = _results_server_container(docs)
        result_ports = _port_by_name(results_server)
        result_env = _env_by_name(results_server)
        assert result_ports["results"] == port
        assert result_env["AIPERF_RESULTS_SERVER_PORT"] == str(port)
        assert results_server["livenessProbe"]["httpGet"]["port"] == port
        assert results_server["readinessProbe"]["httpGet"]["port"] == port

        operator_env = _env_by_name(_operator_container(docs))
        assert (
            operator_env["AIPERF_OPERATOR_BASE_URL"]
            == f"http://aiperf-operator.{DEFAULT_OPERATOR_NAMESPACE}:{port}"
        )

    @pytest.mark.parametrize(
        "port,expected_message",
        [
            param(0, "/resultsServer/port", id="zero-rejected"),
            param(65536, "/resultsServer/port", id="above-max-rejected"),
        ],
    )  # fmt: skip
    def test_results_server_port_invalid_boundary_rejected_by_schema(
        self, port: int, expected_message: str
    ) -> None:
        result = _helm_template_failure("--set", f"resultsServer.port={port}")
        assert result.returncode != 0
        assert expected_message in result.stderr


# ============================================================
# metrics port disabled-at-zero contract
# ============================================================


@pytest.mark.skipif(not _helm_available(), reason="helm CLI not installed")
class TestMetricsPortDisabledAtZero:
    """``operator.metrics.port=0`` disables every rendered metrics surface."""

    def test_operator_metrics_port_zero_preserves_disable_env_and_omits_ports(
        self,
    ) -> None:
        docs = _helm_template(
            "--set",
            "operator.metrics.port=0",
            "--set",
            "serviceMonitor.enabled=true",
            "--set",
            "networkPolicy.enabled=true",
        )

        operator = _operator_container(docs)
        assert _env_by_name(operator)["AIPERF_METRICS_PORT"] == "0"
        assert "metrics" not in _port_by_name(operator)

        service = _find(docs, "Service", "aiperf-operator")
        assert "metrics" not in {port["name"] for port in service["spec"]["ports"]}
        assert [doc for doc in docs if doc.get("kind") == "ServiceMonitor"] == []

        netpol = _find(docs, "NetworkPolicy", "aiperf-operator")
        ingress_ports = {
            port["port"]
            for rule in netpol["spec"]["ingress"]
            for port in rule.get("ports", [])
        }
        assert 0 not in ingress_ports
        assert 9090 not in ingress_ports

    def test_operator_metrics_port_override_propagates_without_touching_base_url(
        self,
    ) -> None:
        docs = _helm_template(
            "--set",
            "operator.metrics.port=32091",
            "--set",
            "serviceMonitor.enabled=true",
            "--set",
            "networkPolicy.enabled=true",
        )

        operator = _operator_container(docs)
        assert _env_by_name(operator)["AIPERF_METRICS_PORT"] == "32091"
        assert _port_by_name(operator)["metrics"] == 32091

        service = _find(docs, "Service", "aiperf-operator")
        service_ports = {port["name"]: port for port in service["spec"]["ports"]}
        assert service_ports["metrics"]["port"] == 32091

        servicemonitor = _find(docs, "ServiceMonitor", "aiperf-operator")
        assert servicemonitor["spec"]["endpoints"][0]["port"] == "metrics"

        netpol = _find(docs, "NetworkPolicy", "aiperf-operator")
        ingress_ports = {
            port["port"]
            for rule in netpol["spec"]["ingress"]
            for port in rule.get("ports", [])
        }
        assert 32091 in ingress_ports
        assert _env_by_name(operator)["AIPERF_OPERATOR_BASE_URL"].endswith(":8081")


# ============================================================
# namespace and share-process namespace values
# ============================================================


@pytest.mark.skipif(not _helm_available(), reason="helm CLI not installed")
class TestNamespaceAndShareProcessValues:
    """Chart defaults must stay aligned with Python Kubernetes constants."""

    def test_default_operator_namespace_constant_is_base_url_namespace(self) -> None:
        docs = _helm_template(namespace=DEFAULT_OPERATOR_NAMESPACE)
        env = _env_by_name(_operator_container(docs))
        assert (
            env["AIPERF_OPERATOR_BASE_URL"]
            == f"http://aiperf-operator.{DEFAULT_OPERATOR_NAMESPACE}:8081"
        )

    @pytest.mark.parametrize(
        "share_process_namespace,expected_env",
        [
            (False, "false"),
            (True, "true"),
        ],
    )  # fmt: skip
    def test_share_process_namespace_value_flows_to_operator_env(
        self, share_process_namespace: bool, expected_env: str
    ) -> None:
        docs = _helm_template(
            "--set",
            f"podTemplate.shareProcessNamespace={str(share_process_namespace).lower()}",
        )
        env = _env_by_name(_operator_container(docs))
        assert env["AIPERF_K8S_SHARE_PROCESS_NAMESPACE"] == expected_env

    def test_share_process_namespace_scalar_string_rejected_by_schema(self) -> None:
        result = _helm_template_failure(
            "--set-string",
            "podTemplate.shareProcessNamespace=true",
        )
        assert result.returncode != 0
        assert "/podTemplate/shareProcessNamespace" in result.stderr


# ============================================================
# resource override preservation
# ============================================================


@pytest.mark.skipif(not _helm_available(), reason="helm CLI not installed")
class TestResourceOverrides:
    """Resource requests and limits survive nested Helm value overrides."""

    def test_operator_and_results_server_resource_overrides_render_verbatim(
        self,
    ) -> None:
        docs = _helm_template(
            "--set",
            "operator.resources.requests.cpu=750m",
            "--set",
            "operator.resources.requests.memory=768Mi",
            "--set",
            "operator.resources.limits.cpu=1500m",
            "--set",
            "operator.resources.limits.memory=2Gi",
            "--set",
            "resultsServer.resources.requests.cpu=250m",
            "--set",
            "resultsServer.resources.requests.memory=640Mi",
            "--set",
            "resultsServer.resources.limits.cpu=900m",
            "--set",
            "resultsServer.resources.limits.memory=1536Mi",
        )

        operator_resources = _operator_container(docs)["resources"]
        assert operator_resources["requests"] == {"cpu": "750m", "memory": "768Mi"}
        assert operator_resources["limits"] == {"cpu": "1500m", "memory": "2Gi"}

        results_resources = _results_server_container(docs)["resources"]
        assert results_resources["requests"] == {"cpu": "250m", "memory": "640Mi"}
        assert results_resources["limits"] == {"cpu": "900m", "memory": "1536Mi"}


# ============================================================
# values.schema.json adversaries and generated CRD consistency
# ============================================================


@pytest.mark.skipif(not _helm_available(), reason="helm CLI not installed")
class TestValuesSchemaAdversarial:
    """Typos, bad enums, and invalid cardinalities are rejected before install."""

    @pytest.mark.parametrize(
        "set_args,expected_message",
        [
            param(
                ("--set", "operator.replicas=0"),
                "/operator/replicas",
                id="operator-replicas-zero-rejected",
            ),
            param(
                ("--set", "image.pullPolicy=Sometimes"),
                "/image/pullPolicy",
                id="image-pull-policy-enum-rejected",
            ),
            param(
                ("--set", "resultsServer.ports=8081"),
                "/resultsServer",
                id="results-server-typo-rejected",
            ),
        ],
    )  # fmt: skip
    def test_values_schema_invalid_values_rejected_with_field_context(
        self, set_args: tuple[str, str], expected_message: str
    ) -> None:
        result = _helm_template_failure(*set_args)
        assert result.returncode != 0
        assert expected_message in result.stderr


class TestGeneratedCrdConsistency:
    """Generated CRD templates stay in sync with the Python generator."""

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "tools" / "generate_crd.py").exists(),
        reason="tools/generate_crd.py not present on this branch (lives on the "
        "parallel k8s CRD-generator branch, not integrated here)",
    )
    def test_crd_templates_match_generator_check(self) -> None:
        # sys.executable (the pytest venv python) already has the project
        # installed; shelling through ``uv run`` risks a cold lock/sync.
        result = subprocess.run(
            [sys.executable, "tools/generate_crd.py", "--check"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
