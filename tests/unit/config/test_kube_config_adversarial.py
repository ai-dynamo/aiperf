# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes configuration and environment parsing.

Focuses on:
- KubeManageOptions preserving kubeconfig, kube-context, and namespace together.
- OperatorEnvironment and K8sEnvironment parsing env vars for ports, resources,
  shareProcessNamespace, and the results-server API URL.
- Invalid env values surfacing as field-specific Pydantic ValidationError entries.
- DEFAULT_OPERATOR_NAMESPACE remaining the single namespace literal for source code
  that needs the chart-default operator namespace.

Out of scope: Kubernetes API behavior and manifest rendering; see sibling tests in
``tests/unit/kubernetes/`` and ``tests/unit/operator/`` for those integration seams.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.kube import KubeManageOptions, KubeOptions
from aiperf.kubernetes.constants import DEFAULT_OPERATOR_NAMESPACE
from aiperf.kubernetes.environment import (
    _K8sEnvironment,
    _PortSettings,
    _resource_settings,
)
from aiperf.kubernetes.results import _kubectl_kube_args
from aiperf.operator.environment import (
    _DashboardSettings,
    _OperatorEnvironment,
    _OperatorServiceSettings,
)

# =============================================================================
# Helpers
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_AIPERF = _REPO_ROOT / "src" / "aiperf"
_ALLOWED_OPERATOR_NAMESPACE_LITERAL_FILES = {
    Path("kubernetes/constants.py"),
    Path("operator/environment.py"),
}


def _validation_fields(exc: ValidationError) -> set[str]:
    """Return the field path names from a Pydantic validation error."""
    return {".".join(str(part) for part in err["loc"]) for err in exc.errors()}


def _source_literals_containing(text: str) -> list[tuple[Path, int, str]]:
    """Find non-docstring string literals under ``src/aiperf`` containing text."""
    hits: list[tuple[Path, int, str]] = []
    for path in _SRC_AIPERF.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstring_nodes: set[ast.Constant] = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not body or not isinstance(body, list):
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstring_nodes.add(first.value)

        relative = path.relative_to(_SRC_AIPERF)
        for node in ast.walk(tree):
            if node in docstring_nodes:
                continue
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and text in node.value
            ):
                hits.append((relative, node.lineno, node.value))
    return hits


# =============================================================================
# KubeManageOptions / kube context propagation
# =============================================================================


class TestKubeManageOptionsAdversarial:
    """Trust-boundary tests for shared Kubernetes CLI management options."""

    def test_kube_manage_options_context_and_kubeconfig_roundtrip_together(
        self,
    ) -> None:
        options = KubeManageOptions.model_validate(
            {
                "kubeconfig": "/home/runner/.kube/dgx-prod.yaml",
                "kube_context": "kind-aiperf-chaos",
                "namespace": "aiperf-benchmarks-canary",
            }
        )

        assert options.model_dump() == {
            "kubeconfig": "/home/runner/.kube/dgx-prod.yaml",
            "kube_context": "kind-aiperf-chaos",
            "namespace": "aiperf-benchmarks-canary",
        }

    def test_kube_options_inherits_manage_context_without_dropping_deploy_fields(
        self,
    ) -> None:
        options = KubeOptions(
            image="nvcr.io/nvidia/aiperf:branch-2026-05-18",
            kubeconfig="/configs/kube/dgx-prod.yaml",
            kube_context="kind-aiperf-scaleout",
            namespace="aiperf-benchmarks",
            total_workers=32,
        )

        dumped = options.model_dump()
        assert dumped["kubeconfig"] == "/configs/kube/dgx-prod.yaml"
        assert dumped["kube_context"] == "kind-aiperf-scaleout"
        assert dumped["namespace"] == "aiperf-benchmarks"
        assert dumped["image"] == "nvcr.io/nvidia/aiperf:branch-2026-05-18"
        assert dumped["total_workers"] == 32

    @pytest.mark.parametrize(
        "kubeconfig,kube_context,expected",
        [
            (None, None, []),
            ("/configs/kube/dgx-prod.yaml", None, ["--kubeconfig", "/configs/kube/dgx-prod.yaml"]),
            (None, "kind-aiperf-chaos", ["--context", "kind-aiperf-chaos"]),
            param(
                "/configs/kube/dgx-prod.yaml",
                "kind-aiperf-chaos",
                ["--kubeconfig", "/configs/kube/dgx-prod.yaml", "--context", "kind-aiperf-chaos"],
                id="both-kubeconfig-and-context-preserved",
            ),
        ],
    )  # fmt: skip
    def test_kubectl_kube_args_context_flags_propagate_in_stable_order(
        self, kubeconfig: str | None, kube_context: str | None, expected: list[str]
    ) -> None:
        assert _kubectl_kube_args(kubeconfig, kube_context) == expected


# =============================================================================
# K8sEnvironment env parsing
# =============================================================================


class TestK8sEnvironmentEnvParsingAdversarial:
    """Environment parser tests for K8s pod ports, resources, and pod flags."""

    @pytest.mark.parametrize(
        "env_var,field,raw,expected",
        [
            ("AIPERF_K8S_PORT_API_SERVICE", "API_SERVICE", "65535", 65535),
            ("AIPERF_K8S_PORT_RESULTS_SIDECAR", "RESULTS_SIDECAR", "19091", 19091),
            ("AIPERF_K8S_PORT_EVENT_BUS_PROXY_PUB_FRONTEND", "EVENT_BUS_PROXY_PUB_FRONTEND", "56630", 56630),
        ],
    )  # fmt: skip
    def test_port_settings_env_values_parse_to_target_field(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_var: str,
        field: str,
        raw: str,
        expected: int,
    ) -> None:
        monkeypatch.setenv(env_var, raw)

        settings = _PortSettings()

        assert getattr(settings, field) == expected

    @pytest.mark.parametrize(
        "env_var,field,raw",
        [
            param("AIPERF_K8S_PORT_API_SERVICE", "API_SERVICE", "0", id="api-service-zero"),
            param("AIPERF_K8S_PORT_RESULTS_SIDECAR", "RESULTS_SIDECAR", "65536", id="results-sidecar-over-max"),
            param("AIPERF_K8S_PORT_EVENT_BUS_PROXY_SUB_BACKEND", "EVENT_BUS_PROXY_SUB_BACKEND", "not-a-port", id="event-bus-nonnumeric"),
        ],
    )  # fmt: skip
    def test_port_settings_invalid_env_values_raise_with_field_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_var: str,
        field: str,
        raw: str,
    ) -> None:
        monkeypatch.setenv(env_var, raw)

        with pytest.raises(ValidationError) as exc_info:
            _PortSettings()

        assert field in _validation_fields(exc_info.value)

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
        ],
    )  # fmt: skip
    def test_share_process_namespace_env_accepts_boolean_spellings(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
    ) -> None:
        monkeypatch.setenv("AIPERF_K8S_SHARE_PROCESS_NAMESPACE", raw)

        settings = _K8sEnvironment()

        assert settings.SHARE_PROCESS_NAMESPACE is expected

    def test_share_process_namespace_invalid_env_value_raises_with_field_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_K8S_SHARE_PROCESS_NAMESPACE", "sometimes")

        with pytest.raises(ValidationError) as exc_info:
            _K8sEnvironment()

        assert "SHARE_PROCESS_NAMESPACE" in _validation_fields(exc_info.value)

    @pytest.mark.parametrize(
        "factory_prefix,env_prefix,expected_cpu,expected_memory,burstable",
        [
            param("RESULTS_SIDECAR_", "AIPERF_K8S_RESULTS_SIDECAR_", "150m", "384Mi", False, id="results-sidecar-guaranteed"),
            param("EVENT_BUS_PROXY_", "AIPERF_K8S_EVENT_BUS_PROXY_", "250m", "128Mi", True, id="event-bus-proxy-burstable"),
        ],
    )  # fmt: skip
    def test_resource_settings_env_override_preserves_qos_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory_prefix: str,
        env_prefix: str,
        expected_cpu: str,
        expected_memory: str,
        burstable: bool,
    ) -> None:
        monkeypatch.setenv(f"{env_prefix}CPU", expected_cpu)
        monkeypatch.setenv(f"{env_prefix}MEMORY", expected_memory)

        settings = _resource_settings(factory_prefix, "25m", "192Mi")
        resources = settings.to_k8s_resources(burstable=burstable)

        assert resources["requests"] == {"cpu": expected_cpu, "memory": expected_memory}
        if burstable:
            assert "limits" not in resources
        else:
            assert resources["limits"] == resources["requests"]


# =============================================================================
# OperatorEnvironment env parsing
# =============================================================================


class TestOperatorEnvironmentEnvParsingAdversarial:
    """Environment parser tests for operator ports and results-server identity."""

    def test_operator_metrics_port_zero_disables_metrics_server(self) -> None:
        settings = _OperatorEnvironment(METRICS_PORT=0)

        assert settings.METRICS_PORT == 0

    @pytest.mark.parametrize(
        "raw,field",
        [
            param("-1", "METRICS_PORT", id="metrics-port-negative"),
            param("65536", "METRICS_PORT", id="metrics-port-over-max"),
            param("0", "PORT", id="dashboard-port-zero-is-valid-control"),
            param("65536", "PORT", id="dashboard-port-over-max"),
        ],
    )  # fmt: skip
    def test_operator_port_invalid_env_values_raise_with_field_name(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, field: str
    ) -> None:
        if field == "METRICS_PORT":
            monkeypatch.setenv("AIPERF_METRICS_PORT", raw)
            factory = _OperatorEnvironment
        else:
            monkeypatch.setenv("AIPERF_DASHBOARD_PORT", raw)
            factory = _DashboardSettings

        if raw == "0" and field == "PORT":
            assert factory().PORT == 0  # type: ignore[attr-defined]
            return

        with pytest.raises(ValidationError) as exc_info:
            factory()

        assert field in _validation_fields(exc_info.value)

    def test_operator_service_default_base_url_uses_default_namespace_and_results_port(
        self,
    ) -> None:
        settings = _OperatorServiceSettings()

        assert (
            f"http://aiperf-operator.{DEFAULT_OPERATOR_NAMESPACE}:8081"
        ) == settings.BASE_URL

    def test_operator_service_base_url_env_override_preserves_custom_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "AIPERF_OPERATOR_BASE_URL",
            "https://aiperf-operator.gpu-platform.example.com",
        )

        settings = _OperatorServiceSettings()

        assert settings.BASE_URL == "https://aiperf-operator.gpu-platform.example.com"


# =============================================================================
# DEFAULT_OPERATOR_NAMESPACE contract
# =============================================================================


class TestDefaultOperatorNamespaceAdversarial:
    """Regression locks for chart-default operator namespace usage."""

    @pytest.mark.parametrize(
        "module_path,function_name,parameter_name",
        [
            param("aiperf.kubernetes.client_pods", "find_operator_pod", "namespace", id="find-operator-pod"),
            param("aiperf.kubernetes.client_pods", "resolve_operator_namespace", "default", id="resolve-operator-namespace"),
            param("aiperf.kubernetes.results", "retrieve_sweep_results_from_operator", "operator_namespace", id="retrieve-sweep-results"),
            param("aiperf.kubernetes.results_operator", "retrieve_results_from_operator", "operator_namespace", id="retrieve-results"),
        ],
    )  # fmt: skip
    def test_public_defaults_use_default_operator_namespace_constant(
        self, module_path: str, function_name: str, parameter_name: str
    ) -> None:
        module = __import__(module_path, fromlist=[function_name])
        function = getattr(module, function_name)
        parameter = inspect.signature(function).parameters[parameter_name]

        assert parameter.default == DEFAULT_OPERATOR_NAMESPACE

    def test_source_has_no_hardcoded_operator_namespace_outside_constant_owners(
        self,
    ) -> None:
        hits = [
            hit
            for hit in _source_literals_containing(DEFAULT_OPERATOR_NAMESPACE)
            if hit[0] not in _ALLOWED_OPERATOR_NAMESPACE_LITERAL_FILES
        ]

        assert hits == []
