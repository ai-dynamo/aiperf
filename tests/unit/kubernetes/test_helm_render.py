# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chart-rendering invariant tests for the aiperf-operator Helm chart.

These tests shell out to ``helm template`` and assert chart-output invariants
that round-3 / round-4 audits identified as load-bearing — the kind of bug
that sails through code review because the chart YAML "looks right" but the
rendered Service/ServiceMonitor/Deployment combo silently breaks a feature.

Specifically locks in:
- The single FastAPI surface (``AIPERF_OPERATOR_BASE_URL``) is auto-templated
  from ``Release.Name`` + ``Release.Namespace`` + ``Values.resultsServer.port``.
  Caught by the round-2 collapse audit only after a live dgx run 404'd on
  ``status.apiUrl``.
- The Service exposes the metrics port when ``operator.metrics.port>0``,
  AND the ServiceMonitor scrapes ``port: metrics`` (not ``port: health``).
  Caught by round-3 chart audit — Prometheus scraping was silently broken
  in production because the Service didn't expose 9090.

Each test is a pure ``helm template`` invocation; no kind cluster, no apply.
Runs as a unit test so a regression fails fast in pre-commit / CI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

CHART_PATH = Path(__file__).parents[3] / "deploy" / "helm" / "aiperf-operator"


def _helm_available() -> bool:
    """Report whether the chart can be rendered here.

    Needs both the ``helm`` CLI and the chart itself; the ``aiperf-operator``
    chart is supplied by the operator port, not by ``aiperf.kubernetes``.
    """
    return shutil.which("helm") is not None and CHART_PATH.exists()


pytestmark = pytest.mark.skipif(
    not _helm_available(), reason="helm CLI not installed; chart-render tests skipped"
)


def _helm_template(
    *extra: str, namespace: str = "test-ns", release: str = "aiperf-operator"
) -> list[dict]:
    """Run ``helm template`` and return parsed YAML docs.

    Default release name is ``aiperf-operator`` (matches ``Chart.Name``) so
    the chart's ``fullname`` helper resolves to a clean ``aiperf-operator``
    instead of ``<release>-aiperf-operator``. Override per-test only when
    asserting fullname behavior under release-name divergence.

    ``--api-versions monitoring.coreos.com/v1`` is set so the ServiceMonitor
    template renders even on a stripped-down test environment without the
    Prometheus Operator CRDs.
    """
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
        cmd, capture_output=True, text=True, check=False, env=os.environ.copy()
    )
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit={result.returncode}): {result.stderr}"
        )
    return [
        doc
        for doc in yaml.safe_load_all(result.stdout)
        if doc and isinstance(doc, dict)
    ]


def _find(docs: list[dict], kind: str, name: str) -> dict:
    """Return the first doc with matching kind+name; raise on miss."""
    for doc in docs:
        if doc.get("kind") == kind and doc.get("metadata", {}).get("name") == name:
            return doc
    available = [(d.get("kind"), d.get("metadata", {}).get("name")) for d in docs]
    raise AssertionError(f"{kind}/{name} not found in chart output. Got: {available}")


# ============================================================
# Benchmark namespace ownership and RBAC
# ============================================================


class TestBenchmarkNamespaceOwnership:
    """Namespace creation can be external while benchmark RBAC stays chart-owned."""

    def test_existing_custom_namespace_omits_namespace_but_keeps_rbac(self) -> None:
        docs = _helm_template(
            "--set-string",
            "benchmarkNamespace.name=team-benchmarks",
            "--set",
            "benchmarkNamespace.create=false",
        )

        namespaces = {
            doc.get("metadata", {}).get("name")
            for doc in docs
            if doc.get("kind") == "Namespace"
        }
        assert "team-benchmarks" not in namespaces

        role = _find(docs, "Role", "aiperf-operator-benchmark")
        role_binding = _find(docs, "RoleBinding", "aiperf-operator-benchmark")
        assert role["metadata"]["namespace"] == "team-benchmarks"
        assert role_binding["metadata"]["namespace"] == "team-benchmarks"


# ============================================================
# AIPERF_OPERATOR_BASE_URL auto-templated from Release identity
# ============================================================


class TestOperatorBaseUrlEnvInjection:
    """``AIPERF_OPERATOR_BASE_URL`` is auto-templated from chart values."""

    def test_default_install_uses_release_fullname_and_results_port(self) -> None:
        docs = _helm_template(namespace="aiperf-system")
        deploy = _find(docs, "Deployment", "aiperf-operator")
        operator_container = next(
            c
            for c in deploy["spec"]["template"]["spec"]["containers"]
            if c["name"] == "operator"
        )
        env_by_name = {e["name"]: e["value"] for e in operator_container.get("env", [])}
        assert (
            env_by_name.get("AIPERF_OPERATOR_BASE_URL")
            == "http://aiperf-operator.aiperf-system:8081"
        )

    def test_custom_namespace_propagates(self) -> None:
        docs = _helm_template(namespace="perf-team-aiperf")
        deploy = _find(docs, "Deployment", "aiperf-operator")
        operator_container = next(
            c
            for c in deploy["spec"]["template"]["spec"]["containers"]
            if c["name"] == "operator"
        )
        env_by_name = {e["name"]: e["value"] for e in operator_container.get("env", [])}
        assert (
            env_by_name.get("AIPERF_OPERATOR_BASE_URL")
            == "http://aiperf-operator.perf-team-aiperf:8081"
        )

    def test_fullname_override_propagates(self) -> None:
        docs = _helm_template("--set", "fullnameOverride=my-aiperf")
        deploy = _find(docs, "Deployment", "my-aiperf")
        operator_container = next(
            c
            for c in deploy["spec"]["template"]["spec"]["containers"]
            if c["name"] == "operator"
        )
        env_by_name = {e["name"]: e["value"] for e in operator_container.get("env", [])}
        assert (
            env_by_name.get("AIPERF_OPERATOR_BASE_URL")
            == "http://my-aiperf.test-ns:8081"
        )

    def test_results_server_port_override_propagates(self) -> None:
        docs = _helm_template("--set", "resultsServer.port=9001")
        deploy = _find(docs, "Deployment", "aiperf-operator")
        operator_container = next(
            c
            for c in deploy["spec"]["template"]["spec"]["containers"]
            if c["name"] == "operator"
        )
        env_by_name = {e["name"]: e["value"] for e in operator_container.get("env", [])}
        assert (
            env_by_name.get("AIPERF_OPERATOR_BASE_URL")
            == "http://aiperf-operator.test-ns:9001"
        )

    def test_no_obsolete_results_base_url_env(self) -> None:
        """Pre-collapse, the chart emitted a separate AIPERF_OPERATOR_RESULTS_BASE_URL.
        The collapse commit (6b344438a) deleted it; lock in absence."""
        docs = _helm_template()
        deploy = _find(docs, "Deployment", "aiperf-operator")
        operator_container = next(
            c
            for c in deploy["spec"]["template"]["spec"]["containers"]
            if c["name"] == "operator"
        )
        env_names = {e["name"] for e in operator_container.get("env", [])}
        assert "AIPERF_OPERATOR_RESULTS_BASE_URL" not in env_names


class TestNoRuntimeUiOverride:
    """The operator UI is served only from the packaged static bundle."""

    def test_deployment_has_no_ui_override_runtime_plumbing(self) -> None:
        docs = _helm_template()
        deploy = _find(docs, "Deployment", "aiperf-operator")
        pod_spec = deploy["spec"]["template"]["spec"]
        assert "initContainers" not in pod_spec
        assert "ui-override" not in {v["name"] for v in pod_spec.get("volumes", [])}

        results_server = next(
            c for c in pod_spec["containers"] if c["name"] == "results-server"
        )
        env_names = {e["name"] for e in results_server.get("env", [])}
        volume_mount_names = {m["name"] for m in results_server.get("volumeMounts", [])}
        assert "AIPERF_DEV_UI_OVERRIDE_DIR" not in env_names
        assert "ui-override" not in volume_mount_names


# ============================================================
# Service / ServiceMonitor / NetworkPolicy metrics-port wiring
# ============================================================


class TestMetricsPortExposure:
    """Round-3 chart audit: Prometheus scraping was silently broken because
    the Service didn't expose the metrics port and the ServiceMonitor scraped
    ``port: health`` (8080) where there's no FastAPI.
    """

    def test_service_exposes_metrics_port_by_default(self) -> None:
        docs = _helm_template()
        svc = _find(docs, "Service", "aiperf-operator")
        port_names = {p["name"]: p for p in svc["spec"]["ports"]}
        assert "metrics" in port_names, (
            f"Service missing 'metrics' port; got: {list(port_names)}"
        )
        assert port_names["metrics"]["port"] == 9090
        assert port_names["metrics"]["targetPort"] == "metrics"

    def test_service_omits_metrics_port_when_disabled(self) -> None:
        """Setting ``operator.metrics.port=0`` disables metrics; Service should
        not expose a port that isn't being listened on."""
        docs = _helm_template("--set", "operator.metrics.port=0")
        svc = _find(docs, "Service", "aiperf-operator")
        port_names = {p["name"] for p in svc["spec"]["ports"]}
        assert "metrics" not in port_names, (
            f"Service exposes metrics port even when disabled: {port_names}"
        )

    def test_deployment_omits_metrics_container_port_when_disabled(self) -> None:
        docs = _helm_template("--set", "operator.metrics.port=0")
        deploy = _find(docs, "Deployment", "aiperf-operator")
        operator_container = next(
            c
            for c in deploy["spec"]["template"]["spec"]["containers"]
            if c["name"] == "operator"
        )
        port_names = {p["name"] for p in operator_container.get("ports", [])}
        assert "metrics" not in port_names

    def test_service_metrics_port_follows_override(self) -> None:
        docs = _helm_template("--set", "operator.metrics.port=9100")
        svc = _find(docs, "Service", "aiperf-operator")
        port_names = {p["name"]: p["port"] for p in svc["spec"]["ports"]}
        assert port_names["metrics"] == 9100

    def test_servicemonitor_scrapes_metrics_port_not_health(self) -> None:
        """Pre-fix ServiceMonitor scraped ``port: health`` (kopf's healthz only,
        no /metrics) — locking in that it now scrapes ``port: metrics``."""
        docs = _helm_template("--set", "serviceMonitor.enabled=true")
        sm = _find(docs, "ServiceMonitor", "aiperf-operator")
        endpoints = sm["spec"]["endpoints"]
        assert len(endpoints) == 1
        assert endpoints[0]["port"] == "metrics", (
            f"ServiceMonitor must scrape 'metrics', got '{endpoints[0]['port']}'"
        )
        assert endpoints[0]["path"] == "/metrics"

    def test_servicemonitor_omitted_when_metrics_disabled(self) -> None:
        docs = _helm_template(
            "--set",
            "serviceMonitor.enabled=true",
            "--set",
            "operator.metrics.port=0",
        )
        service_monitors = [doc for doc in docs if doc.get("kind") == "ServiceMonitor"]
        assert service_monitors == []

    def test_networkpolicy_allows_metrics_ingress(self) -> None:
        docs = _helm_template("--set", "networkPolicy.enabled=true")
        netpol = _find(docs, "NetworkPolicy", "aiperf-operator")
        all_ingress_ports = {
            p["port"]
            for rule in netpol["spec"]["ingress"]
            for p in rule.get("ports", [])
        }
        assert 9090 in all_ingress_ports, (
            f"NetworkPolicy ingress missing metrics port 9090; got: {all_ingress_ports}"
        )

    def test_networkpolicy_omits_metrics_when_disabled(self) -> None:
        docs = _helm_template(
            "--set",
            "networkPolicy.enabled=true",
            "--set",
            "operator.metrics.port=0",
        )
        netpol = _find(docs, "NetworkPolicy", "aiperf-operator")
        all_ingress_ports = {
            p["port"]
            for rule in netpol["spec"]["ingress"]
            for p in rule.get("ports", [])
        }
        assert 9090 not in all_ingress_ports, (
            "NetworkPolicy still allows 9090 ingress when metrics disabled"
        )


# ============================================================
# benchmarkRbacNamespaces propagates into NetworkPolicy
# ============================================================


class TestBenchmarkRbacNamespaceIngress:
    """Round-4 RBAC audit: ``benchmarkRbacNamespaces`` (multi-tenant benchmark
    RBAC) was missing from NetworkPolicy ingress allow-list, so sweep
    controllers in those namespaces silently couldn't reach the operator's
    results-server.
    """

    def test_extra_benchmark_namespaces_in_ingress(self) -> None:
        docs = _helm_template(
            "--set",
            "networkPolicy.enabled=true",
            "--set",
            "benchmarkRbacNamespaces={team-a,team-b}",
        )
        netpol = _find(docs, "NetworkPolicy", "aiperf-operator")
        all_source_namespaces: set[str] = set()
        for rule in netpol["spec"]["ingress"]:
            for src in rule.get("from", []):
                ns = (
                    src.get("namespaceSelector", {})
                    .get("matchLabels", {})
                    .get("kubernetes.io/metadata.name")
                )
                if ns:
                    all_source_namespaces.add(ns)
        assert "team-a" in all_source_namespaces
        assert "team-b" in all_source_namespaces


# ============================================================
# helm-test pods cover BOTH FastAPI surfaces
# ============================================================


class TestHelmTestCoverage:
    """Round-3: helm-test pod only probed kopf's :8080/healthz, missing the
    results-server FastAPI surface where every /api/v1/* router lives.
    Round-4: also assert AIPerfSweep CRD installation (was AIPerfJob-only).
    """

    def test_health_test_probes_results_server_fastapi(self) -> None:
        docs = _helm_template()
        test_pod = _find(docs, "Pod", "aiperf-operator-test-health")
        args = test_pod["spec"]["containers"][0]["args"][0]
        assert "/api/v1/jobs" in args, (
            "helm-test must probe the FastAPI surface, not just kopf health"
        )
        assert ":8080/healthz" in args, "kopf health probe still required"

    def test_health_test_selector_targets_operator_component(self) -> None:
        docs = _helm_template()
        test_pod = _find(docs, "Pod", "aiperf-operator-test-health")
        args = test_pod["spec"]["containers"][0]["args"][0]
        assert "app.kubernetes.io/component=operator" in args

    def test_crd_test_covers_both_aiperfjob_and_aiperfsweep(self) -> None:
        docs = _helm_template()
        test_pod = _find(docs, "Pod", "aiperf-operator-test-crd")
        args = test_pod["spec"]["containers"][0]["args"][0]
        assert "aiperfjobs.aiperf.nvidia.com" in args
        assert "aiperfsweeps.aiperf.nvidia.com" in args


class TestClusterRolePrivileges:
    """ClusterRole grants only verbs the operator uses cluster-wide."""

    def test_serviceaccount_verbs_are_limited_to_create_and_reads(self) -> None:
        docs = _helm_template()
        cluster_role = _find(docs, "ClusterRole", "aiperf-operator")
        serviceaccount_rules = [
            rule
            for rule in cluster_role["rules"]
            if rule.get("apiGroups") == [""]
            and rule.get("resources") == ["serviceaccounts"]
        ]
        assert len(serviceaccount_rules) == 1
        assert set(serviceaccount_rules[0]["verbs"]) == {
            "create",
            "get",
            "list",
            "watch",
        }


# ============================================================
# AIPerfJob / AIPerfSweep CRD chart-default parity
# ============================================================


def _crd_spec_properties(docs: list[dict], crd_name: str) -> dict:
    """Return the openAPIV3Schema spec.properties map for a rendered CRD."""
    crd = _find(docs, "CustomResourceDefinition", crd_name)
    schema = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
    return schema["properties"]["spec"]["properties"]


class TestCrdChartDefaultParity:
    """Both CRDs must agree on the .Values-driven spec defaults.

    AIPerfJob and AIPerfSweep share the workload spec shape; the generator
    once applied the defaults.image / defaults.imagePullPolicy substitutions
    only to the AIPerfJob CRD, so a chart deployed with a custom image gave
    AIPerfSweep CRs the hardcoded nvcr.io:latest default silently.
    """

    BOTH_CRDS = ("aiperfjobs.aiperf.nvidia.com", "aiperfsweeps.aiperf.nvidia.com")

    def test_custom_image_repository_and_tag_propagate_to_both_crds(self) -> None:
        docs = _helm_template(
            "--set",
            "image.repository=example.com/custom/aiperf",
            "--set",
            "image.tag=v9.9.9",
        )
        for crd_name in self.BOTH_CRDS:
            props = _crd_spec_properties(docs, crd_name)
            assert props["image"]["default"] == "example.com/custom/aiperf:v9.9.9", (
                f"{crd_name} spec.image default must follow the chart image"
            )

    def test_defaults_image_override_propagates_to_both_crds(self) -> None:
        docs = _helm_template("--set", "defaults.image=ghcr.io/x/aiperf:dev")
        for crd_name in self.BOTH_CRDS:
            props = _crd_spec_properties(docs, crd_name)
            assert props["image"]["default"] == "ghcr.io/x/aiperf:dev"

    def test_defaults_image_pull_policy_propagates_to_both_crds(self) -> None:
        docs = _helm_template("--set", "defaults.imagePullPolicy=Always")
        for crd_name in self.BOTH_CRDS:
            props = _crd_spec_properties(docs, crd_name)
            assert props["imagePullPolicy"].get("default") == "Always"

    def test_unset_image_pull_policy_omits_default_on_both_crds(self) -> None:
        docs = _helm_template("--set", "defaults.imagePullPolicy=null")
        for crd_name in self.BOTH_CRDS:
            props = _crd_spec_properties(docs, crd_name)
            assert "default" not in props["imagePullPolicy"], (
                f"{crd_name} must defer imagePullPolicy to K8s when the chart "
                f"value is unset"
            )


# ============================================================
# serverMetricsDiscoveryNamespaces RoleBinding subjects
# ============================================================


class TestMetricsDiscoveryRoleBindingSubjects:
    """Cross-namespace discovery RoleBindings honor the entry's SA list.

    Plain string entries keep the historical behavior (bind the benchmark
    namespaces' `default` ServiceAccount); the object form binds the listed
    ServiceAccounts instead so pods running under a custom
    podTemplate.serviceAccountName are not silently denied 'pods: list'.
    """

    def test_plain_string_entry_binds_default_serviceaccount(self) -> None:
        docs = _helm_template(
            "--set", "serverMetricsDiscoveryNamespaces={dynamo-server}"
        )
        binding = _find(docs, "RoleBinding", "aiperf-operator-metrics-discovery")
        assert binding["metadata"]["namespace"] == "dynamo-server"
        assert {(s["name"], s["namespace"]) for s in binding["subjects"]} == {
            ("default", "aiperf-benchmarks")
        }

    def test_object_entry_binds_listed_serviceaccounts(self) -> None:
        docs = _helm_template(
            "--set",
            "serverMetricsDiscoveryNamespaces[0].namespace=dynamo-server",
            "--set",
            "serverMetricsDiscoveryNamespaces[0].serviceAccounts[0]=aiperf-bench",
            "--set",
            "serverMetricsDiscoveryNamespaces[0].serviceAccounts[1]=other-sa",
        )
        binding = _find(docs, "RoleBinding", "aiperf-operator-metrics-discovery")
        assert binding["metadata"]["namespace"] == "dynamo-server"
        assert {(s["name"], s["namespace"]) for s in binding["subjects"]} == {
            ("aiperf-bench", "aiperf-benchmarks"),
            ("other-sa", "aiperf-benchmarks"),
        }

    def test_object_entry_without_serviceaccounts_falls_back_to_default(self) -> None:
        docs = _helm_template(
            "--set", "serverMetricsDiscoveryNamespaces[0].namespace=dynamo-server"
        )
        binding = _find(docs, "RoleBinding", "aiperf-operator-metrics-discovery")
        assert {(s["name"], s["namespace"]) for s in binding["subjects"]} == {
            ("default", "aiperf-benchmarks")
        }
