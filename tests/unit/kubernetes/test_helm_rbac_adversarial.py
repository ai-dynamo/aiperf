# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial Helm RBAC and network-boundary tests.

Focuses on least-privilege chart invariants that protect operator installs:
- operator RBAC for pod-restart events remains read-only on pods and pod logs.
- benchmark pod Roles can patch JobSet/AIPerfJob status without cluster-admin verbs.
- service account subjects bind to the rendered release namespace and configured name.
- helm-test CRD checks can only read the two AIPerf CRDs they verify.
- results-server is the only public FastAPI service port; dashboard stays pod-local.

Out of scope: live Kubernetes authorization checks and applied NetworkPolicy behavior;
``tests/kubernetes/test_helm.py`` owns cluster-backed install coverage.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from pytest import param

from aiperf.kubernetes.constants import DEFAULT_OPERATOR_NAMESPACE

CHART_PATH = Path(__file__).parents[3] / "deploy" / "helm" / "aiperf-operator"

K8sDoc = dict[str, object]
K8sRule = dict[str, object]


def _helm_available() -> bool:
    """Report whether the chart can be rendered here.

    Needs both the ``helm`` CLI and the chart itself; the ``aiperf-operator``
    chart is supplied by the operator port, not by ``aiperf.kubernetes``.
    """
    return shutil.which("helm") is not None and CHART_PATH.exists()


def _run_helm_template(
    *extra: str,
    namespace: str = DEFAULT_OPERATOR_NAMESPACE,
    release: str = "aiperf-operator",
) -> subprocess.CompletedProcess[str]:
    """Attempt a chart render and return the raw result, failure included."""
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
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )


def _helm_template(
    *extra: str,
    namespace: str = DEFAULT_OPERATOR_NAMESPACE,
    release: str = "aiperf-operator",
) -> list[K8sDoc]:
    """Render the chart with optional CRD APIs present and return YAML documents."""
    result = _run_helm_template(*extra, namespace=namespace, release=release)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed for release {release!r} in namespace {namespace!r}: "
            f"{result.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(result.stdout) if isinstance(doc, dict)]


def _as_mapping(value: object, context: str) -> dict[str, object]:
    assert isinstance(value, dict), f"{context} must be a mapping, got {type(value)}"
    return value


def _as_list(value: object, context: str) -> list[object]:
    assert isinstance(value, list), f"{context} must be a list, got {type(value)}"
    return value


def _find(docs: list[K8sDoc], kind: str, name: str) -> K8sDoc:
    """Return the rendered Kubernetes object identified by kind and name."""
    for doc in docs:
        metadata = _as_mapping(doc.get("metadata", {}), f"{kind} metadata")
        if doc.get("kind") == kind and metadata.get("name") == name:
            return doc
    available = [
        (doc.get("kind"), _as_mapping(doc.get("metadata", {}), "metadata").get("name"))
        for doc in docs
    ]
    raise AssertionError(f"{kind}/{name} not found in chart output. Got: {available}")


def _rules(resource: K8sDoc) -> list[K8sRule]:
    rules = _as_list(resource.get("rules", []), f"{resource.get('kind')} rules")
    return [rule for rule in rules if isinstance(rule, dict)]


def _resource_rules(
    resource: K8sDoc,
    *,
    api_group: str,
    resource_name: str,
) -> list[K8sRule]:
    matched: list[K8sRule] = []
    for rule in _rules(resource):
        api_groups = _as_list(rule.get("apiGroups", []), "rule apiGroups")
        resources = _as_list(rule.get("resources", []), "rule resources")
        if api_group in api_groups and resource_name in resources:
            matched.append(rule)
    return matched


def _verbs(rule: K8sRule) -> set[str]:
    return {str(verb) for verb in _as_list(rule.get("verbs", []), "rule verbs")}


def _resources(rule: K8sRule) -> set[str]:
    return {
        str(resource)
        for resource in _as_list(rule.get("resources", []), "rule resources")
    }


def _api_groups(rule: K8sRule) -> set[str]:
    return {
        str(group) for group in _as_list(rule.get("apiGroups", []), "rule apiGroups")
    }


def _metadata_name(resource: K8sDoc) -> str:
    metadata = _as_mapping(resource["metadata"], "metadata")
    name = metadata["name"]
    assert isinstance(name, str)
    return name


def _metadata_namespace(resource: K8sDoc) -> str:
    metadata = _as_mapping(resource["metadata"], "metadata")
    namespace = metadata["namespace"]
    assert isinstance(namespace, str)
    return namespace


pytestmark = pytest.mark.skipif(not _helm_available(), reason="helm CLI not installed")


# ============================================================
# Operator ClusterRole least privilege
# ============================================================


class TestOperatorClusterRoleLeastPrivilege:
    """Cluster-scoped operator permissions stay specific to reconciler behavior."""

    def test_clusterrole_pods_restart_event_path_is_read_only(self) -> None:
        docs = _helm_template()
        cluster_role = _find(docs, "ClusterRole", "aiperf-operator")

        pod_rules = _resource_rules(cluster_role, api_group="", resource_name="pods")
        assert len(pod_rules) == 1
        assert _resources(pod_rules[0]) == {"pods", "pods/log"}
        assert _verbs(pod_rules[0]) == {"get", "list", "watch"}
        assert {"create", "delete", "patch", "update"}.isdisjoint(_verbs(pod_rules[0]))

    @pytest.mark.parametrize(
        "kind,resource_name",
        [
            ("ClusterRole", "aiperf-operator"),
            ("Role", "aiperf-operator-benchmark"),
            param("ClusterRole", "aiperf-operator-tests", id="helm-test-clusterrole"),
            param("Role", "aiperf-operator-tests", id="helm-test-role"),
        ],
    )  # fmt: skip
    def test_rbac_rules_do_not_grant_wildcard_cluster_admin_shapes(
        self, kind: str, resource_name: str
    ) -> None:
        docs = _helm_template()
        resource = _find(docs, kind, resource_name)

        for rule in _rules(resource):
            assert "*" not in _api_groups(rule), (
                f"{kind}/{resource_name} has wildcard API group"
            )
            assert "*" not in _resources(rule), (
                f"{kind}/{resource_name} has wildcard resource"
            )
            assert "*" not in _verbs(rule), f"{kind}/{resource_name} has wildcard verb"

    def test_clusterrole_jobsets_has_manage_verbs_but_status_is_read_only(self) -> None:
        docs = _helm_template()
        cluster_role = _find(docs, "ClusterRole", "aiperf-operator")

        jobset_rules = _resource_rules(
            cluster_role, api_group="jobset.x-k8s.io", resource_name="jobsets"
        )
        assert len(jobset_rules) == 1
        assert _verbs(jobset_rules[0]) == {
            "create",
            "delete",
            "get",
            "list",
            "patch",
            "update",
            "watch",
        }

        status_rules = _resource_rules(
            cluster_role, api_group="jobset.x-k8s.io", resource_name="jobsets/status"
        )
        assert len(status_rules) == 1
        assert _verbs(status_rules[0]) == {"get", "list", "watch"}

    def test_clusterrole_grants_kueue_localqueues_read_for_preflight(self) -> None:
        """Preflight reads Kueue LocalQueues; without this grant the check 403s.

        Also asserts the unused ``workloads`` grant is gone — nothing in the
        operator reads Kueue Workloads.
        """
        docs = _helm_template()
        cluster_role = _find(docs, "ClusterRole", "aiperf-operator")

        localqueue_rules = _resource_rules(
            cluster_role, api_group="kueue.x-k8s.io", resource_name="localqueues"
        )
        assert len(localqueue_rules) == 1
        assert {"get", "list"}.issubset(_verbs(localqueue_rules[0]))

        for rule in _rules(cluster_role):
            if "kueue.x-k8s.io" in _api_groups(rule):
                assert "workloads" not in _resources(rule), (
                    "operator ClusterRole still grants unused kueue workloads"
                )


# ============================================================
# Benchmark namespace Role contracts
# ============================================================


class TestBenchmarkRoleStatusPatchContract:
    """Benchmark pods get only namespace-local verbs needed for completion signaling."""

    def test_benchmark_role_pods_are_read_only_for_peer_discovery(self) -> None:
        docs = _helm_template()
        role = _find(docs, "Role", "aiperf-operator-benchmark")

        pod_rules = _resource_rules(role, api_group="", resource_name="pods")
        assert len(pod_rules) == 1
        assert _resources(pod_rules[0]) == {"pods"}
        assert _verbs(pod_rules[0]) == {"get", "list", "watch"}

    @pytest.mark.parametrize(
        "api_group,resources",
        [
            ("jobset.x-k8s.io", {"jobsets"}),
            param(
                "aiperf.nvidia.com",
                {"aiperfjobs", "aiperfjobs/status"},
                id="aiperfjob-status-patch",
            ),
        ],
    )  # fmt: skip
    def test_benchmark_role_status_patch_targets_have_no_lifecycle_admin_verbs(
        self, api_group: str, resources: set[str]
    ) -> None:
        """``patch`` is the only write verb; ``update`` is a whole-object PUT.

        Nothing in a benchmark pod replaces a whole JobSet or AIPerfJob -- both
        writers issue JSON-patch annotation/status updates -- so ``update``
        would let a compromised pod rewrite ``spec``, not just ``status``.
        """
        docs = _helm_template()
        role = _find(docs, "Role", "aiperf-operator-benchmark")
        matching_rules = [
            rule
            for rule in _rules(role)
            if _api_groups(rule) == {api_group} and _resources(rule) == resources
        ]
        assert len(matching_rules) == 1
        assert _verbs(matching_rules[0]) == {"get", "list", "watch", "patch"}
        assert {"create", "delete", "update"}.isdisjoint(_verbs(matching_rules[0]))

    def test_extra_benchmark_namespace_rolebinding_stays_in_its_namespace(self) -> None:
        docs = _helm_template(
            "--set",
            "benchmarkRbacNamespaces={vision-benchmarks,audio-benchmarks}",
        )
        rolebindings = [
            doc
            for doc in docs
            if doc.get("kind") == "RoleBinding"
            and _metadata_name(doc) == "aiperf-operator-benchmark"
        ]
        namespaces = {_metadata_namespace(binding) for binding in rolebindings}
        assert namespaces == {
            "aiperf-benchmarks",
            "vision-benchmarks",
            "audio-benchmarks",
        }

        for binding in rolebindings:
            subject = _as_mapping(
                _as_list(binding["subjects"], "RoleBinding subjects")[0],
                "RoleBinding subject",
            )
            assert subject == {
                "kind": "ServiceAccount",
                "name": "default",
                "namespace": _metadata_namespace(binding),
            }


# ============================================================
# ServiceAccount and helm-test RBAC bindings
# ============================================================


class TestServiceAccountBindingContract:
    """Rendered subjects must bind the service accounts that Pods actually use."""

    def test_custom_serviceaccount_name_propagates_to_deployment_and_binding(
        self,
    ) -> None:
        docs = _helm_template(
            "--set",
            "serviceAccount.name=aiperf-operator-runner",
            namespace="observability-aiperf",
        )
        service_account = _find(docs, "ServiceAccount", "aiperf-operator-runner")
        assert _metadata_namespace(service_account) == "observability-aiperf"

        deployment = _find(docs, "Deployment", "aiperf-operator")
        pod_spec = _as_mapping(
            _as_mapping(deployment["spec"], "Deployment spec")
            .get("template", {})
            .get("spec", {}),
            "Deployment pod spec",
        )
        assert pod_spec["serviceAccountName"] == "aiperf-operator-runner"

        binding = _find(docs, "ClusterRoleBinding", "aiperf-operator")
        subject = _as_mapping(
            _as_list(binding["subjects"], "ClusterRoleBinding subjects")[0],
            "ClusterRoleBinding subject",
        )
        assert subject == {
            "kind": "ServiceAccount",
            "name": "aiperf-operator-runner",
            "namespace": "observability-aiperf",
        }

    def test_serviceaccount_create_false_without_a_name_fails_the_render(self) -> None:
        """An unnamed pre-provisioned SA used to silently become ``default``.

        The namespace ``default`` ServiceAccount holds none of the operator's
        ClusterRole, so the install succeeded and then 403'd on every reconcile.
        The chart must reject it at render time instead.
        """
        result = _run_helm_template("--set", "serviceAccount.create=false")

        assert result.returncode != 0, (
            "serviceAccount.create=false with an empty name must not render"
        )
        assert "serviceAccount.name is required" in result.stderr, result.stderr
        assert "serviceAccountName: default" not in result.stdout

    @pytest.mark.parametrize(
        ("overrides", "expected_name"),
        [
            param((), "aiperf-operator", id="create-true-default-name"),
            param(
                ("--set", "serviceAccount.create=false",
                 "--set", "serviceAccount.name=aiperf-operator-sa"),
                "aiperf-operator-sa",
                id="create-false-explicit-name",
            ),
        ],
    )  # fmt: skip
    def test_legitimate_serviceaccount_configurations_still_render(
        self, overrides: tuple[str, ...], expected_name: str
    ) -> None:
        docs = _helm_template(*overrides)

        deployment = _find(docs, "Deployment", "aiperf-operator")
        pod_spec = _as_mapping(
            _as_mapping(deployment["spec"], "Deployment spec")
            .get("template", {})
            .get("spec", {}),
            "Deployment pod spec",
        )
        assert pod_spec["serviceAccountName"] == expected_name

    def test_helm_test_crd_clusterrole_is_resource_name_scoped_to_aiperf_crds(
        self,
    ) -> None:
        docs = _helm_template()
        test_role = _find(docs, "ClusterRole", "aiperf-operator-tests")
        rules = _rules(test_role)
        assert len(rules) == 1
        assert _api_groups(rules[0]) == {"apiextensions.k8s.io"}
        assert _resources(rules[0]) == {"customresourcedefinitions"}
        assert _verbs(rules[0]) == {"get"}
        assert set(_as_list(rules[0].get("resourceNames", []), "resourceNames")) == {
            "aiperfjobs.aiperf.nvidia.com",
            "aiperfsweeps.aiperf.nvidia.com",
        }

    @pytest.mark.parametrize(
        "pod_name",
        [
            "aiperf-operator-test-crd",
            "aiperf-operator-test-health",
        ],
    )  # fmt: skip
    def test_helm_test_pods_use_dedicated_test_serviceaccount(
        self, pod_name: str
    ) -> None:
        docs = _helm_template()
        pod = _find(docs, "Pod", pod_name)
        pod_spec = _as_mapping(pod["spec"], "test pod spec")
        assert pod_spec["serviceAccountName"] == "aiperf-operator-tests"


# ============================================================
# "This chart creates no cluster-scoped RBAC" opt-out
# ============================================================

#: The documented opt-out for a cluster whose policy forbids chart-managed
#: cluster-scoped RBAC. ``rbac.create=false`` alone is not enough: the
#: ``helm test`` hook scaffolding is gated separately on ``tests.enabled``
#: because it serves a different purpose and is torn down with the release.
_NO_CLUSTER_RBAC_OPT_OUT = (
    "--set", "rbac.create=false",
    "--set", "serviceAccount.create=false",
    "--set", "serviceAccount.name=preprovisioned-operator-sa",
    "--set", "tests.enabled=false",
)  # fmt: skip

_CLUSTER_SCOPED_RBAC_KINDS = {"ClusterRole", "ClusterRoleBinding"}


class TestClusterScopedRbacOptOut:
    """``rbac.create=false`` plus ``tests.enabled=false`` must emit zero
    ClusterRole/ClusterRoleBinding.

    The ``helm test`` hook RBAC used to render unconditionally, so a cluster
    whose policy is "this chart creates no cluster-scoped RBAC" could not reach
    that state through any documented flag combination.
    """

    def test_documented_opt_out_emits_no_cluster_scoped_rbac(self) -> None:
        docs = _helm_template(*_NO_CLUSTER_RBAC_OPT_OUT)
        leftover = [
            f"{doc['kind']}/{_metadata_name(doc)}"
            for doc in docs
            if doc.get("kind") in _CLUSTER_SCOPED_RBAC_KINDS
        ]
        assert leftover == [], f"chart still emits cluster-scoped RBAC: {leftover}"

    @pytest.mark.parametrize(
        "overrides,expected_names",
        [
            param(
                ("--set", "rbac.create=false",
                 "--set", "serviceAccount.create=false",
                 "--set", "serviceAccount.name=preprovisioned-operator-sa"),
                {"aiperf-operator-tests"},
                id="rbac-create-false-alone-leaves-the-test-hook-rbac",
            ),
            param(
                _NO_CLUSTER_RBAC_OPT_OUT,
                set(),
                id="tests-disabled-too-leaves-nothing",
            ),
        ],
    )  # fmt: skip
    def test_rbac_create_false_does_not_by_itself_disable_test_hook_rbac(
        self, overrides: tuple[str, ...], expected_names: set[str]
    ) -> None:
        """``tests.enabled`` is deliberately independent of ``rbac.create``.

        Gating the test scaffolding on ``rbac.create`` would strip the test
        ServiceAccount while still rendering the hook Pods that reference it, so
        ``helm test`` would fail on pod creation rather than report no tests.
        """
        docs = _helm_template(*overrides)
        names = {
            _metadata_name(doc)
            for doc in docs
            if doc.get("kind") in _CLUSTER_SCOPED_RBAC_KINDS
        }
        assert names == expected_names

    @pytest.mark.parametrize(
        "kind,name",
        [
            param("ServiceAccount", "aiperf-operator-tests", id="test-sa"),
            param("Role", "aiperf-operator-tests", id="test-role"),
            param("RoleBinding", "aiperf-operator-tests", id="test-rolebinding"),
            param("ClusterRole", "aiperf-operator-tests", id="test-clusterrole"),
            param("ClusterRoleBinding", "aiperf-operator-tests", id="test-crb"),
            param("Pod", "aiperf-operator-test-crd", id="test-crd-hook-pod"),
            param("Pod", "aiperf-operator-test-health", id="test-health-hook-pod"),
        ],
    )  # fmt: skip
    def test_tests_disabled_removes_every_test_object(
        self, kind: str, name: str
    ) -> None:
        """The hook Pods and their RBAC are gated together.

        Disabling only one side would leave Pods pointing at a nonexistent
        ServiceAccount, which fails ``helm test`` confusingly instead of
        skipping it.
        """
        docs = _helm_template("--set", "tests.enabled=false")
        matches = [
            doc
            for doc in docs
            if doc.get("kind") == kind and _metadata_name(doc) == name
        ]
        assert matches == [], f"{kind}/{name} still rendered with tests.enabled=false"

    @pytest.mark.parametrize(
        "kind,name",
        [
            param("ServiceAccount", "aiperf-operator-tests", id="test-sa"),
            param("ClusterRole", "aiperf-operator-tests", id="test-clusterrole"),
            param("Pod", "aiperf-operator-test-crd", id="test-crd-hook-pod"),
            param("Pod", "aiperf-operator-test-health", id="test-health-hook-pod"),
        ],
    )  # fmt: skip
    def test_tests_enabled_by_default_so_helm_test_keeps_working(
        self, kind: str, name: str
    ) -> None:
        """The opt-out is opt-in: a default install still ships `helm test`."""
        _find(_helm_template(), kind, name)


# ============================================================
# Results-server service and NetworkPolicy boundary
# ============================================================


class TestResultsServerNetworkBoundary:
    """The Service exposes health/results/metrics only; dashboard remains pod-local."""

    @pytest.mark.parametrize(
        "extra_args,expected_ports",
        [
            ((), {"health", "results", "metrics"}),
            param(("--set", "operator.metrics.port=0"), {"health", "results"}, id="metrics-disabled"),
            param(("--set", "dashboard.enabled=true"), {"health", "results", "metrics"}, id="dashboard-pod-local"),
        ],
    )  # fmt: skip
    def test_service_port_names_are_limited_to_operator_public_surfaces(
        self, extra_args: tuple[str, ...], expected_ports: set[str]
    ) -> None:
        docs = _helm_template(*extra_args)
        service = _find(docs, "Service", "aiperf-operator")
        ports = _as_list(_as_mapping(service["spec"], "Service spec")["ports"], "ports")
        port_names = {_as_mapping(port, "Service port")["name"] for port in ports}
        assert port_names == expected_ports
        assert "dashboard" not in port_names

    def test_networkpolicy_ingress_does_not_expose_dashboard_port(self) -> None:
        docs = _helm_template(
            "--set",
            "networkPolicy.enabled=true",
            "--set",
            "dashboard.enabled=true",
            "--set",
            "dashboard.port=8099",
            "--set",
            "networkPolicy.allowedIngressCIDRs={10.42.0.0/16}",
        )
        netpol = _find(docs, "NetworkPolicy", "aiperf-operator")
        spec = _as_mapping(netpol["spec"], "NetworkPolicy spec")
        ingress = _as_list(spec["ingress"], "NetworkPolicy ingress")
        all_ports = {
            _as_mapping(port, "NetworkPolicy port")["port"]
            for rule in ingress
            for port in _as_list(
                _as_mapping(rule, "ingress rule").get("ports", []), "ports"
            )
        }
        assert all_ports == {8080, 8081, 9090}
        assert 8099 not in all_ports

    def test_networkpolicy_allows_results_from_release_and_benchmark_namespaces(
        self,
    ) -> None:
        docs = _helm_template(
            "--set",
            "networkPolicy.enabled=true",
            "--set",
            "benchmarkNamespace.name=mlperf-benchmarks",
            "--set",
            "benchmarkRbacNamespaces={vision-benchmarks}",
            namespace="operator-control-plane",
        )
        netpol = _find(docs, "NetworkPolicy", "aiperf-operator")
        spec = _as_mapping(netpol["spec"], "NetworkPolicy spec")
        ingress = _as_list(spec["ingress"], "NetworkPolicy ingress")
        first_rule = _as_mapping(ingress[0], "primary ingress rule")
        source_namespaces = set()
        for source in _as_list(first_rule["from"], "primary ingress sources"):
            source_map = _as_mapping(source, "ingress source")
            namespace_selector = _as_mapping(
                source_map.get("namespaceSelector", {}), "namespace selector"
            )
            match_labels = _as_mapping(
                namespace_selector.get("matchLabels", {}), "namespace match labels"
            )
            namespace = match_labels.get("kubernetes.io/metadata.name")
            if isinstance(namespace, str):
                source_namespaces.add(namespace)

        assert source_namespaces == {
            "operator-control-plane",
            "mlperf-benchmarks",
            "vision-benchmarks",
        }
        primary_ports = {
            _as_mapping(port, "NetworkPolicy port")["port"]
            for port in _as_list(first_rule["ports"], "primary ingress ports")
        }
        assert {8080, 8081, 9090}.issubset(primary_ports)
