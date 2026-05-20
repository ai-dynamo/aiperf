# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D725-D739 infra/network/security chaos cases for DynamoGraphDeployment.

The cluster-scoped scenarios in this file are deliberately conservative. Tests
that mutate shared control-plane components, node state, webhook configuration,
or operator RBAC require explicit opt-in plus a uniquely discoverable target;
otherwise they skip before making any change.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import orjson
import pytest

from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_DGD_LABEL = "nvidia.com/dynamo-graph-deployment-name"
_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_SYSTEM_FAULT_ENV = "AIPERF_DYNAMO_CHAOS_ALLOW_SYSTEM_FAULTS"
_NODE_FAULT_LABEL = "aiperf.nvidia.com/chaos-node-faults"
_PROBE_IMAGE = "busybox:1.36"
_BOGUS_IMAGE = "nonexistent.example.com/dynamo:nope"
_NETWORK_POLICY_CNI_NEEDLES = ("cilium", "calico", "tigera", "canal", "antrea")


@dataclass(frozen=True)
class WorkloadRef:
    """A scalable workload target plus its original replica count."""

    namespace: str
    name: str
    replicas: int


@dataclass(frozen=True)
class RBACTarget:
    """A reversible Role or ClusterRole mutation target."""

    kind: Literal["role", "clusterrole"]
    name: str
    namespace: str | None
    rules: list[dict[str, Any]]
    mutated_rules: list[dict[str, Any]]

    @property
    def display_name(self) -> str:
        """Return a human-readable kubectl resource address."""
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


async def test_d725_networkpolicy_enforcement_sanity_preflight(
    kubectl: KubectlClient,
) -> None:
    """D725: prove NetworkPolicy is enforced before DNS/egress chaos tests run."""
    await _skip_unless_network_policy_cni_detected(kubectl, case_id="D725")
    namespace = "d725-netpol-preflight"
    await kubectl.create_namespace(namespace)
    try:
        await _run_probe_pod(kubectl, namespace=namespace, name="probe")
        pre = await _exec_probe(
            kubectl, namespace, "probe", "nslookup kubernetes.default.svc"
        )
        if pre.returncode != 0:
            pytest.skip(
                f"D725: DNS probe image is not usable before policy: {pre.stderr}"
            )

        await kubectl.apply(_deny_all_egress_policy(namespace, "d725-deny-all"))
        blocked = await _exec_probe(
            kubectl,
            namespace,
            "probe",
            "nslookup kubernetes.default.svc",
            timeout=20,
        )
        assert blocked.returncode != 0, (
            "D725: NetworkPolicy deny-all egress did not block DNS from the "
            "probe pod; the cluster CNI is not enforcing policies"
        )
    finally:
        await _delete_namespace(kubectl, namespace)


async def test_d726_coredns_scale_zero_blocks_dgd_startup_and_recovers(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - installs DGD CRD/operator
) -> None:
    """D726: CoreDNS scale-to-zero during DGD startup must not falsely succeed."""
    _skip_unless_system_faults_enabled("D726 scales CoreDNS to zero")
    coredns = await _find_coredns_deployment(kubectl)
    if coredns.replicas < 1:
        pytest.skip("D726: CoreDNS deployment has zero replicas before fault")

    namespace = "d726-coredns-zero"
    name = "d726-test"
    await kubectl.create_namespace(namespace)
    try:
        await _scale_workload(kubectl, coredns, replicas=0)
        await _wait_for_deployment_available(kubectl, coredns, expect_available=False)
        await kubectl.apply(_minimal_dgd_manifest(name, namespace), namespace=namespace)
        observed = await _observe_dgd_state(kubectl, name, namespace, timeout_s=45.0)
        assert observed != "successful", (
            "D726: DGD reported successful while CoreDNS was scaled to zero "
            f"(observed state={observed!r})"
        )
    finally:
        await _scale_workload(kubectl, coredns, replicas=coredns.replicas)
        await _wait_for_deployment_available(kubectl, coredns, expect_available=True)
        await _delete_namespace(kubectl, namespace)


async def test_d727_dns_denied_by_networkpolicy_blocks_worker_startup(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - installs DGD CRD/operator
) -> None:
    """D727: namespace DNS egress denial surfaces as non-successful DGD startup."""
    await _skip_unless_network_policy_cni_detected(kubectl, case_id="D727")
    namespace = "d727-dns-deny"
    name = "d727-test"
    await kubectl.create_namespace(namespace)
    try:
        await kubectl.apply(_deny_dns_egress_policy(namespace, "d727-deny-dns"))
        await kubectl.apply(_minimal_dgd_manifest(name, namespace), namespace=namespace)
        observed = await _observe_dgd_state(kubectl, name, namespace, timeout_s=90.0)
        assert observed != "successful", (
            "D727: DGD reported successful while DNS egress was denied by "
            f"NetworkPolicy (observed state={observed!r})"
        )
        status_text = await _dgd_status_text(kubectl, namespace=namespace, name=name)
        assert _mentions_any(
            status_text, ("dns", "lookup", "resolve", "network", "egress")
        ), (
            "D727: DNS-denied DGD did not surface a DNS/network hint in "
            f"status; observed status={status_text!r}"
        )
    finally:
        await _delete_namespace(kubectl, namespace)


async def test_d728_frontend_only_dns_policy_does_not_break_backend_creation(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - installs DGD CRD/operator
) -> None:
    """D728: DNS broken only for frontend pods should preserve backend children."""
    await _skip_unless_network_policy_cni_detected(kubectl, case_id="D728")
    namespace = "d728-frontend-dns"
    name = "d728-test"
    await kubectl.create_namespace(namespace)
    try:
        await kubectl.apply(_minimal_dgd_manifest(name, namespace), namespace=namespace)
        frontend_selector = await _wait_for_frontend_selector(kubectl, namespace, name)
        await kubectl.apply(
            _deny_dns_for_selector_policy(
                namespace, "d728-deny-frontend-dns", frontend_selector
            )
        )
        child_names = await _list_dgd_children(kubectl, namespace=namespace, name=name)
        assert child_names, "D728: no DGD child resources were created before DNS fault"
        observed = await _observe_dgd_state(kubectl, name, namespace, timeout_s=45.0)
        assert observed != "failed" or child_names, (
            "D728: frontend-only DNS fault removed or prevented all backend "
            "children instead of isolating the frontend dataplane"
        )
    finally:
        await _delete_namespace(kubectl, namespace)


async def test_d729_kube_proxy_restart_preserves_service_dataplane(
    kubectl: KubectlClient,
    dynamo_endpoint_url: str,
) -> None:
    """D729: kube-proxy restart during service churn should recover endpoint access."""
    _skip_unless_system_faults_enabled("D729 restarts kube-proxy")
    target = await _find_kube_proxy_daemonset(kubectl)
    pre = await _http_get_status(dynamo_endpoint_url.rstrip("/") + "/models")
    if pre >= 500:
        pytest.skip(
            f"D729: Dynamo endpoint unhealthy before kube-proxy restart: HTTP {pre}"
        )

    await kubectl.run(
        "rollout",
        "restart",
        "daemonset",
        target.name,
        "-n",
        target.namespace,
        check=True,
    )
    await kubectl.run(
        "rollout",
        "status",
        "daemonset",
        target.name,
        "-n",
        target.namespace,
        "--timeout=180s",
        check=True,
    )
    post = await _http_get_status(dynamo_endpoint_url.rstrip("/") + "/models")
    assert post < 500, (
        f"D729: Dynamo endpoint unhealthy after kube-proxy restart: {post}"
    )


async def test_d730_node_notready_fault_self_skips_without_safe_prerequisites(
    kubectl: KubectlClient,
) -> None:
    """D730: NodeNotReady is only allowed on explicitly labelled safe nodes."""
    _skip_unless_system_faults_enabled("D730 mutates node readiness")
    nodes = await _safe_fault_nodes(kubectl, required_value="notready")
    if not nodes:
        pytest.skip(
            f"D730: no node labelled {_NODE_FAULT_LABEL}=notready; refusing "
            "to synthesize NodeNotReady on an arbitrary cluster node"
        )
    pytest.skip(
        "D730: safe node was found, but this harness has no provider-specific "
        "NotReady primitive; use a provider fault injector rather than kubectl"
    )


async def test_d731_node_drain_with_dgd_pod_eviction_recovers(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - installs DGD CRD/operator
) -> None:
    """D731: drain an explicitly safe node and confirm DGD reconciliation recovers."""
    _skip_unless_system_faults_enabled("D731 drains a Kubernetes node")
    nodes = await _safe_fault_nodes(kubectl, required_value="drain")
    if len(nodes) != 1:
        pytest.skip(
            f"D731 requires exactly one node labelled {_NODE_FAULT_LABEL}=drain; "
            f"found {nodes!r}"
        )

    namespace = "d731-node-drain"
    name = "d731-test"
    node = nodes[0]
    await kubectl.create_namespace(namespace)
    try:
        await kubectl.apply(_minimal_dgd_manifest(name, namespace), namespace=namespace)
        await kubectl.run(
            "drain",
            node,
            "--ignore-daemonsets",
            "--delete-emptydir-data",
            "--force",
            "--timeout=120s",
            check=True,
            timeout=150,
        )
        await kubectl.run("uncordon", node, check=False)
        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)
    finally:
        await kubectl.run("uncordon", node, check=False)
        await _delete_namespace(kubectl, namespace)


async def test_d732_pdb_blocks_voluntary_eviction(
    kubectl: KubectlClient,
) -> None:
    """D732: a single-replica PDB must reject voluntary eviction."""
    namespace = "d732-pdb-blocks"
    await kubectl.create_namespace(namespace)
    try:
        await kubectl.apply(_pdb_probe_manifest(namespace), namespace=namespace)
        await kubectl.run(
            "wait",
            "--for=condition=Ready",
            "pod/d732-pod",
            "-n",
            namespace,
            "--timeout=90s",
            check=True,
        )
        result = await _create_eviction_result(
            kubectl,
            namespace=namespace,
            pod="d732-pod",
        )
        assert result.returncode != 0, "D732: eviction unexpectedly bypassed the PDB"
        assert _mentions_any(
            result.stderr + result.stdout, ("disruption", "pdb", "429")
        ), (
            "D732: eviction failed, but not with a PDB/disruption reason: "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    finally:
        await _delete_namespace(kubectl, namespace)


async def test_d733_serviceaccount_token_automount_disabled_surfaces_cleanly(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - installs DGD CRD/operator
) -> None:
    """D733: DGD with automountServiceAccountToken=false does not opaque-hang."""
    namespace = "d733-sa-token-off"
    name = "d733-test"
    await kubectl.create_namespace(namespace)
    try:
        await kubectl.apply(
            _service_account_manifest("d733-sa", namespace, automount=False)
        )
        await kubectl.apply(
            _minimal_dgd_manifest(
                name,
                namespace,
                extra_pod_spec={
                    "serviceAccountName": "d733-sa",
                    "automountServiceAccountToken": False,
                    "mainContainer": {
                        "image": _BOGUS_IMAGE,
                        "imagePullPolicy": "IfNotPresent",
                    },
                },
            ),
            namespace=namespace,
        )
        pod = await _wait_for_dgd_pod(
            kubectl, namespace=namespace, name=name, timeout_s=90.0
        )
        pod_spec = await _get_json(kubectl, "pod", pod, namespace=namespace)
        assert pod_spec.get("spec", {}).get("automountServiceAccountToken") is False, (
            "D733: service account token automount=false did not propagate to "
            f"DGD child pod {pod}"
        )
    finally:
        await _delete_namespace(kubectl, namespace)


async def test_d734_missing_serviceaccount_surfaces_actionable_failure(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - installs DGD CRD/operator
) -> None:
    """D734: missing ServiceAccount should surface as an actionable pod/DGD error."""
    namespace = "d734-missing-sa"
    name = "d734-test"
    await kubectl.create_namespace(namespace)
    try:
        await kubectl.apply(
            _minimal_dgd_manifest(
                name,
                namespace,
                extra_pod_spec={
                    "serviceAccountName": "d734-missing",
                    "mainContainer": {
                        "image": _BOGUS_IMAGE,
                        "imagePullPolicy": "IfNotPresent",
                    },
                },
            ),
            namespace=namespace,
        )
        text = await _wait_for_events_or_status(
            kubectl,
            namespace=namespace,
            name=name,
            needles=("serviceaccount", "d734-missing", "not found", "forbidden"),
            timeout_s=120.0,
        )
        assert _mentions_any(text, ("serviceaccount", "d734-missing")), (
            f"D734: missing ServiceAccount was not named in status/events: {text!r}"
        )
    finally:
        await _delete_namespace(kubectl, namespace)


async def test_d735_rbac_revoke_operator_dgd_watch_self_skips_unless_reversible(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - ensures operator exists before RBAC discovery
) -> None:
    """D735: revoke DGD watch/list RBAC only when a reversible grant is found."""
    _skip_unless_system_faults_enabled("D735 revokes operator DGD watch/list RBAC")
    target = await _find_reversible_operator_rbac(
        kubectl,
        api_group="nvidia.com",
        resource="dynamographdeployments",
        verbs=("watch", "list"),
        case_id="D735",
    )
    await _patch_rbac_rules(kubectl, target, target.mutated_rules)
    try:
        probe = await kubectl.run(
            "auth",
            "can-i",
            "watch",
            "dynamographdeployments.nvidia.com",
            "--as",
            await _operator_sa_user(kubectl),
            check=False,
        )
        assert probe.returncode != 0 or "no" in probe.stdout.lower(), (
            f"D735: RBAC revoke did not remove watch permission from {target.display_name}"
        )
    finally:
        await _patch_rbac_rules(kubectl, target, target.rules)


async def test_d736_rbac_revoke_operator_child_create_self_skips_unless_reversible(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - ensures operator exists before RBAC discovery
) -> None:
    """D736: revoke child create RBAC only when a reversible grant is found."""
    _skip_unless_system_faults_enabled(
        "D736 revokes operator child-resource create RBAC"
    )
    target = await _find_reversible_operator_rbac(
        kubectl,
        api_group="apps",
        resource="deployments",
        verbs=("create",),
        case_id="D736",
    )
    await _patch_rbac_rules(kubectl, target, target.mutated_rules)
    try:
        probe = await kubectl.run(
            "auth",
            "can-i",
            "create",
            "deployments.apps",
            "--as",
            await _operator_sa_user(kubectl),
            check=False,
        )
        assert probe.returncode != 0 or "no" in probe.stdout.lower(), (
            f"D736: RBAC revoke did not remove create permission from {target.display_name}"
        )
    finally:
        await _patch_rbac_rules(kubectl, target, target.rules)


async def test_d737_webhook_service_unavailable_blocks_or_defers_dgd_apply(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - installs webhook/CRD/operator
) -> None:
    """D737: unavailable validating webhook should block apply or prevent children."""
    target = await _find_dgd_webhook_deployment(kubectl)
    namespace = "d737-webhook-unavailable"
    name = "d737-test"
    await kubectl.create_namespace(namespace)
    applied = False
    try:
        await _scale_workload(kubectl, target, replicas=0)
        await _wait_for_deployment_available(kubectl, target, expect_available=False)
        result = await _apply_manifest_result(
            kubectl,
            _minimal_dgd_manifest(name, namespace),
            namespace=namespace,
        )
        if result.returncode != 0:
            assert _mentions_any(
                result.stderr, ("webhook", "admission", "endpoint", "timeout")
            ), (
                "D737: apply failed while webhook was unavailable, but error "
                f"did not name admission/webhook unavailability: {result.stderr!r}"
            )
            return
        applied = True
        await asyncio.sleep(10.0)
        children = await _list_dgd_children(kubectl, namespace=namespace, name=name)
        assert not children, (
            "D737: DGD was admitted while webhook was unavailable and children "
            f"were created before restore: {children!r}"
        )
    finally:
        await _scale_workload(kubectl, target, replicas=target.replicas)
        await _wait_for_deployment_available(kubectl, target, expect_available=True)
        if applied:
            await _delete_dgd(kubectl, namespace=namespace, name=name)
        await _delete_namespace(kubectl, namespace)


async def test_d738_webhook_timeout_failurepolicy_behavior_is_reversible(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - installs webhook/CRD/operator
) -> None:
    """D738: webhook timeout/failurePolicy mutation must be reversible and explicit."""
    _skip_unless_system_faults_enabled("D738 patches validating webhook configuration")
    config_name, webhook_index, original = await _find_dgd_webhook_config(kubectl)
    patched = dict(original)
    patched["timeoutSeconds"] = 1
    patched["failurePolicy"] = "Fail"
    await _patch_webhook(kubectl, config_name, webhook_index, patched)
    try:
        current = await _get_json(
            kubectl, "validatingwebhookconfiguration", config_name
        )
        webhook = current["webhooks"][webhook_index]
        assert webhook.get("timeoutSeconds") == 1
        assert webhook.get("failurePolicy") == "Fail"
    finally:
        await _patch_webhook(kubectl, config_name, webhook_index, original)


async def test_d739_finalizer_cleanup_under_namespace_deletion(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - installs DGD CRD/operator
) -> None:
    """D739: namespace deletion with a DGD must not leave finalizers stuck."""
    namespace = "d739-finalizer-cleanup"
    name = "d739-test"
    await kubectl.create_namespace(namespace)
    await kubectl.apply(_minimal_dgd_manifest(name, namespace), namespace=namespace)
    await _delete_namespace(kubectl, namespace)
    gone = await _wait_for_namespace_absent(
        kubectl, namespace=namespace, timeout_s=120.0
    )
    assert gone, (
        "D739: namespace deletion did not complete within 120s; a DGD or child "
        "resource finalizer may be stuck"
    )


async def _skip_unless_network_policy_cni_detected(
    kubectl: KubectlClient,
    *,
    case_id: str,
) -> None:
    api_result = await kubectl.run(
        "api-resources",
        "--api-group=networking.k8s.io",
        "-o",
        "name",
        check=False,
    )
    if api_result.returncode != 0 or "networkpolicies" not in api_result.stdout:
        pytest.skip(
            f"{case_id}: cluster does not expose networking.k8s.io NetworkPolicy"
        )

    pods_result = await kubectl.run("get", "pods", "-A", "-o", "json", check=False)
    if pods_result.returncode != 0:
        pytest.skip(f"{case_id}: cannot inspect CNI pods before NetworkPolicy test")
    pod_data = orjson.loads(pods_result.stdout or b"{}")
    cni_text = " ".join(
        f"{item.get('metadata', {}).get('namespace', '')}/"
        f"{item.get('metadata', {}).get('name', '')} "
        f"{item.get('metadata', {}).get('labels', {})}"
        for item in pod_data.get("items", [])
    ).lower()
    if not any(needle in cni_text for needle in _NETWORK_POLICY_CNI_NEEDLES):
        pytest.skip(
            f"{case_id}: requires a NetworkPolicy-enforcing CNI such as Cilium "
            "or Calico; kindnet accepts policies but does not enforce them"
        )


def _skip_unless_system_faults_enabled(reason: str) -> None:
    enabled = os.environ.get(_SYSTEM_FAULT_ENV, "").lower() in {"1", "true", "yes"}
    if not enabled:
        pytest.skip(
            f"{reason}; set {_SYSTEM_FAULT_ENV}=1 to allow shared-cluster mutation"
        )


async def _run_probe_pod(kubectl: KubectlClient, *, namespace: str, name: str) -> None:
    await kubectl.run(
        "run",
        name,
        "-n",
        namespace,
        "--image",
        _PROBE_IMAGE,
        "--restart=Never",
        "--command",
        "--",
        "sleep",
        "3600",
        check=True,
    )
    ready = await kubectl.run(
        "wait",
        "--for=condition=Ready",
        f"pod/{name}",
        "-n",
        namespace,
        "--timeout=90s",
        check=False,
    )
    if ready.returncode != 0:
        pytest.skip(
            f"probe pod {namespace}/{name} did not become Ready: {ready.stderr}"
        )


async def _exec_probe(
    kubectl: KubectlClient,
    namespace: str,
    pod: str,
    command: str,
    *,
    timeout: int = 30,
) -> Any:
    return await kubectl.run(
        "exec",
        "-n",
        namespace,
        pod,
        "--",
        "sh",
        "-c",
        command,
        check=False,
        timeout=timeout,
    )


async def _get_json(
    kubectl: KubectlClient,
    resource: str,
    name: str | None = None,
    *,
    namespace: str | None = None,
) -> dict[str, Any]:
    args = ["get", resource]
    if name is not None:
        args.append(name)
    if namespace is not None:
        args.extend(["-n", namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=True)
    return orjson.loads(result.stdout)


async def _apply_manifest_result(
    kubectl: KubectlClient,
    manifest: str,
    *,
    namespace: str,
) -> SimpleNamespace:
    cmd = kubectl._build_cmd("apply", "-f", "-", namespace=namespace)  # noqa: SLF001
    return await _kubectl_stdin(cmd, stdin=manifest)


async def _create_eviction_result(
    kubectl: KubectlClient,
    *,
    namespace: str,
    pod: str,
) -> SimpleNamespace:
    manifest = orjson.dumps(
        {
            "apiVersion": "policy/v1",
            "kind": "Eviction",
            "metadata": {"name": pod, "namespace": namespace},
        }
    ).decode()
    cmd = kubectl._build_cmd(  # noqa: SLF001
        "create",
        "--raw",
        f"/api/v1/namespaces/{namespace}/pods/{pod}/eviction",
        "-f",
        "-",
    )
    return await _kubectl_stdin(cmd, stdin=manifest, timeout_s=20.0)


async def _kubectl_stdin(
    cmd: list[str],
    *,
    stdin: str,
    timeout_s: float = 120.0,
) -> SimpleNamespace:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin.encode()),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return SimpleNamespace(
            returncode=-9,
            stdout="",
            stderr=f"kubectl stdin command timed out after {timeout_s}s",
        )
    return SimpleNamespace(
        returncode=proc.returncode,
        stdout=stdout.decode() if stdout else "",
        stderr=stderr.decode() if stderr else "",
    )


async def _delete_namespace(kubectl: KubectlClient, namespace: str) -> None:
    await kubectl.run(
        "delete",
        "namespace",
        namespace,
        "--wait=false",
        "--ignore-not-found",
        check=False,
    )


async def _wait_for_namespace_absent(
    kubectl: KubectlClient,
    *,
    namespace: str,
    timeout_s: float,
) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        result = await kubectl.run("get", "namespace", namespace, check=False)
        if result.returncode != 0:
            return True
        await asyncio.sleep(2.0)
    return False


async def _delete_dgd(kubectl: KubectlClient, *, namespace: str, name: str) -> None:
    await kubectl.run(
        "delete",
        "dynamographdeployment",
        name,
        "-n",
        namespace,
        "--wait=false",
        "--ignore-not-found",
        check=False,
    )


def _minimal_dgd_manifest(
    name: str,
    namespace: str,
    *,
    extra_pod_spec: dict[str, Any] | None = None,
) -> str:
    pod_spec = extra_pod_spec or {
        "mainContainer": {"image": _BOGUS_IMAGE, "imagePullPolicy": "IfNotPresent"}
    }
    manifest = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": pod_spec,
                }
            }
        },
    }
    return orjson.dumps(manifest).decode()


def _deny_all_egress_policy(namespace: str, name: str) -> str:
    return orjson.dumps(
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {"name": name, "namespace": namespace},
            "spec": {"podSelector": {}, "policyTypes": ["Egress"], "egress": []},
        }
    ).decode()


def _deny_dns_egress_policy(namespace: str, name: str) -> str:
    return _deny_dns_for_selector_policy(namespace, name, selector={})


def _deny_dns_for_selector_policy(
    namespace: str,
    name: str,
    selector: dict[str, str],
) -> str:
    match_labels = selector if selector else {}
    return orjson.dumps(
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {"name": name, "namespace": namespace},
            "spec": {
                "podSelector": {"matchLabels": match_labels},
                "policyTypes": ["Egress"],
                "egress": [
                    {
                        "to": [{"ipBlock": {"cidr": "0.0.0.0/0"}}],
                        "ports": [
                            {"protocol": "TCP", "port": 1},
                            {"protocol": "UDP", "port": 1},
                        ],
                    }
                ],
            },
        }
    ).decode()


async def _observe_dgd_state(
    kubectl: KubectlClient,
    name: str,
    namespace: str,
    *,
    timeout_s: float,
) -> str:
    deadline = asyncio.get_event_loop().time() + timeout_s
    last_state = "<unobserved>"
    while asyncio.get_event_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "jsonpath={.status.state}",
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            last_state = result.stdout.strip()
            if last_state in {"successful", "failed"}:
                return last_state
        await asyncio.sleep(2.0)
    return last_state


async def _dgd_status_text(kubectl: KubectlClient, *, namespace: str, name: str) -> str:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        namespace,
        "-o",
        "jsonpath={.status}",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _mentions_any(text: str, needles: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(needle.lower() in lower for needle in needles)


async def _wait_for_frontend_selector(
    kubectl: KubectlClient,
    namespace: str,
    name: str,
) -> dict[str, str]:
    pod = await _wait_for_dgd_pod(
        kubectl, namespace=namespace, name=name, timeout_s=90.0
    )
    data = await _get_json(kubectl, "pod", pod, namespace=namespace)
    labels = data.get("metadata", {}).get("labels", {})
    for key in (
        "app.kubernetes.io/component",
        "nvidia.com/dynamo-component",
        "nvidia.com/dynamo-graph-deployment-name",
    ):
        value = labels.get(key)
        if value:
            return {key: value}
    pytest.skip(f"D728: frontend pod {pod} had no stable selector labels: {labels!r}")


async def _wait_for_dgd_pod(
    kubectl: KubectlClient,
    *,
    namespace: str,
    name: str,
    timeout_s: float,
) -> str:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"{_DGD_LABEL}={name}",
            "-o",
            "jsonpath={.items[0].metadata.name}",
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        await asyncio.sleep(2.0)
    raise AssertionError(f"DGD {namespace}/{name} did not create a child pod")


async def _list_dgd_children(
    kubectl: KubectlClient,
    *,
    namespace: str,
    name: str,
) -> list[str]:
    result = await kubectl.run(
        "get",
        "deployment,service,configmap,role,rolebinding,serviceaccount,pod",
        "-n",
        namespace,
        "-l",
        f"{_DGD_LABEL}={name}",
        "-o",
        "name",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


async def _find_coredns_deployment(kubectl: KubectlClient) -> WorkloadRef:
    for namespace, name in (("kube-system", "coredns"), ("kube-system", "kube-dns")):
        result = await kubectl.run(
            "get",
            "deployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            data = orjson.loads(result.stdout)
            return WorkloadRef(
                namespace=namespace,
                name=name,
                replicas=int(data.get("spec", {}).get("replicas") or 0),
            )
    pytest.skip("D726: no kube-system CoreDNS/kube-dns Deployment found")


async def _find_kube_proxy_daemonset(kubectl: KubectlClient) -> WorkloadRef:
    result = await kubectl.run(
        "get",
        "daemonset",
        "-A",
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("D729: cannot list DaemonSets to find kube-proxy")
    data = orjson.loads(result.stdout)
    matches = []
    for item in data.get("items", []):
        name = item.get("metadata", {}).get("name", "")
        namespace = item.get("metadata", {}).get("namespace", "")
        if "kube-proxy" in f"{namespace}/{name}":
            matches.append(WorkloadRef(namespace=namespace, name=name, replicas=1))
    if len(matches) != 1:
        pytest.skip(
            f"D729 requires exactly one kube-proxy DaemonSet; found {matches!r}"
        )
    return matches[0]


async def _scale_workload(
    kubectl: KubectlClient,
    target: WorkloadRef,
    *,
    replicas: int,
) -> None:
    await kubectl.run(
        "scale",
        "deployment",
        target.name,
        "-n",
        target.namespace,
        f"--replicas={replicas}",
        check=True,
    )


async def _wait_for_deployment_available(
    kubectl: KubectlClient,
    target: WorkloadRef,
    *,
    expect_available: bool,
) -> None:
    if expect_available:
        await kubectl.run(
            "rollout",
            "status",
            "deployment",
            target.name,
            "-n",
            target.namespace,
            "--timeout=180s",
            check=False,
        )
        return
    await asyncio.sleep(5.0)


async def _http_get_status(url: str) -> int:
    import aiohttp

    try:
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.0)) as session,
            session.get(url) as resp,
        ):
            await resp.read()
            return resp.status
    except aiohttp.ClientError:
        return 599


async def _safe_fault_nodes(
    kubectl: KubectlClient,
    *,
    required_value: str,
) -> list[str]:
    result = await kubectl.run(
        "get",
        "nodes",
        "-l",
        f"{_NODE_FAULT_LABEL}={required_value}",
        "-o",
        "jsonpath={.items[*].metadata.name}",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return result.stdout.split()


def _pdb_probe_manifest(namespace: str) -> str:
    docs = [
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "d732-pod",
                "namespace": namespace,
                "labels": {"app": "d732"},
            },
            "spec": {
                "containers": [
                    {
                        "name": "pause",
                        "image": "registry.k8s.io/pause:3.9",
                    }
                ]
            },
        },
        {
            "apiVersion": "policy/v1",
            "kind": "PodDisruptionBudget",
            "metadata": {"name": "d732-pdb", "namespace": namespace},
            "spec": {"minAvailable": 1, "selector": {"matchLabels": {"app": "d732"}}},
        },
    ]
    return "\n---\n".join(orjson.dumps(doc).decode() for doc in docs)


def _service_account_manifest(name: str, namespace: str, *, automount: bool) -> str:
    return orjson.dumps(
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {"name": name, "namespace": namespace},
            "automountServiceAccountToken": automount,
        }
    ).decode()


async def _wait_for_events_or_status(
    kubectl: KubectlClient,
    *,
    namespace: str,
    name: str,
    needles: tuple[str, ...],
    timeout_s: float,
) -> str:
    deadline = asyncio.get_event_loop().time() + timeout_s
    combined = ""
    while asyncio.get_event_loop().time() < deadline:
        status = await _dgd_status_text(kubectl, namespace=namespace, name=name)
        events = await kubectl.run("get", "events", "-n", namespace, check=False)
        combined = f"{status}\n{events.stdout}\n{events.stderr}"
        if _mentions_any(combined, needles):
            return combined
        await asyncio.sleep(2.0)
    return combined


async def _operator_sa_user(kubectl: KubectlClient) -> str:
    sa = await _operator_service_account(kubectl)
    return f"system:serviceaccount:{_OPERATOR_NAMESPACE}:{sa}"


async def _operator_service_account(kubectl: KubectlClient) -> str:
    result = await kubectl.run(
        "get",
        "deployment",
        "-n",
        _OPERATOR_NAMESPACE,
        "-l",
        _OPERATOR_SELECTOR,
        "-o",
        "json",
        check=True,
    )
    deployments = orjson.loads(result.stdout).get("items", [])
    if len(deployments) != 1:
        names = [
            item.get("metadata", {}).get("name", "<unnamed>") for item in deployments
        ]
        pytest.skip(
            "operator RBAC test requires exactly one Dynamo operator deployment; "
            f"found {names!r}"
        )
    return deployments[0]["spec"]["template"]["spec"].get(
        "serviceAccountName", "default"
    )


async def _find_reversible_operator_rbac(
    kubectl: KubectlClient,
    *,
    api_group: str,
    resource: str,
    verbs: tuple[str, ...],
    case_id: str,
) -> RBACTarget:
    service_account = await _operator_service_account(kubectl)
    candidates = await _operator_bound_targets(kubectl, service_account)
    matches: list[RBACTarget] = []
    for target in candidates:
        mutated = _without_rule_verbs(target.rules, api_group, resource, verbs)
        if mutated != target.rules:
            matches.append(
                RBACTarget(
                    kind=target.kind,
                    name=target.name,
                    namespace=target.namespace,
                    rules=target.rules,
                    mutated_rules=mutated,
                )
            )
    if len(matches) != 1:
        inspected = (
            ", ".join(candidate.display_name for candidate in candidates) or "<none>"
        )
        pytest.skip(
            f"{case_id}: requires exactly one reversible RBAC target for "
            f"{api_group}/{resource} verbs={verbs!r}; inspected {inspected}"
        )
    return matches[0]


async def _operator_bound_targets(
    kubectl: KubectlClient,
    service_account: str,
) -> list[RBACTarget]:
    refs: list[tuple[Literal["role", "clusterrole"], str, str | None]] = []
    refs.extend(
        await _bound_role_refs(
            kubectl,
            "rolebinding",
            service_account,
            namespaced=True,
        )
    )
    refs.extend(
        await _bound_role_refs(
            kubectl,
            "clusterrolebinding",
            service_account,
            namespaced=False,
        )
    )
    targets: list[RBACTarget] = []
    for kind, name, namespace in refs:
        target = await _load_rbac_target(kubectl, kind, name, namespace)
        if target is not None:
            targets.append(target)
    return targets


async def _bound_role_refs(
    kubectl: KubectlClient,
    binding_kind: Literal["rolebinding", "clusterrolebinding"],
    service_account: str,
    *,
    namespaced: bool,
) -> list[tuple[Literal["role", "clusterrole"], str, str | None]]:
    args = ["get", binding_kind]
    if namespaced:
        args.extend(["-n", _OPERATOR_NAMESPACE])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=True)
    bindings = orjson.loads(result.stdout).get("items", [])
    refs: list[tuple[Literal["role", "clusterrole"], str, str | None]] = []
    for binding in bindings:
        subjects = binding.get("subjects", [])
        if not _has_operator_subject(subjects, service_account):
            continue
        role_ref = binding.get("roleRef", {})
        ref_kind = role_ref.get("kind", "").lower()
        if ref_kind not in {"role", "clusterrole"}:
            continue
        namespace = (
            binding.get("metadata", {}).get("namespace") if ref_kind == "role" else None
        )
        refs.append((ref_kind, role_ref["name"], namespace))
    return refs


async def _load_rbac_target(
    kubectl: KubectlClient,
    kind: Literal["role", "clusterrole"],
    name: str,
    namespace: str | None,
) -> RBACTarget | None:
    args = ["get", kind, name]
    if namespace is not None:
        args.extend(["-n", namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        return None
    obj = orjson.loads(result.stdout)
    return RBACTarget(
        kind=kind,
        name=name,
        namespace=namespace,
        rules=obj.get("rules", []),
        mutated_rules=obj.get("rules", []),
    )


def _has_operator_subject(subjects: list[dict[str, Any]], service_account: str) -> bool:
    for subject in subjects:
        if (
            subject.get("kind") == "ServiceAccount"
            and subject.get("name") == service_account
            and subject.get("namespace") == _OPERATOR_NAMESPACE
        ):
            return True
    return False


def _without_rule_verbs(
    rules: list[dict[str, Any]],
    api_group: str,
    resource: str,
    verbs: tuple[str, ...],
) -> list[dict[str, Any]]:
    mutated: list[dict[str, Any]] = []
    verb_set = set(verbs)
    for rule in rules:
        if _rule_matches(rule, api_group, resource, verb_set):
            new_rule = dict(rule)
            new_rule["verbs"] = [
                verb for verb in rule.get("verbs", []) if verb not in verb_set
            ]
            if new_rule["verbs"]:
                mutated.append(new_rule)
        else:
            mutated.append(dict(rule))
    return mutated


def _rule_matches(
    rule: dict[str, Any],
    api_group: str,
    resource: str,
    verbs: set[str],
) -> bool:
    api_groups = set(rule.get("apiGroups", []))
    resources = set(rule.get("resources", []))
    rule_verbs = set(rule.get("verbs", []))
    return (
        api_group in api_groups
        and resource in resources
        and verbs.issubset(rule_verbs)
        and "*" not in api_groups
        and "*" not in resources
        and "*" not in rule_verbs
    )


async def _patch_rbac_rules(
    kubectl: KubectlClient,
    target: RBACTarget,
    rules: list[dict[str, Any]],
) -> None:
    patch = orjson.dumps({"rules": rules}).decode()
    args = ["patch", target.kind, target.name, "--type=merge", f"-p={patch}"]
    if target.namespace is not None:
        args.extend(["-n", target.namespace])
    await kubectl.run(*args, check=True)


async def _find_dgd_webhook_deployment(kubectl: KubectlClient) -> WorkloadRef:
    services = await _dgd_webhook_services(kubectl)
    if not services:
        pytest.skip("D737: no DGD validating webhook service found")
    candidates: list[WorkloadRef] = []
    for namespace, service_name in services:
        service = await _get_json(kubectl, "service", service_name, namespace=namespace)
        selector = service.get("spec", {}).get("selector") or {}
        if not selector:
            continue
        candidates.extend(
            await _deployments_matching_selector(kubectl, namespace, selector)
        )
    unique = {
        (candidate.namespace, candidate.name): candidate for candidate in candidates
    }
    if len(unique) != 1:
        pytest.skip(f"D737: webhook deployment not uniquely identified: {candidates!r}")
    return next(iter(unique.values()))


async def _dgd_webhook_services(kubectl: KubectlClient) -> list[tuple[str, str]]:
    result = await kubectl.run(
        "get",
        "validatingwebhookconfigurations",
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    data = orjson.loads(result.stdout)
    services: list[tuple[str, str]] = []
    for item in data.get("items", []):
        for webhook in item.get("webhooks", []):
            if not _webhook_validates_dgd(webhook.get("rules", []) or []):
                continue
            service = webhook.get("clientConfig", {}).get("service") or {}
            namespace = service.get("namespace", "")
            name = service.get("name", "")
            if namespace and name:
                services.append((namespace, name))
    return services


def _webhook_validates_dgd(rules: list[dict[str, Any]]) -> bool:
    for rule in rules:
        groups = rule.get("apiGroups") or []
        resources = rule.get("resources") or []
        if "nvidia.com" in groups and any(
            str(resource).startswith("dynamographdeployments") for resource in resources
        ):
            return True
    return False


async def _deployments_matching_selector(
    kubectl: KubectlClient,
    namespace: str,
    selector: dict[str, str],
) -> list[WorkloadRef]:
    data = await _get_json(kubectl, "deployment", namespace=namespace)
    matches: list[WorkloadRef] = []
    for item in data.get("items", []):
        labels = (
            item.get("spec", {})
            .get("template", {})
            .get("metadata", {})
            .get("labels", {})
        )
        if all(labels.get(key) == value for key, value in selector.items()):
            matches.append(
                WorkloadRef(
                    namespace=namespace,
                    name=item.get("metadata", {}).get("name", ""),
                    replicas=int(item.get("spec", {}).get("replicas") or 0),
                )
            )
    return matches


async def _find_dgd_webhook_config(
    kubectl: KubectlClient,
) -> tuple[str, int, dict[str, Any]]:
    result = await kubectl.run(
        "get",
        "validatingwebhookconfigurations",
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("D738: cannot list ValidatingWebhookConfiguration objects")
    data = orjson.loads(result.stdout)
    matches: list[tuple[str, int, dict[str, Any]]] = []
    for item in data.get("items", []):
        name = item.get("metadata", {}).get("name", "")
        for idx, webhook in enumerate(item.get("webhooks", [])):
            if _webhook_validates_dgd(webhook.get("rules", []) or []):
                matches.append((name, idx, webhook))
    if len(matches) != 1:
        pytest.skip(f"D738: DGD webhook config not uniquely identified: {matches!r}")
    return matches[0]


async def _patch_webhook(
    kubectl: KubectlClient,
    config_name: str,
    webhook_index: int,
    webhook: dict[str, Any],
) -> None:
    patch = orjson.dumps(
        [{"op": "replace", "path": f"/webhooks/{webhook_index}", "value": webhook}]
    ).decode()
    await kubectl.run(
        "patch",
        "validatingwebhookconfiguration",
        config_name,
        "--type=json",
        f"-p={patch}",
        check=True,
    )
