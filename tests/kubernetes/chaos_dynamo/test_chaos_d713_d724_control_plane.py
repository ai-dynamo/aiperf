# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D713-D724 -- Dynamo infra/control-plane chaos coverage.

These scenarios exercise operator retry/idempotence and Kubernetes pod-template
controls that commonly break during CRD/operator upgrades. The tests use
isolated namespaces and only mutate resources they create, except for explicit
control-plane chaos cases that dry-run CRD deletion or delete one self-healing
operator/CNI pod.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import orjson
import pytest
import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_DGD_CRD = "dynamographdeployments.nvidia.com"
_DGD_KIND = "dynamographdeployment"
_DGD_LABEL = "nvidia.com/dynamo-graph-deployment-name"
_WORKER_LABEL = "nvidia.com/dynamo-component-type=worker"
_STATUS_CONFLICT_WINDOW_S = 20.0
_STATUS_TIMEOUT_S = 120.0
_POD_TIMEOUT_S = 120.0
_RUNTIME_CLASS_NAME = "nvidia"
_CNI_DAEMONSET_CANDIDATES = (
    "cilium",
    "calico-node",
    "kube-flannel-ds",
    "kindnet",
    "kindnetd",
)


def _namespace(case_id: str) -> str:
    """Return the isolated namespace for one D7xx scenario."""
    return f"{case_id.lower()}-dynamo-chaos"


async def test_d713_status_conflict_retry_reaches_terminal_state(
    kubectl: KubectlClient,
    dynamo_operator: None,  # noqa: ARG001 - fixture ensures the operator is installed
) -> None:
    """Patch DGD status concurrently and require operator status retry recovery."""
    namespace = _namespace("d713")
    deployer = _alpha_deployer(kubectl, namespace)
    name = deployer._deployment_name()
    stop = asyncio.Event()
    patch_task: asyncio.Task[None] | None = None

    await kubectl.delete_namespace(namespace, wait=True)
    await kubectl.create_namespace(namespace)
    try:
        await kubectl.apply(deployer.generate_manifest())
        patch_task = asyncio.create_task(
            _status_patch_loop(kubectl, namespace, name, stop),
            name="d713-status-conflict-patcher",
        )
        await asyncio.sleep(_STATUS_CONFLICT_WINDOW_S)
        stop.set()
        await patch_task
        observed = await wait_for_dgd_state(
            kubectl,
            name,
            namespace,
            "successful",
            timeout=300.0,
        )
        assert observed == "successful"
    finally:
        stop.set()
        if patch_task is not None:
            await asyncio.gather(patch_task, return_exceptions=True)
        await _delete_namespace_fast(kubectl, namespace)


async def test_d714_initial_status_after_operator_kill(
    kubectl: KubectlClient,
    dynamo_operator: None,  # noqa: ARG001 - fixture ensures the operator is installed
) -> None:
    """Kill the operator just after CR creation and require status initialization."""
    namespace = _namespace("d714")
    deployer = _alpha_deployer(kubectl, namespace)
    name = deployer._deployment_name()

    await kubectl.delete_namespace(namespace, wait=True)
    await kubectl.create_namespace(namespace)
    try:
        await kubectl.apply(deployer.generate_manifest())
        await _kill_operator_pod(kubectl)
        state = await _wait_for_non_empty_dgd_state(
            kubectl,
            namespace=namespace,
            name=name,
            timeout=_STATUS_TIMEOUT_S,
        )
        assert state in {"pending", "successful", "failed"}, (
            f"D714: operator restart initialized unexpected DGD state {state!r}"
        )
        await _wait_for_operator_available(kubectl, timeout=180.0)
    finally:
        await _delete_namespace_fast(kubectl, namespace)


async def test_d715_crd_delete_dry_run_guard_preserves_dgd_crd(
    kubectl: KubectlClient,
    dynamo_operator: None,  # noqa: ARG001 - fixture ensures the CRD is installed
) -> None:
    """Dry-run CRD deletion must not remove the DynamoGraphDeployment CRD."""
    before_uid = await _crd_uid(kubectl)
    result = await kubectl.run(
        "delete",
        "crd",
        _DGD_CRD,
        "--dry-run=server",
        "-o",
        "yaml",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            "D715 requires delete dry-run permission on the Dynamo CRD; "
            f"kubectl returned: {result.stderr.strip() or result.stdout.strip()}"
        )
    after_uid = await _crd_uid(kubectl)
    assert after_uid == before_uid, "D715: CRD UID changed after dry-run delete"
    assert "deletionTimestamp" not in result.stdout, (
        "D715: dry-run output included deletionTimestamp, suggesting a real delete"
    )


async def test_d716_read_only_cache_volume_mount_reaches_child_pod(
    kubectl: KubectlClient,
    dynamo_operator: None,  # noqa: ARG001 - fixture ensures the operator is installed
) -> None:
    """A read-only model/cache mount in the DGD must propagate to child pods."""
    namespace = _namespace("d716")
    pod = await _apply_beta_manifest_and_wait_for_worker_pod(
        kubectl,
        namespace,
        mutate_worker=lambda component: _set_worker_volume_mount(
            component,
            volume_name="d716-cache",
            mount_path="/models",
            read_only=True,
        ),
    )
    mount = _main_container(pod)["volumeMounts"][0]
    assert mount["name"] == "d716-cache"
    assert mount["mountPath"] == "/models"
    assert mount["readOnly"] is True


async def test_d717_pod_and_container_security_contexts_reach_child_pod(
    kubectl: KubectlClient,
    dynamo_operator: None,  # noqa: ARG001 - fixture ensures the operator is installed
) -> None:
    """Pod-level and container-level security contexts must survive reconcile."""
    namespace = _namespace("d717")
    pod = await _apply_beta_manifest_and_wait_for_worker_pod(
        kubectl,
        namespace,
        mutate_worker=lambda component: _set_security_contexts(component),
    )
    assert pod["spec"]["securityContext"]["runAsNonRoot"] is True
    container_security = _main_container(pod)["securityContext"]
    assert container_security["runAsUser"] == 1000
    assert container_security["runAsGroup"] == 1000


async def test_d718_container_capability_drop_reaches_child_pod(
    kubectl: KubectlClient,
    dynamo_operator: None,  # noqa: ARG001 - fixture ensures the operator is installed
) -> None:
    """Capability drop settings must propagate without operator default loss."""
    namespace = _namespace("d718")
    pod = await _apply_beta_manifest_and_wait_for_worker_pod(
        kubectl,
        namespace,
        mutate_worker=lambda component: _set_container_security(
            component,
            {"capabilities": {"drop": ["ALL"]}},
        ),
    )
    assert _main_container(pod)["securityContext"]["capabilities"]["drop"] == ["ALL"]


async def test_d719_read_only_root_filesystem_reaches_child_pod(
    kubectl: KubectlClient,
    dynamo_operator: None,  # noqa: ARG001 - fixture ensures the operator is installed
) -> None:
    """readOnlyRootFilesystem must survive the DGD -> Deployment -> Pod path."""
    namespace = _namespace("d719")
    pod = await _apply_beta_manifest_and_wait_for_worker_pod(
        kubectl,
        namespace,
        mutate_worker=lambda component: _set_container_security(
            component,
            {"readOnlyRootFilesystem": True},
        ),
    )
    assert _main_container(pod)["securityContext"]["readOnlyRootFilesystem"] is True


async def test_d720_privilege_escalation_disabled_reaches_child_pod(
    kubectl: KubectlClient,
    dynamo_operator: None,  # noqa: ARG001 - fixture ensures the operator is installed
) -> None:
    """allowPrivilegeEscalation=false must survive operator reconciliation."""
    namespace = _namespace("d720")
    pod = await _apply_beta_manifest_and_wait_for_worker_pod(
        kubectl,
        namespace,
        mutate_worker=lambda component: _set_container_security(
            component,
            {"allowPrivilegeEscalation": False},
        ),
    )
    assert _main_container(pod)["securityContext"]["allowPrivilegeEscalation"] is False


async def test_d721_runtimeclass_reaches_child_pod_when_available(
    kubectl: KubectlClient,
    dynamo_operator: None,  # noqa: ARG001 - fixture ensures the operator is installed
) -> None:
    """runtimeClassName=nvidia must propagate on clusters that advertise it."""
    await _require_runtime_class(kubectl, _RUNTIME_CLASS_NAME, case_id="D721")
    namespace = _namespace("d721")
    pod = await _apply_beta_manifest_and_wait_for_worker_pod(
        kubectl,
        namespace,
        mutate_worker=lambda component: _pod_spec(component).__setitem__(
            "runtimeClassName",
            _RUNTIME_CLASS_NAME,
        ),
    )
    assert pod["spec"]["runtimeClassName"] == _RUNTIME_CLASS_NAME


async def test_d722_toleration_for_cluster_taint_reaches_child_pod(
    kubectl: KubectlClient,
    dynamo_operator: None,  # noqa: ARG001 - fixture ensures the operator is installed
) -> None:
    """A toleration for an existing NoSchedule taint must propagate to pods."""
    taint = await _require_schedulable_taint(kubectl, case_id="D722")
    namespace = _namespace("d722")
    pod = await _apply_beta_manifest_and_wait_for_worker_pod(
        kubectl,
        namespace,
        mutate_worker=lambda component: _pod_spec(component).__setitem__(
            "tolerations",
            [taint],
        ),
    )
    assert taint in pod["spec"].get("tolerations", [])


async def test_d723_affinity_reaches_child_pod(
    kubectl: KubectlClient,
    dynamo_operator: None,  # noqa: ARG001 - fixture ensures the operator is installed
) -> None:
    """Node affinity in a DGD podTemplate must propagate to the worker pod."""
    namespace = _namespace("d723")
    affinity = {
        "nodeAffinity": {
            "preferredDuringSchedulingIgnoredDuringExecution": [
                {
                    "weight": 1,
                    "preference": {
                        "matchExpressions": [
                            {
                                "key": "kubernetes.io/os",
                                "operator": "In",
                                "values": ["linux"],
                            }
                        ]
                    },
                }
            ]
        }
    }
    pod = await _apply_beta_manifest_and_wait_for_worker_pod(
        kubectl,
        namespace,
        mutate_worker=lambda component: _pod_spec(component).__setitem__(
            "affinity",
            affinity,
        ),
    )
    assert pod["spec"]["affinity"] == affinity


async def test_d724_topology_spread_survives_single_cni_pod_kill(
    kubectl: KubectlClient,
    dynamo_operator: None,  # noqa: ARG001 - fixture ensures the operator is installed
) -> None:
    """Kill one self-healing CNI pod and require topologySpread propagation."""
    cni = await _require_cni_daemonset(kubectl, case_id="D724")
    namespace = _namespace("d724")
    constraint = {
        "maxSkew": 1,
        "topologyKey": "kubernetes.io/hostname",
        "whenUnsatisfiable": "ScheduleAnyway",
        "labelSelector": {"matchLabels": {"d724-spread": "worker"}},
    }

    await _kill_one_cni_pod(kubectl, cni)
    pod = await _apply_beta_manifest_and_wait_for_worker_pod(
        kubectl,
        namespace,
        mutate_worker=lambda component: _set_topology_spread(component, constraint),
    )
    assert constraint in pod["spec"].get("topologySpreadConstraints", [])
    await _wait_for_daemonset_available(kubectl, cni, timeout=180.0)


def _alpha_deployer(kubectl: KubectlClient, namespace: str) -> DynamoDeployer:
    """Return a small v1alpha1 DGD deployer for status-oriented tests."""
    return DynamoDeployer(
        kubectl,
        DynamoConfig(
            namespace=namespace,
            gpu_count=0,
            api_version="v1alpha1",
        ),
    )


async def _status_patch_loop(
    kubectl: KubectlClient,
    namespace: str,
    name: str,
    stop: asyncio.Event,
) -> None:
    """Continuously patch status to force resourceVersion churn during reconcile."""
    saw_status_subresource = False
    patch_num = 0
    while not stop.is_set():
        patch_num += 1
        payload = {
            "status": {
                "state": "pending",
                "d713ExternalPatch": str(patch_num),
            }
        }
        result = await kubectl.run(
            "patch",
            f"{_DGD_KIND}/status",
            name,
            "-n",
            namespace,
            "--type=merge",
            f"-p={orjson.dumps(payload).decode()}",
            check=False,
        )
        if result.returncode == 0:
            saw_status_subresource = True
        elif "not found" not in result.stderr.lower():
            pytest.skip(
                "D713 requires permission to patch the DGD status subresource; "
                f"kubectl returned: {result.stderr.strip() or result.stdout.strip()}"
            )
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except TimeoutError:
            continue
    if not saw_status_subresource:
        pytest.skip("D713 could not patch the DGD status subresource before timeout")


async def _wait_for_non_empty_dgd_state(
    kubectl: KubectlClient,
    *,
    namespace: str,
    name: str,
    timeout: float,
) -> str:
    """Poll until the DGD has any non-empty ``status.state``."""
    deadline = asyncio.get_running_loop().time() + timeout
    last = "<unobserved>"
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            _DGD_KIND,
            name,
            "-n",
            namespace,
            "-o",
            "jsonpath={.status.state}",
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        last = result.stderr.strip() or result.stdout.strip() or "<empty>"
        await asyncio.sleep(2.0)
    raise TimeoutError(
        f"D714: {namespace}/{name} did not receive initial status within "
        f"{timeout}s; last={last!r}"
    )


async def _crd_uid(kubectl: KubectlClient) -> str:
    """Return the DynamoGraphDeployment CRD UID, skipping if it cannot be read."""
    result = await kubectl.run(
        "get",
        "crd",
        _DGD_CRD,
        "-o",
        "jsonpath={.metadata.uid}",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        pytest.skip(
            "D715 requires read access to the DynamoGraphDeployment CRD; "
            f"kubectl returned: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


async def _apply_beta_manifest_and_wait_for_worker_pod(
    kubectl: KubectlClient,
    namespace: str,
    mutate_worker: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Apply one v1beta1 DGD and return a generated worker pod JSON object."""
    await _require_dgd_version(kubectl, "v1beta1")
    deployer = DynamoDeployer(
        kubectl,
        DynamoConfig(namespace=namespace, gpu_count=0, api_version="v1beta1"),
    )
    docs = list(yaml.safe_load_all(deployer.generate_manifest()))
    dgd = docs[1]
    worker = _worker_component(dgd)
    mutate_worker(worker)

    await kubectl.delete_namespace(namespace, wait=True)
    try:
        await kubectl.apply("\n---\n".join(yaml.safe_dump(doc) for doc in docs))
        return await _wait_for_labeled_pod_json(
            kubectl,
            namespace=namespace,
            label_selector=f"{_DGD_LABEL}={deployer._deployment_name()},{_WORKER_LABEL}",
            timeout=_POD_TIMEOUT_S,
        )
    finally:
        await _delete_namespace_fast(kubectl, namespace)


async def _require_dgd_version(kubectl: KubectlClient, version: str) -> None:
    """Skip when the installed DGD CRD does not serve a requested version."""
    result = await kubectl.run("get", "crd", _DGD_CRD, "-o", "json", check=False)
    if result.returncode != 0:
        pytest.skip(
            "D7xx pod-template tests require DGD CRD inspection; "
            f"kubectl returned: {result.stderr.strip() or result.stdout.strip()}"
        )
    crd = orjson.loads(result.stdout or b"{}")
    versions = crd.get("spec", {}).get("versions", [])
    served = [item.get("name") for item in versions if item.get("served")]
    if version not in served:
        pytest.skip(
            f"D7xx pod-template tests require served DGD CRD version {version!r}; "
            f"served versions: {served or '<none>'}"
        )


def _worker_component(dgd: dict[str, Any]) -> dict[str, Any]:
    """Return the first worker component from a v1beta1 DGD manifest."""
    for component in dgd["spec"]["components"]:
        if component.get("type") == "worker":
            return component
    raise AssertionError("D7xx helper could not find worker component in DGD manifest")


def _pod_spec(component: dict[str, Any]) -> dict[str, Any]:
    """Return a mutable component podTemplate.spec dict."""
    return component["podTemplate"]["spec"]


def _component_container(component: dict[str, Any]) -> dict[str, Any]:
    """Return the component's primary container dict."""
    return _pod_spec(component)["containers"][0]


def _set_worker_volume_mount(
    component: dict[str, Any],
    *,
    volume_name: str,
    mount_path: str,
    read_only: bool,
) -> None:
    """Add a cache-like emptyDir volume and read-only mount to a component."""
    pod_spec = _pod_spec(component)
    pod_spec["volumes"] = [{"name": volume_name, "emptyDir": {}}]
    container = _component_container(component)
    container["volumeMounts"] = [
        {"name": volume_name, "mountPath": mount_path, "readOnly": read_only}
    ]


def _set_security_contexts(component: dict[str, Any]) -> None:
    """Set both pod and container security contexts on the component."""
    _pod_spec(component)["securityContext"] = {"runAsNonRoot": True}
    _set_container_security(component, {"runAsUser": 1000, "runAsGroup": 1000})


def _set_container_security(
    component: dict[str, Any],
    security_context: dict[str, Any],
) -> None:
    """Merge container securityContext values into the primary container."""
    container = _component_container(component)
    existing = dict(container.get("securityContext") or {})
    existing.update(security_context)
    container["securityContext"] = existing


def _set_topology_spread(component: dict[str, Any], constraint: dict[str, Any]) -> None:
    """Add a topologySpreadConstraint and matching labels to the component."""
    component["podTemplate"]["metadata"].setdefault("labels", {})["d724-spread"] = (
        "worker"
    )
    _pod_spec(component)["topologySpreadConstraints"] = [constraint]


def _main_container(pod: dict[str, Any]) -> dict[str, Any]:
    """Return the pod's Dynamo main container, falling back to the first container."""
    containers = pod["spec"].get("containers") or []
    for container in containers:
        if container.get("name") == "main":
            return container
    if not containers:
        raise AssertionError("D7xx helper found a pod with no containers")
    return containers[0]


async def _wait_for_labeled_pod_json(
    kubectl: KubectlClient,
    *,
    namespace: str,
    label_selector: str,
    timeout: float,
) -> dict[str, Any]:
    """Poll until a pod matching ``label_selector`` exists and return its JSON."""
    deadline = asyncio.get_running_loop().time() + timeout
    last = "<unobserved>"
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            label_selector,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = orjson.loads(result.stdout)
            items = data.get("items") or []
            if items:
                return items[0]
            last = "<no matching pods>"
        else:
            last = result.stderr.strip() or result.stdout.strip() or "<empty>"
        await asyncio.sleep(2.0)
    raise TimeoutError(
        f"D7xx helper found no pod matching {label_selector!r} in {namespace!r} "
        f"within {timeout}s; last={last!r}"
    )


async def _require_runtime_class(
    kubectl: KubectlClient,
    name: str,
    *,
    case_id: str,
) -> None:
    """Skip unless the requested RuntimeClass exists in the cluster."""
    result = await kubectl.run("get", "runtimeclass", name, check=False)
    if result.returncode != 0:
        pytest.skip(
            f"{case_id} requires RuntimeClass {name!r}; "
            f"kubectl returned: {result.stderr.strip() or result.stdout.strip()}"
        )


async def _require_schedulable_taint(
    kubectl: KubectlClient,
    *,
    case_id: str,
) -> dict[str, str]:
    """Return a toleration for an existing NoSchedule/NoExecute taint, or skip."""
    result = await kubectl.run("get", "nodes", "-o", "json", check=False)
    if result.returncode != 0:
        pytest.skip(
            f"{case_id} requires node-list permission; "
            f"kubectl returned: {result.stderr.strip() or result.stdout.strip()}"
        )
    nodes = orjson.loads(result.stdout or b"{}").get("items", [])
    for node in nodes:
        for taint in node.get("spec", {}).get("taints", []) or []:
            effect = taint.get("effect")
            if effect not in {"NoSchedule", "NoExecute"}:
                continue
            toleration = {
                "key": str(taint.get("key", "")),
                "operator": "Exists",
                "effect": str(effect),
            }
            if toleration["key"]:
                return toleration
    pytest.skip(f"{case_id} requires a cluster node with a NoSchedule/NoExecute taint")


async def _require_cni_daemonset(
    kubectl: KubectlClient,
    *,
    case_id: str,
) -> str:
    """Return a known self-healing CNI DaemonSet name, or skip explicitly."""
    result = await kubectl.run(
        "get", "daemonset", "-n", "kube-system", "-o", "json", check=False
    )
    if result.returncode != 0:
        pytest.skip(
            f"{case_id} requires kube-system DaemonSet list permission; "
            f"kubectl returned: {result.stderr.strip() or result.stdout.strip()}"
        )
    items = orjson.loads(result.stdout or b"{}").get("items", [])
    names = {item.get("metadata", {}).get("name", "") for item in items}
    for candidate in _CNI_DAEMONSET_CANDIDATES:
        if candidate in names:
            return candidate
    pytest.skip(
        f"{case_id} requires a known self-healing CNI DaemonSet; found: "
        f"{', '.join(sorted(names)) or '<none>'}"
    )


async def _kill_one_cni_pod(kubectl: KubectlClient, daemonset: str) -> None:
    """Delete one pod owned by a CNI DaemonSet; the DaemonSet recreates it."""
    pod = await _daemonset_pod_name(kubectl, daemonset)
    result = await kubectl.run(
        "delete",
        "pod",
        pod,
        "-n",
        "kube-system",
        "--wait=false",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            f"D724 requires permission to delete one {daemonset!r} CNI pod; "
            f"kubectl returned: {result.stderr.strip() or result.stdout.strip()}"
        )


async def _daemonset_pod_name(kubectl: KubectlClient, daemonset: str) -> str:
    """Return one pod name owned by ``daemonset`` in kube-system."""
    result = await kubectl.run(
        "get",
        "pods",
        "-n",
        "kube-system",
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            "D724 requires kube-system pod list permission; "
            f"kubectl returned: {result.stderr.strip() or result.stdout.strip()}"
        )
    data = orjson.loads(result.stdout or b"{}")
    for pod in data.get("items", []):
        owners = pod.get("metadata", {}).get("ownerReferences", []) or []
        if any(
            owner.get("kind") == "DaemonSet" and owner.get("name") == daemonset
            for owner in owners
        ):
            return str(pod["metadata"]["name"])
    pytest.skip(f"D724 found CNI DaemonSet {daemonset!r} but no owned pods")


async def _wait_for_daemonset_available(
    kubectl: KubectlClient,
    daemonset: str,
    *,
    timeout: float,
) -> None:
    """Wait until a DaemonSet reports all desired pods available."""
    deadline = asyncio.get_running_loop().time() + timeout
    last = "<unobserved>"
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "daemonset",
            daemonset,
            "-n",
            "kube-system",
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            status = orjson.loads(result.stdout or b"{}").get("status", {})
            desired = int(status.get("desiredNumberScheduled", 0))
            available = int(status.get("numberAvailable", 0))
            last = f"desired={desired} available={available}"
            if desired > 0 and available >= desired:
                return
        else:
            last = result.stderr.strip() or result.stdout.strip() or "<empty>"
        await asyncio.sleep(2.0)
    raise TimeoutError(
        f"D724: CNI DaemonSet {daemonset!r} did not recover within {timeout}s; {last}"
    )


async def _kill_operator_pod(kubectl: KubectlClient) -> None:
    """Force-delete Dynamo operator pods and let the Deployment recreate them."""
    result = await kubectl.run(
        "delete",
        "pod",
        "-l",
        _OPERATOR_SELECTOR,
        "-n",
        _OPERATOR_NAMESPACE,
        "--force",
        "--grace-period=0",
        "--ignore-not-found",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            "D714 requires permission to delete the Dynamo operator pod; "
            f"kubectl returned: {result.stderr.strip() or result.stdout.strip()}"
        )


async def _wait_for_operator_available(
    kubectl: KubectlClient, *, timeout: float
) -> None:
    """Wait until the Dynamo operator Deployment is Available again."""
    deadline = asyncio.get_running_loop().time() + timeout
    last = "<unobserved>"
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "deployment",
            "-n",
            _OPERATOR_NAMESPACE,
            "-l",
            _OPERATOR_SELECTOR,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            items = orjson.loads(result.stdout or b"{}").get("items", [])
            if items:
                status = items[0].get("status", {})
                desired = int(status.get("replicas", 0))
                available = int(status.get("availableReplicas", 0))
                last = f"desired={desired} available={available}"
                if desired > 0 and available >= desired:
                    return
        else:
            last = result.stderr.strip() or result.stdout.strip() or "<empty>"
        await asyncio.sleep(2.0)
    raise TimeoutError(
        f"Dynamo operator did not become Available within {timeout}s; {last}"
    )


async def _delete_namespace_fast(kubectl: KubectlClient, namespace: str) -> None:
    """Delete a test namespace without blocking the next collected test."""
    await kubectl.run(
        "delete",
        "namespace",
        namespace,
        "--wait=false",
        "--ignore-not-found",
        check=False,
    )
