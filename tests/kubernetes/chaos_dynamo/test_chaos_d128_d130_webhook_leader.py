# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D128-D130 -- Dynamo webhook and leader-election control-plane faults."""

from __future__ import annotations

import asyncio
import base64
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import orjson
import pytest

from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_WEBHOOK_KIND = "validatingwebhookconfiguration"
_WEBHOOK_GROUP = "nvidia.com"
_DGD_RESOURCE = "dynamographdeployments"
_BAD_CA_BUNDLE = base64.b64encode(b"d128-not-a-serving-ca").decode()
_D128_NAMESPACE = "d128-webhook-bad-ca"
_D129_NAMESPACE = "d129-webhook-timeout-budget"
_APPLY_TIMEOUT_BUDGET_S = 20.0
_SUCCESS_TIMEOUT_S = 600.0


@dataclass(frozen=True, slots=True)
class _WebhookTarget:
    """Exact validating webhook entry that admits DynamoGraphDeployments."""

    config_name: str
    webhook_index: int
    webhook_name: str
    ca_bundle: str | None
    timeout_seconds: int | None
    service_namespace: str
    service_name: str


@dataclass(frozen=True, slots=True)
class _DeploymentTarget:
    """Deployment backing the DGD admission webhook Service."""

    namespace: str
    name: str
    replicas: int


@dataclass(frozen=True, slots=True)
class _LeaseTarget:
    """Leader-election Lease selected for D130."""

    namespace: str
    name: str
    uid: str
    holder_identity: str


async def test_d128_webhook_bad_ca_bundle_rejects_dgd_admission(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures webhook/CRD are installed
) -> None:
    """Patch the DGD validating webhook CA bundle and require admission failure."""
    target = await _unique_dgd_webhook(kubectl, case="D128")
    if target.ca_bundle is None:
        pytest.skip(
            f"D128 requires webhook {target.config_name}/{target.webhook_name} to "
            "declare clientConfig.caBundle so it can be restored exactly"
        )

    await kubectl.create_namespace(_D128_NAMESPACE)
    try:
        await _patch_webhook_field(
            kubectl,
            target,
            field="caBundle",
            value=_BAD_CA_BUNDLE,
        )
        manifest = _dgd_manifest(kubectl, namespace=_D128_NAMESPACE)
        result = await _apply_manifest_result(
            kubectl, manifest, namespace=_D128_NAMESPACE
        )
        assert result.returncode != 0, (
            "D128: DGD admission succeeded while the validating webhook caBundle "
            f"was patched to an invalid CA on {target.config_name}/"
            f"{target.webhook_name}; stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        message = f"{result.stdout}\n{result.stderr}".lower()
        assert any(
            term in message
            for term in (
                "certificate",
                "x509",
                "tls",
                "ca",
                "webhook",
                "unknown authority",
            )
        ), (
            "D128: DGD admission failed after CA-bundle mismatch, but the error "
            f"did not identify webhook/TLS/CA validation: {message!r}"
        )
    finally:
        await _patch_webhook_field(
            kubectl,
            target,
            field="caBundle",
            value=target.ca_bundle,
        )
        await kubectl.run(
            "delete",
            "namespace",
            _D128_NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def test_d129_webhook_timeout_budget_is_bounded(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures webhook/CRD are installed
) -> None:
    """Set webhook timeoutSeconds=1, remove endpoints, and assert fast failure."""
    webhook = await _unique_dgd_webhook(kubectl, case="D129")
    deployment = await _unique_webhook_deployment(kubectl, webhook, case="D129")

    await kubectl.create_namespace(_D129_NAMESPACE)
    started = time.monotonic()
    try:
        await _patch_webhook_field(kubectl, webhook, field="timeoutSeconds", value=1)
        await _scale_deployment(kubectl, deployment, replicas=0)
        await _wait_deployment_available_replicas(
            kubectl,
            deployment,
            replicas=0,
            timeout=90.0,
        )

        manifest = _dgd_manifest(kubectl, namespace=_D129_NAMESPACE)
        result = await _apply_manifest_result(
            kubectl, manifest, namespace=_D129_NAMESPACE
        )
        elapsed = time.monotonic() - started

        assert result.returncode != 0, (
            "D129: DGD admission succeeded while webhook timeoutSeconds=1 and "
            f"backing deployment {deployment.namespace}/{deployment.name} had zero replicas"
        )
        assert elapsed < _APPLY_TIMEOUT_BUDGET_S, (
            f"D129: webhook admission failure exceeded timeout budget; elapsed={elapsed:.2f}s "
            f"budget={_APPLY_TIMEOUT_BUDGET_S:.2f}s stderr={result.stderr!r}"
        )
        message = f"{result.stdout}\n{result.stderr}".lower()
        assert any(
            term in message
            for term in (
                "timeout",
                "context deadline exceeded",
                "no endpoints available",
                "connection refused",
                "webhook",
            )
        ), (
            "D129: DGD admission failed under webhook timeout fault, but the "
            f"message did not identify timeout/webhook unavailability: {message!r}"
        )
    finally:
        if webhook.timeout_seconds is None:
            await _remove_webhook_field(kubectl, webhook, field="timeoutSeconds")
        else:
            await _patch_webhook_field(
                kubectl,
                webhook,
                field="timeoutSeconds",
                value=webhook.timeout_seconds,
            )
        await _scale_deployment(kubectl, deployment, replicas=deployment.replicas)
        if deployment.replicas > 0:
            await kubectl.run(
                "rollout",
                "status",
                "deployment",
                deployment.name,
                "-n",
                deployment.namespace,
                "--timeout=180s",
                check=False,
            )
        await kubectl.run(
            "delete",
            "namespace",
            _D129_NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def test_d130_leader_election_lease_disruption_recovers(
    kubectl: KubectlClient,
    dynamo_server,  # noqa: ANN001 - fixture provides a live reconcile target
    dynamo_deployment_namespace: str,
) -> None:
    """Delete the Dynamo operator leader-election Lease and assert reacquisition."""
    lease = await _unique_dynamo_leader_lease(kubectl)
    dgd_name = await _unique_dgd_name(kubectl, dynamo_deployment_namespace, case="D130")

    await kubectl.run(
        "delete",
        "lease",
        lease.name,
        "-n",
        lease.namespace,
        "--wait=false",
        check=True,
    )
    recreated = await _wait_for_lease_recreated(kubectl, lease, timeout=90.0)
    await _wait_operator_available(kubectl, timeout=180.0)
    await wait_for_dgd_state(
        kubectl,
        dgd_name,
        dynamo_deployment_namespace,
        "successful",
        timeout=_SUCCESS_TIMEOUT_S,
    )

    assert recreated.uid != lease.uid, (
        f"D130: leader-election Lease {lease.namespace}/{lease.name} retained old "
        f"uid={lease.uid!r} after deletion; disruption did not recreate/reacquire it"
    )
    assert recreated.holder_identity, (
        f"D130: leader-election Lease {lease.namespace}/{lease.name} was recreated "
        "without spec.holderIdentity"
    )


async def _unique_dgd_webhook(kubectl: KubectlClient, *, case: str) -> _WebhookTarget:
    result = await kubectl.run("get", _WEBHOOK_KIND, "-o", "json", check=False)
    if result.returncode != 0 or not result.stdout.strip():
        pytest.skip(
            f"{case} requires permission to inspect validating webhook configs; "
            f"stderr={result.stderr.strip()!r}"
        )

    candidates: list[_WebhookTarget] = []
    for item in orjson.loads(result.stdout).get("items", []):
        config_name = item.get("metadata", {}).get("name", "")
        for index, webhook in enumerate(item.get("webhooks") or []):
            if not _webhook_validates_dgd(webhook.get("rules") or []):
                continue
            client_config = webhook.get("clientConfig") or {}
            service = client_config.get("service") or {}
            service_namespace = service.get("namespace", "")
            service_name = service.get("name", "")
            if not service_namespace or not service_name:
                continue
            candidates.append(
                _WebhookTarget(
                    config_name=config_name,
                    webhook_index=index,
                    webhook_name=webhook.get("name", f"webhooks[{index}]"),
                    ca_bundle=client_config.get("caBundle"),
                    timeout_seconds=webhook.get("timeoutSeconds"),
                    service_namespace=service_namespace,
                    service_name=service_name,
                )
            )

    if len(candidates) != 1:
        labels = [f"{c.config_name}/{c.webhook_name}" for c in candidates]
        pytest.skip(
            f"{case} requires exactly one DGD validating webhook target; "
            f"found {labels if labels else '<none>'}"
        )
    return candidates[0]


def _webhook_validates_dgd(rules: list[dict[str, Any]]) -> bool:
    for rule in rules:
        groups = rule.get("apiGroups") or []
        resources = rule.get("resources") or []
        if _WEBHOOK_GROUP in groups and any(
            str(resource).startswith(_DGD_RESOURCE) for resource in resources
        ):
            return True
    return False


async def _unique_webhook_deployment(
    kubectl: KubectlClient,
    webhook: _WebhookTarget,
    *,
    case: str,
) -> _DeploymentTarget:
    service = await _get_json(
        kubectl,
        "service",
        webhook.service_name,
        namespace=webhook.service_namespace,
        check=False,
    )
    if not service:
        pytest.skip(
            f"{case} requires webhook Service {webhook.service_namespace}/"
            f"{webhook.service_name} to exist"
        )
    selector = service.get("spec", {}).get("selector") or {}
    if not selector:
        pytest.skip(
            f"{case} requires webhook Service {webhook.service_namespace}/"
            f"{webhook.service_name} to have a non-empty selector"
        )

    deployments = await _get_json(
        kubectl,
        "deployment",
        namespace=webhook.service_namespace,
        check=True,
    )
    matches: list[_DeploymentTarget] = []
    for item in deployments.get("items", []):
        labels = (
            item.get("spec", {})
            .get("template", {})
            .get("metadata", {})
            .get(
                "labels",
                {},
            )
        )
        if all(labels.get(key) == value for key, value in selector.items()):
            matches.append(
                _DeploymentTarget(
                    namespace=webhook.service_namespace,
                    name=item.get("metadata", {}).get("name", ""),
                    replicas=int(item.get("spec", {}).get("replicas") or 0),
                )
            )
    if len(matches) != 1:
        labels = [f"{item.namespace}/{item.name}" for item in matches]
        pytest.skip(
            f"{case} requires exactly one Deployment backing webhook Service "
            f"{webhook.service_namespace}/{webhook.service_name}; "
            f"selector={selector!r}, matches={labels if labels else '<none>'}"
        )
    return matches[0]


async def _patch_webhook_field(
    kubectl: KubectlClient,
    webhook: _WebhookTarget,
    *,
    field: str,
    value: str | int,
) -> None:
    patch = [
        {
            "op": "replace",
            "path": f"/webhooks/{webhook.webhook_index}/{_json_pointer_escape(field_path(field))}",
            "value": value,
        }
    ]
    await kubectl.run(
        "patch",
        _WEBHOOK_KIND,
        webhook.config_name,
        "--type=json",
        f"-p={orjson.dumps(patch).decode()}",
        check=True,
    )


async def _remove_webhook_field(
    kubectl: KubectlClient,
    webhook: _WebhookTarget,
    *,
    field: str,
) -> None:
    patch = [
        {
            "op": "remove",
            "path": f"/webhooks/{webhook.webhook_index}/{_json_pointer_escape(field_path(field))}",
        }
    ]
    await kubectl.run(
        "patch",
        _WEBHOOK_KIND,
        webhook.config_name,
        "--type=json",
        f"-p={orjson.dumps(patch).decode()}",
        check=False,
    )


def field_path(field: str) -> str:
    if field == "caBundle":
        return "clientConfig/caBundle"
    return field


def _json_pointer_escape(path: str) -> str:
    return "/".join(
        part.replace("~", "~0").replace("/", "~1") for part in path.split("/")
    )


async def _apply_manifest_result(
    kubectl: KubectlClient,
    manifest: str,
    *,
    namespace: str,
) -> subprocess.CompletedProcess[str]:
    cmd = ["kubectl"]
    if kubectl.context:
        cmd.extend(["--context", kubectl.context])
    if kubectl.kubeconfig:
        cmd.extend(["--kubeconfig", kubectl.kubeconfig])
    cmd.extend(["apply", "-n", namespace, "-f", "-"])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=manifest.encode())
    return subprocess.CompletedProcess(
        cmd,
        proc.returncode,
        stdout.decode() if stdout else "",
        stderr.decode() if stderr else "",
    )


def _dgd_manifest(kubectl: KubectlClient, *, namespace: str) -> str:
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=namespace,
        api_version="v1alpha1",
    )
    return DynamoDeployer(kubectl, config).generate_manifest()


async def _scale_deployment(
    kubectl: KubectlClient,
    target: _DeploymentTarget,
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


async def _wait_deployment_available_replicas(
    kubectl: KubectlClient,
    target: _DeploymentTarget,
    *,
    replicas: int,
    timeout: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        deployment = await _get_json(
            kubectl,
            "deployment",
            target.name,
            namespace=target.namespace,
            check=False,
        )
        status = deployment.get("status", {})
        if int(status.get("availableReplicas") or 0) == replicas:
            return
        await asyncio.sleep(1.0)
    pytest.fail(
        f"D129: webhook deployment {target.namespace}/{target.name} did not reach "
        f"availableReplicas={replicas} within {timeout}s"
    )


async def _unique_dynamo_leader_lease(kubectl: KubectlClient) -> _LeaseTarget:
    operator_pods = await _operator_pod_names(kubectl)
    if not operator_pods:
        pytest.skip(
            "D130 requires at least one Dynamo operator pod matching selector "
            f"{_OPERATOR_SELECTOR!r} in namespace {_OPERATOR_NAMESPACE!r}"
        )

    result = await kubectl.run(
        "get",
        "lease",
        "-n",
        _OPERATOR_NAMESPACE,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        pytest.skip(
            "D130 requires coordination.k8s.io Lease list access in the Dynamo "
            f"operator namespace; stderr={result.stderr.strip()!r}"
        )

    candidates: list[_LeaseTarget] = []
    for item in orjson.loads(result.stdout).get("items", []):
        metadata = item.get("metadata", {})
        spec = item.get("spec", {})
        name = metadata.get("name", "")
        holder = spec.get("holderIdentity", "")
        holder_matches_operator = any(pod in holder for pod in operator_pods)
        name_matches_operator = "dynamo" in name and "leader" in name
        if holder_matches_operator or name_matches_operator:
            candidates.append(
                _LeaseTarget(
                    namespace=_OPERATOR_NAMESPACE,
                    name=name,
                    uid=metadata.get("uid", ""),
                    holder_identity=holder,
                )
            )

    if len(candidates) != 1:
        labels = [f"{lease.namespace}/{lease.name}" for lease in candidates]
        pytest.skip(
            "D130 requires exactly one Dynamo operator leader-election Lease; "
            f"found {labels if labels else '<none>'}"
        )
    return candidates[0]


async def _operator_pod_names(kubectl: KubectlClient) -> list[str]:
    result = await kubectl.run(
        "get",
        "pod",
        "-n",
        _OPERATOR_NAMESPACE,
        "-l",
        _OPERATOR_SELECTOR,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return [
        item.get("metadata", {}).get("name", "")
        for item in orjson.loads(result.stdout).get("items", [])
        if item.get("metadata", {}).get("name")
    ]


async def _wait_for_lease_recreated(
    kubectl: KubectlClient,
    old: _LeaseTarget,
    *,
    timeout: float,
) -> _LeaseTarget:
    deadline = asyncio.get_running_loop().time() + timeout
    last_uid = "<unobserved>"
    while True:
        lease = await _get_json(
            kubectl,
            "lease",
            old.name,
            namespace=old.namespace,
            check=False,
        )
        metadata = lease.get("metadata", {})
        spec = lease.get("spec", {})
        uid = metadata.get("uid", "")
        if uid:
            last_uid = uid
        if uid and uid != old.uid:
            return _LeaseTarget(
                namespace=old.namespace,
                name=old.name,
                uid=uid,
                holder_identity=spec.get("holderIdentity", ""),
            )
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"D130: leader-election Lease {old.namespace}/{old.name} was not "
                f"recreated within {timeout}s; old_uid={old.uid!r}, last_uid={last_uid!r}"
            )
        await asyncio.sleep(1.0)


async def _unique_dgd_name(
    kubectl: KubectlClient,
    namespace: str,
    *,
    case: str,
) -> str:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            f"{case} requires list/get access to DGD resources in namespace {namespace!r}; "
            f"stderr={result.stderr.strip()!r}"
        )
    items = orjson.loads(result.stdout or b"{}").get("items", [])
    if len(items) != 1:
        names = [item.get("metadata", {}).get("name", "<unnamed>") for item in items]
        pytest.skip(
            f"{case} requires exactly one DGD in namespace {namespace!r}; "
            f"found {names if names else '<none>'}"
        )
    return items[0]["metadata"]["name"]


async def _wait_operator_available(kubectl: KubectlClient, *, timeout: float) -> None:
    result = await kubectl.run(
        "wait",
        "-n",
        _OPERATOR_NAMESPACE,
        "-l",
        _OPERATOR_SELECTOR,
        "deployment",
        "--for=condition=Available",
        f"--timeout={int(timeout)}s",
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "D130: Dynamo operator deployment did not become Available after "
            f"leader-election Lease disruption; stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )


async def _get_json(
    kubectl: KubectlClient,
    resource: str,
    name: str | None = None,
    *,
    namespace: str,
    check: bool,
) -> dict[str, Any]:
    args = ["get", resource]
    if name is not None:
        args.append(name)
    args.extend(["-n", namespace, "-o", "json"])
    result = await kubectl.run(*args, check=check)
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    loaded = orjson.loads(result.stdout)
    if not isinstance(loaded, dict):
        raise TypeError(f"expected {resource} JSON object, got {type(loaded)!r}")
    return loaded
