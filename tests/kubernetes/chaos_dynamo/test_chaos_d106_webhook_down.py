# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D106 -- webhook outage fails DGD admission or prevents child creation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import orjson
import pytest

from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


_DGD_NAME = "d106-test"
_DGD_NAMESPACE = "d106-webhook-down"
_WEBHOOK_KIND = "validatingwebhookconfigurations"
_WEBHOOK_GROUP = "nvidia.com"
_DGD_RESOURCE = "dynamographdeployment"
_DGD_LABELS = (
    "nvidia.com/dynamo-graph-deployment-name",
    "nvidia.com/dynamographdeployment",
)
_CHILD_KINDS = (
    "deployment",
    "service",
    "configmap",
    "role",
    "rolebinding",
    "serviceaccount",
    "pod",
)
_NO_CHILD_GRACE_S = 10.0


@dataclass(frozen=True)
class WebhookDeployment:
    """Kubernetes deployment backing the DGD validating webhook service."""

    namespace: str
    name: str
    replicas: int
    selector: dict[str, str]
    candidates: tuple[str, ...]


async def test_d106_webhook_down_blocks_dgd_or_leaves_no_children(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture installs the DGD CRD/webhook
) -> None:
    """Scale the DGD webhook deployment to zero and assert no partial reconcile.

    The D106 contract accepts either failure mode while the webhook is down:
    apiserver admission can fail fast with a webhook availability error, or the
    CR can be admitted while the controller/webhook deployment is unavailable.
    In the admitted case, the test asserts no child resources are created before
    restoring the deployment.
    """
    target = await _find_dgd_webhook_deployment(kubectl)
    if target is None:
        pytest.skip("D106: no DGD validating webhook service was found")

    await kubectl.create_namespace(_DGD_NAMESPACE)
    apply_succeeded = False
    try:
        await _scale_deployment(kubectl, target, replicas=0)
        await _wait_for_deployment_replicas(kubectl, target, replicas=0, timeout=60.0)

        manifest = _dgd_manifest(kubectl)
        try:
            await kubectl.apply(manifest, namespace=_DGD_NAMESPACE)
        except RuntimeError as exc:
            message = str(exc).lower()
            assert any(
                term in message
                for term in (
                    "webhook",
                    "validating",
                    "admission",
                    "connection refused",
                    "no endpoints available",
                    "service unavailable",
                    "context deadline exceeded",
                    "timeout",
                )
            ), (
                "D106: DGD apply failed while webhook deployment was scaled "
                "down, but the error did not name admission/webhook "
                f"unavailability: {exc!s}"
            )
        else:
            apply_succeeded = True
            await asyncio.sleep(_NO_CHILD_GRACE_S)
            children = await _list_dgd_children(
                kubectl,
                namespace=_DGD_NAMESPACE,
                name=_DGD_NAME,
            )
            assert not children, (
                "D106: DGD was admitted while webhook deployment "
                f"{target.namespace}/{target.name} was scaled to zero, but "
                f"child resources were created before restore: {children}"
            )
    finally:
        await _scale_deployment(kubectl, target, replicas=target.replicas)
        if target.replicas > 0:
            await kubectl.run(
                "rollout",
                "status",
                "deployment",
                target.name,
                "-n",
                target.namespace,
                "--timeout=120s",
                check=False,
            )
        if apply_succeeded:
            await kubectl.run(
                "delete",
                _DGD_RESOURCE,
                _DGD_NAME,
                "-n",
                _DGD_NAMESPACE,
                "--wait=false",
                "--ignore-not-found",
                check=False,
            )
        await kubectl.run(
            "delete",
            "namespace",
            _DGD_NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _find_dgd_webhook_deployment(
    kubectl: KubectlClient,
) -> WebhookDeployment | None:
    """Find the single Deployment selected by the DGD webhook Service.

    Returns ``None`` only when no DGD webhook exists. Ambiguous or selectorless
    webhook services skip with inspected candidates instead of guessing which
    deployment is safe to scale down.
    """
    webhook_services = await _dgd_webhook_services(kubectl)
    if not webhook_services:
        return None

    candidates: list[WebhookDeployment] = []
    inspected: list[str] = []
    for namespace, service_name in webhook_services:
        service = await _get_json(
            kubectl,
            "service",
            service_name,
            namespace=namespace,
            check=False,
        )
        selector = service.get("spec", {}).get("selector") or {}
        if not selector:
            inspected.append(f"service/{namespace}/{service_name}: selector=<empty>")
            continue

        deployments = await _deployments_matching_selector(
            kubectl,
            namespace=namespace,
            selector=selector,
        )
        inspected.append(
            f"service/{namespace}/{service_name}: selector={selector!r} "
            f"deployments={deployments!r}"
        )
        candidates.extend(deployments)

    unique = {
        (candidate.namespace, candidate.name): candidate for candidate in candidates
    }
    if len(unique) != 1:
        pytest.skip(
            "D106: DGD webhook deployment could not be uniquely identified; "
            f"inspected candidates: {inspected!r}"
        )
    target = next(iter(unique.values()))
    return WebhookDeployment(
        namespace=target.namespace,
        name=target.name,
        replicas=target.replicas,
        selector=target.selector,
        candidates=tuple(inspected),
    )


async def _dgd_webhook_services(kubectl: KubectlClient) -> list[tuple[str, str]]:
    result = await kubectl.run("get", _WEBHOOK_KIND, "-o", "json", check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return []

    data = orjson.loads(result.stdout)
    services: list[tuple[str, str]] = []
    for item in data.get("items", []):
        for webhook in item.get("webhooks", []):
            rules = webhook.get("rules", []) or []
            if not _webhook_validates_dgd(rules):
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
        if _WEBHOOK_GROUP in groups and any(
            str(resource).startswith("dynamographdeployments") for resource in resources
        ):
            return True
    return False


async def _deployments_matching_selector(
    kubectl: KubectlClient,
    *,
    namespace: str,
    selector: dict[str, str],
) -> list[WebhookDeployment]:
    data = await _get_json(kubectl, "deployment", namespace=namespace, check=True)
    matches: list[WebhookDeployment] = []
    for item in data.get("items", []):
        labels = (
            item.get("spec", {})
            .get("template", {})
            .get("metadata", {})
            .get("labels", {})
        )
        if all(labels.get(key) == value for key, value in selector.items()):
            spec = item.get("spec", {})
            matches.append(
                WebhookDeployment(
                    namespace=namespace,
                    name=item.get("metadata", {}).get("name", ""),
                    replicas=int(spec.get("replicas") or 0),
                    selector=selector,
                    candidates=(),
                )
            )
    return matches


async def _scale_deployment(
    kubectl: KubectlClient,
    target: WebhookDeployment,
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


async def _wait_for_deployment_replicas(
    kubectl: KubectlClient,
    target: WebhookDeployment,
    *,
    replicas: int,
    timeout: float,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "deployment",
            target.name,
            "-n",
            target.namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            deployment = orjson.loads(result.stdout)
            status = deployment.get("status", {})
            ready = int(status.get("readyReplicas") or 0)
            available = int(status.get("availableReplicas") or 0)
            updated = int(status.get("updatedReplicas") or 0)
            if ready == replicas and available == replicas and updated == replicas:
                return
        await asyncio.sleep(1.0)
    pytest.fail(
        "D106: timed out waiting for webhook deployment "
        f"{target.namespace}/{target.name} to reach replicas={replicas}"
    )


def _dgd_manifest(kubectl: KubectlClient) -> str:
    config = DynamoConfig(
        name=_DGD_NAME,
        model_name="Qwen/Qwen3-0.6B",
        namespace=_DGD_NAMESPACE,
        api_version="v1alpha1",
    )
    return DynamoDeployer(kubectl, config).generate_manifest()


async def _list_dgd_children(
    kubectl: KubectlClient,
    *,
    namespace: str,
    name: str,
) -> list[str]:
    children: list[str] = []
    for kind in _CHILD_KINDS:
        for label in _DGD_LABELS:
            result = await kubectl.run(
                "get",
                kind,
                "-n",
                namespace,
                "-l",
                f"{label}={name}",
                "-o",
                "name",
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                children.extend(
                    f"{kind}/{line.strip().split('/', 1)[-1]}"
                    for line in result.stdout.splitlines()
                    if line.strip()
                )
    return sorted(set(children))


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
    return orjson.loads(result.stdout)
