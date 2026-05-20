# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D118 -- DGD validating webhook fails closed when unavailable."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import orjson
import pytest

from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_NAMESPACE = "d118-webhook-fail-closed"
_WEBHOOK_GROUP = "nvidia.com"
_DGD_RESOURCE = "dynamographdeployment"
_DGD_NAME = "dynamo-agg"


@dataclass(frozen=True)
class WebhookTarget:
    """Validating webhook configuration plus the Deployment behind its service."""

    config_name: str
    webhook_name: str
    failure_policy: str | None
    deployment_namespace: str
    deployment_name: str
    deployment_replicas: int


async def test_d118_webhook_unavailable_with_fail_policy_fail_rejects_dgd(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture installs DGD webhook
) -> None:
    """Set failurePolicy=Fail, stop webhook pods, and assert admission rejects."""
    target = await _find_webhook_target(kubectl, case_id="D118")
    await kubectl.create_namespace(_NAMESPACE)
    try:
        await _patch_failure_policy(kubectl, target, policy="Fail")
        await _scale_deployment(kubectl, target, replicas=0)
        await _wait_deployment_replicas(kubectl, target, replicas=0, timeout=90.0)

        manifest = DynamoDeployer(
            kubectl,
            DynamoConfig(namespace=_NAMESPACE, api_version="v1alpha1"),
        ).generate_manifest()
        try:
            await kubectl.apply(manifest, namespace=_NAMESPACE)
        except RuntimeError as exc:
            message = str(exc).lower()
        else:
            pytest.fail(
                "D118 expected DGD admission to fail while failurePolicy=Fail "
                "and the validating webhook Deployment was scaled to zero"
            )

        assert any(
            term in message
            for term in (
                "webhook",
                "validating",
                "admission",
                "no endpoints available",
                "connection refused",
                "service unavailable",
                "context deadline exceeded",
                "timeout",
            )
        ), f"D118 expected webhook availability error, got {message!r}"
        get_result = await kubectl.run(
            "get",
            _DGD_RESOURCE,
            _DGD_NAME,
            "-n",
            _NAMESPACE,
            check=False,
        )
        assert get_result.returncode != 0, (
            "D118: fail-closed admission still created DGD"
        )
    finally:
        await _scale_deployment(kubectl, target, replicas=target.deployment_replicas)
        if target.deployment_replicas > 0:
            await _wait_deployment_replicas(
                kubectl,
                target,
                replicas=target.deployment_replicas,
                timeout=180.0,
            )
        await _restore_failure_policy(kubectl, target)
        await kubectl.run(
            "delete",
            "namespace",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _find_webhook_target(
    kubectl: KubectlClient, *, case_id: str
) -> WebhookTarget:
    result = await kubectl.run(
        "get",
        "validatingwebhookconfigurations",
        "-o",
        "json",
        check=True,
    )
    candidates: list[WebhookTarget] = []
    inspected: list[str] = []
    for config in orjson.loads(result.stdout).get("items", []):
        config_name = config.get("metadata", {}).get("name", "")
        for webhook in config.get("webhooks", []) or []:
            if not _webhook_validates_dgd(webhook.get("rules", []) or []):
                continue
            service = webhook.get("clientConfig", {}).get("service") or {}
            namespace = service.get("namespace")
            name = service.get("name")
            if not namespace or not name:
                inspected.append(f"{config_name}/{webhook.get('name')}: no service")
                continue
            deployment = await _deployment_for_service(kubectl, namespace, name)
            inspected.append(
                f"{config_name}/{webhook.get('name')} -> {namespace}/{name}"
            )
            if deployment is not None:
                candidates.append(
                    WebhookTarget(
                        config_name=config_name,
                        webhook_name=webhook.get("name", ""),
                        failure_policy=webhook.get("failurePolicy"),
                        deployment_namespace=namespace,
                        deployment_name=deployment["name"],
                        deployment_replicas=deployment["replicas"],
                    )
                )
    if len(candidates) != 1:
        pytest.skip(
            f"{case_id} requires exactly one DGD validating webhook Deployment; "
            f"inspected={inspected!r}"
        )
    return candidates[0]


def _webhook_validates_dgd(rules: list[dict[str, Any]]) -> bool:
    for rule in rules:
        groups = rule.get("apiGroups") or []
        resources = rule.get("resources") or []
        if _WEBHOOK_GROUP in groups and any(
            str(resource).startswith("dynamographdeployments") for resource in resources
        ):
            return True
    return False


async def _deployment_for_service(
    kubectl: KubectlClient,
    namespace: str,
    service_name: str,
) -> dict[str, Any] | None:
    service = await _get_json(kubectl, "service", service_name, namespace=namespace)
    selector = service.get("spec", {}).get("selector") or {}
    if not selector:
        return None
    deployments = await _get_json(kubectl, "deployment", namespace=namespace)
    matches = []
    for item in deployments.get("items", []):
        labels = (
            item.get("spec", {})
            .get("template", {})
            .get("metadata", {})
            .get("labels", {})
        )
        if all(labels.get(key) == value for key, value in selector.items()):
            matches.append(
                {
                    "name": item.get("metadata", {}).get("name", ""),
                    "replicas": int(item.get("spec", {}).get("replicas") or 0),
                }
            )
    if len(matches) != 1:
        return None
    return matches[0]


async def _patch_failure_policy(
    kubectl: KubectlClient,
    target: WebhookTarget,
    *,
    policy: str,
) -> None:
    patch = [
        {
            "op": "add" if target.failure_policy is None else "replace",
            "path": f"/webhooks/{await _webhook_index(kubectl, target)}/failurePolicy",
            "value": policy,
        }
    ]
    await kubectl.run(
        "patch",
        "validatingwebhookconfiguration",
        target.config_name,
        "--type=json",
        f"-p={orjson.dumps(patch).decode()}",
        check=True,
    )


async def _restore_failure_policy(
    kubectl: KubectlClient, target: WebhookTarget
) -> None:
    if target.failure_policy is None:
        patch = [
            {
                "op": "remove",
                "path": f"/webhooks/{await _webhook_index(kubectl, target)}/failurePolicy",
            }
        ]
    else:
        patch = [
            {
                "op": "replace",
                "path": f"/webhooks/{await _webhook_index(kubectl, target)}/failurePolicy",
                "value": target.failure_policy,
            }
        ]
    await kubectl.run(
        "patch",
        "validatingwebhookconfiguration",
        target.config_name,
        "--type=json",
        f"-p={orjson.dumps(patch).decode()}",
        check=False,
    )


async def _webhook_index(kubectl: KubectlClient, target: WebhookTarget) -> int:
    data = await _get_json(
        kubectl, "validatingwebhookconfiguration", target.config_name
    )
    for index, webhook in enumerate(data.get("webhooks", []) or []):
        if webhook.get("name") == target.webhook_name:
            return index
    raise AssertionError(f"D118: webhook {target.webhook_name!r} disappeared")


async def _scale_deployment(
    kubectl: KubectlClient,
    target: WebhookTarget,
    *,
    replicas: int,
) -> None:
    await kubectl.run(
        "scale",
        "deployment",
        target.deployment_name,
        "-n",
        target.deployment_namespace,
        f"--replicas={replicas}",
        check=True,
    )


async def _wait_deployment_replicas(
    kubectl: KubectlClient,
    target: WebhookTarget,
    *,
    replicas: int,
    timeout: float,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        deployment = await _get_json(
            kubectl,
            "deployment",
            target.deployment_name,
            namespace=target.deployment_namespace,
        )
        status = deployment.get("status", {})
        ready = int(status.get("readyReplicas") or 0)
        available = int(status.get("availableReplicas") or 0)
        updated = int(status.get("updatedReplicas") or 0)
        if ready == replicas and available == replicas and updated == replicas:
            return
        await asyncio.sleep(1.0)
    pytest.fail(
        f"D118: webhook deployment {target.deployment_namespace}/"
        f"{target.deployment_name} did not reach replicas={replicas}"
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
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    return orjson.loads(result.stdout)
