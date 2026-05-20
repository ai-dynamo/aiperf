# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D119 -- webhook failurePolicy=Ignore admits only schema-valid invalid specs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import orjson
import pytest

from dev.versions import DYNAMO_VERSION
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_NAMESPACE = "d119-webhook-fail-open"
_NAME = "d119-invalid"
_WEBHOOK_GROUP = "nvidia.com"


@dataclass(frozen=True)
class WebhookTarget:
    """Validating webhook configuration plus the Deployment behind its service."""

    config_name: str
    webhook_name: str
    failure_policy: str | None
    deployment_namespace: str
    deployment_name: str
    deployment_replicas: int


async def test_d119_fail_open_invalid_spec_is_admitted_but_not_successful(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture installs DGD webhook
) -> None:
    """Set failurePolicy=Ignore and assert a webhook-only invalid spec is not run."""
    target = await _find_webhook_target(kubectl)
    await kubectl.create_namespace(_NAMESPACE)
    try:
        await _patch_failure_policy(kubectl, target, policy="Ignore")
        await _scale_deployment(kubectl, target, replicas=0)
        await _wait_deployment_replicas(kubectl, target, replicas=0, timeout=90.0)

        try:
            await kubectl.apply(
                orjson.dumps(_invalid_but_schema_valid_manifest()).decode(),
                namespace=_NAMESPACE,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "strict decoding" in message or "unknown field" in message:
                pytest.skip(
                    "D119 requires an invalid DGD shape that is accepted by the CRD "
                    "schema when the webhook is fail-open; this cluster rejects the "
                    f"probe at schema admission: {message!r}"
                )
            pytest.fail(f"D119 fail-open apply failed unexpectedly: {message!r}")

        dgd = await _read_dgd(kubectl)
        assert dgd["metadata"]["name"] == _NAME
        state = await _observe_state(kubectl, timeout=45.0)
        assert state != "successful", (
            "D119: fail-open admitted a webhook-invalid DGD and the operator "
            "reported state='successful' instead of surfacing validation failure"
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


def _invalid_but_schema_valid_manifest() -> dict[str, Any]:
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": _NAME, "namespace": _NAMESPACE},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": f"nvcr.io/nvidia/ai-dynamo/vllm-runtime:{DYNAMO_VERSION}",
                        }
                    },
                },
                "BrokenWorker": {
                    "componentType": "worker",
                    "replicas": 1,
                    "dynamoNamespace": "missing-frontend-linkage",
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": f"nvcr.io/nvidia/ai-dynamo/vllm-runtime:{DYNAMO_VERSION}",
                        }
                    },
                },
            }
        },
    }


async def _observe_state(kubectl: KubectlClient, *, timeout: float) -> str | None:
    deadline = asyncio.get_event_loop().time() + timeout
    last_state: str | None = None
    while asyncio.get_event_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            _NAME,
            "-n",
            _NAMESPACE,
            "-o",
            "jsonpath={.status.state}",
            check=False,
        )
        if result.returncode == 0:
            last_state = result.stdout.strip() or None
            if last_state in {"failed", "successful"}:
                return last_state
        await asyncio.sleep(2.0)
    return last_state


async def _read_dgd(kubectl: KubectlClient) -> dict[str, Any]:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        _NAME,
        "-n",
        _NAMESPACE,
        "-o",
        "json",
        check=True,
    )
    return orjson.loads(result.stdout)


async def _find_webhook_target(kubectl: KubectlClient) -> WebhookTarget:
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
            "D119 requires exactly one DGD validating webhook Deployment; "
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
    raise AssertionError(f"D119: webhook {target.webhook_name!r} disappeared")


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
        f"D119: webhook deployment {target.deployment_namespace}/"
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
