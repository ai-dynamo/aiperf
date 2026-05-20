# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D1xx Dynamo operator/admission chaos scenarios."""

from __future__ import annotations

import asyncio
import base64
import subprocess
import time
from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    Literal,
)

import orjson
import pytest
from pytest import param

from aiperf.common.aiperf_logger import AIPerfLogger
from dev.versions import DYNAMO_VERSION
from tests.kubernetes.chaos.toxiproxy import (
    TOXIPROXY_APISERVER_PORT,
    ToxiproxyInjector,
)
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.chaos_dynamo.conftest import (
    DYNAMO_TOXIPROXY_NAMESPACE,
    DYNAMO_TOXIPROXY_SERVICE,
    wait_for_dgd_state,
)
from tests.kubernetes.gpu.dynamo.helpers import (
    DynamoConfig,
    DynamoDeployer,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


# D101

logger = AIPerfLogger(__name__)


async def test_d101_kill_operator_mid_dgd_apply(
    faults,
    kubectl,
    dynamo_operator,
    dynamo_deployment_namespace,
) -> None:
    """Apply DGD, kill operator during reconcile, assert resume to state=successful.

    Targets: ``dynamographdeployment_controller.go:119`` - the main Reconcile loop.
    The fault validates that controller-runtime's restart resumes from the last
    apiserver-observed state without orphaning children.

    The test applies the manifest manually instead of using ``deploy()`` so it
    controls fault timing relative to apply-return:

    1. Build a small DGD manifest (Qwen-tiny-style) by reusing
       ``DynamoConfig`` / ``DynamoDeployer.generate_manifest()`` with default
       ``v1beta1``. Do NOT call ``deploy()`` - we apply manually so we control
       fault timing relative to apply-return.

       from tests.kubernetes.gpu.dynamo.deployer import DynamoDeployer
       deployer = DynamoDeployer(kubectl, dynamo_config)
       manifest = deployer.generate_manifest()
       name = dynamo_config.name
       namespace = dynamo_deployment_namespace

    2. ``kubectl apply`` the manifest and capture name + namespace.

       await kubectl.apply(manifest)

    3. As soon as ``kubectl apply`` returns, inject the fault. A short sleep is
       fine; the goal is "kill the operator while it's mid-Reconcile", which on
       kind happens within 1-3s of apply.

       async with faults.inject(
           "operator.kill",
           target={
               "selector": "app.kubernetes.io/name=dynamo-operator",
               "ns": "dynamo-system",
           },
       ):
           # Restore for operator.kill is a no-op; kubelet auto-recreates.
           pass

    4. Wait for the operator deployment to be Available again (60s timeout).
       Uses a label-selector wait so the assertion spans the v0.9.x release
       name (``dynamo-operator``) and the v1.x release name
       (``dynamo-platform-dynamo-operator-controller-manager``).

       await kubectl.run(
           "wait",
           "-n",
           "dynamo-system",
           "-l",
           "app.kubernetes.io/name=dynamo-operator",
           "deployment",
           "--for=condition=Available",
           "--timeout=60s",
           check=True,
       )

    5. Wait for the DGD to reach ``status.state=successful``.

       from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
       await wait_for_dgd_state(
           kubectl, name, namespace, "successful", timeout=300.0,
       )

    6. Read the DGD as JSON and assert:
         - ``status.observedGeneration == metadata.generation``
         - ``status.state == "successful"``

       import orjson
       result = await kubectl.run(
           "get", "dynamographdeployment", name,
           "-n", namespace, "-o", "json",
           check=True,
       )
       dgd = orjson.loads(result.stdout)
       assert dgd["status"]["state"] == "successful"
       assert (
           dgd["status"]["observedGeneration"]
           == dgd["metadata"]["generation"]
       )

    7. List child resources and assert each carries the DGD as an
       ``ownerReference`` (no orphans).

       for kind in ("deployment", "service", "configmap", "role", "rolebinding"):
           res = await kubectl.run(
               "get", kind,
               "-n", namespace,
               "-l", f"nvidia.com/dynamographdeployment={name}",
               "-o", "json",
               check=False,
           )
           if res.returncode != 0:
               continue
           items = orjson.loads(res.stdout).get("items", [])
           for item in items:
               owners = item.get("metadata", {}).get("ownerReferences", [])
               assert any(
                   o.get("kind") == "DynamoGraphDeployment" and o.get("name") == name
                   for o in owners
               ), f"orphan {kind}/{item['metadata']['name']} has no DGD owner"

    8. Cleanup in a ``finally``: best-effort async delete (``--wait=false``).

       await kubectl.run(
           "delete", "dynamographdeployment", name,
           "-n", namespace, "--wait=false",
           check=False,
       )
    """
    await _run_d101_assertion(faults, kubectl, dynamo_deployment_namespace)


async def _run_d101_assertion(
    faults,
    kubectl,
    dynamo_deployment_namespace: str,
) -> None:
    """Full D101 assertion body for the operator-kill recovery scenario."""
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=dynamo_deployment_namespace,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    manifest = deployer.generate_manifest()
    name = deployer._deployment_name()
    namespace = dynamo_deployment_namespace

    try:
        await kubectl.apply(manifest, namespace=namespace)
        logger.info(
            f"D101: applied DGD {name} in ns {namespace}; injecting operator.kill"
        )

        async with faults.inject(
            "operator.kill",
            target={
                "selector": "app.kubernetes.io/name=dynamo-operator",
                "ns": "dynamo-system",
            },
        ):
            # Restore for operator.kill is a no-op; kubelet auto-recreates the pod.
            pass

        await kubectl.run(
            "wait",
            "-n",
            "dynamo-system",
            "-l",
            "app.kubernetes.io/name=dynamo-operator",
            "deployment",
            "--for=condition=Available",
            "--timeout=60s",
            check=True,
        )
        logger.info("D101: dynamo-operator deployment Available again post-kill")

        await wait_for_dgd_state(
            kubectl,
            name,
            namespace,
            "successful",
            timeout=600.0,
        )

        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=True,
        )
        dgd = orjson.loads(result.stdout)
        assert dgd["status"]["state"] == "successful", (
            f"DGD {name} ended in state={dgd['status'].get('state')!r}, expected 'successful'"
        )
        metadata_generation = dgd["metadata"]["generation"]
        observed_generation = dgd["status"].get("observedGeneration")
        assert observed_generation == metadata_generation, (
            f"observedGeneration={observed_generation} != generation={metadata_generation}"
        )
        logger.info(
            f"D101: DGD {name} reconciled successfully "
            f"(generation={metadata_generation}, observedGeneration={observed_generation})"
        )
    finally:
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


# D102

logger = AIPerfLogger(__name__)

_DGD_NAME = "dynamo-agg"
_DGD_NAMESPACE = "d102-double-delete"
_DGD_ESTABLISHED_TIMEOUT_S = 60.0
_DGD_DELETE_TIMEOUT_S = 90.0


async def test_d102_rapid_double_delete_dgd_is_idempotent(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD CRD/operator exist
) -> None:
    """Delete a newly-created DGD twice quickly and assert no finalizer wedge.

    The first delete should mark the DGD for removal; the second delete, issued
    within one second, must be harmless even if it races apiserver deletion. The
    observable contract is that the CR disappears rather than remaining in a
    terminating state with stuck finalizers.
    """
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=_DGD_NAMESPACE,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)

    await kubectl.create_namespace(_DGD_NAMESPACE)
    try:
        await kubectl.apply(deployer.generate_manifest())
        established = await _wait_for_dgd_established(
            kubectl,
            name=_DGD_NAME,
            namespace=_DGD_NAMESPACE,
            timeout=_DGD_ESTABLISHED_TIMEOUT_S,
        )
        assert established, await _dgd_observed_status_text(
            kubectl,
            name=_DGD_NAME,
            namespace=_DGD_NAMESPACE,
            prefix=(
                f"D102: DGD {_DGD_NAMESPACE}/{_DGD_NAME} never became readable "
                f"within {_DGD_ESTABLISHED_TIMEOUT_S}s after apply"
            ),
        )

        first_delete = await kubectl.run(
            "delete",
            "dynamographdeployment",
            _DGD_NAME,
            "-n",
            _DGD_NAMESPACE,
            "--wait=false",
            check=False,
        )
        assert first_delete.returncode == 0, (
            "D102: first DGD delete failed; "
            f"stdout={first_delete.stdout!r} stderr={first_delete.stderr!r}"
        )

        second_delete = await kubectl.run(
            "delete",
            "dynamographdeployment",
            _DGD_NAME,
            "-n",
            _DGD_NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )
        assert second_delete.returncode == 0, (
            "D102: second DGD delete was not idempotent; "
            f"stdout={second_delete.stdout!r} stderr={second_delete.stderr!r}"
        )

        disappeared = await _wait_for_dgd_absent(
            kubectl,
            name=_DGD_NAME,
            namespace=_DGD_NAMESPACE,
            timeout=_DGD_DELETE_TIMEOUT_S,
        )
        assert disappeared, await _dgd_observed_status_text(
            kubectl,
            name=_DGD_NAME,
            namespace=_DGD_NAMESPACE,
            prefix=(
                f"D102: DGD {_DGD_NAMESPACE}/{_DGD_NAME} still existed "
                f"{_DGD_DELETE_TIMEOUT_S}s after rapid double-delete"
            ),
        )
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _DGD_NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _wait_for_dgd_established(
    kubectl: KubectlClient,
    *,
    name: str,
    namespace: str,
    timeout: float,
) -> bool:
    """Return True once the apiserver can read the DGD JSON document."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                dgd = orjson.loads(result.stdout)
            except orjson.JSONDecodeError as exc:
                logger.debug(lambda exc=exc: f"D102 DGD JSON parse failed: {exc!r}")
            else:
                metadata = dgd.get("metadata", {})
                if metadata.get("uid") and metadata.get("resourceVersion"):
                    return True
        await asyncio.sleep(0.5)
    return False


async def _wait_for_dgd_absent(
    kubectl: KubectlClient,
    *,
    name: str,
    namespace: str,
    timeout: float,
) -> bool:
    """Return True once ``kubectl get`` reports the DGD is gone."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            check=False,
        )
        if result.returncode != 0:
            return True
        await asyncio.sleep(1.0)
    return False


async def _dgd_observed_status_text(
    kubectl: KubectlClient,
    *,
    name: str,
    namespace: str,
    prefix: str,
) -> str:
    """Build a failure message with current status and finalizer context."""
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return (
            f"{prefix}; current read failed with "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    try:
        dgd = orjson.loads(result.stdout)
    except orjson.JSONDecodeError as exc:
        return f"{prefix}; current DGD JSON could not be parsed: {exc!r}"

    metadata = dgd.get("metadata", {})
    status = dgd.get("status", {})
    return (
        f"{prefix}; status={status!r}; "
        f"finalizers={metadata.get('finalizers', [])!r}; "
        f"deletionTimestamp={metadata.get('deletionTimestamp')!r}"
    )


# D104


async def test_d104_invalid_dgd_replicas_negative(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD CRD/webhook exists
) -> None:
    """Apply DGD with replicas=-1; assert admission rejects the invalid spec.

    The v1alpha1 CRD now rejects negative replicas at apiserver admission time,
    before a CR exists for the operator to drive into ``status.state=failed``.
    This keeps D104 runnable against its intended validation signal without
    polling for a resource the apiserver correctly never creates.
    """
    name = "d104-test"
    ns = "d104-invalid"
    manifest = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name, "namespace": ns},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": -1,  # INVALID
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": f"nvcr.io/nvidia/ai-dynamo/vllm-runtime:{DYNAMO_VERSION}",
                        }
                    },
                }
            }
        },
    }

    await kubectl.create_namespace(ns)
    try:
        try:
            await kubectl.apply(orjson.dumps(manifest).decode(), namespace=ns)
        except RuntimeError as exc:
            message = str(exc).lower()
        else:
            pytest.fail("expected replicas=-1 manifest to fail admission")

        assert "replicas" in message, (
            f"expected admission error to mention replicas, got {message!r}"
        )
        assert "greater than or equal to 0" in message or "minimum" in message, (
            "expected admission error to name the non-negative constraint, "
            f"got {message!r}"
        )

        get_result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            ns,
            check=False,
        )
        assert get_result.returncode != 0, (
            "invalid DGD should be rejected at admission, not created for "
            "operator reconciliation"
        )
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            ns,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


# D105

logger = AIPerfLogger(__name__)

_DGD_NAMESPACE = "d105-recreate-same-name"
_ABSENT_TIMEOUT_S = 5.0
_SUCCESS_TIMEOUT_S = 600.0


async def test_d105_recreate_same_dgd_name_reconciles_successfully(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the Dynamo operator is installed
) -> None:
    """Delete a successful DGD and re-create the same name within 5 seconds.

    The first apply is a prerequisite probe: if a clean minimal DGD cannot
    reach ``state=successful``, the cluster cannot start Dynamo workloads and
    this scenario cannot distinguish tombstone handling from baseline startup
    failure. Once that baseline succeeds, the second same-name CR must also
    reach ``state=successful`` and must have a different UID from the deleted
    predecessor.
    """
    config = DynamoConfig.single_gpu_disagg(
        namespace=_DGD_NAMESPACE,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    manifest = deployer.generate_manifest()
    name = deployer._deployment_name()

    try:
        await kubectl.apply(manifest, namespace=_DGD_NAMESPACE)
        try:
            await wait_for_dgd_state(
                kubectl,
                name,
                _DGD_NAMESPACE,
                "successful",
                timeout=_SUCCESS_TIMEOUT_S,
            )
        except TimeoutError as exc:
            status = await _dgd_status_snapshot(kubectl, name=name)
            pytest.skip(
                "D105 requires the cluster to start a minimal Dynamo workload; "
                f"baseline DGD {_DGD_NAMESPACE}/{name} did not reach "
                f"state='successful' within {_SUCCESS_TIMEOUT_S}s. "
                f"Status: {status!r}. Error: {exc}"
            )

        first_dgd = await _read_dgd_json(kubectl, name=name)
        first_uid = first_dgd["metadata"]["uid"]
        logger.info(f"D105: baseline DGD {_DGD_NAMESPACE}/{name} uid={first_uid}")

        await kubectl.run(
            "delete",
            "dynamographdeployment",
            name,
            "-n",
            _DGD_NAMESPACE,
            "--wait=false",
            check=True,
        )
        await _d105_wait_for_dgd_absent(kubectl, name=name, timeout=_ABSENT_TIMEOUT_S)

        await kubectl.apply(manifest, namespace=_DGD_NAMESPACE)
        second_dgd = await _read_dgd_json(kubectl, name=name)
        second_uid = second_dgd["metadata"]["uid"]
        assert second_uid != first_uid, (
            f"D105 expected a fresh same-name DGD after deletion, but "
            f"{_DGD_NAMESPACE}/{name} still has uid={second_uid!r}"
        )

        observed_state = await wait_for_dgd_state(
            kubectl,
            name,
            _DGD_NAMESPACE,
            "successful",
            timeout=_SUCCESS_TIMEOUT_S,
        )
        assert observed_state == "successful"

        final_dgd = await _read_dgd_json(kubectl, name=name)
        metadata_generation = final_dgd["metadata"]["generation"]
        observed_generation = final_dgd["status"].get("observedGeneration")
        assert observed_generation == metadata_generation, (
            f"D105 same-name DGD observedGeneration={observed_generation} "
            f"!= generation={metadata_generation}"
        )
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _DGD_NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _d105_wait_for_dgd_absent(
    kubectl: KubectlClient,
    *,
    name: str,
    timeout: float,
) -> None:
    """Wait until the first same-name DGD is gone so the re-apply is a new CR."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            _DGD_NAMESPACE,
            "-o",
            "jsonpath={.metadata.uid}",
            check=False,
        )
        if result.returncode != 0:
            return
        await asyncio.sleep(0.25)

    status = await _dgd_status_snapshot(kubectl, name=name)
    raise AssertionError(
        f"D105 expected {_DGD_NAMESPACE}/{name} to be deleted within "
        f"{timeout}s before same-name re-create; last status: {status!r}"
    )


async def _read_dgd_json(kubectl: KubectlClient, *, name: str) -> dict[str, Any]:
    """Read a DynamoGraphDeployment as a parsed JSON object."""
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        _DGD_NAMESPACE,
        "-o",
        "json",
        check=True,
    )
    return orjson.loads(result.stdout)


async def _dgd_status_snapshot(kubectl: KubectlClient, *, name: str) -> str:
    """Return status/finalizers text for D105 failure and skip messages."""
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        _DGD_NAMESPACE,
        "-o",
        "jsonpath={.status}{' finalizers='}{.metadata.finalizers}",
        check=False,
    )
    if result.returncode != 0:
        return result.stderr.strip() or result.stdout.strip()
    return result.stdout.strip()


# D106

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


# D107

logger = AIPerfLogger(__name__)

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_REVOKED_STATE_WINDOW_S = 45.0


@dataclass(frozen=True)
class RBACTarget:
    """A reversible RBAC object that grants the Dynamo operator deployment patch."""

    kind: Literal["role", "clusterrole"]
    name: str
    namespace: str | None
    rules: list[dict[str, Any]]
    mutated_rules: list[dict[str, Any]]

    @property
    def display_name(self) -> str:
        """Return a kubectl-addressable target name for skip/failure messages."""
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


async def test_d107_operator_rbac_revoked_mid_reconcile(
    kubectl: KubectlClient,
    dynamo_operator: None,
    dynamo_deployment_namespace: str,
) -> None:
    """Revoke deployments/patch and assert DGD status does not falsely succeed."""
    target = await _find_reversible_deployment_patch_target(kubectl)
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=dynamo_deployment_namespace,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    manifest = deployer.generate_manifest()
    name = deployer._deployment_name()
    namespace = dynamo_deployment_namespace
    rbac_revoked = False

    try:
        try:
            await _revoke_deployment_patch(kubectl, target)
            rbac_revoked = True
            logger.info(f"D107: revoked deployments/patch from {target.display_name}")

            await kubectl.apply(manifest, namespace=namespace)
            observed = await _observe_dgd_while_rbac_revoked(
                kubectl,
                name,
                namespace,
                timeout_s=_REVOKED_STATE_WINDOW_S,
            )
            assert observed != "successful", (
                f"DGD {name} reported status.state='successful' while "
                f"deployments/patch was revoked from {target.display_name}"
            )
        finally:
            if rbac_revoked:
                await _restore_deployment_patch(kubectl, target)
                logger.info(f"D107: restored original rules on {target.display_name}")

        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)
    finally:
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


async def _find_reversible_deployment_patch_target(
    kubectl: KubectlClient,
) -> RBACTarget:
    service_account = await _operator_service_account(kubectl)
    candidates = await _operator_bound_targets(kubectl, service_account)
    inspected = [candidate.display_name for candidate in candidates]

    reversible = [
        target for target in candidates if _has_explicit_deployment_patch(target)
    ]
    if len(reversible) != 1:
        pytest.skip(
            "D107 requires exactly one reversible operator RBAC target granting "
            f"deployments/patch to {service_account}; inspected targets: "
            f"{', '.join(inspected) if inspected else '<none>'}"
        )

    target = reversible[0]
    mutated_rules = _without_deployment_patch(target.rules)
    if mutated_rules == target.rules:
        pytest.skip(
            f"D107 found {target.display_name} but could not build reversible "
            "rules without deployments/patch"
        )
    return RBACTarget(
        kind=target.kind,
        name=target.name,
        namespace=target.namespace,
        rules=target.rules,
        mutated_rules=mutated_rules,
    )


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
            "D107 requires exactly one Dynamo operator deployment; inspected deployments: "
            f"{', '.join(names) if names else '<none>'}"
        )
    return deployments[0]["spec"]["template"]["spec"].get(
        "serviceAccountName",
        "default",
    )


async def _operator_bound_targets(
    kubectl: KubectlClient,
    service_account: str,
) -> list[RBACTarget]:
    role_bindings = await _bound_role_refs(
        kubectl,
        "rolebinding",
        service_account,
        namespaced=True,
    )
    cluster_role_bindings = await _bound_role_refs(
        kubectl,
        "clusterrolebinding",
        service_account,
        namespaced=False,
    )
    targets: list[RBACTarget] = []
    for kind, name, namespace in role_bindings + cluster_role_bindings:
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


def _has_explicit_deployment_patch(target: RBACTarget) -> bool:
    return any(_rule_grants_explicit_deployment_patch(rule) for rule in target.rules)


def _rule_grants_explicit_deployment_patch(rule: dict[str, Any]) -> bool:
    api_groups = set(rule.get("apiGroups", []))
    resources = set(rule.get("resources", []))
    verbs = set(rule.get("verbs", []))
    return (
        "apps" in api_groups
        and "deployments" in resources
        and "patch" in verbs
        and "*" not in api_groups
        and "*" not in resources
        and "*" not in verbs
    )


def _without_deployment_patch(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mutated: list[dict[str, Any]] = []
    for rule in rules:
        copied = dict(rule)
        if _rule_grants_explicit_deployment_patch(rule):
            copied["verbs"] = [
                verb for verb in rule.get("verbs", []) if verb != "patch"
            ]
        mutated.append(copied)
    return mutated


async def _revoke_deployment_patch(kubectl: KubectlClient, target: RBACTarget) -> None:
    await _patch_rules(kubectl, target, target.mutated_rules)


async def _restore_deployment_patch(kubectl: KubectlClient, target: RBACTarget) -> None:
    await _patch_rules(kubectl, target, target.rules)


async def _patch_rules(
    kubectl: KubectlClient,
    target: RBACTarget,
    rules: list[dict[str, Any]],
) -> None:
    patch = orjson.dumps({"rules": rules}).decode()
    args = ["patch", target.kind, target.name, "--type=merge", f"-p={patch}"]
    if target.namespace is not None:
        args.extend(["-n", target.namespace])
    await kubectl.run(*args, check=True)


async def _observe_dgd_while_rbac_revoked(
    kubectl: KubectlClient,
    name: str,
    namespace: str,
    *,
    timeout_s: float,
) -> str | None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_state: str | None = None
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            dgd = orjson.loads(result.stdout)
            last_state = dgd.get("status", {}).get("state")
            if last_state == "successful":
                return last_state
        await asyncio.sleep(2.0)
    return last_state


# D108

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_PROXY_NAME = "d108-apiserver"
_PROXY_HOST = (
    f"{DYNAMO_TOXIPROXY_SERVICE}.{DYNAMO_TOXIPROXY_NAMESPACE}.svc.cluster.local"
)


@dataclass(frozen=True, slots=True)
class _DeploymentEnv:
    name: str
    env: dict[str, str]


async def test_d108_apiserver_pause_during_reconcile_recovers(
    kubectl: KubectlClient,
    dynamo_toxiproxy: ToxiproxyInjector,
    dynamo_deployment_namespace: str,
) -> None:
    deployment_env = await _operator_deployment_env(kubectl)
    await dynamo_toxiproxy.reset()
    await dynamo_toxiproxy.add_proxy(
        name=_PROXY_NAME,
        listen=f"0.0.0.0:{TOXIPROXY_APISERVER_PORT}",
        upstream="kubernetes.default.svc:443",
    )

    name = ""
    namespace = dynamo_deployment_namespace
    routed = False
    try:
        await _route_operator_to_apiserver_proxy(kubectl, deployment_env.name)
        routed = await _wait_operator_available(kubectl, timeout_s=90.0)
        if not routed:
            await _restore_operator_env(kubectl, deployment_env)
            pytest.skip(
                "D108 requires the Dynamo operator to run with apiserver traffic "
                "routed through toxiproxy; patched deployment did not become "
                "Available, likely because the operator image lacks a TLS/SNI "
                "override for KUBERNETES_SERVICE_HOST=toxiproxy"
            )

        config = DynamoConfig(
            model_name="Qwen/Qwen3-0.6B",
            namespace=namespace,
            api_version="v1alpha1",
        )
        deployer = DynamoDeployer(kubectl, config)
        name = deployer._deployment_name()
        await kubectl.apply(deployer.generate_manifest(), namespace=namespace)

        await dynamo_toxiproxy.add_toxic(_PROXY_NAME, "timeout", {"timeout": 0})
        await asyncio.sleep(30.0)
        await dynamo_toxiproxy.remove_toxic(_PROXY_NAME, "timeout_downstream")

        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=420.0)
    finally:
        if name:
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
        if routed:
            await _restore_operator_env(kubectl, deployment_env)
        await dynamo_toxiproxy.reset()


async def _operator_deployment_env(kubectl: KubectlClient) -> _DeploymentEnv:
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
    import orjson

    deployments = orjson.loads(result.stdout or b"{}").get("items", [])
    if len(deployments) != 1:
        names = [
            item.get("metadata", {}).get("name", "<unnamed>") for item in deployments
        ]
        pytest.skip(
            "D108 requires exactly one Dynamo operator deployment; found "
            f"{', '.join(names) if names else '<none>'}"
        )
    deployment = deployments[0]
    containers = (
        deployment.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
    )
    if not containers:
        pytest.skip("D108 requires the Dynamo operator deployment to have a container")
    env = {
        str(item.get("name")): str(item.get("value"))
        for item in containers[0].get("env", [])
        if item.get("name") and item.get("value") is not None
    }
    return _DeploymentEnv(name=deployment["metadata"]["name"], env=env)


async def _route_operator_to_apiserver_proxy(
    kubectl: KubectlClient, deployment_name: str
) -> None:
    await kubectl.run(
        "set",
        "env",
        f"deployment/{deployment_name}",
        f"KUBERNETES_SERVICE_HOST={_PROXY_HOST}",
        f"KUBERNETES_SERVICE_PORT={TOXIPROXY_APISERVER_PORT}",
        "-n",
        _OPERATOR_NAMESPACE,
        check=True,
    )


async def _restore_operator_env(
    kubectl: KubectlClient, deployment_env: _DeploymentEnv
) -> None:
    args = ["set", "env", f"deployment/{deployment_env.name}"]
    for key in ("KUBERNETES_SERVICE_HOST", "KUBERNETES_SERVICE_PORT"):
        if key in deployment_env.env:
            args.append(f"{key}={deployment_env.env[key]}")
        else:
            args.append(f"{key}-")
    args.extend(["-n", _OPERATOR_NAMESPACE])
    await kubectl.run(*args, check=False)
    await _wait_operator_available(kubectl, timeout_s=90.0)


async def _wait_operator_available(kubectl: KubectlClient, *, timeout_s: float) -> bool:
    result = await kubectl.run(
        "wait",
        "-n",
        _OPERATOR_NAMESPACE,
        "-l",
        _OPERATOR_SELECTOR,
        "deployment",
        "--for=condition=Available",
        f"--timeout={int(timeout_s)}s",
        check=False,
    )
    return result.returncode == 0


# D109

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_OBSERVE_REVOKED_S = 45.0


@dataclass(frozen=True, slots=True)
class _RbacOwner:
    scope: Literal["role", "clusterrole"]
    name: str
    namespace: str | None

    @property
    def label(self) -> str:
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


async def _d109_operator_service_account(kubectl: KubectlClient) -> str:
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
    deployments = orjson.loads(result.stdout or b"{}").get("items", [])
    if len(deployments) != 1:
        names = [
            item.get("metadata", {}).get("name", "<unnamed>") for item in deployments
        ]
        pytest.skip(
            "Dynamo RBAC chaos requires exactly one operator deployment; found "
            f"{', '.join(names) if names else '<none>'}"
        )
    return str(
        deployments[0]["spec"]["template"]["spec"].get("serviceAccountName", "default")
    )


async def _find_unique_operator_rbac_owner(
    kubectl: KubectlClient,
    *,
    api_group: str,
    resource: str,
    verb: str,
    case_id: str,
) -> _RbacOwner:
    service_account = await _d109_operator_service_account(kubectl)
    candidates: list[_RbacOwner] = []
    inspected: list[str] = []
    for scope, name, namespace in await _operator_bound_role_refs(
        kubectl, service_account
    ):
        inspected.append(f"{scope}/{namespace + '/' if namespace else ''}{name}")
        body = await _load_rbac(kubectl, scope, name, namespace)
        if body is None:
            continue
        if _has_exact_rule(body.get("rules") or [], api_group, resource, verb):
            candidates.append(_RbacOwner(scope=scope, name=name, namespace=namespace))
    if len(candidates) != 1:
        pytest.skip(
            f"{case_id} requires exactly one operator-bound RBAC rule granting "
            f"{verb!r} on {resource!r} apiGroup={api_group!r}; candidates="
            f"{', '.join(c.label for c in candidates) or '<none>'}; inspected="
            f"{', '.join(inspected) if inspected else '<none>'}"
        )
    return candidates[0]


async def _operator_bound_role_refs(
    kubectl: KubectlClient,
    service_account: str,
) -> list[tuple[Literal["role", "clusterrole"], str, str | None]]:
    refs: list[tuple[Literal["role", "clusterrole"], str, str | None]] = []
    for binding_kind, namespaced in (
        ("rolebinding", True),
        ("clusterrolebinding", False),
    ):
        args = ["get", binding_kind, "-o", "json"]
        if namespaced:
            args.insert(2, "-n")
            args.insert(3, _OPERATOR_NAMESPACE)
        result = await kubectl.run(*args, check=True)
        for binding in orjson.loads(result.stdout or b"{}").get("items", []):
            if not _d109_has_operator_subject(
                binding.get("subjects") or [], service_account
            ):
                continue
            role_ref = binding.get("roleRef") or {}
            scope = str(role_ref.get("kind", "")).lower()
            if scope not in {"role", "clusterrole"}:
                continue
            namespace = (
                binding.get("metadata", {}).get("namespace")
                if scope == "role"
                else None
            )
            refs.append((scope, str(role_ref.get("name", "")), namespace))
    return refs


def _d109_has_operator_subject(
    subjects: list[dict[str, Any]], service_account: str
) -> bool:
    return any(
        subject.get("kind") == "ServiceAccount"
        and subject.get("name") == service_account
        and subject.get("namespace") == _OPERATOR_NAMESPACE
        for subject in subjects
    )


async def _load_rbac(
    kubectl: KubectlClient,
    scope: Literal["role", "clusterrole"],
    name: str,
    namespace: str | None,
) -> dict[str, Any] | None:
    args = ["get", scope, name]
    if namespace is not None:
        args.extend(["-n", namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        return None
    return dict(orjson.loads(result.stdout or b"{}"))


def _has_exact_rule(
    rules: list[dict[str, Any]],
    api_group: str,
    resource: str,
    verb: str,
) -> bool:
    for rule in rules:
        if "*" in (rule.get("apiGroups") or []):
            continue
        if "*" in (rule.get("resources") or []):
            continue
        if "*" in (rule.get("verbs") or []):
            continue
        if (
            api_group in (rule.get("apiGroups") or [])
            and resource in (rule.get("resources") or [])
            and verb in (rule.get("verbs") or [])
        ):
            return True
    return False


def _rbac_target(owner: _RbacOwner) -> dict[str, str]:
    target = {"scope": owner.scope, "name": owner.name}
    if owner.namespace is not None:
        target["ns"] = owner.namespace
    return target


async def _apply_fresh_dgd(kubectl: KubectlClient, namespace: str) -> tuple[str, str]:
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=namespace,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()
    await kubectl.apply(deployer.generate_manifest(), namespace=namespace)
    return name, namespace


async def _observe_not_successful(
    kubectl: KubectlClient,
    name: str,
    namespace: str,
    *,
    case_id: str,
    timeout_s: float = _OBSERVE_REVOKED_S,
) -> str | None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_state: str | None = None
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            body = orjson.loads(result.stdout or b"{}")
            last_state = body.get("status", {}).get("state")
            assert last_state != "successful", (
                f"{case_id}: DGD {namespace}/{name} reached successful while "
                "required operator RBAC was revoked"
            )
        await asyncio.sleep(2.0)
    return last_state


async def _delete_dgd(kubectl: KubectlClient, name: str, namespace: str) -> None:
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


async def test_d109_status_subresource_rbac_revoked_mid_reconcile(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    owner = await _find_unique_operator_rbac_owner(
        kubectl,
        api_group="nvidia.com",
        resource="dynamographdeployments/status",
        verb="update",
        case_id="D109",
    )
    faults = request.getfixturevalue("faults")
    name = ""
    namespace = dynamo_deployment_namespace
    try:
        async with faults.inject(
            "cluster.rbac.revoke",
            target=_rbac_target(owner),
            api_group="nvidia.com",
            resource="dynamographdeployments/status",
            verb="update",
        ):
            name, namespace = await _apply_fresh_dgd(kubectl, namespace)
            await _observe_not_successful(kubectl, name, namespace, case_id="D109")
        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)
    finally:
        if name:
            await _delete_dgd(kubectl, name, namespace)


# D110

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_OBSERVE_REVOKED_S = 45.0


@dataclass(frozen=True, slots=True)
class _d110_RbacOwner:
    scope: Literal["role", "clusterrole"]
    name: str
    namespace: str | None

    @property
    def label(self) -> str:
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


async def _d110_operator_service_account(kubectl: KubectlClient) -> str:
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
    deployments = orjson.loads(result.stdout or b"{}").get("items", [])
    if len(deployments) != 1:
        names = [
            item.get("metadata", {}).get("name", "<unnamed>") for item in deployments
        ]
        pytest.skip(
            "Dynamo RBAC chaos requires exactly one operator deployment; found "
            f"{', '.join(names) if names else '<none>'}"
        )
    return str(
        deployments[0]["spec"]["template"]["spec"].get("serviceAccountName", "default")
    )


async def _d110_find_unique_operator_rbac_owner(
    kubectl: KubectlClient,
    *,
    api_group: str,
    resource: str,
    verb: str,
    case_id: str,
) -> _d110_RbacOwner:
    service_account = await _d110_operator_service_account(kubectl)
    candidates: list[_d110_RbacOwner] = []
    inspected: list[str] = []
    for scope, name, namespace in await _d110_operator_bound_role_refs(
        kubectl, service_account
    ):
        inspected.append(f"{scope}/{namespace + '/' if namespace else ''}{name}")
        body = await _d110_load_rbac(kubectl, scope, name, namespace)
        if body is None:
            continue
        if _d110_has_exact_rule(body.get("rules") or [], api_group, resource, verb):
            candidates.append(
                _d110_RbacOwner(scope=scope, name=name, namespace=namespace)
            )
    if len(candidates) != 1:
        pytest.skip(
            f"{case_id} requires exactly one operator-bound RBAC rule granting "
            f"{verb!r} on {resource!r} apiGroup={api_group!r}; candidates="
            f"{', '.join(c.label for c in candidates) or '<none>'}; inspected="
            f"{', '.join(inspected) if inspected else '<none>'}"
        )
    return candidates[0]


async def _d110_operator_bound_role_refs(
    kubectl: KubectlClient,
    service_account: str,
) -> list[tuple[Literal["role", "clusterrole"], str, str | None]]:
    refs: list[tuple[Literal["role", "clusterrole"], str, str | None]] = []
    for binding_kind, namespaced in (
        ("rolebinding", True),
        ("clusterrolebinding", False),
    ):
        args = ["get", binding_kind, "-o", "json"]
        if namespaced:
            args.insert(2, "-n")
            args.insert(3, _OPERATOR_NAMESPACE)
        result = await kubectl.run(*args, check=True)
        for binding in orjson.loads(result.stdout or b"{}").get("items", []):
            if not _d110_has_operator_subject(
                binding.get("subjects") or [], service_account
            ):
                continue
            role_ref = binding.get("roleRef") or {}
            scope = str(role_ref.get("kind", "")).lower()
            if scope not in {"role", "clusterrole"}:
                continue
            namespace = (
                binding.get("metadata", {}).get("namespace")
                if scope == "role"
                else None
            )
            refs.append((scope, str(role_ref.get("name", "")), namespace))
    return refs


def _d110_has_operator_subject(
    subjects: list[dict[str, Any]], service_account: str
) -> bool:
    return any(
        subject.get("kind") == "ServiceAccount"
        and subject.get("name") == service_account
        and subject.get("namespace") == _OPERATOR_NAMESPACE
        for subject in subjects
    )


async def _d110_load_rbac(
    kubectl: KubectlClient,
    scope: Literal["role", "clusterrole"],
    name: str,
    namespace: str | None,
) -> dict[str, Any] | None:
    args = ["get", scope, name]
    if namespace is not None:
        args.extend(["-n", namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        return None
    return dict(orjson.loads(result.stdout or b"{}"))


def _d110_has_exact_rule(
    rules: list[dict[str, Any]],
    api_group: str,
    resource: str,
    verb: str,
) -> bool:
    for rule in rules:
        if "*" in (rule.get("apiGroups") or []):
            continue
        if "*" in (rule.get("resources") or []):
            continue
        if "*" in (rule.get("verbs") or []):
            continue
        if (
            api_group in (rule.get("apiGroups") or [])
            and resource in (rule.get("resources") or [])
            and verb in (rule.get("verbs") or [])
        ):
            return True
    return False


def _d110_rbac_target(owner: _d110_RbacOwner) -> dict[str, str]:
    target = {"scope": owner.scope, "name": owner.name}
    if owner.namespace is not None:
        target["ns"] = owner.namespace
    return target


async def _d110_apply_fresh_dgd(
    kubectl: KubectlClient, namespace: str
) -> tuple[str, str]:
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=namespace,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()
    await kubectl.apply(deployer.generate_manifest(), namespace=namespace)
    return name, namespace


async def _d110_observe_not_successful(
    kubectl: KubectlClient,
    name: str,
    namespace: str,
    *,
    case_id: str,
    timeout_s: float = _OBSERVE_REVOKED_S,
) -> str | None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_state: str | None = None
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            body = orjson.loads(result.stdout or b"{}")
            last_state = body.get("status", {}).get("state")
            assert last_state != "successful", (
                f"{case_id}: DGD {namespace}/{name} reached successful while "
                "required operator RBAC was revoked"
            )
        await asyncio.sleep(2.0)
    return last_state


async def _d110_delete_dgd(kubectl: KubectlClient, name: str, namespace: str) -> None:
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


async def test_d110_finalizer_rbac_revoked_during_delete(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    owner = await _d110_find_unique_operator_rbac_owner(
        kubectl,
        api_group="nvidia.com",
        resource="dynamographdeployments/finalizers",
        verb="update",
        case_id="D110",
    )
    faults = request.getfixturevalue("faults")
    name, namespace = await _d110_apply_fresh_dgd(kubectl, dynamo_deployment_namespace)
    try:
        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)
        before = await _get_dgd(kubectl, name, namespace)
        if not before.get("metadata", {}).get("finalizers"):
            pytest.skip(f"D110 requires {namespace}/{name} to carry a DGD finalizer")
        async with faults.inject(
            "cluster.rbac.revoke",
            target=_d110_rbac_target(owner),
            api_group="nvidia.com",
            resource="dynamographdeployments/finalizers",
            verb="update",
        ):
            await _d110_delete_dgd(kubectl, name, namespace)
            blocked = await _wait_for_deletion_timestamp(
                kubectl, name, namespace, timeout_s=30.0
            )
            assert blocked, (
                f"D110: DGD {namespace}/{name} was not observed stuck in Terminating "
                "while finalizer update RBAC was revoked"
            )
        await _wait_for_gone(kubectl, name, namespace, timeout_s=180.0)
    finally:
        await _d110_delete_dgd(kubectl, name, namespace)


async def _get_dgd(kubectl: KubectlClient, name: str, namespace: str) -> dict[str, Any]:
    result = await kubectl.run(
        "get", "dynamographdeployment", name, "-n", namespace, "-o", "json", check=True
    )
    return dict(orjson.loads(result.stdout or b"{}"))


async def _wait_for_deletion_timestamp(
    kubectl: KubectlClient, name: str, namespace: str, *, timeout_s: float
) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            body = orjson.loads(result.stdout or b"{}")
            if body.get("metadata", {}).get("deletionTimestamp"):
                return True
        else:
            return False
        await asyncio.sleep(1.0)
    return False


async def _wait_for_gone(
    kubectl: KubectlClient, name: str, namespace: str, *, timeout_s: float
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get", "dynamographdeployment", name, "-n", namespace, check=False
        )
        if result.returncode != 0:
            return
        await asyncio.sleep(2.0)
    raise AssertionError(
        f"D110: DGD {namespace}/{name} still exists after RBAC restore"
    )


# D111

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_OBSERVE_REVOKED_S = 45.0


@dataclass(frozen=True, slots=True)
class _d111_RbacOwner:
    scope: Literal["role", "clusterrole"]
    name: str
    namespace: str | None

    @property
    def label(self) -> str:
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


async def _d111_operator_service_account(kubectl: KubectlClient) -> str:
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
    deployments = orjson.loads(result.stdout or b"{}").get("items", [])
    if len(deployments) != 1:
        names = [
            item.get("metadata", {}).get("name", "<unnamed>") for item in deployments
        ]
        pytest.skip(
            "Dynamo RBAC chaos requires exactly one operator deployment; found "
            f"{', '.join(names) if names else '<none>'}"
        )
    return str(
        deployments[0]["spec"]["template"]["spec"].get("serviceAccountName", "default")
    )


async def _d111_find_unique_operator_rbac_owner(
    kubectl: KubectlClient,
    *,
    api_group: str,
    resource: str,
    verb: str,
    case_id: str,
) -> _d111_RbacOwner:
    service_account = await _d111_operator_service_account(kubectl)
    candidates: list[_d111_RbacOwner] = []
    inspected: list[str] = []
    for scope, name, namespace in await _d111_operator_bound_role_refs(
        kubectl, service_account
    ):
        inspected.append(f"{scope}/{namespace + '/' if namespace else ''}{name}")
        body = await _d111_load_rbac(kubectl, scope, name, namespace)
        if body is None:
            continue
        if _d111_has_exact_rule(body.get("rules") or [], api_group, resource, verb):
            candidates.append(
                _d111_RbacOwner(scope=scope, name=name, namespace=namespace)
            )
    if len(candidates) != 1:
        pytest.skip(
            f"{case_id} requires exactly one operator-bound RBAC rule granting "
            f"{verb!r} on {resource!r} apiGroup={api_group!r}; candidates="
            f"{', '.join(c.label for c in candidates) or '<none>'}; inspected="
            f"{', '.join(inspected) if inspected else '<none>'}"
        )
    return candidates[0]


async def _d111_operator_bound_role_refs(
    kubectl: KubectlClient,
    service_account: str,
) -> list[tuple[Literal["role", "clusterrole"], str, str | None]]:
    refs: list[tuple[Literal["role", "clusterrole"], str, str | None]] = []
    for binding_kind, namespaced in (
        ("rolebinding", True),
        ("clusterrolebinding", False),
    ):
        args = ["get", binding_kind, "-o", "json"]
        if namespaced:
            args.insert(2, "-n")
            args.insert(3, _OPERATOR_NAMESPACE)
        result = await kubectl.run(*args, check=True)
        for binding in orjson.loads(result.stdout or b"{}").get("items", []):
            if not _d111_has_operator_subject(
                binding.get("subjects") or [], service_account
            ):
                continue
            role_ref = binding.get("roleRef") or {}
            scope = str(role_ref.get("kind", "")).lower()
            if scope not in {"role", "clusterrole"}:
                continue
            namespace = (
                binding.get("metadata", {}).get("namespace")
                if scope == "role"
                else None
            )
            refs.append((scope, str(role_ref.get("name", "")), namespace))
    return refs


def _d111_has_operator_subject(
    subjects: list[dict[str, Any]], service_account: str
) -> bool:
    return any(
        subject.get("kind") == "ServiceAccount"
        and subject.get("name") == service_account
        and subject.get("namespace") == _OPERATOR_NAMESPACE
        for subject in subjects
    )


async def _d111_load_rbac(
    kubectl: KubectlClient,
    scope: Literal["role", "clusterrole"],
    name: str,
    namespace: str | None,
) -> dict[str, Any] | None:
    args = ["get", scope, name]
    if namespace is not None:
        args.extend(["-n", namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        return None
    return dict(orjson.loads(result.stdout or b"{}"))


def _d111_has_exact_rule(
    rules: list[dict[str, Any]],
    api_group: str,
    resource: str,
    verb: str,
) -> bool:
    for rule in rules:
        if "*" in (rule.get("apiGroups") or []):
            continue
        if "*" in (rule.get("resources") or []):
            continue
        if "*" in (rule.get("verbs") or []):
            continue
        if (
            api_group in (rule.get("apiGroups") or [])
            and resource in (rule.get("resources") or [])
            and verb in (rule.get("verbs") or [])
        ):
            return True
    return False


def _d111_rbac_target(owner: _d111_RbacOwner) -> dict[str, str]:
    target = {"scope": owner.scope, "name": owner.name}
    if owner.namespace is not None:
        target["ns"] = owner.namespace
    return target


async def _d111_apply_fresh_dgd(
    kubectl: KubectlClient, namespace: str
) -> tuple[str, str]:
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=namespace,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()
    await kubectl.apply(deployer.generate_manifest(), namespace=namespace)
    return name, namespace


async def _d111_observe_not_successful(
    kubectl: KubectlClient,
    name: str,
    namespace: str,
    *,
    case_id: str,
    timeout_s: float = _OBSERVE_REVOKED_S,
) -> str | None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_state: str | None = None
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            body = orjson.loads(result.stdout or b"{}")
            last_state = body.get("status", {}).get("state")
            assert last_state != "successful", (
                f"{case_id}: DGD {namespace}/{name} reached successful while "
                "required operator RBAC was revoked"
            )
        await asyncio.sleep(2.0)
    return last_state


async def _d111_delete_dgd(kubectl: KubectlClient, name: str, namespace: str) -> None:
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


async def test_d111_operator_loses_dgd_list_watch_rbac(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    list_owner = await _d111_find_unique_operator_rbac_owner(
        kubectl,
        api_group="nvidia.com",
        resource="dynamographdeployments",
        verb="list",
        case_id="D111",
    )
    watch_owner = await _d111_find_unique_operator_rbac_owner(
        kubectl,
        api_group="nvidia.com",
        resource="dynamographdeployments",
        verb="watch",
        case_id="D111",
    )
    name, namespace = await _single_existing_or_fresh_dgd(
        kubectl, dynamo_deployment_namespace
    )
    faults = request.getfixturevalue("faults")
    async with (
        faults.inject(
            "cluster.rbac.revoke",
            target=_d111_rbac_target(list_owner),
            api_group="nvidia.com",
            resource="dynamographdeployments",
            verb="list",
        ),
        faults.inject(
            "cluster.rbac.revoke",
            target=_d111_rbac_target(watch_owner),
            api_group="nvidia.com",
            resource="dynamographdeployments",
            verb="watch",
        ),
    ):
        async with faults.inject(
            "operator.kill",
            target={"selector": _OPERATOR_SELECTOR, "ns": _OPERATOR_NAMESPACE},
        ):
            pass
        await asyncio.sleep(20.0)
        logs = await kubectl.run(
            "logs",
            "-n",
            _OPERATOR_NAMESPACE,
            "-l",
            _OPERATOR_SELECTOR,
            "--tail=200",
            check=False,
        )
        text = (logs.stdout + logs.stderr).lower()
        assert "forbidden" in text or "cannot list" in text or "cannot watch" in text, (
            "D111: operator logs did not show DGD list/watch RBAC denial while "
            "both verbs were revoked after restart"
        )
    await kubectl.run(
        "wait",
        "-n",
        _OPERATOR_NAMESPACE,
        "-l",
        _OPERATOR_SELECTOR,
        "deployment",
        "--for=condition=Available",
        "--timeout=90s",
        check=True,
    )
    await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)


async def _single_existing_or_fresh_dgd(
    kubectl: KubectlClient, namespace: str
) -> tuple[str, str]:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        "-n",
        namespace,
        "-o",
        "jsonpath={.items[*].metadata.name}",
        check=False,
    )
    names = result.stdout.split() if result.returncode == 0 else []
    if len(names) == 1:
        return names[0], namespace
    if len(names) > 1:
        pytest.skip(
            f"D111 requires at most one existing DGD in {namespace}; found {names!r}"
        )
    return await _d111_apply_fresh_dgd(kubectl, namespace)


# D112

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_OBSERVE_REVOKED_S = 45.0


@dataclass(frozen=True, slots=True)
class _d112_RbacOwner:
    scope: Literal["role", "clusterrole"]
    name: str
    namespace: str | None

    @property
    def label(self) -> str:
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


async def _d112_operator_service_account(kubectl: KubectlClient) -> str:
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
    deployments = orjson.loads(result.stdout or b"{}").get("items", [])
    if len(deployments) != 1:
        names = [
            item.get("metadata", {}).get("name", "<unnamed>") for item in deployments
        ]
        pytest.skip(
            "Dynamo RBAC chaos requires exactly one operator deployment; found "
            f"{', '.join(names) if names else '<none>'}"
        )
    return str(
        deployments[0]["spec"]["template"]["spec"].get("serviceAccountName", "default")
    )


async def _d112_find_unique_operator_rbac_owner(
    kubectl: KubectlClient,
    *,
    api_group: str,
    resource: str,
    verb: str,
    case_id: str,
) -> _d112_RbacOwner:
    service_account = await _d112_operator_service_account(kubectl)
    candidates: list[_d112_RbacOwner] = []
    inspected: list[str] = []
    for scope, name, namespace in await _d112_operator_bound_role_refs(
        kubectl, service_account
    ):
        inspected.append(f"{scope}/{namespace + '/' if namespace else ''}{name}")
        body = await _d112_load_rbac(kubectl, scope, name, namespace)
        if body is None:
            continue
        if _d112_has_exact_rule(body.get("rules") or [], api_group, resource, verb):
            candidates.append(
                _d112_RbacOwner(scope=scope, name=name, namespace=namespace)
            )
    if len(candidates) != 1:
        pytest.skip(
            f"{case_id} requires exactly one operator-bound RBAC rule granting "
            f"{verb!r} on {resource!r} apiGroup={api_group!r}; candidates="
            f"{', '.join(c.label for c in candidates) or '<none>'}; inspected="
            f"{', '.join(inspected) if inspected else '<none>'}"
        )
    return candidates[0]


async def _d112_operator_bound_role_refs(
    kubectl: KubectlClient,
    service_account: str,
) -> list[tuple[Literal["role", "clusterrole"], str, str | None]]:
    refs: list[tuple[Literal["role", "clusterrole"], str, str | None]] = []
    for binding_kind, namespaced in (
        ("rolebinding", True),
        ("clusterrolebinding", False),
    ):
        args = ["get", binding_kind, "-o", "json"]
        if namespaced:
            args.insert(2, "-n")
            args.insert(3, _OPERATOR_NAMESPACE)
        result = await kubectl.run(*args, check=True)
        for binding in orjson.loads(result.stdout or b"{}").get("items", []):
            if not _d112_has_operator_subject(
                binding.get("subjects") or [], service_account
            ):
                continue
            role_ref = binding.get("roleRef") or {}
            scope = str(role_ref.get("kind", "")).lower()
            if scope not in {"role", "clusterrole"}:
                continue
            namespace = (
                binding.get("metadata", {}).get("namespace")
                if scope == "role"
                else None
            )
            refs.append((scope, str(role_ref.get("name", "")), namespace))
    return refs


def _d112_has_operator_subject(
    subjects: list[dict[str, Any]], service_account: str
) -> bool:
    return any(
        subject.get("kind") == "ServiceAccount"
        and subject.get("name") == service_account
        and subject.get("namespace") == _OPERATOR_NAMESPACE
        for subject in subjects
    )


async def _d112_load_rbac(
    kubectl: KubectlClient,
    scope: Literal["role", "clusterrole"],
    name: str,
    namespace: str | None,
) -> dict[str, Any] | None:
    args = ["get", scope, name]
    if namespace is not None:
        args.extend(["-n", namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        return None
    return dict(orjson.loads(result.stdout or b"{}"))


def _d112_has_exact_rule(
    rules: list[dict[str, Any]],
    api_group: str,
    resource: str,
    verb: str,
) -> bool:
    for rule in rules:
        if "*" in (rule.get("apiGroups") or []):
            continue
        if "*" in (rule.get("resources") or []):
            continue
        if "*" in (rule.get("verbs") or []):
            continue
        if (
            api_group in (rule.get("apiGroups") or [])
            and resource in (rule.get("resources") or [])
            and verb in (rule.get("verbs") or [])
        ):
            return True
    return False


def _d112_rbac_target(owner: _d112_RbacOwner) -> dict[str, str]:
    target = {"scope": owner.scope, "name": owner.name}
    if owner.namespace is not None:
        target["ns"] = owner.namespace
    return target


async def _d112_apply_fresh_dgd(
    kubectl: KubectlClient, namespace: str
) -> tuple[str, str]:
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=namespace,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()
    await kubectl.apply(deployer.generate_manifest(), namespace=namespace)
    return name, namespace


async def _d112_observe_not_successful(
    kubectl: KubectlClient,
    name: str,
    namespace: str,
    *,
    case_id: str,
    timeout_s: float = _OBSERVE_REVOKED_S,
) -> str | None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_state: str | None = None
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            body = orjson.loads(result.stdout or b"{}")
            last_state = body.get("status", {}).get("state")
            assert last_state != "successful", (
                f"{case_id}: DGD {namespace}/{name} reached successful while "
                "required operator RBAC was revoked"
            )
        await asyncio.sleep(2.0)
    return last_state


async def _d112_delete_dgd(kubectl: KubectlClient, name: str, namespace: str) -> None:
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


async def test_d112_child_deployment_create_rbac_revoked_before_apply(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    owner = await _d112_find_unique_operator_rbac_owner(
        kubectl, api_group="apps", resource="deployments", verb="create", case_id="D112"
    )
    faults = request.getfixturevalue("faults")
    name = ""
    namespace = dynamo_deployment_namespace
    try:
        async with faults.inject(
            "cluster.rbac.revoke",
            target=_d112_rbac_target(owner),
            api_group="apps",
            resource="deployments",
            verb="create",
        ):
            name, namespace = await _d112_apply_fresh_dgd(kubectl, namespace)
            await _d112_observe_not_successful(kubectl, name, namespace, case_id="D112")
            count = await _count_children(kubectl, namespace, name, "deployment")
            assert count == 0, (
                f"D112: operator created {count} child Deployment(s) without create RBAC"
            )
        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)
    finally:
        if name:
            await _d112_delete_dgd(kubectl, name, namespace)


async def _count_children(
    kubectl: KubectlClient, namespace: str, name: str, kind: str
) -> int:
    result = await kubectl.run(
        "get",
        kind,
        "-n",
        namespace,
        "-l",
        f"nvidia.com/dynamographdeployment={name}",
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return 0
    return len(orjson.loads(result.stdout or b"{}").get("items", []))


# D113

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_OBSERVE_REVOKED_S = 45.0


@dataclass(frozen=True, slots=True)
class _d113_RbacOwner:
    scope: Literal["role", "clusterrole"]
    name: str
    namespace: str | None

    @property
    def label(self) -> str:
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


async def _d113_operator_service_account(kubectl: KubectlClient) -> str:
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
    deployments = orjson.loads(result.stdout or b"{}").get("items", [])
    if len(deployments) != 1:
        names = [
            item.get("metadata", {}).get("name", "<unnamed>") for item in deployments
        ]
        pytest.skip(
            "Dynamo RBAC chaos requires exactly one operator deployment; found "
            f"{', '.join(names) if names else '<none>'}"
        )
    return str(
        deployments[0]["spec"]["template"]["spec"].get("serviceAccountName", "default")
    )


async def _d113_find_unique_operator_rbac_owner(
    kubectl: KubectlClient,
    *,
    api_group: str,
    resource: str,
    verb: str,
    case_id: str,
) -> _d113_RbacOwner:
    service_account = await _d113_operator_service_account(kubectl)
    candidates: list[_d113_RbacOwner] = []
    inspected: list[str] = []
    for scope, name, namespace in await _d113_operator_bound_role_refs(
        kubectl, service_account
    ):
        inspected.append(f"{scope}/{namespace + '/' if namespace else ''}{name}")
        body = await _d113_load_rbac(kubectl, scope, name, namespace)
        if body is None:
            continue
        if _d113_has_exact_rule(body.get("rules") or [], api_group, resource, verb):
            candidates.append(
                _d113_RbacOwner(scope=scope, name=name, namespace=namespace)
            )
    if len(candidates) != 1:
        pytest.skip(
            f"{case_id} requires exactly one operator-bound RBAC rule granting "
            f"{verb!r} on {resource!r} apiGroup={api_group!r}; candidates="
            f"{', '.join(c.label for c in candidates) or '<none>'}; inspected="
            f"{', '.join(inspected) if inspected else '<none>'}"
        )
    return candidates[0]


async def _d113_operator_bound_role_refs(
    kubectl: KubectlClient,
    service_account: str,
) -> list[tuple[Literal["role", "clusterrole"], str, str | None]]:
    refs: list[tuple[Literal["role", "clusterrole"], str, str | None]] = []
    for binding_kind, namespaced in (
        ("rolebinding", True),
        ("clusterrolebinding", False),
    ):
        args = ["get", binding_kind, "-o", "json"]
        if namespaced:
            args.insert(2, "-n")
            args.insert(3, _OPERATOR_NAMESPACE)
        result = await kubectl.run(*args, check=True)
        for binding in orjson.loads(result.stdout or b"{}").get("items", []):
            if not _d113_has_operator_subject(
                binding.get("subjects") or [], service_account
            ):
                continue
            role_ref = binding.get("roleRef") or {}
            scope = str(role_ref.get("kind", "")).lower()
            if scope not in {"role", "clusterrole"}:
                continue
            namespace = (
                binding.get("metadata", {}).get("namespace")
                if scope == "role"
                else None
            )
            refs.append((scope, str(role_ref.get("name", "")), namespace))
    return refs


def _d113_has_operator_subject(
    subjects: list[dict[str, Any]], service_account: str
) -> bool:
    return any(
        subject.get("kind") == "ServiceAccount"
        and subject.get("name") == service_account
        and subject.get("namespace") == _OPERATOR_NAMESPACE
        for subject in subjects
    )


async def _d113_load_rbac(
    kubectl: KubectlClient,
    scope: Literal["role", "clusterrole"],
    name: str,
    namespace: str | None,
) -> dict[str, Any] | None:
    args = ["get", scope, name]
    if namespace is not None:
        args.extend(["-n", namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        return None
    return dict(orjson.loads(result.stdout or b"{}"))


def _d113_has_exact_rule(
    rules: list[dict[str, Any]],
    api_group: str,
    resource: str,
    verb: str,
) -> bool:
    for rule in rules:
        if "*" in (rule.get("apiGroups") or []):
            continue
        if "*" in (rule.get("resources") or []):
            continue
        if "*" in (rule.get("verbs") or []):
            continue
        if (
            api_group in (rule.get("apiGroups") or [])
            and resource in (rule.get("resources") or [])
            and verb in (rule.get("verbs") or [])
        ):
            return True
    return False


def _d113_rbac_target(owner: _d113_RbacOwner) -> dict[str, str]:
    target = {"scope": owner.scope, "name": owner.name}
    if owner.namespace is not None:
        target["ns"] = owner.namespace
    return target


async def _d113_apply_fresh_dgd(
    kubectl: KubectlClient, namespace: str
) -> tuple[str, str]:
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=namespace,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()
    await kubectl.apply(deployer.generate_manifest(), namespace=namespace)
    return name, namespace


async def _d113_observe_not_successful(
    kubectl: KubectlClient,
    name: str,
    namespace: str,
    *,
    case_id: str,
    timeout_s: float = _OBSERVE_REVOKED_S,
) -> str | None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_state: str | None = None
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            body = orjson.loads(result.stdout or b"{}")
            last_state = body.get("status", {}).get("state")
            assert last_state != "successful", (
                f"{case_id}: DGD {namespace}/{name} reached successful while "
                "required operator RBAC was revoked"
            )
        await asyncio.sleep(2.0)
    return last_state


async def _d113_delete_dgd(kubectl: KubectlClient, name: str, namespace: str) -> None:
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


async def test_d113_child_service_create_rbac_revoked_before_apply(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    owner = await _d113_find_unique_operator_rbac_owner(
        kubectl, api_group="", resource="services", verb="create", case_id="D113"
    )
    faults = request.getfixturevalue("faults")
    name = ""
    namespace = dynamo_deployment_namespace
    try:
        async with faults.inject(
            "cluster.rbac.revoke",
            target=_d113_rbac_target(owner),
            api_group="",
            resource="services",
            verb="create",
        ):
            name, namespace = await _d113_apply_fresh_dgd(kubectl, namespace)
            await _d113_observe_not_successful(kubectl, name, namespace, case_id="D113")
            count = await _d113_count_children(kubectl, namespace, name, "service")
            assert count == 0, (
                f"D113: operator created {count} child Service(s) without create RBAC"
            )
        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)
    finally:
        if name:
            await _d113_delete_dgd(kubectl, name, namespace)


async def _d113_count_children(
    kubectl: KubectlClient, namespace: str, name: str, kind: str
) -> int:
    result = await kubectl.run(
        "get",
        kind,
        "-n",
        namespace,
        "-l",
        f"nvidia.com/dynamographdeployment={name}",
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return 0
    return len(orjson.loads(result.stdout or b"{}").get("items", []))


# D114

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_OBSERVE_REVOKED_S = 45.0


@dataclass(frozen=True, slots=True)
class _d114_RbacOwner:
    scope: Literal["role", "clusterrole"]
    name: str
    namespace: str | None

    @property
    def label(self) -> str:
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


async def _d114_operator_service_account(kubectl: KubectlClient) -> str:
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
    deployments = orjson.loads(result.stdout or b"{}").get("items", [])
    if len(deployments) != 1:
        names = [
            item.get("metadata", {}).get("name", "<unnamed>") for item in deployments
        ]
        pytest.skip(
            "Dynamo RBAC chaos requires exactly one operator deployment; found "
            f"{', '.join(names) if names else '<none>'}"
        )
    return str(
        deployments[0]["spec"]["template"]["spec"].get("serviceAccountName", "default")
    )


async def _d114_find_unique_operator_rbac_owner(
    kubectl: KubectlClient,
    *,
    api_group: str,
    resource: str,
    verb: str,
    case_id: str,
) -> _d114_RbacOwner:
    service_account = await _d114_operator_service_account(kubectl)
    candidates: list[_d114_RbacOwner] = []
    inspected: list[str] = []
    for scope, name, namespace in await _d114_operator_bound_role_refs(
        kubectl, service_account
    ):
        inspected.append(f"{scope}/{namespace + '/' if namespace else ''}{name}")
        body = await _d114_load_rbac(kubectl, scope, name, namespace)
        if body is None:
            continue
        if _d114_has_exact_rule(body.get("rules") or [], api_group, resource, verb):
            candidates.append(
                _d114_RbacOwner(scope=scope, name=name, namespace=namespace)
            )
    if len(candidates) != 1:
        pytest.skip(
            f"{case_id} requires exactly one operator-bound RBAC rule granting "
            f"{verb!r} on {resource!r} apiGroup={api_group!r}; candidates="
            f"{', '.join(c.label for c in candidates) or '<none>'}; inspected="
            f"{', '.join(inspected) if inspected else '<none>'}"
        )
    return candidates[0]


async def _d114_operator_bound_role_refs(
    kubectl: KubectlClient,
    service_account: str,
) -> list[tuple[Literal["role", "clusterrole"], str, str | None]]:
    refs: list[tuple[Literal["role", "clusterrole"], str, str | None]] = []
    for binding_kind, namespaced in (
        ("rolebinding", True),
        ("clusterrolebinding", False),
    ):
        args = ["get", binding_kind, "-o", "json"]
        if namespaced:
            args.insert(2, "-n")
            args.insert(3, _OPERATOR_NAMESPACE)
        result = await kubectl.run(*args, check=True)
        for binding in orjson.loads(result.stdout or b"{}").get("items", []):
            if not _d114_has_operator_subject(
                binding.get("subjects") or [], service_account
            ):
                continue
            role_ref = binding.get("roleRef") or {}
            scope = str(role_ref.get("kind", "")).lower()
            if scope not in {"role", "clusterrole"}:
                continue
            namespace = (
                binding.get("metadata", {}).get("namespace")
                if scope == "role"
                else None
            )
            refs.append((scope, str(role_ref.get("name", "")), namespace))
    return refs


def _d114_has_operator_subject(
    subjects: list[dict[str, Any]], service_account: str
) -> bool:
    return any(
        subject.get("kind") == "ServiceAccount"
        and subject.get("name") == service_account
        and subject.get("namespace") == _OPERATOR_NAMESPACE
        for subject in subjects
    )


async def _d114_load_rbac(
    kubectl: KubectlClient,
    scope: Literal["role", "clusterrole"],
    name: str,
    namespace: str | None,
) -> dict[str, Any] | None:
    args = ["get", scope, name]
    if namespace is not None:
        args.extend(["-n", namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        return None
    return dict(orjson.loads(result.stdout or b"{}"))


def _d114_has_exact_rule(
    rules: list[dict[str, Any]],
    api_group: str,
    resource: str,
    verb: str,
) -> bool:
    for rule in rules:
        if "*" in (rule.get("apiGroups") or []):
            continue
        if "*" in (rule.get("resources") or []):
            continue
        if "*" in (rule.get("verbs") or []):
            continue
        if (
            api_group in (rule.get("apiGroups") or [])
            and resource in (rule.get("resources") or [])
            and verb in (rule.get("verbs") or [])
        ):
            return True
    return False


def _d114_rbac_target(owner: _d114_RbacOwner) -> dict[str, str]:
    target = {"scope": owner.scope, "name": owner.name}
    if owner.namespace is not None:
        target["ns"] = owner.namespace
    return target


async def _d114_apply_fresh_dgd(
    kubectl: KubectlClient, namespace: str
) -> tuple[str, str]:
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=namespace,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()
    await kubectl.apply(deployer.generate_manifest(), namespace=namespace)
    return name, namespace


async def _d114_observe_not_successful(
    kubectl: KubectlClient,
    name: str,
    namespace: str,
    *,
    case_id: str,
    timeout_s: float = _OBSERVE_REVOKED_S,
) -> str | None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_state: str | None = None
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            body = orjson.loads(result.stdout or b"{}")
            last_state = body.get("status", {}).get("state")
            assert last_state != "successful", (
                f"{case_id}: DGD {namespace}/{name} reached successful while "
                "required operator RBAC was revoked"
            )
        await asyncio.sleep(2.0)
    return last_state


async def _d114_delete_dgd(kubectl: KubectlClient, name: str, namespace: str) -> None:
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


async def test_d114_configmap_create_rbac_revoked_before_apply(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    owner = await _d114_find_unique_operator_rbac_owner(
        kubectl, api_group="", resource="configmaps", verb="create", case_id="D114"
    )
    faults = request.getfixturevalue("faults")
    name = ""
    namespace = dynamo_deployment_namespace
    try:
        async with faults.inject(
            "cluster.rbac.revoke",
            target=_d114_rbac_target(owner),
            api_group="",
            resource="configmaps",
            verb="create",
        ):
            name, namespace = await _d114_apply_fresh_dgd(kubectl, namespace)
            await _d114_observe_not_successful(kubectl, name, namespace, case_id="D114")
            count = await _d114_count_children(kubectl, namespace, name, "configmap")
            assert count == 0, (
                f"D114: operator created {count} child ConfigMap(s) without create RBAC"
            )
        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)
    finally:
        if name:
            await _d114_delete_dgd(kubectl, name, namespace)


async def _d114_count_children(
    kubectl: KubectlClient, namespace: str, name: str, kind: str
) -> int:
    result = await kubectl.run(
        "get",
        kind,
        "-n",
        namespace,
        "-l",
        f"nvidia.com/dynamographdeployment={name}",
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return 0
    return len(orjson.loads(result.stdout or b"{}").get("items", []))


# D115

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_CHILD_KINDS = ("deployment", "service", "configmap", "role", "rolebinding")


async def test_d115_operator_kill_during_dgd_delete_cleans_orphans(
    faults: Any,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=dynamo_deployment_namespace,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()
    namespace = dynamo_deployment_namespace
    try:
        await kubectl.apply(deployer.generate_manifest(), namespace=namespace)
        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)
        before = await _child_names(kubectl, namespace, name)
        if not any(before.values()):
            pytest.skip(
                f"D115 requires child resources labelled with DGD {namespace}/{name}; found none"
            )

        await kubectl.run(
            "delete",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "--wait=false",
            check=True,
        )
        async with faults.inject(
            "operator.kill",
            target={"selector": _OPERATOR_SELECTOR, "ns": _OPERATOR_NAMESPACE},
        ):
            pass

        await kubectl.run(
            "wait",
            "-n",
            _OPERATOR_NAMESPACE,
            "-l",
            _OPERATOR_SELECTOR,
            "deployment",
            "--for=condition=Available",
            "--timeout=90s",
            check=True,
        )
        await _wait_dgd_gone(kubectl, name, namespace, timeout_s=240.0)
        leftovers = await _child_names(kubectl, namespace, name)
        assert not any(leftovers.values()), (
            f"D115: child resources remained after operator-kill delete recovery: {leftovers!r}"
        )
    finally:
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


async def _wait_dgd_gone(
    kubectl: KubectlClient, name: str, namespace: str, *, timeout_s: float
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get", "dynamographdeployment", name, "-n", namespace, check=False
        )
        if result.returncode != 0:
            return
        await asyncio.sleep(2.0)
    raise AssertionError(
        f"D115: DGD {namespace}/{name} still exists after delete recovery"
    )


async def _child_names(
    kubectl: KubectlClient, namespace: str, dgd_name: str
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for kind in _CHILD_KINDS:
        result = await kubectl.run(
            "get",
            kind,
            "-n",
            namespace,
            "-l",
            f"nvidia.com/dynamographdeployment={dgd_name}",
            "-o",
            "json",
            check=False,
        )
        if result.returncode != 0:
            out[kind] = []
            continue
        items = orjson.loads(result.stdout or b"{}").get("items", [])
        out[kind] = [str(item.get("metadata", {}).get("name", "")) for item in items]
    return out


# D116

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_NAMESPACE = "d116-patch-operator-down"
_SUCCESS_TIMEOUT_S = 600.0


async def test_d116_patch_while_operator_down_reconciles_latest_generation(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures operator and CRD exist
) -> None:
    """Scale the operator down, patch the DGD spec, restore, and assert catch-up."""
    deployment = await _single_operator_deployment(kubectl)
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=_NAMESPACE,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()

    try:
        await kubectl.apply(deployer.generate_manifest(), namespace=_NAMESPACE)
        try:
            await wait_for_dgd_state(
                kubectl,
                name,
                _NAMESPACE,
                "successful",
                timeout=_SUCCESS_TIMEOUT_S,
            )
        except TimeoutError as exc:
            status = await _status_snapshot(kubectl, name)
            pytest.skip(
                "D116 requires a baseline DGD to reach state='successful' before "
                f"operator-down patching; status={status!r}; error={exc}"
            )

        try:
            await _scale_operator(kubectl, deployment, replicas=0)
            await _d116_wait_operator_available(kubectl, available=False, timeout=90.0)

            patch = {"spec": {"services": {"Frontend": {"replicas": 0}}}}
            await kubectl.run(
                "patch",
                "dynamographdeployment",
                name,
                "-n",
                _NAMESPACE,
                "--type=merge",
                f"-p={orjson.dumps(patch).decode()}",
                check=True,
            )
            patched = await _read_dgd(kubectl, name)
            patched_generation = patched["metadata"]["generation"]
        finally:
            await _scale_operator(kubectl, deployment, replicas=deployment["replicas"])
            if deployment["replicas"] > 0:
                await _d116_wait_operator_available(
                    kubectl, available=True, timeout=180.0
                )

        await wait_for_dgd_state(
            kubectl,
            name,
            _NAMESPACE,
            "successful",
            timeout=_SUCCESS_TIMEOUT_S,
        )
        final_dgd = await _read_dgd(kubectl, name)
        assert final_dgd["metadata"]["generation"] == patched_generation
        assert final_dgd["status"].get("observedGeneration") == patched_generation, (
            "D116: operator restored after an offline patch but did not observe "
            f"latest generation {patched_generation}; status={final_dgd.get('status')!r}"
        )
        assert final_dgd["spec"]["services"]["Frontend"].get("replicas") == 0
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _single_operator_deployment(kubectl: KubectlClient) -> dict[str, Any]:
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
    items = orjson.loads(result.stdout).get("items", [])
    if len(items) != 1:
        names = [item.get("metadata", {}).get("name", "<unnamed>") for item in items]
        pytest.skip(
            "D116 requires exactly one Dynamo operator Deployment; found "
            f"{names if names else '<none>'}"
        )
    item = items[0]
    return {
        "name": item["metadata"]["name"],
        "replicas": int(item.get("spec", {}).get("replicas") or 1),
    }


async def _scale_operator(
    kubectl: KubectlClient,
    deployment: dict[str, Any],
    *,
    replicas: int,
) -> None:
    await kubectl.run(
        "scale",
        "deployment",
        deployment["name"],
        "-n",
        _OPERATOR_NAMESPACE,
        f"--replicas={replicas}",
        check=True,
    )


async def _d116_wait_operator_available(
    kubectl: KubectlClient,
    *,
    available: bool,
    timeout: float,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
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
            items = orjson.loads(result.stdout).get("items", [])
            ready = sum(
                int(item.get("status", {}).get("availableReplicas") or 0)
                for item in items
            )
            if (ready > 0) is available:
                return
        await asyncio.sleep(1.0)
    pytest.fail(f"D116: operator available={available} was not observed in {timeout}s")


async def _read_dgd(kubectl: KubectlClient, name: str) -> dict[str, Any]:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        _NAMESPACE,
        "-o",
        "json",
        check=True,
    )
    return orjson.loads(result.stdout)


async def _status_snapshot(kubectl: KubectlClient, name: str) -> str:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        _NAMESPACE,
        "-o",
        "jsonpath={.status}",
        check=False,
    )
    return result.stdout.strip() or result.stderr.strip()


# D117

_NAMESPACE = "d117-rapid-spec-patch"
_SUCCESS_TIMEOUT_S = 600.0


async def test_d117_rapid_spec_patches_converge_on_final_spec(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures operator and CRD exist
) -> None:
    """Patch frontend replicas several times and assert final observedGeneration."""
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=_NAMESPACE,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()

    try:
        await kubectl.apply(deployer.generate_manifest(), namespace=_NAMESPACE)
        try:
            await wait_for_dgd_state(
                kubectl,
                name,
                _NAMESPACE,
                "successful",
                timeout=_SUCCESS_TIMEOUT_S,
            )
        except TimeoutError as exc:
            status = await _d117_status_snapshot(kubectl, name)
            pytest.skip(
                "D117 requires a baseline DGD to reach state='successful' before "
                f"rapid spec patching; status={status!r}; error={exc}"
            )

        final_replicas = 1
        for replicas in (2, 0, final_replicas):
            await _patch_frontend_replicas(kubectl, name=name, replicas=replicas)
            await asyncio.sleep(0.25)

        patched = await _d117_read_dgd(kubectl, name)
        final_generation = patched["metadata"]["generation"]
        assert patched["spec"]["services"]["Frontend"].get("replicas") == final_replicas

        await wait_for_dgd_state(
            kubectl,
            name,
            _NAMESPACE,
            "successful",
            timeout=_SUCCESS_TIMEOUT_S,
        )
        final = await _d117_read_dgd(kubectl, name)
        assert final["status"].get("observedGeneration") == final_generation, (
            "D117: rapid spec patches did not converge to the final generation; "
            f"generation={final_generation}, status={final.get('status')!r}"
        )
        assert final["spec"]["services"]["Frontend"].get("replicas") == final_replicas
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _patch_frontend_replicas(
    kubectl: KubectlClient,
    *,
    name: str,
    replicas: int,
) -> None:
    patch = {"spec": {"services": {"Frontend": {"replicas": replicas}}}}
    await kubectl.run(
        "patch",
        "dynamographdeployment",
        name,
        "-n",
        _NAMESPACE,
        "--type=merge",
        f"-p={orjson.dumps(patch).decode()}",
        check=True,
    )


async def _d117_read_dgd(kubectl: KubectlClient, name: str) -> dict[str, Any]:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        _NAMESPACE,
        "-o",
        "json",
        check=True,
    )
    return orjson.loads(result.stdout)


async def _d117_status_snapshot(kubectl: KubectlClient, name: str) -> str:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        _NAMESPACE,
        "-o",
        "jsonpath={.status}",
        check=False,
    )
    return result.stdout.strip() or result.stderr.strip()


# D118

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
        await _d118_scale_deployment(kubectl, target, replicas=0)
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
        await _d118_scale_deployment(
            kubectl, target, replicas=target.deployment_replicas
        )
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
            if not _d118_webhook_validates_dgd(webhook.get("rules", []) or []):
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


def _d118_webhook_validates_dgd(rules: list[dict[str, Any]]) -> bool:
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
    service = await _d118_get_json(
        kubectl, "service", service_name, namespace=namespace
    )
    selector = service.get("spec", {}).get("selector") or {}
    if not selector:
        return None
    deployments = await _d118_get_json(kubectl, "deployment", namespace=namespace)
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
    data = await _d118_get_json(
        kubectl, "validatingwebhookconfiguration", target.config_name
    )
    for index, webhook in enumerate(data.get("webhooks", []) or []):
        if webhook.get("name") == target.webhook_name:
            return index
    raise AssertionError(f"D118: webhook {target.webhook_name!r} disappeared")


async def _d118_scale_deployment(
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
        deployment = await _d118_get_json(
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


async def _d118_get_json(
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


# D119

_NAMESPACE = "d119-webhook-fail-open"
_NAME = "d119-invalid"
_WEBHOOK_GROUP = "nvidia.com"


@dataclass(frozen=True)
class _d119_WebhookTarget:
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
    target = await _d119_find_webhook_target(kubectl)
    await kubectl.create_namespace(_NAMESPACE)
    try:
        await _d119_patch_failure_policy(kubectl, target, policy="Ignore")
        await _d119_scale_deployment(kubectl, target, replicas=0)
        await _d119_wait_deployment_replicas(kubectl, target, replicas=0, timeout=90.0)

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

        dgd = await _d119_read_dgd(kubectl)
        assert dgd["metadata"]["name"] == _NAME
        state = await _observe_state(kubectl, timeout=45.0)
        assert state != "successful", (
            "D119: fail-open admitted a webhook-invalid DGD and the operator "
            "reported state='successful' instead of surfacing validation failure"
        )
    finally:
        await _d119_scale_deployment(
            kubectl, target, replicas=target.deployment_replicas
        )
        if target.deployment_replicas > 0:
            await _d119_wait_deployment_replicas(
                kubectl,
                target,
                replicas=target.deployment_replicas,
                timeout=180.0,
            )
        await _d119_restore_failure_policy(kubectl, target)
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


async def _d119_read_dgd(kubectl: KubectlClient) -> dict[str, Any]:
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


async def _d119_find_webhook_target(kubectl: KubectlClient) -> _d119_WebhookTarget:
    result = await kubectl.run(
        "get",
        "validatingwebhookconfigurations",
        "-o",
        "json",
        check=True,
    )
    candidates: list[_d119_WebhookTarget] = []
    inspected: list[str] = []
    for config in orjson.loads(result.stdout).get("items", []):
        config_name = config.get("metadata", {}).get("name", "")
        for webhook in config.get("webhooks", []) or []:
            if not _d119_webhook_validates_dgd(webhook.get("rules", []) or []):
                continue
            service = webhook.get("clientConfig", {}).get("service") or {}
            namespace = service.get("namespace")
            name = service.get("name")
            if not namespace or not name:
                inspected.append(f"{config_name}/{webhook.get('name')}: no service")
                continue
            deployment = await _d119_deployment_for_service(kubectl, namespace, name)
            inspected.append(
                f"{config_name}/{webhook.get('name')} -> {namespace}/{name}"
            )
            if deployment is not None:
                candidates.append(
                    _d119_WebhookTarget(
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


def _d119_webhook_validates_dgd(rules: list[dict[str, Any]]) -> bool:
    for rule in rules:
        groups = rule.get("apiGroups") or []
        resources = rule.get("resources") or []
        if _WEBHOOK_GROUP in groups and any(
            str(resource).startswith("dynamographdeployments") for resource in resources
        ):
            return True
    return False


async def _d119_deployment_for_service(
    kubectl: KubectlClient,
    namespace: str,
    service_name: str,
) -> dict[str, Any] | None:
    service = await _d119_get_json(
        kubectl, "service", service_name, namespace=namespace
    )
    selector = service.get("spec", {}).get("selector") or {}
    if not selector:
        return None
    deployments = await _d119_get_json(kubectl, "deployment", namespace=namespace)
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


async def _d119_patch_failure_policy(
    kubectl: KubectlClient,
    target: _d119_WebhookTarget,
    *,
    policy: str,
) -> None:
    patch = [
        {
            "op": "add" if target.failure_policy is None else "replace",
            "path": f"/webhooks/{await _d119_webhook_index(kubectl, target)}/failurePolicy",
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


async def _d119_restore_failure_policy(
    kubectl: KubectlClient, target: _d119_WebhookTarget
) -> None:
    if target.failure_policy is None:
        patch = [
            {
                "op": "remove",
                "path": f"/webhooks/{await _d119_webhook_index(kubectl, target)}/failurePolicy",
            }
        ]
    else:
        patch = [
            {
                "op": "replace",
                "path": f"/webhooks/{await _d119_webhook_index(kubectl, target)}/failurePolicy",
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


async def _d119_webhook_index(
    kubectl: KubectlClient, target: _d119_WebhookTarget
) -> int:
    data = await _d119_get_json(
        kubectl, "validatingwebhookconfiguration", target.config_name
    )
    for index, webhook in enumerate(data.get("webhooks", []) or []):
        if webhook.get("name") == target.webhook_name:
            return index
    raise AssertionError(f"D119: webhook {target.webhook_name!r} disappeared")


async def _d119_scale_deployment(
    kubectl: KubectlClient,
    target: _d119_WebhookTarget,
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


async def _d119_wait_deployment_replicas(
    kubectl: KubectlClient,
    target: _d119_WebhookTarget,
    *,
    replicas: int,
    timeout: float,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        deployment = await _d119_get_json(
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


async def _d119_get_json(
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


# D120

_NAMESPACE = "d120-unserved-version"
_NAME = "d120-unserved"
_UNSERVED_VERSION = "v99alpha99"


async def test_d120_unserved_dgd_api_version_is_rejected(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures DGD CRD exists
) -> None:
    """Apply a DGD with a version not served by the CRD and assert rejection."""
    served_versions = await _served_dgd_versions(kubectl)
    if _UNSERVED_VERSION in served_versions:
        pytest.skip(
            f"D120 requires {_UNSERVED_VERSION!r} to be unserved; served={served_versions!r}"
        )

    await kubectl.create_namespace(_NAMESPACE)
    try:
        try:
            await kubectl.apply(
                orjson.dumps(_manifest()).decode(), namespace=_NAMESPACE
            )
        except RuntimeError as exc:
            message = str(exc).lower()
        else:
            pytest.fail(
                f"D120 expected nvidia.com/{_UNSERVED_VERSION} DGD to fail admission"
            )

        assert any(
            term in message
            for term in (
                "no matches for kind",
                "no kind",
                "not registered",
                "could not find",
                "unable to recognize",
            )
        ), f"D120 expected unserved-version error, got {message!r}"
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            _NAME,
            "-n",
            _NAMESPACE,
            check=False,
        )
        assert result.returncode != 0, "D120: unserved-version DGD was persisted"
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _served_dgd_versions(kubectl: KubectlClient) -> list[str]:
    result = await kubectl.run(
        "get",
        "crd",
        "dynamographdeployments.nvidia.com",
        "-o",
        "json",
        check=True,
    )
    crd = orjson.loads(result.stdout)
    return [
        version.get("name", "")
        for version in crd.get("spec", {}).get("versions", [])
        if version.get("served") is True
    ]


def _manifest() -> dict[str, object]:
    return {
        "apiVersion": f"nvidia.com/{_UNSERVED_VERSION}",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": _NAME, "namespace": _NAMESPACE},
        "spec": {"services": {}},
    }


# D121

_NAMESPACE = "d121-unknown-spec-field"
_NAME = "d121-unknown-field"
_UNKNOWN_FIELD = "definitelyUnknownD121Field"


async def test_d121_unknown_dgd_spec_field_is_rejected(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures CRD/webhook exist
) -> None:
    """Apply a DGD with an unknown spec key and assert strict admission failure."""
    await kubectl.create_namespace(_NAMESPACE)
    try:
        try:
            await kubectl.apply(
                orjson.dumps(_d121_manifest()).decode(), namespace=_NAMESPACE
            )
        except RuntimeError as exc:
            message = str(exc).lower()
        else:
            pytest.fail("D121 expected unknown DGD spec field to fail admission")

        assert _UNKNOWN_FIELD.lower() in message or "unknown field" in message, (
            "D121 expected admission error to name the unknown field; "
            f"message={message!r}"
        )
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            _NAME,
            "-n",
            _NAMESPACE,
            check=False,
        )
        assert result.returncode != 0, "D121: DGD with unknown spec field was persisted"
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


def _d121_manifest() -> dict[str, object]:
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": _NAME, "namespace": _NAMESPACE},
        "spec": {
            _UNKNOWN_FIELD: True,
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": f"nvcr.io/nvidia/ai-dynamo/vllm-runtime:{DYNAMO_VERSION}",
                        }
                    },
                }
            },
        },
    }


# D122

_NAMESPACE = "d122-invalid-component"


def _service(component_type: str) -> dict[str, Any]:
    return {
        "componentType": component_type,
        "replicas": 1,
        "extraPodSpec": {
            "mainContainer": {
                "image": f"nvcr.io/nvidia/ai-dynamo/vllm-runtime:{DYNAMO_VERSION}",
            }
        },
    }


@pytest.mark.parametrize(
    ("name", "services", "expected_terms"),
    [
        param(
            "d122-invalid-type",
            {
                "Frontend": _service("frontend"),
                "BadWorker": _service("not-a-dynamo-component"),
            },
            ("componenttype", "component type", "not-a-dynamo-component", "unsupported"),
            id="invalid-component-type",
        ),
        param(
            "d122-invalid-key",
            {
                "Frontend": _service("frontend"),
                "bad key with spaces": _service("worker"),
            },
            ("bad key with spaces", "service", "key", "metadata.name"),
            id="invalid-service-key",
        ),
    ],
)  # fmt: skip
async def test_d122_invalid_component_type_or_key_rejected(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures CRD/webhook exist
    name: str,
    services: dict[str, Any],
    expected_terms: tuple[str, ...],
) -> None:
    """Apply invalid component definitions and assert they do not persist."""
    await kubectl.create_namespace(_NAMESPACE)
    try:
        try:
            await kubectl.apply(
                orjson.dumps(_d122_manifest(name=name, services=services)).decode(),
                namespace=_NAMESPACE,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
        else:
            pytest.fail(f"D122 expected invalid component manifest {name!r} to fail")

        assert any(term in message for term in expected_terms), (
            "D122 expected admission error to identify invalid component type/key; "
            f"terms={expected_terms!r}, message={message!r}"
        )
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            _NAMESPACE,
            check=False,
        )
        assert result.returncode != 0, (
            f"D122: invalid DGD {_NAMESPACE}/{name} persisted"
        )
    finally:
        await kubectl.run(
            "delete",
            "dynamographdeployment",
            name,
            "-n",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )
        await kubectl.run(
            "delete",
            "namespace",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


def _d122_manifest(*, name: str, services: dict[str, Any]) -> dict[str, Any]:
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name, "namespace": _NAMESPACE},
        "spec": {"services": services},
    }


# D123

_NAMESPACE = "d123-status-condition-freshness"
_SUCCESS_TIMEOUT_S = 600.0


async def test_d123_status_conditions_fresh_after_spec_patch(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures operator and CRD exist
) -> None:
    """Patch a successful DGD and assert status conditions match latest generation."""
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=_NAMESPACE,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()

    try:
        await kubectl.apply(deployer.generate_manifest(), namespace=_NAMESPACE)
        try:
            await wait_for_dgd_state(
                kubectl,
                name,
                _NAMESPACE,
                "successful",
                timeout=_SUCCESS_TIMEOUT_S,
            )
        except TimeoutError as exc:
            status = await _d123_status_snapshot(kubectl, name)
            pytest.skip(
                "D123 requires a baseline DGD to reach state='successful' before "
                f"condition freshness checks; status={status!r}; error={exc}"
            )

        await _d123_patch_frontend_replicas(kubectl, name=name, replicas=0)
        patched = await _d123_read_dgd(kubectl, name)
        patched_generation = patched["metadata"]["generation"]

        await wait_for_dgd_state(
            kubectl,
            name,
            _NAMESPACE,
            "successful",
            timeout=_SUCCESS_TIMEOUT_S,
        )
        final = await _d123_read_dgd(kubectl, name)
        status = final.get("status", {})
        assert status.get("observedGeneration") == patched_generation, (
            "D123: status.observedGeneration is stale after spec patch; "
            f"generation={patched_generation}, status={status!r}"
        )

        conditions = status.get("conditions") or []
        if not conditions:
            pytest.skip(
                "D123 requires DynamoGraphDeployment status.conditions to be "
                "present; this operator exposes no conditions on a successful DGD"
            )
        stale = _stale_conditions(conditions, patched_generation)
        assert not stale, (
            "D123: status.conditions contain stale observedGeneration values after "
            f"generation {patched_generation}: {stale!r}"
        )
        future = _future_transition_conditions(conditions)
        assert not future, (
            f"D123: condition transition times are in the future: {future!r}"
        )
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _d123_patch_frontend_replicas(
    kubectl: KubectlClient,
    *,
    name: str,
    replicas: int,
) -> None:
    patch = {"spec": {"services": {"Frontend": {"replicas": replicas}}}}
    await kubectl.run(
        "patch",
        "dynamographdeployment",
        name,
        "-n",
        _NAMESPACE,
        "--type=merge",
        f"-p={orjson.dumps(patch).decode()}",
        check=True,
    )


async def _d123_read_dgd(kubectl: KubectlClient, name: str) -> dict[str, Any]:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        _NAMESPACE,
        "-o",
        "json",
        check=True,
    )
    return orjson.loads(result.stdout)


async def _d123_status_snapshot(kubectl: KubectlClient, name: str) -> str:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        _NAMESPACE,
        "-o",
        "jsonpath={.status}",
        check=False,
    )
    return result.stdout.strip() or result.stderr.strip()


def _stale_conditions(
    conditions: list[dict[str, Any]],
    generation: int,
) -> list[dict[str, Any]]:
    stale: list[dict[str, Any]] = []
    for condition in conditions:
        observed = condition.get("observedGeneration")
        if observed is not None and observed != generation:
            stale.append(condition)
    return stale


def _future_transition_conditions(
    conditions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    now = datetime.now(tz=timezone.utc)
    future: list[dict[str, Any]] = []
    for condition in conditions:
        timestamp = condition.get("lastTransitionTime")
        if not timestamp:
            continue
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if parsed > now:
            future.append(condition)
    return future


# D124-D127

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_CHILD_LABELS = (
    "nvidia.com/dynamo-graph-deployment-name",
    "nvidia.com/dynamographdeployment",
)
_SUCCESS_TIMEOUT_S = 600.0
_CHILD_RECREATE_TIMEOUT_S = 180.0
_D127_NAMESPACE = "d127-namespace-finalizer-delete"


@dataclass(frozen=True, slots=True)
class _DgdRef:
    """DynamoGraphDeployment identity used by reconcile assertions."""

    namespace: str
    name: str


@dataclass(frozen=True, slots=True)
class _ChildRef:
    """Owned child resource selected for a delete/recreate scenario."""

    kind: Literal["deployment", "service"]
    namespace: str
    name: str
    uid: str


async def test_d124_child_deployment_deleted_is_recreated_by_operator(
    kubectl: KubectlClient,
    dynamo_server,  # noqa: ANN001 - fixture provides a ready baseline DGD
    dynamo_deployment_namespace: str,
) -> None:
    """Delete one child Deployment and require the DGD controller to recreate it."""
    dgd = await _unique_dgd_in_namespace(
        kubectl, dynamo_deployment_namespace, case="D124"
    )
    child = await _unique_owned_child(kubectl, dgd, kind="deployment", case="D124")

    await kubectl.run(
        "delete",
        "deployment",
        child.name,
        "-n",
        child.namespace,
        "--wait=false",
        check=True,
    )
    recreated = await _wait_for_child_recreated(kubectl, child)
    await kubectl.run(
        "rollout",
        "status",
        "deployment",
        recreated.name,
        "-n",
        recreated.namespace,
        "--timeout=180s",
        check=True,
    )

    assert recreated.uid != child.uid, (
        f"D124: child Deployment {child.namespace}/{child.name} still has old "
        f"uid={child.uid!r} after delete; operator did not recreate it"
    )


async def test_d125_child_service_deleted_is_recreated_by_operator(
    kubectl: KubectlClient,
    dynamo_server,  # noqa: ANN001 - fixture provides a ready baseline DGD
    dynamo_deployment_namespace: str,
) -> None:
    """Delete one child Service and require the DGD controller to recreate it."""
    dgd = await _unique_dgd_in_namespace(
        kubectl, dynamo_deployment_namespace, case="D125"
    )
    child = await _unique_owned_child(kubectl, dgd, kind="service", case="D125")

    await kubectl.run(
        "delete",
        "service",
        child.name,
        "-n",
        child.namespace,
        "--wait=false",
        check=True,
    )
    recreated = await _wait_for_child_recreated(kubectl, child)

    assert recreated.uid != child.uid, (
        f"D125: child Service {child.namespace}/{child.name} still has old "
        f"uid={child.uid!r} after delete; operator did not recreate it"
    )


async def test_d126_operator_restart_with_stale_workqueue_converges(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_server,  # noqa: ANN001 - fixture provides a ready baseline DGD
    dynamo_deployment_namespace: str,
) -> None:
    """Queue repeated DGD metadata events, restart the operator, and assert convergence."""
    dgd = await _unique_dgd_in_namespace(
        kubectl, dynamo_deployment_namespace, case="D126"
    )
    before_children = await _owned_child_names(kubectl, dgd)
    if not any(before_children.values()):
        pytest.skip(
            f"D126 requires existing DGD-owned child resources for {dgd.namespace}/{dgd.name}"
        )

    for idx in range(8):
        await kubectl.run(
            "annotate",
            "dynamographdeployment",
            dgd.name,
            "-n",
            dgd.namespace,
            f"chaos.dynamo.nvidia.com/d126-workqueue-{idx}={idx}",
            "--overwrite",
            check=True,
        )

    async with faults.inject(
        "operator.kill",
        target={"selector": _OPERATOR_SELECTOR, "ns": _OPERATOR_NAMESPACE},
    ):
        pass

    await _d124_d127_wait_operator_available(kubectl, timeout=180.0)
    await wait_for_dgd_state(
        kubectl,
        dgd.name,
        dgd.namespace,
        "successful",
        timeout=_SUCCESS_TIMEOUT_S,
    )
    after_children = await _owned_child_names(kubectl, dgd)

    assert before_children == after_children, (
        "D126: operator restart after queued DGD events changed owned child set; "
        f"before={before_children!r}, after={after_children!r}"
    )


async def test_d127_namespace_delete_with_dgd_finalizer_completes(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures operator and CRD exist
) -> None:
    """Delete an isolated namespace while its DGD carries an operator finalizer."""
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=_D127_NAMESPACE,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()

    await kubectl.run(
        "delete",
        "namespace",
        _D127_NAMESPACE,
        "--wait=false",
        "--ignore-not-found",
        check=False,
    )
    await _wait_namespace_absent(kubectl, _D127_NAMESPACE, timeout=120.0)

    try:
        await kubectl.apply(deployer.generate_manifest(), namespace=_D127_NAMESPACE)
        finalizers = await _wait_for_dgd_finalizers(
            kubectl,
            name=name,
            namespace=_D127_NAMESPACE,
            timeout=90.0,
        )
        if not finalizers:
            pytest.skip(
                "D127 requires the Dynamo operator to add a DGD finalizer before "
                "namespace deletion; none appeared within 90s"
            )

        await kubectl.run(
            "delete",
            "namespace",
            _D127_NAMESPACE,
            "--wait=false",
            check=True,
        )
        await _wait_namespace_absent(kubectl, _D127_NAMESPACE, timeout=240.0)
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _D127_NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _unique_dgd_in_namespace(
    kubectl: KubectlClient,
    namespace: str,
    *,
    case: str,
) -> _DgdRef:
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
            f"{case} requires list/get access to DynamoGraphDeployments in "
            f"namespace {namespace!r}; kubectl stderr={result.stderr.strip()!r}"
        )
    items = orjson.loads(result.stdout or b"{}").get("items", [])
    if len(items) != 1:
        names = [item.get("metadata", {}).get("name", "<unnamed>") for item in items]
        pytest.skip(
            f"{case} requires exactly one DGD in namespace {namespace!r}; "
            f"found {names if names else '<none>'}"
        )
    return _DgdRef(namespace=namespace, name=items[0]["metadata"]["name"])


async def _unique_owned_child(
    kubectl: KubectlClient,
    dgd: _DgdRef,
    *,
    kind: Literal["deployment", "service"],
    case: str,
) -> _ChildRef:
    children = await _owned_children(kubectl, dgd, kind=kind)
    if not children:
        pytest.skip(
            f"{case} requires at least one DGD-owned child {kind} for "
            f"{dgd.namespace}/{dgd.name}; none were found by ownerReference or labels"
        )
    frontend = [child for child in children if child.name.endswith("-frontend")]
    candidates = frontend or children
    if len(candidates) != 1:
        names = [child.name for child in candidates]
        pytest.skip(
            f"{case} requires a unique child {kind} target; candidates={names!r}"
        )
    return candidates[0]


async def _owned_children(
    kubectl: KubectlClient,
    dgd: _DgdRef,
    *,
    kind: Literal["deployment", "service"],
) -> list[_ChildRef]:
    by_name: dict[str, _ChildRef] = {}
    for args in _child_list_arg_sets(kind, dgd.name):
        result = await kubectl.run(
            *args, "-n", dgd.namespace, "-o", "json", check=False
        )
        if result.returncode != 0 or not result.stdout.strip():
            continue
        for item in orjson.loads(result.stdout).get("items", []):
            metadata = item.get("metadata", {})
            name = metadata.get("name", "")
            uid = metadata.get("uid", "")
            if not name or not uid:
                continue
            if _owned_by_dgd(item, dgd.name) or _labeled_for_dgd(item, dgd.name):
                by_name[name] = _ChildRef(
                    kind=kind,
                    namespace=dgd.namespace,
                    name=name,
                    uid=uid,
                )
    return sorted(by_name.values(), key=lambda child: child.name)


def _child_list_arg_sets(kind: str, dgd_name: str) -> list[list[str]]:
    args = [["get", kind]]
    args.extend(["get", kind, "-l", f"{label}={dgd_name}"] for label in _CHILD_LABELS)
    return args


def _owned_by_dgd(item: dict[str, Any], dgd_name: str) -> bool:
    owners = item.get("metadata", {}).get("ownerReferences") or []
    return any(
        owner.get("kind") == "DynamoGraphDeployment" and owner.get("name") == dgd_name
        for owner in owners
        if isinstance(owner, dict)
    )


def _labeled_for_dgd(item: dict[str, Any], dgd_name: str) -> bool:
    labels = item.get("metadata", {}).get("labels") or {}
    return any(labels.get(label) == dgd_name for label in _CHILD_LABELS)


async def _wait_for_child_recreated(
    kubectl: KubectlClient,
    old: _ChildRef,
) -> _ChildRef:
    deadline = asyncio.get_running_loop().time() + _CHILD_RECREATE_TIMEOUT_S
    last_seen = "<not observed>"
    while True:
        result = await kubectl.run(
            "get",
            old.kind,
            old.name,
            "-n",
            old.namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            item = orjson.loads(result.stdout)
            uid = item.get("metadata", {}).get("uid", "")
            last_seen = uid or "<missing uid>"
            if uid and uid != old.uid:
                return _ChildRef(
                    kind=old.kind,
                    namespace=old.namespace,
                    name=old.name,
                    uid=uid,
                )
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"{old.kind} {old.namespace}/{old.name} was not recreated with a "
                f"new UID within {_CHILD_RECREATE_TIMEOUT_S}s after deletion; "
                f"old_uid={old.uid!r}, last_seen_uid={last_seen!r}"
            )
        await asyncio.sleep(2.0)


async def _owned_child_names(
    kubectl: KubectlClient, dgd: _DgdRef
) -> dict[str, list[str]]:
    names: dict[str, list[str]] = {}
    for kind in ("deployment", "service"):
        children = await _owned_children(kubectl, dgd, kind=kind)
        names[kind] = [child.name for child in children]
    return names


async def _d124_d127_wait_operator_available(
    kubectl: KubectlClient, *, timeout: float
) -> None:
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
            "D126: Dynamo operator deployment did not become Available after restart; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )


async def _wait_for_dgd_finalizers(
    kubectl: KubectlClient,
    *,
    name: str,
    namespace: str,
    timeout: float,
) -> list[str]:
    deadline = asyncio.get_running_loop().time() + timeout
    finalizers: list[str] = []
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            body = orjson.loads(result.stdout)
            finalizers = list(body.get("metadata", {}).get("finalizers") or [])
            if finalizers:
                return finalizers
        await asyncio.sleep(1.0)
    return finalizers


async def _wait_namespace_absent(
    kubectl: KubectlClient,
    namespace: str,
    *,
    timeout: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    last_phase = "<unobserved>"
    while True:
        result = await kubectl.run(
            "get", "namespace", namespace, "-o", "json", check=False
        )
        if result.returncode != 0:
            return
        if result.stdout.strip():
            body = orjson.loads(result.stdout)
            last_phase = body.get("status", {}).get("phase", "<missing phase>")
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail(
                f"D127: namespace {namespace!r} still exists after {timeout}s; "
                f"last_phase={last_phase!r}"
            )
        await asyncio.sleep(2.0)


# D128-D130

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
        manifest = _d128_d130_dgd_manifest(kubectl, namespace=_D128_NAMESPACE)
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
        await _d128_d130_scale_deployment(kubectl, deployment, replicas=0)
        await _wait_deployment_available_replicas(
            kubectl,
            deployment,
            replicas=0,
            timeout=90.0,
        )

        manifest = _d128_d130_dgd_manifest(kubectl, namespace=_D129_NAMESPACE)
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
        await _d128_d130_scale_deployment(
            kubectl, deployment, replicas=deployment.replicas
        )
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
    await _d128_d130_wait_operator_available(kubectl, timeout=180.0)
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
            if not _d128_d130_webhook_validates_dgd(webhook.get("rules") or []):
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


def _d128_d130_webhook_validates_dgd(rules: list[dict[str, Any]]) -> bool:
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
    service = await _d128_d130_get_json(
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

    deployments = await _d128_d130_get_json(
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


def _d128_d130_dgd_manifest(kubectl: KubectlClient, *, namespace: str) -> str:
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=namespace,
        api_version="v1alpha1",
    )
    return DynamoDeployer(kubectl, config).generate_manifest()


async def _d128_d130_scale_deployment(
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
        deployment = await _d128_d130_get_json(
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
        lease = await _d128_d130_get_json(
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


async def _d128_d130_wait_operator_available(
    kubectl: KubectlClient, *, timeout: float
) -> None:
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


async def _d128_d130_get_json(
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
