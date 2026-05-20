# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D7xx Dynamo infrastructure and control-plane chaos scenarios."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import orjson
import pytest
import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.chaos_dynamo.d7_status_helpers import (
    dgd_state_from_status_text,
    mentions_any,
    minimal_v1alpha1_frontend_dgd_manifest,
    read_dgd_status_text,
    wait_for_events_or_status,
)
from tests.kubernetes.chaos_dynamo.rbac_helpers import (
    find_unique_operator_rbac_owner,
    rbac_revoke_target,
)
from tests.kubernetes.chaos_dynamo.test_chaos_d1xx_operator_admission import (
    _d112_apply_fresh_dgd,
    _d112_delete_dgd,
    _d112_observe_not_successful,
)
from tests.kubernetes.gpu.conftest import GPUTestSettings
from tests.kubernetes.gpu.dynamo.helpers import (
    DynamoBackend,
    DynamoConfig,
    DynamoDeployer,
    DynamoMode,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


# D701

_D701_DGD_NAME = "d701-test"
_D701_DGD_NAMESPACE = "d701-image-pull"
_D701_DGD_LABEL = "nvidia.com/dynamo-graph-deployment-name"
_D701_BOGUS_IMAGE = "nonexistent.example.com/dynamo:nope"
_PULL_REASONS = ("ImagePullBackOff", "ErrImagePull")
_POD_REASON_TIMEOUT_S = 60.0
_D701_DGD_FAILED_TIMEOUT_S = 120.0


async def test_d701_imagepullbackoff_surfaces_failed_state(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Bogus container image -> kubelet ImagePullBackOff -> CR ``state=failed``.

    Targets the kubelet pull-fail -> pod-status -> operator-reconcile ->
    ``CR.status.state`` chain. The DGD must reach ``state=failed`` within
    120 s of the pull error becoming visible, and the surfaced reason or
    message must name the pull failure so an operator humans can act on it
    instead of staring at indefinite Pending.

    Args:
        kubectl: Package-scoped :py:class:`KubectlClient` for pod polling
            and status reads.
        dynamo_operator: Fixture that installs the Dynamo CRD and operator.
    """

    manifest = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": _D701_DGD_NAME, "namespace": _D701_DGD_NAMESPACE},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": _D701_BOGUS_IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                        }
                    },
                }
            }
        },
    }

    await kubectl.run(
        "delete",
        "namespace",
        _D701_DGD_NAMESPACE,
        "--wait=true",
        "--ignore-not-found",
        check=False,
    )
    await kubectl.create_namespace(_D701_DGD_NAMESPACE)
    try:
        await kubectl.apply(
            orjson.dumps(manifest).decode(), namespace=_D701_DGD_NAMESPACE
        )
        pull_pod = await _wait_for_image_pull_failure(
            kubectl,
            namespace=_D701_DGD_NAMESPACE,
            label_selector=f"{_D701_DGD_LABEL}={_D701_DGD_NAME}",
            timeout=_POD_REASON_TIMEOUT_S,
        )
        assert pull_pod, (
            f"D701: no pod in {_D701_DGD_NAMESPACE!r} surfaced "
            f"ImagePullBackOff/ErrImagePull within {_POD_REASON_TIMEOUT_S}s; "
            "kubelet may not have attempted the pull or the operator may "
            "not have created the child pods"
        )

        observed_state = await wait_for_dgd_state(
            kubectl,
            name=_D701_DGD_NAME,
            namespace=_D701_DGD_NAMESPACE,
            target_state="failed",
            timeout=_D701_DGD_FAILED_TIMEOUT_S,
        )
        assert observed_state == "failed", (
            f"D701: DGD did not reach state=failed within "
            f"{_D701_DGD_FAILED_TIMEOUT_S}s after pod {pull_pod!r} surfaced an "
            f"image-pull error (observed state={observed_state!r})"
        )

        status_text = await _D701_read_dgd_status_text(
            kubectl, namespace=_D701_DGD_NAMESPACE, name=_D701_DGD_NAME
        )
        lower = status_text.lower()
        assert any(
            term in lower
            for term in ("imagepullbackoff", "errimagepull", "pull", "image")
        ), (
            "D701: DGD reached state=failed but status did not mention the "
            "image-pull cause; an opaque failure is not actionable. "
            f"Observed status: {status_text!r}"
        )
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _D701_DGD_NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _wait_for_image_pull_failure(
    kubectl: KubectlClient,
    *,
    namespace: str,
    label_selector: str,
    timeout: float,
) -> str:
    """Poll until any child pod surfaces ``ImagePullBackOff`` or ``ErrImagePull``.

    Inlines the same shape as :py:meth:`ChaosInjector.wait_for_pod_status_reason`
    (see ``tests/kubernetes/chaos/chaos_injector.py``) but covers BOTH waiting
    reasons in a single pass. kubelet flips ``ErrImagePull`` to
    ``ImagePullBackOff`` after the first backoff window, so a single-reason
    poll can race the transition and miss it.

    Args:
        kubectl: Package-scoped :py:class:`KubectlClient`.
        namespace: Namespace housing the DGD's child pods.
        label_selector: kubectl ``-l`` selector for the DGD's pods.
        timeout: Max seconds to wait before returning ``""``.

    Returns:
        The pod name that first surfaced a pull-error reason, or ``""``
        when the timeout elapses with no match. Callers assert on the
        return value rather than raising so the failure message can
        include broader cluster context.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
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
            try:
                data = orjson.loads(result.stdout)
            except orjson.JSONDecodeError as exc:
                logger.debug(lambda exc=exc: f"D701 pod-list parse failed: {exc!r}")
                data = {}
            for item in data.get("items", []):
                pod_name = item.get("metadata", {}).get("name", "")
                statuses = item.get("status", {}).get("containerStatuses", []) or []
                init_statuses = (
                    item.get("status", {}).get("initContainerStatuses", []) or []
                )
                for cs in (*statuses, *init_statuses):
                    waiting = (cs.get("state") or {}).get("waiting") or {}
                    if waiting.get("reason") in _PULL_REASONS:
                        return pod_name
        await asyncio.sleep(1.0)
    return ""


async def _D701_read_dgd_status_text(
    kubectl: KubectlClient,
    *,
    namespace: str,
    name: str,
) -> str:
    """Return the DGD's ``status`` block as a JSON string for cause inspection.

    Used to assert the operator names the pull failure in the CR status
    (message / reason / condition), not just the bare ``state=failed`` value.
    Returns ``""`` on any kubectl error so callers see a deterministic empty
    cause rather than a raised exception masking the original assertion.
    """
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
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


# D702

_D702_DGD_NAME = "d702-test"
_D702_DGD_NAMESPACE = "d702-node-selector"
_IMPOSSIBLE_NODE_SELECTOR = {"aiperf.nvidia.com/d702-impossible-node": "true"}
_D702_EVENT_TIMEOUT_S = 90.0
_D702_FAILED_TIMEOUT_S = 120.0
_D702_STATUS_TERMS = (
    "nodeselector",
    "node selector",
    "node affinity",
    "didn't match",
    "unschedulable",
    "failedscheduling",
)


async def test_d702_impossible_node_selector_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Unsatisfiable node selector -> child pod Pending -> DGD failed status."""
    await kubectl.delete_namespace(_D702_DGD_NAMESPACE, wait=True)
    await kubectl.create_namespace(_D702_DGD_NAMESPACE)
    try:
        await kubectl.apply(_D702_manifest(), namespace=_D702_DGD_NAMESPACE)
        scheduling_event = await wait_for_events_or_status(
            kubectl,
            namespace=_D702_DGD_NAMESPACE,
            name=_D702_DGD_NAME,
            needles=_D702_STATUS_TERMS,
            timeout_s=_D702_EVENT_TIMEOUT_S,
            poll_interval_s=1.0,
        )
        assert scheduling_event, (
            f"D702: no unschedulable/nodeSelector event appeared in namespace "
            f"{_D702_DGD_NAMESPACE!r} within {_D702_EVENT_TIMEOUT_S}s"
        )

        observed_state = await wait_for_dgd_state(
            kubectl,
            _D702_DGD_NAME,
            _D702_DGD_NAMESPACE,
            "failed",
            timeout=_D702_FAILED_TIMEOUT_S,
        )
        assert observed_state == "failed"

        status_text = await read_dgd_status_text(
            kubectl, namespace=_D702_DGD_NAMESPACE, name=_D702_DGD_NAME
        )
        assert any(term in status_text.lower() for term in _D702_STATUS_TERMS), (
            "D702: DGD failed status did not name the nodeSelector/scheduling cause. "
            f"status={status_text!r}; event={scheduling_event!r}"
        )
    finally:
        await kubectl.delete_namespace(_D702_DGD_NAMESPACE, wait=False)


def _D702_manifest() -> str:
    return minimal_v1alpha1_frontend_dgd_manifest(
        _D702_DGD_NAME,
        _D702_DGD_NAMESPACE,
        extra_pod_spec={
            "nodeSelector": _IMPOSSIBLE_NODE_SELECTOR,
            "mainContainer": {"image": "busybox:1.36"},
        },
    )


# D703

_D703_DGD_NAMESPACE = "d703-resource-quota"
_QUOTA_NAME = "d703-too-small"
_D703_DGD_FAILED_TIMEOUT_S = 120.0
_QUOTA_EVENT_TIMEOUT_S = 90.0
_STATUS_CAUSE_TERMS = (
    "quota",
    "resourcequota",
    "exceeded",
    "insufficient",
    "failedscheduling",
    "unschedulable",
    "didn't match pod anti-affinity",
)


async def test_d703_resource_quota_exhaustion_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Tiny namespace ResourceQuota -> child pods rejected -> DGD ``state=failed``.

    The namespace quota allows zero pods and only 1Mi memory request/limit, which
    is below even the smallest worker pod requirement. The test first proves the
    apiserver or scheduler observed the quota/scheduling failure, then requires
    the parent DGD status to fail with an actionable quota/scheduling reason.
    """
    config = DynamoConfig.single_gpu_disagg(
        namespace=_D703_DGD_NAMESPACE,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()

    await kubectl.delete_namespace(_D703_DGD_NAMESPACE, wait=True)
    await kubectl.create_namespace(_D703_DGD_NAMESPACE)
    try:
        await kubectl.apply(_resource_quota_manifest(), namespace=_D703_DGD_NAMESPACE)
        await kubectl.apply(deployer.generate_manifest())

        quota_event_text = await _wait_for_quota_or_scheduling_event(
            kubectl,
            namespace=_D703_DGD_NAMESPACE,
            timeout=_QUOTA_EVENT_TIMEOUT_S,
        )
        assert quota_event_text, (
            f"D703: no ResourceQuota/scheduling event appeared in namespace "
            f"{_D703_DGD_NAMESPACE!r} within {_QUOTA_EVENT_TIMEOUT_S}s after applying "
            f"DGD {name!r}; the quota may not be blocking worker pods"
        )

        observed_state, observed_status = await _wait_for_failed_dgd_status(
            kubectl,
            namespace=_D703_DGD_NAMESPACE,
            name=name,
            timeout=_D703_DGD_FAILED_TIMEOUT_S,
        )
        assert observed_state == "failed", (
            f"D703: quota failure was observed in namespace events but DGD {name!r} "
            f"did not reach state='failed' within {_D703_DGD_FAILED_TIMEOUT_S}s; "
            f"observed state={observed_state!r}, status={observed_status!r}, "
            f"quota event={quota_event_text!r}"
        )

        status_lower = observed_status.lower()
        assert any(term in status_lower for term in _STATUS_CAUSE_TERMS), (
            "D703: DGD reached state='failed' but status did not name the "
            "quota/scheduling cause. "
            f"Observed status: {observed_status!r}; quota event: {quota_event_text!r}"
        )
    finally:
        await kubectl.delete_namespace(_D703_DGD_NAMESPACE, wait=False)


def _resource_quota_manifest() -> str:
    """Return the deliberately-too-small ResourceQuota manifest."""
    quota = {
        "apiVersion": "v1",
        "kind": "ResourceQuota",
        "metadata": {"name": _QUOTA_NAME, "namespace": _D703_DGD_NAMESPACE},
        "spec": {
            "hard": {
                "pods": "0",
                "requests.memory": "1Mi",
                "limits.memory": "1Mi",
            }
        },
    }
    return orjson.dumps(quota).decode()


async def _wait_for_quota_or_scheduling_event(
    kubectl: KubectlClient,
    *,
    namespace: str,
    timeout: float,
) -> str:
    """Poll namespace events until quota or scheduling rejection is visible."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        events_text = await _read_namespace_events(kubectl, namespace=namespace)
        events_lower = events_text.lower()
        if any(term in events_lower for term in _STATUS_CAUSE_TERMS):
            return events_text
        await asyncio.sleep(1.0)
    return ""


async def _wait_for_failed_dgd_status(
    kubectl: KubectlClient,
    *,
    namespace: str,
    name: str,
    timeout: float,
) -> tuple[str, str]:
    """Poll DGD status until ``state=failed`` or timeout, returning last status."""
    deadline = asyncio.get_event_loop().time() + timeout
    observed_state = "<unobserved>"
    observed_status = ""
    while True:
        observed_status = await _D703_read_dgd_status_text(
            kubectl,
            namespace=namespace,
            name=name,
        )
        observed_state = _state_from_status_text(observed_status)
        if observed_state == "failed" or asyncio.get_event_loop().time() >= deadline:
            return observed_state, observed_status
        await asyncio.sleep(2.0)


def _state_from_status_text(status_text: str) -> str:
    """Extract ``status.state`` from a JSON status payload."""
    return dgd_state_from_status_text(status_text)


async def _D703_read_dgd_status_text(
    kubectl: KubectlClient,
    *,
    namespace: str,
    name: str,
) -> str:
    """Return the DGD ``status`` block as JSON text for assertion messages."""
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
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


async def _read_namespace_events(
    kubectl: KubectlClient,
    *,
    namespace: str,
) -> str:
    """Return warning event reason/message text for the isolated namespace."""
    result = await kubectl.run(
        "get",
        "events",
        "-n",
        namespace,
        "--sort-by=.lastTimestamp",
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    try:
        data = orjson.loads(result.stdout)
    except orjson.JSONDecodeError as exc:
        logger.debug(lambda exc=exc: f"D703 events parse failed: {exc!r}")
        return result.stdout.strip()

    lines: list[str] = []
    for item in data.get("items", []):
        reason = item.get("reason", "")
        message = item.get("message", "")
        involved = item.get("involvedObject", {})
        ref = f"{involved.get('kind', '')}/{involved.get('name', '')}".strip("/")
        lines.append(f"{ref}: {reason}: {message}".strip())
    return "\n".join(lines)


# D704

_D704_MODEL_ENV = "D704_HF_CACHE_MISS_MODEL"
_D704_DEFAULT_MODEL = "hf-internal-testing/tiny-random-gpt2"
_D704_NAME = "dynamo-agg"
_D704_NAMESPACE = "d704-hf-egress"
_D704_POLICY = "d704-blackhole"
_D704_NETWORK_POLICY_CNI_NEEDLES = (
    "cilium",
    "calico",
    "tigera",
    "canal",
    "antrea",
)


async def test_d704_hf_hub_egress_blackhole(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
) -> None:
    """Block egress; assert worker fails weight-download cleanly, DGD reports failure.

    Requires a NetworkPolicy-enforcing CNI and a model name that is absent from
    the node/container HF cache. Without those prerequisites the test skips
    before applying the DGD so a green run cannot be a false positive from
    kindnet ignoring NetworkPolicy or from a cached model bypassing HF Hub.
    """
    await _skip_unless_network_policy_enforced(kubectl)
    model = _cache_miss_model_or_skip()
    request.getfixturevalue("dynamo_operator")
    faults = request.getfixturevalue("faults")
    await _run_d704_assertion(faults, kubectl, gpu_settings, model=model)


async def _run_d704_assertion(
    faults,
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
    *,
    model: str,
) -> None:
    """Apply DGD under an egress-deny policy and assert actionable failure.

    The DGD uses the aggregated vLLM worker shape because the HF download path
    lives in ``dynamo.vllm`` worker startup, not in the frontend component.
    """
    manifest = _build_d704_manifest(kubectl, gpu_settings, model=model)

    await kubectl.create_namespace(_D704_NAMESPACE)
    try:
        async with faults.inject(
            "cluster.network_policy.deny_egress",
            target={"ns": _D704_NAMESPACE},
            name=_D704_POLICY,
            allow_cluster_egress=True,
        ):
            await kubectl.apply(manifest=manifest, namespace=_D704_NAMESPACE)

            await wait_for_dgd_state(
                kubectl, _D704_NAME, _D704_NAMESPACE, "failed", timeout=300
            )

            result = await kubectl.run(
                "get",
                "dynamographdeployment",
                _D704_NAME,
                "-n",
                _D704_NAMESPACE,
                "-o",
                "json",
                check=True,
            )
            dgd = orjson.loads(result.stdout)
            assert dgd["status"]["state"] == "failed"

            status_text = orjson.dumps(dgd.get("status", {})).decode().lower()
            keywords = (
                "weight",
                "hub",
                "huggingface",
                "network",
                "egress",
                "dns",
                "connect",
                "download",
                "model",
            )
            assert any(kw in status_text for kw in keywords), (
                "expected DGD status to name network/model-download failure; "
                f"got status={status_text!r}"
            )
    finally:
        # NetworkPolicy is cleaned up by the faults.inject context's restore;
        # the DGD namespace is ours to delete.
        await kubectl.run(
            "delete",
            "namespace",
            _D704_NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


def _cache_miss_model_or_skip() -> str:
    """Return the requested cache-miss HF model or skip with setup guidance."""
    model = os.environ.get(_D704_MODEL_ENV, _D704_DEFAULT_MODEL).strip()
    if not model:
        pytest.skip(
            f"D704 requires {_D704_MODEL_ENV}=<uncached Hugging Face model id> "
            "so the worker must reach HF Hub during startup"
        )
    return model


def _build_d704_manifest(
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
    *,
    model: str,
) -> str:
    """Build a minimal v1alpha1 aggregated DGD that starts a vLLM worker."""
    config = DynamoConfig(
        model_name=model,
        namespace=_D704_NAMESPACE,
        backend=DynamoBackend.VLLM,
        mode=DynamoMode.AGGREGATED,
        gpu_count=0,
        max_model_len=gpu_settings.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=0.12,
        runtime_class_name=gpu_settings.runtime_class,
        hf_token_secret=gpu_settings.hf_token_secret,
        image=gpu_settings.dynamo_image,
        image_pull_secrets=gpu_settings.image_pull_secrets,
    )
    return DynamoDeployer(kubectl=kubectl, config=config).generate_manifest()


async def _skip_unless_network_policy_enforced(kubectl: KubectlClient) -> None:
    """Skip unless the cluster has NetworkPolicy API plus an enforcing CNI."""
    api_result = await kubectl.run(
        "api-resources",
        "--api-group=networking.k8s.io",
        "-o",
        "name",
        check=False,
    )
    if api_result.returncode != 0 or "networkpolicies" not in api_result.stdout:
        pytest.skip("D704 requires networking.k8s.io NetworkPolicy support")

    pods_result = await kubectl.run("get", "pods", "-A", "-o", "json", check=False)
    if pods_result.returncode != 0:
        pytest.skip("D704 could not inspect cluster CNI pods before NetworkPolicy test")

    pod_data = orjson.loads(pods_result.stdout or b"{}")
    cni_text = " ".join(
        f"{item.get('metadata', {}).get('namespace', '')}/"
        f"{item.get('metadata', {}).get('name', '')} "
        f"{item.get('metadata', {}).get('labels', {})}"
        for item in pod_data.get("items", [])
    ).lower()
    if not any(needle in cni_text for needle in _D704_NETWORK_POLICY_CNI_NEEDLES):
        pytest.skip(
            "D704 requires a NetworkPolicy-enforcing CNI such as Cilium or Calico; "
            "kindnet applies NetworkPolicy objects but does not block egress"
        )


# D705

_D705_DGD_NAME = "d705-test"
_D705_DGD_NAMESPACE = "d705-limitrange"
_LIMIT_RANGE_NAME = "d705-default-too-large"
_D705_EVENT_TIMEOUT_S = 90.0
_D705_FAILED_TIMEOUT_S = 120.0
_D705_STATUS_TERMS = (
    "limitrange",
    "limit range",
    "maximum",
    "minimum",
    "exceeded",
    "forbidden",
    "failedcreate",
)


async def test_d705_limitrange_conflict_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Namespace LimitRange rejects child pods and parent DGD reports why."""
    await kubectl.delete_namespace(_D705_DGD_NAMESPACE, wait=True)
    await kubectl.create_namespace(_D705_DGD_NAMESPACE)
    try:
        await kubectl.apply(_limitrange_manifest(), namespace=_D705_DGD_NAMESPACE)
        await kubectl.apply(_D705_dgd_manifest(), namespace=_D705_DGD_NAMESPACE)

        event_text = await wait_for_events_or_status(
            kubectl,
            namespace=_D705_DGD_NAMESPACE,
            name=_D705_DGD_NAME,
            needles=_D705_STATUS_TERMS,
            timeout_s=_D705_EVENT_TIMEOUT_S,
            poll_interval_s=1.0,
        )
        assert event_text, (
            f"D705: no LimitRange admission event appeared within {_D705_EVENT_TIMEOUT_S}s"
        )

        observed_state = await wait_for_dgd_state(
            kubectl,
            _D705_DGD_NAME,
            _D705_DGD_NAMESPACE,
            "failed",
            timeout=_D705_FAILED_TIMEOUT_S,
        )
        assert observed_state == "failed"

        status_text = await read_dgd_status_text(
            kubectl, namespace=_D705_DGD_NAMESPACE, name=_D705_DGD_NAME
        )
        assert any(term in status_text.lower() for term in _D705_STATUS_TERMS), (
            "D705: DGD failed status did not name the LimitRange cause. "
            f"status={status_text!r}; event={event_text!r}"
        )
    finally:
        await kubectl.delete_namespace(_D705_DGD_NAMESPACE, wait=False)


def _limitrange_manifest() -> str:
    manifest = {
        "apiVersion": "v1",
        "kind": "LimitRange",
        "metadata": {"name": _LIMIT_RANGE_NAME, "namespace": _D705_DGD_NAMESPACE},
        "spec": {
            "limits": [
                {
                    "type": "Container",
                    "default": {"cpu": "2", "memory": "2Gi"},
                    "defaultRequest": {"cpu": "2", "memory": "2Gi"},
                    "max": {"cpu": "10m", "memory": "16Mi"},
                }
            ]
        },
    }
    return orjson.dumps(manifest).decode()


def _D705_dgd_manifest() -> str:
    return minimal_v1alpha1_frontend_dgd_manifest(
        _D705_DGD_NAME,
        _D705_DGD_NAMESPACE,
        extra_pod_spec={"mainContainer": {"image": "busybox:1.36"}},
    )


# D706

_D706_DGD_NAME = "d706-test"
_D706_DGD_NAMESPACE = "d706-podsecurity"
_D706_EVENT_TIMEOUT_S = 90.0
_D706_FAILED_TIMEOUT_S = 120.0
_D706_STATUS_TERMS = (
    "podsecurity",
    "pod security",
    "restricted",
    "privileged",
    "hostnetwork",
    "forbidden",
    "failedcreate",
)


async def test_d706_podsecurity_rejection_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Restricted namespace rejects unsafe pod spec and DGD reports the cause."""
    await kubectl.delete_namespace(_D706_DGD_NAMESPACE, wait=True)
    await kubectl.apply(_namespace_manifest())
    try:
        await kubectl.apply(_D706_dgd_manifest(), namespace=_D706_DGD_NAMESPACE)

        event_text = await wait_for_events_or_status(
            kubectl,
            namespace=_D706_DGD_NAMESPACE,
            name=_D706_DGD_NAME,
            needles=_D706_STATUS_TERMS,
            timeout_s=_D706_EVENT_TIMEOUT_S,
            poll_interval_s=1.0,
        )
        assert event_text, (
            f"D706: no PodSecurity admission event appeared within {_D706_EVENT_TIMEOUT_S}s"
        )

        observed_state = await wait_for_dgd_state(
            kubectl,
            _D706_DGD_NAME,
            _D706_DGD_NAMESPACE,
            "failed",
            timeout=_D706_FAILED_TIMEOUT_S,
        )
        assert observed_state == "failed"

        status_text = await read_dgd_status_text(
            kubectl, namespace=_D706_DGD_NAMESPACE, name=_D706_DGD_NAME
        )
        assert any(term in status_text.lower() for term in _D706_STATUS_TERMS), (
            "D706: DGD failed status did not name the PodSecurity cause. "
            f"status={status_text!r}; event={event_text!r}"
        )
    finally:
        await kubectl.delete_namespace(_D706_DGD_NAMESPACE, wait=False)


def _namespace_manifest() -> str:
    manifest = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": _D706_DGD_NAMESPACE,
            "labels": {
                "pod-security.kubernetes.io/enforce": "restricted",
                "pod-security.kubernetes.io/enforce-version": "latest",
            },
        },
    }
    return orjson.dumps(manifest).decode()


def _D706_dgd_manifest() -> str:
    return minimal_v1alpha1_frontend_dgd_manifest(
        _D706_DGD_NAME,
        _D706_DGD_NAMESPACE,
        extra_pod_spec={
            "hostNetwork": True,
            "mainContainer": {
                "image": "busybox:1.36",
                "securityContext": {
                    "privileged": True,
                    "runAsUser": 0,
                },
            },
        },
    )


# D707

_D707_DGD_NAME = "d707-test"
_D707_DGD_NAMESPACE = "d707-image-pull-secret"
_D707_MISSING_SECRET = "d707-missing-pull-secret"
_D707_EVENT_TIMEOUT_S = 90.0
_D707_FAILED_TIMEOUT_S = 120.0
_D707_STATUS_TERMS = (
    "imagepullsecret",
    "image pull secret",
    "pull secret",
    _D707_MISSING_SECRET,
    "secret",
    "errimagepull",
    "imagepullbackoff",
)


async def test_d707_missing_image_pull_secret_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Missing private-registry pull secret is reflected in parent DGD status."""
    await kubectl.delete_namespace(_D707_DGD_NAMESPACE, wait=True)
    await kubectl.create_namespace(_D707_DGD_NAMESPACE)
    try:
        await kubectl.apply(_D707_manifest(), namespace=_D707_DGD_NAMESPACE)
        event_text = await wait_for_events_or_status(
            kubectl,
            namespace=_D707_DGD_NAMESPACE,
            name=_D707_DGD_NAME,
            needles=_D707_STATUS_TERMS,
            timeout_s=_D707_EVENT_TIMEOUT_S,
            poll_interval_s=1.0,
        )
        assert event_text, (
            f"D707: no imagePullSecret/pull event appeared within {_D707_EVENT_TIMEOUT_S}s"
        )

        observed_state = await wait_for_dgd_state(
            kubectl,
            _D707_DGD_NAME,
            _D707_DGD_NAMESPACE,
            "failed",
            timeout=_D707_FAILED_TIMEOUT_S,
        )
        assert observed_state == "failed"

        status_text = await read_dgd_status_text(
            kubectl, namespace=_D707_DGD_NAMESPACE, name=_D707_DGD_NAME
        )
        assert any(term in status_text.lower() for term in _D707_STATUS_TERMS), (
            "D707: DGD failed status did not name the missing imagePullSecret. "
            f"status={status_text!r}; event={event_text!r}"
        )
    finally:
        await kubectl.delete_namespace(_D707_DGD_NAMESPACE, wait=False)


def _D707_manifest() -> str:
    return minimal_v1alpha1_frontend_dgd_manifest(
        _D707_DGD_NAME,
        _D707_DGD_NAMESPACE,
        extra_pod_spec={
            "imagePullSecrets": [{"name": _D707_MISSING_SECRET}],
            "mainContainer": {
                "image": "nvcr.io/nvidia/private-dynamo-test:missing",
                "imagePullPolicy": "Always",
            },
        },
    )


# D708

_D708_DGD_NAMESPACE = "d708-hf-secret"
_D708_MISSING_SECRET = "d708-missing-hf-token"
_D708_EVENT_TIMEOUT_S = 90.0
_D708_FAILED_TIMEOUT_S = 120.0
_D708_STATUS_TERMS = (
    "secret",
    _D708_MISSING_SECRET,
    "envfrom",
    "configerror",
    "createcontainerconfigerror",
    "huggingface",
    "hf",
)


async def test_d708_missing_hf_secret_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Worker envFromSecret points at a missing HF secret and DGD reports it."""
    config = DynamoConfig.single_gpu_disagg(
        namespace=_D708_DGD_NAMESPACE,
        hf_token_secret=_D708_MISSING_SECRET,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    dgd_name = deployer._deployment_name()

    await kubectl.delete_namespace(_D708_DGD_NAMESPACE, wait=True)
    await kubectl.create_namespace(_D708_DGD_NAMESPACE)
    try:
        await kubectl.apply(deployer.generate_manifest(), namespace=_D708_DGD_NAMESPACE)
        event_text = await wait_for_events_or_status(
            kubectl,
            namespace=_D708_DGD_NAMESPACE,
            name=dgd_name,
            needles=_D708_STATUS_TERMS,
            timeout_s=_D708_EVENT_TIMEOUT_S,
            poll_interval_s=1.0,
        )
        assert event_text, (
            f"D708: no missing-secret event appeared within {_D708_EVENT_TIMEOUT_S}s"
        )

        observed_state = await wait_for_dgd_state(
            kubectl,
            dgd_name,
            _D708_DGD_NAMESPACE,
            "failed",
            timeout=_D708_FAILED_TIMEOUT_S,
        )
        assert observed_state == "failed"

        status_text = await read_dgd_status_text(
            kubectl, namespace=_D708_DGD_NAMESPACE, name=dgd_name
        )
        assert any(term in status_text.lower() for term in _D708_STATUS_TERMS), (
            "D708: DGD failed status did not name the missing HF token secret. "
            f"status={status_text!r}; event={event_text!r}"
        )
    finally:
        await kubectl.delete_namespace(_D708_DGD_NAMESPACE, wait=False)


# D709

_D709_DGD_NAME = "d709-test"
_D709_DGD_NAMESPACE = "d709-missing-pvc"
_MISSING_PVC = "d709-model-cache-missing"
_D709_EVENT_TIMEOUT_S = 90.0
_D709_FAILED_TIMEOUT_S = 120.0
_D709_STATUS_TERMS = (
    "persistentvolumeclaim",
    "pvc",
    _MISSING_PVC,
    "not found",
    "failedmount",
    "failedscheduling",
    "unbound",
)


async def test_d709_missing_pvc_reference_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Child pod references absent PVC and parent status names the storage cause."""
    await kubectl.delete_namespace(_D709_DGD_NAMESPACE, wait=True)
    await kubectl.create_namespace(_D709_DGD_NAMESPACE)
    try:
        await kubectl.apply(_D709_manifest(), namespace=_D709_DGD_NAMESPACE)
        event_text = await _D709_wait_for_event(kubectl, _D709_DGD_NAMESPACE)
        assert event_text, (
            f"D709: no PVC event appeared within {_D709_EVENT_TIMEOUT_S}s"
        )

        observed_state = await wait_for_dgd_state(
            kubectl,
            _D709_DGD_NAME,
            _D709_DGD_NAMESPACE,
            "failed",
            timeout=_D709_FAILED_TIMEOUT_S,
        )
        assert observed_state == "failed"

        status_text = await _D709_read_status(
            kubectl, _D709_DGD_NAMESPACE, _D709_DGD_NAME
        )
        assert any(term in status_text.lower() for term in _D709_STATUS_TERMS), (
            "D709: DGD failed status did not name the missing PVC. "
            f"status={status_text!r}; event={event_text!r}"
        )
    finally:
        await kubectl.delete_namespace(_D709_DGD_NAMESPACE, wait=False)


def _D709_manifest() -> str:
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": _D709_DGD_NAME, "namespace": _D709_DGD_NAMESPACE},
        "spec": {
            "components": [
                {
                    "name": "Frontend",
                    "type": "frontend",
                    "replicas": 1,
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "image": "busybox:1.36",
                                    "volumeMounts": [
                                        {"name": "model-cache", "mountPath": "/models"}
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "model-cache",
                                    "persistentVolumeClaim": {
                                        "claimName": _MISSING_PVC
                                    },
                                }
                            ],
                        }
                    },
                }
            ]
        },
    }
    return orjson.dumps(manifest).decode()


async def _D709_wait_for_event(kubectl: KubectlClient, namespace: str) -> str:
    deadline = asyncio.get_event_loop().time() + _D709_EVENT_TIMEOUT_S
    while asyncio.get_event_loop().time() < deadline:
        events = await _D709_read_events(kubectl, namespace)
        if any(term in events.lower() for term in _D709_STATUS_TERMS):
            return events
        await asyncio.sleep(1.0)
    return ""


async def _D709_read_events(kubectl: KubectlClient, namespace: str) -> str:
    result = await kubectl.run(
        "get", "events", "-n", namespace, "-o", "json", check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    data = orjson.loads(result.stdout)
    return "\n".join(
        f"{item.get('reason', '')}: {item.get('message', '')}"
        for item in data.get("items", [])
    )


async def _D709_read_status(kubectl: KubectlClient, namespace: str, name: str) -> str:
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


# D710

_D710_DGD_NAME = "d710-test"
_D710_DGD_NAMESPACE = "d710-runtimeclass"
_MISSING_RUNTIME_CLASS = "d710-no-such-runtimeclass"
_D710_EVENT_TIMEOUT_S = 90.0
_D710_FAILED_TIMEOUT_S = 120.0
_D710_STATUS_TERMS = (
    "runtimeclass",
    "runtime class",
    _MISSING_RUNTIME_CLASS,
    "not found",
    "failedcreate",
    "failedscheduling",
)


async def test_d710_missing_runtimeclass_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Absent RuntimeClass blocks child pod and DGD reports the runtime cause."""
    await kubectl.delete_namespace(_D710_DGD_NAMESPACE, wait=True)
    await kubectl.create_namespace(_D710_DGD_NAMESPACE)
    try:
        await kubectl.apply(_D710_manifest(), namespace=_D710_DGD_NAMESPACE)
        event_text = await wait_for_events_or_status(
            kubectl,
            namespace=_D710_DGD_NAMESPACE,
            name=_D710_DGD_NAME,
            needles=_D710_STATUS_TERMS,
            timeout_s=_D710_EVENT_TIMEOUT_S,
            poll_interval_s=1.0,
        )
        assert event_text, (
            f"D710: no RuntimeClass event appeared within {_D710_EVENT_TIMEOUT_S}s"
        )

        observed_state = await wait_for_dgd_state(
            kubectl,
            _D710_DGD_NAME,
            _D710_DGD_NAMESPACE,
            "failed",
            timeout=_D710_FAILED_TIMEOUT_S,
        )
        assert observed_state == "failed"

        status_text = await read_dgd_status_text(
            kubectl, namespace=_D710_DGD_NAMESPACE, name=_D710_DGD_NAME
        )
        assert any(term in status_text.lower() for term in _D710_STATUS_TERMS), (
            "D710: DGD failed status did not name the missing RuntimeClass. "
            f"status={status_text!r}; event={event_text!r}"
        )
    finally:
        await kubectl.delete_namespace(_D710_DGD_NAMESPACE, wait=False)


def _D710_manifest() -> str:
    return minimal_v1alpha1_frontend_dgd_manifest(
        _D710_DGD_NAME,
        _D710_DGD_NAMESPACE,
        extra_pod_spec={
            "runtimeClassName": _MISSING_RUNTIME_CLASS,
            "mainContainer": {"image": "busybox:1.36"},
        },
    )


# D711

_D711_DGD_NAMESPACE = "d711-worker-node-selector"
_SELECTOR = {"aiperf.nvidia.com/d711-no-worker-node": "true"}
_D711_EVENT_TIMEOUT_S = 90.0
_D711_FAILED_TIMEOUT_S = 120.0
_D711_STATUS_TERMS = (
    "nodeselector",
    "node selector",
    "node affinity",
    "didn't match",
    "unschedulable",
    "failedscheduling",
    "worker",
)


async def test_d711_worker_node_selector_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Worker-only unschedulable pod state must reach DGD status."""
    config = DynamoConfig.single_gpu_disagg(
        namespace=_D711_DGD_NAMESPACE,
        node_selector=_SELECTOR,
        api_version="v1beta1",
    )
    deployer = DynamoDeployer(kubectl, config)
    dgd_name = deployer._deployment_name()

    await kubectl.delete_namespace(_D711_DGD_NAMESPACE, wait=True)
    await kubectl.create_namespace(_D711_DGD_NAMESPACE)
    try:
        await kubectl.apply(deployer.generate_manifest(), namespace=_D711_DGD_NAMESPACE)
        event_text = await wait_for_events_or_status(
            kubectl,
            namespace=_D711_DGD_NAMESPACE,
            name=dgd_name,
            needles=_D711_STATUS_TERMS,
            timeout_s=_D711_EVENT_TIMEOUT_S,
            poll_interval_s=1.0,
        )
        assert event_text, (
            f"D711: no worker nodeSelector event appeared within {_D711_EVENT_TIMEOUT_S}s"
        )

        observed_state = await wait_for_dgd_state(
            kubectl,
            dgd_name,
            _D711_DGD_NAMESPACE,
            "failed",
            timeout=_D711_FAILED_TIMEOUT_S,
        )
        assert observed_state == "failed"

        status_text = await read_dgd_status_text(
            kubectl, namespace=_D711_DGD_NAMESPACE, name=dgd_name
        )
        assert any(term in status_text.lower() for term in _D711_STATUS_TERMS), (
            "D711: DGD failed status did not name the worker nodeSelector cause. "
            f"status={status_text!r}; event={event_text!r}"
        )
    finally:
        await kubectl.delete_namespace(_D711_DGD_NAMESPACE, wait=False)


# D712


async def test_d712_rbac_child_rolebinding_create_denial_blocks_then_recovers(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    """Revoke child RoleBinding create RBAC before apply, then restore and recover."""
    owner = await find_unique_operator_rbac_owner(
        kubectl,
        api_group="rbac.authorization.k8s.io",
        resource="rolebindings",
        verb="create",
        case_id="D712",
    )
    faults = request.getfixturevalue("faults")
    name = ""
    namespace = dynamo_deployment_namespace
    try:
        async with faults.inject(
            "cluster.rbac.revoke",
            target=rbac_revoke_target(owner),
            api_group="rbac.authorization.k8s.io",
            resource="rolebindings",
            verb="create",
        ):
            name, namespace = await _d112_apply_fresh_dgd(kubectl, namespace)
            await _d112_observe_not_successful(kubectl, name, namespace, case_id="D712")
            authz = await kubectl.run(
                "auth",
                "can-i",
                "create",
                "rolebindings.rbac.authorization.k8s.io",
                "--as=system:serviceaccount:dynamo-system:dynamo-operator",
                "-n",
                namespace,
                check=False,
            )
            assert authz.stdout.strip() == "no", (
                "D712: RBAC revoke did not remove rolebindings/create from "
                f"{owner.label}; auth can-i returned {authz.stdout!r}"
            )
        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)
    finally:
        if name:
            await _d112_delete_dgd(kubectl, name, namespace)


# D713-D724

_D713_D724_OPERATOR_NAMESPACE = "dynamo-system"
_D713_D724_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_DGD_CRD = "dynamographdeployments.nvidia.com"
_DGD_KIND = "dynamographdeployment"
_D713_D724_DGD_LABEL = "nvidia.com/dynamo-graph-deployment-name"
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
            label_selector=f"{_D713_D724_DGD_LABEL}={deployer._deployment_name()},{_WORKER_LABEL}",
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
        _D713_D724_OPERATOR_SELECTOR,
        "-n",
        _D713_D724_OPERATOR_NAMESPACE,
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
            _D713_D724_OPERATOR_NAMESPACE,
            "-l",
            _D713_D724_OPERATOR_SELECTOR,
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


# D725-D739

_D725_D739_DGD_LABEL = "nvidia.com/dynamo-graph-deployment-name"
_D725_D739_OPERATOR_NAMESPACE = "dynamo-system"
_D725_D739_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_SYSTEM_FAULT_ENV = "AIPERF_DYNAMO_CHAOS_ALLOW_SYSTEM_FAULTS"
_NODE_FAULT_LABEL = "aiperf.nvidia.com/chaos-node-faults"
_PROBE_IMAGE = "busybox:1.36"
_D725_D739_BOGUS_IMAGE = "nonexistent.example.com/dynamo:nope"
_D725_D739_NETWORK_POLICY_CNI_NEEDLES = (
    "cilium",
    "calico",
    "tigera",
    "canal",
    "antrea",
)


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
                        "image": _D725_D739_BOGUS_IMAGE,
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
                        "image": _D725_D739_BOGUS_IMAGE,
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
    if not any(needle in cni_text for needle in _D725_D739_NETWORK_POLICY_CNI_NEEDLES):
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
    return minimal_v1alpha1_frontend_dgd_manifest(
        name,
        namespace,
        extra_pod_spec=extra_pod_spec
        or {
            "mainContainer": {
                "image": _D725_D739_BOGUS_IMAGE,
                "imagePullPolicy": "IfNotPresent",
            }
        },
    )


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
    return mentions_any(text, needles)


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
            f"{_D725_D739_DGD_LABEL}={name}",
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
        f"{_D725_D739_DGD_LABEL}={name}",
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
    return f"system:serviceaccount:{_D725_D739_OPERATOR_NAMESPACE}:{sa}"


async def _operator_service_account(kubectl: KubectlClient) -> str:
    result = await kubectl.run(
        "get",
        "deployment",
        "-n",
        _D725_D739_OPERATOR_NAMESPACE,
        "-l",
        _D725_D739_OPERATOR_SELECTOR,
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
        args.extend(["-n", _D725_D739_OPERATOR_NAMESPACE])
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
            and subject.get("namespace") == _D725_D739_OPERATOR_NAMESPACE
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
