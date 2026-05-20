# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D701 -- ImagePullBackOff propagates to CR ``status.state=failed``.

Validates the kubelet -> pod-status -> operator-reconcile -> CR chain that
turns an infra-level image-pull failure into an actionable CR status,
rather than leaving the DGD opaquely Pending forever.

Scenario (D-series catalog, section D7xx):

* Apply a ``DynamoGraphDeployment`` whose Frontend component points at a
  non-existent image (``nonexistent.example.com/dynamo:nope``).
* Wait for kubelet to surface ``ImagePullBackOff`` / ``ErrImagePull`` on
  one of the child pods.
* Assert the Dynamo operator reads the pod state and transitions the CR
  ``status.state`` to ``"failed"`` within 120 s of the pull failure
  becoming visible.

Scaffold-grade: the assertion body is gated behind a ``pytest.skip`` until
the test runs against a live Dynamo cluster. The full setup -- manifest,
fault injection, polling loops -- is in place so a follow-up branch can
delete the ``skip`` and exercise the path end-to-end.
"""

from __future__ import annotations

import asyncio

import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


_DGD_NAME = "d701-test"
_DGD_NAMESPACE = "dynamo-server"
_BOGUS_IMAGE = "nonexistent.example.com/dynamo:nope"
_PULL_REASONS = ("ImagePullBackOff", "ErrImagePull")
_POD_REASON_TIMEOUT_S = 60.0
_DGD_FAILED_TIMEOUT_S = 120.0


async def test_d701_imagepullbackoff_surfaces_failed_state(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
) -> None:
    """Bogus container image -> kubelet ImagePullBackOff -> CR ``state=failed``.

    Targets the kubelet pull-fail -> pod-status -> operator-reconcile ->
    ``CR.status.state`` chain. The DGD must reach ``state=failed`` within
    120 s of the pull error becoming visible, and the surfaced reason or
    message must name the pull failure so an operator humans can act on it
    instead of staring at indefinite Pending.

    Args:
        faults: D-series fault registry; ``crd.apply_invalid`` is wired to
            apply the supplied DGD manifest and delete the CR on restore.
        kubectl: Package-scoped :py:class:`KubectlClient` for pod polling
            and status reads.
    """
    pytest.skip("scaffold landed; assertion-body pending real-cluster validation")

    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": _DGD_NAME, "namespace": _DGD_NAMESPACE},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": _BOGUS_IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                        }
                    },
                }
            }
        },
    }

    async with faults.inject(
        "crd.apply_invalid",
        target={"ns": _DGD_NAMESPACE, "name": _DGD_NAME},
        manifest=manifest,
    ):
        pull_pod = await _wait_for_image_pull_failure(
            kubectl,
            namespace=_DGD_NAMESPACE,
            label_selector=f"nvidia.com/dynamographdeployment={_DGD_NAME}",
            timeout=_POD_REASON_TIMEOUT_S,
        )
        assert pull_pod, (
            f"D701: no pod in {_DGD_NAMESPACE!r} surfaced "
            f"ImagePullBackOff/ErrImagePull within {_POD_REASON_TIMEOUT_S}s; "
            "kubelet may not have attempted the pull or the operator may "
            "not have created the child pods"
        )

        observed_state = await wait_for_dgd_state(
            kubectl,
            name=_DGD_NAME,
            namespace=_DGD_NAMESPACE,
            target_state="failed",
            timeout=_DGD_FAILED_TIMEOUT_S,
        )
        assert observed_state == "failed", (
            f"D701: DGD did not reach state=failed within "
            f"{_DGD_FAILED_TIMEOUT_S}s after pod {pull_pod!r} surfaced an "
            f"image-pull error (observed state={observed_state!r})"
        )

        status_text = await _read_dgd_status_text(
            kubectl, namespace=_DGD_NAMESPACE, name=_DGD_NAME
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


async def _read_dgd_status_text(
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
