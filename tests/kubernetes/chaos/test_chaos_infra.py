# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chaos: real-world infrastructure faults.

Covers chaos-expansion scenarios K1, K2, K3. These
scenarios exercise operator behavior under faults that originate OUTSIDE
the operator's code path — a broken container image, an unreachable
endpoint hostname, and a namespace-level admission quota.

Exercises these operator code paths:

* ``src/aiperf/operator/handlers/create.py::on_create`` — CR accepted and
  the JobSet created, but downstream pods never become Ready because
  kubelet rejects the image pull (K1) or the quota admission webhook
  rejects pod creation (K3). Tests that the operator now propagates those
  infrastructure failures to CR phase ``Failed`` rather than retrying
  silently.
* ``src/aiperf/operator/handlers/monitor.py::_monitor_tick`` — when the
  benchmark endpoint URL never resolves / never responds, the
  SystemController surfaces a connection error rather than hanging
  forever (K2).
"""

from __future__ import annotations

import asyncio
import time

import orjson
import pytest

from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

pytestmark = [pytest.mark.asyncio, pytest.mark.k8s_slow]


async def _force_delete(kubectl: KubectlClient, namespace: str, name: str) -> None:
    """Best-effort CR delete; used as the unconditional finally-path."""
    await kubectl.run(
        "delete",
        "aiperfjob",
        name,
        "-n",
        namespace,
        "--ignore-not-found",
        "--wait=false",
        check=False,
    )


async def _phase(kubectl: KubectlClient, namespace: str, name: str) -> str:
    """Return the CR's current ``.status.phase`` or empty string when unset."""
    res = await kubectl.run(
        "get",
        "aiperfjob",
        name,
        "-n",
        namespace,
        "-o",
        "jsonpath={.status.phase}",
        check=False,
    )
    return res.stdout.strip()


async def _status_text(kubectl: KubectlClient, namespace: str, name: str) -> str:
    """Return serialized AIPerfJob status text for cause assertions."""
    res = await kubectl.run(
        "get",
        "aiperfjob",
        name,
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if not res.stdout.strip():
        return ""
    body = orjson.loads(res.stdout)
    status = body.get("status", {})
    return orjson.dumps(status).decode()


async def _namespace_events_text(kubectl: KubectlClient, namespace: str) -> str:
    """Return serialized namespace events text for cause assertions."""
    res = await kubectl.run(
        "get",
        "events",
        "-n",
        namespace,
        "--sort-by=.lastTimestamp",
        "-o",
        "json",
        check=False,
    )
    return res.stdout


@pytest.mark.timeout(300)
async def test_k1_image_pull_backoff_surfaces_pending(
    operator_ready: OperatorDeployer,  # noqa: ARG001  (operator must be running to reconcile the CR)
    chaos_injector: ChaosInjector,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """A non-existent image surfaces ImagePullBackOff on JobSet pods.

    Creates a CR pointing at ``ghcr.io/does-not-exist/nope:404`` with
    ``imagePullPolicy=IfNotPresent`` so kubelet actually attempts to
    pull. Asserts:

    * Within 90 s, at least one pod in ``operator_job_namespace`` has
      container status ``ImagePullBackOff`` or ``ErrImagePull``.
    * ``kubectl describe pod`` surfaces the pull failure in the event
      stream (``Failed to pull image`` or ``ErrImagePull`` / ``Back-off``).
    * The CR ``.status.phase`` transitions to ``Failed`` within 120 s
      after the pod pull failure is visible.
    """
    name = "chaos-k1"
    bad_image_config = AIPerfJobConfig(
        concurrency=1,
        request_count=10,
        warmup_request_count=0,
        image="ghcr.io/does-not-exist/nope:404",
        image_pull_policy="IfNotPresent",
    )
    try:
        await operator_ready.create_job(
            config=bad_image_config, name=name, namespace=operator_job_namespace
        )

        # Wait for any pod in the JobSet to surface an image-pull waiting
        # reason. Either ErrImagePull (first attempt) or ImagePullBackOff
        # (subsequent backoff) is acceptable evidence; poll each in turn.
        pull_pod = ""
        for reason in ("ErrImagePull", "ImagePullBackOff"):
            try:
                pull_pod = await chaos_injector.wait_for_pod_status_reason(
                    operator_job_namespace,
                    label_selector=f"jobset.sigs.k8s.io/jobset-name=aiperf-{name}",
                    reason=reason,
                    timeout=90.0,
                )
                break
            except TimeoutError:
                continue
        assert pull_pod, (
            f"K1: no pod in {operator_job_namespace} surfaced "
            "ErrImagePull/ImagePullBackOff within 90 s; kubelet may not "
            "have attempted the pull (check imagePullPolicy) or the "
            "operator may not have created the JobSet"
        )

        # `kubectl describe pod` should show an event mentioning the
        # pull failure; surface this for the human reviewer and assert
        # on it as the "clear cause" requirement.
        describe = await kubectl.run(
            "describe",
            "pod",
            pull_pod,
            "-n",
            operator_job_namespace,
            check=False,
        )
        lower = describe.stdout.lower()
        assert (
            "failed to pull image" in lower
            or "errimagepull" in lower
            or "imagepullbackoff" in lower
        ), (
            f"K1: `kubectl describe pod {pull_pod}` did not surface a "
            "pull-error event. Operator may be hiding the pod's event "
            "stream or the pod is stuck in a different failure mode.\n"
            f"Describe output head:\n{describe.stdout[:2000]}"
        )

        # The CR must now reach Failed once the pod pull error is visible.
        deadline = time.monotonic() + 120.0
        observed_phase = ""
        while time.monotonic() < deadline:
            observed_phase = await _phase(kubectl, operator_job_namespace, name)
            if observed_phase == "Failed":
                break
            await asyncio.sleep(2.0)
        assert observed_phase == "Failed", (
            f"K1: CR did not reach Failed within 120 s after pod "
            f"{pull_pod!r} surfaced ErrImagePull/ImagePullBackOff "
            f"(observed phase={observed_phase!r})"
        )
    finally:
        await _force_delete(kubectl, operator_job_namespace, name)


@pytest.mark.timeout(300)
async def test_k2_dns_resolution_failure_fails_fast(
    operator_ready: OperatorDeployer,  # noqa: ARG001  (operator must be running to reconcile the CR)
    chaos_injector: ChaosInjector,
    operator_job_namespace: str,
    kubectl: KubectlClient,
    k8s_settings,  # noqa: ANN001  (pytest fixture typed as Any via duck-typing)
) -> None:
    """Unresolvable endpoint hostname surfaces a DNS error and fails fast.

    Creates a CR whose ``endpoint.urls`` point at a ``.invalid`` TLD
    (RFC 2606 — guaranteed never to resolve). The expectation:

    * The CR reaches terminal ``Failed`` phase within 120 s.
    * The CR status text names a DNS, resolution, endpoint, unreachable,
      or ``.invalid`` cause.

    A ``.invalid`` TLD is used over an unreachable-but-valid host so the
    failure mode is deterministic across every cluster's DNS setup.
    """
    name = "chaos-k2"
    dns_failure_config = AIPerfJobConfig(
        endpoint_url="http://this-hostname-does-not-exist.invalid:8000/v1",
        concurrency=1,
        request_count=10,
        warmup_request_count=0,
        image=k8s_settings.aiperf_image,
    )
    try:
        await operator_ready.create_job(
            config=dns_failure_config, name=name, namespace=operator_job_namespace
        )

        # Wait up to 120 s for the CR to transition to Failed. A Running
        # phase is fine in the interim — the DNS failure may only surface
        # after the worker attempts the first request.
        deadline = time.monotonic() + 120.0
        observed_phase = ""
        while time.monotonic() < deadline:
            observed_phase = await _phase(kubectl, operator_job_namespace, name)
            if observed_phase == "Failed":
                break
            await asyncio.sleep(2.0)
        assert observed_phase == "Failed", (
            f"K2: CR did not reach Failed within 120 s (observed "
            f"phase={observed_phase!r}); DNS failure may be retried "
            "forever inside the worker instead of being surfaced to the "
            "operator"
        )

        # The Failed status must name the endpoint/DNS problem domain; an
        # unrelated False condition is not enough evidence for this scenario.
        status_text = await _status_text(kubectl, operator_job_namespace, name)
        lower_status = status_text.lower()
        assert any(
            term in lower_status
            for term in ("dns", "resolution", "endpoint", "unreachable", ".invalid")
        ), (
            "K2: CR reached Failed but status did not mention a DNS, "
            "resolution, endpoint, unreachable, or .invalid cause. "
            f"Observed status: {status_text!r}"
        )
    finally:
        await _force_delete(kubectl, operator_job_namespace, name)


@pytest.mark.timeout(300)
async def test_k3_resource_quota_exhaustion_fails_fast(
    operator_ready: OperatorDeployer,  # noqa: ARG001  (operator must be running to reconcile the CR)
    chaos_injector: ChaosInjector,
    operator_job_namespace: str,
    kubectl: KubectlClient,
    k8s_settings,  # noqa: ANN001
) -> None:
    """A tight memory ResourceQuota blocks pod admission; CR surfaces cause.

    Applies a ``ResourceQuota`` capping namespace memory at 256Mi (well
    below any controller/worker request) and creates a CR. Asserts:

    * Within 120 s the CR reaches ``Failed`` and either CR status text
      mentions quota or namespace events mention the quota admission
      rejection.

    The quota is removed unconditionally in ``finally`` so subsequent
    tests in the same namespace can make forward progress. We do not
    assert forward progress here — the CR itself may be latched Failed
    by the operator; a second CR in the same namespace after quota
    removal is the cleaner recovery signal, covered by the sibling
    chaos suites that run after this file.
    """
    name = "chaos-k3"
    quota_name = "chaos-k3-quota"
    config = AIPerfJobConfig(
        concurrency=1,
        request_count=10,
        warmup_request_count=0,
        image=k8s_settings.aiperf_image,
    )
    quota_applied = False
    try:
        await chaos_injector.apply_resource_quota(
            operator_job_namespace,
            quota_name,
            hard_limits={
                "requests.memory": "256Mi",
                "limits.memory": "256Mi",
            },
        )
        quota_applied = True

        await operator_ready.create_job(
            config=config, name=name, namespace=operator_job_namespace
        )

        # Poll for up to 120 s: quota admission rejection must now surface
        # on the CR as Failed, not only in namespace events.
        deadline = time.monotonic() + 120.0
        observed_phase = ""
        while time.monotonic() < deadline:
            observed_phase = await _phase(kubectl, operator_job_namespace, name)
            if observed_phase == "Failed":
                break
            await asyncio.sleep(2.0)

        assert observed_phase == "Failed", (
            f"K3: CR did not reach Failed within 120 s after ResourceQuota "
            f"{quota_name!r} rejected pod admission (observed phase="
            f"{observed_phase!r})"
        )

        status_text = await _status_text(kubectl, operator_job_namespace, name)
        events_text = await _namespace_events_text(kubectl, operator_job_namespace)
        lower_status = status_text.lower()
        lower_events = events_text.lower()
        assert (
            "quota" in lower_status
            or "exceeded quota" in lower_events
            or "forbidden: exceeded" in lower_events
        ), (
            "K3: CR reached Failed but neither CR status mentioned quota "
            "nor namespace events mentioned an exceeded quota admission "
            "rejection. "
            f"Observed status: {status_text!r}\n"
            f"Observed events: {events_text[-4000:]!r}"
        )
    finally:
        await _force_delete(kubectl, operator_job_namespace, name)
        if quota_applied:
            await chaos_injector.delete_resource_quota(
                operator_job_namespace, quota_name
            )
