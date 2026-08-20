# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chaos (unified): real-world infrastructure faults (K1, K2, K3).

Unified-API port of :py:mod:`tests.kubernetes.chaos.test_chaos_infra`. Covers
chaos-expansion scenarios K1, K2, K3:

* K1 -- non-existent container image surfaces ``ImagePullBackOff`` on
  JobSet pods and the operator transitions the CR to ``Failed`` instead of
  retrying silently.
* K2 -- unresolvable endpoint hostname (``.invalid`` TLD per RFC 2606) is
  surfaced to the CR as ``Failed`` with a DNS/resolution/endpoint cause
  rather than hanging the worker forever.
* K3 -- a tight namespace ``ResourceQuota`` blocks pod admission; the
  operator surfaces the quota admission rejection on the CR within 120 s.

Differences from the legacy module:

* CR application flows through ``faults.inject("crd.apply_invalid", ...)``
  rather than :py:meth:`OperatorDeployer.create_job`. The registry's LIFO
  restore handles CR deletion on block exit, so the tests no longer need a
  bespoke ``_force_delete`` finally.
* K3's ``ResourceQuota`` lifecycle is owned by
  ``faults.inject("cluster.resource_quota", ...)`` -- the restore handle
  deletes the quota even when the body raises, matching the legacy
  ``finally: chaos_injector.delete_resource_quota`` semantics with one
  less call site.
* ``wait_for_aiperfjob_phase`` replaces the inline poll loops from the
  legacy file. The helper accepts a tuple of acceptable phases and
  surfaces the last observed phase in the :py:class:`TimeoutError`
  message; tests still catch the timeout to add scenario-specific context.

The :py:meth:`ChaosInjector.wait_for_pod_status_reason` call in K1 is the
one legacy primitive that does not yet have a unified equivalent -- it is
a generic kubelet-status poller, not a fault injection, and the chaos_aiperf
conftest exposes the legacy ``chaos_injector`` fixture so the helper is
reachable without going through the registry.
"""

from __future__ import annotations

import orjson
import pytest

from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.chaos_aiperf.conftest import wait_for_aiperfjob_phase
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

pytestmark = [pytest.mark.asyncio, pytest.mark.k8s_slow]


async def _status_text(kubectl: KubectlClient, namespace: str, name: str) -> str:
    """Return serialized AIPerfJob ``.status`` JSON for cause assertions.

    Empty string when the CR has no status block (e.g. the apply was
    rejected by the apiserver before the operator reconciled). Callers
    pattern-match on substrings so an empty result fails the assertion
    with a useful "no status observed" report rather than raising.
    """
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
    """Return serialized namespace events JSON for cause assertions.

    Used by K3 to find the apiserver's ``exceeded quota`` admission
    rejection when the CR ``.status`` itself does not name the quota
    (some operator paths surface only the bare ``Failed`` phase and rely
    on the namespace event stream for the human-readable cause).
    """
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
async def test_k1_image_pull_backoff_surfaces_pending_unified(
    operator_ready: OperatorDeployer,  # noqa: ARG001  (operator must be running to reconcile the CR)
    faults,  # noqa: ANN001  (InjectorRegistry; typed at fixture site)
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """A non-existent image surfaces ``ImagePullBackOff`` on JobSet pods.

    Applies an AIPerfJob whose ``spec.image`` is
    ``ghcr.io/does-not-exist/nope:404`` with ``imagePullPolicy=IfNotPresent``
    so kubelet actually attempts the pull. Asserts:

    * Within 90 s, at least one pod in ``operator_job_namespace`` has a
      container status whose ``waiting.reason`` is ``ImagePullBackOff`` or
      ``ErrImagePull``.
    * ``kubectl describe pod`` surfaces the pull failure in the event
      stream so a human reviewer can act on it.
    * The CR ``.status.phase`` transitions to ``Failed`` within 120 s after
      the pod pull failure is visible.

    CR lifecycle (apply + delete-on-exit) is owned by
    ``faults.inject("crd.apply_invalid", ...)``. The CR is technically a
    well-formed AIPerfJob -- the apiserver accepts it -- but the resulting
    workload is "invalid" in the sense the chaos catalog cares about: the
    image will never pull, so the JobSet's pods can never start.
    """
    name = "chaos-k1-unified"
    bad_image_config = AIPerfJobConfig(
        concurrency=1,
        request_count=10,
        warmup_request_count=0,
        image="ghcr.io/does-not-exist/nope:404",
        image_pull_policy="IfNotPresent",
    )
    manifest = bad_image_config.to_cr_manifest(name, operator_job_namespace)

    async with faults.inject(
        "crd.apply_invalid",
        target={"ns": operator_job_namespace, "name": name},
        manifest=manifest,
    ):
        chaos_injector = ChaosInjector(kubectl=kubectl)
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

        # `kubectl describe pod` should show an event mentioning the pull
        # failure; surface this for the human reviewer and assert on it as
        # the "clear cause" requirement.
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
        try:
            observed_phase = await wait_for_aiperfjob_phase(
                kubectl,
                operator_job_namespace,
                name,
                ("Failed",),
                timeout=120.0,
            )
        except TimeoutError as exc:
            pytest.fail(
                f"K1: CR did not reach Failed within 120 s after pod "
                f"{pull_pod!r} surfaced ErrImagePull/ImagePullBackOff "
                f"({exc!s})"
            )
        assert observed_phase == "Failed"


@pytest.mark.timeout(300)
async def test_k2_dns_resolution_failure_fails_fast_unified(
    operator_ready: OperatorDeployer,  # noqa: ARG001  (operator must be running to reconcile the CR)
    faults,  # noqa: ANN001
    operator_job_namespace: str,
    kubectl: KubectlClient,
    k8s_settings,  # noqa: ANN001  (pytest fixture typed as Any via duck-typing)
) -> None:
    """Unresolvable endpoint hostname surfaces a DNS error and fails fast.

    Applies an AIPerfJob whose ``endpoint.urls`` point at a ``.invalid``
    TLD (RFC 2606 -- guaranteed never to resolve). The expectation:

    * The CR reaches terminal ``Failed`` phase within 120 s.
    * The CR ``.status`` JSON names a DNS, resolution, endpoint,
      unreachable, or ``.invalid`` cause.

    A ``.invalid`` TLD is used over an unreachable-but-valid host so the
    failure mode is deterministic across every cluster's DNS setup. CR
    lifecycle flows through ``crd.apply_invalid``; the registry's
    restore deletes the CR on block exit.
    """
    name = "chaos-k2-unified"
    dns_failure_config = AIPerfJobConfig(
        endpoint_url="http://this-hostname-does-not-exist.invalid:8000/v1",
        concurrency=1,
        request_count=10,
        warmup_request_count=0,
        image=k8s_settings.aiperf_image,
    )
    manifest = dns_failure_config.to_cr_manifest(name, operator_job_namespace)

    async with faults.inject(
        "crd.apply_invalid",
        target={"ns": operator_job_namespace, "name": name},
        manifest=manifest,
    ):
        # Wait up to 120 s for the CR to transition to Failed. A Running
        # phase is fine in the interim -- the DNS failure may only surface
        # after the worker attempts the first request.
        try:
            observed_phase = await wait_for_aiperfjob_phase(
                kubectl,
                operator_job_namespace,
                name,
                ("Failed",),
                timeout=120.0,
            )
        except TimeoutError as exc:
            pytest.fail(
                f"K2: CR did not reach Failed within 120 s; DNS failure "
                "may be retried forever inside the worker instead of being "
                f"surfaced to the operator ({exc!s})"
            )
        assert observed_phase == "Failed"

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


@pytest.mark.timeout(300)
async def test_k3_resource_quota_exhaustion_fails_fast_unified(
    operator_ready: OperatorDeployer,  # noqa: ARG001  (operator must be running to reconcile the CR)
    faults,  # noqa: ANN001
    operator_job_namespace: str,
    kubectl: KubectlClient,
    k8s_settings,  # noqa: ANN001
) -> None:
    """A tight memory ``ResourceQuota`` blocks pod admission; CR surfaces cause.

    Stacks two unified-API faults:

    1. ``cluster.resource_quota`` -- applies a ``ResourceQuota`` capping
       namespace memory at 256Mi (well below any controller/worker request).
    2. ``crd.apply_invalid`` -- applies a well-formed AIPerfJob whose pods
       cannot be admitted because of the quota.

    Asserts that within 120 s the CR reaches ``Failed`` and either the
    CR ``.status`` text mentions ``quota`` or namespace events mention
    the quota admission rejection (``exceeded quota`` / ``forbidden:
    exceeded``).

    Restore order matters: the registry's LIFO unwind deletes the
    AIPerfJob first (inner ``crd.apply_invalid``) and then the
    ``ResourceQuota`` (outer ``cluster.resource_quota``), so subsequent
    tests in the same namespace can make forward progress. We do not
    assert forward progress here -- the CR itself may be latched Failed
    by the operator; a second CR in the same namespace after quota
    removal is the cleaner recovery signal, covered by sibling chaos
    suites that run after this file.
    """
    name = "chaos-k3-unified"
    quota_name = "chaos-k3-quota-unified"
    config = AIPerfJobConfig(
        concurrency=1,
        request_count=10,
        warmup_request_count=0,
        image=k8s_settings.aiperf_image,
    )
    manifest = config.to_cr_manifest(name, operator_job_namespace)

    async with (
        faults.inject(
            "cluster.resource_quota",
            target={"ns": operator_job_namespace},
            name=quota_name,
            hard_limits={
                "requests.memory": "256Mi",
                "limits.memory": "256Mi",
            },
        ),
        faults.inject(
            "crd.apply_invalid",
            target={"ns": operator_job_namespace, "name": name},
            manifest=manifest,
        ),
    ):
        # Poll for up to 120 s: quota admission rejection must now
        # surface on the CR as Failed, not only in namespace events.
        try:
            observed_phase = await wait_for_aiperfjob_phase(
                kubectl,
                operator_job_namespace,
                name,
                ("Failed",),
                timeout=120.0,
            )
        except TimeoutError as exc:
            pytest.fail(
                f"K3: CR did not reach Failed within 120 s after "
                f"ResourceQuota {quota_name!r} rejected pod admission "
                f"({exc!s})"
            )
        assert observed_phase == "Failed"

        status_text = await _status_text(kubectl, operator_job_namespace, name)
        events_text = await _namespace_events_text(kubectl, operator_job_namespace)
        lower_status = status_text.lower()
        lower_events = events_text.lower()
        assert (
            "quota" in lower_status
            or "exceeded quota" in lower_events
            or "forbidden: exceeded" in lower_events
        ), (
            "K3: CR reached Failed but neither CR status mentioned "
            "quota nor namespace events mentioned an exceeded quota "
            "admission rejection. "
            f"Observed status: {status_text!r}\n"
            f"Observed events: {events_text[-4000:]!r}"
        )
