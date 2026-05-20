# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D301 - Toxiproxy ``reset_peer`` toxic on NIXL side-channel during prefill->decode.

Wave-0 #4 scenario from the D-series chaos matrix. Exercises the NIXL
KV-transfer status path documented in ``lib/kvbm-physical/src/transfer/
notifications/nixl_status.rs:30`` ("NIXL transfer status check failed"): a
``reset_peer`` toxic dropped onto the NIXL side-channel port mid-handoff
should propagate up the worker stack into a structured SSE error frame at
the frontend, **not** crash the decode pod.

The NIXL side-channel port is ``5600 + engineID`` per the upstream
``failover_vllm.go:35`` convention; with a single-engine deployment the
target is ``5600``. We front it with a Toxiproxy proxy named ``nixl-0`` on
``0.0.0.0:20040`` so the toxic can be applied to a stable address.

This test is currently a **scaffold** that asserts the toxiproxy contract
(proxy reachable, reset_peer toxic applies and clears) and is marked
``pytest.skip()`` at the assertion-of-product-behaviour point. The full
behavioural assertion (decode worker survives + frontend SSE shape +
``dynamo_component_errors_total{error_type="response_stream"}``
increments) requires redirecting the decode worker's
``VLLM_NIXL_SIDE_CHANNEL_HOST`` env var to the Toxiproxy ``Service``,
which is a ``DynamoDeployer`` change that lands in a follow-up phase.
"""

from __future__ import annotations

import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


NIXL_PROXY_NAME = "nixl-0"
"""Toxiproxy proxy name fronting the NIXL side-channel for engine 0."""

NIXL_PROXY_LISTEN = "0.0.0.0:20040"
"""Listen address inside the toxiproxy pod. Must be one of the ports
exposed by the chaos-toxiproxy Service (20040 is reserved in
``tests/kubernetes/chaos_common/fixtures/toxiproxy.yaml`` for NIXL)."""

NIXL_ENGINE_PORT = 5600
"""NIXL side-channel base port. Per ``failover_vllm.go:35`` the actual
port is ``5600 + engineID``; this scaffold targets engine 0."""

_DECODE_LABEL_SELECTOR = "nvidia.com/dynamo-sub-component-type=decode"
"""Label injected by the Dynamo operator onto decode worker pods. Source
of truth: ``deploy/operator/internal/consts/consts.go:59-60`` (mirrored
in ``tests/kubernetes/gpu/dynamo/helpers.py``)."""

_PREFILL_LABEL_SELECTOR = "nvidia.com/dynamo-sub-component-type=prefill"
"""Same label, ``prefill`` value. Both must be present for the scenario
to be meaningful (D301 is a disagg-only fault)."""


async def _list_pods_with_label(
    kubectl: KubectlClient, namespace: str, selector: str
) -> list[str]:
    """Return pod names in ``namespace`` matching ``selector``.

    Thin wrapper around ``kubectl get pods -l <selector>`` so the
    pre-condition probes below stay one-liners. Returns ``[]`` on any
    non-zero exit so a missing CRD/namespace surfaces as a skip rather
    than an opaque failure.
    """
    result = await kubectl.run(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        selector,
        "-o",
        "jsonpath={.items[*].metadata.name}",
        check=False,
    )
    if result.returncode != 0:
        return []
    return [name for name in result.stdout.strip().split() if name]


async def _get_pod_ip(kubectl: KubectlClient, namespace: str, pod_name: str) -> str:
    """Return ``status.podIP`` for ``pod_name`` in ``namespace``.

    Raises ``RuntimeError`` with the pod identity in the message if the
    field is missing -- a Pending pod has no IP yet and the test cannot
    point Toxiproxy at it.
    """
    result = await kubectl.run(
        "get",
        "pod",
        pod_name,
        "-n",
        namespace,
        "-o",
        "jsonpath={.status.podIP}",
        check=True,
    )
    ip = result.stdout.strip()
    if not ip:
        raise RuntimeError(
            f"pod {namespace}/{pod_name} has no status.podIP; "
            f"D301 requires a Running decode pod to address NIXL"
        )
    return ip


async def test_d301_nixl_reset_peer_disagg(
    faults,
    kubectl: KubectlClient,
    dynamo_endpoint_url: str,
    dynamo_toxiproxy,
    dynamo_deployment_namespace: str,
) -> None:
    """NIXL ``reset_peer`` mid-KV-handoff -> client gets structured 500, decode survives.

    Wave-0 scaffold. Confirms:

    1. Disagg pre-conditions hold (one prefill + one decode pod labelled by the
       Dynamo operator) -- otherwise ``pytest.skip``.
    2. The Toxiproxy ``nixl-0`` proxy can be created against the live decode
       pod's IP on ``5600`` and the ``network.reset_peer`` fault applies and
       cleans up via the unified registry contract.

    The full behavioural assertion (streaming client receives the structured
    SSE error frame, decode pod is not in ``CrashLoopBackOff``, frontend
    ``dynamo_component_errors_total{error_type="response_stream"}``
    increments by exactly 1) is deferred until ``DynamoDeployer`` learns to
    redirect ``VLLM_NIXL_SIDE_CHANNEL_HOST`` to the Toxiproxy ``Service`` at
    deploy time. See module docstring.
    """
    # 1. Pre-condition: disagg deployment exists.
    decode_pods = await _list_pods_with_label(
        kubectl, dynamo_deployment_namespace, _DECODE_LABEL_SELECTOR
    )
    prefill_pods = await _list_pods_with_label(
        kubectl, dynamo_deployment_namespace, _PREFILL_LABEL_SELECTOR
    )
    if not decode_pods or not prefill_pods:
        pytest.skip(
            f"D301 requires disagg deployment; observed "
            f"prefill_pods={prefill_pods!r}, decode_pods={decode_pods!r} "
            f"in ns={dynamo_deployment_namespace!r}"
        )

    # 2. Resolve the decode worker pod IP so toxiproxy has an upstream.
    decode_pod_name = decode_pods[0]
    decode_ip = await _get_pod_ip(kubectl, dynamo_deployment_namespace, decode_pod_name)
    upstream = f"{decode_ip}:{NIXL_ENGINE_PORT}"
    logger.info(
        f"D301: NIXL side-channel upstream resolved to {upstream} "
        f"(decode pod {dynamo_deployment_namespace}/{decode_pod_name})"
    )

    # 3. Stand up the Toxiproxy proxy. ``finally``-bounded so a partial
    # setup never leaks across the suite.
    proxy_created = False
    try:
        await dynamo_toxiproxy.add_proxy(
            name=NIXL_PROXY_NAME,
            listen=NIXL_PROXY_LISTEN,
            upstream=upstream,
        )
        proxy_created = True

        # 4. Apply the ``reset_peer`` toxic via the unified registry. Empty
        # attributes -- reset_peer is parameterless in toxiproxy.
        async with faults.inject(
            "network.reset_peer",
            target={"proxy": NIXL_PROXY_NAME},
            attributes={},
        ) as applied:
            assert applied.spec.fault_id == "network.reset_peer"
            assert applied.metadata.get("proxy_name") == NIXL_PROXY_NAME

            # TODO(D301-followup): once VLLM_NIXL_SIDE_CHANNEL_HOST can be
            # pointed at the Toxiproxy Service at deploy time, perform a
            # streaming POST to ``{dynamo_endpoint_url}/chat/completions``
            # with a non-trivial prompt, sleep ~1s for prefill, then read
            # the SSE stream and assert:
            #   - error frame ``{"error": {"code": 500, ...}}`` then
            #     ``data: [DONE]``;
            #   - decode pod containerStatuses still Running (not
            #     CrashLoopBackOff);
            #   - frontend ``dynamo_component_errors_total{
            #     error_type="response_stream"}`` incremented by exactly 1.
            # See ``scrape_frontend_metrics`` in this package's conftest.
            pytest.skip(
                "D301 scaffold landed; full behavioural assertion needs "
                "DynamoDeployer to redirect VLLM_NIXL_SIDE_CHANNEL_HOST to "
                f"the Toxiproxy Service (dynamo_endpoint_url={dynamo_endpoint_url!r})"
            )
    finally:
        if proxy_created:
            try:
                await dynamo_toxiproxy.remove_proxy(NIXL_PROXY_NAME)
            except Exception as exc:
                # Per the chaos_common cleanup contract, restore failures
                # must not mask the original test signal -- the
                # package-scoped ``dynamo_toxiproxy.reset()`` in
                # conftest's ``faults`` teardown will catch any leak.
                logger.warning(lambda exc=exc: f"D301 remove_proxy failed: {exc!r}")
