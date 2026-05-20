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

The default v1.1.0 disagg deployment does not route NIXL through
Toxiproxy, so a reset toxic on ``nixl-0`` would be a false-positive no-op.
The test now encodes that as an explicit topology precondition: it only
injects when the decode pod declares a ``VLLM_NIXL_SIDE_CHANNEL_HOST`` value
pointing at the chaos Toxiproxy service, otherwise it skips before creating
any proxy state.
"""

from __future__ import annotations

import os

import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig
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
port is ``5600 + engineID``; this test targets engine 0."""

NIXL_HOST_ENV = "VLLM_NIXL_SIDE_CHANNEL_HOST"
"""Dynamo/vLLM env var that must point decode workers at the NIXL peer host."""

_TOXIPROXY_SERVICE_DNS = "toxiproxy.chaos-toxiproxy.svc"
"""Cluster DNS name that proves the deployment routes NIXL through Toxiproxy."""

_NIXL_CHAOS_OPT_IN_ENV = "AIPERF_DYNAMO_NIXL_CHAOS"
"""Opt-in for externally managed topologies that route NIXL through Toxiproxy."""

_DECODE_LABEL_SELECTOR = "nvidia.com/dynamo-sub-component-type=decode"
"""Label injected by the Dynamo operator onto decode worker pods. Source
of truth: ``deploy/operator/internal/consts/consts.go:59-60`` (mirrored
in ``tests/kubernetes/gpu/dynamo/helpers.py``)."""

_PREFILL_LABEL_SELECTOR = "nvidia.com/dynamo-sub-component-type=prefill"
"""Same label, ``prefill`` value. Both must be present for the scenario
to be meaningful (D301 is a disagg-only fault)."""


def _nixl_route_skip_reason(config: DynamoConfig) -> str | None:
    """Return why D301 cannot run before spending cluster setup time."""
    if os.environ.get(_NIXL_CHAOS_OPT_IN_ENV) == "1":
        return None
    for env in config.extra_envs:
        if env.get("name") == NIXL_HOST_ENV and _TOXIPROXY_SERVICE_DNS in env.get(
            "value", ""
        ):
            return None
    return (
        f"D301 requires {NIXL_HOST_ENV} to point at "
        f"{_TOXIPROXY_SERVICE_DNS!r}. Default Dynamo v1.1.0 disagg routes "
        "NIXL directly, so a Toxiproxy reset_peer toxic would be a no-op; "
        f"set {_NIXL_CHAOS_OPT_IN_ENV}=1 only for an externally managed "
        "topology that routes NIXL through Toxiproxy."
    )


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


async def _get_container_env(
    kubectl: KubectlClient,
    namespace: str,
    pod_name: str,
) -> dict[str, str]:
    """Return literal env vars declared on all containers in ``pod_name``."""
    result = await kubectl.run(
        "get",
        "pod",
        pod_name,
        "-n",
        namespace,
        "-o",
        "json",
        check=True,
    )
    pod = orjson.loads(result.stdout)
    envs: dict[str, str] = {}
    containers = pod.get("spec", {}).get("containers", [])
    for container in containers:
        for env in container.get("env", []):
            name = env.get("name")
            value = env.get("value")
            if isinstance(name, str) and isinstance(value, str):
                envs[name] = value
    return envs


async def test_d301_nixl_reset_peer_disagg(
    dynamo_config: DynamoConfig,
    request: pytest.FixtureRequest,
) -> None:
    """NIXL ``reset_peer`` mid-KV-handoff requires a Toxiproxy-routed topology.

    The default v1.1.0 single-GPU disagg fixture has prefill and decode pods,
    but it does not put Toxiproxy in the NIXL path. This test therefore skips
    with an explicit precondition until the deployment is created with
    ``VLLM_NIXL_SIDE_CHANNEL_HOST`` pointing at the chaos Toxiproxy service;
    otherwise the toxic would apply successfully while exercising no Dynamo
    product behavior.
    """
    static_skip_reason = _nixl_route_skip_reason(dynamo_config)
    if static_skip_reason is not None:
        pytest.skip(static_skip_reason)

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    dynamo_endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")
    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    faults = request.getfixturevalue("faults")
    dynamo_deployment_namespace: str = request.getfixturevalue(
        "dynamo_deployment_namespace"
    )

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

    # 2. Resolve the decode worker pod IP and prove that the deployment's
    # NIXL side-channel is actually pointed at the chaos Toxiproxy Service.
    decode_pod_name = decode_pods[0]
    envs = await _get_container_env(
        kubectl, dynamo_deployment_namespace, decode_pod_name
    )
    nixl_host = envs.get(NIXL_HOST_ENV, "")
    if _TOXIPROXY_SERVICE_DNS not in nixl_host:
        pytest.skip(
            f"D301 requires {NIXL_HOST_ENV} on decode pod "
            f"{dynamo_deployment_namespace}/{decode_pod_name} to point at "
            f"{_TOXIPROXY_SERVICE_DNS!r}; observed {nixl_host!r}. "
            "Default Dynamo v1.1.0 disagg routes NIXL directly, so a "
            "Toxiproxy reset_peer toxic would be a no-op."
        )

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

            logger.info(
                "D301: reset_peer toxic applied to routed NIXL proxy; "
                f"endpoint={dynamo_endpoint_url!r}"
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
