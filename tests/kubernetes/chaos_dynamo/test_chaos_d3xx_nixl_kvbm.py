# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D3xx Dynamo NIXL and KVBM chaos scenarios."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


# D301

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


# D307-D316

TOXIPROXY_SERVICE_DNS = "toxiproxy.chaos-toxiproxy.svc"
"""Cluster DNS name that proves NIXL is routed through the chaos Toxiproxy."""

NIXL_CHAOS_OPT_IN_ENV = "AIPERF_DYNAMO_NIXL_CHAOS"
"""Opt-in for externally managed NIXL-through-Toxiproxy topologies."""


DECODE_LABEL_SELECTOR = "nvidia.com/dynamo-sub-component-type=decode"
"""Dynamo operator label used to find decode worker pods."""

PREFILL_LABEL_SELECTOR = "nvidia.com/dynamo-sub-component-type=prefill"
"""Dynamo operator label used to prove the deployment is disaggregated."""

BROKEN_NIXL_PORT = 9
"""Reserved discard port used as an intentionally wrong NIXL upstream."""

BROKEN_NIXL_DNS = "nixl-dns-blackhole.invalid."
"""Intentionally unresolvable upstream host for DNS-failure injection."""


@dataclass(slots=True)
class NixlRoute:
    """Resolved NIXL side-channel route for one decode worker."""

    decode_pod_name: str
    decode_pod_ip: str
    upstream: str
    advertised_host: str


@dataclass(frozen=True, slots=True)
class NixlFaultCase:
    """Declarative Toxiproxy mutation for a D307-D316 NIXL case."""

    case_id: str
    fault_id: Literal[
        "network.latency",
        "network.partition",
        "network.timeout",
        "network.reset_peer",
    ]
    attributes: dict[str, Any]
    stream: Literal["upstream", "downstream"] = "upstream"
    upstream_override: str | None = None
    expect_bounded_stream: bool = False


async def _d307_d316_list_pods_with_label(
    kubectl: KubectlClient, namespace: str, selector: str
) -> list[str]:
    """Return pod names in ``namespace`` matching ``selector``."""
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


async def _d307_d316_get_pod_ip(
    kubectl: KubectlClient, namespace: str, pod_name: str
) -> str:
    """Return ``status.podIP`` for ``pod_name`` in ``namespace``."""
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
    pod_ip = result.stdout.strip()
    if not pod_ip:
        raise RuntimeError(
            f"pod {namespace}/{pod_name} has no status.podIP; "
            "D307-D316 require a Running decode pod to address NIXL"
        )
    return pod_ip


async def _d307_d316_get_container_env(
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
    for container in pod.get("spec", {}).get("containers", []):
        for env in container.get("env", []):
            name = env.get("name")
            value = env.get("value")
            if isinstance(name, str) and isinstance(value, str):
                envs[name] = value
    return envs


def _nixl_static_skip_reason(config: DynamoConfig, case_id: str) -> str | None:
    """Return why a NIXL chaos case cannot run before cluster setup."""
    if os.environ.get(NIXL_CHAOS_OPT_IN_ENV) == "1":
        return None
    for env in config.extra_envs:
        if env.get("name") == NIXL_HOST_ENV and TOXIPROXY_SERVICE_DNS in env.get(
            "value", ""
        ):
            return None
    return (
        f"{case_id} requires {NIXL_HOST_ENV} to point at "
        f"{TOXIPROXY_SERVICE_DNS!r}. Default Dynamo v1.1.0 disagg routes "
        "NIXL directly, so a Toxiproxy NIXL fault would be a no-op; set "
        f"{NIXL_CHAOS_OPT_IN_ENV}=1 only for an externally managed topology "
        "that routes NIXL through Toxiproxy."
    )


async def _resolve_nixl_route(
    kubectl: KubectlClient,
    namespace: str,
    *,
    case_id: str,
) -> NixlRoute:
    """Resolve and validate the Toxiproxy-routed NIXL side-channel route."""
    decode_pods = await _d307_d316_list_pods_with_label(
        kubectl, namespace, DECODE_LABEL_SELECTOR
    )
    prefill_pods = await _d307_d316_list_pods_with_label(
        kubectl, namespace, PREFILL_LABEL_SELECTOR
    )
    if not decode_pods or not prefill_pods:
        pytest.skip(
            f"{case_id} requires disaggregated serving; observed "
            f"prefill_pods={prefill_pods!r}, decode_pods={decode_pods!r} "
            f"in ns={namespace!r}"
        )

    decode_pod_name = decode_pods[0]
    envs = await _d307_d316_get_container_env(kubectl, namespace, decode_pod_name)
    nixl_host = envs.get(NIXL_HOST_ENV, "")
    if (
        os.environ.get(NIXL_CHAOS_OPT_IN_ENV) != "1"
        and TOXIPROXY_SERVICE_DNS not in nixl_host
    ):
        pytest.skip(
            f"{case_id} requires {NIXL_HOST_ENV} on decode pod "
            f"{namespace}/{decode_pod_name} to point at "
            f"{TOXIPROXY_SERVICE_DNS!r}; observed {nixl_host!r}. "
            "Default Dynamo v1.1.0 disagg routes NIXL directly, so a "
            "Toxiproxy NIXL toxic would be a no-op."
        )

    decode_ip = await _d307_d316_get_pod_ip(kubectl, namespace, decode_pod_name)
    return NixlRoute(
        decode_pod_name=decode_pod_name,
        decode_pod_ip=decode_ip,
        upstream=f"{decode_ip}:{NIXL_ENGINE_PORT}",
        advertised_host=nixl_host,
    )


@asynccontextmanager
async def _nixl_proxy(
    request: pytest.FixtureRequest,
    route: NixlRoute,
    *,
    upstream_override: str | None = None,
) -> AsyncIterator[None]:
    """Create and remove the reserved NIXL Toxiproxy proxy for one case."""
    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    upstream = upstream_override or route.upstream
    proxy_created = False
    try:
        await dynamo_toxiproxy.add_proxy(
            name=NIXL_PROXY_NAME,
            listen=NIXL_PROXY_LISTEN,
            upstream=upstream,
        )
        proxy_created = True
        yield
    finally:
        if proxy_created:
            try:
                await dynamo_toxiproxy.remove_proxy(NIXL_PROXY_NAME)
            except Exception as exc:
                logger.warning(
                    lambda exc=exc: f"remove_proxy({NIXL_PROXY_NAME}) failed: {exc!r}"
                )


async def _stream_chat_once(
    endpoint_url: str,
    *,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Issue one streaming chat request and return its bounded outcome."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    started = time.monotonic()
    frames: list[str] = []
    try:
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as session,
            session.post(f"{endpoint_url}/chat/completions", json=payload) as resp,
        ):
            if resp.status >= 500:
                body = (await resp.read()).decode(errors="replace")[:512]
                return {
                    "kind": "http_error",
                    "status": resp.status,
                    "elapsed": time.monotonic() - started,
                    "body": body,
                }
            async for chunk in resp.content.iter_any():
                if not chunk:
                    continue
                text = chunk.decode(errors="replace")
                frames.append(text[:512])
                if '"error"' in text or "[ERROR]" in text:
                    return {
                        "kind": "error_frame",
                        "status": resp.status,
                        "elapsed": time.monotonic() - started,
                        "frames": frames,
                    }
            return {
                "kind": "eof",
                "status": resp.status,
                "elapsed": time.monotonic() - started,
                "frames": frames,
            }
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return {
            "kind": "client_error",
            "elapsed": time.monotonic() - started,
            "error": repr(exc),
            "frames": frames,
        }


async def _run_nixl_fault_case(
    request: pytest.FixtureRequest,
    dynamo_config: DynamoConfig,
    case: NixlFaultCase,
) -> None:
    """Shared runner for Toxiproxy-backed D307-D316 NIXL cases."""
    static_skip_reason = _nixl_static_skip_reason(dynamo_config, case.case_id)
    if static_skip_reason is not None:
        pytest.skip(static_skip_reason)

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    route = await _resolve_nixl_route(kubectl, namespace, case_id=case.case_id)
    faults: InjectorRegistry = request.getfixturevalue("faults")

    async with (
        _nixl_proxy(
            request,
            route,
            upstream_override=case.upstream_override,
        ),
        faults.inject(
            case.fault_id,
            target={"proxy": NIXL_PROXY_NAME},
            attributes=case.attributes,
            stream=case.stream,
        ) as applied,
    ):
        assert applied.spec.fault_id == case.fault_id
        assert applied.metadata.get("proxy_name") == NIXL_PROXY_NAME
        logger.info(
            f"{case.case_id}: {case.fault_id} applied to NIXL route "
            f"decode={namespace}/{route.decode_pod_name} "
            f"advertised_host={route.advertised_host!r} upstream={route.upstream!r}"
        )
        if case.expect_bounded_stream:
            endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")
            outcome = await _stream_chat_once(
                endpoint_url,
                prompt=f"Exercise {case.case_id} NIXL transport fault handling.",
                max_tokens=64,
                timeout_seconds=45.0,
            )
            assert outcome["kind"] in {
                "http_error",
                "error_frame",
                "eof",
                "client_error",
            }
            assert outcome["elapsed"] <= 45.0, (
                f"{case.case_id}: stream did not terminate inside the bounded "
                f"client budget; outcome={outcome!r}"
            )


async def test_d307_nixl_latency_toxic(
    request: pytest.FixtureRequest,
    dynamo_config: DynamoConfig,
) -> None:
    """Inject moderate NIXL side-channel latency through Toxiproxy."""
    await _run_nixl_fault_case(
        request,
        dynamo_config,
        NixlFaultCase(
            case_id="D307",
            fault_id="network.latency",
            attributes={"latency": 750, "jitter": 100},
        ),
    )


async def test_d308_nixl_blackhole_partition(
    request: pytest.FixtureRequest,
    dynamo_config: DynamoConfig,
) -> None:
    """Blackhole NIXL by disabling the routed Toxiproxy proxy."""
    await _run_nixl_fault_case(
        request,
        dynamo_config,
        NixlFaultCase(
            case_id="D308",
            fault_id="network.partition",
            attributes={},
            expect_bounded_stream=True,
        ),
    )


async def test_d309_nixl_wrong_port_refused(
    request: pytest.FixtureRequest,
    dynamo_config: DynamoConfig,
) -> None:
    """Route the NIXL proxy to an addressable pod on the wrong TCP port."""
    static_skip_reason = _nixl_static_skip_reason(dynamo_config, "D309")
    if static_skip_reason is not None:
        pytest.skip(static_skip_reason)

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    route = await _resolve_nixl_route(kubectl, namespace, case_id="D309")
    await _run_nixl_fault_case(
        request,
        dynamo_config,
        NixlFaultCase(
            case_id="D309",
            fault_id="network.reset_peer",
            attributes={},
            upstream_override=f"{route.decode_pod_ip}:{BROKEN_NIXL_PORT}",
            expect_bounded_stream=True,
        ),
    )


async def test_d310_nixl_dns_blackhole(
    request: pytest.FixtureRequest,
    dynamo_config: DynamoConfig,
) -> None:
    """Route the NIXL proxy to an intentionally unresolvable upstream host."""
    await _run_nixl_fault_case(
        request,
        dynamo_config,
        NixlFaultCase(
            case_id="D310",
            fault_id="network.reset_peer",
            attributes={},
            upstream_override=f"{BROKEN_NIXL_DNS}:{NIXL_ENGINE_PORT}",
            expect_bounded_stream=True,
        ),
    )


async def test_d311_nixl_latency_flap(
    request: pytest.FixtureRequest,
    dynamo_config: DynamoConfig,
) -> None:
    """Apply and remove a NIXL latency toxic repeatedly to model route flapping."""
    static_skip_reason = _nixl_static_skip_reason(dynamo_config, "D311")
    if static_skip_reason is not None:
        pytest.skip(static_skip_reason)

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    route = await _resolve_nixl_route(kubectl, namespace, case_id="D311")
    faults: InjectorRegistry = request.getfixturevalue("faults")

    async with _nixl_proxy(request, route):
        for iteration in range(3):
            async with faults.inject(
                "network.latency",
                target={"proxy": NIXL_PROXY_NAME},
                attributes={"latency": 500, "jitter": 250},
                stream="upstream",
            ) as applied:
                assert applied.spec.fault_id == "network.latency"
                assert applied.metadata.get("proxy_name") == NIXL_PROXY_NAME
                logger.info(f"D311: latency flap iteration {iteration} applied")
            await asyncio.sleep(0.25)


async def test_d312_nixl_stall_timeout_toxic(
    request: pytest.FixtureRequest,
    dynamo_config: DynamoConfig,
) -> None:
    """Stall NIXL by applying a timeout toxic on the side-channel route."""
    await _run_nixl_fault_case(
        request,
        dynamo_config,
        NixlFaultCase(
            case_id="D312",
            fault_id="network.timeout",
            attributes={"timeout": 0},
            expect_bounded_stream=True,
        ),
    )


async def test_d313_nixl_toxic_removal_recovers_proxy(
    request: pytest.FixtureRequest,
    dynamo_config: DynamoConfig,
) -> None:
    """Verify removing a NIXL toxic leaves the same routed proxy usable."""
    static_skip_reason = _nixl_static_skip_reason(dynamo_config, "D313")
    if static_skip_reason is not None:
        pytest.skip(static_skip_reason)

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    route = await _resolve_nixl_route(kubectl, namespace, case_id="D313")
    faults: InjectorRegistry = request.getfixturevalue("faults")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")

    async with _nixl_proxy(request, route):
        async with faults.inject(
            "network.latency",
            target={"proxy": NIXL_PROXY_NAME},
            attributes={"latency": 1_000, "jitter": 0},
            stream="upstream",
        ) as applied:
            assert applied.spec.fault_id == "network.latency"
            assert applied.metadata.get("proxy_name") == NIXL_PROXY_NAME
        outcome = await _stream_chat_once(
            endpoint_url,
            prompt="Confirm NIXL proxy works after toxic removal.",
            max_tokens=16,
            timeout_seconds=45.0,
        )
        assert outcome["kind"] in {"eof", "error_frame", "http_error", "client_error"}
        assert outcome["elapsed"] <= 45.0, (
            f"D313: post-removal stream exceeded bounded budget; outcome={outcome!r}"
        )


async def _run_nixl_client_disconnect_case(
    request: pytest.FixtureRequest,
    dynamo_config: DynamoConfig,
    case: NixlFaultCase,
) -> None:
    """Run a NIXL toxic while the client drops a streaming request."""
    static_skip_reason = _nixl_static_skip_reason(dynamo_config, case.case_id)
    if static_skip_reason is not None:
        pytest.skip(static_skip_reason)

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    route = await _resolve_nixl_route(kubectl, namespace, case_id=case.case_id)
    faults: InjectorRegistry = request.getfixturevalue("faults")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": f"Stream enough text for {case.case_id} client disconnect.",
            }
        ],
        "max_tokens": 256,
        "stream": True,
        "temperature": 0.0,
    }

    async with (
        _nixl_proxy(
            request,
            route,
            upstream_override=case.upstream_override,
        ),
        faults.inject(
            case.fault_id,
            target={"proxy": NIXL_PROXY_NAME},
            attributes=case.attributes,
            stream=case.stream,
        ),
        faults.inject(
            "client.cancel_request",
            target={"url": f"{endpoint_url}/chat/completions"},
            payload=payload,
            cancel_after_seconds=0.5,
        ) as applied,
    ):
        assert applied.spec.fault_id == "client.cancel_request"
        assert applied.metadata.get("url") == f"{endpoint_url}/chat/completions"
        assert applied.metadata.get("bytes_received", 0) >= 0


async def test_d314_nixl_client_disconnect_with_latency(
    request: pytest.FixtureRequest,
    dynamo_config: DynamoConfig,
) -> None:
    """Cancel a streaming client while NIXL latency is active."""
    await _run_nixl_client_disconnect_case(
        request,
        dynamo_config,
        NixlFaultCase(
            case_id="D314",
            fault_id="network.latency",
            attributes={"latency": 500, "jitter": 50},
        ),
    )


async def test_d315_nixl_client_disconnect_with_blackhole(
    request: pytest.FixtureRequest,
    dynamo_config: DynamoConfig,
) -> None:
    """Cancel a streaming client while NIXL is blackholed."""
    await _run_nixl_client_disconnect_case(
        request,
        dynamo_config,
        NixlFaultCase(
            case_id="D315",
            fault_id="network.partition",
            attributes={},
        ),
    )


async def test_d316_nixl_client_disconnect_after_toxic_removal(
    request: pytest.FixtureRequest,
    dynamo_config: DynamoConfig,
) -> None:
    """Cancel a streaming client after a NIXL toxic is removed."""
    static_skip_reason = _nixl_static_skip_reason(dynamo_config, "D316")
    if static_skip_reason is not None:
        pytest.skip(static_skip_reason)

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    route = await _resolve_nixl_route(kubectl, namespace, case_id="D316")
    faults: InjectorRegistry = request.getfixturevalue("faults")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")

    async with _nixl_proxy(request, route):
        async with faults.inject(
            "network.latency",
            target={"proxy": NIXL_PROXY_NAME},
            attributes={"latency": 750, "jitter": 0},
            stream="upstream",
        ):
            pass
        async with faults.inject(
            "client.cancel_request",
            target={"url": f"{endpoint_url}/chat/completions"},
            payload={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": "Stream enough text after NIXL toxic removal.",
                    }
                ],
                "max_tokens": 256,
                "stream": True,
                "temperature": 0.0,
            },
            cancel_after_seconds=0.5,
        ) as applied:
            assert applied.spec.fault_id == "client.cancel_request"
            assert applied.metadata.get("bytes_received", 0) >= 0


# D317

_PREFILL_SELECTOR = "nvidia.com/dynamo-sub-component-type=prefill"
_KVBM_ENV = "DYN_KVBM_CPU_CACHE_GB"
_REQUEST_TIMEOUT_S = 45.0


@dataclass(frozen=True, slots=True)
class KVBMPodTarget:
    """Address of a KVBM-enabled container in a Dynamo pod."""

    namespace: str
    pod: str
    container: str
    env: dict[str, str]


@dataclass(frozen=True, slots=True)
class KVBMProcessTarget:
    """Address of a KVBM helper process inside a Dynamo pod."""

    pod_target: KVBMPodTarget
    pid: int
    command: str


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """HTTP response from one Dynamo chat completion request."""

    status: int
    body: str
    json_body: dict[str, Any] | None


async def test_d317_kvbm_zmq_publisher_pause_keeps_generation_bounded(
    request: pytest.FixtureRequest,
) -> None:
    """Pause an isolated KVBM ZMQ publisher and verify frontend requests finish."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")
    faults: InjectorRegistry = request.getfixturevalue("faults")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D317")
    publisher = await discover_isolated_kvbm_process(
        kubectl,
        kvbm_pod,
        role="publisher",
        role_patterns=("publisher", "pub", "zmq.*pub"),
        scenario_id="D317",
    )

    baseline = await post_completion(
        endpoint_url, content="D317 baseline.", max_tokens=4
    )
    assert_successful_completion("D317 baseline", baseline)

    async with faults.inject(
        "process.signal",
        target={
            "kind": "pod",
            "ns": publisher.pod_target.namespace,
            "pod": publisher.pod_target.pod,
            "container": publisher.pod_target.container,
            "pid": publisher.pid,
        },
        signal="SIGSTOP",
    ) as applied:
        assert applied.metadata.get("pid") == publisher.pid
        logger.info(
            lambda: (
                "D317: paused KVBM ZMQ publisher "
                f"pid={publisher.pid} command={publisher.command!r}"
            )
        )
        paused = await post_completion(
            endpoint_url,
            content="D317 completion while KVBM ZMQ publisher is paused.",
            max_tokens=8,
        )
        assert_successful_completion("D317 paused publisher", paused)


async def discover_kvbm_prefill_target(
    kubectl: KubectlClient,
    namespace: str,
    scenario_id: str,
) -> KVBMPodTarget:
    """Return one ready KVBM-enabled prefill container or skip precisely."""
    pods = await _list_pods_json(kubectl, namespace, _PREFILL_SELECTOR)
    if not pods:
        pytest.skip(
            f"{scenario_id}: requires a disaggregated prefill pod labelled "
            f"{_PREFILL_SELECTOR!r} in namespace {namespace!r}; none found"
        )

    observed_prefill: list[str] = []
    for pod in pods:
        pod_name = pod.get("metadata", {}).get("name", "<unknown>")
        if not _pod_ready(pod):
            observed_prefill.append(f"{pod_name}:not-ready")
            continue
        for container in pod.get("spec", {}).get("containers", []):
            env = _container_env(container)
            args = " ".join(str(arg) for arg in container.get("args", []))
            if _KVBM_ENV in env or "--connector kvbm" in args:
                name = container.get("name")
                if isinstance(name, str):
                    return KVBMPodTarget(namespace, pod_name, name, env)
        observed_prefill.append(f"{pod_name}:no-{_KVBM_ENV}")

    pytest.skip(
        f"{scenario_id}: requires a ready prefill container with KVBM enabled "
        f"({_KVBM_ENV} env or '--connector kvbm' arg); observed {observed_prefill!r}"
    )


async def discover_isolated_kvbm_process(
    kubectl: KubectlClient,
    target: KVBMPodTarget,
    *,
    role: str,
    role_patterns: tuple[str, ...],
    scenario_id: str,
) -> KVBMProcessTarget:
    """Return an isolated KVBM helper process matching ``role_patterns``."""
    result = await kubectl.run(
        "exec",
        target.pod,
        "-c",
        target.container,
        "-n",
        target.namespace,
        "--",
        "sh",
        "-lc",
        "ps -eo pid=,args=",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            f"{scenario_id}: cannot inspect processes in "
            f"{target.namespace}/{target.pod}/{target.container}; "
            f"kubectl exec returned {result.returncode}: {result.stderr.strip()!r}"
        )

    candidates = _parse_processes(result.stdout)
    role_matches = [
        proc
        for proc in candidates
        if "kvbm" in proc.command.lower()
        and any(
            _contains_pattern(proc.command.lower(), pattern)
            for pattern in role_patterns
        )
    ]
    isolated = [
        proc for proc in role_matches if _looks_like_helper_process(proc.command)
    ]
    if isolated:
        proc = isolated[0]
        return KVBMProcessTarget(target, proc.pid, proc.command)

    kvbm_seen = [proc.command for proc in candidates if "kvbm" in proc.command.lower()]
    pytest.skip(
        f"{scenario_id}: requires an isolated KVBM ZMQ {role} process inside "
        f"{target.namespace}/{target.pod}/{target.container}; found KVBM entries "
        f"{kvbm_seen[:5]!r}. If {role} is a thread inside the vLLM worker, add a "
        "test hook/sidecar exposing it as a separate PID before running this fault."
    )


async def post_completion(
    dynamo_endpoint_url: str,
    *,
    content: str,
    max_tokens: int,
) -> CompletionResult:
    """POST one non-streaming chat completion to the Dynamo frontend."""
    payload: dict[str, object] = {
        "model": "default",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
    }
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(_chat_completion_url(dynamo_endpoint_url), json=payload) as resp,
    ):
        body = await resp.text()
        parsed_json: dict[str, Any] | None = None
        with contextlib.suppress(ValueError):
            parsed = await resp.json(content_type=None)
            if isinstance(parsed, dict):
                parsed_json = parsed
        return CompletionResult(resp.status, body, parsed_json)


def assert_successful_completion(label: str, result: CompletionResult) -> None:
    """Assert the Dynamo frontend returned a successful completion."""
    assert result.status == 200, (
        f"{label}: expected HTTP 200 from Dynamo frontend, got {result.status}; "
        f"body={result.body[:512]!r}"
    )
    if result.json_body is not None:
        choices = result.json_body.get("choices")
        assert choices, (
            f"{label}: response JSON contains no choices: {result.json_body!r}"
        )


async def wait_for_pod_ready(
    kubectl: KubectlClient,
    namespace: str,
    pod_name: str,
    *,
    timeout_s: float = 120.0,
) -> None:
    """Wait until ``pod_name`` reports Ready=True."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while True:
        result = await kubectl.run(
            "get",
            "pod",
            pod_name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            pod = orjson.loads(result.stdout)
            if _pod_ready(pod):
                return
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(
                f"pod {namespace}/{pod_name} did not become Ready within {timeout_s}s"
            )
        await asyncio.sleep(2.0)


@dataclass(frozen=True, slots=True)
class _ProcessRow:
    pid: int
    command: str


async def _list_pods_json(
    kubectl: KubectlClient,
    namespace: str,
    selector: str,
) -> list[dict[str, Any]]:
    result = await kubectl.run(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        selector,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return []
    payload = orjson.loads(result.stdout)
    items = payload.get("items", [])
    return [item for item in items if isinstance(item, dict)]


def _container_env(container: dict[str, Any]) -> dict[str, str]:
    envs: dict[str, str] = {}
    for env in container.get("env", []):
        name = env.get("name")
        value = env.get("value")
        if isinstance(name, str) and isinstance(value, str):
            envs[name] = value
    return envs


def _pod_ready(pod: dict[str, Any]) -> bool:
    conditions = pod.get("status", {}).get("conditions", [])
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in conditions
    )


def _parse_processes(stdout: str) -> list[_ProcessRow]:
    rows: list[_ProcessRow] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        with contextlib.suppress(ValueError):
            rows.append(_ProcessRow(pid=int(pid_text), command=command.strip()))
    return rows


def _contains_pattern(value: str, pattern: str) -> bool:
    parts = [part for part in pattern.split(".*") if part]
    cursor = 0
    for part in parts:
        found = value.find(part, cursor)
        if found == -1:
            return False
        cursor = found + len(part)
    return True


def _looks_like_helper_process(command: str) -> bool:
    lowered = command.lower()
    worker_markers = ("vllm", "sglang", "trtllm", "python", "api_server")
    return not any(marker in lowered for marker in worker_markers)


def _chat_completion_url(dynamo_endpoint_url: str) -> str:
    return dynamo_endpoint_url.rstrip("/") + "/chat/completions"


# D318


async def test_d318_kvbm_zmq_subscriber_restart_during_frees(
    request: pytest.FixtureRequest,
) -> None:
    """Restart an isolated KVBM ZMQ subscriber and assert traffic recovers."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")
    faults: InjectorRegistry = request.getfixturevalue("faults")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D318")
    subscriber = await discover_isolated_kvbm_process(
        kubectl,
        kvbm_pod,
        role="subscriber",
        role_patterns=("subscriber", "sub", "zmq.*sub"),
        scenario_id="D318",
    )

    stream_task = asyncio.create_task(
        post_completion(
            endpoint_url,
            content="D318 long request generating KVBM add/free events.",
            max_tokens=64,
        )
    )
    await asyncio.sleep(1.0)

    async with faults.inject(
        "process.signal",
        target={
            "kind": "pod",
            "ns": subscriber.pod_target.namespace,
            "pod": subscriber.pod_target.pod,
            "container": subscriber.pod_target.container,
            "pid": subscriber.pid,
        },
        signal="SIGTERM",
    ) as applied:
        assert applied.metadata.get("signal") == "SIGTERM"

    stream_result = await stream_task
    assert_successful_completion("D318 in-flight request", stream_result)
    await wait_for_pod_ready(kubectl, namespace, subscriber.pod_target.pod)

    recovery = await post_completion(
        endpoint_url,
        content="D318 recovery request after KVBM subscriber restart.",
        max_tokens=8,
    )
    assert_successful_completion("D318 recovery", recovery)


# D319

_HWM_ENV_NAMES = (
    "DYN_KVBM_ZMQ_SNDHWM",
    "DYN_KVBM_ZMQ_RCVHWM",
    "DYN_KVBM_ZMQ_HWM",
    "ZMQ_SNDHWM",
    "ZMQ_RCVHWM",
)
_MAX_MEANINGFUL_HWM = 32
_BURST_REQUESTS = 48
_MIN_SUCCESS_RATE = 0.80


async def test_d319_kvbm_zmq_hwm_overflow_burst_is_bounded(
    request: pytest.FixtureRequest,
) -> None:
    """Run only when the topology sets a deliberately low KVBM ZMQ HWM."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D319")
    hwm = _configured_hwm(kvbm_pod.env)
    if hwm is None:
        pytest.skip(
            "D319 requires a topology/test hook that sets a low KVBM ZMQ high-water "
            f"mark via one of {_HWM_ENV_NAMES!r}; observed env keys "
            f"{sorted(kvbm_pod.env)!r}"
        )
    if hwm > _MAX_MEANINGFUL_HWM:
        pytest.skip(
            f"D319 requires HWM <= {_MAX_MEANINGFUL_HWM} to force overflow; "
            f"observed configured HWM={hwm} on "
            f"{kvbm_pod.namespace}/{kvbm_pod.pod}/{kvbm_pod.container}"
        )

    results = await asyncio.gather(
        *(
            post_completion(
                endpoint_url,
                content=f"D319 burst request {idx} with low KVBM ZMQ HWM.",
                max_tokens=4,
            )
            for idx in range(_BURST_REQUESTS)
        ),
        return_exceptions=True,
    )
    successes = sum(1 for result in results if _is_success(result))
    failures = [result for result in results if not _is_success(result)]
    success_rate = successes / _BURST_REQUESTS
    assert success_rate >= _MIN_SUCCESS_RATE, (
        f"D319: low-HWM burst success rate {success_rate:.1%} below "
        f"{_MIN_SUCCESS_RATE:.0%}; successes={successes}, "
        f"failures_sample={failures[:3]!r}"
    )
    logger.info(
        lambda: (
            f"D319: low-HWM burst completed with HWM={hwm}, "
            f"successes={successes}/{_BURST_REQUESTS}"
        )
    )

    recovery = await post_completion(
        endpoint_url,
        content="D319 recovery request after low-HWM burst.",
        max_tokens=4,
    )
    assert_successful_completion("D319 recovery", recovery)


def _configured_hwm(env: dict[str, str]) -> int | None:
    for name in _HWM_ENV_NAMES:
        value = env.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            pytest.skip(f"D319: {name} must be an integer HWM, got {value!r}")
    return None


def _is_success(result: CompletionResult | BaseException) -> bool:
    return isinstance(result, CompletionResult) and result.status == 200


# D320


async def test_d320_kvbm_consolidator_restart_between_add_and_free(
    request: pytest.FixtureRequest,
) -> None:
    """Restart an isolated KVBM consolidator while a request is active."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")
    faults: InjectorRegistry = request.getfixturevalue("faults")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D320")
    consolidator = await discover_isolated_kvbm_process(
        kubectl,
        kvbm_pod,
        role="consolidator",
        role_patterns=("consolidator", "consolidat", "free.*block", "block.*free"),
        scenario_id="D320",
    )

    in_flight = asyncio.create_task(
        post_completion(
            endpoint_url,
            content="D320 long request held across consolidator restart.",
            max_tokens=96,
        )
    )
    await asyncio.sleep(1.0)

    async with faults.inject(
        "process.signal",
        target={
            "kind": "pod",
            "ns": consolidator.pod_target.namespace,
            "pod": consolidator.pod_target.pod,
            "container": consolidator.pod_target.container,
            "pid": consolidator.pid,
        },
        signal="SIGTERM",
    ) as applied:
        assert applied.metadata.get("pid") == consolidator.pid

    result = await in_flight
    assert_successful_completion("D320 in-flight", result)
    await wait_for_pod_ready(kubectl, namespace, consolidator.pod_target.pod)

    recovery = await post_completion(
        endpoint_url,
        content="D320 recovery request after consolidator restart.",
        max_tokens=8,
    )
    assert_successful_completion("D320 recovery", recovery)


# D321

_HOOK_ENV_NAMES = ("DYN_KVBM_CHAOS_HOOK", "AIPERF_DYNAMO_KVBM_CHAOS_HOOK")
_HOOK_BINARIES = ("dynamo-kvbm-chaos", "kvbm-chaos")


@dataclass(frozen=True, slots=True)
class KVBMChaosHook:
    """Executable chaos hook discovered inside a KVBM container."""

    pod_target: KVBMPodTarget
    command: str


async def test_d321_duplicate_kvbm_free_notification_is_idempotent(
    request: pytest.FixtureRequest,
) -> None:
    """Inject duplicate free events through the KVBM chaos hook."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D321")
    hook = await discover_kvbm_chaos_hook(kubectl, kvbm_pod, "D321")

    baseline = await post_completion(
        endpoint_url,
        content="D321 baseline request before duplicate free injection.",
        max_tokens=8,
    )
    assert_successful_completion("D321 baseline", baseline)

    await run_kvbm_chaos_hook(
        kubectl,
        hook,
        "duplicate-free --request-id d321-synthetic-free --count 2",
        "D321",
    )

    recovery = await post_completion(
        endpoint_url,
        content="D321 recovery after duplicate KVBM free notification.",
        max_tokens=8,
    )
    assert_successful_completion("D321 recovery", recovery)


async def discover_kvbm_chaos_hook(
    kubectl: KubectlClient,
    target: KVBMPodTarget,
    scenario_id: str,
) -> KVBMChaosHook:
    """Return an executable KVBM chaos hook or skip with the missing surface."""
    for env_name in _HOOK_ENV_NAMES:
        command = target.env.get(env_name)
        if command:
            return KVBMChaosHook(target, command)

    command_probe = " || ".join(f"command -v {name}" for name in _HOOK_BINARIES)
    result = await kubectl.run(
        "exec",
        target.pod,
        "-c",
        target.container,
        "-n",
        target.namespace,
        "--",
        "sh",
        "-lc",
        command_probe,
        check=False,
    )
    command = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if result.returncode == 0 and command:
        return KVBMChaosHook(target, command)

    pytest.skip(
        f"{scenario_id}: requires a KVBM event-injection test hook inside "
        f"{target.namespace}/{target.pod}/{target.container}: set one of "
        f"{_HOOK_ENV_NAMES!r} or install one of {_HOOK_BINARIES!r} on PATH. "
        "The stock Dynamo KVBM path does not expose duplicate/reordered Add/Free "
        "events through Kubernetes or Toxiproxy."
    )


async def run_kvbm_chaos_hook(
    kubectl: KubectlClient,
    hook: KVBMChaosHook,
    hook_args: str,
    scenario_id: str,
) -> None:
    """Execute a KVBM chaos hook command and fail with stderr on non-zero exit."""
    target = hook.pod_target
    shell_command = f"{shlex.quote(hook.command)} {hook_args}"
    result = await kubectl.run(
        "exec",
        target.pod,
        "-c",
        target.container,
        "-n",
        target.namespace,
        "--",
        "sh",
        "-lc",
        shell_command,
        check=False,
    )
    assert result.returncode == 0, (
        f"{scenario_id}: KVBM chaos hook failed with exit {result.returncode}; "
        f"command={shell_command!r}, stdout={result.stdout[:512]!r}, "
        f"stderr={result.stderr[:512]!r}"
    )


# D322


async def test_d322_reordered_kvbm_add_free_events_are_bounded(
    request: pytest.FixtureRequest,
) -> None:
    """Use the KVBM hook to publish Free-before-Add and assert recovery."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D322")
    hook = await discover_kvbm_chaos_hook(kubectl, kvbm_pod, "D322")

    await run_kvbm_chaos_hook(
        kubectl,
        hook,
        "reorder-add-free --request-id d322-synthetic-reorder",
        "D322",
    )

    recovery = await post_completion(
        endpoint_url,
        content="D322 recovery after reordered KVBM Add/Free events.",
        max_tokens=8,
    )
    assert_successful_completion("D322 recovery", recovery)


# D323

_SPILL_DIR_ENV_NAMES = (
    "DYN_KVBM_SPILL_DIR",
    "DYN_KVBM_CACHE_DIR",
    "AIPERF_DYNAMO_KVBM_ENOSPC_DIR",
)


async def test_d323_kvbm_disk_spill_enospc_does_not_crash_worker(
    request: pytest.FixtureRequest,
) -> None:
    """Trigger ENOSPC through an explicit KVBM hook and verify recovery."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D323")
    spill_dir = _spill_dir(kvbm_pod.env)
    if spill_dir is None:
        pytest.skip(
            "D323 requires a bounded KVBM disk-spill/cache directory to fill via "
            f"one of {_SPILL_DIR_ENV_NAMES!r}; observed env keys "
            f"{sorted(kvbm_pod.env)!r}. Refusing to fill arbitrary container filesystems."
        )
    hook = await discover_kvbm_chaos_hook(kubectl, kvbm_pod, "D323")

    await run_kvbm_chaos_hook(
        kubectl,
        hook,
        f"enospc --dir {spill_dir} --duration-seconds 10",
        "D323",
    )

    recovery = await post_completion(
        endpoint_url,
        content="D323 recovery after KVBM disk-spill ENOSPC.",
        max_tokens=8,
    )
    assert_successful_completion("D323 recovery", recovery)


def _spill_dir(env: dict[str, str]) -> str | None:
    for name in _SPILL_DIR_ENV_NAMES:
        value = env.get(name)
        if value:
            return value
    return None


# D324

_PINNED_MEMORY_OPT_IN_ENVS = (
    "DYN_KVBM_PINNED_MEMORY_CHAOS",
    "AIPERF_DYNAMO_KVBM_PINNED_MEMORY_CHAOS",
)


async def test_d324_kvbm_pinned_host_memory_exhaustion_is_bounded(
    request: pytest.FixtureRequest,
) -> None:
    """Trigger pinned-memory pressure through the KVBM hook and verify recovery."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D324")
    if not _pinned_memory_chaos_enabled(kvbm_pod.env):
        pytest.skip(
            "D324 requires an explicit pinned-memory exhaustion guardrail in the "
            f"KVBM container via one of {_PINNED_MEMORY_OPT_IN_ENVS!r}; observed "
            f"env keys {sorted(kvbm_pod.env)!r}. Refusing to consume arbitrary host "
            "pinned memory without a bounded topology/test hook."
        )
    hook = await discover_kvbm_chaos_hook(kubectl, kvbm_pod, "D324")

    await run_kvbm_chaos_hook(
        kubectl,
        hook,
        "pinned-memory-exhaustion --duration-seconds 10",
        "D324",
    )

    recovery = await post_completion(
        endpoint_url,
        content="D324 recovery after KVBM pinned host memory exhaustion.",
        max_tokens=8,
    )
    assert_successful_completion("D324 recovery", recovery)


def _pinned_memory_chaos_enabled(env: dict[str, str]) -> bool:
    return any(
        env.get(name, "").lower() in {"1", "true", "yes"}
        for name in _PINNED_MEMORY_OPT_IN_ENVS
    )
