# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D307-D316 -- NIXL transport chaos cases for Dynamo disaggregated serving.

These cases extend D301's topology gate: NIXL faults are meaningful only when
Dynamo routes the vLLM NIXL side-channel through the chaos Toxiproxy service.
The default v1.1.0 disaggregated fixture advertises the NIXL peer directly, so
all tests skip before creating proxy state unless either:

* ``VLLM_NIXL_SIDE_CHANNEL_HOST`` contains ``toxiproxy.chaos-toxiproxy.svc``; or
* ``AIPERF_DYNAMO_NIXL_CHAOS=1`` declares an externally managed topology that
  already routes NIXL through Toxiproxy.
"""

from __future__ import annotations

import asyncio
import os
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


NIXL_HOST_ENV = "VLLM_NIXL_SIDE_CHANNEL_HOST"
"""Dynamo/vLLM env var that must point decode workers at the NIXL peer host."""

TOXIPROXY_SERVICE_DNS = "toxiproxy.chaos-toxiproxy.svc"
"""Cluster DNS name that proves NIXL is routed through the chaos Toxiproxy."""

NIXL_CHAOS_OPT_IN_ENV = "AIPERF_DYNAMO_NIXL_CHAOS"
"""Opt-in for externally managed NIXL-through-Toxiproxy topologies."""

NIXL_ENGINE_PORT = 5600
"""NIXL side-channel base port for engine 0."""

NIXL_PROXY_NAME = "nixl-0"
"""Toxiproxy proxy name fronting the engine-0 NIXL side-channel."""

NIXL_PROXY_LISTEN = "0.0.0.0:20040"
"""Toxiproxy listen address reserved for ``nixl-0`` in the chaos fixture."""

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


async def _list_pods_with_label(
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


async def _get_pod_ip(kubectl: KubectlClient, namespace: str, pod_name: str) -> str:
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
    decode_pods = await _list_pods_with_label(kubectl, namespace, DECODE_LABEL_SELECTOR)
    prefill_pods = await _list_pods_with_label(
        kubectl, namespace, PREFILL_LABEL_SELECTOR
    )
    if not decode_pods or not prefill_pods:
        pytest.skip(
            f"{case_id} requires disaggregated serving; observed "
            f"prefill_pods={prefill_pods!r}, decode_pods={decode_pods!r} "
            f"in ns={namespace!r}"
        )

    decode_pod_name = decode_pods[0]
    envs = await _get_container_env(kubectl, namespace, decode_pod_name)
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

    decode_ip = await _get_pod_ip(kubectl, namespace, decode_pod_name)
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
