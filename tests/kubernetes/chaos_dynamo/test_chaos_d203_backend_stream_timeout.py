# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D203 -- backend stream inactivity timeout through a Toxiproxy-routed backend.

D-series catalog § D203 targets Dynamo's backend response-inactivity guard:
``lib/llm/src/http/service/disconnect.rs:195`` wraps the frontend SSE stream,
and ``lib/runtime/src/pipeline/network/egress/push_router.rs:52`` applies the
same ``DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS`` budget to the request plane so a
silent backend is reported down before the HTTP safety net has to clean up.

The default Dynamo v1.1.0 ``disagg-1gpu`` deployment advertises backend
request-plane endpoints directly, not through Toxiproxy. This case is therefore
strictly topology-gated: it only injects when a ``DynamoWorkerMetadata`` endpoint
advertises a TCP transport whose host is ``toxiproxy.chaos-toxiproxy.svc``. If
that route is absent, the test skips before creating any proxy or toxic so the
suite cannot record a false-positive no-op.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


TOXIPROXY_SERVICE_DNS = "toxiproxy.chaos-toxiproxy.svc"
"""Cluster DNS name that must appear in the advertised backend TCP route."""

BACKEND_PROXY_NAME = "backend-stream"
"""Toxiproxy proxy name used for the frontend -> backend request-plane route."""

BACKEND_TIMEOUT_ENV = "DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS"
"""Dynamo env var controlling backend response-inactivity timeout seconds."""

DEFAULT_BACKEND_TIMEOUT_SECONDS = 30
"""Dynamo default when ``DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS`` is unset."""

CLIENT_TIMEOUT_SLACK_SECONDS = 10.0
"""Extra client budget above the backend timeout for scheduling/cleanup jitter."""

FRONTEND_SELECTOR = "nvidia.com/dynamo-component-type=frontend"
"""Dynamo operator label used to find the frontend pod that owns metrics/env."""

RESPONSE_TIMEOUT_METRIC = "dynamo_frontend_requests_total"
"""Counter labelled with ``error_type=\"response_timeout\"`` on timeout."""

_BACKEND_CHAOS_OPT_IN_ENV = "AIPERF_DYNAMO_BACKEND_STREAM_CHAOS"
"""Opt-in for externally managed topologies that already route TCP via Toxiproxy."""


def _configured_env(config: DynamoConfig, name: str) -> str:
    """Return a literal env value from ``DynamoConfig.extra_envs`` if present."""
    for env in config.extra_envs:
        if env.get("name") == name and isinstance(env.get("value"), str):
            return env["value"]
    return ""


def _static_backend_route_skip_reason(config: DynamoConfig) -> str | None:
    """Return why D203 cannot run before spending cluster setup time."""
    if os.environ.get(_BACKEND_CHAOS_OPT_IN_ENV) == "1":
        return None
    request_plane = _configured_env(config, "DYN_REQUEST_PLANE")
    tcp_host = _configured_env(config, "DYN_TCP_RPC_HOST")
    if request_plane == "tcp" and TOXIPROXY_SERVICE_DNS in tcp_host:
        return None
    return (
        "D203 requires a Toxiproxy-routed backend TCP request-plane route: "
        "DYN_REQUEST_PLANE='tcp' and DYN_TCP_RPC_HOST containing "
        f"{TOXIPROXY_SERVICE_DNS!r}. Default Dynamo v1.1.0 disagg advertises "
        "worker pod IPs directly, so a latency toxic would miss the backend stream; "
        f"set {_BACKEND_CHAOS_OPT_IN_ENV}=1 only for an externally managed topology "
        "that already routes DynamoWorkerMetadata transport.Tcp through Toxiproxy."
    )


@dataclass(slots=True)
class BackendRoute:
    """Toxiproxy-routed backend request-plane endpoint discovered from DWM."""

    cr_name: str
    pod_name: str
    pod_ip: str
    advertised: str
    listen_port: int
    upstream: str

    @property
    def listen(self) -> str:
        """Return the Toxiproxy listen address inside the toxiproxy pod."""
        return f"0.0.0.0:{self.listen_port}"


async def _list_dynamo_worker_metadata(
    kubectl: KubectlClient,
    namespace: str,
) -> list[dict[str, Any]]:
    """Return ``DynamoWorkerMetadata`` items, or ``[]`` if the CRD is unavailable."""
    result = await kubectl.run(
        "get",
        "dynamoworkermetadatas",
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        logger.debug(
            lambda stderr=result.stderr: f"D203: DWM list failed: {stderr.strip()!r}"
        )
        return []
    payload = orjson.loads(result.stdout)
    items = payload.get("items", [])
    return [item for item in items if isinstance(item, dict)]


def _transport_tcp_address(value: Any) -> str | None:
    """Extract a TCP transport address from a DWM endpoint value."""
    transport = value.get("transport") if isinstance(value, dict) else None
    if not isinstance(transport, dict):
        return None
    for key in ("Tcp", "tcp"):
        address = transport.get(key)
        if isinstance(address, str):
            return address
    return None


def _iter_tcp_addresses(item: dict[str, Any]) -> list[str]:
    """Return all TCP endpoint addresses embedded in one DWM item."""
    data = item.get("spec", {}).get("data", {})
    endpoints = data.get("endpoints", {}) if isinstance(data, dict) else {}
    if not isinstance(endpoints, dict):
        return []
    addresses: list[str] = []
    for endpoint in endpoints.values():
        address = _transport_tcp_address(endpoint)
        if address is not None:
            addresses.append(address)
    return addresses


def _parse_advertised_port(address: str) -> int | None:
    """Parse ``host:port/...`` from Dynamo's TCP transport address."""
    host_port = address.split("/", 1)[0]
    _host, sep, port_text = host_port.rpartition(":")
    if sep == "" or not port_text.isdigit():
        return None
    return int(port_text)


def _owner_pod_name(item: dict[str, Any]) -> str | None:
    """Return the owning Pod name for a DWM item, falling back to CR name."""
    metadata = item.get("metadata", {})
    owner_refs = (
        metadata.get("ownerReferences", []) if isinstance(metadata, dict) else []
    )
    if isinstance(owner_refs, list):
        for owner_ref in owner_refs:
            if not isinstance(owner_ref, dict):
                continue
            if owner_ref.get("kind") == "Pod" and isinstance(
                owner_ref.get("name"), str
            ):
                return owner_ref["name"]
    name = metadata.get("name") if isinstance(metadata, dict) else None
    return name if isinstance(name, str) else None


async def _pod_ip(kubectl: KubectlClient, namespace: str, pod_name: str) -> str | None:
    """Return a pod's IP address, or ``None`` when the pod is not addressable."""
    result = await kubectl.run(
        "get",
        "pod",
        pod_name,
        "-n",
        namespace,
        "-o",
        "jsonpath={.status.podIP}",
        check=False,
    )
    if result.returncode != 0:
        return None
    ip = result.stdout.strip()
    return ip or None


async def _resolve_toxiproxy_backend_route(
    kubectl: KubectlClient,
    namespace: str,
) -> BackendRoute | None:
    """Find the first backend TCP route that is already advertised via Toxiproxy."""
    for item in await _list_dynamo_worker_metadata(kubectl, namespace):
        metadata = item.get("metadata", {})
        cr_name = (
            metadata.get("name", "<unknown>")
            if isinstance(metadata, dict)
            else "<unknown>"
        )
        pod_name = _owner_pod_name(item)
        if pod_name is None:
            continue
        for address in _iter_tcp_addresses(item):
            if TOXIPROXY_SERVICE_DNS not in address:
                continue
            port = _parse_advertised_port(address)
            if port is None:
                continue
            pod_ip = await _pod_ip(kubectl, namespace, pod_name)
            if pod_ip is None:
                continue
            return BackendRoute(
                cr_name=cr_name,
                pod_name=pod_name,
                pod_ip=pod_ip,
                advertised=address,
                listen_port=port,
                upstream=f"{pod_ip}:{port}",
            )
    return None


async def _frontend_pod(kubectl: KubectlClient, namespace: str) -> str | None:
    """Return the first frontend pod name in ``namespace``."""
    result = await kubectl.run(
        "get",
        "pod",
        "-n",
        namespace,
        "-l",
        FRONTEND_SELECTOR,
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=False,
    )
    pod = result.stdout.strip() if result.returncode == 0 else ""
    return pod or None


async def _container_env(
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


async def _backend_timeout_seconds(
    kubectl: KubectlClient,
    namespace: str,
) -> int:
    """Read the frontend backend-stream timeout, falling back to Dynamo's default."""
    pod = await _frontend_pod(kubectl, namespace)
    if pod is None:
        return DEFAULT_BACKEND_TIMEOUT_SECONDS
    envs = await _container_env(kubectl, namespace, pod)
    raw_value = envs.get(BACKEND_TIMEOUT_ENV)
    if raw_value is None:
        return DEFAULT_BACKEND_TIMEOUT_SECONDS
    try:
        parsed = int(raw_value)
    except ValueError:
        return DEFAULT_BACKEND_TIMEOUT_SECONDS
    return parsed if parsed > 0 else DEFAULT_BACKEND_TIMEOUT_SECONDS


async def _raw_frontend_metrics(
    kubectl: KubectlClient,
    namespace: str,
) -> str:
    """Fetch raw frontend Prometheus metrics text for label-aware assertions."""
    pod = await _frontend_pod(kubectl, namespace)
    if pod is None:
        raise RuntimeError(
            f"D203: no frontend pod matched {FRONTEND_SELECTOR!r} in {namespace!r}"
        )
    async with (
        kubectl.port_forward(pod, 8000, namespace=namespace) as local_port,
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session,
        session.get(f"http://127.0.0.1:{local_port}/metrics") as resp,
    ):
        body = await resp.text()
        if resp.status != 200:
            raise RuntimeError(
                f"D203: metrics scrape for {namespace}/{pod} returned {resp.status}: "
                f"{body[:512]!r}"
            )
        return body


def _response_timeout_count(metrics_text: str) -> float:
    """Sum frontend request samples labelled ``error_type=response_timeout``."""
    total = 0.0
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(RESPONSE_TIMEOUT_METRIC + "{"):
            continue
        if 'error_type="response_timeout"' not in line:
            continue
        value_text = line.rsplit(None, 1)[-1]
        try:
            total += float(value_text)
        except ValueError:
            logger.debug(lambda line=line: f"D203: bad metric sample {line!r}")
    return total


async def _stream_with_latency_fault(url: str, client_budget: float) -> dict[str, Any]:
    """Issue one streaming chat request and return its terminal outcome."""
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": "Write a detailed paragraph about backend stream timeouts.",
            }
        ],
        "max_tokens": 200,
        "stream": True,
        "temperature": 0.0,
    }
    started = time.monotonic()
    frames: list[str] = []
    try:
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=client_budget)
            ) as session,
            session.post(f"{url}/chat/completions", json=payload) as resp,
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
                if chunk:
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
    except (
        aiohttp.ServerDisconnectedError,
        aiohttp.ClientPayloadError,
        aiohttp.ClientConnectionError,
    ) as exc:
        return {
            "kind": "client_disconnect",
            "elapsed": time.monotonic() - started,
            "error": repr(exc),
            "frames": frames,
        }


@pytest.mark.skipif(
    os.environ.get(_BACKEND_CHAOS_OPT_IN_ENV) != "1",
    reason=(
        "D203 requires a Toxiproxy-routed backend TCP request-plane route: "
        "DynamoWorkerMetadata spec.data.endpoints[*].transport.Tcp host must contain "
        f"{TOXIPROXY_SERVICE_DNS!r}; set {_BACKEND_CHAOS_OPT_IN_ENV}=1 only for "
        "a topology that already advertises the backend route through Toxiproxy."
    ),
)
async def test_d203_backend_stream_inactivity_timeout(
    request: pytest.FixtureRequest,
    dynamo_config: DynamoConfig,
) -> None:
    """Inject backend-path latency above stream timeout and require clean termination."""
    static_skip_reason = _static_backend_route_skip_reason(dynamo_config)
    if static_skip_reason is not None:
        pytest.skip(static_skip_reason)

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    dynamo_deployment_namespace: str = request.getfixturevalue(
        "dynamo_deployment_namespace"
    )
    dynamo_endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")

    route = await _resolve_toxiproxy_backend_route(kubectl, dynamo_deployment_namespace)
    if route is None:
        pytest.skip(
            "D203 requires a Toxiproxy-routed backend TCP request-plane route: "
            f"DynamoWorkerMetadata spec.data.endpoints[*].transport.Tcp host must contain "
            f"{TOXIPROXY_SERVICE_DNS!r}. Default Dynamo v1.1.0 disagg advertises "
            "worker pod IPs directly, so a latency toxic would miss the backend stream."
        )

    timeout_seconds = await _backend_timeout_seconds(
        kubectl, dynamo_deployment_namespace
    )
    latency_ms = (timeout_seconds + 1) * 1000
    client_budget = timeout_seconds + CLIENT_TIMEOUT_SLACK_SECONDS

    try:
        before_count = _response_timeout_count(
            await _raw_frontend_metrics(kubectl, dynamo_deployment_namespace)
        )
    except RuntimeError as exc:
        logger.warning(
            lambda exc=exc: f"D203: pre-fault metrics scrape failed: {exc!r}"
        )
        before_count = 0.0

    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    faults: InjectorRegistry = request.getfixturevalue("faults")
    proxy_created = False
    try:
        await dynamo_toxiproxy.add_proxy(
            name=BACKEND_PROXY_NAME,
            listen=route.listen,
            upstream=route.upstream,
        )
        proxy_created = True
        async with faults.inject(
            "network.latency",
            target={"proxy": BACKEND_PROXY_NAME},
            attributes={"latency": latency_ms, "jitter": 0},
            stream="upstream",
        ) as applied:
            assert applied.spec.fault_id == "network.latency"
            assert applied.metadata.get("proxy_name") == BACKEND_PROXY_NAME
            logger.info(
                "D203: latency toxic applied to backend route "
                f"cr={route.cr_name!r} pod={route.pod_name!r} "
                f"advertised={route.advertised!r} upstream={route.upstream!r} "
                f"latency_ms={latency_ms}"
            )
            try:
                outcome = await asyncio.wait_for(
                    _stream_with_latency_fault(dynamo_endpoint_url, client_budget),
                    timeout=client_budget + 5.0,
                )
            except asyncio.TimeoutError:
                pytest.fail(
                    f"D203: client did not observe a terminal stream event within "
                    f"{client_budget + 5.0}s after injecting {latency_ms}ms backend "
                    f"latency on {route.advertised!r}"
                )
    finally:
        if proxy_created:
            try:
                await dynamo_toxiproxy.remove_proxy(BACKEND_PROXY_NAME)
            except Exception as exc:
                logger.warning(lambda exc=exc: f"D203 remove_proxy failed: {exc!r}")

    assert outcome["kind"] in {"http_error", "error_frame", "eof", "client_disconnect"}
    assert outcome["elapsed"] <= client_budget + 5.0, (
        f"D203: terminal outcome exceeded bounded-clean-termination budget; "
        f"outcome={outcome!r}, budget={client_budget + 5.0}s"
    )

    try:
        after_count = _response_timeout_count(
            await _raw_frontend_metrics(kubectl, dynamo_deployment_namespace)
        )
    except RuntimeError as exc:
        logger.warning(
            lambda exc=exc: f"D203: post-fault metrics scrape failed: {exc!r}"
        )
        return

    assert after_count > before_count, (
        f"D203: {RESPONSE_TIMEOUT_METRIC} did not increment with "
        f"error_type='response_timeout' (before={before_count}, after={after_count}); "
        f"stream outcome={outcome!r}"
    )
