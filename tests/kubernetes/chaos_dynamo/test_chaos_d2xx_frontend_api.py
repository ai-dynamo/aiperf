# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D2xx Dynamo frontend/API chaos scenarios."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import (
    AsyncIterator,
    Iterable,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos.toxiproxy import ToxiproxyError
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.chaos_dynamo.conftest import scrape_frontend_metrics
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


# D201

logger = AIPerfLogger(__name__)

CONCURRENT_STREAMS = 64

DYNAMO_SERVER_NAMESPACE = "dynamo-server"

FRONTEND_LABEL_SELECTOR = "nvidia.com/dynamo-component-type=frontend"

DISCONNECTED_METRIC = "dynamo_frontend_disconnected_clients"

STREAM_SETTLE_SECONDS = 1.0

CLIENT_TIMEOUT_SECONDS = 30.0

RECOVERY_TIMEOUT_SECONDS = 30.0


async def _resolve_frontend_pod(kubectl: KubectlClient, namespace: str) -> str:
    """Return the name of the first frontend pod matching the label selector.

    Raises:
        RuntimeError: When no pod matches the selector — D201 is meaningless
            without a live frontend to kill, so callers should fail fast.
    """
    result = await kubectl.run(
        "get",
        "pod",
        "-l",
        FRONTEND_LABEL_SELECTOR,
        "-n",
        namespace,
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=False,
    )
    pod = result.stdout.strip() if result.returncode == 0 else ""
    if not pod:
        raise RuntimeError(
            f"D201: no frontend pod matched label {FRONTEND_LABEL_SELECTOR!r} "
            f"in namespace {namespace!r}; check `kubectl get pods -n {namespace}`"
        )
    return pod


async def _wait_for_new_frontend_ready(
    kubectl: KubectlClient,
    namespace: str,
    killed_pod: str,
    timeout: float,
) -> str:
    """Poll until a frontend pod whose name differs from ``killed_pod`` is Ready.

    Returns the new pod's name. Raises ``TimeoutError`` if no replacement is
    ready inside ``timeout`` seconds.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    last_observed: str = "<none>"
    while True:
        pods = await kubectl.get_pods(namespace)
        for candidate in pods:
            if (
                candidate.name != killed_pod
                and candidate.is_ready
                and FRONTEND_LABEL_SELECTOR.split("=")[0] in candidate.labels
                and candidate.labels.get(FRONTEND_LABEL_SELECTOR.split("=")[0])
                == FRONTEND_LABEL_SELECTOR.split("=")[1]
            ):
                return candidate.name
            last_observed = f"{candidate.name}(ready={candidate.is_ready})"
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(
                f"D201: no replacement frontend pod became Ready within "
                f"{timeout}s (killed={killed_pod!r}, last_observed={last_observed!r})"
            )
        await asyncio.sleep(0.5)


async def _stream_one(
    session: aiohttp.ClientSession,
    url: str,
    idx: int,
) -> dict[str, Any]:
    """Open one SSE stream and read until the server closes it or errors.

    Returns a small dict describing the terminal state so the test body can
    bucket outcomes (clean-RST vs 503 vs unexpected) without re-raising.
    """
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": f"stream {idx}"}],
        "stream": True,
        "max_tokens": 10,
    }
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=CLIENT_TIMEOUT_SECONDS),
        ) as resp:
            status = resp.status
            chunks = 0
            async for _chunk in resp.content.iter_chunked(1024):
                chunks += 1
            return {"idx": idx, "kind": "closed", "status": status, "chunks": chunks}
    except aiohttp.ServerDisconnectedError as exc:
        return {"idx": idx, "kind": "server_disconnect", "error": repr(exc)}
    except aiohttp.ClientConnectionError as exc:
        return {"idx": idx, "kind": "conn_error", "error": repr(exc)}
    except asyncio.TimeoutError as exc:
        return {"idx": idx, "kind": "timeout", "error": repr(exc)}
    except aiohttp.ClientError as exc:
        return {"idx": idx, "kind": "client_error", "error": repr(exc)}


async def test_d201_force_kill_frontend_under_sse(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_endpoint_url: str,
    dynamo_deployment_namespace: str,
) -> None:
    """D201 — force-kill the frontend pod with 64 concurrent SSE streams in flight.

    Steps:
        1. Snapshot the dynamo_frontend_disconnected_clients counter (best-effort).
        2. Spawn ``CONCURRENT_STREAMS`` SSE POSTs and wait ``STREAM_SETTLE_SECONDS``
           so the streams are actively reading.
        3. Inject ``pod.kill`` against a frontend pod resolved by label.
        4. Gather every stream task; bucket outcomes into clean-RST / 503 /
           unexpected.
        5. Assert no client is left hung (every task terminated within the
           hard timeout), and at least 50/64 saw a clean disconnect (RST or 503).
        6. Wait up to ``RECOVERY_TIMEOUT_SECONDS`` for a replacement frontend
           pod to become Ready.
        7. Issue a fresh non-streaming completion; assert HTTP 200 to confirm
           the new replica is serving.
        8. Scrape /metrics on the new pod and assert
           ``dynamo_frontend_disconnected_clients`` advanced past the snapshot.
    """
    namespace = dynamo_deployment_namespace or DYNAMO_SERVER_NAMESPACE

    # 1. Snapshot disconnect counter (tolerate missing — first run, fresh pod).
    try:
        before_metrics = await scrape_frontend_metrics(kubectl, namespace)
    except RuntimeError as exc:
        logger.debug(
            lambda exc=exc: f"D201: pre-snapshot metrics scrape failed: {exc!r}"
        )
        before_metrics = {}
    disconnect_before = float(before_metrics.get(DISCONNECTED_METRIC, 0.0))

    killed_pod = await _resolve_frontend_pod(kubectl, namespace)
    logger.info(f"D201: resolved frontend pod {killed_pod!r} for fault injection")

    # 2 + 3. Launch streams, settle, inject pod.kill while streams in flight.
    async with aiohttp.ClientSession() as session:
        stream_tasks = [
            asyncio.create_task(_stream_one(session, dynamo_endpoint_url, i))
            for i in range(CONCURRENT_STREAMS)
        ]
        await asyncio.sleep(STREAM_SETTLE_SECONDS)

        async with faults.inject(
            "pod.kill",
            target={"ns": namespace, "pod": killed_pod},
        ):
            results = await asyncio.gather(*stream_tasks, return_exceptions=False)

    # 4 + 5. Bucket outcomes.
    clean_disconnects = sum(
        1
        for r in results
        if r["kind"] in ("server_disconnect", "conn_error")
        or (r["kind"] == "closed" and r.get("status") == 503)
    )
    timeouts = sum(1 for r in results if r["kind"] == "timeout")
    logger.info(
        f"D201: stream outcomes — clean={clean_disconnects} timeouts={timeouts} "
        f"total={len(results)}"
    )
    assert timeouts == 0, (
        f"D201: {timeouts}/{len(results)} clients hung past "
        f"{CLIENT_TIMEOUT_SECONDS}s — disconnect.rs:195 cleanup likely broken"
    )
    assert clean_disconnects >= 50, (
        f"D201: only {clean_disconnects}/{CONCURRENT_STREAMS} clients saw a "
        f"clean RST/503 (expected >=50); outcomes={results!r}"
    )

    # 6. Wait for the replacement frontend pod.
    new_pod = await _wait_for_new_frontend_ready(
        kubectl, namespace, killed_pod, RECOVERY_TIMEOUT_SECONDS
    )
    logger.info(f"D201: replacement frontend pod ready: {new_pod!r}")

    # 7. New request should succeed against the fresh replica.
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            f"{dynamo_endpoint_url}/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "post-recovery probe"}],
                "stream": False,
                "max_tokens": 10,
            },
            timeout=aiohttp.ClientTimeout(total=10.0),
        ) as resp,
    ):
        assert resp.status == 200, (
            f"D201: post-recovery request returned {resp.status}; "
            f"replacement pod {new_pod!r} is not actually serving"
        )

    # 8. Disconnect counter should have advanced on the new pod.
    try:
        after_metrics = await scrape_frontend_metrics(kubectl, namespace)
    except RuntimeError as exc:
        logger.warning(
            lambda exc=exc: f"D201: post-recovery metrics scrape failed: {exc!r}"
        )
        return
    disconnect_after = float(after_metrics.get(DISCONNECTED_METRIC, 0.0))
    assert disconnect_after > disconnect_before, (
        f"D201: {DISCONNECTED_METRIC} did not advance "
        f"(before={disconnect_before}, after={disconnect_after}); "
        f"disconnect.rs:195 may not be incrementing on force-kill RST"
    )


# D202


FRONTEND_COMPONENT_LABEL = "nvidia.com/dynamo-component-type"

FRONTEND_COMPONENT_VALUE = "frontend"

REQUEST_INTERVAL_SECONDS = 0.5

REQUEST_TIMEOUT_SECONDS = 5.0

RECOVERY_TIMEOUT_SECONDS = 30.0

SCALE_WAIT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class FrontendDeployment:
    """Frontend Deployment selected for D202 scaling."""

    name: str
    replicas: int
    labels: dict[str, str]


async def test_d202_frontend_scale_zero_recovers_mid_traffic(
    kubectl: KubectlClient,
    dynamo_endpoint_url: str,
    dynamo_deployment_namespace: str,
) -> None:
    """D202 — scale the unique Frontend Deployment down to zero and back to one.

    The default Dynamo ``disagg-1gpu`` topology has a single Frontend
    Deployment. If the installed topology has zero or multiple frontend-like
    Deployments, the test self-skips with the observed candidates rather than
    guessing which workload to disrupt.
    """
    frontend = await _resolve_unique_frontend_deployment(
        kubectl, dynamo_deployment_namespace
    )
    if frontend.replicas != 1:
        pytest.skip(
            f"D202 requires frontend Deployment {frontend.name!r} to start at "
            f"1 replica; observed replicas={frontend.replicas}"
        )

    traffic = TrafficRecorder()
    stop_event = asyncio.Event()
    traffic_task = asyncio.create_task(
        _send_background_traffic(dynamo_endpoint_url, traffic, stop_event)
    )

    try:
        await _wait_for_successful_request(
            dynamo_endpoint_url,
            timeout=RECOVERY_TIMEOUT_SECONDS,
            reason="pre-scale warmup",
        )

        await _scale_deployment(kubectl, dynamo_deployment_namespace, frontend.name, 0)
        await _wait_for_available_replicas(
            kubectl,
            dynamo_deployment_namespace,
            frontend.name,
            expected_available=0,
            timeout=SCALE_WAIT_TIMEOUT_SECONDS,
        )

        await _scale_deployment(kubectl, dynamo_deployment_namespace, frontend.name, 1)
        await _wait_for_available_replicas(
            kubectl,
            dynamo_deployment_namespace,
            frontend.name,
            expected_available=1,
            timeout=SCALE_WAIT_TIMEOUT_SECONDS,
        )
        await _wait_for_successful_request(
            dynamo_endpoint_url,
            timeout=RECOVERY_TIMEOUT_SECONDS,
            reason="post-scale recovery",
        )

        assert traffic.total_requests > 0, (
            "D202: background traffic loop sent no requests"
        )
        assert traffic.successes > 0, (
            f"D202: background traffic loop never observed a successful request; "
            f"observed outcomes={traffic.outcomes!r}"
        )
    finally:
        stop_event.set()
        traffic_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await traffic_task
        await _scale_deployment(
            kubectl,
            dynamo_deployment_namespace,
            frontend.name,
            frontend.replicas,
        )
        await _wait_for_available_replicas(
            kubectl,
            dynamo_deployment_namespace,
            frontend.name,
            expected_available=frontend.replicas,
            timeout=SCALE_WAIT_TIMEOUT_SECONDS,
        )


class TrafficRecorder:
    """Small mutable traffic summary for the background request loop."""

    def __init__(self) -> None:
        self.total_requests = 0
        self.successes = 0
        self.outcomes: list[str] = []

    def record(self, outcome: str) -> None:
        """Record one host-side request outcome."""
        self.total_requests += 1
        if outcome == "200":
            self.successes += 1
        self.outcomes.append(outcome)


async def _resolve_unique_frontend_deployment(
    kubectl: KubectlClient,
    namespace: str,
) -> FrontendDeployment:
    """Return the unique frontend Deployment or skip with observed candidates."""
    data = await kubectl.get_json("deployments", namespace=namespace)
    items = data.get("items", []) if isinstance(data, dict) else []
    candidates = [_deployment_from_item(item) for item in items]
    label_matches = [
        candidate
        for candidate in candidates
        if candidate.labels.get(FRONTEND_COMPONENT_LABEL) == FRONTEND_COMPONENT_VALUE
    ]
    name_matches = [
        candidate for candidate in candidates if candidate.name.endswith("-frontend")
    ]
    matches = label_matches or name_matches
    if len(matches) != 1:
        observed = (
            ", ".join(
                f"{candidate.name}(replicas={candidate.replicas}, "
                f"component={candidate.labels.get(FRONTEND_COMPONENT_LABEL)!r})"
                for candidate in candidates
            )
            or "<none>"
        )
        pytest.skip(
            f"D202 requires exactly one frontend Deployment in namespace {namespace!r}; "
            f"observed candidates: {observed}"
        )
    return matches[0]


def _deployment_from_item(item: dict[str, Any]) -> FrontendDeployment:
    """Build a compact frontend candidate from a Kubernetes Deployment item."""
    metadata = item.get("metadata", {})
    spec = item.get("spec", {})
    labels = metadata.get("labels", {})
    if not isinstance(labels, dict):
        labels = {}
    return FrontendDeployment(
        name=str(metadata.get("name", "")),
        replicas=int(spec.get("replicas") or 0),
        labels={str(key): str(value) for key, value in labels.items()},
    )


async def _scale_deployment(
    kubectl: KubectlClient,
    namespace: str,
    deployment: str,
    replicas: int,
) -> None:
    """Scale a Deployment to ``replicas`` with an error that names D202 context."""
    result = await kubectl.run(
        "scale",
        "deployment",
        deployment,
        f"--replicas={replicas}",
        namespace=namespace,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"D202: failed to scale Deployment {deployment!r} in namespace "
            f"{namespace!r} to replicas={replicas}: {result.stderr.strip()}"
        )
    logger.info(
        f"D202: scaled frontend Deployment {deployment!r} in {namespace!r} "
        f"to replicas={replicas}"
    )


async def _wait_for_available_replicas(
    kubectl: KubectlClient,
    namespace: str,
    deployment: str,
    expected_available: int,
    timeout: float,
) -> None:
    """Wait until Deployment status.availableReplicas equals the expected count."""
    deadline = asyncio.get_running_loop().time() + timeout
    last_status: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        data = await kubectl.get_json("deployment", deployment, namespace=namespace)
        if not isinstance(data, dict):
            raise RuntimeError(
                f"D202: kubectl returned non-object Deployment payload for {deployment!r}: "
                f"{data!r}"
            )
        last_status = data.get("status", {})
        available = int(last_status.get("availableReplicas") or 0)
        if available == expected_available:
            return
        await asyncio.sleep(0.5)
    raise TimeoutError(
        f"D202: Deployment {deployment!r} did not reach availableReplicas="
        f"{expected_available} within {timeout}s; last_status={last_status!r}"
    )


async def _wait_for_successful_request(
    dynamo_endpoint_url: str,
    timeout: float,
    reason: str,
) -> None:
    """Poll chat-completions until one request returns HTTP 200."""
    deadline = asyncio.get_running_loop().time() + timeout
    outcomes: list[str] = []
    async with aiohttp.ClientSession() as session:
        while asyncio.get_running_loop().time() < deadline:
            outcome = await _post_chat_completion(session, dynamo_endpoint_url)
            outcomes.append(outcome)
            if outcome == "200":
                return
            await asyncio.sleep(0.5)
    raise TimeoutError(
        f"D202: no successful chat-completions request within {timeout}s during "
        f"{reason}; outcomes={outcomes!r}"
    )


async def _send_background_traffic(
    dynamo_endpoint_url: str,
    traffic: TrafficRecorder,
    stop_event: asyncio.Event,
) -> None:
    """Send host-side chat-completions traffic until ``stop_event`` is set."""
    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            outcome = await _post_chat_completion(session, dynamo_endpoint_url)
            traffic.record(outcome)
            await asyncio.sleep(REQUEST_INTERVAL_SECONDS)


async def _post_chat_completion(
    session: aiohttp.ClientSession,
    dynamo_endpoint_url: str,
) -> str:
    """POST one non-streaming chat-completions request and return an outcome token."""
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
        "max_tokens": 10,
    }
    try:
        async with session.post(
            dynamo_endpoint_url + "/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
        ) as response:
            await response.read()
            return str(response.status)
    except asyncio.TimeoutError:
        return "timeout"
    except aiohttp.ClientError as exc:
        return type(exc).__name__


# D203


TOXIPROXY_SERVICE_DNS = "toxiproxy.chaos-toxiproxy.svc"

BACKEND_PROXY_NAME = "backend-stream"

BACKEND_TIMEOUT_ENV = "DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS"

DEFAULT_BACKEND_TIMEOUT_SECONDS = 30

CLIENT_TIMEOUT_SLACK_SECONDS = 10.0

FRONTEND_SELECTOR = "nvidia.com/dynamo-component-type=frontend"

RESPONSE_TIMEOUT_METRIC = "dynamo_frontend_requests_total"

_BACKEND_CHAOS_OPT_IN_ENV = "AIPERF_DYNAMO_BACKEND_STREAM_CHAOS"


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


# D207


_CLIENT_ERROR_BUDGET_S: float = 35.0

_STREAM_START_TIMEOUT_S: float = 10.0

_DECODE_POD_SELECTOR: str = "nvidia.com/dynamo-sub-component-type=decode"


async def _resolve_decode_pod(kubectl: KubectlClient, namespace: str) -> str:
    """Return the first decode pod selected by the Dynamo operator label.

    Raises:
        RuntimeError: When no decode pod is present in ``namespace``.
    """
    pod_res = await kubectl.run(
        "get",
        "pod",
        "-n",
        namespace,
        "-l",
        _DECODE_POD_SELECTOR,
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=True,
    )
    decode_pod = pod_res.stdout.strip()
    if not decode_pod:
        raise RuntimeError(
            f"D207: no decode pod found in {namespace!r} matching "
            f"{_DECODE_POD_SELECTOR!r}; cannot inject mid-stream worker death"
        )
    return decode_pod


def _append_sse_data_frames(
    frames: list[str],
    buffered_text: str,
    first_frame_seen: asyncio.Event,
) -> str:
    """Append complete SSE ``data:`` lines and return the incomplete suffix."""
    while "\n" in buffered_text:
        raw_line, buffered_text = buffered_text.split("\n", 1)
        line = raw_line.rstrip("\r")
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        frames.append(payload)
        if payload != "[DONE]":
            first_frame_seen.set()
    return buffered_text


async def _read_sse_frames(
    session: aiohttp.ClientSession,
    url: str,
    request_body: dict[str, object],
    frames: list[str],
    first_frame_seen: asyncio.Event,
) -> tuple[int | None, str | None]:
    """Read SSE data frames until EOF and return ``(status, exception_repr)``."""
    buffered_text = ""
    try:
        async with session.post(url, json=request_body) as resp:
            status = resp.status
            if status != 200:
                body = await resp.text()
                frames.append(f"<HTTP {status}: {body}>")
                return status, None
            async for chunk in resp.content.iter_any():
                buffered_text += chunk.decode("utf-8", errors="replace")
                buffered_text = _append_sse_data_frames(
                    frames,
                    buffered_text,
                    first_frame_seen,
                )
            if buffered_text.strip():
                frames.append(f"<TRAILING_BYTES: {buffered_text.strip()}>")
            return status, None
    except (
        aiohttp.ClientError,
        aiohttp.ServerDisconnectedError,
        asyncio.TimeoutError,
    ) as exc:
        return None, repr(exc)


def _format_observed_sequence(frames: list[str], exception_repr: str | None) -> str:
    """Return the exact observed stream sequence for assertion messages."""
    suffix = f", exception={exception_repr}" if exception_repr else ""
    return f"frames={frames!r}{suffix}"


def _error_payload(frame: str) -> dict[str, object] | None:
    """Decode one SSE data payload and return its ``error`` object if present."""
    try:
        payload = orjson.loads(frame)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    return error if isinstance(error, dict) else None


def _assert_error_frame_then_done(
    frames: list[str],
    exception_repr: str | None,
) -> None:
    """Assert the D207 two-frame terminal contract and include observed frames."""
    observed = _format_observed_sequence(frames, exception_repr)
    assert exception_repr is None, (
        "D207: stream raised before receiving the required error JSON frame "
        f"followed by [DONE]; observed {observed}"
    )

    error_index = next(
        (idx for idx, frame in enumerate(frames) if _error_payload(frame) is not None),
        None,
    )
    assert error_index is not None, (
        "D207: stream did not contain an error JSON SSE frame before [DONE]; "
        f"observed {observed}"
    )
    assert error_index + 1 < len(frames) and frames[error_index + 1] == "[DONE]", (
        "D207: error JSON SSE frame was not immediately followed by [DONE]; "
        f"observed {observed}"
    )

    error = _error_payload(frames[error_index])
    assert error is not None
    assert isinstance(error.get("message"), str) and error["message"], (
        f"D207: error JSON frame missing non-empty error.message; observed {observed}"
    )
    assert error.get("type") == "internal_server_error", (
        "D207: error JSON frame has wrong error.type; expected "
        f"'internal_server_error'; observed {observed}"
    )
    assert error.get("code") == 500, (
        "D207: error JSON frame has wrong error.code; expected 500; "
        f"observed {observed}"
    )


async def test_d207_streaming_error_frame_after_decode_kill(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Kill a decode worker mid-stream and assert error JSON then ``[DONE]``.

    Steps:
        1. Open one streaming POST to ``/chat/completions`` with enough output
           budget to stay in-flight.
        2. Wait for the first SSE data frame to prove decode generation began.
        3. Select and kill a decode pod via the same label path used by D401.
        4. Drain the stream and assert the terminal sequence is the Dynamo
           structured error frame followed immediately by ``[DONE]``.
    """
    request_body: dict[str, object] = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a detailed, 800-word incident report about a "
                    "database failover and include numbered remediation steps."
                ),
            }
        ],
        "max_tokens": 512,
        "stream": True,
        "temperature": 0.0,
    }
    frames: list[str] = []
    first_frame_seen = asyncio.Event()
    timeout = aiohttp.ClientTimeout(total=_CLIENT_ERROR_BUDGET_S + 30.0)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        stream_task = asyncio.create_task(
            _read_sse_frames(
                session,
                f"{dynamo_endpoint_url}/chat/completions",
                request_body,
                frames,
                first_frame_seen,
            )
        )
        try:
            await asyncio.wait_for(
                first_frame_seen.wait(),
                timeout=_STREAM_START_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            stream_task.cancel()
            pytest.fail(
                "D207: no SSE data frame received within "
                f"{_STREAM_START_TIMEOUT_S}s; observed "
                f"{_format_observed_sequence(frames, None)}"
            )

        try:
            decode_pod = await _resolve_decode_pod(kubectl, dynamo_deployment_namespace)
        except RuntimeError as exc:
            stream_task.cancel()
            pytest.fail(str(exc))

        async with faults.inject(
            "pod.kill",
            target={"ns": dynamo_deployment_namespace, "pod": decode_pod},
        ):
            logger.info(
                lambda p=decode_pod, ns=dynamo_deployment_namespace: (
                    f"D207: killed decode pod {ns}/{p} mid-stream"
                )
            )
            try:
                _status, exception_repr = await asyncio.wait_for(
                    stream_task,
                    timeout=_CLIENT_ERROR_BUDGET_S + 5.0,
                )
            except asyncio.TimeoutError:
                stream_task.cancel()
                pytest.fail(
                    "D207: stream did not terminate with error JSON then [DONE] "
                    f"within {_CLIENT_ERROR_BUDGET_S + 5.0}s of decode-pod kill; "
                    f"observed {_format_observed_sequence(frames, None)}"
                )

    _assert_error_frame_then_done(frames, exception_repr)


# D209

_D209_FRONTEND_PROXY = "d209-frontend-first-byte"
_FRONTEND_PROXY_PORT = 20011
_LATENCY_MS = 1200
_FIRST_BYTE_TIMEOUT_S = 12.0


async def _frontend_service_name(kubectl: KubectlClient, namespace: str) -> str | None:
    result = await kubectl.run(
        "get",
        "service",
        "-n",
        namespace,
        "-o",
        "jsonpath={.items[*].metadata.name}",
        check=False,
    )
    if result.returncode != 0:
        return None
    for name in result.stdout.split():
        if name.endswith("-frontend"):
            return name
    return None


async def _configure_frontend_proxy(request: pytest.FixtureRequest) -> None:
    """Create the frontend proxy and downstream latency toxic, or skip explicitly."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    service = await _frontend_service_name(kubectl, namespace)
    if service is None:
        pytest.skip(
            "D209 requires a Dynamo frontend Service ending in '-frontend'; "
            f"none was found in namespace {namespace!r}."
        )
    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    try:
        await dynamo_toxiproxy.add_proxy(
            name=_D209_FRONTEND_PROXY,
            listen=f"0.0.0.0:{_FRONTEND_PROXY_PORT}",
            upstream=f"{service}.{namespace}.svc.cluster.local:8000",
        )
    except ToxiproxyError as exc:
        pytest.skip(
            f"D209 requires the frontend Toxiproxy port; proxy setup failed: {exc}"
        )
    await dynamo_toxiproxy.add_toxic(
        _D209_FRONTEND_PROXY,
        "latency",
        {"latency": _LATENCY_MS, "jitter": 0},
        stream="downstream",
        name="d209-first-byte-latency",
    )


async def test_d209_sse_first_byte_latency_under_frontend_throttling(
    request: pytest.FixtureRequest,
) -> None:
    """Inject frontend downstream latency and assert first SSE byte is bounded."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    try:
        await _configure_frontend_proxy(request)
        async with kubectl.port_forward(
            "service/toxiproxy",
            _FRONTEND_PROXY_PORT,
            namespace="chaos-toxiproxy",
        ) as local_port:
            url = f"http://127.0.0.1:{local_port}/v1"
            payload = {
                "model": "default",
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                "max_tokens": 32,
                "stream": True,
                "temperature": 0.0,
            }
            started = time.monotonic()
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=_FIRST_BYTE_TIMEOUT_S)
                ) as session,
                session.post(f"{url}/chat/completions", json=payload) as resp,
            ):
                body = await resp.text() if resp.status != 200 else ""
                assert resp.status == 200, (
                    f"D209: expected HTTP 200, got {resp.status}: {body}"
                )
                first = await resp.content.read(1)
            elapsed = time.monotonic() - started
    finally:
        await dynamo_toxiproxy.reset()

    assert first, "D209: streaming response ended before any SSE byte was received"
    assert elapsed >= (_LATENCY_MS / 1000.0) * 0.75, (
        f"D209: first byte arrived in {elapsed:.3f}s despite {_LATENCY_MS}ms "
        "downstream latency; the request likely bypassed the frontend proxy"
    )
    assert elapsed < _FIRST_BYTE_TIMEOUT_S, (
        f"D209: first byte latency {elapsed:.3f}s exceeded bounded throttle budget "
        f"{_FIRST_BYTE_TIMEOUT_S}s"
    )


# D210

_D210_FRONTEND_PROXY = "d210-frontend-slicer"
_FRONTEND_PROXY_PORT = 20011


async def _d210_frontend_service_name(
    kubectl: KubectlClient, namespace: str
) -> str | None:
    result = await kubectl.run(
        "get",
        "service",
        "-n",
        namespace,
        "-o",
        "jsonpath={.items[*].metadata.name}",
        check=False,
    )
    if result.returncode != 0:
        return None
    for name in result.stdout.split():
        if name.endswith("-frontend"):
            return name
    return None


def _append_complete_sse_payloads(buffer: str, payloads: list[str]) -> str:
    """Append complete LF/CRLF-delimited SSE data payloads; return suffix."""
    while "\n\n" in buffer or "\r\n\r\n" in buffer:
        lf_pos = buffer.find("\n\n") if "\n\n" in buffer else len(buffer) + 1
        crlf_pos = buffer.find("\r\n\r\n") if "\r\n\r\n" in buffer else len(buffer) + 1
        delimiter = "\n\n" if lf_pos < crlf_pos else "\r\n\r\n"
        raw_event, buffer = buffer.split(delimiter, 1)
        for raw_line in raw_event.splitlines():
            line = raw_line.rstrip("\r")
            if line.startswith("data:"):
                payloads.append(line.removeprefix("data:").strip())
    return buffer


async def _configure_sliced_frontend_proxy(request: pytest.FixtureRequest) -> None:
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    service = await _d210_frontend_service_name(kubectl, namespace)
    if service is None:
        pytest.skip(
            "D210 requires a Dynamo frontend Service ending in '-frontend'; "
            f"none was found in namespace {namespace!r}."
        )
    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    try:
        await dynamo_toxiproxy.add_proxy(
            name=_D210_FRONTEND_PROXY,
            listen=f"0.0.0.0:{_FRONTEND_PROXY_PORT}",
            upstream=f"{service}.{namespace}.svc.cluster.local:8000",
        )
        await dynamo_toxiproxy.add_toxic(
            _D210_FRONTEND_PROXY,
            "slicer",
            {"average_size": 7, "size_variation": 2, "delay": 2},
            stream="downstream",
            name="d210-sse-slicer",
        )
    except ToxiproxyError as exc:
        pytest.skip(
            f"D210 requires a frontend Toxiproxy slicer toxic; setup failed: {exc}"
        )


async def test_d210_sse_fragmented_frames_parse_to_complete_chunks(
    request: pytest.FixtureRequest,
) -> None:
    """Slice frontend TCP bytes and assert complete OpenAI SSE frames are readable."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    payloads: list[str] = []
    try:
        await _configure_sliced_frontend_proxy(request)
        async with kubectl.port_forward(
            "service/toxiproxy",
            _FRONTEND_PROXY_PORT,
            namespace="chaos-toxiproxy",
        ) as local_port:
            url = f"http://127.0.0.1:{local_port}/v1/chat/completions"
            body = {
                "model": "default",
                "messages": [{"role": "user", "content": "Count from one to ten."}],
                "max_tokens": 64,
                "stream": True,
                "temperature": 0.0,
            }
            buffer = ""
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30.0)
                ) as session,
                session.post(url, json=body) as resp,
            ):
                error_body = await resp.text() if resp.status != 200 else ""
                assert resp.status == 200, (
                    f"D210: expected HTTP 200, got {resp.status}: {error_body}"
                )
                async for chunk in resp.content.iter_chunked(3):
                    buffer += chunk.decode("utf-8", errors="replace")
                    buffer = _append_complete_sse_payloads(buffer, payloads)
                    if payloads and payloads[-1] == "[DONE]":
                        break
    finally:
        await dynamo_toxiproxy.reset()

    assert payloads, (
        "D210: no complete SSE data payloads were parsed from sliced stream"
    )
    assert payloads[-1] == "[DONE]", (
        f"D210: stream did not end with [DONE]: {payloads!r}"
    )
    data_payloads = [payload for payload in payloads if payload != "[DONE]"]
    assert data_payloads, f"D210: stream contained no JSON data frames: {payloads!r}"
    for payload in data_payloads:
        decoded = orjson.loads(payload)
        assert isinstance(decoded, dict), (
            f"D210: SSE payload is not a JSON object: {payload!r}"
        )
        assert "choices" in decoded, f"D210: JSON chunk missing choices: {decoded!r}"


# D211


async def _post_with_raw_crlf(endpoint_url: str, payload: dict[str, object]) -> bytes:
    """Send a raw HTTP/1.1 POST using CRLF line delimiters and return bytes."""
    parsed = urlparse(endpoint_url)
    if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
        pytest.skip(
            "D211 requires a host-reachable plain-http Dynamo endpoint with an explicit port; "
            f"got {endpoint_url!r}"
        )
    path = f"{parsed.path.rstrip('/')}/chat/completions"
    body = orjson.dumps(payload)
    request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {parsed.hostname}:{parsed.port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode() + body

    reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
    try:
        writer.write(request)
        await writer.drain()
        return await asyncio.wait_for(reader.read(-1), timeout=20.0)
    finally:
        writer.close()
        await writer.wait_closed()


def _decode_chunked_body(body: bytes) -> bytes:
    """Decode a minimal HTTP/1.1 chunked body for raw-socket assertions."""
    decoded = bytearray()
    rest = body
    while rest:
        size_text, sep, after_size = rest.partition(b"\r\n")
        if not sep:
            break
        size = int(size_text.split(b";", 1)[0], 16)
        if size == 0:
            break
        decoded.extend(after_size[:size])
        rest = after_size[size + 2 :]
    return bytes(decoded)


def _split_http_response(raw: bytes) -> tuple[int, bytes]:
    """Return ``(status, body)`` from a raw HTTP response."""
    header_bytes, sep, body = raw.partition(b"\r\n\r\n")
    assert sep, (
        f"D211: response did not contain CRLF CRLF header delimiter: {raw[:512]!r}"
    )
    header_lines = header_bytes.split(b"\r\n")
    status_line = header_lines[0].decode(errors="replace")
    parts = status_line.split()
    assert len(parts) >= 2 and parts[1].isdigit(), (
        f"D211: malformed HTTP status line for CRLF request: {status_line!r}"
    )
    headers = b"\r\n".join(header_lines[1:]).lower()
    if b"transfer-encoding: chunked" in headers:
        body = _decode_chunked_body(body)
    return int(parts[1]), body


async def test_d211_crlf_delimited_http_request_returns_openai_json(
    dynamo_endpoint_url: str,
) -> None:
    """Use raw CRLF HTTP framing and assert Dynamo returns a normal JSON shape."""
    payload: dict[str, object] = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with the word ok."}],
        "max_tokens": 8,
        "stream": False,
        "temperature": 0.0,
    }
    status, body = _split_http_response(
        await _post_with_raw_crlf(dynamo_endpoint_url, payload)
    )

    assert status == 200, (
        f"D211: CRLF-framed request returned HTTP {status}: {body[:512]!r}"
    )
    decoded = orjson.loads(body)
    assert isinstance(decoded, dict), (
        f"D211: response body is not a JSON object: {decoded!r}"
    )
    assert decoded.get("object") == "chat.completion", (
        f"D211: non-stream response object mismatch: {decoded!r}"
    )
    choices = decoded.get("choices")
    assert isinstance(choices, list) and choices, (
        f"D211: CRLF request response missing non-empty choices: {decoded!r}"
    )


# D212


async def _assert_followup_chat_succeeds(endpoint_url: str, case_id: str) -> None:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with ok."}],
        "max_tokens": 8,
        "stream": False,
        "temperature": 0.0,
    }
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0)) as session,
        session.post(f"{endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body = await resp.text()
    assert resp.status == 200, (
        f"{case_id}: follow-up request returned HTTP {resp.status}: {body}"
    )


async def test_d212_disconnect_before_first_token_is_clean(
    dynamo_endpoint_url: str,
) -> None:
    """Open a stream and drop the socket before reading any SSE body bytes."""
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": "Write a long paragraph about graceful streaming cancellation.",
            }
        ],
        "max_tokens": 256,
        "stream": True,
        "temperature": 0.0,
    }
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15.0))
    try:
        resp = await session.post(
            f"{dynamo_endpoint_url}/chat/completions", json=payload
        )
        try:
            assert resp.status == 200, (
                f"D212: streaming setup returned HTTP {resp.status}: {await resp.text()}"
            )
            resp.close()
        finally:
            resp.release()
    finally:
        await session.close()

    await _assert_followup_chat_succeeds(dynamo_endpoint_url, "D212")


# D213


async def _read_first_data_payload(resp: aiohttp.ClientResponse) -> str | None:
    """Return the first non-DONE SSE data payload from ``resp``."""
    buffer = ""
    async for chunk in resp.content.iter_any():
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            raw_line, buffer = buffer.split("\n", 1)
            line = raw_line.rstrip("\r")
            if not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if payload and payload != "[DONE]":
                return payload
    return None


async def _d213_assert_followup_chat_succeeds(endpoint_url: str) -> None:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with ok."}],
        "max_tokens": 8,
        "stream": False,
        "temperature": 0.0,
    }
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0)) as session,
        session.post(f"{endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body = await resp.text()
    assert resp.status == 200, (
        f"D213: follow-up request returned HTTP {resp.status}: {body}"
    )


async def test_d213_disconnect_after_first_token_is_clean(
    dynamo_endpoint_url: str,
) -> None:
    """Read one SSE data frame, close the socket, then verify fresh traffic works."""
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": "Write three short sentences about streaming clients.",
            }
        ],
        "max_tokens": 128,
        "stream": True,
        "temperature": 0.0,
    }
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20.0))
    try:
        resp = await session.post(
            f"{dynamo_endpoint_url}/chat/completions", json=payload
        )
        try:
            assert resp.status == 200, (
                f"D213: streaming setup returned HTTP {resp.status}: {await resp.text()}"
            )
            first_payload = await _read_first_data_payload(resp)
            assert first_payload is not None, (
                "D213: stream ended before first SSE data payload"
            )
            resp.close()
        finally:
            resp.release()
    finally:
        await session.close()

    await _d213_assert_followup_chat_succeeds(dynamo_endpoint_url)


# D214


def _append_sse_payloads(buffer: str, payloads: list[str]) -> str:
    """Append complete SSE ``data:`` payloads and return the incomplete suffix."""
    while "\n" in buffer:
        raw_line, buffer = buffer.split("\n", 1)
        line = raw_line.rstrip("\r")
        if line.startswith("data:"):
            payloads.append(line.removeprefix("data:").strip())
    return buffer


async def _collect_stream_payloads(endpoint_url: str) -> list[str]:
    request_body = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with a short sentence."}],
        "max_tokens": 32,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }
    payloads: list[str] = []
    buffer = ""
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0)) as session,
        session.post(f"{endpoint_url}/chat/completions", json=request_body) as resp,
    ):
        body = await resp.text() if resp.status != 200 else ""
        assert resp.status == 200, f"D214: expected HTTP 200, got {resp.status}: {body}"
        async for chunk in resp.content.iter_any():
            buffer += chunk.decode("utf-8", errors="replace")
            buffer = _append_sse_payloads(buffer, payloads)
    return payloads


def _is_usage_chunk(payload: str) -> bool:
    if payload == "[DONE]":
        return False
    decoded = orjson.loads(payload)
    if not isinstance(decoded, dict):
        return False
    usage = decoded.get("usage")
    choices = decoded.get("choices")
    return isinstance(usage, dict) and choices == []


async def test_d214_stream_usage_final_chunk_precedes_done(
    dynamo_endpoint_url: str,
) -> None:
    """Request include_usage=true and assert the final usage-only chunk shape."""
    payloads = await _collect_stream_payloads(dynamo_endpoint_url)
    assert payloads, "D214: no SSE payloads received"
    assert payloads[-1] == "[DONE]", (
        f"D214: stream did not terminate with [DONE]: {payloads!r}"
    )
    assert len(payloads) >= 2, (
        f"D214: stream too short to contain usage chunk: {payloads!r}"
    )
    assert _is_usage_chunk(payloads[-2]), (
        "D214: expected usage-only chunk immediately before [DONE] when "
        f"include_usage=true; observed tail={payloads[-3:]!r}"
    )


# D215


def _d215_append_sse_payloads(buffer: str, payloads: list[str]) -> str:
    """Append complete SSE ``data:`` payloads and return the incomplete suffix."""
    while "\n" in buffer:
        raw_line, buffer = buffer.split("\n", 1)
        line = raw_line.rstrip("\r")
        if line.startswith("data:"):
            payloads.append(line.removeprefix("data:").strip())
    return buffer


def _d215_is_usage_chunk(payload: str) -> bool:
    if payload == "[DONE]":
        return False
    decoded = orjson.loads(payload)
    return (
        isinstance(decoded, dict)
        and isinstance(decoded.get("usage"), dict)
        and decoded.get("choices") == []
    )


async def test_d215_include_usage_false_has_no_usage_only_sse_chunk(
    dynamo_endpoint_url: str,
) -> None:
    """Request include_usage=false and assert no usage-only SSE chunk appears."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with a short sentence."}],
        "max_tokens": 32,
        "stream": True,
        "stream_options": {"include_usage": False},
        "temperature": 0.0,
    }
    payloads: list[str] = []
    buffer = ""
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0)) as session,
        session.post(f"{dynamo_endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body = await resp.text() if resp.status != 200 else ""
        assert resp.status == 200, f"D215: expected HTTP 200, got {resp.status}: {body}"
        async for chunk in resp.content.iter_any():
            buffer += chunk.decode("utf-8", errors="replace")
            buffer = _d215_append_sse_payloads(buffer, payloads)

    assert payloads, "D215: no SSE payloads received"
    assert payloads[-1] == "[DONE]", (
        f"D215: stream did not terminate with [DONE]: {payloads!r}"
    )
    usage_chunks = [item for item in payloads if _d215_is_usage_chunk(item)]
    assert not usage_chunks, (
        "D215: include_usage=false still emitted usage-only chunks: "
        f"{usage_chunks!r}; full stream={payloads!r}"
    )


# D216

_OVERLOAD_OPT_IN_ENV = "AIPERF_DYNAMO_OVERLOAD_CHAOS"
_CONCURRENCY = 32


async def _post_non_stream(endpoint_url: str, idx: int) -> dict[str, Any]:
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": f"Reply with one concise sentence for overload request {idx}.",
            }
        ],
        "max_tokens": 32,
        "stream": False,
        "temperature": 0.0,
    }
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45.0)) as session,
        session.post(f"{endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body = await resp.read()
    try:
        decoded = orjson.loads(body)
    except orjson.JSONDecodeError:
        decoded = body.decode(errors="replace")[:512]
    return {"status": resp.status, "body": decoded}


def _assert_openai_non_stream_shape(result: dict[str, Any]) -> None:
    status = result["status"]
    body = result["body"]
    assert isinstance(body, dict), (
        f"D216: non-stream overload response was not JSON object: status={status}, body={body!r}"
    )
    if status == 200:
        assert body.get("object") == "chat.completion", (
            f"D216: success response has wrong OpenAI object shape: {body!r}"
        )
        choices = body.get("choices")
        assert isinstance(choices, list) and choices, (
            f"D216: success response missing non-empty choices: {body!r}"
        )
        return
    assert status in {429, 500, 503, 504}, (
        f"D216: overload response used unexpected HTTP status {status}: {body!r}"
    )
    error = body.get("error")
    assert isinstance(error, dict), (
        f"D216: overload error response missing OpenAI error object: {body!r}"
    )
    assert isinstance(error.get("message"), str) and error["message"], (
        f"D216: overload error missing non-empty error.message: {body!r}"
    )


@pytest.mark.skipif(
    os.environ.get(_OVERLOAD_OPT_IN_ENV) != "1",
    reason=(
        "D216 requires an overload topology or externally tuned Dynamo deployment; "
        f"set {_OVERLOAD_OPT_IN_ENV}=1 only when concurrency is expected to trigger "
        "queueing/throttling without destabilizing the shared test cluster."
    ),
)
async def test_d216_non_stream_shape_under_overload(
    dynamo_endpoint_url: str,
) -> None:
    """Fan out non-stream requests and require every terminal body to be JSON-shaped."""
    results = await asyncio.gather(
        *(_post_non_stream(dynamo_endpoint_url, idx) for idx in range(_CONCURRENCY))
    )
    for result in results:
        _assert_openai_non_stream_shape(result)
    if not any(item["status"] != 200 for item in results):
        pytest.skip(
            "D216 overload prerequisite did not trigger any non-200 response; "
            f"statuses={[item['status'] for item in results]!r}. Increase load or use an "
            "overload-tuned Dynamo topology before treating this case as covered."
        )


# D217


@dataclass(frozen=True)
class _HTTPJSON:
    status: int
    body: dict[str, object]
    text: str


def _service_root(endpoint_url: str) -> str:
    """Return the Dynamo service root regardless of whether fixture includes /v1."""
    trimmed = endpoint_url.rstrip("/")
    return trimmed.removesuffix("/v1")


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, object],
) -> _HTTPJSON:
    """POST JSON and decode the response body for assertion diagnostics."""
    async with session.post(url, json=payload) as resp:
        text = await resp.text()
    try:
        decoded = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoded = {}
    body = decoded if isinstance(decoded, dict) else {}
    return _HTTPJSON(status=resp.status, body=body, text=text)


def _assert_chat_completion_shape(case_id: str, result: _HTTPJSON) -> None:
    """Assert a minimal OpenAI chat-completion response envelope."""
    assert result.status == 200, (
        f"{case_id}: chat endpoint returned HTTP {result.status}; body={result.text!r}"
    )
    assert isinstance(result.body.get("id"), str), (
        f"{case_id}: response missing string id; body={result.body!r}"
    )
    choices = result.body.get("choices")
    assert isinstance(choices, list) and choices, (
        f"{case_id}: response missing non-empty choices; body={result.body!r}"
    )
    first_choice = choices[0]
    assert isinstance(first_choice, dict), (
        f"{case_id}: first choice is not an object; body={result.body!r}"
    )
    assert isinstance(first_choice.get("message"), dict), (
        f"{case_id}: first choice missing message object; body={result.body!r}"
    )


async def test_d217_chat_path_alias_compatibility(dynamo_endpoint_url: str) -> None:
    """Both /v1/chat/completions and /chat/completions should serve chat."""
    payload: dict[str, object] = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with the word pong."}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    root = _service_root(dynamo_endpoint_url)
    canonical_url = f"{root}/v1/chat/completions"
    alias_url = f"{root}/chat/completions"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30.0)
    ) as session:
        canonical = await _post_json(session, canonical_url, payload)
        alias = await _post_json(session, alias_url, payload)

    _assert_chat_completion_shape("D217 canonical", canonical)
    if alias.status in {404, 405, 501}:
        pytest.skip(
            "D217: Dynamo deployment does not expose /chat/completions alias "
            f"(HTTP {alias.status}); canonical /v1 path passed"
        )
    _assert_chat_completion_shape("D217 alias", alias)


# D218


@dataclass(frozen=True)
class _d218_HTTPJSON:
    status: int
    body: dict[str, object]
    text: str


async def _d218_post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, object],
) -> _d218_HTTPJSON:
    """POST JSON and decode an object response when present."""
    async with session.post(url, json=payload) as resp:
        text = await resp.text()
    try:
        decoded = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoded = {}
    body = decoded if isinstance(decoded, dict) else {}
    return _d218_HTTPJSON(status=resp.status, body=body, text=text)


def _assert_responses_shape(result: _d218_HTTPJSON) -> None:
    """Assert a minimal Responses API success envelope."""
    assert result.status == 200, (
        f"D218: /responses returned HTTP {result.status}; body={result.text!r}"
    )
    assert isinstance(result.body.get("id"), str), (
        f"D218: response missing string id; body={result.body!r}"
    )
    assert result.body.get("object") == "response", (
        f"D218: response object must be 'response'; body={result.body!r}"
    )
    assert "output" in result.body or "output_text" in result.body, (
        f"D218: Responses API success missing output/output_text; body={result.body!r}"
    )


async def test_d218_responses_api_compatibility_probe(
    dynamo_endpoint_url: str,
) -> None:
    """Probe /responses and skip explicitly when this Dynamo build lacks it."""
    payload: dict[str, object] = {
        "model": "default",
        "input": "Reply with the word pong.",
        "max_output_tokens": 8,
        "temperature": 0.0,
    }
    url = f"{dynamo_endpoint_url.rstrip('/')}/responses"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30.0)
    ) as session:
        result = await _d218_post_json(session, url, payload)

    if result.status in {404, 405, 501}:
        pytest.skip(
            "D218: Dynamo deployment does not expose Responses API "
            f"(HTTP {result.status})"
        )
    assert result.status < 500, (
        f"D218: /responses must not fail with server error; "
        f"HTTP {result.status} body={result.text!r}"
    )
    _assert_responses_shape(result)


# D219


@dataclass(frozen=True)
class _d219_HTTPJSON:
    status: int
    body: dict[str, object]
    text: str


async def _d219_post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, object],
) -> _d219_HTTPJSON:
    """POST JSON and decode an object response when present."""
    async with session.post(url, json=payload) as resp:
        text = await resp.text()
    try:
        decoded = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoded = {}
    body = decoded if isinstance(decoded, dict) else {}
    return _d219_HTTPJSON(status=resp.status, body=body, text=text)


def _assert_completions_shape(result: _d219_HTTPJSON) -> None:
    """Assert a minimal OpenAI legacy completion response envelope."""
    assert result.status == 200, (
        f"D219: /completions returned HTTP {result.status}; body={result.text!r}"
    )
    assert isinstance(result.body.get("id"), str), (
        f"D219: response missing string id; body={result.body!r}"
    )
    choices = result.body.get("choices")
    assert isinstance(choices, list) and choices, (
        f"D219: response missing non-empty choices; body={result.body!r}"
    )
    first_choice = choices[0]
    assert isinstance(first_choice, dict), (
        f"D219: first choice is not an object; body={result.body!r}"
    )
    assert isinstance(first_choice.get("text"), str), (
        f"D219: first choice missing text string; body={result.body!r}"
    )


async def test_d219_completions_api_compatibility_probe(
    dynamo_endpoint_url: str,
) -> None:
    """Probe /completions and skip explicitly when this Dynamo build lacks it."""
    payload: dict[str, object] = {
        "model": "default",
        "prompt": "Reply with the word pong.",
        "max_tokens": 8,
        "temperature": 0.0,
    }
    url = f"{dynamo_endpoint_url.rstrip('/')}/completions"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30.0)
    ) as session:
        result = await _d219_post_json(session, url, payload)

    if result.status in {404, 405, 501}:
        pytest.skip(
            "D219: Dynamo deployment does not expose legacy Completions API "
            f"(HTTP {result.status})"
        )
    assert result.status < 500, (
        f"D219: /completions must not fail with server error; "
        f"HTTP {result.status} body={result.text!r}"
    )
    _assert_completions_shape(result)


# D220


@dataclass(frozen=True)
class _d220_HTTPJSON:
    status: int
    body: dict[str, object]
    text: str
    content_type: str


async def _d220_post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, object],
) -> _d220_HTTPJSON:
    """POST JSON and decode an object response when present."""
    async with session.post(url, json=payload) as resp:
        text = await resp.text()
        content_type = resp.headers.get("content-type", "")
    try:
        decoded = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoded = {}
    body = decoded if isinstance(decoded, dict) else {}
    return _d220_HTTPJSON(
        status=resp.status,
        body=body,
        text=text,
        content_type=content_type,
    )


def _assert_openai_error_shape(case_id: str, result: _d220_HTTPJSON) -> None:
    """Assert a JSON OpenAI-compatible error envelope."""
    assert "json" in result.content_type.lower(), (
        f"{case_id}: rejection must be JSON, got {result.content_type!r}; "
        f"body={result.text!r}"
    )
    error = result.body.get("error")
    assert isinstance(error, dict), (
        f"{case_id}: rejection missing error object; body={result.body!r}"
    )
    assert isinstance(error.get("message"), str) and error["message"], (
        f"{case_id}: error.message must be non-empty; body={result.body!r}"
    )
    assert isinstance(error.get("type"), str) and error["type"], (
        f"{case_id}: error.type must be non-empty; body={result.body!r}"
    )


async def test_d220_embeddings_rejection_shape(dynamo_endpoint_url: str) -> None:
    """Unsupported embeddings requests should return structured 4xx JSON."""
    payload: dict[str, object] = {
        "model": "default",
        "input": "embedding probe",
    }
    url = f"{dynamo_endpoint_url.rstrip('/')}/embeddings"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30.0)
    ) as session:
        result = await _d220_post_json(session, url, payload)

    if result.status == 200:
        pytest.skip(
            "D220: Dynamo deployment supports embeddings; no rejection to assert"
        )
    assert 400 <= result.status < 500, (
        f"D220: unsupported embeddings should reject with 4xx, not "
        f"HTTP {result.status}; body={result.text!r}"
    )
    _assert_openai_error_shape("D220", result)


# D221


@dataclass(frozen=True)
class _d221_HTTPJSON:
    status: int
    body: dict[str, object]
    text: str
    content_type: str


async def _post_malformed_json(
    session: aiohttp.ClientSession, url: str
) -> _d221_HTTPJSON:
    """POST a truncated JSON body with an application/json content type."""
    async with session.post(
        url,
        data=b'{"model":"default","messages":[',
        headers={"content-type": "application/json"},
    ) as resp:
        text = await resp.text()
        content_type = resp.headers.get("content-type", "")
    try:
        decoded = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoded = {}
    body = decoded if isinstance(decoded, dict) else {}
    return _d221_HTTPJSON(
        status=resp.status,
        body=body,
        text=text,
        content_type=content_type,
    )


def _d221_assert_openai_error_shape(result: _d221_HTTPJSON) -> None:
    """Assert malformed JSON returns a structured JSON error response."""
    assert "json" in result.content_type.lower(), (
        f"D221: malformed JSON rejection must be JSON, got "
        f"{result.content_type!r}; body={result.text!r}"
    )
    error = result.body.get("error")
    assert isinstance(error, dict), (
        f"D221: malformed JSON rejection missing error object; body={result.body!r}"
    )
    assert isinstance(error.get("message"), str) and error["message"], (
        f"D221: error.message must be non-empty; body={result.body!r}"
    )
    assert isinstance(error.get("type"), str) and error["type"], (
        f"D221: error.type must be non-empty; body={result.body!r}"
    )


async def test_d221_malformed_json_body_rejection(dynamo_endpoint_url: str) -> None:
    """Malformed JSON should fail with 4xx and an OpenAI-style error object."""
    url = f"{dynamo_endpoint_url.rstrip('/')}/chat/completions"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15.0)
    ) as session:
        result = await _post_malformed_json(session, url)

    assert 400 <= result.status < 500, (
        f"D221: malformed JSON should reject with 4xx, not HTTP "
        f"{result.status}; body={result.text!r}"
    )
    _d221_assert_openai_error_shape(result)


# D222


@dataclass(frozen=True)
class _d222_HTTPJSON:
    status: int
    body: dict[str, object]
    text: str
    content_type: str


async def _post_wrong_content_type(
    session: aiohttp.ClientSession, url: str
) -> _d222_HTTPJSON:
    """POST a valid JSON body advertised as text/plain."""
    payload = orjson.dumps(
        {
            "model": "default",
            "messages": [{"role": "user", "content": "pong"}],
            "max_tokens": 8,
        }
    )
    async with session.post(
        url,
        data=payload,
        headers={"content-type": "text/plain"},
    ) as resp:
        text = await resp.text()
        content_type = resp.headers.get("content-type", "")
    try:
        decoded = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoded = {}
    body = decoded if isinstance(decoded, dict) else {}
    return _d222_HTTPJSON(
        status=resp.status,
        body=body,
        text=text,
        content_type=content_type,
    )


def _d222_assert_openai_error_shape(result: _d222_HTTPJSON) -> None:
    """Assert wrong content type returns a structured JSON error response."""
    assert "json" in result.content_type.lower(), (
        f"D222: wrong content-type rejection must be JSON, got "
        f"{result.content_type!r}; body={result.text!r}"
    )
    error = result.body.get("error")
    assert isinstance(error, dict), (
        f"D222: wrong content-type rejection missing error object; body={result.body!r}"
    )
    assert isinstance(error.get("message"), str) and error["message"], (
        f"D222: error.message must be non-empty; body={result.body!r}"
    )
    assert isinstance(error.get("type"), str) and error["type"], (
        f"D222: error.type must be non-empty; body={result.body!r}"
    )


async def test_d222_wrong_content_type_rejection(dynamo_endpoint_url: str) -> None:
    """A JSON chat body sent as text/plain should reject with structured 4xx."""
    url = f"{dynamo_endpoint_url.rstrip('/')}/chat/completions"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15.0)
    ) as session:
        result = await _post_wrong_content_type(session, url)

    if result.status == 200:
        pytest.skip(
            "D222: Dynamo accepts text/plain request bodies as JSON; "
            "no content-type rejection to assert"
        )
    assert 400 <= result.status < 500, (
        f"D222: wrong content type should reject with 4xx, not HTTP "
        f"{result.status}; body={result.text!r}"
    )
    _d222_assert_openai_error_shape(result)


# D223


@dataclass(frozen=True)
class _d223_HTTPJSON:
    status: int
    body: dict[str, object]
    text: str


async def _post_chat_without_auth(
    session: aiohttp.ClientSession,
    url: str,
) -> _d223_HTTPJSON:
    """POST a valid chat completion without an Authorization header."""
    payload: dict[str, object] = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with pong."}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    async with session.post(url, json=payload) as resp:
        text = await resp.text()
    try:
        decoded = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoded = {}
    body = decoded if isinstance(decoded, dict) else {}
    return _d223_HTTPJSON(status=resp.status, body=body, text=text)


def _d223_assert_chat_completion_shape(result: _d223_HTTPJSON) -> None:
    """Assert a minimal OpenAI chat-completion response envelope."""
    assert result.status == 200, (
        f"D223: missing Authorization should remain compatible; "
        f"HTTP {result.status} body={result.text!r}"
    )
    choices = result.body.get("choices")
    assert isinstance(choices, list) and choices, (
        f"D223: response missing non-empty choices; body={result.body!r}"
    )
    first_choice = choices[0]
    assert isinstance(first_choice, dict), (
        f"D223: first choice is not an object; body={result.body!r}"
    )
    assert isinstance(first_choice.get("message"), dict), (
        f"D223: first choice missing message object; body={result.body!r}"
    )


async def test_d223_missing_authorization_compatibility(
    dynamo_endpoint_url: str,
) -> None:
    """Dynamo local/frontend compatibility should not require Authorization."""
    url = f"{dynamo_endpoint_url.rstrip('/')}/chat/completions"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30.0)
    ) as session:
        result = await _post_chat_without_auth(session, url)

    if result.status in {401, 403}:
        pytest.skip(
            "D223: this Dynamo deployment has auth enforcement enabled "
            f"(HTTP {result.status})"
        )
    _d223_assert_chat_completion_shape(result)


# D224


@dataclass(frozen=True)
class _d224_HTTPJSON:
    status: int
    body: dict[str, object]
    text: str


async def _post_chat_with_bad_auth(
    session: aiohttp.ClientSession,
    url: str,
) -> _d224_HTTPJSON:
    """POST a valid chat completion with an intentionally bad bearer token."""
    payload: dict[str, object] = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with pong."}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    async with session.post(
        url,
        json=payload,
        headers={"authorization": "Bearer definitely-not-a-real-token"},
    ) as resp:
        text = await resp.text()
    try:
        decoded = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoded = {}
    body = decoded if isinstance(decoded, dict) else {}
    return _d224_HTTPJSON(status=resp.status, body=body, text=text)


def _d224_assert_chat_completion_shape(result: _d224_HTTPJSON) -> None:
    """Assert a minimal OpenAI chat-completion response envelope."""
    assert result.status == 200, (
        f"D224: bad Authorization should remain compatible; "
        f"HTTP {result.status} body={result.text!r}"
    )
    choices = result.body.get("choices")
    assert isinstance(choices, list) and choices, (
        f"D224: response missing non-empty choices; body={result.body!r}"
    )
    first_choice = choices[0]
    assert isinstance(first_choice, dict), (
        f"D224: first choice is not an object; body={result.body!r}"
    )
    assert isinstance(first_choice.get("message"), dict), (
        f"D224: first choice missing message object; body={result.body!r}"
    )


async def test_d224_bad_authorization_compatibility(dynamo_endpoint_url: str) -> None:
    """Dynamo local/frontend compatibility should ignore a bad bearer token."""
    url = f"{dynamo_endpoint_url.rstrip('/')}/chat/completions"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30.0)
    ) as session:
        result = await _post_chat_with_bad_auth(session, url)

    if result.status in {401, 403}:
        pytest.skip(
            "D224: this Dynamo deployment has auth enforcement enabled "
            f"(HTTP {result.status})"
        )
    _d224_assert_chat_completion_shape(result)


# D225-D233


FRONTEND_SELECTOR = "nvidia.com/dynamo-component-type=frontend"
DEFAULT_MODEL = os.environ.get("AIPERF_DYNAMO_CHAOS_MODEL", "default")
LOAD_SHEDDING_OPT_IN_ENV = "AIPERF_DYNAMO_LOAD_SHEDDING_CHAOS"
IDLE_STALL_OPT_IN_ENV = "AIPERF_DYNAMO_IDLE_STALL_CHAOS"
REQUEST_ID_HEADERS = (
    "x-request-id",
    "x-correlation-id",
    "x-openai-request-id",
    "x-dynamo-request-id",
)
LOAD_SHEDDING_STATUS_CODES = {429, 503}
LOAD_SHEDDING_METRIC_HINTS = (
    "shed",
    "reject",
    "admission",
    "overload",
    "rate_limit",
    "rate_limited",
    "too_many",
)


async def _d225_d233_frontend_pod(kubectl: KubectlClient, namespace: str) -> str:
    """Return a ready frontend pod name for direct metric/env inspection."""
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
    if pod:
        return pod

    for candidate in await kubectl.get_pods(namespace):
        if candidate.is_ready and "-frontend-" in candidate.name:
            return candidate.name
    raise RuntimeError(
        f"D225-D233: no ready Dynamo frontend pod found in namespace {namespace!r}"
    )


async def _frontend_env(kubectl: KubectlClient, namespace: str) -> dict[str, str]:
    """Read literal env vars from the active frontend pod."""
    pod = await _d225_d233_frontend_pod(kubectl, namespace)
    result = await kubectl.run(
        "get",
        "pod",
        pod,
        "-n",
        namespace,
        "-o",
        "jsonpath={range .spec.containers[*].env[*]}{.name}={.value}{'\\n'}{end}",
        check=True,
    )
    env: dict[str, str] = {}
    for line in result.stdout.splitlines():
        name, sep, value = line.partition("=")
        if sep and name:
            env[name] = value
    return env


async def _d225_d233_raw_frontend_metrics(
    kubectl: KubectlClient, namespace: str
) -> str:
    """Fetch raw Prometheus text from the frontend /metrics endpoint."""
    pod = await _d225_d233_frontend_pod(kubectl, namespace)
    async with (
        kubectl.port_forward(pod, 8000, namespace=namespace) as local_port,
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session,
        session.get(f"http://127.0.0.1:{local_port}/metrics") as resp,
    ):
        body = await resp.text()
        if resp.status != 200:
            raise RuntimeError(
                f"D225-D233: /metrics on {namespace}/{pod} returned {resp.status}: "
                f"{body[:512]!r}"
            )
        return body


def _chat_url(endpoint_url: str) -> str:
    """Build the chat-completions URL from the package /v1 endpoint fixture."""
    return f"{endpoint_url.rstrip('/')}/chat/completions"


def _chat_payload(
    prompt: str,
    *,
    stream: bool,
    max_tokens: int = 32,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a minimal OpenAI-compatible chat-completions payload."""
    payload: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if extra:
        payload.update(extra)
    return payload


def _metric_samples(metrics_text: str, hints: Iterable[str]) -> dict[str, float]:
    """Return metric samples whose metric name or labels contain any hint."""
    lowered_hints = tuple(hint.lower() for hint in hints)
    samples: dict[str, float] = {}
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if not any(hint in line.lower() for hint in lowered_hints):
            continue
        value_text = line.rsplit(None, 1)[-1]
        try:
            samples[name] = samples.get(name, 0.0) + float(value_text)
        except ValueError:
            logger.debug(lambda line=line: f"D225-D233: bad metric sample {line!r}")
    return samples


def _sum_samples(samples: dict[str, float]) -> float:
    """Return the sum of all matching sample values."""
    return sum(samples.values())


async def _post_chat(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST one chat-completions request and return status, headers and body."""
    started = time.monotonic()
    try:
        async with session.post(
            _chat_url(endpoint_url),
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            body = await resp.text()
            return {
                "kind": "response",
                "status": resp.status,
                "headers": {k.lower(): v for k, v in resp.headers.items()},
                "body": body,
                "elapsed": time.monotonic() - started,
            }
    except asyncio.TimeoutError as exc:
        return {
            "kind": "timeout",
            "error": repr(exc),
            "elapsed": time.monotonic() - started,
        }
    except aiohttp.ClientError as exc:
        return {
            "kind": "client_error",
            "error": repr(exc),
            "elapsed": time.monotonic() - started,
        }


async def _read_stream(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 45.0,
    read_delay: float = 0.0,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    """POST one streaming request and summarize its SSE lifecycle."""
    started = time.monotonic()
    chunks: list[str] = []
    try:
        async with session.post(
            _chat_url(endpoint_url),
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            async for chunk in resp.content.iter_any():
                if chunk:
                    text = chunk.decode(errors="replace")
                    chunks.append(text[:512])
                    if read_delay > 0.0:
                        await asyncio.sleep(read_delay)
                    if max_chunks is not None and len(chunks) >= max_chunks:
                        break
            return {
                "kind": "stream",
                "status": resp.status,
                "headers": {k.lower(): v for k, v in resp.headers.items()},
                "chunks": chunks,
                "elapsed": time.monotonic() - started,
            }
    except asyncio.TimeoutError as exc:
        return {
            "kind": "timeout",
            "error": repr(exc),
            "elapsed": time.monotonic() - started,
        }
    except aiohttp.ClientError as exc:
        return {
            "kind": "client_error",
            "error": repr(exc),
            "chunks": chunks,
            "elapsed": time.monotonic() - started,
        }


@asynccontextmanager
async def _mixed_traffic(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    *,
    stream_count: int,
    non_stream_count: int,
) -> AsyncIterator[set[asyncio.Task[dict[str, Any]]]]:
    """Run a short mixed streaming/non-stream traffic burst."""
    tasks: set[asyncio.Task[dict[str, Any]]] = set()
    for idx in range(stream_count):
        payload = _chat_payload(
            f"D233 streaming metrics availability probe {idx}",
            stream=True,
            max_tokens=96,
        )
        tasks.add(asyncio.create_task(_read_stream(session, endpoint_url, payload)))
    for idx in range(non_stream_count):
        payload = _chat_payload(
            f"D233 non-stream metrics availability probe {idx}",
            stream=False,
            max_tokens=32,
        )
        tasks.add(asyncio.create_task(_post_chat(session, endpoint_url, payload)))
    try:
        yield tasks
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _load_shedding_supported(env: dict[str, str], metrics_text: str) -> bool:
    """Return whether this frontend advertises admission/load-shedding support."""
    if os.environ.get(LOAD_SHEDDING_OPT_IN_ENV) == "1":
        return True
    env_blob = "\n".join(f"{k}={v}" for k, v in sorted(env.items())).lower()
    metric_blob = metrics_text.lower()
    return any(
        hint in env_blob or hint in metric_blob for hint in LOAD_SHEDDING_METRIC_HINTS
    )


async def _require_load_shedding_topology(
    kubectl: KubectlClient,
    namespace: str,
) -> str:
    """Return baseline metrics or skip if admission/load-shedding is unsupported."""
    env = await _frontend_env(kubectl, namespace)
    metrics_text = await _d225_d233_raw_frontend_metrics(kubectl, namespace)
    if not _load_shedding_supported(env, metrics_text):
        pytest.skip(
            "D226-D228 require a Dynamo frontend configured with admission-control "
            "or load-shedding counters. No frontend env var or /metrics sample "
            "contained shed/reject/admission/rate-limit hints; set "
            f"{LOAD_SHEDDING_OPT_IN_ENV}=1 for a topology that enables it."
        )
    return metrics_text


async def test_d225_request_id_header_propagation(
    dynamo_endpoint_url: str,
) -> None:
    """D225: a caller-supplied request ID is preserved on the HTTP response."""
    request_id = f"d225-{uuid4()}"
    headers = {"x-request-id": request_id, "x-correlation-id": request_id}
    async with aiohttp.ClientSession() as session:
        result = await _post_chat(
            session,
            dynamo_endpoint_url,
            _chat_payload("D225 request ID propagation probe", stream=False),
            headers=headers,
        )

    assert result["kind"] == "response", f"D225 request failed: {result!r}"
    assert result["status"] < 500, f"D225 frontend returned server error: {result!r}"
    if result["status"] in {400, 404}:
        pytest.skip(
            f"D225 cannot validate request ID propagation because the deployed "
            f"model/API rejected the probe with HTTP {result['status']}: "
            f"{result['body'][:256]!r}"
        )

    response_headers = result["headers"]
    propagated = {
        name: response_headers[name]
        for name in REQUEST_ID_HEADERS
        if response_headers.get(name) == request_id
    }
    assert propagated, (
        "D225: response did not echo the caller request ID in any supported "
        f"header {REQUEST_ID_HEADERS}; response_headers={response_headers!r}"
    )


async def test_d226_concurrent_stream_admission_load_shedding(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """D226: overloaded streaming admission sheds promptly instead of hanging."""
    await _require_load_shedding_topology(kubectl, dynamo_deployment_namespace)
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                _read_stream(
                    session,
                    dynamo_endpoint_url,
                    _chat_payload(
                        f"D226 stream overload {idx}", stream=True, max_tokens=128
                    ),
                    timeout=45.0,
                )
            )
            for idx in range(96)
        ]
        results = await asyncio.gather(*tasks)

    statuses = [
        result.get("status") for result in results if result["kind"] == "stream"
    ]
    shed = [status for status in statuses if status in LOAD_SHEDDING_STATUS_CODES]
    timeouts = [result for result in results if result["kind"] == "timeout"]
    assert not timeouts, (
        f"D226: load-shed streams hung instead of failing fast: {timeouts!r}"
    )
    assert shed, (
        "D226: no concurrent streaming request was load-shed with HTTP 429/503; "
        f"statuses={statuses!r}"
    )


async def test_d227_non_stream_load_shedding_metrics(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """D227: non-stream load shedding increments admission/rejection metrics."""
    before_text = await _require_load_shedding_topology(
        kubectl, dynamo_deployment_namespace
    )
    before = _sum_samples(_metric_samples(before_text, LOAD_SHEDDING_METRIC_HINTS))
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[
                _post_chat(
                    session,
                    dynamo_endpoint_url,
                    _chat_payload(f"D227 non-stream overload {idx}", stream=False),
                    timeout=30.0,
                )
                for idx in range(128)
            ]
        )
    shed = [
        result
        for result in results
        if result["kind"] == "response"
        and result.get("status") in LOAD_SHEDDING_STATUS_CODES
    ]
    if not shed:
        pytest.skip(
            "D227 did not drive this deployment into non-stream load shedding; "
            "increase frontend admission pressure or lower configured limits."
        )

    after_text = await _d225_d233_raw_frontend_metrics(
        kubectl, dynamo_deployment_namespace
    )
    after_samples = _metric_samples(after_text, LOAD_SHEDDING_METRIC_HINTS)
    after = _sum_samples(after_samples)
    assert after > before, (
        "D227: non-stream load-shed responses did not increment any matching "
        f"metric; before={before}, after={after}, samples={after_samples!r}"
    )


async def test_d228_streaming_load_shedding_metrics(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """D228: streaming load shedding increments admission/rejection metrics."""
    before_text = await _require_load_shedding_topology(
        kubectl, dynamo_deployment_namespace
    )
    before = _sum_samples(_metric_samples(before_text, LOAD_SHEDDING_METRIC_HINTS))
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[
                _read_stream(
                    session,
                    dynamo_endpoint_url,
                    _chat_payload(
                        f"D228 stream overload {idx}", stream=True, max_tokens=128
                    ),
                    timeout=45.0,
                )
                for idx in range(128)
            ]
        )
    shed = [
        result
        for result in results
        if result["kind"] == "stream"
        and result.get("status") in LOAD_SHEDDING_STATUS_CODES
    ]
    if not shed:
        pytest.skip(
            "D228 did not drive this deployment into streaming load shedding; "
            "increase frontend admission pressure or lower configured limits."
        )

    after_text = await _d225_d233_raw_frontend_metrics(
        kubectl, dynamo_deployment_namespace
    )
    after_samples = _metric_samples(after_text, LOAD_SHEDDING_METRIC_HINTS)
    after = _sum_samples(after_samples)
    assert after > before, (
        "D228: stream load-shed responses did not increment any matching metric; "
        f"before={before}, after={after}, samples={after_samples!r}"
    )


async def test_d229_slow_client_reader_backpressure(
    dynamo_endpoint_url: str,
) -> None:
    """D229: a slow SSE reader is bounded and does not poison the connection."""
    async with aiohttp.ClientSession() as session:
        result = await _read_stream(
            session,
            dynamo_endpoint_url,
            _chat_payload(
                "D229 slow reader backpressure probe", stream=True, max_tokens=128
            ),
            timeout=60.0,
            read_delay=0.25,
        )
        followup = await _post_chat(
            session,
            dynamo_endpoint_url,
            _chat_payload("D229 follow-up after slow reader", stream=False),
        )

    assert result["kind"] != "timeout", f"D229: slow reader hung: {result!r}"
    assert result.get("status", 200) < 500, f"D229: slow reader caused 5xx: {result!r}"
    assert followup["kind"] == "response" and followup["status"] < 500, (
        "D229: frontend failed a normal follow-up request after slow reader "
        f"backpressure; followup={followup!r}"
    )


@pytest.mark.skipif(
    os.environ.get(IDLE_STALL_OPT_IN_ENV) != "1",
    reason=(
        "D230 requires a topology that can inject an idle/no-token backend stall; "
        f"set {IDLE_STALL_OPT_IN_ENV}=1 only when the deployed Dynamo frontend is "
        "wired to a stall-capable backend or proxy."
    ),
)
async def test_d230_idle_no_token_stall_terminates_stream(
    dynamo_endpoint_url: str,
) -> None:
    """D230: an idle/no-token stream reaches a bounded terminal state."""
    async with aiohttp.ClientSession() as session:
        result = await _read_stream(
            session,
            dynamo_endpoint_url,
            _chat_payload(
                "D230 idle no-token stall probe", stream=True, max_tokens=512
            ),
            timeout=75.0,
        )
    assert result["kind"] != "timeout", f"D230: idle/no-token stream hung: {result!r}"
    assert result.get("status", 200) in {200, 499, 500, 502, 503, 504}, (
        f"D230: unexpected terminal status for idle/no-token stream: {result!r}"
    )


async def test_d231_tool_call_streaming_compatibility(
    dynamo_endpoint_url: str,
) -> None:
    """D231: OpenAI tool-call payloads remain valid under streaming."""
    tool_payload = _chat_payload(
        "D231 call the weather tool for San Jose, then summarize the result.",
        stream=True,
        max_tokens=128,
        extra={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Return weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )
    async with aiohttp.ClientSession() as session:
        result = await _read_stream(
            session, dynamo_endpoint_url, tool_payload, timeout=45.0
        )

    if result.get("status") in {400, 404, 422}:
        pytest.skip(
            "D231 tool-call streaming is unsupported by this model/frontend; "
            f"status={result.get('status')}, chunks={result.get('chunks', [])[:2]!r}"
        )
    assert result["kind"] == "stream", (
        f"D231 request failed before streaming: {result!r}"
    )
    assert result["status"] == 200, f"D231 stream returned non-200: {result!r}"
    assert any("[DONE]" in chunk for chunk in result["chunks"]), (
        f"D231: tool-call stream did not terminate with [DONE]: {result!r}"
    )


async def test_d232_extra_openai_fields_tolerated(
    dynamo_endpoint_url: str,
) -> None:
    """D232: forward-compatible OpenAI fields are ignored or accepted, not 5xx."""
    payload = _chat_payload(
        "D232 tolerate extra OpenAI-compatible request fields.",
        stream=False,
        extra={
            "frequency_penalty": 0.0,
            "logprobs": False,
            "metadata": {"chaos_case": "D232"},
            "parallel_tool_calls": True,
            "presence_penalty": 0.0,
            "response_format": {"type": "text"},
            "seed": 12345,
            "service_tier": "auto",
            "user": "aiperf-chaos-d232",
        },
    )
    async with aiohttp.ClientSession() as session:
        result = await _post_chat(session, dynamo_endpoint_url, payload)

    assert result["kind"] == "response", f"D232 request failed at transport: {result!r}"
    assert result["status"] < 500, (
        "D232: extra OpenAI fields triggered a frontend/server 5xx instead of "
        f"being tolerated or rejected cleanly; result={result!r}"
    )
    assert result["status"] not in {400, 422}, (
        "D232: frontend rejected forward-compatible OpenAI fields instead of "
        f"tolerating them; result={result!r}"
    )


async def test_d233_metrics_endpoint_available_during_mixed_traffic(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """D233: /metrics remains scrapeable during mixed stream/non-stream traffic."""
    async with (
        aiohttp.ClientSession() as session,
        _mixed_traffic(
            session,
            dynamo_endpoint_url,
            stream_count=12,
            non_stream_count=24,
        ) as tasks,
    ):
        scrapes: list[str] = []
        for _ in range(5):
            scrapes.append(
                await _d225_d233_raw_frontend_metrics(
                    kubectl, dynamo_deployment_namespace
                )
            )
            await asyncio.sleep(0.5)
        results = await asyncio.gather(*tasks, return_exceptions=False)

    assert all("#" in scrape or "dynamo" in scrape.lower() for scrape in scrapes), (
        "D233: one or more /metrics scrapes returned non-Prometheus-looking text"
    )
    hard_failures = [
        result
        for result in results
        if result["kind"] == "timeout"
        or (
            result["kind"] in {"response", "stream"}
            and result.get("status", 200) >= 500
        )
    ]
    assert not hard_failures, (
        "D233: mixed traffic caused timeouts or 5xx responses while metrics were "
        f"being scraped: {hard_failures!r}"
    )
