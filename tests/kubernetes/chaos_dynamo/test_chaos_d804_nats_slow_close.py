# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D804 -- NATS slow-close toxic against the stats bus.

Scenario:
    Apply a Toxiproxy ``slow_close`` toxic to Dynamo's NATS route while
    host-side SSE traffic continues against the frontend. NATS carries
    service-stats requests such as ``$SRV.STATS.<service>``; slow close should
    time out cleanly and recover after the toxic is removed, not leave the
    router metrics scrape path permanently broken.

Pre-condition:
    The live Dynamo topology must actually route NATS traffic through the
    reserved chaos Toxiproxy endpoint. The default v1.1.0 ``disagg-1gpu``
    topology talks to NATS directly, so this test skips with the observed NATS
    selector and missing proxy route instead of applying a no-op toxic.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_dynamo.conftest import scrape_frontend_metrics
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

NATS_NAMESPACE = "dynamo-system"
"""Namespace where the Dynamo platform chart installs NATS."""

NATS_SELECTOR = "app=nats"
"""Default NATS pod selector used by ``StoreInjector``."""

NATS_PROXY_NAME = "nats"
"""Toxiproxy proxy name reserved for NATS in the unified chaos fixture."""

NATS_PROXY_LISTEN = "0.0.0.0:20020"
"""Listen address inside the toxiproxy pod; port 20020 is reserved for NATS."""

NATS_PROXY_ROUTE = "toxiproxy.chaos-toxiproxy.svc:20020"
"""Cluster route that proves Dynamo clients traverse the NATS Toxiproxy proxy."""

NATS_SERVICE_PORT = 4222
"""NATS client port exposed by the platform chart."""

SLOW_CLOSE_DELAY_MS = 5_000
"""Delay injected when either side closes the NATS TCP connection."""

TOXIC_WINDOW_SECS = 15.0
"""Duration to hold the slow-close toxic while frontend traffic runs."""

RECOVERY_SECS = 20.0
"""Budget after toxic removal for metrics scrape recovery."""

CONCURRENCY = 4
"""Background SSE workers hitting the frontend during the toxic window."""

REQUESTS_PER_TASK = 4
"""Sequential requests each worker submits before exiting."""

REQUEST_INTERVAL_SECS = 0.25
"""Inter-request pause inside each worker loop."""

COMPLETED_METRIC = "dynamo_frontend_requests_total"
"""Frontend request counter used as a cluster-side traffic signal."""

ROUTER_OVERHEAD_METRIC_PREFIX = "dynamo_component_router_overhead_total_ms"
"""Router overhead histogram prefix that should still scrape after recovery."""

_NATS_CHAOS_OPT_IN_ENV = "AIPERF_DYNAMO_NATS_CHAOS"
"""Opt-in proving the target topology routes NATS through Toxiproxy."""


@dataclass(frozen=True, slots=True)
class _NatsRouteObservation:
    """Observed NATS route prerequisites for D804."""

    nats_pods: list[str]
    nats_service: str | None
    routed_values: list[str]

    @property
    def is_routed_through_toxiproxy(self) -> bool:
        """Return whether any observed pod config points NATS at Toxiproxy."""
        return any(NATS_PROXY_ROUTE in value for value in self.routed_values)

    def skip_reason(self) -> str:
        """Build the prerequisite skip reason with concrete observed state."""
        return (
            f"D804 requires live NATS traffic to route through {NATS_PROXY_ROUTE!r} "
            f"so store.nats.slow_close reaches product traffic. Observed "
            f"NATS selector {NATS_SELECTOR!r} in namespace {NATS_NAMESPACE!r}: "
            f"pods={self.nats_pods!r}, service={self.nats_service!r}, "
            f"route_values={self.routed_values!r}; missing proxy route. "
            "Default Dynamo v1.1.0 disagg routes NATS directly."
        )


def _d804_static_skip_reason() -> str | None:
    """Return why the default topology cannot run D804 before cluster setup."""
    if os.environ.get(_NATS_CHAOS_OPT_IN_ENV) == "1":
        return None
    return (
        f"D804 requires live NATS traffic to route through {NATS_PROXY_ROUTE!r} "
        "before cluster setup; "
        f"observed NATS selector {NATS_SELECTOR!r} in namespace {NATS_NAMESPACE!r}; "
        "missing proxy route. Default Dynamo v1.1.0 disagg routes NATS "
        f"directly. Set {_NATS_CHAOS_OPT_IN_ENV}=1 only for an externally "
        "managed topology that routes NATS through Toxiproxy."
    )


async def test_d804_nats_slow_close_recovers_metrics_and_traffic(
    request: pytest.FixtureRequest,
) -> None:
    """Inject NATS slow-close only when the topology routes NATS via Toxiproxy."""
    static_skip_reason = _d804_static_skip_reason()
    if static_skip_reason is not None:
        pytest.skip(static_skip_reason)

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    dynamo_deployment_namespace: str = request.getfixturevalue(
        "dynamo_deployment_namespace"
    )

    observation = await _observe_nats_route(kubectl, dynamo_deployment_namespace)
    if not observation.is_routed_through_toxiproxy:
        pytest.skip(observation.skip_reason())
    if observation.nats_service is None:
        pytest.skip(
            f"D804 observed routed NATS config but no NATS service for selector "
            f"{NATS_SELECTOR!r} in namespace {NATS_NAMESPACE!r}; "
            f"pods={observation.nats_pods!r}"
        )

    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    faults = request.getfixturevalue("faults")
    dynamo_endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")
    upstream = f"{observation.nats_service}.{NATS_NAMESPACE}.svc:{NATS_SERVICE_PORT}"

    proxy_created = False
    try:
        await dynamo_toxiproxy.add_proxy(
            name=NATS_PROXY_NAME,
            listen=NATS_PROXY_LISTEN,
            upstream=upstream,
        )
        proxy_created = True
        await _run_slow_close_assertion(
            faults=faults,
            kubectl=kubectl,
            dynamo_endpoint_url=dynamo_endpoint_url,
            dynamo_deployment_namespace=dynamo_deployment_namespace,
        )
    finally:
        if proxy_created:
            try:
                await dynamo_toxiproxy.remove_proxy(NATS_PROXY_NAME)
            except Exception as exc:
                logger.warning(lambda exc=exc: f"D804 remove_proxy failed: {exc!r}")


async def _run_slow_close_assertion(
    *,
    faults,  # noqa: ANN001 - InjectorRegistry, see chaos_dynamo.conftest
    kubectl: KubectlClient,
    dynamo_endpoint_url: str,
    dynamo_deployment_namespace: str,
) -> None:
    """Apply the slow-close toxic and verify traffic plus metrics recovery."""
    metrics_before = await scrape_frontend_metrics(kubectl, dynamo_deployment_namespace)
    stop_event = asyncio.Event()
    request_counter = {"completed": 0, "errors": 0}
    workers = [
        asyncio.create_task(
            _traffic_worker(dynamo_endpoint_url, stop_event, request_counter, idx)
        )
        for idx in range(CONCURRENCY)
    ]

    try:
        async with faults.inject(
            "store.nats.slow_close",
            target={"proxy": NATS_PROXY_NAME},
            attributes={"delay": SLOW_CLOSE_DELAY_MS},
            stream="downstream",
        ) as applied:
            assert applied.spec.fault_id == "network.slow_close"
            assert applied.metadata.get("proxy_name") == NATS_PROXY_NAME
            await asyncio.sleep(TOXIC_WINDOW_SECS)
            metrics_during = await _try_scrape_frontend_metrics(
                kubectl, dynamo_deployment_namespace
            )

        await asyncio.sleep(RECOVERY_SECS)
        metrics_after = await _scrape_frontend_metrics_with_retries(
            kubectl, dynamo_deployment_namespace
        )

        completed_during = float(request_counter["completed"])
        if metrics_during is not None:
            completed_during = max(
                completed_during,
                _metric_delta(metrics_during, metrics_before, COMPLETED_METRIC),
            )
        completed_after = max(
            float(request_counter["completed"]),
            _metric_delta(metrics_after, metrics_before, COMPLETED_METRIC),
        )
        assert completed_during > 0 or completed_after > 0, (
            "D804: frontend traffic neither continued during NATS slow-close nor "
            f"recovered afterward (client_completed={request_counter['completed']}, "
            f"client_errors={request_counter['errors']}, "
            f"metrics_before_completed={metrics_before.get(COMPLETED_METRIC, 0.0)}, "
            f"metrics_after_completed={metrics_after.get(COMPLETED_METRIC, 0.0)})"
        )
        _assert_router_overhead_metric_recovered(metrics_before, metrics_after)
    finally:
        stop_event.set()
        for worker in workers:
            worker.cancel()
        for worker in workers:
            with contextlib.suppress(asyncio.CancelledError):
                await worker


async def _traffic_worker(
    dynamo_endpoint_url: str,
    stop_event: asyncio.Event,
    request_counter: dict[str, int],
    idx: int,
) -> None:
    """Issue short streaming chat requests until the test asks workers to stop."""
    async with aiohttp.ClientSession() as session:
        for _ in range(REQUESTS_PER_TASK):
            if stop_event.is_set():
                return
            payload = {
                "model": "Qwen/Qwen3-0.6B",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "max_tokens": 10,
            }
            try:
                async with session.post(
                    dynamo_endpoint_url + "/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    async for _chunk in resp.content.iter_chunked(1024):
                        pass
                request_counter["completed"] += 1
            except (aiohttp.ClientError, TimeoutError) as exc:
                logger.warning(
                    lambda exc=exc, idx=idx: f"D804 worker {idx} error: {exc!r}"
                )
                request_counter["errors"] += 1
            await asyncio.sleep(REQUEST_INTERVAL_SECS)


async def _observe_nats_route(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> _NatsRouteObservation:
    """Inspect live pods/services to decide whether D804 can affect NATS."""
    nats_pods = await _list_pods_with_selector(kubectl, NATS_NAMESPACE, NATS_SELECTOR)
    nats_service = await _find_nats_service(kubectl)
    route_values: list[str] = []
    for namespace in dict.fromkeys([dynamo_deployment_namespace, NATS_NAMESPACE]):
        route_values.extend(await _collect_nats_route_values(kubectl, namespace))
    return _NatsRouteObservation(
        nats_pods=nats_pods,
        nats_service=nats_service,
        routed_values=sorted(set(route_values)),
    )


async def _list_pods_with_selector(
    kubectl: KubectlClient,
    namespace: str,
    selector: str,
) -> list[str]:
    """Return pod names matching ``selector`` or ``[]`` if the lookup fails."""
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


async def _find_nats_service(kubectl: KubectlClient) -> str | None:
    """Return a NATS service name suitable for the Toxiproxy upstream."""
    result = await kubectl.run(
        "get",
        "services",
        "-n",
        NATS_NAMESPACE,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return None
    services = orjson.loads(result.stdout).get("items", [])
    for service in services:
        metadata = service.get("metadata", {})
        labels = metadata.get("labels", {})
        ports = service.get("spec", {}).get("ports", [])
        exposes_nats_port = any(port.get("port") == NATS_SERVICE_PORT for port in ports)
        name = metadata.get("name")
        if exposes_nats_port and (
            labels.get("app") == "nats" or (isinstance(name, str) and "nats" in name)
        ):
            return name
    return None


async def _collect_nats_route_values(
    kubectl: KubectlClient,
    namespace: str,
) -> list[str]:
    """Return env/arg values mentioning NATS or the reserved Toxiproxy route."""
    result = await kubectl.run(
        "get",
        "pods",
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return []
    pods = orjson.loads(result.stdout).get("items", [])
    values: list[str] = []
    for pod in pods:
        spec = pod.get("spec", {})
        for container in spec.get("containers", []):
            values.extend(_container_nats_values(container))
    return values


def _container_nats_values(container: dict[str, object]) -> list[str]:
    """Extract literal NATS route hints from one Kubernetes container spec."""
    values: list[str] = []
    for env in container.get("env", []):
        if not isinstance(env, dict):
            continue
        name = env.get("name")
        value = env.get("value")
        if not isinstance(value, str):
            continue
        if _mentions_nats_route(name, value):
            values.append(f"{name}={value}")
    for field in ("command", "args"):
        items = container.get(field, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str) and _mentions_nats_route(None, item):
                values.append(item)
    return values


def _mentions_nats_route(name: object, value: str) -> bool:
    """Return whether an env/arg value is relevant to the NATS route probe."""
    name_text = name if isinstance(name, str) else ""
    haystack = f"{name_text}={value}".lower()
    return "nats" in haystack or NATS_PROXY_ROUTE in value


async def _try_scrape_frontend_metrics(
    kubectl: KubectlClient,
    namespace: str,
) -> dict[str, float] | None:
    """Scrape metrics once, returning ``None`` if slow-close causes a timeout."""
    try:
        return await scrape_frontend_metrics(kubectl, namespace)
    except (RuntimeError, aiohttp.ClientError, TimeoutError) as exc:
        logger.warning(
            lambda exc=exc: f"D804 metrics scrape during toxic failed: {exc!r}"
        )
        return None


async def _scrape_frontend_metrics_with_retries(
    kubectl: KubectlClient,
    namespace: str,
) -> dict[str, float]:
    """Retry metrics scrape after toxic removal; failure here is permanent."""
    last_exc: Exception | None = None
    for _ in range(6):
        try:
            return await scrape_frontend_metrics(kubectl, namespace)
        except (RuntimeError, aiohttp.ClientError, TimeoutError) as exc:
            last_exc = exc
            await asyncio.sleep(5.0)
    raise AssertionError(
        "D804: frontend metrics scrape did not recover after NATS slow-close "
        f"toxic removal; last_error={last_exc!r}"
    )


def _assert_router_overhead_metric_recovered(
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
) -> None:
    """Assert the router-overhead histogram did not disappear after recovery."""
    before_keys = _router_overhead_keys(metrics_before)
    after_keys = _router_overhead_keys(metrics_after)
    if before_keys:
        assert after_keys, (
            "D804: router overhead histogram disappeared after NATS slow-close; "
            f"before_keys={before_keys!r}, sample_after_keys={sorted(metrics_after)[:20]!r}"
        )


def _router_overhead_keys(metrics: dict[str, float]) -> list[str]:
    """Return metric keys belonging to the router overhead histogram."""
    return sorted(
        key for key in metrics if key.startswith(ROUTER_OVERHEAD_METRIC_PREFIX)
    )


def _metric_delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    """Return the non-negative increment in a counter between two scrapes."""
    return max(after.get(key, 0.0) - before.get(key, 0.0), 0.0)
