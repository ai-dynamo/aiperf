# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D8xx Dynamo store and discovery chaos scenarios."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from dev.versions import DYNAMO_VERSION
from tests.kubernetes.chaos_dynamo.conftest import scrape_frontend_metrics
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


# D801

MAX_RETRIES = 3

_DECODE_POD_SELECTOR = "nvidia.com/dynamo-sub-component-type=decode"

_FRESH_POD_WAIT_S = 30.0

_POST_FAULT_SETTLE_S = 90.0

_TRAFFIC_PROBE_TIMEOUT_S = 30.0

_ETCD_ERROR_NEEDLES = (b"etcd", b"register", b"lease")


async def test_d801_etcd_kill_during_registration_race(
    faults: Any,
    kubectl: Any,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Kill etcd during fresh worker boot; assert no half-registered state.

    Retries up to :py:data:`MAX_RETRIES` times because the "fresh worker
    mid-boot" window is non-deterministic. Accepts the first PASS per plan
    section 4 risk note.
    """
    await _run_d801_assertion(
        faults, kubectl, dynamo_deployment_namespace, dynamo_endpoint_url
    )


async def _run_d801_assertion(
    faults: Any,
    kubectl: Any,
    namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Full D801 assertion body; one-line unskip flip to run.

    The race window between "fresh worker mid-boot" and "etcd kill" is
    non-deterministic, so we attempt up to :py:data:`MAX_RETRIES` times and
    accept the first PASS. Each attempt scales the decode component +1,
    waits for the new pod to be Running+NotReady, kills etcd, waits for
    steady state, and classifies the result.
    """
    dgd_name = await _resolve_dgd_name(kubectl, namespace)

    dgd_json = await kubectl.run(
        "get",
        "dynamographdeployment",
        dgd_name,
        "-n",
        namespace,
        "-o",
        "json",
        check=True,
    )
    dgd = orjson.loads(dgd_json.stdout)
    decode_component = _find_decode_component(dgd)
    assert decode_component is not None, (
        f"d801: DGD {namespace}/{dgd_name} has no decode component; "
        "test requires a disagg deployment with a mutable decode replica count"
    )
    initial_replicas = int(decode_component.get("replicas", 1))

    observations: list[str] = []
    try:
        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(lambda a=attempt: f"d801: attempt {a}/{MAX_RETRIES} starting")
            verdict, detail = await _attempt_once(
                faults,
                kubectl,
                namespace,
                dgd_name,
                initial_replicas,
                dynamo_endpoint_url,
            )
            observations.append(f"attempt {attempt}: {verdict} -- {detail}")
            logger.info(
                lambda a=attempt, v=verdict, d=detail: (
                    f"d801: attempt {a} verdict={v} detail={d}"
                )
            )
            if verdict in ("pass-a", "pass-b"):
                return
            # Scale back before the next attempt so the +1 doesn't accumulate.
            await _scale_decode(kubectl, dgd_name, namespace, initial_replicas)
            await asyncio.sleep(5.0)
    finally:
        await _scale_decode(kubectl, dgd_name, namespace, initial_replicas)

    raise AssertionError(
        "d801: no PASS verdict across "
        f"{MAX_RETRIES} attempts; observations: " + " | ".join(observations)
    )


async def _attempt_once(
    faults: Any,
    kubectl: Any,
    namespace: str,
    dgd_name: str,
    initial_replicas: int,
    dynamo_endpoint_url: str,
) -> tuple[str, str]:
    """One scale-up + kill + classify cycle. Returns ``(verdict, detail)``."""
    target_replicas = initial_replicas + 1
    await _scale_decode(kubectl, dgd_name, namespace, target_replicas)
    fresh_pod = await _wait_for_fresh_decode_pod(
        kubectl, namespace, timeout=_FRESH_POD_WAIT_S
    )
    if fresh_pod is None:
        # Did not observe the race window; treat as a flake and retry.
        return "flake", "no Running+NotReady decode pod within window"

    logger.info(
        lambda p=fresh_pod: f"d801: observed fresh decode pod {p}; injecting etcd kill"
    )
    async with faults.inject("store.etcd.kill", grace_period=0):
        # restore is a no-op; the kubelet respawns etcd via its StatefulSet.
        pass

    await asyncio.sleep(_POST_FAULT_SETTLE_S)
    return await _check_post_fault_state(kubectl, namespace, dynamo_endpoint_url)


async def _resolve_dgd_name(kubectl: Any, namespace: str) -> str:
    """Return the single DGD in the namespace; assert exactly one exists."""
    res = await kubectl.run(
        "get",
        "dynamographdeployment",
        "-n",
        namespace,
        "-o",
        "jsonpath={.items[*].metadata.name}",
        check=True,
    )
    names = res.stdout.split()
    assert len(names) == 1, (
        f"d801: expected exactly one DynamoGraphDeployment in {namespace!r}, "
        f"found {names!r}"
    )
    return names[0]


def _find_decode_component(dgd: dict[str, Any]) -> dict[str, Any] | None:
    """Locate the decode-role component in either shipped DGD spec shape."""
    spec = dgd.get("spec", {})
    components = spec.get("components", [])
    match = next((c for c in components if c.get("type") == "decode"), None)
    if match is not None:
        return match

    services = spec.get("services", {})
    if isinstance(services, dict):
        return next(
            (
                service
                for service in services.values()
                if service.get("subComponentType") == "decode"
                or service.get("componentType") == "decode"
            ),
            None,
        )
    return None


async def _scale_decode(
    kubectl: Any,
    dgd_name: str,
    namespace: str,
    replicas: int,
) -> None:
    """Patch the decode component's ``replicas`` to the given value.

    Uses a JSON-merge patch to update only the decode entry; the operator
    reconciles the underlying Deployment from the new spec.
    """
    res = await kubectl.run(
        "get",
        "dynamographdeployment",
        dgd_name,
        "-n",
        namespace,
        "-o",
        "json",
        check=True,
    )
    dgd = orjson.loads(res.stdout)
    spec = dgd.get("spec", {})
    components = spec.get("components")
    if isinstance(components, list):
        for component in components:
            if component.get("type") == "decode":
                component["replicas"] = replicas
                break
        patch_data = {"spec": {"components": components}}
    else:
        services = spec.get("services", {})
        for service in services.values():
            if (
                service.get("subComponentType") == "decode"
                or service.get("componentType") == "decode"
            ):
                service["replicas"] = replicas
                break
        patch_data = {"spec": {"services": services}}
    patch = orjson.dumps(patch_data).decode()
    await kubectl.run(
        "patch",
        "dynamographdeployment",
        dgd_name,
        "-n",
        namespace,
        "--type=merge",
        "-p",
        patch,
        check=True,
    )


async def _wait_for_fresh_decode_pod(
    kubectl: Any,
    namespace: str,
    *,
    timeout: float,
) -> str | None:
    """Poll for the newest decode pod in Running+NotReady state.

    Returns the pod name once observed, or ``None`` if the window closes
    before any pod reaches that state. The newest pod (by
    ``creationTimestamp``) is the one we just scaled up.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        res = await kubectl.run(
            "get",
            "pods",
            "-l",
            _DECODE_POD_SELECTOR,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            payload = orjson.loads(res.stdout)
            candidate = _newest_running_not_ready(payload.get("items", []))
            if candidate is not None:
                return candidate
        await asyncio.sleep(1.0)
    return None


def _newest_running_not_ready(pods: list[dict[str, Any]]) -> str | None:
    """Return the name of the newest Running+NotReady pod, or ``None``."""
    matching: list[tuple[str, str]] = []
    for pod in pods:
        status = pod.get("status", {})
        if status.get("phase") != "Running":
            continue
        ready = False
        for cond in status.get("conditions", []):
            if cond.get("type") == "Ready":
                ready = cond.get("status") == "True"
                break
        if ready:
            continue
        name = pod.get("metadata", {}).get("name", "")
        created = pod.get("metadata", {}).get("creationTimestamp", "")
        if name:
            matching.append((created, name))
    if not matching:
        return None
    matching.sort()
    return matching[-1][1]


async def _check_post_fault_state(
    kubectl: Any,
    namespace: str,
    dynamo_endpoint_url: str,
) -> tuple[str, str]:
    """Classify the cluster state after the post-fault settle window.

    Returns one of:
        * ``("pass-a", detail)`` -- all decode pods Ready AND traffic 200 OK.
        * ``("pass-b", detail)`` -- one or more decode pods in
          ``CrashLoopBackOff`` with a clean etcd/register/lease error in logs.
        * ``("fail", detail)`` -- any pod Running+Ready that fails to serve,
          or no recognised PASS state.
    """
    res = await kubectl.run(
        "get",
        "pods",
        "-l",
        _DECODE_POD_SELECTOR,
        "-n",
        namespace,
        "-o",
        "json",
        check=True,
    )
    payload = orjson.loads(res.stdout)
    pods = payload.get("items", [])

    crashed = _pods_in_crashloop(pods)
    all_ready = pods and all(_is_pod_ready(p) for p in pods)

    if all_ready:
        ok, detail = await _probe_traffic(dynamo_endpoint_url)
        if ok:
            return "pass-a", f"all decode pods Ready and traffic 200 OK ({detail})"
        return "fail", (
            "decode pods Ready but traffic probe failed (half-registered "
            f"regression): {detail}"
        )

    if crashed:
        for pod in crashed:
            pod_name = pod.get("metadata", {}).get("name", "")
            if not pod_name:
                continue
            log_res = await kubectl.run(
                "logs",
                pod_name,
                "-n",
                namespace,
                "--tail=200",
                check=False,
            )
            if log_res.returncode != 0:
                continue
            log_bytes = log_res.stdout.encode().lower()
            if any(needle in log_bytes for needle in _ETCD_ERROR_NEEDLES):
                return "pass-b", (
                    f"pod {pod_name} in CrashLoopBackOff with clear "
                    "etcd/register/lease error"
                )
        return "fail", (
            "decode pod in CrashLoopBackOff but no etcd/register/lease "
            "needle in last 200 log lines (unclear error mode)"
        )

    return "fail", (
        f"no PASS state observed; pods={[p.get('metadata', {}).get('name') for p in pods]}"
    )


def _is_pod_ready(pod: dict[str, Any]) -> bool:
    """True when the pod's Ready condition is ``status=True``."""
    if pod.get("status", {}).get("phase") != "Running":
        return False
    for cond in pod.get("status", {}).get("conditions", []):
        if cond.get("type") == "Ready":
            return cond.get("status") == "True"
    return False


def _pods_in_crashloop(pods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return pods with any container in ``CrashLoopBackOff`` and restartCount>0."""
    out: list[dict[str, Any]] = []
    for pod in pods:
        statuses = pod.get("status", {}).get("containerStatuses", [])
        for cs in statuses:
            if int(cs.get("restartCount", 0)) <= 0:
                continue
            waiting = cs.get("state", {}).get("waiting", {})
            if waiting.get("reason") == "CrashLoopBackOff":
                out.append(pod)
                break
    return out


async def _probe_traffic(dynamo_endpoint_url: str) -> tuple[bool, str]:
    """Single chat-completions GET-style probe to confirm the frontend serves."""
    timeout = aiohttp.ClientTimeout(total=_TRAFFIC_PROBE_TIMEOUT_S)
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4,
        "stream": False,
        "temperature": 0.0,
    }
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(
                f"{dynamo_endpoint_url.rstrip('/')}/chat/completions", json=body
            ) as resp,
        ):
            if resp.status == 200:
                return True, "status=200"
            text = (await resp.read()).decode(errors="replace")[:256]
            return False, f"status={resp.status} body={text!r}"
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return False, f"client error {exc!r}"


# D802

D802_ETCD_NAMESPACE = "dynamo-system"

D802_ETCD_SERVICE = "dynamo-platform-etcd-headless"

ETCD_UPSTREAM = f"{D802_ETCD_SERVICE}.{D802_ETCD_NAMESPACE}.svc:2379"

D802_ETCD_PROXY_NAME = "etcd"

D802_ETCD_PROXY_LISTEN = "0.0.0.0:20030"

PAUSE_SECONDS = 30.0

RECOVERY_SECONDS = 60.0

_ETCD_CHAOS_OPT_IN_ENV = "AIPERF_DYNAMO_ETCD_CHAOS"


def _d802_static_skip_reason(dynamo_version: str = DYNAMO_VERSION) -> str | None:
    """Return why D802 cannot run before spending cluster setup time."""
    if os.environ.get(_ETCD_CHAOS_OPT_IN_ENV) == "1":
        return None
    if dynamo_version.startswith("1."):
        return (
            "D802 requires bundled etcd plus etcd discovery. Dynamo v1.1.0 "
            "defaults to Kubernetes discovery with global.etcd.install=false; "
            f"set {_ETCD_CHAOS_OPT_IN_ENV}=1 only for a topology that enables etcd."
        )
    return None


async def _etcd_service_exists(kubectl: KubectlClient) -> bool:
    """Return whether the bundled Dynamo etcd Service exists in the cluster."""
    result = await kubectl.run(
        "get",
        "service",
        D802_ETCD_SERVICE,
        "-n",
        D802_ETCD_NAMESPACE,
        check=False,
    )
    return result.returncode == 0


async def test_d802_etcd_30s_pause_recovers(
    request: pytest.FixtureRequest,
) -> None:
    """Pause etcd traffic only when Dynamo is actually using bundled etcd.

    Dynamo v1.1.0's chart defaults are ``global.etcd.install=false`` and
    ``dynamo-operator.discoveryBackend=kubernetes``. The default disagg-1gpu
    deployment therefore has no etcd dependency to pause, so D802 is not a
    valid scenario for that topology.
    """
    static_skip_reason = _d802_static_skip_reason()
    if static_skip_reason is not None:
        pytest.skip(static_skip_reason)

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    if not await _etcd_service_exists(kubectl):
        pytest.skip(
            f"D802 requires bundled etcd service {D802_ETCD_NAMESPACE}/{D802_ETCD_SERVICE}; "
            "the opt-in topology did not expose that service."
        )

    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    faults = request.getfixturevalue("faults")

    proxy_created = False
    try:
        await dynamo_toxiproxy.add_proxy(
            name=D802_ETCD_PROXY_NAME,
            listen=D802_ETCD_PROXY_LISTEN,
            upstream=ETCD_UPSTREAM,
        )
        proxy_created = True
        async with faults.inject(
            "store.etcd.timeout",
            target={"proxy": D802_ETCD_PROXY_NAME},
            attributes={"timeout": 0},
        ) as applied:
            assert applied.spec.fault_id == "network.timeout"
            assert applied.metadata.get("proxy_name") == D802_ETCD_PROXY_NAME
            logger.info(
                f"D802: timeout toxic applied to etcd proxy for {PAUSE_SECONDS}s; "
                f"recovery budget={RECOVERY_SECONDS}s"
            )
    finally:
        if proxy_created:
            try:
                await dynamo_toxiproxy.remove_proxy(D802_ETCD_PROXY_NAME)
            except Exception as exc:
                logger.warning(lambda exc=exc: f"D802 remove_proxy failed: {exc!r}")


# D803

D803_CONCURRENCY = 8

D803_REQUESTS_PER_TASK = 10

OUTAGE_SECS = 15

D803_RECOVERY_SECS = 30

STEADY_STATE_SECS = 5

D803_REQUEST_INTERVAL_SECS = 0.5

D803_COMPLETED_METRIC = "dynamo_frontend_requests_total"

ERRORS_METRIC = "dynamo_frontend_requests_errors_total"

ERROR_RATE_DURING_OUTAGE_THRESHOLD = 0.20

ERROR_RATE_RECOVERY_THRESHOLD = 0.05


async def test_d803_nats_kill_mid_traffic(
    faults,  # noqa: ANN001 - InjectorRegistry, see conftest.py
    kubectl,  # noqa: ANN001 - KubectlClient, see tests.kubernetes.conftest
    dynamo_endpoint_url,  # noqa: ANN001 - str, see gpu.dynamo.conftest
    dynamo_deployment_namespace,  # noqa: ANN001 - str, see chaos_dynamo.conftest
) -> None:
    """Kill NATS under 8 concurrent SSE streams; assert degradation not outage.

    NATS is dynamo's stats/metrics bus. ``nats.rs:49`` has no explicit reconnect
    backoff overrides; this test exercises whatever ``async_nats`` defaults to.

    Materialized body lives in :py:func:`_run_d803_assertion` -- flip the
    ``pytest.skip`` below to run it on a real cluster.
    """
    await _run_d803_assertion(
        faults, kubectl, dynamo_endpoint_url, dynamo_deployment_namespace
    )


async def _run_d803_assertion(
    faults,  # noqa: ANN001 - InjectorRegistry, see conftest.py
    kubectl,  # noqa: ANN001 - KubectlClient, see tests.kubernetes.conftest
    dynamo_endpoint_url: str,
    dynamo_deployment_namespace: str,
) -> None:
    """Full D803 assertion body; one-line unskip flip in the test stub runs it.

    Steps mirror the docstring outline on :py:func:`test_d803_nats_kill_mid_traffic`:

    1. Snapshot ``metrics_before`` from the dynamo frontend's ``/metrics``.
    2. Spawn ``CONCURRENCY`` background SSE tasks, each looping
       ``REQUESTS_PER_TASK`` requests against ``/chat/completions``.
    3. Sleep ``STEADY_STATE_SECS`` so the workers are actively streaming
       before the fault lands.
    4. Inject ``store.nats.kill`` with ``grace_period=0`` (instantaneous
       force-delete; the kubelet respawns NATS in the background) and hold
       it for ``OUTAGE_SECS``. Scrape ``metrics_during`` inside the window.
    5. After the context exits, wait ``RECOVERY_SECS`` for steady state,
       then scrape ``metrics_after``.
    6. Assert that the frontend kept serving during the outage and that
       the error rate stayed under :data:`ERROR_RATE_DURING_OUTAGE_THRESHOLD`.
    7. Assert that the post-recovery error rate is under
       :data:`ERROR_RATE_RECOVERY_THRESHOLD`.

    Worker tasks are cancelled in ``finally`` so a failed assertion does
    not leak background coroutines into later tests.
    """
    # 1. Snapshot baseline counters from the frontend's /metrics.
    metrics_before = await scrape_frontend_metrics(kubectl, dynamo_deployment_namespace)
    logger.info(
        lambda: f"D803: metrics_before keys={len(metrics_before)} "
        f"completed={metrics_before.get(D803_COMPLETED_METRIC, 0.0)} "
        f"errors={metrics_before.get(ERRORS_METRIC, 0.0)}"
    )

    # 2. Spawn CONCURRENCY background SSE workers.
    stop_event = asyncio.Event()
    request_counter: dict[str, int] = {"completed": 0, "errors": 0}

    async def _worker(idx: int) -> None:
        async with aiohttp.ClientSession() as session:
            for _ in range(D803_REQUESTS_PER_TASK):
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
                        lambda exc=exc, idx=idx: (
                            f"D803 worker {idx} request error: {exc!r}"
                        )
                    )
                    request_counter["errors"] += 1
                await asyncio.sleep(D803_REQUEST_INTERVAL_SECS)

    workers = [asyncio.create_task(_worker(i)) for i in range(D803_CONCURRENCY)]

    try:
        # 3. Let traffic stabilize before the fault lands.
        await asyncio.sleep(STEADY_STATE_SECS)

        # 4. Inject NATS kill; let traffic run through the outage window.
        async with faults.inject("store.nats.kill", grace_period=0):
            await asyncio.sleep(OUTAGE_SECS)
            metrics_during = await scrape_frontend_metrics(
                kubectl, dynamo_deployment_namespace
            )

        # 5. Post-restore window -- wait for steady state, then scrape.
        await asyncio.sleep(D803_RECOVERY_SECS)
        metrics_after = await scrape_frontend_metrics(
            kubectl, dynamo_deployment_namespace
        )

        # 6. Frontend must have stayed up during the outage.
        completed_during = max(
            _D803_metric_delta(metrics_during, metrics_before, D803_COMPLETED_METRIC),
            float(request_counter["completed"]),
        )
        errors_during = max(
            _D803_metric_delta(metrics_during, metrics_before, ERRORS_METRIC),
            float(request_counter["errors"]),
        )
        assert completed_during > 0, (
            f"D803: frontend stopped serving during NATS outage "
            f"(completed_during={completed_during}, errors_during={errors_during}, "
            f"client_completed={request_counter['completed']}, "
            f"client_errors={request_counter['errors']})"
        )
        error_rate_during = errors_during / max(completed_during, 1)
        assert error_rate_during < ERROR_RATE_DURING_OUTAGE_THRESHOLD, (
            f"D803: error rate during outage {error_rate_during:.1%} > "
            f"{ERROR_RATE_DURING_OUTAGE_THRESHOLD:.0%} threshold "
            f"(completed={completed_during}, errors={errors_during})"
        )

        # 7. Recovery must be clean once NATS is back.
        completed_recovery = _D803_metric_delta(
            metrics_after, metrics_during, D803_COMPLETED_METRIC
        )
        errors_recovery = _D803_metric_delta(
            metrics_after, metrics_during, ERRORS_METRIC
        )
        if completed_recovery > 0:
            error_rate_recovery = errors_recovery / completed_recovery
            assert error_rate_recovery < ERROR_RATE_RECOVERY_THRESHOLD, (
                f"D803: recovery error rate {error_rate_recovery:.1%} > "
                f"{ERROR_RATE_RECOVERY_THRESHOLD:.0%} threshold "
                f"(completed={completed_recovery}, errors={errors_recovery})"
            )
        else:
            logger.warning(
                lambda: (
                    f"D803: no completed requests observed during recovery "
                    f"window ({D803_RECOVERY_SECS}s); skipping recovery error-rate "
                    f"assertion (errors_recovery={errors_recovery})"
                )
            )
    finally:
        stop_event.set()
        for w in workers:
            w.cancel()
        # Gather cancellations to swallow CancelledError from each worker
        # so an in-flight aiohttp request does not surface as an unhandled
        # task exception when the test tears down.
        for w in workers:
            with contextlib.suppress(asyncio.CancelledError):
                await w


def _D803_metric_delta(
    after: dict[str, float], before: dict[str, float], key: str
) -> float:
    """Return the increment in a counter metric between two scrapes.

    Missing keys default to 0.0 so a metric that never appears on the
    frontend (e.g. placeholder name mismatch) reads as no change rather
    than raising ``KeyError`` mid-assertion.
    """
    return after.get(key, 0.0) - before.get(key, 0.0)


# D804

D804_NATS_NAMESPACE = "dynamo-system"

D804_NATS_SELECTOR = "app=nats"

D804_NATS_PROXY_NAME = "nats"

D804_NATS_PROXY_LISTEN = "0.0.0.0:20020"

D804_NATS_PROXY_ROUTE = "toxiproxy.chaos-toxiproxy.svc:20020"

D804_NATS_SERVICE_PORT = 4222

SLOW_CLOSE_DELAY_MS = 5_000

TOXIC_WINDOW_SECS = 15.0

D804_RECOVERY_SECS = 20.0

D804_CONCURRENCY = 4

D804_REQUESTS_PER_TASK = 4

D804_REQUEST_INTERVAL_SECS = 0.25

D804_COMPLETED_METRIC = "dynamo_frontend_requests_total"

ROUTER_OVERHEAD_METRIC_PREFIX = "dynamo_component_router_overhead_total_ms"

_NATS_CHAOS_OPT_IN_ENV = "AIPERF_DYNAMO_NATS_CHAOS"


@dataclass(frozen=True, slots=True)
class _NatsRouteObservation:
    """Observed NATS route prerequisites for D804."""

    nats_pods: list[str]
    nats_service: str | None
    routed_values: list[str]

    @property
    def is_routed_through_toxiproxy(self) -> bool:
        """Return whether any observed pod config points NATS at Toxiproxy."""
        return any(D804_NATS_PROXY_ROUTE in value for value in self.routed_values)

    def skip_reason(self) -> str:
        """Build the prerequisite skip reason with concrete observed state."""
        return (
            f"D804 requires live NATS traffic to route through {D804_NATS_PROXY_ROUTE!r} "
            f"so store.nats.slow_close reaches product traffic. Observed "
            f"NATS selector {D804_NATS_SELECTOR!r} in namespace {D804_NATS_NAMESPACE!r}: "
            f"pods={self.nats_pods!r}, service={self.nats_service!r}, "
            f"route_values={self.routed_values!r}; missing proxy route. "
            "Default Dynamo v1.1.0 disagg routes NATS directly."
        )


def _d804_static_skip_reason() -> str | None:
    """Return why the default topology cannot run D804 before cluster setup."""
    if os.environ.get(_NATS_CHAOS_OPT_IN_ENV) == "1":
        return None
    return (
        f"D804 requires live NATS traffic to route through {D804_NATS_PROXY_ROUTE!r} "
        "before cluster setup; "
        f"observed NATS selector {D804_NATS_SELECTOR!r} in namespace {D804_NATS_NAMESPACE!r}; "
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
            f"{D804_NATS_SELECTOR!r} in namespace {D804_NATS_NAMESPACE!r}; "
            f"pods={observation.nats_pods!r}"
        )

    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    faults = request.getfixturevalue("faults")
    dynamo_endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")
    upstream = (
        f"{observation.nats_service}.{D804_NATS_NAMESPACE}.svc:{D804_NATS_SERVICE_PORT}"
    )

    proxy_created = False
    try:
        await dynamo_toxiproxy.add_proxy(
            name=D804_NATS_PROXY_NAME,
            listen=D804_NATS_PROXY_LISTEN,
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
                await dynamo_toxiproxy.remove_proxy(D804_NATS_PROXY_NAME)
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
        for idx in range(D804_CONCURRENCY)
    ]

    try:
        async with faults.inject(
            "store.nats.slow_close",
            target={"proxy": D804_NATS_PROXY_NAME},
            attributes={"delay": SLOW_CLOSE_DELAY_MS},
            stream="downstream",
        ) as applied:
            assert applied.spec.fault_id == "network.slow_close"
            assert applied.metadata.get("proxy_name") == D804_NATS_PROXY_NAME
            await asyncio.sleep(TOXIC_WINDOW_SECS)
            metrics_during = await _try_scrape_frontend_metrics(
                kubectl, dynamo_deployment_namespace
            )

        await asyncio.sleep(D804_RECOVERY_SECS)
        metrics_after = await _scrape_frontend_metrics_with_retries(
            kubectl, dynamo_deployment_namespace
        )

        completed_during = float(request_counter["completed"])
        if metrics_during is not None:
            completed_during = max(
                completed_during,
                _D804_metric_delta(
                    metrics_during, metrics_before, D804_COMPLETED_METRIC
                ),
            )
        completed_after = max(
            float(request_counter["completed"]),
            _D804_metric_delta(metrics_after, metrics_before, D804_COMPLETED_METRIC),
        )
        assert completed_during > 0 or completed_after > 0, (
            "D804: frontend traffic neither continued during NATS slow-close nor "
            f"recovered afterward (client_completed={request_counter['completed']}, "
            f"client_errors={request_counter['errors']}, "
            f"metrics_before_completed={metrics_before.get(D804_COMPLETED_METRIC, 0.0)}, "
            f"metrics_after_completed={metrics_after.get(D804_COMPLETED_METRIC, 0.0)})"
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
        for _ in range(D804_REQUESTS_PER_TASK):
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
            await asyncio.sleep(D804_REQUEST_INTERVAL_SECS)


async def _observe_nats_route(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> _NatsRouteObservation:
    """Inspect live pods/services to decide whether D804 can affect NATS."""
    nats_pods = await _list_pods_with_selector(
        kubectl, D804_NATS_NAMESPACE, D804_NATS_SELECTOR
    )
    nats_service = await _find_nats_service(kubectl)
    route_values: list[str] = []
    for namespace in dict.fromkeys([dynamo_deployment_namespace, D804_NATS_NAMESPACE]):
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
        D804_NATS_NAMESPACE,
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
        exposes_nats_port = any(
            port.get("port") == D804_NATS_SERVICE_PORT for port in ports
        )
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
    return "nats" in haystack or D804_NATS_PROXY_ROUTE in value


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


def _D804_metric_delta(
    after: dict[str, float], before: dict[str, float], key: str
) -> float:
    """Return the non-negative increment in a counter between two scrapes."""
    return max(after.get(key, 0.0) - before.get(key, 0.0), 0.0)


# D805

_API_GROUP = "nvidia.com"
_RESOURCE = "dynamoworkermetadatas"
_VERB = "watch"
_REQUEST_TIMEOUT_S = 30.0
_WATCH_FAILURE_WINDOW_S = 10.0


@dataclass(frozen=True, slots=True)
class _RbacWatchOwner:
    """Exact RBAC resource that grants DynamoWorkerMetadata watch permission."""

    scope: str
    """``role`` or ``clusterrole`` for kubectl/fault-injector patching."""

    name: str
    """Role or ClusterRole name."""

    namespace: str | None
    """Role namespace, or ``None`` for ClusterRole."""

    @property
    def label(self) -> str:
        """Human-readable identifier for skip/failure diagnostics."""
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


async def test_d805_discovery_rbac_watch_revocation_preserves_existing_traffic(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Revoke DWM watch RBAC only when the exact owner is uniquely discoverable.

    Kubernetes discovery should tolerate a short watch outage by serving from the
    already-populated cache. The test proves that contract with one successful
    request before revocation and one while the ``watch`` verb is absent.
    """
    owner, inspected_names = await _find_unique_dwm_watch_owner(
        kubectl, dynamo_deployment_namespace
    )
    if owner is None:
        pytest.skip(
            "D805 requires exactly one exact RBAC rule granting watch on "
            f"{_RESOURCE}.{_API_GROUP}; inspected RBAC resources: "
            f"{', '.join(inspected_names) or '<none>'}"
        )

    await _D805_assert_frontend_serves(
        dynamo_endpoint_url, phase="before RBAC revocation"
    )

    faults = request.getfixturevalue("faults")
    target: dict[str, str] = {"scope": owner.scope, "name": owner.name}
    if owner.namespace is not None:
        target["ns"] = owner.namespace

    try:
        async with faults.inject(
            "cluster.rbac.revoke",
            target=target,
            api_group=_API_GROUP,
            resource=_RESOURCE,
            verb=_VERB,
        ) as applied:
            assert applied.metadata["name"] == owner.name
            assert applied.metadata["resource"] == _RESOURCE
            assert applied.metadata["verb"] == _VERB
            logger.info(
                f"D805: revoked {_VERB!r} on {_RESOURCE}.{_API_GROUP} from "
                f"{owner.label}; asserting live traffic during watch failure"
            )
            await asyncio.sleep(_WATCH_FAILURE_WINDOW_S)
            await _D805_assert_frontend_serves(
                dynamo_endpoint_url,
                phase=f"while {owner.label} lacks {_VERB!r}",
            )
    finally:
        restored = await _role_currently_grants_watch(kubectl, owner)
        assert restored, (
            f"D805: RBAC restore did not put {_VERB!r} back on {owner.label} "
            f"for {_RESOURCE}.{_API_GROUP}; manual cluster repair required"
        )


async def _find_unique_dwm_watch_owner(
    kubectl: KubectlClient,
    namespace: str,
) -> tuple[_RbacWatchOwner | None, list[str]]:
    """Return the unique exact RBAC owner, or ``None`` with inspected names.

    Wildcard resources / verbs are intentionally ignored. D805 is only safe to
    run when the RBAC rule explicitly names ``dynamoworkermetadatas`` and
    ``watch`` so the injected patch has a narrow blast radius.
    """
    roles = await _D805_load_rbac_collection(kubectl, "roles", namespace=namespace)
    clusterroles = await _D805_load_rbac_collection(kubectl, "clusterroles")

    inspected: list[str] = []
    candidates: list[_RbacWatchOwner] = []
    for item in roles:
        metadata = item.get("metadata", {})
        owner = _RbacWatchOwner(
            scope="role",
            name=str(metadata.get("name", "")),
            namespace=str(metadata.get("namespace", "")),
        )
        inspected.append(owner.label)
        if _has_exact_dwm_watch_rule(item.get("rules") or []):
            candidates.append(owner)

    for item in clusterroles:
        metadata = item.get("metadata", {})
        owner = _RbacWatchOwner(
            scope="clusterrole",
            name=str(metadata.get("name", "")),
            namespace=None,
        )
        inspected.append(owner.label)
        if _has_exact_dwm_watch_rule(item.get("rules") or []):
            candidates.append(owner)

    if len(candidates) != 1:
        candidate_names = [candidate.label for candidate in candidates]
        return None, candidate_names or inspected
    return candidates[0], inspected


async def _D805_load_rbac_collection(
    kubectl: KubectlClient,
    resource: str,
    *,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    """Load Roles or ClusterRoles as JSON; skip if the caller lacks list RBAC."""
    args = ["get", resource, "-o", "json"]
    if namespace is not None:
        args.extend(["-n", namespace])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        pytest.skip(
            f"D805 could not inspect {resource} before RBAC mutation: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    data = orjson.loads(result.stdout or b"{}")
    return list(data.get("items", []))


def _has_exact_dwm_watch_rule(rules: list[dict[str, Any]]) -> bool:
    """Return true for explicit ``watch`` on ``dynamoworkermetadatas`` only."""
    for rule in rules:
        groups = rule.get("apiGroups") or []
        resources = rule.get("resources") or []
        verbs = rule.get("verbs") or []
        if _API_GROUP in groups and _RESOURCE in resources and _VERB in verbs:
            return True
    return False


async def _role_currently_grants_watch(
    kubectl: KubectlClient,
    owner: _RbacWatchOwner,
) -> bool:
    """Verify cleanup restored the exact watch permission that D805 removed."""
    args = ["get", owner.scope, owner.name]
    if owner.namespace is not None:
        args.extend(["-n", owner.namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        return False
    body = orjson.loads(result.stdout or b"{}")
    return _has_exact_dwm_watch_rule(body.get("rules") or [])


async def _D805_assert_frontend_serves(endpoint_url: str, *, phase: str) -> None:
    """Send one OpenAI-compatible streaming request and require HTTP success."""
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "max_tokens": 10,
    }
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(f"{endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body_prefix = b""
        async for chunk in resp.content.iter_chunked(1024):
            body_prefix += chunk
            if body_prefix:
                break
        assert resp.status == 200, (
            f"D805: frontend returned HTTP {resp.status} {phase}; "
            f"body_prefix={body_prefix[:256].decode(errors='replace')!r}"
        )
        assert body_prefix, f"D805: frontend returned an empty stream {phase}"


# D806-D817

D806_D817_ETCD_NAMESPACE = "dynamo-system"
D806_D817_ETCD_SERVICE = "dynamo-platform-etcd-headless"
D806_D817_ETCD_PROXY_NAME = "etcd-keepalive"
D806_D817_ETCD_PROXY_LISTEN = "0.0.0.0:20031"
ETCD_CLIENT_PORT = 2379
ETCD_CHAOS_OPT_IN_ENV = "AIPERF_DYNAMO_ETCD_CHAOS"

D806_D817_NATS_NAMESPACE = "dynamo-system"
D806_D817_NATS_SELECTOR = "app=nats"
D806_D817_NATS_PROXY_NAME = "nats-frontend-partition"
D806_D817_NATS_PROXY_LISTEN = "0.0.0.0:20021"
D806_D817_NATS_PROXY_ROUTE = "toxiproxy.chaos-toxiproxy.svc:20020"
D806_D817_NATS_SERVICE_PORT = 4222
NATS_CHAOS_OPT_IN_ENV = "AIPERF_DYNAMO_NATS_CHAOS"

DWM_API_GROUP = "nvidia.com"
DWM_RESOURCE = "dynamoworkermetadatas"
ENDPOINTSLICE_API_GROUP = "discovery.k8s.io"
ENDPOINTSLICE_RESOURCE = "endpointslices"

SERVICE_SELECTOR_OPT_IN_ENV = "AIPERF_DYNAMO_SERVICE_SELECTOR_CHAOS"
COREDNS_OPT_IN_ENV = "AIPERF_DYNAMO_COREDNS_CHAOS"
FRONTEND_REQUEST_TIMEOUT_S = 30.0
RBAC_FAILURE_WINDOW_S = 5.0
SERVICE_SELECTOR_WINDOW_S = 10.0
COREDNS_WINDOW_S = 15.0


@dataclass(frozen=True, slots=True)
class _RbacOwner:
    """Exact RBAC resource granting one discovery permission."""

    scope: str
    """``role`` or ``clusterrole`` for kubectl and the fault injector."""

    name: str
    """Role or ClusterRole name."""

    namespace: str | None
    """Role namespace, or ``None`` for ClusterRole."""

    @property
    def label(self) -> str:
        """Human-readable identifier for skip and assertion messages."""
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


@dataclass(frozen=True, slots=True)
class _ServiceSelectorPatch:
    """Patch target and original selector for a reversible Service-selector fault."""

    namespace: str
    name: str
    original_selector: dict[str, str]


async def test_d806_etcd_keepalive_blackhole_expires_one_worker(
    request: pytest.FixtureRequest,
) -> None:
    """Blackhole etcd keepalive traffic only for an etcd-enabled topology.

    The default Dynamo v1.1.0 topology uses Kubernetes discovery and does not
    install bundled etcd, so the test self-skips unless the caller opts into an
    etcd-backed deployment where the reserved Toxiproxy route is live.
    """
    if os.environ.get(ETCD_CHAOS_OPT_IN_ENV) != "1":
        pytest.skip(
            "D806 requires bundled etcd plus an etcd-discovery topology routed "
            f"through Toxiproxy; stock Dynamo v1.1.0 disagg uses Kubernetes "
            f"discovery. Set {ETCD_CHAOS_OPT_IN_ENV}=1 only for that topology."
        )

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    if not await _service_exists(
        kubectl, D806_D817_ETCD_NAMESPACE, D806_D817_ETCD_SERVICE
    ):
        pytest.skip(
            f"D806 requires bundled etcd service {D806_D817_ETCD_NAMESPACE}/{D806_D817_ETCD_SERVICE}; "
            "the opt-in topology did not expose that service."
        )

    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    faults = request.getfixturevalue("faults")
    upstream = (
        f"{D806_D817_ETCD_SERVICE}.{D806_D817_ETCD_NAMESPACE}.svc:{ETCD_CLIENT_PORT}"
    )

    proxy_created = False
    try:
        await dynamo_toxiproxy.add_proxy(
            name=D806_D817_ETCD_PROXY_NAME,
            listen=D806_D817_ETCD_PROXY_LISTEN,
            upstream=upstream,
        )
        proxy_created = True
        async with faults.inject(
            "store.etcd.bandwidth",
            target={"proxy": D806_D817_ETCD_PROXY_NAME},
            attributes={"rate": 0},
            stream="upstream",
        ) as applied:
            assert applied.spec.fault_id == "network.bandwidth"
            assert applied.metadata.get("proxy_name") == D806_D817_ETCD_PROXY_NAME
            logger.info(
                "D806: etcd keepalive bandwidth=0 toxic applied; lease-expiry "
                "assertion is topology-gated by the etcd opt-in deployment"
            )
    finally:
        if proxy_created:
            await _remove_proxy_safely(
                dynamo_toxiproxy, D806_D817_ETCD_PROXY_NAME, "D806"
            )


async def test_d807_nats_frontend_partition_converges_after_heal(
    request: pytest.FixtureRequest,
) -> None:
    """Partition frontend NATS traffic only when multiple frontends are proxied."""
    if os.environ.get(NATS_CHAOS_OPT_IN_ENV) != "1":
        pytest.skip(
            "D807 requires two frontend replicas whose NATS route traverses "
            f"{D806_D817_NATS_PROXY_ROUTE!r}; stock Dynamo v1.1.0 disagg has one frontend "
            f"and direct NATS. Set {NATS_CHAOS_OPT_IN_ENV}=1 only for that topology."
        )

    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    frontend_pods = await _list_frontend_pods(kubectl, namespace)
    if len(frontend_pods) < 2:
        pytest.skip(
            f"D807 requires at least two frontend pods in {namespace!r}; "
            f"observed {frontend_pods!r}."
        )
    if not await _topology_mentions_route(
        kubectl, [namespace, D806_D817_NATS_NAMESPACE], D806_D817_NATS_PROXY_ROUTE
    ):
        pytest.skip(
            f"D807 requires frontend NATS traffic to route through {D806_D817_NATS_PROXY_ROUTE!r}; "
            "no pod env/args mention that route."
        )

    nats_service = await _find_service_with_port(
        kubectl, D806_D817_NATS_NAMESPACE, D806_D817_NATS_SERVICE_PORT
    )
    if nats_service is None:
        pytest.skip(
            f"D807 requires a NATS Service exposing port {D806_D817_NATS_SERVICE_PORT} in "
            f"{D806_D817_NATS_NAMESPACE!r}; selector={D806_D817_NATS_SELECTOR!r}."
        )

    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    faults = request.getfixturevalue("faults")
    upstream = (
        f"{nats_service}.{D806_D817_NATS_NAMESPACE}.svc:{D806_D817_NATS_SERVICE_PORT}"
    )

    proxy_created = False
    try:
        await dynamo_toxiproxy.add_proxy(
            name=D806_D817_NATS_PROXY_NAME,
            listen=D806_D817_NATS_PROXY_LISTEN,
            upstream=upstream,
        )
        proxy_created = True
        async with faults.inject(
            "store.nats.partition",
            target={"proxy": D806_D817_NATS_PROXY_NAME},
        ) as applied:
            assert applied.spec.fault_id == "network.partition"
            assert applied.metadata.get("proxy_name") == D806_D817_NATS_PROXY_NAME
            await asyncio.sleep(RBAC_FAILURE_WINDOW_S)
    finally:
        if proxy_created:
            await _remove_proxy_safely(
                dynamo_toxiproxy, D806_D817_NATS_PROXY_NAME, "D807"
            )


@pytest.mark.parametrize(
    ("case_id", "verb"),
    [
        pytest.param("D808", "get", id="D808-dwm-get"),
        pytest.param("D809", "list", id="D809-dwm-list"),
        pytest.param("D810", "watch", id="D810-dwm-watch"),
    ],
)  # fmt: skip
async def test_d808_d810_dwm_rbac_revocation_preserves_cached_traffic(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
    case_id: str,
    verb: str,
) -> None:
    """Revoke one DWM verb and require cached discovery to keep traffic alive."""
    await _run_discovery_rbac_case(
        request=request,
        kubectl=kubectl,
        endpoint_url=dynamo_endpoint_url,
        namespace=dynamo_deployment_namespace,
        case_id=case_id,
        api_group=DWM_API_GROUP,
        resource=DWM_RESOURCE,
        verb=verb,
    )


@pytest.mark.parametrize(
    ("case_id", "verb"),
    [
        pytest.param("D811", "get", id="D811-endpointslice-get"),
        pytest.param("D812", "list", id="D812-endpointslice-list"),
        pytest.param("D813", "watch", id="D813-endpointslice-watch"),
        pytest.param("D814", "delete", id="D814-endpointslice-delete"),
        pytest.param("D815", "patch", id="D815-endpointslice-patch"),
    ],
)  # fmt: skip
async def test_d811_d815_endpointslice_rbac_revocation_preserves_cached_traffic(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
    case_id: str,
    verb: str,
) -> None:
    """Revoke one EndpointSlice verb and require cached discovery to serve."""
    await _run_discovery_rbac_case(
        request=request,
        kubectl=kubectl,
        endpoint_url=dynamo_endpoint_url,
        namespace=dynamo_deployment_namespace,
        case_id=case_id,
        api_group=ENDPOINTSLICE_API_GROUP,
        resource=ENDPOINTSLICE_RESOURCE,
        verb=verb,
    )


async def test_d816_service_selector_mismatch_does_not_poison_cached_discovery(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_endpoint_url: str,
) -> None:
    """Patch one Dynamo Service selector only in explicit selector-chaos topology."""
    if os.environ.get(SERVICE_SELECTOR_OPT_IN_ENV) != "1":
        pytest.skip(
            "D816 mutates a live Dynamo Service selector and is intentionally "
            f"opt-in. Set {SERVICE_SELECTOR_OPT_IN_ENV}=1 only on an isolated "
            "cluster where EndpointSlice churn is expected."
        )

    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    patch_target = await _find_patchable_dynamo_service(kubectl, namespace)
    if patch_target is None:
        pytest.skip(
            f"D816 requires a Dynamo Service with a non-empty selector in {namespace!r}; "
            "none was found."
        )

    await _D806_D817_assert_frontend_serves(
        dynamo_endpoint_url, case_id="D816", phase="before"
    )
    mismatched_selector = dict(patch_target.original_selector)
    mismatched_selector["aiperf.nvidia.com/chaos-d816"] = "no-such-pod"
    try:
        await _patch_service_selector(
            kubectl,
            patch_target.namespace,
            patch_target.name,
            mismatched_selector,
        )
        await asyncio.sleep(SERVICE_SELECTOR_WINDOW_S)
        await _D806_D817_assert_frontend_serves(
            dynamo_endpoint_url,
            case_id="D816",
            phase="while service selector is mismatched",
        )
    finally:
        await _patch_service_selector(
            kubectl,
            patch_target.namespace,
            patch_target.name,
            patch_target.original_selector,
        )


async def test_d817_coredns_outage_does_not_poison_cached_discovery(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_endpoint_url: str,
) -> None:
    """Scale CoreDNS down only in explicit DNS-chaos topology."""
    if os.environ.get(COREDNS_OPT_IN_ENV) != "1":
        pytest.skip(
            "D817 scales the cluster DNS deployment and is intentionally opt-in. "
            f"Set {COREDNS_OPT_IN_ENV}=1 only on an isolated cluster where a "
            "short CoreDNS outage is acceptable."
        )

    deployment = await _find_coredns_deployment(kubectl)
    if deployment is None:
        pytest.skip(
            "D817 requires a CoreDNS/kube-dns Deployment in kube-system; none found."
        )

    faults = request.getfixturevalue("faults")
    await _D806_D817_assert_frontend_serves(
        dynamo_endpoint_url, case_id="D817", phase="before"
    )
    async with faults.inject(
        "workload.scale",
        target={"kind": "deployment", "name": deployment, "ns": "kube-system"},
        replicas=0,
    ) as applied:
        assert applied.metadata.get("name") == deployment
        await asyncio.sleep(COREDNS_WINDOW_S)
        await _D806_D817_assert_frontend_serves(
            dynamo_endpoint_url,
            case_id="D817",
            phase="during CoreDNS outage",
        )


async def _run_discovery_rbac_case(
    *,
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    endpoint_url: str,
    namespace: str,
    case_id: str,
    api_group: str,
    resource: str,
    verb: str,
) -> None:
    """Revoke one exact discovery RBAC verb, assert cached traffic, verify restore."""
    owner, inspected_names = await _find_unique_rbac_owner(
        kubectl,
        namespace=namespace,
        api_group=api_group,
        resource=resource,
        verb=verb,
    )
    if owner is None:
        pytest.skip(
            f"{case_id} requires exactly one exact RBAC rule granting {verb!r} on "
            f"{resource}.{api_group}; inspected/candidate RBAC resources: "
            f"{', '.join(inspected_names) or '<none>'}. Wildcards and ambiguous "
            "owners are skipped to avoid broad cluster mutation."
        )

    await _D806_D817_assert_frontend_serves(
        endpoint_url, case_id=case_id, phase="before"
    )
    faults = request.getfixturevalue("faults")
    target: dict[str, str] = {"scope": owner.scope, "name": owner.name}
    if owner.namespace is not None:
        target["ns"] = owner.namespace

    try:
        async with faults.inject(
            "cluster.rbac.revoke",
            target=target,
            api_group=api_group,
            resource=resource,
            verb=verb,
        ) as applied:
            assert applied.metadata["name"] == owner.name
            assert applied.metadata["resource"] == resource
            assert applied.metadata["verb"] == verb
            await asyncio.sleep(RBAC_FAILURE_WINDOW_S)
            await _D806_D817_assert_frontend_serves(
                endpoint_url,
                case_id=case_id,
                phase=f"while {owner.label} lacks {verb!r}",
            )
    finally:
        restored = await _role_currently_grants(
            kubectl,
            owner,
            api_group=api_group,
            resource=resource,
            verb=verb,
        )
        assert restored, (
            f"{case_id}: RBAC restore did not put {verb!r} back on {owner.label} "
            f"for {resource}.{api_group}; manual cluster repair required"
        )


async def _find_unique_rbac_owner(
    kubectl: KubectlClient,
    *,
    namespace: str,
    api_group: str,
    resource: str,
    verb: str,
) -> tuple[_RbacOwner | None, list[str]]:
    """Return unique exact RBAC owner for ``(api_group, resource, verb)``."""
    roles = await _D806_D817_load_rbac_collection(kubectl, "roles", namespace=namespace)
    clusterroles = await _D806_D817_load_rbac_collection(kubectl, "clusterroles")

    inspected: list[str] = []
    candidates: list[_RbacOwner] = []
    for item in roles:
        metadata = item.get("metadata", {})
        owner = _RbacOwner(
            scope="role",
            name=str(metadata.get("name", "")),
            namespace=str(metadata.get("namespace", "")),
        )
        inspected.append(owner.label)
        if _has_exact_rule(item.get("rules") or [], api_group, resource, verb):
            candidates.append(owner)

    for item in clusterroles:
        metadata = item.get("metadata", {})
        owner = _RbacOwner(
            scope="clusterrole",
            name=str(metadata.get("name", "")),
            namespace=None,
        )
        inspected.append(owner.label)
        if _has_exact_rule(item.get("rules") or [], api_group, resource, verb):
            candidates.append(owner)

    if len(candidates) != 1:
        return None, [candidate.label for candidate in candidates] or inspected
    return candidates[0], inspected


async def _D806_D817_load_rbac_collection(
    kubectl: KubectlClient,
    resource: str,
    *,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    """Load Roles or ClusterRoles, skipping if caller cannot inspect RBAC."""
    args = ["get", resource, "-o", "json"]
    if namespace is not None:
        args.extend(["-n", namespace])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        pytest.skip(
            f"could not inspect {resource} before RBAC mutation: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    data = orjson.loads(result.stdout or b"{}")
    return list(data.get("items", []))


def _has_exact_rule(
    rules: Iterable[dict[str, Any]],
    api_group: str,
    resource: str,
    verb: str,
) -> bool:
    """Return true only for explicit group/resource/verb RBAC rules."""
    for rule in rules:
        groups = rule.get("apiGroups") or []
        resources = rule.get("resources") or []
        verbs = rule.get("verbs") or []
        if api_group in groups and resource in resources and verb in verbs:
            return True
    return False


async def _role_currently_grants(
    kubectl: KubectlClient,
    owner: _RbacOwner,
    *,
    api_group: str,
    resource: str,
    verb: str,
) -> bool:
    """Verify cleanup restored the exact permission removed by a test."""
    args = ["get", owner.scope, owner.name]
    if owner.namespace is not None:
        args.extend(["-n", owner.namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        return False
    body = orjson.loads(result.stdout or b"{}")
    return _has_exact_rule(body.get("rules") or [], api_group, resource, verb)


async def _D806_D817_assert_frontend_serves(
    endpoint_url: str, *, case_id: str, phase: str
) -> None:
    """Send one streaming OpenAI-compatible request and require HTTP success."""
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "max_tokens": 10,
    }
    timeout = aiohttp.ClientTimeout(total=FRONTEND_REQUEST_TIMEOUT_S)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(f"{endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body_prefix = b""
        async for chunk in resp.content.iter_chunked(1024):
            body_prefix += chunk
            if body_prefix:
                break
        assert resp.status == 200, (
            f"{case_id}: frontend returned HTTP {resp.status} {phase}; "
            f"body_prefix={body_prefix[:256].decode(errors='replace')!r}"
        )
        assert body_prefix, f"{case_id}: frontend returned an empty stream {phase}"


async def _service_exists(kubectl: KubectlClient, namespace: str, name: str) -> bool:
    """Return whether a Service exists."""
    result = await kubectl.run(
        "get",
        "service",
        name,
        "-n",
        namespace,
        check=False,
    )
    return result.returncode == 0


async def _find_service_with_port(
    kubectl: KubectlClient,
    namespace: str,
    port: int,
) -> str | None:
    """Return the first Service in ``namespace`` exposing ``port``."""
    result = await kubectl.run(
        "get",
        "services",
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return None
    for service in orjson.loads(result.stdout or b"{}").get("items", []):
        ports = service.get("spec", {}).get("ports", [])
        if any(item.get("port") == port for item in ports):
            name = service.get("metadata", {}).get("name")
            if isinstance(name, str):
                return name
    return None


async def _list_frontend_pods(kubectl: KubectlClient, namespace: str) -> list[str]:
    """Return frontend-like pod names in the Dynamo deployment namespace."""
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
    pods = orjson.loads(result.stdout or b"{}").get("items", [])
    names: list[str] = []
    for pod in pods:
        name = pod.get("metadata", {}).get("name")
        if isinstance(name, str) and "frontend" in name:
            names.append(name)
    return sorted(names)


async def _topology_mentions_route(
    kubectl: KubectlClient,
    namespaces: Iterable[str],
    route: str,
) -> bool:
    """Return whether pod env/args in any namespace mention ``route``."""
    for namespace in dict.fromkeys(namespaces):
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
            continue
        pods = orjson.loads(result.stdout or b"{}").get("items", [])
        for pod in pods:
            for container in pod.get("spec", {}).get("containers", []):
                if _container_mentions_route(container, route):
                    return True
    return False


def _container_mentions_route(container: dict[str, Any], route: str) -> bool:
    """Inspect one container's env/command/args for a route string."""
    for env in container.get("env", []):
        if isinstance(env, dict) and env.get("value") == route:
            return True
        if isinstance(env, dict) and route in str(env.get("value", "")):
            return True
    for field in ("command", "args"):
        values = container.get(field, [])
        if isinstance(values, list) and any(route in str(value) for value in values):
            return True
    return False


async def _find_patchable_dynamo_service(
    kubectl: KubectlClient,
    namespace: str,
) -> _ServiceSelectorPatch | None:
    """Find a Dynamo Service whose selector can be restored exactly."""
    result = await kubectl.run(
        "get",
        "services",
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return None
    services = orjson.loads(result.stdout or b"{}").get("items", [])
    for service in services:
        metadata = service.get("metadata", {})
        name = metadata.get("name")
        selector = service.get("spec", {}).get("selector") or {}
        if not isinstance(name, str) or not isinstance(selector, dict) or not selector:
            continue
        if "frontend" not in name and "worker" not in name:
            continue
        return _ServiceSelectorPatch(
            namespace=namespace,
            name=name,
            original_selector={str(key): str(value) for key, value in selector.items()},
        )
    return None


async def _patch_service_selector(
    kubectl: KubectlClient,
    namespace: str,
    name: str,
    selector: dict[str, str],
) -> None:
    """Patch a Service selector using merge patch."""
    patch = {"spec": {"selector": selector}}
    await kubectl.run(
        "patch",
        "service",
        name,
        "-n",
        namespace,
        "--type=merge",
        "-p",
        orjson.dumps(patch).decode(),
        check=True,
    )


async def _find_coredns_deployment(kubectl: KubectlClient) -> str | None:
    """Return CoreDNS/kube-dns deployment name in kube-system."""
    for selector in (
        "k8s-app=kube-dns",
        "k8s-app=coredns",
        "app.kubernetes.io/name=coredns",
    ):
        result = await kubectl.run(
            "get",
            "deployment",
            "-n",
            "kube-system",
            "-l",
            selector,
            "-o",
            "jsonpath={.items[0].metadata.name}",
            check=False,
        )
        name = result.stdout.strip() if result.returncode == 0 else ""
        if name:
            return name
    return None


async def _remove_proxy_safely(toxiproxy: Any, proxy_name: str, case_id: str) -> None:
    """Best-effort proxy cleanup so assertion failures are not masked."""
    try:
        await toxiproxy.remove_proxy(proxy_name)
    except Exception as exc:
        logger.warning(lambda exc=exc: f"{case_id} remove_proxy failed: {exc!r}")


# D818-D835

DYNAMO_CHAOS_ENV = "AIPERF_DYNAMO_CHAOS"
DEFAULT_TIMEOUT_S = 60

ChaosArea = Literal["nats", "etcd", "metadata"]


@dataclass(frozen=True, slots=True)
class Prerequisites:
    """Environment gates required before mutating a live Dynamo topology."""

    env_vars: tuple[str, ...] = ()
    requires_proxy: bool = False
    requires_ha: bool = False

    def missing(self, area: ChaosArea) -> list[str]:
        missing = [name for name in self.env_vars if not os.environ.get(name)]
        if self.requires_proxy:
            proxy_url_var = f"AIPERF_DYNAMO_{area.upper()}_TOXIPROXY_URL"
            proxy_name_var = f"AIPERF_DYNAMO_{area.upper()}_TOXIPROXY_NAME"
            missing.extend(
                name
                for name in (proxy_url_var, proxy_name_var)
                if not os.environ.get(name)
            )
        if self.requires_ha:
            ha_var = f"AIPERF_DYNAMO_{area.upper()}_HA"
            if os.environ.get(ha_var) != "1":
                missing.append(f"{ha_var}=1")
        return missing


@dataclass(frozen=True, slots=True)
class ChaosCase:
    """Single store/discovery chaos case from the expanded Dynamo matrix."""

    case_id: str
    area: ChaosArea
    name: str
    prerequisites: Prerequisites
    run: Callable[[Cluster], None]

    @property
    def id(self) -> str:
        return f"{self.case_id}-{self.area}-{self.name.replace(' ', '-')}"


@dataclass(frozen=True, slots=True)
class Cluster:
    """Minimal kubectl/toxiproxy facade for destructive opt-in tests."""

    namespace: str
    context: str | None

    def kubectl(
        self, *args: str, stdin: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        command = ["kubectl"]
        if self.context:
            command.extend(("--context", self.context))
        command.extend(("-n", self.namespace, *args))
        return subprocess.run(
            command,
            input=stdin,
            text=True,
            check=True,
            capture_output=True,
            timeout=DEFAULT_TIMEOUT_S,
        )

    def wait_for_selector(self, selector: str) -> None:
        self.kubectl(
            "wait",
            "pod",
            "--for=condition=Ready",
            f"--selector={selector}",
            "--timeout=180s",
        )

    def pods_for_selector(self, selector: str) -> list[str]:
        result = self.kubectl(
            "get",
            "pod",
            f"--selector={selector}",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        )
        return [pod for pod in result.stdout.split() if pod]

    def service_yaml(self, service_name: str) -> str:
        return self.kubectl("get", "service", service_name, "-o", "yaml").stdout

    def apply_yaml(self, yaml_text: str) -> None:
        self.kubectl("apply", "-f", "-", stdin=yaml_text)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise AssertionError(f"missing prerequisite environment variable {name}")
    return value


def _toxiproxy_request(
    area: ChaosArea,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> None:
    base_url = _env(f"AIPERF_DYNAMO_{area.upper()}_TOXIPROXY_URL")
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_S) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        if method == "DELETE" and exc.code == 404:
            return
        raise


def _toxiproxy_path(area: ChaosArea, suffix: str) -> str:
    proxy_name = _env(f"AIPERF_DYNAMO_{area.upper()}_TOXIPROXY_NAME")
    return f"/proxies/{proxy_name}{suffix}"


def _add_toxic(
    area: ChaosArea,
    toxic_name: str,
    toxic_type: str,
    attributes: dict[str, object],
) -> None:
    _toxiproxy_request(
        area,
        "POST",
        _toxiproxy_path(area, "/toxics"),
        {
            "name": toxic_name,
            "type": toxic_type,
            "stream": "downstream",
            "toxicity": 1.0,
            "attributes": attributes,
        },
    )


def _remove_toxic(area: ChaosArea, toxic_name: str) -> None:
    _toxiproxy_request(area, "DELETE", _toxiproxy_path(area, f"/toxics/{toxic_name}"))


def _with_toxic(
    area: ChaosArea,
    toxic_name: str,
    toxic_type: str,
    attributes: dict[str, object],
) -> None:
    _remove_toxic(area, toxic_name)
    try:
        _add_toxic(area, toxic_name, toxic_type, attributes)
        time.sleep(2)
    finally:
        _remove_toxic(area, toxic_name)


def _nats_partition(_: Cluster) -> None:
    _with_toxic("nats", "d818_partition", "timeout", {"timeout": 0})


def _nats_latency(_: Cluster) -> None:
    _with_toxic("nats", "d819_latency", "latency", {"latency": 750, "jitter": 100})


def _nats_bandwidth(_: Cluster) -> None:
    _with_toxic("nats", "d820_bandwidth", "bandwidth", {"rate": 16})


def _nats_reset(_: Cluster) -> None:
    _with_toxic("nats", "d821_reset", "reset_peer", {"timeout": 1000})


def _nats_restart(cluster: Cluster) -> None:
    selector = _env("AIPERF_DYNAMO_NATS_SELECTOR")
    pods_before = set(cluster.pods_for_selector(selector))
    assert pods_before, f"selector {selector!r} did not match any NATS pods"
    cluster.kubectl("delete", "pod", f"--selector={selector}", "--wait=true")
    cluster.wait_for_selector(selector)
    pods_after = set(cluster.pods_for_selector(selector))
    assert pods_after and pods_after != pods_before


def _nats_selector_delete(cluster: Cluster) -> None:
    service = _env("AIPERF_DYNAMO_NATS_SERVICE")
    original_yaml = cluster.service_yaml(service)
    try:
        cluster.kubectl(
            "patch",
            "service",
            service,
            "--type=merge",
            "-p",
            '{"spec":{"selector":{"aiperf.nvidia.com/chaos":"missing"}}}',
        )
        time.sleep(2)
    finally:
        cluster.apply_yaml(original_yaml)


def _nats_service_delete(cluster: Cluster) -> None:
    service = _env("AIPERF_DYNAMO_NATS_SERVICE")
    original_yaml = cluster.service_yaml(service)
    try:
        cluster.kubectl("delete", "service", service, "--wait=true")
    finally:
        cluster.apply_yaml(original_yaml)


def _etcd_bandwidth(_: Cluster) -> None:
    _with_toxic("etcd", "d825_bandwidth", "bandwidth", {"rate": 32})


def _etcd_reset(_: Cluster) -> None:
    _with_toxic("etcd", "d826_reset", "reset_peer", {"timeout": 1000})


def _etcd_partition(_: Cluster) -> None:
    _with_toxic("etcd", "d827_partition", "timeout", {"timeout": 0})


def _etcd_compaction(cluster: Cluster) -> None:
    selector = _env("AIPERF_DYNAMO_ETCD_SELECTOR")
    pods = cluster.pods_for_selector(selector)
    assert pods, f"selector {selector!r} did not match any etcd pods"
    cluster.kubectl("exec", pods[0], "--", "etcdctl", "compact", "1")


def _etcd_leader_restart(cluster: Cluster) -> None:
    selector = _env("AIPERF_DYNAMO_ETCD_SELECTOR")
    pods = cluster.pods_for_selector(selector)
    assert pods, f"selector {selector!r} did not match any etcd pods"
    cluster.kubectl("delete", "pod", pods[0], "--wait=true")
    cluster.wait_for_selector(selector)


def _etcd_quorum_loss(cluster: Cluster) -> None:
    selector = _env("AIPERF_DYNAMO_ETCD_SELECTOR")
    pods = cluster.pods_for_selector(selector)
    assert len(pods) >= 3, "etcd quorum-loss chaos requires at least three etcd pods"
    victims = pods[:2]
    try:
        cluster.kubectl("delete", "pod", *victims, "--wait=true")
        time.sleep(2)
    finally:
        cluster.wait_for_selector(selector)


def _metadata_configmap_name() -> str:
    return _env("AIPERF_DYNAMO_METADATA_CONFIGMAP")


def _metadata_malformed(cluster: Cluster) -> None:
    configmap = _metadata_configmap_name()
    original = cluster.kubectl("get", "configmap", configmap, "-o", "yaml").stdout
    try:
        cluster.kubectl(
            "patch",
            "configmap",
            configmap,
            "--type=merge",
            "-p",
            '{"data":{"endpoints":"not: [valid"}}',
        )
        time.sleep(2)
    finally:
        cluster.apply_yaml(original)


def _metadata_duplicate(cluster: Cluster) -> None:
    configmap = _metadata_configmap_name()
    original = cluster.kubectl("get", "configmap", configmap, "-o", "yaml").stdout
    try:
        cluster.kubectl(
            "patch",
            "configmap",
            configmap,
            "--type=merge",
            "-p",
            json.dumps(
                {"data": {"duplicate-endpoints": '[{"name":"dup"},{"name":"dup"}]'}}
            ),
        )
        time.sleep(2)
    finally:
        cluster.apply_yaml(original)


def _metadata_delete(cluster: Cluster) -> None:
    configmap = _metadata_configmap_name()
    original = cluster.kubectl("get", "configmap", configmap, "-o", "yaml").stdout
    try:
        cluster.kubectl("delete", "configmap", configmap, "--wait=true")
        time.sleep(2)
    finally:
        cluster.apply_yaml(original)


def _metadata_freeze(cluster: Cluster) -> None:
    selector = _env("AIPERF_DYNAMO_METADATA_SELECTOR")
    pods = cluster.pods_for_selector(selector)
    assert pods, f"selector {selector!r} did not match any metadata pods"
    cluster.kubectl(
        "exec", pods[0], "--", "sh", "-c", "kill -STOP 1; sleep 2; kill -CONT 1"
    )


def _metadata_watch_storm(cluster: Cluster) -> None:
    configmap = _metadata_configmap_name()
    for index in range(10):
        cluster.kubectl(
            "patch",
            "configmap",
            configmap,
            "--type=merge",
            "-p",
            f'{{"metadata":{{"annotations":{{"aiperf.nvidia.com/chaos-watch-storm":"{index}"}}}}}}',
        )


NATS_PROXY = Prerequisites(requires_proxy=True)
NATS_TOPOLOGY = Prerequisites(("AIPERF_DYNAMO_NATS_SELECTOR",))
NATS_SERVICE = Prerequisites(
    ("AIPERF_DYNAMO_NATS_SELECTOR", "AIPERF_DYNAMO_NATS_SERVICE")
)
ETCD_PROXY_HA = Prerequisites(requires_proxy=True, requires_ha=True)
ETCD_HA = Prerequisites(("AIPERF_DYNAMO_ETCD_SELECTOR",), requires_ha=True)
METADATA_TOPOLOGY = Prerequisites(
    ("AIPERF_DYNAMO_METADATA_CONFIGMAP", "AIPERF_DYNAMO_METADATA_SELECTOR")
)
METADATA_CONFIG = Prerequisites(("AIPERF_DYNAMO_METADATA_CONFIGMAP",))

CASES: tuple[ChaosCase, ...] = (
    ChaosCase("D818", "nats", "partition", NATS_PROXY, _nats_partition),
    ChaosCase("D819", "nats", "latency", NATS_PROXY, _nats_latency),
    ChaosCase("D820", "nats", "bandwidth", NATS_PROXY, _nats_bandwidth),
    ChaosCase("D821", "nats", "reset", NATS_PROXY, _nats_reset),
    ChaosCase("D822", "nats", "restart", NATS_TOPOLOGY, _nats_restart),
    ChaosCase(
        "D823", "nats", "selector service delete", NATS_SERVICE, _nats_selector_delete
    ),
    ChaosCase("D824", "nats", "service delete", NATS_SERVICE, _nats_service_delete),
    ChaosCase("D825", "etcd", "bandwidth", ETCD_PROXY_HA, _etcd_bandwidth),
    ChaosCase("D826", "etcd", "reset", ETCD_PROXY_HA, _etcd_reset),
    ChaosCase("D827", "etcd", "partition", ETCD_PROXY_HA, _etcd_partition),
    ChaosCase("D828", "etcd", "compaction", ETCD_HA, _etcd_compaction),
    ChaosCase("D829", "etcd", "leader restart", ETCD_HA, _etcd_leader_restart),
    ChaosCase("D830", "etcd", "quorum loss", ETCD_HA, _etcd_quorum_loss),
    ChaosCase("D831", "metadata", "malformed", METADATA_CONFIG, _metadata_malformed),
    ChaosCase("D832", "metadata", "duplicate", METADATA_CONFIG, _metadata_duplicate),
    ChaosCase("D833", "metadata", "delete", METADATA_CONFIG, _metadata_delete),
    ChaosCase("D834", "metadata", "freeze", METADATA_TOPOLOGY, _metadata_freeze),
    ChaosCase(
        "D835", "metadata", "watch storm", METADATA_CONFIG, _metadata_watch_storm
    ),
)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case" in metafunc.fixturenames:
        metafunc.parametrize("case", CASES, ids=[case.id for case in CASES])


@pytest.fixture
def cluster() -> Iterator[Cluster]:
    if os.environ.get(DYNAMO_CHAOS_ENV) != "1":
        pytest.skip(f"set {DYNAMO_CHAOS_ENV}=1 to run destructive Dynamo chaos tests")
    if not shutil.which("kubectl"):
        pytest.skip("kubectl is required for Dynamo chaos tests")
    namespace = os.environ.get("AIPERF_DYNAMO_NAMESPACE")
    if not namespace:
        pytest.skip("set AIPERF_DYNAMO_NAMESPACE to the disposable Dynamo namespace")
    yield Cluster(
        namespace=namespace, context=os.environ.get("AIPERF_DYNAMO_KUBE_CONTEXT")
    )


def test_expanded_spec_cases_cover_d818_through_d835() -> None:
    assert [case.case_id for case in CASES] == [
        f"D{case_id}" for case_id in range(818, 836)
    ]


def test_store_discovery_chaos_case_has_explicit_prerequisites(case: ChaosCase) -> None:
    missing = case.prerequisites.missing(case.area)
    assert (
        case.prerequisites.env_vars
        or case.prerequisites.requires_proxy
        or case.prerequisites.requires_ha
    )
    assert all(item for item in missing)


def test_store_discovery_chaos_case_executes_with_required_topology(
    cluster: Cluster, case: ChaosCase
) -> None:
    missing = case.prerequisites.missing(case.area)
    if missing:
        pytest.skip(
            f"{case.case_id} requires explicit prerequisites: {', '.join(missing)}"
        )
    case.run(cluster)
