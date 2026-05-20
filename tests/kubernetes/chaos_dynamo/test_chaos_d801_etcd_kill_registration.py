# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D801 -- etcd kill during decode-worker registration race.

Known flake risk per plan section 4. Timing-sensitive; retry up to 3 times.

Scenario (Wave-0 #3):
    Kill the etcd pod via ``faults.inject("store.etcd.kill")`` while a fresh
    decode worker pod is mid-boot -- the window where ``register_model`` has
    been issued (worker_factory.py:398, 433) and the etcd lease grant
    (runtime/src/transports/etcd/lease.rs:21) is in flight.

Assertion:
    No half-registered state. Either the worker retries to registration
    success within ~90s (lease TTL + timeout) **or** it fails cleanly into a
    ``CrashLoopBackOff`` with a clear error message. A worker stuck in the
    router roster while not actually serving requests is a FAIL.
"""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

MAX_RETRIES = 3
"""Per plan section 4: accept first PASS, or 2-of-3 if any flake."""

_DECODE_POD_SELECTOR = "nvidia.com/dynamo-sub-component-type=decode"
"""Label set by the dynamo-operator on every decode-role pod."""

_FRESH_POD_WAIT_S = 30.0
"""How long to wait for the scaled-up replica to enter Running+NotReady."""

_POST_FAULT_SETTLE_S = 90.0
"""Lease TTL + reconnect timeout. After this, the worker has either
re-registered cleanly or crashed with a clear error -- anything still
half-registered is the regression we are testing for."""

_TRAFFIC_PROBE_TIMEOUT_S = 30.0
"""Per-request HTTP timeout for the PASS-A traffic probe."""

_ETCD_ERROR_NEEDLES = (b"etcd", b"register", b"lease")
"""Lowercased substrings that must appear in a CrashLoopBackOff log to
qualify as PASS-B (clean error, not a hang). All three categories are
common phrasing in the runtime's registration path."""


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
    pytest.skip(
        "scaffold landed; awaiting Dynamo deployment with mutable decode-worker replicas"
    )
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
    """Locate the decode-role component in a v1beta1 DGD spec."""
    components = dgd.get("spec", {}).get("components", [])
    return next(
        (c for c in components if c.get("type") == "decode"),
        None,
    )


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
    components = dgd.get("spec", {}).get("components", [])
    for component in components:
        if component.get("type") == "decode":
            component["replicas"] = replicas
            break
    patch = orjson.dumps({"spec": {"components": components}}).decode()
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
                f"{dynamo_endpoint_url}/v1/chat/completions", json=body
            ) as resp,
        ):
            if resp.status == 200:
                return True, "status=200"
            text = (await resp.read()).decode(errors="replace")[:256]
            return False, f"status={resp.status} body={text!r}"
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return False, f"client error {exc!r}"
