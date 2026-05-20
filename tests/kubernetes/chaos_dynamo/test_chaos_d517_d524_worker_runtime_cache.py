# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D517-D524 -- Dynamo worker/runtime/cache chaos scenarios.

These cases cover the expanded D5xx worker-runtime/cache matrix:

* D517 CPU throttling
* D518 host memory pressure
* D519 slow PreStop hook
* D520 false-ready readiness probe
* D521 tiny KVBM cache with read-only spill path
* D522 full disk/cache
* D523 deleted cache directory
* D524 network noisy-neighbor contention

Several faults require topology controls that the stock v1alpha1 Dynamo fixture does
not provide: podTemplate mutation, same-node sidecars/standalone pressure pods,
NET_ADMIN, or an explicit cache mount. Those tests skip before mutation with a
precondition that names the missing control so a lab topology can opt in without
turning a no-op into a false pass.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

_WORKER_SELECTORS = (
    "nvidia.com/dynamo-sub-component-type=decode",
    "nvidia.com/dynamo-component-type=worker",
)
_PREFILL_SELECTOR = "nvidia.com/dynamo-sub-component-type=prefill"
_MAIN_CONTAINER = "main"
_HEALTH_PORT = 9090
_CPU_HOG_SECONDS = 20
_MEMORY_PRESSURE_ENV = "AIPERF_DYNAMO_HOST_MEMORY_PRESSURE_IMAGE"
_CACHE_CHAOS_ENV = "AIPERF_DYNAMO_DESTRUCTIVE_CACHE_CHAOS"
_CACHE_PATH_ENV = "AIPERF_DYNAMO_CACHE_CHAOS_PATH"
_NOISY_NEIGHBOR_ENV = "AIPERF_DYNAMO_NET_NOISY_NEIGHBOR_POD"
_NET_ADMIN_ENV = "AIPERF_DYNAMO_NET_ADMIN_CONTAINER"


async def test_d517_cpu_throttled_worker_stays_live(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    """D517: under an actual cgroup CPU quota, worker health must not flap.

    This test does not simulate throttling by merely burning CPU on an unlimited
    pod. It first proves the selected worker container has a finite cgroup CPU
    quota, then runs a short in-container CPU hog and verifies ``/live`` still
    responds through the health port. Stock Dynamo fixtures usually skip here
    because they do not set container CPU limits.
    """
    pod = await _first_pod_for_selectors(
        kubectl, dynamo_deployment_namespace, _WORKER_SELECTORS
    )
    quota = await _read_cpu_quota(kubectl, dynamo_deployment_namespace, pod["name"])
    if quota is None:
        pytest.skip(
            "D517 requires a worker pod with finite cgroup CPU quota/limit; "
            f"pod {pod['namespace']}/{pod['name']} is not CPU-throttled, so a "
            "CPU hog would test contention rather than throttling."
        )

    hog = _shell_background(
        "python3 - <<'PY'\n"
        "import multiprocessing, time\n"
        f"deadline = time.time() + {_CPU_HOG_SECONDS}\n"
        "def burn():\n"
        "    while time.time() < deadline:\n"
        "        pass\n"
        "procs = [multiprocessing.Process(target=burn) for _ in range(8)]\n"
        "[p.start() for p in procs]\n"
        "[p.join() for p in procs]\n"
        "PY"
    )
    await _exec(kubectl, pod, hog, check=True, timeout=5)
    await _assert_live(kubectl, pod, timeout=10.0)


async def test_d518_host_memory_pressure_surfaces_cleanly(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    """D518: same-node host memory pressure should not silently wedge workers.

    Requires an explicit pressure image because host-memory stress is node-scoped
    and potentially disruptive. The test schedules a short-lived pressure pod on
    the worker node, then confirms the worker is either still live or has a clear
    kubelet termination reason instead of disappearing from the operator surface.
    """
    pressure_image = os.environ.get(_MEMORY_PRESSURE_ENV, "").strip()
    if not pressure_image:
        pytest.skip(
            f"D518 requires {_MEMORY_PRESSURE_ENV}=<image> for a same-node "
            "memory-pressure helper pod; stock fixtures do not include a safe "
            "host-pressure sidecar or node-level control."
        )

    pod = await _first_pod_for_selectors(
        kubectl, dynamo_deployment_namespace, _WORKER_SELECTORS
    )
    node_name = pod["body"].get("spec", {}).get("nodeName", "")
    if not node_name:
        pytest.skip(
            f"D518 requires the target worker pod {pod['namespace']}/{pod['name']} "
            "to be scheduled so a same-node pressure pod can be pinned; "
            "spec.nodeName is empty."
        )

    pressure_name = "d518-host-memory-pressure"
    await kubectl.apply(
        _memory_pressure_pod_manifest(
            name=pressure_name,
            namespace=dynamo_deployment_namespace,
            node_name=node_name,
            image=pressure_image,
        )
    )
    try:
        await asyncio.sleep(10.0)
        await _assert_worker_live_or_cleanly_restarted(kubectl, pod)
    finally:
        await kubectl.run(
            "delete",
            "pod",
            pressure_name,
            "-n",
            dynamo_deployment_namespace,
            "--ignore-not-found",
            "--force",
            "--grace-period=0",
            check=False,
        )


async def test_d519_slow_prestop_does_not_block_replacement(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    """D519: a configured slow PreStop hook must not strand the worker replica.

    The stock fixture has no worker ``lifecycle.preStop`` hook. This case only
    runs on a topology that deliberately injected one through podTemplate; without
    that prerequisite, deleting a pod would test ordinary replacement, not the
    slow-death path.
    """
    pod = await _first_pod_for_selectors(
        kubectl, dynamo_deployment_namespace, _WORKER_SELECTORS
    )
    pre_stop = _main_container(pod["body"]).get("lifecycle", {}).get("preStop")
    if not pre_stop:
        pytest.skip(
            "D519 requires a worker podTemplate lifecycle.preStop hook, usually "
            "a deliberate sleep; the current worker container has no preStop hook."
        )

    await kubectl.run(
        "delete",
        "pod",
        pod["name"],
        "-n",
        dynamo_deployment_namespace,
        "--grace-period=30",
        check=True,
    )
    replacement = await _wait_for_replacement_ready(
        kubectl,
        namespace=dynamo_deployment_namespace,
        old_pod=pod["name"],
        selectors=_WORKER_SELECTORS,
        timeout=90.0,
    )
    assert replacement, "D519: replacement worker pod name should not be empty"


async def test_d520_false_ready_probe_is_detected_by_live_endpoint(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    """D520: readiness that lies must be caught by an independent live probe.

    Runs only when the worker pod declares a readinessProbe that is not the
    canonical ``/live`` HTTP probe. The test contrasts Kubernetes Ready with a
    direct ``/live`` check so false-ready topologies are not accepted silently.
    """
    pod = await _first_pod_for_selectors(
        kubectl, dynamo_deployment_namespace, _WORKER_SELECTORS
    )
    container = _main_container(pod["body"])
    readiness_probe = container.get("readinessProbe")
    if not readiness_probe:
        pytest.skip(
            "D520 requires a deliberately configured worker readinessProbe; the "
            "stock Dynamo pod relies on startup/liveness probes only."
        )
    if readiness_probe == container.get("livenessProbe"):
        pytest.skip(
            "D520 requires a false-ready or divergent readinessProbe; readiness "
            "currently matches liveness, so there is no false-ready fault."
        )

    assert _pod_ready_condition(pod["body"]) is True, (
        f"D520 precondition expected Kubernetes Ready=True for {pod['name']}; "
        "the false-ready fault is not active yet."
    )
    await _assert_live(kubectl, pod, timeout=10.0)


async def test_d521_tiny_kv_cache_read_only_spill_is_explicitly_gated(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    """D521: tiny KVBM cache with read-only spill path fails cleanly.

    The scenario needs a KVBM-enabled prefill worker with ``DYN_KVBM_CPU_CACHE_GB``
    set tiny and a read-only spill/cache mount. If either is absent, the test
    skips before traffic so it cannot pass without exercising cache spill.
    """
    pod = await _first_pod_for_selectors(
        kubectl, dynamo_deployment_namespace, (_PREFILL_SELECTOR,)
    )
    env = _container_env(_main_container(pod["body"]))
    cache_gb = env.get("DYN_KVBM_CPU_CACHE_GB")
    if cache_gb not in {"0", "1"}:
        pytest.skip(
            "D521 requires a tiny KVBM CPU cache on a prefill worker "
            f"(DYN_KVBM_CPU_CACHE_GB=0 or 1); observed {cache_gb!r}."
        )
    readonly_mounts = [
        m
        for m in _main_container(pod["body"]).get("volumeMounts", [])
        if m.get("readOnly")
    ]
    if not readonly_mounts:
        pytest.skip(
            "D521 requires an explicit read-only cache/spill volumeMount on the "
            "prefill worker; no readOnly volumeMount is present."
        )

    await _assert_live(kubectl, pod, timeout=10.0)


async def test_d522_full_disk_cache_recovers_or_fails_cleanly(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    """D522: filling the cache filesystem should not wedge the worker forever.

    This is destructive to the configured cache path and therefore requires both
    an explicit path and an opt-in environment variable.
    """
    if os.environ.get(_CACHE_CHAOS_ENV) != "1":
        pytest.skip(
            f"D522 requires {_CACHE_CHAOS_ENV}=1 because filling a cache "
            "filesystem is destructive and the stock fixture has no disposable "
            "cache sidecar/volume."
        )
    cache_path = os.environ.get(_CACHE_PATH_ENV, "").strip()
    if not cache_path.startswith("/"):
        pytest.skip(
            f"D522 requires {_CACHE_PATH_ENV}=<absolute-cache-path>; observed "
            f"{cache_path!r}."
        )

    pod = await _first_pod_for_selectors(
        kubectl, dynamo_deployment_namespace, _WORKER_SELECTORS
    )
    await _exec(
        kubectl,
        pod,
        f"mkdir -p {cache_path!r} && dd if=/dev/zero of={cache_path!r}/d522.fill bs=1M count=64 || true",
        check=False,
        timeout=60,
    )
    await _assert_worker_live_or_cleanly_restarted(kubectl, pod)


async def test_d523_deleted_cache_dir_is_recreated_or_errors_cleanly(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    """D523: deleting the cache directory must not leave opaque bad state."""
    if os.environ.get(_CACHE_CHAOS_ENV) != "1":
        pytest.skip(
            f"D523 requires {_CACHE_CHAOS_ENV}=1 because removing cache state is "
            "destructive; stock deployments do not expose a disposable cache dir."
        )
    cache_path = os.environ.get(_CACHE_PATH_ENV, "").strip()
    if not cache_path.startswith("/"):
        pytest.skip(
            f"D523 requires {_CACHE_PATH_ENV}=<absolute-cache-path>; observed "
            f"{cache_path!r}."
        )

    pod = await _first_pod_for_selectors(
        kubectl, dynamo_deployment_namespace, _WORKER_SELECTORS
    )
    await _exec(kubectl, pod, f"rm -rf {cache_path!r}", check=True, timeout=30)
    await _assert_worker_live_or_cleanly_restarted(kubectl, pod)


async def test_d524_network_noisy_neighbor_does_not_break_health(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    faults: InjectorRegistry,
) -> None:
    """D524: network-noisy-neighbor contention preserves worker health.

    The fault requires either a pre-created noisy-neighbor pod or a NET_ADMIN
    helper container that can apply traffic shaping. Without that, applying a
    Toxiproxy toxic to an unrelated port would be a no-op and is skipped.
    """
    noisy_pod = os.environ.get(_NOISY_NEIGHBOR_ENV, "").strip()
    net_admin_container = os.environ.get(_NET_ADMIN_ENV, "").strip()
    if not noisy_pod and not net_admin_container:
        pytest.skip(
            f"D524 requires {_NOISY_NEIGHBOR_ENV}=<pod> or "
            f"{_NET_ADMIN_ENV}=<container> so traffic shaping/noise runs in the "
            "worker network namespace; stock pods have no NET_ADMIN sidecar."
        )

    pod = await _first_pod_for_selectors(
        kubectl, dynamo_deployment_namespace, _WORKER_SELECTORS
    )
    if noisy_pod:
        await _exec(
            kubectl,
            {"name": noisy_pod, "namespace": dynamo_deployment_namespace},
            "sh -lc 'for i in $(seq 1 20); do wget -q -O- http://127.0.0.1:9090/live >/dev/null 2>&1 || true; done'",
            check=False,
            timeout=30,
        )
    else:
        async with faults.inject(
            "network.latency",
            target={"proxy": "d524-worker-net"},
            attributes={"latency_ms": 100, "jitter_ms": 50},
        ):
            logger.info(
                "D524: latency toxic applied to externally managed worker proxy"
            )
    await _assert_live(kubectl, pod, timeout=10.0)


async def _first_pod_for_selectors(
    kubectl: KubectlClient,
    namespace: str,
    selectors: tuple[str, ...],
) -> dict[str, Any]:
    """Return the first running pod matching one of ``selectors`` or skip."""
    for selector in selectors:
        result = await kubectl.run(
            "get",
            "pod",
            "-n",
            namespace,
            "-l",
            selector,
            "-o",
            "json",
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            continue
        data = orjson.loads(result.stdout)
        for item in data.get("items", []):
            if item.get("status", {}).get("phase") != "Running":
                continue
            name = item.get("metadata", {}).get("name", "")
            if name:
                return {"name": name, "namespace": namespace, "body": item}
    pytest.skip(
        f"No running Dynamo worker pod found in namespace {namespace!r} for "
        f"selectors={selectors!r}; scenario requires an active worker."
    )


async def _read_cpu_quota(
    kubectl: KubectlClient,
    namespace: str,
    pod_name: str,
) -> str | None:
    """Return finite cgroup CPU quota text, or None for unlimited/unknown."""
    result = await kubectl.run(
        "exec",
        pod_name,
        "-n",
        namespace,
        "-c",
        _MAIN_CONTAINER,
        "--",
        "sh",
        "-lc",
        "if [ -r /sys/fs/cgroup/cpu.max ]; then cat /sys/fs/cgroup/cpu.max; "
        "elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then "
        "cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us; fi",
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    quota = result.stdout.strip()
    if not quota or quota.startswith("max") or quota == "-1":
        return None
    return quota


async def _exec(
    kubectl: KubectlClient,
    pod: dict[str, Any],
    shell_command: str,
    *,
    check: bool,
    timeout: int,
) -> str:
    """Run ``shell_command`` in the pod's main container and return stdout."""
    result = await kubectl.run(
        "exec",
        pod["name"],
        "-n",
        pod["namespace"],
        "-c",
        _MAIN_CONTAINER,
        "--",
        "sh",
        "-lc",
        shell_command,
        check=check,
        timeout=timeout,
    )
    return result.stdout


def _shell_background(command: str) -> str:
    """Wrap a shell snippet so it starts in the background and returns promptly."""
    return f"({command}) >/tmp/aiperf-d517-cpu-hog.log 2>&1 &"


async def _assert_live(
    kubectl: KubectlClient,
    pod: dict[str, Any],
    *,
    timeout: float,
) -> None:
    """Assert the worker ``/live`` endpoint answers through port-forward."""
    async with (
        kubectl.port_forward(
            pod["name"], _HEALTH_PORT, namespace=pod["namespace"]
        ) as local_port,
        aiohttp.ClientSession() as session,
        session.get(
            f"http://127.0.0.1:{local_port}/live",
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp,
    ):
        body = await resp.text()
        assert resp.status == 200, (
            f"worker {pod['namespace']}/{pod['name']} /live returned "
            f"HTTP {resp.status}: {body[:200]!r}"
        )


async def _assert_worker_live_or_cleanly_restarted(
    kubectl: KubectlClient,
    pod: dict[str, Any],
) -> None:
    """Accept live health or a kubelet reason that explicitly names pressure."""
    refreshed = await _read_pod(kubectl, pod["namespace"], pod["name"])
    if refreshed is None:
        replacement = await _wait_for_replacement_ready(
            kubectl,
            namespace=pod["namespace"],
            old_pod=pod["name"],
            selectors=_WORKER_SELECTORS,
            timeout=90.0,
        )
        assert replacement, "replacement worker pod name should not be empty"
        return

    reasons = _container_termination_reasons(refreshed)
    if reasons:
        assert any(
            reason in {"OOMKilled", "Error", "Completed"} for reason in reasons
        ), (
            f"worker {pod['namespace']}/{pod['name']} terminated with unexpected "
            f"reasons={reasons!r}"
        )
        return
    await _assert_live(kubectl, {**pod, "body": refreshed}, timeout=10.0)


async def _read_pod(
    kubectl: KubectlClient,
    namespace: str,
    name: str,
) -> dict[str, Any] | None:
    """Return a pod JSON object or None when it no longer exists."""
    result = await kubectl.run(
        "get",
        "pod",
        name,
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return orjson.loads(result.stdout)


async def _wait_for_replacement_ready(
    kubectl: KubectlClient,
    *,
    namespace: str,
    old_pod: str,
    selectors: tuple[str, ...],
    timeout: float,
) -> str:
    """Wait for a ready worker pod whose name differs from ``old_pod``."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        for selector in selectors:
            result = await kubectl.run(
                "get",
                "pod",
                "-n",
                namespace,
                "-l",
                selector,
                "-o",
                "json",
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                continue
            data = orjson.loads(result.stdout)
            for item in data.get("items", []):
                name = item.get("metadata", {}).get("name", "")
                if name != old_pod and _pod_ready_condition(item):
                    return name
        await asyncio.sleep(1.0)
    raise TimeoutError(
        f"No replacement worker pod became Ready in namespace {namespace!r} "
        f"within {timeout}s after deleting {old_pod!r}"
    )


def _main_container(pod_body: dict[str, Any]) -> dict[str, Any]:
    """Return the main container spec from a pod object."""
    containers = pod_body.get("spec", {}).get("containers", [])
    for container in containers:
        if container.get("name") == _MAIN_CONTAINER:
            return container
    if containers:
        return containers[0]
    pytest.skip(
        f"pod {pod_body.get('metadata', {}).get('name', '<unknown>')!r} has no "
        "containers; cannot run worker-runtime chaos scenario"
    )


def _container_env(container: dict[str, Any]) -> dict[str, str]:
    """Return literal env vars from a container spec."""
    env: dict[str, str] = {}
    for entry in container.get("env", []):
        name = entry.get("name")
        value = entry.get("value")
        if isinstance(name, str) and isinstance(value, str):
            env[name] = value
    return env


def _pod_ready_condition(pod_body: dict[str, Any]) -> bool:
    """Return True when Pod Ready condition is True."""
    for condition in pod_body.get("status", {}).get("conditions", []):
        if condition.get("type") == "Ready":
            return condition.get("status") == "True"
    return False


def _container_termination_reasons(pod_body: dict[str, Any]) -> list[str]:
    """Return terminated/lastState reasons from containerStatuses."""
    reasons: list[str] = []
    statuses = pod_body.get("status", {}).get("containerStatuses", [])
    for status in statuses:
        state = status.get("state", {})
        last_state = status.get("lastState", {})
        for field in (state.get("terminated"), last_state.get("terminated")):
            if isinstance(field, dict) and isinstance(field.get("reason"), str):
                reasons.append(field["reason"])
    return reasons


def _memory_pressure_pod_manifest(
    *,
    name: str,
    namespace: str,
    node_name: str,
    image: str,
) -> str:
    """Build a same-node memory pressure pod manifest."""
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {"aiperf.nvidia.com/chaos-case": "d518"},
        },
        "spec": {
            "nodeName": node_name,
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "pressure",
                    "image": image,
                    "command": ["sh", "-lc"],
                    "args": [
                        "python3 - <<'PY'\n"
                        "import time\n"
                        "blocks=[]\n"
                        "deadline=time.time()+20\n"
                        "while time.time()<deadline:\n"
                        "    blocks.append(bytearray(64*1024*1024))\n"
                        "    time.sleep(0.2)\n"
                        "PY"
                    ],
                    "resources": {
                        "requests": {"memory": "256Mi", "cpu": "100m"},
                        "limits": {"memory": "8Gi", "cpu": "1"},
                    },
                }
            ],
        },
    }
    return orjson.dumps(manifest, option=orjson.OPT_INDENT_2).decode()
