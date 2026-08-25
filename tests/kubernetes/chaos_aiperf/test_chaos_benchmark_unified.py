# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B1/B2/B3 unified-API ports - see chaos/test_chaos_benchmark.py for legacy.

Exercises the benchmark-runtime fault paths against the mock inference
server through the unified
:py:class:`tests.kubernetes.chaos_common.registry.InjectorRegistry`
``faults.inject(...)`` interface rather than the legacy
:py:class:`MockServerInjector` / :py:class:`ToxiproxyInjector` direct
calls. Each scenario maps to one unified fault id:

* B1 -> ``workload.set_env`` against the mock-server Deployment to flip
  ``AIPERF_MOCK_FORCE_STATUS=500``; benchmark must drive the AIPerfJob
  to ``Completed`` and surface non-zero ``error_request_count``.
* B2 -> ``pod.kill`` against a discovered mock-server pod name; the
  Deployment respawns the pod and the benchmark must tolerate the flap.
* B3 -> ``network.latency`` toxic via the package-scoped Toxiproxy proxy;
  ``request_latency.p99`` must reflect the injected 500 ms delay.

The function-scope :py:data:`faults` registry inherited from
:py:mod:`tests.kubernetes.chaos_aiperf.conftest` already pre-registers
:py:class:`PodInjector`, :py:class:`WorkloadInjector`,
:py:class:`NetworkInjector`, and :py:class:`CRDInjector`, so no extra
``register()`` calls are needed in this module.

Mock-server contract (see ``tests/aiperf_mock_server/config.py``):

* ``AIPERF_MOCK_FORCE_STATUS`` (int, HTTP status code): force every
  response to this status. B1 sets it to ``500`` to make the benchmark
  see a hard error stream.
* Deployment + Service name: ``aiperf-mock-server`` in ``default``,
  port 8000 (see ``dev/deploy/mock-server.yaml``).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.chaos.toxiproxy import (
    TOXIPROXY_MOCK_SERVER_PORT,
    TOXIPROXY_NAMESPACE,
    TOXIPROXY_SERVICE,
    ToxiproxyInjector,
)
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

pytestmark = [pytest.mark.asyncio, pytest.mark.k8s_slow]

MOCK_SERVER_NAMESPACE = "default"
"""Namespace where ``tests/kubernetes/conftest.py::mock_server`` deploys."""

MOCK_SERVER_DEPLOYMENT = "aiperf-mock-server"
"""Deployment + Service name for the mock inference server."""

MOCK_SERVER_SERVICE_PORT = 8000
"""Service port the mock server listens on."""


@pytest.fixture
def longrun_config(k8s_settings) -> AIPerfJobConfig:  # noqa: ANN001 - test-fixture dataclass
    """Duration-based AIPerfJob so faults can land mid-profiling.

    Mirrors the identically named fixture in
    :py:mod:`tests.kubernetes.chaos.test_chaos_benchmark`; kept local
    because pytest does not share package-private fixtures across
    sibling test packages (``chaos/`` vs. ``chaos_aiperf/``).
    """
    return AIPerfJobConfig(
        concurrency=3,
        request_count=None,
        benchmark_duration=120.0,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )


def _metric_avg(metrics: dict[str, Any], key: str) -> float | None:
    """Return the average value for a metric key in the CR status results.

    ``status.results.metrics`` values may be either a scalar (int/float)
    or a distribution dict ``{"avg": ..., "p99": ..., ...}`` depending on
    the metric. This helper normalises both shapes so callers do not
    branch on the per-metric schema.
    """
    val = metrics.get(key)
    if isinstance(val, dict):
        avg = val.get("avg")
        return float(avg) if avg is not None else None
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _metric_p99(metrics: dict[str, Any], key: str) -> float | None:
    """Return the p99 value for a distribution metric, or None if missing."""
    val = metrics.get(key)
    if isinstance(val, dict):
        p99 = val.get("p99")
        return float(p99) if p99 is not None else None
    return None


async def _force_delete_cr(kubectl: KubectlClient, namespace: str, name: str) -> None:
    """Drop the AIPerfJob CR without blocking on finalizer settlement.

    Mirrors the teardown pattern in :py:mod:`tests.kubernetes.chaos.test_chaos_cancellation`
    so a failed assertion never leaves an AIPerfJob around to poison the
    next test.
    """
    await kubectl.run(
        "delete",
        "aiperfjob",
        name,
        "-n",
        namespace,
        "--ignore-not-found",
        "--wait=false",
        check=False,
    )


async def _resolve_mock_server_pod(kubectl: KubectlClient, namespace: str) -> str:
    """Return the name of a running mock-server pod for ``pod.kill`` targeting.

    The unified ``pod.kill`` fault id takes a specific pod name in its
    ``target`` payload (see :py:class:`PodInjector` ABC contract), unlike
    the legacy :py:meth:`MockServerInjector.delete_pod` which selects by
    ``app=aiperf-mock-server`` label. We replicate the label-based
    discovery here and pass the first hit through to ``pod.kill``.

    Raises:
        RuntimeError: When no pod matches the selector -- B2 is meaningless
            without a live mock-server to kill.
    """
    res = await kubectl.run(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        f"app={MOCK_SERVER_DEPLOYMENT}",
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=False,
    )
    pod = res.stdout.strip() if res.returncode == 0 else ""
    if not pod:
        raise RuntimeError(
            f"B2: no mock-server pod matched label app={MOCK_SERVER_DEPLOYMENT!r} "
            f"in namespace {namespace!r}; check `kubectl get pods -n {namespace}`"
        )
    return pod


@pytest.fixture
def waiter(kubectl: KubectlClient) -> ChaosInjector:
    """Plain-helper :py:class:`ChaosInjector` for ``wait_for_*`` calls.

    The unified API only exposes fault injection (``inject`` /
    ``compose``); polling helpers like ``wait_for_phase`` remain on the
    legacy :py:class:`ChaosInjector`. The task contract explicitly
    allows reusing those helpers verbatim.
    """
    return ChaosInjector(kubectl)


async def test_b1_mock_server_500s_mid_run_unified(
    operator_ready: OperatorDeployer,
    faults: InjectorRegistry,
    waiter: ChaosInjector,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """B1 unified port: flip mock-server to 500s mid-run via ``workload.set_env``.

    Exercises the worker error-handling path in ``src/aiperf/workers/``
    when the inference server returns HTTP 500s: the worker must record
    a ``RequestFailure`` credit (not hang), the records-manager must
    surface it as a non-zero ``error_request_count`` metric, and the
    system-controller must still drive the CR to ``Completed`` rather
    than stalling on the bad endpoint.

    Unified mapping:
        ``mock_server_injector.patch_env("AIPERF_MOCK_FORCE_STATUS", "500")``
        becomes ``async with faults.inject("workload.set_env", target={"ns":
        "default", "deployment": "aiperf-mock-server"},
        env_var="AIPERF_MOCK_FORCE_STATUS", value="500"): ...``.

    Tolerances:
        - CR phase ``Completed`` within 240 s (benchmark_duration=120 s
          plus generous startup + teardown margin).
        - ``metrics.error_request_count`` average > 0 (strict check:
          forcing every response to 500 makes a zero count a real bug,
          not a tolerance issue).
        - No assertion on the exact error count because the env patch
          rolls the Deployment forward and some warmup requests may land
          before the faulty pod is Ready.
    """
    name = "chaos-b1-unified"
    try:
        await operator_ready.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )
        await waiter.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        async with faults.inject(
            "workload.set_env",
            target={"ns": MOCK_SERVER_NAMESPACE, "deployment": MOCK_SERVER_DEPLOYMENT},
            env_var="AIPERF_MOCK_FORCE_STATUS",
            value="500",
        ):
            # Wait for the rolling update to publish the faulty replica. A
            # failure here surfaces as a timeout on the Completion wait.
            await kubectl.wait_for_rollout(
                "deployment",
                MOCK_SERVER_DEPLOYMENT,
                namespace=MOCK_SERVER_NAMESPACE,
                timeout=60,
            )

            status = await operator_ready.wait_for_job_completion(
                name, operator_job_namespace, timeout=240
            )

        assert status.is_completed, (
            f"AIPerfJob {name} did not complete under forced-500 injection: "
            f"phase={status.phase}, error={status.error}"
        )

        assert status.results is not None, (
            f"AIPerfJob {name} completed but CR status.results is None; "
            "records-manager did not surface metrics"
        )
        metrics = status.results.get("metrics", {})
        error_avg = _metric_avg(metrics, "error_request_count")
        assert error_avg is not None and error_avg > 0, (
            "expected non-zero error_request_count when mock-server forces "
            f"500 on every response, got metrics={metrics}"
        )
    finally:
        await _force_delete_cr(kubectl, operator_job_namespace, name)


async def test_b2_mock_server_restart_mid_run_unified(
    operator_ready: OperatorDeployer,
    faults: InjectorRegistry,
    waiter: ChaosInjector,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """B2 unified port: force-kill mock-server pod mid-run via ``pod.kill``.

    Exercises the worker reconnect path when the upstream inference
    server flaps: ``pod.kill`` force-deletes the pod without SIGTERM,
    the Deployment recreates it, and the worker's HTTP client must
    reconnect without poisoning the benchmark.

    Unified mapping:
        ``mock_server_injector.delete_pod()`` becomes ``async with
        faults.inject("pod.kill", target={"ns": "default", "pod":
        <discovered pod name>}): ...``. The mock-server pod name is
        resolved via the same ``app=aiperf-mock-server`` label selector
        the legacy injector used internally.

    Tolerances:
        - New mock-server pod Ready within 60 s of deletion.
        - CR reaches ``Completed`` within 240 s.
        - ``metrics.request_count`` average > 0 (benchmark did not
          zero-out). Exact request count is NOT asserted because some
          in-flight requests fail during the pod flap, and that is
          expected / honest behaviour.
    """
    name = "chaos-b2-unified"
    try:
        await operator_ready.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )
        await waiter.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        mock_pod = await _resolve_mock_server_pod(kubectl, MOCK_SERVER_NAMESPACE)
        async with faults.inject(
            "pod.kill",
            target={"ns": MOCK_SERVER_NAMESPACE, "pod": mock_pod},
        ):
            # Deployment must bring a fresh pod Ready before we assert
            # completion; otherwise slow image pulls can masquerade as
            # benchmark regressions.
            ready = await kubectl.wait_for_rollout(
                "deployment",
                MOCK_SERVER_DEPLOYMENT,
                namespace=MOCK_SERVER_NAMESPACE,
                timeout=60,
            )
            assert ready, (
                f"mock-server Deployment did not roll over within 60 s of "
                f"pod delete in namespace {MOCK_SERVER_NAMESPACE}"
            )

            status = await operator_ready.wait_for_job_completion(
                name, operator_job_namespace, timeout=240
            )

        assert status.is_completed, (
            f"AIPerfJob {name} did not complete after mock-server flap: "
            f"phase={status.phase}, error={status.error}"
        )

        assert status.results is not None, (
            "records-manager did not surface metrics after mock-server flap"
        )
        metrics = status.results.get("metrics", {})
        request_avg = _metric_avg(metrics, "request_count")
        assert request_avg is not None and request_avg > 0, (
            f"expected non-zero request_count after mock-server flap, "
            f"got metrics={metrics}"
        )
    finally:
        await _force_delete_cr(kubectl, operator_job_namespace, name)


async def test_b3_mock_server_latency_injection_unified(
    operator_ready: OperatorDeployer,
    faults: InjectorRegistry,
    toxiproxy_injector: ToxiproxyInjector,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """B3 unified port: inject 500 ms latency via ``network.latency``.

    Exercises the benchmark's tolerance of slow upstreams: the worker's
    HTTP client must not time out at its default budget and the
    ``request_latency`` distribution must surface the injected delay
    rather than being clipped by a timeout cascade.

    Unified mapping:
        ``toxiproxy_injector.add_toxic("mock-server", "latency",
        attributes={...})`` becomes ``async with faults.inject(
        "network.latency", target={"proxy": "mock-server"},
        attributes={"latency": 500, "jitter": 50}): ...``. Proxy
        *creation* (``add_proxy``) remains a direct ``ToxiproxyInjector``
        call because the unified network domain only adds/removes toxics
        on existing proxies; setup is not a fault.

    Wiring:
        - Toxiproxy's ``mock-server`` proxy listens on
          ``0.0.0.0:{TOXIPROXY_MOCK_SERVER_PORT}`` (exposed by the
          fixture Service as port ``mock-server``).
        - ``AIPerfJobConfig.endpoint_url`` points at
          ``http://toxiproxy.<ns>.svc.cluster.local:20010/v1`` so every
          worker request transits toxiproxy before hitting the mock
          server.

    Tolerances:
        - ``metrics.request_latency.p99 > 400 ms`` (lower bound: injected
          downstream latency of 500 ms +/- 50 ms jitter, minus mock-server's
          own ~20 ms TTFT variance). Metrics are served in ms (see
          ``RequestLatencyMetric.display_unit``).
    """
    name = "chaos-b3-unified"
    toxiproxy_endpoint = (
        f"http://{TOXIPROXY_SERVICE}.{TOXIPROXY_NAMESPACE}.svc.cluster.local:"
        f"{TOXIPROXY_MOCK_SERVER_PORT}/v1"
    )
    try:
        await toxiproxy_injector.add_proxy(
            name="mock-server",
            listen=f"0.0.0.0:{TOXIPROXY_MOCK_SERVER_PORT}",
            upstream=(
                f"{MOCK_SERVER_DEPLOYMENT}.{MOCK_SERVER_NAMESPACE}.svc.cluster.local:"
                f"{MOCK_SERVER_SERVICE_PORT}"
            ),
        )

        cfg = AIPerfJobConfig(
            concurrency=longrun_config.concurrency,
            request_count=None,
            benchmark_duration=60.0,
            warmup_request_count=longrun_config.warmup_request_count,
            image=longrun_config.image,
            endpoint_url=toxiproxy_endpoint,
        )
        async with faults.inject(
            "network.latency",
            target={"proxy": "mock-server"},
            attributes={"latency": 500, "jitter": 50},
        ):
            await operator_ready.create_job(
                config=cfg, name=name, namespace=operator_job_namespace
            )
            status = await operator_ready.wait_for_job_completion(
                name, operator_job_namespace, timeout=300
            )

        assert status.is_completed, (
            f"AIPerfJob {name} did not complete with 500 ms toxiproxy latency "
            f"injection: phase={status.phase}, error={status.error}"
        )
        assert status.results is not None, (
            "records-manager did not surface metrics under latency injection"
        )
        metrics = status.results.get("metrics", {})
        p99 = _metric_p99(metrics, "request_latency")
        assert p99 is not None and p99 > 400.0, (
            f"expected p99 request_latency > 400 ms with 500 ms toxic "
            f"(unit=ms per RequestLatencyMetric.display_unit), "
            f"got {p99!r} from metrics={metrics}"
        )
    finally:
        await _force_delete_cr(kubectl, operator_job_namespace, name)
