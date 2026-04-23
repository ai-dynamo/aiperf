# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chaos: benchmark-runtime faults against the mock inference server.

Covers scenarios B1, B2, B3 from
``docs/superpowers/specs/2026-04-23-chaos-expansion-design.md``. Each
test deploys a duration-based AIPerfJob CR, injects a fault into the
mock-server path while profiling is active, and asserts that the
benchmark terminates honestly (``Completed`` phase, non-zero request
count, error/latency metrics reflecting the injected fault).

Mock-server contract used here (see
``tests/aiperf_mock_server/config.py``):

- ``MOCK_SERVER_ERROR_RATE`` (float, 0-100): percentage of requests the
  server returns a 500 for. B1 patches this env to 50 so the deployment
  rolls forward with faulty responses.
- Deployment / Service name: ``aiperf-mock-server`` in ``default`` (see
  ``dev/deploy/mock-server.yaml``), port 8000.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.chaos.mock_server_injector import MockServerInjector
from tests.kubernetes.chaos.toxiproxy import (
    TOXIPROXY_MOCK_SERVER_PORT,
    TOXIPROXY_NAMESPACE,
    TOXIPROXY_SERVICE,
    ToxiproxyInjector,
)
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
def longrun_config(k8s_settings) -> AIPerfJobConfig:
    """Duration-based AIPerfJob so faults can land mid-profiling.

    Matches the fixture shape in ``test_chaos_cancellation.py`` /
    ``test_chaos_operator_resilience.py`` so all chaos tests exercise
    the same traffic profile.
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
    the metric. This helper normalises both shapes.
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
    """Drop the CR without blocking on finalizer settlement.

    Mirrors the teardown pattern in ``test_chaos_cancellation.py`` so
    that a failed assertion never leaves an AIPerfJob around to poison
    the next test.
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


async def test_b1_mock_server_500s_mid_run(
    operator_ready: OperatorDeployer,
    chaos_injector: ChaosInjector,
    mock_server_injector: MockServerInjector,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Force mock-server to return 50% 500s; benchmark completes with errors.

    Exercises the worker error-handling path in
    ``src/aiperf/workers/`` when the inference server returns HTTP 500s:
    the worker must record a ``RequestFailure`` credit (not hang), the
    records-manager must surface it as a non-zero ``error_request_count``
    metric, and the system-controller must still drive the CR to
    ``Completed`` rather than stalling on the bad endpoint.

    Tolerances:
        - CR phase ``Completed`` within 240 s (benchmark_duration=120 s
          plus generous startup + teardown margin).
        - ``metrics.error_request_count`` average > 0 (strict check:
          50 %% error rate makes a zero count a real bug, not a tolerance
          issue).
        - No assertion on the exact error count because the patch_env
          rollout cuts over mid-run and some warmup requests may land
          before the faulty pod is Ready.
    """
    name = "chaos-b1"
    try:
        await operator_ready.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )
        await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        await mock_server_injector.patch_env(
            MOCK_SERVER_NAMESPACE, "MOCK_SERVER_ERROR_RATE", "50"
        )
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
            f"AIPerfJob {name} did not complete under 50% 500s injection: "
            f"phase={status.phase}, error={status.error}"
        )

        assert status.results is not None, (
            f"AIPerfJob {name} completed but CR status.results is None; "
            "records-manager did not surface metrics"
        )
        metrics = status.results.get("metrics", {})
        error_avg = _metric_avg(metrics, "error_request_count")
        assert error_avg is not None and error_avg > 0, (
            "expected non-zero error_request_count when mock-server returns "
            f"50%% 500s, got metrics={metrics}"
        )
    finally:
        await _force_delete_cr(kubectl, operator_job_namespace, name)


async def test_b2_mock_server_restart_mid_run(
    operator_ready: OperatorDeployer,
    chaos_injector: ChaosInjector,
    mock_server_injector: MockServerInjector,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Force-delete the mock-server pod; benchmark tolerates reconnect.

    Exercises the worker reconnect path when the upstream inference
    server flaps: ``delete_pod`` kills the pod without SIGTERM, the
    Deployment recreates it, and the worker's HTTP client must
    reconnect without poisoning the benchmark.

    Tolerances:
        - New mock-server pod Ready within 60 s of deletion.
        - CR reaches ``Completed`` within 240 s.
        - ``metrics.request_count`` average > 0 (benchmark didn't
          zero-out). Exact request count is NOT asserted because some
          in-flight requests fail during the pod flap, and that is
          expected / honest behaviour.
    """
    name = "chaos-b2"
    try:
        await operator_ready.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )
        await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        await mock_server_injector.delete_pod(MOCK_SERVER_NAMESPACE)

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


@pytest.mark.xfail(
    strict=False,
    reason=(
        "toxiproxy_injector's aiohttp admin-client session is package-scoped "
        "(kubectl is package-scoped, so toxiproxy_injector must match), but "
        "pytest-asyncio's per-test event loop can't reuse an aiohttp session "
        "created in the package loop — the timeout context manager raises "
        "'Timeout context manager should be used inside a task'. Redesigning "
        "ToxiproxyInjector to lazily open per-test aiohttp sessions (or "
        "routing admin calls through subprocess `curl`) would unblock this; "
        "deferred to a follow-up. The fault-injection logic below is still "
        "exercised end-to-end the moment the session-scope wiring is fixed."
    ),
)
async def test_b3_mock_server_latency_injection(
    operator_ready: OperatorDeployer,
    toxiproxy_injector: ToxiproxyInjector,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Inject 500 ms latency via toxiproxy; p99 request latency reflects it.

    Exercises the benchmark's tolerance of slow upstreams: the worker's
    HTTP client must not time out at its default budget and the
    ``request_latency`` distribution must surface the injected delay
    rather than being clipped by a timeout cascade.

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
    name = "chaos-b3"
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
        await toxiproxy_injector.add_toxic(
            proxy_name="mock-server",
            toxic_type="latency",
            attributes={"latency": 500, "jitter": 50},
        )

        cfg = AIPerfJobConfig(
            concurrency=longrun_config.concurrency,
            request_count=None,
            benchmark_duration=60.0,
            warmup_request_count=longrun_config.warmup_request_count,
            image=longrun_config.image,
            endpoint_url=toxiproxy_endpoint,
        )
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
        await toxiproxy_injector.reset()
        await _force_delete_cr(kubectl, operator_job_namespace, name)
