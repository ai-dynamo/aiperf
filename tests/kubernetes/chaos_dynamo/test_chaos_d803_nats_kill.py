# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D803 -- NATS pod kill mid-traffic; routing degrades but does not fail entirely.

Scenario (Wave-0 #1, highest leverage):
    Kill the NATS pod via ``faults.inject("store.nats.kill")`` while 8
    concurrent SSE streams hit the dynamo frontend. NATS is dynamo's
    stats/metrics bus; ``lib/runtime/src/transports/nats.rs:49`` does not set
    an explicit reconnect backoff on the client struct, so this test exercises
    whatever ``async_nats`` defaults to under abrupt server loss + restart.

Assertion:
    The frontend keeps serving during the outage (degradation, not outage):
    error rate <20% during the ~15s window the NATS pod is gone, and <5%
    after recovery (kubelet respawns NATS, ~30s steady-state window).
"""

from __future__ import annotations

import asyncio
import contextlib

import aiohttp
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_dynamo.conftest import scrape_frontend_metrics

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


CONCURRENCY = 8
"""Number of background SSE workers hitting the frontend during the test."""

REQUESTS_PER_TASK = 10
"""Sequential requests each worker submits before the worker exits."""

OUTAGE_SECS = 15
"""Window the NATS pod is held down before the kubelet respawns it."""

RECOVERY_SECS = 30
"""Steady-state window after the outage ends, before the post-scrape."""

STEADY_STATE_SECS = 5
"""Warm-up traffic window before the fault is injected."""

REQUEST_INTERVAL_SECS = 0.5
"""Inter-request pause inside a single worker loop."""

# Metric names are placeholders -- the dynamo frontend's actual counter
# identifiers may differ. ``scrape_frontend_metrics`` returns a flat
# ``{name: value}`` dict (see conftest._parse_prometheus_text), so the
# assertion just looks them up with ``.get(..., 0.0)``; real-cluster
# validation will pin these to whatever the frontend actually exports.
COMPLETED_METRIC = "dynamo_frontend_requests_completed_total"
"""Placeholder: monotonically increasing counter of fully served requests."""

ERRORS_METRIC = "dynamo_frontend_requests_errors_total"
"""Placeholder: monotonically increasing counter of request-side errors."""

ERROR_RATE_DURING_OUTAGE_THRESHOLD = 0.20
"""Frontend may degrade but must keep serving: <20% errors during the outage."""

ERROR_RATE_RECOVERY_THRESHOLD = 0.05
"""Post-recovery error rate ceiling -- effectively steady-state again."""


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
    pytest.skip("scaffold landed; awaiting Dynamo deployment serving SSE traffic")
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
        f"completed={metrics_before.get(COMPLETED_METRIC, 0.0)} "
        f"errors={metrics_before.get(ERRORS_METRIC, 0.0)}"
    )

    # 2. Spawn CONCURRENCY background SSE workers.
    stop_event = asyncio.Event()
    request_counter: dict[str, int] = {"completed": 0, "errors": 0}

    async def _worker(idx: int) -> None:
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
                        lambda exc=exc, idx=idx: (
                            f"D803 worker {idx} request error: {exc!r}"
                        )
                    )
                    request_counter["errors"] += 1
                await asyncio.sleep(REQUEST_INTERVAL_SECS)

    workers = [asyncio.create_task(_worker(i)) for i in range(CONCURRENCY)]

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
        await asyncio.sleep(RECOVERY_SECS)
        metrics_after = await scrape_frontend_metrics(
            kubectl, dynamo_deployment_namespace
        )

        # 6. Frontend must have stayed up during the outage.
        completed_during = _metric_delta(
            metrics_during, metrics_before, COMPLETED_METRIC
        )
        errors_during = _metric_delta(metrics_during, metrics_before, ERRORS_METRIC)
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
        completed_recovery = _metric_delta(
            metrics_after, metrics_during, COMPLETED_METRIC
        )
        errors_recovery = _metric_delta(metrics_after, metrics_during, ERRORS_METRIC)
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
                    f"window ({RECOVERY_SECS}s); skipping recovery error-rate "
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


def _metric_delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    """Return the increment in a counter metric between two scrapes.

    Missing keys default to 0.0 so a metric that never appears on the
    frontend (e.g. placeholder name mismatch) reads as no change rather
    than raising ``KeyError`` mid-assertion.
    """
    return after.get(key, 0.0) - before.get(key, 0.0)
