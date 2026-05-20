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

import pytest

from aiperf.common.aiperf_logger import AIPerfLogger

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


async def test_d803_nats_kill_mid_traffic(
    faults,  # noqa: ANN001 - InjectorRegistry, see conftest.py
    kubectl,  # noqa: ANN001 - KubectlClient, see tests.kubernetes.conftest
    dynamo_endpoint_url,  # noqa: ANN001 - str, see gpu.dynamo.conftest
    dynamo_deployment_namespace,  # noqa: ANN001 - str, see chaos_dynamo.conftest
) -> None:
    """Kill NATS under 8 concurrent SSE streams; assert degradation not outage.

    NATS is dynamo's stats/metrics bus. ``nats.rs:49`` has no explicit reconnect
    backoff overrides; this test exercises whatever ``async_nats`` defaults to.

    Implementation outline (pending real-cluster validation):

    1. Snapshot ``metrics_before = await scrape_frontend_metrics(kubectl,
       dynamo_deployment_namespace)``. Track ``completed_total``,
       ``errors_total`` counters.
    2. Spawn 8 concurrent SSE POST tasks against ``dynamo_endpoint_url +
       "/chat/completions"``, each in a loop that submits sequential requests
       (e.g. 10 per task = 80 total over the test window).
    3. Wait 5s for steady-state traffic.
    4. ``async with faults.inject("store.nats.kill", grace_period=0):``
        - Fault is instantaneous -- pod kill returns when kubectl returns.
        - ``await asyncio.sleep(15)`` to let traffic run through the outage.
        - ``metrics_during = await scrape_frontend_metrics(...)``.
    5. After context exits (no-op restore; kubelet respawns NATS), wait 30s
       for steady state. ``metrics_after = await scrape_frontend_metrics(...)``.
    6. Compute and assert::

        completed_during = metrics_during["completed_total"] - metrics_before["completed_total"]
        errors_during    = metrics_during["errors_total"] - metrics_before["errors_total"]
        assert completed_during > 0, "frontend stopped serving during NATS outage"
        error_rate_during = errors_during / max(completed_during, 1)
        assert error_rate_during < 0.20, f"error rate {error_rate_during:.1%} >20% during outage"

        completed_recovery = metrics_after["completed_total"] - metrics_during["completed_total"]
        errors_recovery    = metrics_after["errors_total"] - metrics_during["errors_total"]
        error_rate_recovery = errors_recovery / max(completed_recovery, 1)
        assert error_rate_recovery < 0.05, f"recovery error rate {error_rate_recovery:.1%}"

    7. Cancel the 8 background tasks in ``finally``.
    """
    pytest.skip("scaffold landed; assertion-body pending real-cluster validation")
