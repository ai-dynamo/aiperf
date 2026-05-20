# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D802 -- 30s etcd timeout via Toxiproxy; frontend recovers without outage.

Scenario (Wave-0 #2):
    Inject a Toxiproxy ``timeout`` toxic with ``timeout=0`` (drop all traffic)
    on the etcd proxy for 30 seconds, then remove it. Targets the keep-alive
    heartbeat in ``lib/runtime/src/transports/etcd/lease.rs:136`` which renews
    the lease at TTL/2 -- pausing etcd long enough to skip one heartbeat but
    short enough that the 60s lease has not yet expired by the time traffic
    resumes.

Assertion:
    Frontend continues to serve a stale roster during the pause window, then
    recovers within ~90s of pause-start (60s lease TTL + 30s timeout). No
    permanent outage; ``completed_total`` is monotonically non-decreasing
    across the during -> after window.

Pre-condition:
    A Toxiproxy proxy named ``etcd`` must exist, forwarding
    ``0.0.0.0:20030`` -> ``<release>-etcd:2379``. Port 20030 is reserved by
    ``tests/kubernetes/chaos_common/fixtures/toxiproxy.yaml``. For Wave-0,
    this test creates the proxy inline via
    ``dynamo_toxiproxy.add_proxy(...)`` and removes it in ``finally``;
    a session-scoped fixture is a follow-up.
"""

from __future__ import annotations

import pytest

from aiperf.common.aiperf_logger import AIPerfLogger

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

ETCD_UPSTREAM = "dynamo-platform-etcd-headless.dynamo-system.svc:2379"
"""Upstream etcd Service the chart deploys (dynamo-platform release)."""

ETCD_PROXY_NAME = "etcd"
"""Toxiproxy proxy name; must match the reserved entry in toxiproxy.yaml."""

ETCD_PROXY_LISTEN = "0.0.0.0:20030"
"""Listen address inside the toxiproxy pod; port 20030 is reserved for etcd."""

PAUSE_SECONDS = 30.0
"""Duration of the etcd-traffic blackout (timeout toxic with ``timeout=0``)."""

RECOVERY_SECONDS = 60.0
"""Lease-TTL-driven recovery budget after the toxic is removed (TTL = 60s)."""


async def test_d802_etcd_30s_pause_recovers(
    faults,  # noqa: ANN001 - InjectorRegistry, see conftest.py
    dynamo_toxiproxy,  # noqa: ANN001 - ToxiproxyInjector, see conftest.py
    kubectl,  # noqa: ANN001 - KubectlClient, see tests.kubernetes.conftest
    dynamo_endpoint_url,  # noqa: ANN001 - str, see gpu.dynamo.conftest
) -> None:
    """Pause etcd traffic 30s via Toxiproxy; assert lease-TTL-bounded recovery.

    Targets ``lease.rs:136`` (keep-alive at TTL/2). NOTE: this test requires
    the dynamo operator + workers be configured to route etcd through
    Toxiproxy via a ``KUBERNETES_SERVICE_HOST``-style override OR a sidecar
    redirect. For Wave-0 scaffold, the test asserts only that the toxic is
    applied and removed -- the actual operator-routing is a follow-up
    (see Wave-5 in ``chaos_dynamo/README.md``).

    Implementation outline (pending operator etcd-via-toxiproxy redirect
    wiring + ``scrape_frontend_metrics`` integration):

    1. Set up the toxiproxy proxy::

        await dynamo_toxiproxy.add_proxy(
            ETCD_PROXY_NAME, ETCD_PROXY_LISTEN, ETCD_UPSTREAM,
        )

    2. Snapshot metrics ``before`` via
       ``scrape_frontend_metrics(kubectl, namespace=...)``.
    3. Inject the timeout toxic; sleep for the pause window::

        async with faults.inject(
            "store.etcd.timeout",
            target={"proxy": ETCD_PROXY_NAME},
            attributes={"timeout": 0},  # 0 = drop all traffic
        ):
            await asyncio.sleep(PAUSE_SECONDS)
            metrics_during = await scrape_frontend_metrics(kubectl, ns)

    4. Wait ``RECOVERY_SECONDS`` for lease-TTL-driven recovery + reconnect::

        await asyncio.sleep(RECOVERY_SECONDS)
        metrics_after = await scrape_frontend_metrics(kubectl, ns)

    5. Assert frontend made progress between ``metrics_during`` and
       ``metrics_after`` (``completed_total`` monotonically non-decreasing;
       some recovery occurred).
    6. ``finally``: ``await dynamo_toxiproxy.remove_proxy(ETCD_PROXY_NAME)``
       (the ``faults`` fixture teardown also calls ``reset()`` which removes
       every proxy, so this is belt-and-braces).
    """
    pytest.skip(
        "scaffold landed; full assertion needs operator etcd-via-toxiproxy "
        "redirect wiring (follow-up: see Wave-5 in chaos_dynamo/README.md)"
    )
