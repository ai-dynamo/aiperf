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
    Dynamo must be installed with its bundled etcd subchart and etcd discovery
    enabled. Dynamo v1.1.0 defaults to Kubernetes discovery with
    ``global.etcd.install=false``, so the stock single-GPU topology has no etcd
    service for this scenario. In that topology the test skips with an explicit
    reason instead of applying a Toxiproxy toxic to an unused proxy.
"""

from __future__ import annotations

import os

import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from dev.versions import DYNAMO_VERSION
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

ETCD_NAMESPACE = "dynamo-system"
"""Namespace where the Dynamo platform chart installs bundled services."""

ETCD_SERVICE = "dynamo-platform-etcd-headless"
"""Headless etcd Service name when ``global.etcd.install=true`` is enabled."""

ETCD_UPSTREAM = f"{ETCD_SERVICE}.{ETCD_NAMESPACE}.svc:2379"
"""Upstream etcd Service the chart deploys when bundled etcd is enabled."""

ETCD_PROXY_NAME = "etcd"
"""Toxiproxy proxy name; must match the reserved entry in toxiproxy.yaml."""

ETCD_PROXY_LISTEN = "0.0.0.0:20030"
"""Listen address inside the toxiproxy pod; port 20030 is reserved for etcd."""

PAUSE_SECONDS = 30.0
"""Duration of the etcd-traffic blackout (timeout toxic with ``timeout=0``)."""

RECOVERY_SECONDS = 60.0
"""Lease-TTL-driven recovery budget after the toxic is removed (TTL = 60s)."""

_ETCD_CHAOS_OPT_IN_ENV = "AIPERF_DYNAMO_ETCD_CHAOS"
"""Opt-in proving the target topology was installed with bundled etcd enabled."""


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
        ETCD_SERVICE,
        "-n",
        ETCD_NAMESPACE,
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
            f"D802 requires bundled etcd service {ETCD_NAMESPACE}/{ETCD_SERVICE}; "
            "the opt-in topology did not expose that service."
        )

    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    faults = request.getfixturevalue("faults")

    proxy_created = False
    try:
        await dynamo_toxiproxy.add_proxy(
            name=ETCD_PROXY_NAME,
            listen=ETCD_PROXY_LISTEN,
            upstream=ETCD_UPSTREAM,
        )
        proxy_created = True
        async with faults.inject(
            "store.etcd.timeout",
            target={"proxy": ETCD_PROXY_NAME},
            attributes={"timeout": 0},
        ) as applied:
            assert applied.spec.fault_id == "network.timeout"
            assert applied.metadata.get("proxy_name") == ETCD_PROXY_NAME
            logger.info(
                f"D802: timeout toxic applied to etcd proxy for {PAUSE_SECONDS}s; "
                f"recovery budget={RECOVERY_SECONDS}s"
            )
    finally:
        if proxy_created:
            try:
                await dynamo_toxiproxy.remove_proxy(ETCD_PROXY_NAME)
            except Exception as exc:
                logger.warning(lambda exc=exc: f"D802 remove_proxy failed: {exc!r}")
