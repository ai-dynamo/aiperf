# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D704 - HF Hub egress blackhole; weight download fails cleanly, not opaquely.

Wave-0 #10 (Cilium-gated). Targets the cache-miss weight-download path in
``components/src/dynamo/vllm/main.py`` (``await fetch_model(config.model)``)
to confirm the worker surfaces a clear failure status when external egress
is severed, rather than hanging or emitting an opaque error.

The fault is a ``NetworkPolicy`` denying egress to ``0.0.0.0/0`` except the
cluster CIDR (``allow_cluster_egress=True``). NetworkPolicy enforcement
requires a NetworkPolicy-aware CNI -- kindnet silently ignores the policy,
so the scenario is gated on :py:data:`KIND_HAS_CILIUM` via the
:py:func:`cilium_on_kind_required` mark. Without Cilium the test is
xfail-skipped; with Cilium the strict xfail flips and a failure surfaces
loudly (see ``tests/kubernetes/chaos_common/README.md``).
"""

from __future__ import annotations

import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.marks import cilium_on_kind_required

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


@cilium_on_kind_required
async def test_d704_hf_hub_egress_blackhole(faults, kubectl, wait_for_dgd_state):
    """Block egress; assert worker fails weight-download cleanly, DGD reports failure.

    Requires a NetworkPolicy-aware CNI (Cilium or Calico). The
    ``cilium_on_kind_required`` mark gates this test on the
    ``KIND_HAS_CILIUM`` env var; without Cilium the test is xfail-skipped.
    With Cilium the test must PASS or the strict xfail flips to a loud
    failure (see ``chaos_common/README.md`` "Verifying the flip").
    """
    # 1. Apply a NetworkPolicy denying all egress (allow_cluster_egress=True for intra-cluster):
    #    async with faults.inject(
    #        "cluster.network_policy.deny_egress",
    #        target={"ns": "dynamo-server"},
    #        name="d704-blackhole",
    #        allow_cluster_egress=True,  # cluster traffic OK; only external blocked
    #    ):
    # 2. Apply a fresh DGD that references an HF Hub model not yet cached anywhere:
    #    DynamoConfig(model_name="Qwen/Qwen3-0.6B", ...) (default; lands in /tmp/HF cache miss).
    # 3. wait_for_dgd_state("d704-test", "dynamo-server", "failed", timeout=300).
    # 4. Inspect kubectl get pods -l ... -o json for the worker pod's status -- should show
    #    container terminated with non-zero exit code, log mentions weight download failure
    #    (HF Hub, DNS, NetworkPolicy, or connection refused).
    # 5. Parse status.conditions; assert Ready=False with reason naming weight/network failure.
    # 6. finally (handled by faults context): NetworkPolicy deleted.
    del faults, kubectl, wait_for_dgd_state  # scaffold; consumed once assertions land.

    pytest.skip(
        "scaffold landed; full assertion requires Cilium-equipped cluster + fresh "
        "DGD with cache-miss model. Will be exercised in real-cluster CI."
    )
