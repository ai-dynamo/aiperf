# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D610 -- stale rank DNS / isolated resolver path does not permanently wedge."""

from __future__ import annotations

import asyncio

import pytest

from tests.kubernetes.chaos_dynamo.multinode_rank_helpers import (
    require_multinode_rank_topology,
    send_chat_completion,
    temporary_network_policy,
    wait_until_chat_serves,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


async def test_d610_stale_dns_rank_recovers_after_dns_egress_restore(
    kubectl: KubectlClient,
    dynamo_config,  # noqa: ANN001 - pytest fixture owns the concrete dataclass
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Block rank DNS egress long enough to stale peer lookup, then restore."""
    topology = await require_multinode_rank_topology(
        kubectl, dynamo_deployment_namespace, dynamo_config, case_id="D610"
    )
    target = topology.non_leader_pod()
    if target is None:
        pytest.skip("D610: detected topology has no non-leader rank pod")

    await send_chat_completion(dynamo_endpoint_url, case_id="D610-warmup")
    async with temporary_network_policy(
        kubectl,
        target,
        name="d610-stale-dns-rank",
        policy_spec={"policyTypes": ["Egress"], "egress": []},
    ):
        await asyncio.sleep(10.0)

    await wait_until_chat_serves(dynamo_endpoint_url, case_id="D610")
