# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D609 -- NCCL/rank port block is bounded and reversible."""

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


async def test_d609_nccl_port_block_recovers_after_policy_restore(
    kubectl: KubectlClient,
    dynamo_config,  # noqa: ANN001 - pytest fixture owns the concrete dataclass
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Deny ingress to a non-leader rank pod, then restore and assert serving."""
    topology = await require_multinode_rank_topology(
        kubectl, dynamo_deployment_namespace, dynamo_config, case_id="D609"
    )
    target = topology.non_leader_pod()
    if target is None:
        pytest.skip("D609: detected topology has no non-leader rank pod")

    await send_chat_completion(dynamo_endpoint_url, case_id="D609-warmup")
    async with temporary_network_policy(
        kubectl,
        target,
        name="d609-nccl-port-block",
        policy_spec={"policyTypes": ["Ingress"], "ingress": []},
    ):
        await asyncio.sleep(10.0)

    await wait_until_chat_serves(dynamo_endpoint_url, case_id="D609")
