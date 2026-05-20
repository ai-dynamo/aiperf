# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D612 -- rank-local clock skew is bounded and reversible when supported."""

from __future__ import annotations

import os

import pytest

from tests.kubernetes.chaos_dynamo.multinode_rank_helpers import (
    maybe_skew_clock,
    require_multinode_rank_topology,
    restore_clock,
    send_chat_completion,
    wait_until_chat_serves,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


async def test_d612_clock_skew_recovers_after_clock_restore(
    kubectl: KubectlClient,
    dynamo_config,  # noqa: ANN001 - pytest fixture owns the concrete dataclass
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Skew one rank clock under explicit opt-in, restore it, and assert serving."""
    if os.environ.get("DYNAMO_CHAOS_ALLOW_CLOCK_SKEW") != "1":
        pytest.skip("D612: set DYNAMO_CHAOS_ALLOW_CLOCK_SKEW=1 to allow clock mutation")

    topology = await require_multinode_rank_topology(
        kubectl, dynamo_deployment_namespace, dynamo_config, case_id="D612"
    )
    target = topology.non_leader_pod() or topology.pods[0]

    await send_chat_completion(dynamo_endpoint_url, case_id="D612-warmup")
    skewed = await maybe_skew_clock(kubectl, target, seconds=120)
    if not skewed:
        pytest.skip(f"D612: {target.name!r} lacks permission to skew CLOCK_REALTIME")
    try:
        await wait_until_chat_serves(
            dynamo_endpoint_url, case_id="D612-skewed", timeout=60.0
        )
    finally:
        await restore_clock(kubectl, target)

    await wait_until_chat_serves(dynamo_endpoint_url, case_id="D612")
