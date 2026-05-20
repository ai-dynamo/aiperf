# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D611 -- filling /dev/shm in one rank is cleaned up and recoverable."""

from __future__ import annotations

import pytest

from tests.kubernetes.chaos_dynamo.multinode_rank_helpers import (
    cleanup_dev_shm,
    fill_dev_shm,
    require_multinode_rank_topology,
    send_chat_completion,
    wait_until_chat_serves,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


async def test_d611_dev_shm_fill_recovers_after_cleanup(
    kubectl: KubectlClient,
    dynamo_config,  # noqa: ANN001 - pytest fixture owns the concrete dataclass
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Fill /dev/shm in a rank pod, remove the file, and assert serving resumes."""
    topology = await require_multinode_rank_topology(
        kubectl, dynamo_deployment_namespace, dynamo_config, case_id="D611"
    )
    target = topology.non_leader_pod() or topology.pods[0]

    await send_chat_completion(dynamo_endpoint_url, case_id="D611-warmup")
    filled = await fill_dev_shm(kubectl, target, case_id="D611")
    if not filled:
        pytest.skip(f"D611: unable to fill /dev/shm in {target.name!r}")
    try:
        await cleanup_dev_shm(kubectl, target)
    finally:
        await cleanup_dev_shm(kubectl, target)

    await wait_until_chat_serves(dynamo_endpoint_url, case_id="D611")
