# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D613 -- deleting a placement object is reconciled or fails loudly."""

from __future__ import annotations

import pytest

from tests.kubernetes.chaos_dynamo.multinode_rank_helpers import (
    find_placement_object,
    require_multinode_rank_topology,
    wait_for_object_recreated,
    wait_until_chat_serves,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


async def test_d613_placement_object_delete_is_reconciled(
    kubectl: KubectlClient,
    dynamo_config,  # noqa: ANN001 - pytest fixture owns the concrete dataclass
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Delete a Grove/LWS scheduling object and assert controller reconciliation."""
    await require_multinode_rank_topology(
        kubectl, dynamo_deployment_namespace, dynamo_config, case_id="D613"
    )
    placement = await find_placement_object(kubectl, dynamo_deployment_namespace)
    if placement is None:
        pytest.skip("D613: no PodGroup/LeaderWorkerSet placement object detected")
    resource, name = placement
    original = await kubectl.run(
        "get",
        resource,
        name,
        "-n",
        dynamo_deployment_namespace,
        "-o",
        "yaml",
        check=True,
    )

    try:
        await kubectl.run(
            "delete",
            resource,
            name,
            "-n",
            dynamo_deployment_namespace,
            "--ignore-not-found",
            check=True,
        )
        await wait_for_object_recreated(
            kubectl,
            dynamo_deployment_namespace,
            resource,
            name,
            case_id="D613",
        )
    finally:
        current = await kubectl.run(
            "get",
            resource,
            name,
            "-n",
            dynamo_deployment_namespace,
            check=False,
        )
        if current.returncode != 0:
            await kubectl.apply(original.stdout)

    await wait_until_chat_serves(dynamo_endpoint_url, case_id="D613")
