# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D614 -- incompatible replacement rank image surfaces as ImagePullBackOff."""

from __future__ import annotations

import pytest

from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.chaos_dynamo.multinode_rank_helpers import (
    require_multinode_rank_topology,
    set_owner_image,
    wait_for_image_pull_failure,
    wait_until_chat_serves,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_BAD_IMAGE = "nonexistent.example.com/dynamo-rank-incompatible:d614"


async def test_d614_incompatible_rank_image_surfaces_pull_failure(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_config,  # noqa: ANN001 - pytest fixture owns the concrete dataclass
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Patch a rank owner to a bad image, kill one rank, and assert pull failure."""
    topology = await require_multinode_rank_topology(
        kubectl, dynamo_deployment_namespace, dynamo_config, case_id="D614"
    )
    target = topology.non_leader_pod()
    if target is None:
        pytest.skip("D614: detected topology has no non-leader rank pod")

    patched = await set_owner_image(kubectl, target, _BAD_IMAGE, case_id="D614")
    if not patched:
        pytest.skip(
            f"D614: cannot patch image on owner {target.owner_kind}/{target.owner_name}"
        )
    try:
        async with faults.inject(
            "pod.kill",
            target={"ns": target.namespace, "pod": target.name},
        ):
            failed_pod = await wait_for_image_pull_failure(
                kubectl, target.namespace, case_id="D614"
            )
        assert failed_pod, "D614: expected replacement pod to hit image-pull failure"
    finally:
        restored = await set_owner_image(
            kubectl, target, target.image, case_id="D614-restore"
        )
        assert restored, (
            f"D614: failed to restore image on {target.owner_kind}/{target.owner_name}; "
            "manual cluster repair required"
        )

    await wait_until_chat_serves(dynamo_endpoint_url, case_id="D614")
