# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D608 -- non-leader rank child death recovers without wedging routing."""

from __future__ import annotations

import pytest

from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.chaos_dynamo.multinode_rank_helpers import (
    find_non_leader_rank_pid,
    require_multinode_rank_topology,
    wait_for_replacement_ready,
    wait_until_chat_serves,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


async def test_d608_non_leader_rank_child_death_recovers(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_config,  # noqa: ANN001 - pytest fixture owns the concrete dataclass
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Kill a non-leader rank child process and assert replacement + serving."""
    topology = await require_multinode_rank_topology(
        kubectl, dynamo_deployment_namespace, dynamo_config, case_id="D608"
    )
    target = topology.non_leader_pod()
    if target is None:
        pytest.skip("D608: detected topology has no non-leader rank pod to kill")
    pid = await find_non_leader_rank_pid(kubectl, target, case_id="D608")
    if pid is None:
        pytest.skip(f"D608: no non-leader rank child PID found in {target.name!r}")

    async with faults.inject(
        "process.signal",
        target={
            "kind": "pod",
            "ns": target.namespace,
            "pod": target.name,
            "container": target.container,
            "pid": pid,
        },
        signal="SIGKILL",
    ):
        await wait_for_replacement_ready(kubectl, target, case_id="D608")

    await wait_until_chat_serves(dynamo_endpoint_url, case_id="D608")
