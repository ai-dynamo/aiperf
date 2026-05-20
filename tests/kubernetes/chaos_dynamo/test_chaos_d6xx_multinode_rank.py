# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D6xx Dynamo multinode/rank chaos scenarios."""

from __future__ import annotations

import asyncio
import os

import pytest

from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.chaos_dynamo.multinode_rank_helpers import (
    cleanup_dev_shm,
    fill_dev_shm,
    find_non_leader_rank_pid,
    find_placement_object,
    maybe_skew_clock,
    require_multinode_rank_topology,
    restore_clock,
    send_chat_completion,
    set_owner_image,
    temporary_network_policy,
    wait_for_image_pull_failure,
    wait_for_object_recreated,
    wait_for_replacement_ready,
    wait_until_chat_serves,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


# D608


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


# D609


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


# D610


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


# D611


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


# D612


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


# D613


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


# D614

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
