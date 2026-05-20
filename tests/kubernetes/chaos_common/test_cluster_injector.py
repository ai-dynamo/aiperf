# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for :py:class:`ClusterInjector`.

Mocks the underlying :py:class:`KubectlClient` so these tests can run on a
laptop without any Kubernetes cluster: they verify dispatch, precondition
handling, sweep-cache recording, and Phase-3 stub markers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.kubernetes.chaos_common.base import FaultPreconditionError, FaultSpec
from tests.kubernetes.chaos_common.injectors.cluster import ClusterInjector


def _make_kubectl_mock() -> MagicMock:
    """Build a :py:class:`KubectlClient` test double with async methods."""
    kubectl = MagicMock()
    kubectl.apply = AsyncMock(return_value="resourcequota/test-quota created")
    kubectl.run = AsyncMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
    return kubectl


def _quota_spec(
    name: str = "test-quota",
    namespace: str = "aiperf-test-cluster",
    hard_limits: dict[str, str] | None = None,
) -> FaultSpec:
    return FaultSpec(
        fault_id="cluster.resource_quota",
        params={
            "name": name,
            "hard_limits": hard_limits or {"requests.memory": "512Mi"},
        },
        target={"ns": namespace},
    )


@pytest.mark.asyncio
async def test_resource_quota_apply_then_restore_deletes_it() -> None:
    kubectl = _make_kubectl_mock()
    injector = ClusterInjector(kubectl)

    with patch(
        "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
    ):
        applied = await injector.inject(_quota_spec())

    assert kubectl.apply.await_count == 1
    # `delete_resource_quota` shells out via `kubectl.run("delete", ...)`.
    assert kubectl.run.await_count == 0

    await applied.restore()

    assert kubectl.run.await_count == 1
    delete_args = kubectl.run.await_args.args
    assert delete_args[0] == "delete"
    assert delete_args[1] == "resourcequota"
    assert delete_args[2] == "test-quota"
    assert "-n" in delete_args
    assert "aiperf-test-cluster" in delete_args
    assert "--ignore-not-found" in delete_args


@pytest.mark.asyncio
async def test_resource_quota_records_mutation_for_sweep() -> None:
    kubectl = _make_kubectl_mock()
    injector = ClusterInjector(kubectl)

    with patch(
        "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
    ) as record:
        await injector.inject(
            _quota_spec(name="quota-x", namespace="aiperf-test-sweep")
        )

    assert record.call_count == 1
    mutation = record.call_args.args[0]
    assert mutation.kind == "resourcequota"
    assert mutation.api_version == "v1"
    assert mutation.name == "quota-x"
    assert mutation.namespace == "aiperf-test-sweep"
    assert mutation.op == "create"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec_kwargs,missing_substring",
    [
        pytest.param(
            {
                "params": {"name": "q", "hard_limits": {"memory": "1Gi"}},
                "target": {},
            },
            "ns",
            id="missing-ns",
        ),
        pytest.param(
            {
                "params": {"hard_limits": {"memory": "1Gi"}},
                "target": {"ns": "aiperf-test-x"},
            },
            "name",
            id="missing-name",
        ),
        pytest.param(
            {
                "params": {"name": "q"},
                "target": {"ns": "aiperf-test-x"},
            },
            "hard_limits",
            id="missing-hard-limits",
        ),
    ],
)  # fmt: skip
async def test_missing_hard_limits_raises_precondition(
    spec_kwargs: dict[str, Any], missing_substring: str
) -> None:
    kubectl = _make_kubectl_mock()
    injector = ClusterInjector(kubectl)
    spec = FaultSpec(fault_id="cluster.resource_quota", **spec_kwargs)

    with (
        patch(
            "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
        ),
        pytest.raises(FaultPreconditionError, match=missing_substring),
    ):
        await injector.inject(spec)

    kubectl.apply.assert_not_awaited()


@pytest.mark.asyncio
async def test_network_policy_stub_raises_not_implemented() -> None:
    injector = ClusterInjector(_make_kubectl_mock())
    spec = FaultSpec(fault_id="cluster.network_policy.deny_egress")

    with pytest.raises(NotImplementedError, match="Phase 3"):
        await injector.inject(spec)


@pytest.mark.asyncio
async def test_rbac_stub_raises_not_implemented() -> None:
    injector = ClusterInjector(_make_kubectl_mock())
    spec = FaultSpec(fault_id="cluster.rbac.revoke")

    with pytest.raises(NotImplementedError, match="Phase 3"):
        await injector.inject(spec)


def test_handles_prefix_match_cluster() -> None:
    assert ClusterInjector.handles("cluster") is True
    assert ClusterInjector.handles("cluster.resource_quota") is True
    assert ClusterInjector.handles("cluster.network_policy.deny_egress") is True
    assert ClusterInjector.handles("cluster.rbac.revoke") is True
    assert ClusterInjector.handles("pod") is False
    assert ClusterInjector.handles("network") is False
    assert ClusterInjector.handles("store") is False
    # Must not false-match a string that merely starts with "cluster" without
    # the dot boundary (e.g. a hypothetical sibling "clustering.foo").
    assert ClusterInjector.handles("clustering.foo") is False
