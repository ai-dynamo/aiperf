# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for :py:class:`ClusterInjector`.

Mocks the underlying :py:class:`KubectlClient` so these tests can run on a
laptop without any Kubernetes cluster: they verify dispatch, precondition
handling, sweep-cache recording, and the manifest / patch shapes that
``cluster.network_policy.deny_egress`` and ``cluster.rbac.revoke`` emit.

These tests intentionally do NOT assert that NetworkPolicy actually blocks
egress; that requires a NetworkPolicy-aware CNI (Cilium/Calico) and lives
behind the ``cilium_on_kind_required`` mark in the live chaos suite. Here
we assert only that the correct manifest is APPLIED and that restore
deletes it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from tests.kubernetes.chaos_common.base import (
    FaultMechanismError,
    FaultPreconditionError,
    FaultSpec,
)
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


# ---------------------------------------------------------------------------
# cluster.network_policy.deny_egress
# ---------------------------------------------------------------------------


def _netpol_spec(
    name: str = "deny-egress",
    namespace: str = "aiperf-test-netpol",
    allow_cluster_egress: bool | None = True,
) -> FaultSpec:
    params: dict[str, Any] = {"name": name}
    if allow_cluster_egress is not None:
        params["allow_cluster_egress"] = allow_cluster_egress
    return FaultSpec(
        fault_id="cluster.network_policy.deny_egress",
        params=params,
        target={"ns": namespace},
    )


@pytest.mark.asyncio
async def test_network_policy_apply_uses_kubectl_apply_with_correct_manifest() -> None:
    kubectl = _make_kubectl_mock()
    injector = ClusterInjector(kubectl)

    with patch(
        "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
    ):
        await injector.inject(_netpol_spec(name="np1", namespace="aiperf-netpol"))

    assert kubectl.apply.await_count == 1
    manifest_str, *_ = kubectl.apply.await_args.args
    manifest = orjson.loads(manifest_str)
    assert manifest["apiVersion"] == "networking.k8s.io/v1"
    assert manifest["kind"] == "NetworkPolicy"
    assert manifest["metadata"]["name"] == "np1"
    assert manifest["metadata"]["namespace"] == "aiperf-netpol"
    assert manifest["spec"]["podSelector"] == {}
    assert manifest["spec"]["policyTypes"] == ["Egress"]
    # allow_cluster_egress defaults to True -> egress list present
    assert manifest["spec"]["egress"] == [{"to": [{"namespaceSelector": {}}]}]


@pytest.mark.asyncio
async def test_network_policy_records_mutation_for_sweep() -> None:
    kubectl = _make_kubectl_mock()
    injector = ClusterInjector(kubectl)

    with patch(
        "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
    ) as record:
        await injector.inject(_netpol_spec(name="np2", namespace="aiperf-netpol-2"))

    assert record.call_count == 1
    mutation = record.call_args.args[0]
    assert mutation.kind == "networkpolicy"
    assert mutation.api_version == "networking.k8s.io/v1"
    assert mutation.name == "np2"
    assert mutation.namespace == "aiperf-netpol-2"
    assert mutation.op == "create"


@pytest.mark.asyncio
async def test_network_policy_restore_deletes_with_ignore_not_found() -> None:
    kubectl = _make_kubectl_mock()
    injector = ClusterInjector(kubectl)

    with patch(
        "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
    ):
        applied = await injector.inject(
            _netpol_spec(name="np3", namespace="aiperf-netpol-3")
        )

    await applied.restore()

    assert kubectl.run.await_count == 1
    delete_args = kubectl.run.await_args.args
    assert delete_args[0] == "delete"
    assert delete_args[1] == "networkpolicy"
    assert delete_args[2] == "np3"
    assert "-n" in delete_args
    assert "aiperf-netpol-3" in delete_args
    assert "--ignore-not-found" in delete_args


@pytest.mark.asyncio
async def test_network_policy_missing_name_raises_precondition() -> None:
    kubectl = _make_kubectl_mock()
    injector = ClusterInjector(kubectl)
    spec = FaultSpec(
        fault_id="cluster.network_policy.deny_egress",
        params={"allow_cluster_egress": True},
        target={"ns": "aiperf-test-netpol"},
    )

    with (
        patch(
            "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
        ),
        pytest.raises(FaultPreconditionError, match="name"),
    ):
        await injector.inject(spec)

    kubectl.apply.assert_not_awaited()


@pytest.mark.asyncio
async def test_network_policy_allow_cluster_egress_false_drops_egress_rules() -> None:
    kubectl = _make_kubectl_mock()
    injector = ClusterInjector(kubectl)

    with patch(
        "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
    ):
        await injector.inject(
            _netpol_spec(
                name="np-default-deny",
                namespace="aiperf-deny",
                allow_cluster_egress=False,
            )
        )

    assert kubectl.apply.await_count == 1
    manifest_str, *_ = kubectl.apply.await_args.args
    manifest = orjson.loads(manifest_str)
    # Default-deny: no `egress` block, just policyTypes=[Egress].
    assert "egress" not in manifest["spec"]
    assert manifest["spec"]["policyTypes"] == ["Egress"]


# ---------------------------------------------------------------------------
# cluster.rbac.revoke
# ---------------------------------------------------------------------------


def _role_json(
    *,
    rules: list[dict[str, Any]] | None = None,
) -> bytes:
    """Return a kubectl-style ``-o json`` role/clusterrole body."""
    body = {
        "kind": "Role",
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "metadata": {"name": "test-role"},
        "rules": rules
        if rules is not None
        else [
            {
                "apiGroups": [""],
                "resources": ["configmaps"],
                "verbs": ["get", "list", "watch"],
            }
        ],
    }
    return orjson.dumps(body)


def _rbac_kubectl_mock(role_body: bytes | None = None) -> MagicMock:
    """Kubectl mock where ``run("get", ...)`` returns a role JSON."""
    kubectl = _make_kubectl_mock()
    body = role_body if role_body is not None else _role_json()

    async def _run(*args: str, **_: Any) -> MagicMock:
        if args and args[0] == "get":
            return MagicMock(returncode=0, stdout=body.decode(), stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    kubectl.run = AsyncMock(side_effect=_run)
    return kubectl


@pytest.mark.asyncio
async def test_rbac_revoke_drops_verb_via_json_patch() -> None:
    kubectl = _rbac_kubectl_mock()
    injector = ClusterInjector(kubectl)
    spec = FaultSpec(
        fault_id="cluster.rbac.revoke",
        target={"scope": "role", "name": "test-role", "ns": "aiperf-rbac"},
        params={"api_group": "", "resource": "configmaps", "verb": "watch"},
    )

    with patch(
        "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
    ):
        await injector.inject(spec)

    # Two run() calls: get, then patch.
    assert kubectl.run.await_count == 2
    get_call = kubectl.run.await_args_list[0]
    patch_call = kubectl.run.await_args_list[1]

    assert get_call.args[0] == "get"
    assert get_call.args[1] == "role"
    assert get_call.args[2] == "test-role"
    assert "-n" in get_call.args and "aiperf-rbac" in get_call.args
    assert "-o" in get_call.args and "json" in get_call.args

    assert patch_call.args[0] == "patch"
    assert patch_call.args[1] == "role"
    assert patch_call.args[2] == "test-role"
    assert "--type=json" in patch_call.args
    # Last positional arg is the patch JSON.
    patch_idx = patch_call.args.index("-p") + 1
    patch_obj = orjson.loads(patch_call.args[patch_idx])
    assert patch_obj == [{"op": "remove", "path": "/rules/0/verbs/2"}]


@pytest.mark.asyncio
async def test_rbac_revoke_clusterrole_requires_no_namespace() -> None:
    kubectl = _rbac_kubectl_mock()
    injector = ClusterInjector(kubectl)
    spec = FaultSpec(
        fault_id="cluster.rbac.revoke",
        target={"scope": "clusterrole", "name": "test-role"},
        params={"api_group": "", "resource": "configmaps", "verb": "watch"},
    )

    with patch(
        "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
    ):
        await injector.inject(spec)

    # Neither call carries -n / namespace.
    for call in kubectl.run.await_args_list:
        assert "-n" not in call.args

    # Explicitly providing ns for clusterrole is rejected.
    bad_spec = FaultSpec(
        fault_id="cluster.rbac.revoke",
        target={
            "scope": "clusterrole",
            "name": "test-role",
            "ns": "should-not-be-here",
        },
        params={"api_group": "", "resource": "configmaps", "verb": "watch"},
    )
    with (
        patch(
            "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
        ),
        pytest.raises(FaultPreconditionError, match="cluster-scoped"),
    ):
        await injector.inject(bad_spec)


@pytest.mark.asyncio
async def test_rbac_revoke_role_requires_namespace() -> None:
    kubectl = _rbac_kubectl_mock()
    injector = ClusterInjector(kubectl)
    spec = FaultSpec(
        fault_id="cluster.rbac.revoke",
        target={"scope": "role", "name": "test-role"},  # no ns
        params={"api_group": "", "resource": "configmaps", "verb": "watch"},
    )

    with (
        patch(
            "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
        ),
        pytest.raises(FaultPreconditionError, match="ns"),
    ):
        await injector.inject(spec)


@pytest.mark.asyncio
async def test_rbac_revoke_records_mutation() -> None:
    kubectl = _rbac_kubectl_mock()
    injector = ClusterInjector(kubectl)
    spec = FaultSpec(
        fault_id="cluster.rbac.revoke",
        target={"scope": "role", "name": "test-role", "ns": "aiperf-rbac"},
        params={"api_group": "", "resource": "configmaps", "verb": "watch"},
    )

    with patch(
        "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
    ) as record:
        await injector.inject(spec)

    assert record.call_count == 1
    mutation = record.call_args.args[0]
    assert mutation.kind == "role"
    assert mutation.api_version == "rbac.authorization.k8s.io/v1"
    assert mutation.name == "test-role"
    assert mutation.op == "patch"
    assert mutation.namespace == "aiperf-rbac"
    assert mutation.payload["rule_index"] == 0
    assert mutation.payload["verb"] == "watch"


@pytest.mark.asyncio
async def test_rbac_revoke_restore_adds_verb_back() -> None:
    kubectl = _rbac_kubectl_mock()
    injector = ClusterInjector(kubectl)
    spec = FaultSpec(
        fault_id="cluster.rbac.revoke",
        target={"scope": "role", "name": "test-role", "ns": "aiperf-rbac"},
        params={"api_group": "", "resource": "configmaps", "verb": "watch"},
    )

    with patch(
        "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
    ):
        applied = await injector.inject(spec)

    # Reset count, then restore.
    pre_restore_calls = kubectl.run.await_count
    await applied.restore()
    assert kubectl.run.await_count == pre_restore_calls + 1

    restore_call = kubectl.run.await_args_list[-1]
    assert restore_call.args[0] == "patch"
    assert restore_call.args[1] == "role"
    assert restore_call.args[2] == "test-role"
    assert "-n" in restore_call.args and "aiperf-rbac" in restore_call.args
    assert "--type=json" in restore_call.args
    patch_idx = restore_call.args.index("-p") + 1
    patch_obj = orjson.loads(restore_call.args[patch_idx])
    assert patch_obj == [{"op": "add", "path": "/rules/0/verbs/-", "value": "watch"}]


@pytest.mark.asyncio
async def test_rbac_revoke_verb_not_found_raises_precondition() -> None:
    # Role has the resource but not the verb -> precondition.
    body = _role_json(
        rules=[
            {
                "apiGroups": [""],
                "resources": ["configmaps"],
                "verbs": ["get", "list"],
            }
        ]
    )
    kubectl = _rbac_kubectl_mock(role_body=body)
    injector = ClusterInjector(kubectl)
    spec = FaultSpec(
        fault_id="cluster.rbac.revoke",
        target={"scope": "role", "name": "test-role", "ns": "aiperf-rbac"},
        params={"api_group": "", "resource": "configmaps", "verb": "watch"},
    )

    with (
        patch(
            "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
        ),
        pytest.raises(FaultPreconditionError, match="locate verb"),
    ):
        await injector.inject(spec)


@pytest.mark.asyncio
async def test_unknown_cluster_fault_id_raises_precondition() -> None:
    kubectl = _make_kubectl_mock()
    injector = ClusterInjector(kubectl)
    spec = FaultSpec(fault_id="cluster.does_not_exist")
    with pytest.raises(FaultPreconditionError, match="does not implement"):
        await injector.inject(spec)


@pytest.mark.asyncio
async def test_network_policy_apply_failure_raises_mechanism_error() -> None:
    kubectl = _make_kubectl_mock()
    kubectl.apply = AsyncMock(side_effect=RuntimeError("apiserver 500"))
    injector = ClusterInjector(kubectl)

    with (
        patch(
            "tests.kubernetes.chaos_common.injectors.cluster.recovery.record_mutation"
        ),
        pytest.raises(FaultMechanismError, match="NetworkPolicy"),
    ):
        await injector.inject(_netpol_spec(name="np-fail", namespace="ns-fail"))
