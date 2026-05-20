# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D712 -- RBAC denial while creating child Role/RoleBinding surfaces in status."""

from __future__ import annotations

import pytest

from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.chaos_dynamo.test_chaos_d112_deployment_create_rbac import (
    _apply_fresh_dgd,
    _delete_dgd,
    _find_unique_operator_rbac_owner,
    _observe_not_successful,
    _rbac_target,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


async def test_d712_rbac_child_rolebinding_create_denial_blocks_then_recovers(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    """Revoke child RoleBinding create RBAC before apply, then restore and recover."""
    owner = await _find_unique_operator_rbac_owner(
        kubectl,
        api_group="rbac.authorization.k8s.io",
        resource="rolebindings",
        verb="create",
        case_id="D712",
    )
    faults = request.getfixturevalue("faults")
    name = ""
    namespace = dynamo_deployment_namespace
    try:
        async with faults.inject(
            "cluster.rbac.revoke",
            target=_rbac_target(owner),
            api_group="rbac.authorization.k8s.io",
            resource="rolebindings",
            verb="create",
        ):
            name, namespace = await _apply_fresh_dgd(kubectl, namespace)
            await _observe_not_successful(kubectl, name, namespace, case_id="D712")
            authz = await kubectl.run(
                "auth",
                "can-i",
                "create",
                "rolebindings.rbac.authorization.k8s.io",
                "--as=system:serviceaccount:dynamo-system:dynamo-operator",
                "-n",
                namespace,
                check=False,
            )
            assert authz.stdout.strip() == "no", (
                "D712: RBAC revoke did not remove rolebindings/create from "
                f"{owner.label}; auth can-i returned {authz.stdout!r}"
            )
        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)
    finally:
        if name:
            await _delete_dgd(kubectl, name, namespace)
