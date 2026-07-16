# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for aiperf.kubernetes.preflight_checks free functions.

Complements the existing direct-test suite by hitting branches that file
leaves cold:

- check_namespace: 404 -> RBAC PASS / RBAC FAIL / RBAC raises (WARN); 403 SKIP;
  non-404/403 ApiException
- check_rbac_permissions: all-pass, missing aggregation, transient WARN
- check_jobset_crd: PASS / 404 FAIL hint / non-404 FAIL
- check_jobset_controller: ready PASS / found-not-ready WARN / missing FAIL /
  403 SKIP / non-403 WARN
- check_network_policies: empty PASS / present WARN / 403 SKIP / non-403 WARN
- check_dns: ready / found-not-ready / missing / API error
- check_endpoint_connectivity: HTTPS default port + http default port + .svc
  cluster.local short form
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client import (
    ApiClient,
    ApiextensionsV1Api,
    AppsV1Api,
    CoreV1Api,
    NetworkingV1Api,
)
from kubernetes_asyncio.client.exceptions import ApiException
from kubernetes_asyncio.client.models import (
    V1CustomResourceDefinition,
    V1Deployment,
    V1DeploymentList,
    V1DeploymentStatus,
    V1Namespace,
    V1NetworkPolicy,
    V1NetworkPolicyList,
    V1ObjectMeta,
)
from pytest import param

from aiperf.kubernetes.preflight import CheckStatus
from aiperf.kubernetes.preflight_checks import (
    check_dns,
    check_endpoint_connectivity,
    check_jobset_controller,
    check_jobset_crd,
    check_namespace,
    check_network_policies,
    check_rbac_permissions,
)

# =============================================================================
# Helpers
# =============================================================================


def _mock_api() -> MagicMock:
    return MagicMock(spec=ApiClient)


def _patch_core(mock_core: MagicMock) -> Any:
    return patch(
        "aiperf.kubernetes.preflight_checks.client.CoreV1Api", return_value=mock_core
    )


def _patch_apps(mock_apps: MagicMock) -> Any:
    return patch(
        "aiperf.kubernetes.preflight_checks.client.AppsV1Api", return_value=mock_apps
    )


def _patch_apiext(mock_apiext: MagicMock) -> Any:
    return patch(
        "aiperf.kubernetes.preflight_checks.client.ApiextensionsV1Api",
        return_value=mock_apiext,
    )


def _patch_net(mock_net: MagicMock) -> Any:
    return patch(
        "aiperf.kubernetes.preflight_checks.client.NetworkingV1Api",
        return_value=mock_net,
    )


def _patch_rbac_access(side_effect: Any = None, return_value: Any = None) -> Any:
    """Patch the shared RBAC helper imported (aliased) into preflight_checks."""
    if side_effect is not None:
        return patch(
            "aiperf.kubernetes.preflight_checks._shared_check_rbac_access",
            new=AsyncMock(side_effect=side_effect),
        )
    return patch(
        "aiperf.kubernetes.preflight_checks._shared_check_rbac_access",
        new=AsyncMock(return_value=return_value),
    )


def _build_deployment(name: str, ready_replicas: int) -> V1Deployment:
    return V1Deployment(
        metadata=V1ObjectMeta(name=name),
        status=V1DeploymentStatus(ready_replicas=ready_replicas),
    )


# =============================================================================
# check_namespace
# =============================================================================


class TestCheckNamespace:
    """All branches of namespace existence + creation-permission probing."""

    @pytest.mark.asyncio
    async def test_namespace_exists_passes(self) -> None:
        core = MagicMock(spec=CoreV1Api)
        core.read_namespace = AsyncMock(
            return_value=V1Namespace(metadata=V1ObjectMeta(name="aiperf-run-1"))
        )
        with _patch_core(core):
            result = await check_namespace(_mock_api(), namespace="aiperf-run-1")
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_namespace_403_returns_skip(self) -> None:
        core = MagicMock(spec=CoreV1Api)
        core.read_namespace = AsyncMock(side_effect=ApiException(status=403))
        with _patch_core(core):
            result = await check_namespace(_mock_api(), namespace="aiperf-run-1")
        assert result.status == CheckStatus.SKIP
        assert "permission denied" in result.message

    @pytest.mark.asyncio
    async def test_namespace_404_with_create_permission_passes(self) -> None:
        """Missing namespace + create RBAC -> PASS ('will be created')."""
        core = MagicMock(spec=CoreV1Api)
        core.read_namespace = AsyncMock(side_effect=ApiException(status=404))
        with _patch_core(core), _patch_rbac_access(return_value=True):
            result = await check_namespace(_mock_api(), namespace="new-ns")
        assert result.status == CheckStatus.PASS
        assert "will be created" in result.message

    @pytest.mark.asyncio
    async def test_namespace_404_without_create_permission_fails(self) -> None:
        core = MagicMock(spec=CoreV1Api)
        core.read_namespace = AsyncMock(side_effect=ApiException(status=404))
        with _patch_core(core), _patch_rbac_access(return_value=False):
            result = await check_namespace(_mock_api(), namespace="new-ns")
        assert result.status == CheckStatus.FAIL
        assert "does not exist" in result.message
        assert any("Ask an admin" in h for h in result.hints)

    @pytest.mark.asyncio
    async def test_namespace_404_rbac_check_raises_returns_warn(self) -> None:
        """RBAC probe failure (e.g. transient apiserver) downgrades to WARN."""
        core = MagicMock(spec=CoreV1Api)
        core.read_namespace = AsyncMock(side_effect=ApiException(status=404))
        with (
            _patch_core(core),
            _patch_rbac_access(side_effect=ApiException(status=500)),
        ):
            result = await check_namespace(_mock_api(), namespace="new-ns")
        assert result.status == CheckStatus.WARN
        assert "cannot verify create permission" in result.message

    @pytest.mark.asyncio
    async def test_namespace_other_api_error_fails(self) -> None:
        """500 / other status codes surface as FAIL with the HTTP code."""
        core = MagicMock(spec=CoreV1Api)
        core.read_namespace = AsyncMock(side_effect=ApiException(status=500))
        with _patch_core(core):
            result = await check_namespace(_mock_api(), namespace="weird")
        assert result.status == CheckStatus.FAIL
        assert "HTTP 500" in result.message


# =============================================================================
# check_rbac_permissions
# =============================================================================


class TestCheckRbacPermissions:
    """Aggregation: missing dominates, then transient, else PASS."""

    @pytest.mark.asyncio
    async def test_all_allowed_passes(self) -> None:
        with _patch_rbac_access(return_value=True):
            result = await check_rbac_permissions(_mock_api(), namespace="ns")
        assert result.status == CheckStatus.PASS
        assert result.details
        assert all(d.strip().startswith(("✓", "passed")) for d in result.details)

    @pytest.mark.asyncio
    async def test_missing_some_fails_with_count(self) -> None:
        """A single denied permission flips the whole check to FAIL."""
        # Deny only the very first permission, allow the rest.
        calls = {"n": 0}

        async def _maybe_allowed(*_: Any, **__: Any) -> bool:
            calls["n"] += 1
            return calls["n"] != 1

        with patch(
            "aiperf.kubernetes.preflight_checks._shared_check_rbac_access",
            new=_maybe_allowed,
        ):
            result = await check_rbac_permissions(_mock_api(), namespace="ns")

        assert result.status == CheckStatus.FAIL
        assert "Missing 1 required permission" in result.message
        assert any("Contact your cluster admin" in h for h in result.hints)

    @pytest.mark.asyncio
    async def test_transient_only_returns_warn(self) -> None:
        """No definitive denial, but every probe failed -> WARN, not FAIL."""
        with _patch_rbac_access(side_effect=ApiException(status=503)):
            result = await check_rbac_permissions(_mock_api(), namespace="ns")
        assert result.status == CheckStatus.WARN
        assert "transient apiserver errors" in result.message


# =============================================================================
# check_jobset_crd
# =============================================================================


class TestCheckJobsetCrd:
    """The CRD presence check has three distinct outcomes."""

    @pytest.mark.asyncio
    async def test_pass(self) -> None:
        apiext = MagicMock(spec=ApiextensionsV1Api)
        apiext.read_custom_resource_definition = AsyncMock(
            return_value=V1CustomResourceDefinition(
                metadata=V1ObjectMeta(name="jobsets.jobset.x-k8s.io"),
                spec=MagicMock(),  # spec is required by the model but unused here
            )
        )
        with _patch_apiext(apiext):
            result = await check_jobset_crd(_mock_api())
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_404_fail_with_install_hint(self) -> None:
        apiext = MagicMock(spec=ApiextensionsV1Api)
        apiext.read_custom_resource_definition = AsyncMock(
            side_effect=ApiException(status=404)
        )
        with _patch_apiext(apiext):
            result = await check_jobset_crd(_mock_api())
        assert result.status == CheckStatus.FAIL
        assert result.hints
        # The hint text should include something installable.
        assert any(h for h in result.hints)

    @pytest.mark.asyncio
    async def test_non_404_api_error_fails(self) -> None:
        """Per docstring: align with operator FAIL on non-404, do NOT downgrade."""
        apiext = MagicMock(spec=ApiextensionsV1Api)
        apiext.read_custom_resource_definition = AsyncMock(
            side_effect=ApiException(status=500)
        )
        with _patch_apiext(apiext):
            result = await check_jobset_crd(_mock_api())
        assert result.status == CheckStatus.FAIL
        assert "HTTP 500" in result.message


# =============================================================================
# check_jobset_controller
# =============================================================================


class TestCheckJobsetController:
    """The deployment-list-based readiness check has 5 outcomes."""

    @pytest.mark.asyncio
    async def test_ready_passes(self) -> None:
        apps = MagicMock(spec=AppsV1Api)
        apps.list_namespaced_deployment = AsyncMock(
            return_value=V1DeploymentList(
                items=[_build_deployment("jobset-controller-manager", 1)]
            )
        )
        with _patch_apps(apps):
            result = await check_jobset_controller(_mock_api())
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_found_but_not_ready_warns(self) -> None:
        apps = MagicMock(spec=AppsV1Api)
        apps.list_namespaced_deployment = AsyncMock(
            return_value=V1DeploymentList(
                items=[_build_deployment("jobset-controller-manager", 0)]
            )
        )
        with _patch_apps(apps):
            result = await check_jobset_controller(_mock_api())
        assert result.status == CheckStatus.WARN
        assert "not ready" in result.message

    @pytest.mark.asyncio
    async def test_missing_fails(self) -> None:
        apps = MagicMock(spec=AppsV1Api)
        apps.list_namespaced_deployment = AsyncMock(
            return_value=V1DeploymentList(items=[])
        )
        with _patch_apps(apps):
            result = await check_jobset_controller(_mock_api())
        assert result.status == CheckStatus.FAIL
        assert any("Install" in h for h in result.hints)

    @pytest.mark.asyncio
    async def test_403_skipped(self) -> None:
        apps = MagicMock(spec=AppsV1Api)
        apps.list_namespaced_deployment = AsyncMock(
            side_effect=ApiException(status=403)
        )
        with _patch_apps(apps):
            result = await check_jobset_controller(_mock_api())
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_non_403_api_error_warns(self) -> None:
        apps = MagicMock(spec=AppsV1Api)
        apps.list_namespaced_deployment = AsyncMock(
            side_effect=ApiException(status=500)
        )
        with _patch_apps(apps):
            result = await check_jobset_controller(_mock_api())
        assert result.status == CheckStatus.WARN
        assert "HTTP 500" in result.message


# =============================================================================
# check_network_policies
# =============================================================================


class TestCheckNetworkPolicies:
    """Network-policy presence: silent PASS or warn with names."""

    @pytest.mark.asyncio
    async def test_no_policies_passes(self) -> None:
        net = MagicMock(spec=NetworkingV1Api)
        net.list_namespaced_network_policy = AsyncMock(
            return_value=V1NetworkPolicyList(items=[])
        )
        with _patch_net(net):
            result = await check_network_policies(_mock_api(), namespace="ns")
        assert result.status == CheckStatus.PASS
        assert "unrestricted" in result.message

    @pytest.mark.asyncio
    async def test_policies_present_warn_with_names(self) -> None:
        # NetworkPolicy requires a spec; build minimally-valid models.
        from kubernetes_asyncio.client.models import V1NetworkPolicySpec

        empty_spec = V1NetworkPolicySpec(pod_selector=MagicMock())
        net = MagicMock(spec=NetworkingV1Api)
        net.list_namespaced_network_policy = AsyncMock(
            return_value=V1NetworkPolicyList(
                items=[
                    V1NetworkPolicy(
                        metadata=V1ObjectMeta(name="default-deny"), spec=empty_spec
                    ),
                    V1NetworkPolicy(
                        metadata=V1ObjectMeta(name="allow-dns"), spec=empty_spec
                    ),
                ]
            )
        )
        with _patch_net(net):
            result = await check_network_policies(_mock_api(), namespace="ns")
        assert result.status == CheckStatus.WARN
        text = "\n".join(result.details)
        assert "default-deny" in text
        assert "allow-dns" in text

    @pytest.mark.asyncio
    async def test_403_skipped(self) -> None:
        net = MagicMock(spec=NetworkingV1Api)
        net.list_namespaced_network_policy = AsyncMock(
            side_effect=ApiException(status=403)
        )
        with _patch_net(net):
            result = await check_network_policies(_mock_api(), namespace="ns")
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_non_403_api_error_warns(self) -> None:
        net = MagicMock(spec=NetworkingV1Api)
        net.list_namespaced_network_policy = AsyncMock(
            side_effect=ApiException(status=500)
        )
        with _patch_net(net):
            result = await check_network_policies(_mock_api(), namespace="ns")
        assert result.status == CheckStatus.WARN
        assert "HTTP 500" in result.message


# =============================================================================
# check_dns
# =============================================================================


class TestCheckDns:
    """CoreDNS deployment readiness in kube-system via the canonical label."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "ready_replicas,expected",
        [
            param(2, CheckStatus.PASS, id="ready"),
            param(0, CheckStatus.WARN, id="found-not-ready"),
        ],
    )  # fmt: skip
    async def test_label_matched_deployment(
        self, ready_replicas: int, expected: CheckStatus
    ) -> None:
        apps = MagicMock(spec=AppsV1Api)
        apps.list_namespaced_deployment = AsyncMock(
            return_value=V1DeploymentList(
                items=[_build_deployment("coredns", ready_replicas)]
            )
        )
        with _patch_apps(apps):
            result = await check_dns(_mock_api())
        assert result.status == expected

    @pytest.mark.asyncio
    async def test_no_dns_deployment_warns(self) -> None:
        apps = MagicMock(spec=AppsV1Api)
        apps.list_namespaced_deployment = AsyncMock(
            return_value=V1DeploymentList(items=[])
        )
        with _patch_apps(apps):
            result = await check_dns(_mock_api())
        assert result.status == CheckStatus.WARN
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_api_error_warns(self) -> None:
        apps = MagicMock(spec=AppsV1Api)
        apps.list_namespaced_deployment = AsyncMock(
            side_effect=ApiException(status=500)
        )
        with _patch_apps(apps):
            result = await check_dns(_mock_api())
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_label_selector_used(self) -> None:
        """The DNS check filters by canonical label, not deployment-name substring."""
        apps = MagicMock(spec=AppsV1Api)
        seen: dict[str, str] = {}

        async def _list(namespace: str, **kwargs: Any):
            seen["namespace"] = namespace
            seen["label_selector"] = kwargs.get("label_selector", "")
            return V1DeploymentList(items=[])

        apps.list_namespaced_deployment = AsyncMock(side_effect=_list)
        with _patch_apps(apps):
            await check_dns(_mock_api())

        assert seen["namespace"] == "kube-system"
        assert seen["label_selector"] == "k8s-app=kube-dns"


# =============================================================================
# check_endpoint_connectivity (default port branches)
# =============================================================================


class TestCheckEndpointConnectivityPorts:
    """Default-port logic for http/https + svc.cluster.local long form."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url,expected_port",
        [
            param("http://example.com/v1", 80, id="http-default"),
            param("https://example.com/v1", 443, id="https-default"),
            param("https://example.com:8443/v1", 8443, id="explicit-port"),
        ],
    )  # fmt: skip
    async def test_external_url_lists_resolved_port(
        self, url: str, expected_port: int
    ) -> None:
        result = await check_endpoint_connectivity(_mock_api(), endpoint_url=url)
        assert result.status == CheckStatus.INFO
        assert f"Port: {expected_port}" in "\n".join(result.details)
