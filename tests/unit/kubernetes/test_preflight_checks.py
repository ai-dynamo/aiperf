# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct tests for aiperf.kubernetes.preflight_checks free functions.

Complements test_preflight.py (which covers the same logic via
CLIPreflightChecker) by pinning down behavior the class-level tests miss:

- REQUIRED_RBAC_PERMISSIONS: enforce the contract of required verbs/resources
- _find_deployment: name_substring match is case-insensitive, returns (found, ready)
- check_cluster_connectivity: the free function itself (distinct from the
  CLIPreflightChecker wrapper which handles ctx-manager entry errors)
- check_kubernetes_version: mirror tests against the free function
- check_endpoint_connectivity: the short-form host.svc path (no namespace) and
  the URL-parse-error branch
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client import (
    ApiClient,
    AppsV1Api,
    CoreV1Api,
    VersionApi,
)
from kubernetes_asyncio.client.exceptions import ApiException
from kubernetes_asyncio.client.models import (
    V1Deployment,
    V1DeploymentList,
    V1DeploymentStatus,
    V1ObjectMeta,
    V1Service,
    VersionInfo,
)
from pytest import param

from aiperf.kubernetes.cr_refs import JOBSET_GROUP
from aiperf.kubernetes.preflight import CheckStatus
from aiperf.kubernetes.preflight_checks import (
    REQUIRED_RBAC_PERMISSIONS,
    _find_deployment,
    check_cluster_connectivity,
    check_endpoint_connectivity,
    check_kubernetes_version,
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


def _patch_version(mock_version: MagicMock) -> Any:
    return patch(
        "aiperf.kubernetes.preflight_checks.client.VersionApi",
        return_value=mock_version,
    )


def _make_version(major: str = "1", minor: str = "28") -> VersionInfo:
    return VersionInfo(
        build_date="2024-01-01T00:00:00Z",
        compiler="gc",
        git_commit="abc",
        git_tree_state="clean",
        git_version=f"v{major}.{minor}.0",
        go_version="go1.21",
        major=major,
        minor=minor,
        platform="linux/amd64",
    )


def _build_deployment(name: str, ready_replicas: int) -> V1Deployment:
    return V1Deployment(
        metadata=V1ObjectMeta(name=name),
        status=V1DeploymentStatus(ready_replicas=ready_replicas),
    )


# =============================================================================
# REQUIRED_RBAC_PERMISSIONS contract
# =============================================================================


class TestRequiredRbacPermissions:
    """The required-permissions table is consumed by deployment runbooks."""

    def test_is_nonempty_tuple_of_triples(self) -> None:
        assert REQUIRED_RBAC_PERMISSIONS
        for entry in REQUIRED_RBAC_PERMISSIONS:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            verb, resource, group = entry
            assert isinstance(verb, str) and verb
            assert isinstance(resource, str) and resource
            assert isinstance(group, str)

    def test_covers_pod_logs_and_configmaps(self) -> None:
        """Core CLI flows need pod-log access and configmap creation."""
        triples = {(v, r, g) for v, r, g in REQUIRED_RBAC_PERMISSIONS}
        assert ("get", "pods/log", "") in triples
        assert ("create", "configmaps", "") in triples

    def test_includes_jobset_crud(self) -> None:
        """Controller must be able to create/get/delete JobSets in its group."""
        group_verbs = {
            (v, r) for v, r, g in REQUIRED_RBAC_PERMISSIONS if g == JOBSET_GROUP
        }
        assert {"create", "get", "delete"} <= {v for v, _ in group_verbs}
        assert ("create", "jobsets") in group_verbs


# =============================================================================
# _find_deployment
# =============================================================================


class TestFindDeployment:
    """Verify substring-match semantics used by JobSet / DNS checks."""

    @pytest.mark.asyncio
    async def test_substring_match_case_insensitive(self) -> None:
        """The name compare is lowercased on the deployment side."""
        deploy = _build_deployment("JobSet-Controller-Manager", ready_replicas=1)
        apps = MagicMock(spec=AppsV1Api)
        apps.list_namespaced_deployment = AsyncMock(
            return_value=V1DeploymentList(items=[deploy])
        )

        with _patch_apps(apps):
            found, ready = await _find_deployment(
                _mock_api(), "jobset-system", "jobset"
            )
        assert found is True
        assert ready is True

    @pytest.mark.asyncio
    async def test_ready_false_when_ready_replicas_zero(self) -> None:
        deploy = _build_deployment("coredns", ready_replicas=0)
        apps = MagicMock(spec=AppsV1Api)
        apps.list_namespaced_deployment = AsyncMock(
            return_value=V1DeploymentList(items=[deploy])
        )
        with _patch_apps(apps):
            found, ready = await _find_deployment(_mock_api(), "kube-system", "coredns")
        assert found is True
        assert ready is False

    @pytest.mark.asyncio
    async def test_no_match_returns_false_false(self) -> None:
        deploy = _build_deployment("something-else", ready_replicas=3)
        apps = MagicMock(spec=AppsV1Api)
        apps.list_namespaced_deployment = AsyncMock(
            return_value=V1DeploymentList(items=[deploy])
        )
        with _patch_apps(apps):
            found, ready = await _find_deployment(_mock_api(), "ns", "jobset")
        assert (found, ready) == (False, False)

    @pytest.mark.asyncio
    async def test_empty_deployment_list(self) -> None:
        apps = MagicMock(spec=AppsV1Api)
        apps.list_namespaced_deployment = AsyncMock(
            return_value=V1DeploymentList(items=[])
        )
        with _patch_apps(apps):
            found, ready = await _find_deployment(_mock_api(), "ns", "anything")
        assert (found, ready) == (False, False)


# =============================================================================
# check_cluster_connectivity (free function)
# =============================================================================


class TestCheckClusterConnectivityFreeFn:
    """Unlike CLIPreflightChecker._check_cluster_connectivity, this function
    receives an already-open ApiClient, so failures are version-API errors."""

    @pytest.mark.asyncio
    async def test_pass(self) -> None:
        version = MagicMock(spec=VersionApi)
        version.get_code = AsyncMock(return_value=_make_version())
        with _patch_version(version):
            result = await check_cluster_connectivity(_mock_api())
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_fail_on_runtime_error(self) -> None:
        version = MagicMock(spec=VersionApi)
        version.get_code = AsyncMock(side_effect=RuntimeError("refused"))
        with _patch_version(version):
            result = await check_cluster_connectivity(_mock_api())
        assert result.status == CheckStatus.FAIL
        assert "refused" in result.message
        assert result.hints


# =============================================================================
# check_kubernetes_version (free function)
# =============================================================================


class TestCheckKubernetesVersionFreeFn:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "major,minor,expected",
        [
            param("1", "24", CheckStatus.PASS, id="min-supported"),
            param("1", "28", CheckStatus.PASS, id="recent"),
            param("2", "0", CheckStatus.PASS, id="future-major"),
            param("1", "23", CheckStatus.FAIL, id="just-below"),
            param("1", "20", CheckStatus.FAIL, id="old"),
        ],
    )  # fmt: skip
    async def test_version_thresholds(
        self, major: str, minor: str, expected: CheckStatus
    ) -> None:
        version = MagicMock(spec=VersionApi)
        version.get_code = AsyncMock(return_value=_make_version(major, minor))
        with _patch_version(version):
            result = await check_kubernetes_version(_mock_api())
        assert result.status == expected

    @pytest.mark.asyncio
    async def test_non_numeric_minor_is_sanitized(self) -> None:
        """'28+' from managed-k8s strips to '28' and passes."""
        vi = _make_version("1", "28+")
        version = MagicMock(spec=VersionApi)
        version.get_code = AsyncMock(return_value=vi)
        with _patch_version(version):
            result = await check_kubernetes_version(_mock_api())
        assert result.status == CheckStatus.PASS


# =============================================================================
# check_endpoint_connectivity (free function)
# =============================================================================


class TestCheckEndpointConnectivityFreeFn:
    @pytest.mark.asyncio
    async def test_no_url_skips(self) -> None:
        result = await check_endpoint_connectivity(_mock_api(), endpoint_url=None)
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_svc_short_form_defaults_namespace(self) -> None:
        """A bare 'svcname.svc' host (no namespace segment) falls back to 'default'."""
        core = MagicMock(spec=CoreV1Api)
        seen: dict[str, str] = {}

        async def _read_service(name: str, namespace: str, **_: Any):
            seen["name"] = name
            seen["namespace"] = namespace
            return V1Service(metadata=V1ObjectMeta(name=name))

        core.read_namespaced_service = AsyncMock(side_effect=_read_service)
        with _patch_core(core):
            result = await check_endpoint_connectivity(
                _mock_api(), endpoint_url="http://my-svc.svc:8080"
            )

        assert result.status == CheckStatus.PASS
        assert seen == {"name": "my-svc", "namespace": "default"}

    @pytest.mark.asyncio
    async def test_svc_with_namespace_uses_it(self) -> None:
        core = MagicMock(spec=CoreV1Api)
        seen: dict[str, str] = {}

        async def _read_service(name: str, namespace: str, **_: Any):
            seen["name"] = name
            seen["namespace"] = namespace
            return V1Service(metadata=V1ObjectMeta(name=name))

        core.read_namespaced_service = AsyncMock(side_effect=_read_service)
        with _patch_core(core):
            result = await check_endpoint_connectivity(
                _mock_api(),
                endpoint_url="http://llm.inference.svc.cluster.local:8080",
            )
        assert result.status == CheckStatus.PASS
        assert seen == {"name": "llm", "namespace": "inference"}

    @pytest.mark.asyncio
    async def test_svc_not_found_returns_fail(self) -> None:
        core = MagicMock(spec=CoreV1Api)
        core.read_namespaced_service = AsyncMock(side_effect=ApiException(status=404))
        with _patch_core(core):
            result = await check_endpoint_connectivity(
                _mock_api(), endpoint_url="http://gone.ns.svc:80"
            )
        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_external_url_is_info(self) -> None:
        result = await check_endpoint_connectivity(
            _mock_api(), endpoint_url="https://api.openai.com/v1"
        )
        assert result.status == CheckStatus.INFO
        assert "Port: 443" in "\n".join(result.details)

    @pytest.mark.asyncio
    async def test_url_parse_error_returns_warn(self) -> None:
        """A URL that raises in urlparse (via AttributeError from non-string)
        surfaces as WARN, not a crash."""
        result = await check_endpoint_connectivity(_mock_api(), endpoint_url=12345)  # type: ignore[arg-type]
        assert result.status == CheckStatus.WARN
        assert "Could not parse" in result.message
