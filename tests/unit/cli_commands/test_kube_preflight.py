# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CLIPreflightChecker.run_quick_checks()."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client import (
    ApiClient,
    ApiextensionsV1Api,
    VersionApi,
)
from kubernetes_asyncio.client.exceptions import ApiException
from kubernetes_asyncio.client.models import (
    V1CustomResourceDefinition,
    V1ObjectMeta,
    VersionInfo,
)

from aiperf.kubernetes.preflight import CheckStatus, CLIPreflightChecker

# =============================================================================
# Fixtures
# =============================================================================


def _version_info() -> VersionInfo:
    return VersionInfo(
        build_date="2024-01-01T00:00:00Z",
        compiler="gc",
        git_commit="abc",
        git_tree_state="clean",
        git_version="v1.28.0",
        go_version="go1.21",
        major="1",
        minor="28",
        platform="linux/amd64",
    )


def _mock_rbac_allowed(allowed: bool) -> MagicMock:
    review = MagicMock()
    review.status = MagicMock()
    review.status.allowed = allowed
    authz = MagicMock()
    authz.create_self_subject_access_review = AsyncMock(return_value=review)
    return authz


@asynccontextmanager
async def _yields(api):
    yield api


@pytest.fixture
def mock_kube_env():
    """Fixture that provides a factory for patching kubernetes_asyncio calls.

    Returns a context-manager factory that patches k8s_client + the typed
    V1 API constructors used by CLIPreflightChecker.
    """

    @asynccontextmanager
    async def _mocks(
        *,
        jobset_crd_error: Exception | None = None,
        rbac_allowed: bool = True,
        connectivity_error: Exception | None = None,
    ):
        api = MagicMock(spec=ApiClient)

        # k8s_client context manager
        if connectivity_error is not None:

            @asynccontextmanager
            async def _boom(**kwargs):
                raise connectivity_error
                yield  # noqa: F841, RET503

            client_patch = patch(
                "aiperf.kubernetes.client.k8s_client",
                new=_boom,
            )
        else:
            client_patch = patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_yields(api),
            )

        # VersionApi.get_code
        version = MagicMock(spec=VersionApi)
        version.get_code = AsyncMock(return_value=_version_info())

        # ApiextensionsV1Api.read_custom_resource_definition
        apiext = MagicMock(spec=ApiextensionsV1Api)
        if jobset_crd_error is not None:
            apiext.read_custom_resource_definition = AsyncMock(
                side_effect=jobset_crd_error
            )
        else:
            apiext.read_custom_resource_definition = AsyncMock(
                return_value=V1CustomResourceDefinition(
                    metadata=V1ObjectMeta(name="jobsets.jobset.x-k8s.io"),
                    spec=MagicMock(),
                )
            )

        authz = _mock_rbac_allowed(rbac_allowed)

        with (
            client_patch,
            patch(
                "aiperf.kubernetes.preflight.client.VersionApi",
                return_value=version,
            ),
            patch(
                "aiperf.kubernetes.preflight.client.ApiextensionsV1Api",
                return_value=apiext,
            ),
            patch(
                "aiperf.kubernetes.preflight_utils.client.AuthorizationV1Api",
                return_value=authz,
            ),
        ):
            yield api

    return _mocks


# =============================================================================
# Quick Checks Tests
# =============================================================================


class TestQuickChecks:
    """Tests for CLIPreflightChecker.run_quick_checks()."""

    @pytest.mark.asyncio
    async def test_quick_checks_passes_healthy_cluster(self, mock_kube_env) -> None:
        """Test quick checks pass on a healthy cluster."""
        async with mock_kube_env():
            checker = CLIPreflightChecker(namespace="default")
            results = await checker.run_quick_checks()

        assert results.passed is True
        assert len(results.checks) == 3

    @pytest.mark.asyncio
    async def test_quick_checks_fails_on_connectivity(self, mock_kube_env) -> None:
        """Test quick checks short-circuit on connectivity failure."""
        async with mock_kube_env(
            connectivity_error=Exception("connection refused"),
        ):
            checker = CLIPreflightChecker(namespace="default")
            results = await checker.run_quick_checks()

        assert results.passed is False
        assert len(results.checks) == 1
        assert results.checks[0].name == "Cluster Connectivity"
        assert results.checks[0].status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_quick_checks_fails_on_jobset_crd(self, mock_kube_env) -> None:
        """Test quick checks fail when JobSet CRD is missing."""
        async with mock_kube_env(jobset_crd_error=ApiException(status=404)):
            checker = CLIPreflightChecker(namespace="default")
            results = await checker.run_quick_checks()

        assert results.passed is False
        crd_check = next(c for c in results.checks if c.name == "JobSet CRD")
        assert crd_check.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_quick_checks_only_runs_three_checks(self, mock_kube_env) -> None:
        """Test that quick checks run exactly 3 checks on success."""
        async with mock_kube_env():
            checker = CLIPreflightChecker(namespace="default")
            results = await checker.run_quick_checks()

        assert len(results.checks) == 3
        check_names = [c.name for c in results.checks]
        assert check_names == [
            "Cluster Connectivity",
            "JobSet CRD",
            "RBAC Permissions",
        ]

    @pytest.mark.asyncio
    async def test_quick_checks_fails_on_rbac(self, mock_kube_env) -> None:
        """Test quick checks fail when RBAC permissions are denied."""
        async with mock_kube_env(rbac_allowed=False):
            checker = CLIPreflightChecker(namespace="default")
            results = await checker.run_quick_checks()

        assert results.passed is False
        rbac_check = next(c for c in results.checks if c.name == "RBAC Permissions")
        assert rbac_check.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_quick_checks_with_endpoint_runs_four_checks(
        self, mock_kube_env
    ) -> None:
        """Test that quick checks include endpoint when endpoint_url is set."""
        async with mock_kube_env():
            checker = CLIPreflightChecker(
                namespace="default", endpoint_url="http://llm:8000/v1"
            )
            results = await checker.run_quick_checks()

        assert len(results.checks) == 4
        check_names = [c.name for c in results.checks]
        assert check_names == [
            "Cluster Connectivity",
            "JobSet CRD",
            "RBAC Permissions",
            "Endpoint Connectivity",
        ]

    @pytest.mark.asyncio
    async def test_check_results_have_duration(self, mock_kube_env) -> None:
        """Test that all check results have duration_ms populated."""
        async with mock_kube_env():
            checker = CLIPreflightChecker(namespace="default")
            results = await checker.run_quick_checks()

        for check in results.checks:
            assert check.duration_ms is not None
            assert check.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_quick_checks_does_not_print(self, mock_kube_env, capsys) -> None:
        """Test that quick checks do not print anything to stdout by default."""
        async with mock_kube_env():
            checker = CLIPreflightChecker(namespace="default")
            await checker.run_quick_checks()

        captured = capsys.readouterr()
        assert captured.out == ""

    @pytest.mark.asyncio
    async def test_quick_checks_show_progress_prints_output(
        self, mock_kube_env, capsys
    ) -> None:
        """Test that quick checks print compact results when show_progress=True."""
        async with mock_kube_env():
            checker = CLIPreflightChecker(namespace="default")
            await checker.run_quick_checks(show_progress=True)

        captured = capsys.readouterr()
        assert "Cluster Connectivity" in captured.out
        assert "JobSet CRD" in captured.out
        assert "RBAC Permissions" in captured.out
