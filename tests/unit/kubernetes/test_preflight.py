# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.preflight module.

Focuses on:
- CheckResult / PreflightResults dataclass behavior
- PreflightChecker individual check methods (mocked k8s API)
- Quick-check and full-check orchestration (short-circuit, ordering)
- Error handling for all k8s API failure modes
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import kr8s
import pytest
from pytest import param

from aiperf.kubernetes.preflight import (
    CheckResult,
    CheckStatus,
    PreflightChecker,
    PreflightResults,
    _format_duration,
)
from tests.harness.k8s import (
    async_list,
    create_not_found_error,
    create_server_error,
    make_kr8s_object,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_checker(**overrides: Any) -> PreflightChecker:
    """Create a PreflightChecker with sensible defaults, overridable per-test."""
    defaults: dict[str, Any] = {
        "namespace": "test-ns",
        "kubeconfig": None,
        "kube_context": None,
        "image": None,
        "image_pull_secrets": None,
        "secrets": None,
        "endpoint_url": None,
        "workers": 1,
    }
    defaults.update(overrides)
    return PreflightChecker(**defaults)


def _mock_api() -> MagicMock:
    """Build a MagicMock kr8s API with commonly-needed async stubs."""
    api = MagicMock(spec=kr8s.Api)
    api.async_version = AsyncMock(
        return_value={"major": "1", "minor": "28", "gitVersion": "v1.28.0"}
    )
    api.async_get = AsyncMock(return_value=[])
    return api


def _mock_call_api_response(json_body: dict[str, Any], status_code: int = 200):
    """Return an async context-manager mock for api.call_api."""
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()

    @asynccontextmanager
    async def _ctx(*args, **kwargs):
        yield resp

    return _ctx


def _mock_call_api_raises(exc: Exception):
    """Return an async context-manager mock for api.call_api that raises."""

    @asynccontextmanager
    async def _ctx(*args, **kwargs):
        raise exc
        yield  # noqa: F841, RET503

    return _ctx


def _node_raw(
    name: str,
    cpu: str,
    memory: str,
    ready: bool = True,
) -> dict[str, Any]:
    """Build a minimal Node .raw dict."""
    conditions = [{"type": "Ready", "status": "True" if ready else "False"}]
    return {
        "metadata": {"name": name, "namespace": ""},
        "status": {
            "conditions": conditions,
            "allocatable": {"cpu": cpu, "memory": memory},
        },
    }


# =============================================================================
# CheckStatus enum
# =============================================================================


class TestCheckStatus:
    """Verify CheckStatus string enum values."""

    @pytest.mark.parametrize(
        "member,value",
        [
            (CheckStatus.PASS, "pass"),
            (CheckStatus.FAIL, "fail"),
            (CheckStatus.WARN, "warn"),
            (CheckStatus.SKIP, "skip"),
            (CheckStatus.INFO, "info"),
        ],
    )  # fmt: skip
    def test_check_status_values(self, member: CheckStatus, value: str) -> None:
        assert member == value
        assert isinstance(member, str)


# =============================================================================
# PreflightResults
# =============================================================================


class TestPreflightResults:
    """Verify aggregation logic on PreflightResults."""

    def test_empty_results_passed(self) -> None:
        """No checks means nothing failed."""
        assert PreflightResults().passed is True

    def test_empty_results_no_warnings(self) -> None:
        assert PreflightResults().has_warnings is False

    def test_passed_true_when_all_pass(self) -> None:
        results = PreflightResults()
        results.add(CheckResult("a", CheckStatus.PASS, "ok"))
        results.add(CheckResult("b", CheckStatus.WARN, "watch out"))
        results.add(CheckResult("c", CheckStatus.INFO, "fyi"))
        assert results.passed is True

    def test_passed_false_when_any_fail(self) -> None:
        results = PreflightResults()
        results.add(CheckResult("a", CheckStatus.PASS, "ok"))
        results.add(CheckResult("b", CheckStatus.FAIL, "broken"))
        assert results.passed is False

    def test_has_warnings_true(self) -> None:
        results = PreflightResults()
        results.add(CheckResult("a", CheckStatus.WARN, "hmm"))
        assert results.has_warnings is True

    def test_has_warnings_false_with_only_pass(self) -> None:
        results = PreflightResults()
        results.add(CheckResult("a", CheckStatus.PASS, "ok"))
        assert results.has_warnings is False

    def test_add_appends_to_checks(self) -> None:
        results = PreflightResults()
        r1 = CheckResult("a", CheckStatus.PASS, "ok")
        r2 = CheckResult("b", CheckStatus.FAIL, "bad")
        results.add(r1)
        results.add(r2)
        assert results.checks == [r1, r2]


# =============================================================================
# _format_duration
# =============================================================================


class TestFormatDuration:
    """Verify duration formatting helper."""

    def test_format_duration_none_returns_empty(self) -> None:
        assert _format_duration(None) == ""

    def test_format_duration_value_returns_formatted(self) -> None:
        assert _format_duration(123.456) == " (123ms)"

    def test_format_duration_zero(self) -> None:
        assert _format_duration(0.0) == " (0ms)"


# =============================================================================
# CheckResult dataclass
# =============================================================================


class TestCheckResult:
    """Verify CheckResult defaults and construction."""

    def test_defaults(self) -> None:
        r = CheckResult("test", CheckStatus.PASS, "msg")
        assert r.details == []
        assert r.hints == []
        assert r.duration_ms is None

    def test_full_construction(self) -> None:
        r = CheckResult(
            "test",
            CheckStatus.FAIL,
            "bad",
            details=["d1"],
            hints=["h1"],
            duration_ms=42.0,
        )
        assert r.name == "test"
        assert r.status == CheckStatus.FAIL
        assert r.details == ["d1"]
        assert r.hints == ["h1"]
        assert r.duration_ms == 42.0


# =============================================================================
# PreflightChecker.__init__
# =============================================================================


class TestPreflightCheckerInit:
    """Verify constructor defaults and list normalization."""

    def test_defaults(self) -> None:
        c = _make_checker()
        assert c.namespace == "test-ns"
        assert c.image_pull_secrets == []
        assert c.secrets == []
        assert c.workers == 1

    def test_lists_normalized_from_none(self) -> None:
        c = PreflightChecker("ns", image_pull_secrets=None, secrets=None)
        assert c.image_pull_secrets == []
        assert c.secrets == []

    def test_lists_preserved_when_provided(self) -> None:
        c = _make_checker(image_pull_secrets=["s1"], secrets=["s2", "s3"])
        assert c.image_pull_secrets == ["s1"]
        assert c.secrets == ["s2", "s3"]


# =============================================================================
# _run_check
# =============================================================================


class TestRunCheck:
    """Verify the _run_check wrapper: timing, error handling."""

    @pytest.mark.asyncio
    async def test_run_check_populates_duration(self) -> None:
        checker = _make_checker()
        expected = CheckResult("t", CheckStatus.PASS, "ok")

        async def fn():
            return expected

        result = await checker._run_check("t", fn)
        assert result.duration_ms is not None
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_run_check_catches_exception(self) -> None:
        checker = _make_checker()

        async def fn():
            raise RuntimeError("boom")

        result = await checker._run_check("exploding", fn)
        assert result.status == CheckStatus.FAIL
        assert "boom" in result.message
        assert result.duration_ms is not None


# =============================================================================
# _check_cluster_connectivity
# =============================================================================


class TestCheckClusterConnectivity:
    """Verify cluster connectivity check."""

    @pytest.mark.asyncio
    async def test_connectivity_pass(self) -> None:
        checker = _make_checker()
        api = _mock_api()

        with patch("aiperf.kubernetes.client.get_api", new=AsyncMock(return_value=api)):
            result = await checker._check_cluster_connectivity()

        assert result.status == CheckStatus.PASS
        assert checker._api is api

    @pytest.mark.asyncio
    async def test_connectivity_fail_on_exception(self) -> None:
        checker = _make_checker()

        with patch(
            "aiperf.kubernetes.client.get_api",
            new=AsyncMock(side_effect=ConnectionError("refused")),
        ):
            result = await checker._check_cluster_connectivity()

        assert result.status == CheckStatus.FAIL
        assert "refused" in result.message
        assert len(result.hints) >= 1


# =============================================================================
# _check_kubernetes_version
# =============================================================================


class TestCheckKubernetesVersion:
    """Verify Kubernetes version compatibility check."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "major,minor,expected_status",
        [
            ("1", "28", CheckStatus.PASS),
            ("1", "24", CheckStatus.PASS),
            ("2", "0", CheckStatus.PASS),
            ("1", "23", CheckStatus.FAIL),
            ("0", "99", CheckStatus.FAIL),
        ],
    )  # fmt: skip
    async def test_version_thresholds(
        self, major: str, minor: str, expected_status: CheckStatus
    ) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_version.return_value = {
            "major": major,
            "minor": minor,
            "gitVersion": f"v{major}.{minor}.0",
        }
        result = await checker._check_kubernetes_version()
        assert result.status == expected_status

    @pytest.mark.asyncio
    async def test_version_with_plus_suffix(self) -> None:
        """GKE/EKS versions like '28+' should parse correctly."""
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_version.return_value = {
            "major": "1",
            "minor": "28+",
            "gitVersion": "v1.28.2-gke.1",
        }
        result = await checker._check_kubernetes_version()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_version_api_error_returns_warn(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_version.side_effect = RuntimeError("timeout")
        result = await checker._check_kubernetes_version()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_version_empty_strings(self) -> None:
        """Empty or None version fields should not crash."""
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_version.return_value = {
            "major": "",
            "minor": None,
            "gitVersion": "unknown",
        }
        result = await checker._check_kubernetes_version()
        assert result.status == CheckStatus.FAIL


# =============================================================================
# _check_namespace
# =============================================================================


class TestCheckNamespace:
    """Verify namespace existence and creation permission checks."""

    @pytest.mark.asyncio
    async def test_namespace_exists(self) -> None:
        checker = _make_checker(namespace="existing")
        checker._api = _mock_api()

        with patch("kr8s.asyncio.objects.Namespace") as MockNs:
            MockNs.get = AsyncMock()
            result = await checker._check_namespace()

        assert result.status == CheckStatus.PASS
        assert "exists" in result.message

    @pytest.mark.asyncio
    async def test_namespace_not_found_but_can_create(self) -> None:
        checker = _make_checker(namespace="new-ns")
        checker._api = _mock_api()
        checker._api.call_api = _mock_call_api_response({"status": {"allowed": True}})

        with patch("kr8s.asyncio.objects.Namespace") as MockNs:
            MockNs.get = AsyncMock(side_effect=kr8s.NotFoundError("not found"))
            result = await checker._check_namespace()

        assert result.status == CheckStatus.PASS
        assert "will be created" in result.message

    @pytest.mark.asyncio
    async def test_namespace_not_found_cannot_create(self) -> None:
        checker = _make_checker(namespace="restricted")
        checker._api = _mock_api()
        checker._api.call_api = _mock_call_api_response({"status": {"allowed": False}})

        with patch("kr8s.asyncio.objects.Namespace") as MockNs:
            MockNs.get = AsyncMock(side_effect=kr8s.NotFoundError("not found"))
            result = await checker._check_namespace()

        assert result.status == CheckStatus.FAIL
        assert len(result.hints) >= 1

    @pytest.mark.asyncio
    async def test_namespace_not_found_permission_check_fails(self) -> None:
        checker = _make_checker(namespace="broken")
        checker._api = _mock_api()
        checker._api.call_api = _mock_call_api_raises(RuntimeError("network"))

        with patch("kr8s.asyncio.objects.Namespace") as MockNs:
            MockNs.get = AsyncMock(side_effect=kr8s.NotFoundError("not found"))
            result = await checker._check_namespace()

        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_namespace_server_error(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()

        with patch("kr8s.asyncio.objects.Namespace") as MockNs:
            MockNs.get = AsyncMock(side_effect=create_server_error(500, "Internal"))
            result = await checker._check_namespace()

        assert result.status == CheckStatus.FAIL
        assert "500" in result.message


# =============================================================================
# _check_rbac_permissions
# =============================================================================


class TestCheckRBACPermissions:
    """Verify RBAC permission checks."""

    @pytest.mark.asyncio
    async def test_all_permissions_granted(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.call_api = _mock_call_api_response({"status": {"allowed": True}})
        result = await checker._check_rbac_permissions()
        assert result.status == CheckStatus.PASS
        assert "All" in result.message

    @pytest.mark.asyncio
    async def test_some_permissions_denied(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()

        call_count = 0

        @asynccontextmanager
        async def _alternating(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.json.return_value = {"status": {"allowed": call_count % 2 == 0}}
            yield resp

        checker._api.call_api = _alternating
        result = await checker._check_rbac_permissions()
        assert result.status == CheckStatus.FAIL
        assert "Missing" in result.message

    @pytest.mark.asyncio
    async def test_rbac_check_exception_treated_as_missing(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.call_api = _mock_call_api_raises(RuntimeError("network"))
        result = await checker._check_rbac_permissions()
        assert result.status == CheckStatus.FAIL
        assert "check failed" in str(result.details)


# =============================================================================
# _check_jobset_crd
# =============================================================================


class TestCheckJobSetCRD:
    """Verify JobSet CRD installation check."""

    @pytest.mark.asyncio
    async def test_crd_installed(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.call_api = _mock_call_api_response({"items": []})
        result = await checker._check_jobset_crd()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_crd_not_found(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.call_api = _mock_call_api_raises(
            create_not_found_error("JobSet CRD")
        )
        result = await checker._check_jobset_crd()
        assert result.status == CheckStatus.FAIL
        assert len(result.hints) >= 1

    @pytest.mark.asyncio
    async def test_crd_server_error(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.call_api = _mock_call_api_raises(
            create_server_error(503, "Unavailable")
        )
        result = await checker._check_jobset_crd()
        assert result.status == CheckStatus.WARN
        assert "503" in result.message


# =============================================================================
# _find_deployment / _check_jobset_controller
# =============================================================================


class TestCheckJobSetController:
    """Verify JobSet controller detection."""

    @pytest.mark.asyncio
    async def test_controller_running(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        deploy = make_kr8s_object(
            {
                "metadata": {
                    "name": "jobset-controller-manager",
                    "namespace": "jobset-system",
                },
                "status": {"readyReplicas": 1},
            }
        )
        checker._api.async_get = MagicMock(return_value=async_list([deploy]))

        result = await checker._check_jobset_controller()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_controller_found_not_ready(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        deploy = make_kr8s_object(
            {
                "metadata": {
                    "name": "jobset-controller-manager",
                    "namespace": "jobset-system",
                },
                "status": {"readyReplicas": 0},
            }
        )
        checker._api.async_get = MagicMock(return_value=async_list([deploy]))

        result = await checker._check_jobset_controller()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_controller_not_found(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_get = MagicMock(return_value=async_list([]))

        result = await checker._check_jobset_controller()
        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_controller_forbidden(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_get = MagicMock(
            side_effect=create_server_error(403, "Forbidden")
        )

        result = await checker._check_jobset_controller()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_controller_other_server_error(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_get = MagicMock(
            side_effect=create_server_error(502, "Bad Gateway")
        )

        result = await checker._check_jobset_controller()
        assert result.status == CheckStatus.WARN
        assert "502" in result.message


# =============================================================================
# _check_resource_quotas
# =============================================================================


class TestCheckResourceQuotas:
    """Verify resource quota detection."""

    @pytest.mark.asyncio
    async def test_no_quotas(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_get = MagicMock(return_value=async_list([]))

        result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_quotas_found(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        quota = make_kr8s_object(
            {
                "metadata": {"name": "compute", "namespace": "test-ns"},
                "status": {
                    "hard": {"cpu": "100", "memory": "256Gi"},
                    "used": {"cpu": "2", "memory": "8Gi"},
                },
            }
        )
        checker._api.async_get = MagicMock(return_value=async_list([quota]))

        result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.INFO
        assert "1 resource quota" in result.message

    @pytest.mark.asyncio
    async def test_quotas_server_error(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_get = MagicMock(
            side_effect=create_server_error(500, "Internal")
        )

        result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.WARN


# =============================================================================
# _check_node_resources
# =============================================================================


class TestCheckNodeResources:
    """Verify node resource estimation."""

    @pytest.mark.asyncio
    async def test_no_nodes(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_get = MagicMock(return_value=async_list([]))

        result = await checker._check_node_resources()
        assert result.status == CheckStatus.FAIL
        assert "No nodes" in result.message

    @pytest.mark.asyncio
    async def test_sufficient_resources(self) -> None:
        checker = _make_checker(workers=1)
        checker._api = _mock_api()
        node = make_kr8s_object(_node_raw("node-1", "16", "64Gi"))
        checker._api.async_get = MagicMock(return_value=async_list([node]))

        result = await checker._check_node_resources()
        assert result.status == CheckStatus.PASS
        assert len(result.details) >= 2

    @pytest.mark.asyncio
    async def test_insufficient_resources(self) -> None:
        checker = _make_checker(workers=1000)
        checker._api = _mock_api()
        node = make_kr8s_object(_node_raw("tiny-node", "1", "1Gi"))
        checker._api.async_get = MagicMock(return_value=async_list([node]))

        result = await checker._check_node_resources()
        assert result.status == CheckStatus.WARN
        assert "not have enough" in result.message

    @pytest.mark.asyncio
    async def test_not_ready_nodes_excluded(self) -> None:
        """Nodes that are not Ready should not contribute to totals."""
        checker = _make_checker(workers=1)
        checker._api = _mock_api()
        ready = make_kr8s_object(_node_raw("ready", "16", "64Gi", ready=True))
        not_ready = make_kr8s_object(_node_raw("sick", "16", "64Gi", ready=False))
        checker._api.async_get = MagicMock(return_value=async_list([ready, not_ready]))

        result = await checker._check_node_resources()
        assert "1 ready nodes" in result.details[0] or "1 nodes" in result.details[0]

    @pytest.mark.asyncio
    async def test_node_api_error_returns_warn(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_get = MagicMock(side_effect=RuntimeError("gone"))

        result = await checker._check_node_resources()
        assert result.status == CheckStatus.WARN


# =============================================================================
# _check_secrets
# =============================================================================


class TestCheckSecrets:
    """Verify secret existence checks."""

    @pytest.mark.asyncio
    async def test_no_secrets_specified(self) -> None:
        checker = _make_checker()
        result = await checker._check_secrets()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_all_secrets_found(self) -> None:
        checker = _make_checker(secrets=["s1", "s2"])
        checker._api = _mock_api()

        with patch("kr8s.asyncio.objects.Secret") as MockSecret:
            MockSecret.get = AsyncMock()
            result = await checker._check_secrets()

        assert result.status == CheckStatus.PASS
        assert "2 secret" in result.message

    @pytest.mark.asyncio
    async def test_missing_secret(self) -> None:
        checker = _make_checker(secrets=["exists", "missing"])
        checker._api = _mock_api()

        async def _get_secret(name, **kwargs):
            if name == "missing":
                raise kr8s.NotFoundError("not found")

        with patch("kr8s.asyncio.objects.Secret") as MockSecret:
            MockSecret.get = AsyncMock(side_effect=_get_secret)
            result = await checker._check_secrets()

        assert result.status == CheckStatus.FAIL
        assert "1 secret" in result.message

    @pytest.mark.asyncio
    async def test_permission_denied_secret(self) -> None:
        checker = _make_checker(secrets=["restricted"])
        checker._api = _mock_api()

        with patch("kr8s.asyncio.objects.Secret") as MockSecret:
            MockSecret.get = AsyncMock(
                side_effect=create_server_error(403, "Forbidden")
            )
            result = await checker._check_secrets()

        assert result.status == CheckStatus.WARN
        assert "permission denied" in str(result.details).lower()

    @pytest.mark.asyncio
    async def test_image_pull_secrets_included(self) -> None:
        checker = _make_checker(
            image_pull_secrets=["pull-secret"], secrets=["app-secret"]
        )
        checker._api = _mock_api()

        with patch("kr8s.asyncio.objects.Secret") as MockSecret:
            MockSecret.get = AsyncMock()
            result = await checker._check_secrets()

        assert result.status == CheckStatus.PASS
        assert "2 secret" in result.message


# =============================================================================
# _check_image
# =============================================================================


class TestCheckImage:
    """Verify image information check."""

    @pytest.mark.asyncio
    async def test_no_image_specified(self) -> None:
        checker = _make_checker(image=None)
        result = await checker._check_image()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_image_specified(self) -> None:
        checker = _make_checker(image="nvcr.io/aiperf:latest")
        result = await checker._check_image()
        assert result.status == CheckStatus.INFO
        assert "nvcr.io/aiperf:latest" in str(result.details)

    @pytest.mark.asyncio
    async def test_image_with_pull_secrets(self) -> None:
        checker = _make_checker(
            image="private.io/img:1", image_pull_secrets=["my-pull"]
        )
        result = await checker._check_image()
        assert result.status == CheckStatus.PASS
        assert "my-pull" in str(result.details)


# =============================================================================
# _check_network_policies
# =============================================================================


class TestCheckNetworkPolicies:
    """Verify network policy detection."""

    @pytest.mark.asyncio
    async def test_no_policies(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_get = MagicMock(return_value=async_list([]))

        result = await checker._check_network_policies()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_policies_found(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        policy = make_kr8s_object(
            {
                "metadata": {"name": "deny-all", "namespace": "test-ns"},
            }
        )
        checker._api.async_get = MagicMock(return_value=async_list([policy]))

        result = await checker._check_network_policies()
        assert result.status == CheckStatus.WARN
        assert "1 network policy" in result.message

    @pytest.mark.asyncio
    async def test_policies_forbidden(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_get = MagicMock(
            side_effect=create_server_error(403, "Forbidden")
        )

        result = await checker._check_network_policies()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_policies_server_error(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_get = MagicMock(
            side_effect=create_server_error(500, "Internal")
        )

        result = await checker._check_network_policies()
        assert result.status == CheckStatus.WARN


# =============================================================================
# _check_dns
# =============================================================================


class TestCheckDNS:
    """Verify DNS resolution check."""

    @pytest.mark.asyncio
    async def test_coredns_running(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        deploy = make_kr8s_object(
            {
                "metadata": {"name": "coredns", "namespace": "kube-system"},
                "status": {"readyReplicas": 2},
            }
        )
        checker._api.async_get = MagicMock(return_value=async_list([deploy]))

        result = await checker._check_dns()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_coredns_found_not_ready(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        deploy = make_kr8s_object(
            {
                "metadata": {"name": "coredns", "namespace": "kube-system"},
                "status": {"readyReplicas": 0},
            }
        )
        checker._api.async_get = MagicMock(return_value=async_list([deploy]))

        result = await checker._check_dns()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_coredns_not_found(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_get = MagicMock(return_value=async_list([]))

        result = await checker._check_dns()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_dns_check_error(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.async_get = MagicMock(side_effect=RuntimeError("timeout"))

        result = await checker._check_dns()
        assert result.status == CheckStatus.WARN


# =============================================================================
# _check_endpoint_connectivity
# =============================================================================


class TestCheckEndpointConnectivity:
    """Verify endpoint connectivity checks."""

    @pytest.mark.asyncio
    async def test_no_endpoint_specified(self) -> None:
        checker = _make_checker(endpoint_url=None)
        result = await checker._check_endpoint_connectivity()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_external_endpoint(self) -> None:
        checker = _make_checker(endpoint_url="https://api.example.com/v1")
        checker._api = _mock_api()
        result = await checker._check_endpoint_connectivity()
        assert result.status == CheckStatus.INFO
        assert "External endpoint" in result.message

    @pytest.mark.asyncio
    async def test_cluster_service_found(self) -> None:
        checker = _make_checker(
            endpoint_url="http://my-llm.inference.svc.cluster.local:8080/v1"
        )
        checker._api = _mock_api()

        with patch("kr8s.asyncio.objects.Service") as MockSvc:
            MockSvc.get = AsyncMock()
            result = await checker._check_endpoint_connectivity()

        assert result.status == CheckStatus.PASS
        assert "my-llm" in result.message

    @pytest.mark.asyncio
    async def test_cluster_service_not_found(self) -> None:
        checker = _make_checker(
            endpoint_url="http://gone.default.svc.cluster.local:8080"
        )
        checker._api = _mock_api()

        with patch("kr8s.asyncio.objects.Service") as MockSvc:
            MockSvc.get = AsyncMock(side_effect=kr8s.NotFoundError("not found"))
            result = await checker._check_endpoint_connectivity()

        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url,expected_port",
        [
            param("https://host.example.com/v1", 443, id="https-default-port"),
            param("http://host.example.com/v1", 80, id="http-default-port"),
            param("http://host.example.com:9090/v1", 9090, id="explicit-port"),
        ],
    )  # fmt: skip
    async def test_port_inference(self, url: str, expected_port: int) -> None:
        checker = _make_checker(endpoint_url=url)
        checker._api = _mock_api()
        result = await checker._check_endpoint_connectivity()
        assert f"Port: {expected_port}" in str(result.details)

    @pytest.mark.asyncio
    async def test_unparseable_url(self) -> None:
        """Malformed URL should not crash, returns WARN."""
        checker = _make_checker(endpoint_url="://bad")
        checker._api = _mock_api()
        result = await checker._check_endpoint_connectivity()
        # The urlparse handles most inputs, but host will be None/"unknown"
        assert result.status in (CheckStatus.INFO, CheckStatus.WARN)


# =============================================================================
# run_quick_checks orchestration
# =============================================================================


class TestRunQuickChecks:
    """Verify quick-check orchestration and short-circuit behavior."""

    @pytest.mark.asyncio
    async def test_quick_checks_all_pass(self) -> None:
        checker = _make_checker()
        checker._api = _mock_api()
        checker._api.call_api = _mock_call_api_response({"status": {"allowed": True}})

        with patch(
            "aiperf.kubernetes.client.get_api",
            new=AsyncMock(return_value=checker._api),
        ):
            results = await checker.run_quick_checks()

        assert results.passed
        assert len(results.checks) == 3

    @pytest.mark.asyncio
    async def test_quick_checks_short_circuits_on_connectivity_failure(self) -> None:
        checker = _make_checker()

        with patch(
            "aiperf.kubernetes.client.get_api",
            new=AsyncMock(side_effect=ConnectionError("refused")),
        ):
            results = await checker.run_quick_checks()

        assert not results.passed
        assert len(results.checks) == 1
        assert results.checks[0].status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_quick_checks_includes_endpoint_when_set(self) -> None:
        checker = _make_checker(endpoint_url="https://api.example.com")
        checker._api = _mock_api()
        checker._api.call_api = _mock_call_api_response({"status": {"allowed": True}})

        with patch(
            "aiperf.kubernetes.client.get_api",
            new=AsyncMock(return_value=checker._api),
        ):
            results = await checker.run_quick_checks()

        assert len(results.checks) == 4
        assert results.checks[3].name == "Endpoint Connectivity"

    @pytest.mark.asyncio
    async def test_quick_checks_no_endpoint_gives_three_checks(self) -> None:
        checker = _make_checker(endpoint_url=None)
        checker._api = _mock_api()
        checker._api.call_api = _mock_call_api_response({"status": {"allowed": True}})

        with patch(
            "aiperf.kubernetes.client.get_api",
            new=AsyncMock(return_value=checker._api),
        ):
            results = await checker.run_quick_checks()

        assert len(results.checks) == 3


# =============================================================================
# run_all_checks orchestration
# =============================================================================


class TestRunAllChecks:
    """Verify full check orchestration."""

    @pytest.mark.asyncio
    async def test_all_checks_short_circuits_on_connectivity_failure(self) -> None:
        checker = _make_checker()

        with patch(
            "aiperf.kubernetes.client.get_api",
            new=AsyncMock(side_effect=ConnectionError("refused")),
        ):
            results = await checker.run_all_checks()

        assert not results.passed
        assert len(results.checks) == 1

    @pytest.mark.asyncio
    async def test_all_checks_runs_all_thirteen(self) -> None:
        """When connectivity passes, all 13 checks should run."""
        checker = _make_checker(
            image="img:1",
            secrets=["s1"],
            endpoint_url="https://external.example.com",
        )
        api = _mock_api()
        api.call_api = _mock_call_api_response({"status": {"allowed": True}})
        api.async_get = MagicMock(return_value=async_list([]))

        with (
            patch(
                "aiperf.kubernetes.client.get_api",
                new=AsyncMock(return_value=api),
            ),
            patch("kr8s.asyncio.objects.Namespace") as MockNs,
            patch("kr8s.asyncio.objects.Secret") as MockSecret,
        ):
            MockNs.get = AsyncMock()
            MockSecret.get = AsyncMock()
            results = await checker.run_all_checks()

        assert len(results.checks) == 13
