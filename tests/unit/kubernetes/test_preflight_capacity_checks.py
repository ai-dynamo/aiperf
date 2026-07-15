# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct tests for aiperf.kubernetes.preflight_capacity_checks free functions.

These complement test_preflight.py (which exercises the same functions
indirectly via CLIPreflightChecker) by targeting the pure-logic helpers and
branches that the class-level tests under-cover:

- _controller_resource_requirements: sum of controller pod resources
- _evaluate_quotas: would-exceed math for both native and requests.* keys
- _node_is_ready / _aggregate_ready_nodes / _any_node_fits: node filtering
- check_resource_quotas: WARN path when a quota would be exceeded
- check_node_resources: FAIL path when no single node fits the largest pod
- check_secrets: mixed found/missing/denied details and non-404/403 HTTP error
- check_image: public-registry INFO vs private-registry WARN branching
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client import ApiClient, CoreV1Api
from kubernetes_asyncio.client.exceptions import ApiException
from kubernetes_asyncio.client.models import (
    V1Node,
    V1NodeCondition,
    V1NodeList,
    V1NodeStatus,
    V1ObjectMeta,
    V1ResourceQuota,
    V1ResourceQuotaList,
    V1ResourceQuotaStatus,
    V1Secret,
)
from pytest import param

from aiperf.kubernetes.preflight import CheckStatus
from aiperf.kubernetes.preflight_capacity_checks import (
    _aggregate_ready_nodes,
    _any_node_fits,
    _controller_resource_requirements,
    _evaluate_quotas,
    _node_is_ready,
    check_image,
    check_node_resources,
    check_resource_quotas,
    check_secrets,
)

# =============================================================================
# Helpers
# =============================================================================


def _mock_api() -> MagicMock:
    return MagicMock(spec=ApiClient)


def _patch_core(mock_core: MagicMock) -> Any:
    return patch(
        "aiperf.kubernetes.preflight_capacity_checks.client.CoreV1Api",
        return_value=mock_core,
    )


def _build_node(
    name: str,
    cpu: str,
    memory: str,
    *,
    ready: bool = True,
    has_status: bool = True,
    has_allocatable: bool = True,
) -> V1Node:
    """Build a V1Node with controllable status/allocatable presence."""
    if not has_status:
        return V1Node(metadata=V1ObjectMeta(name=name), status=None)
    status = V1NodeStatus(
        conditions=[V1NodeCondition(type="Ready", status="True" if ready else "False")],
        allocatable={"cpu": cpu, "memory": memory} if has_allocatable else None,
    )
    return V1Node(metadata=V1ObjectMeta(name=name), status=status)


def _build_quota(
    name: str, *, hard: dict[str, str], used: dict[str, str]
) -> V1ResourceQuota:
    return V1ResourceQuota(
        metadata=V1ObjectMeta(name=name),
        status=V1ResourceQuotaStatus(hard=hard, used=used),
    )


# =============================================================================
# _controller_resource_requirements
# =============================================================================


class TestControllerResourceRequirements:
    """Verify the controller pod aggregate resource calculation."""

    def test_returns_positive_totals(self) -> None:
        """Summing all controller resource keys yields strictly positive CPU/mem."""
        cpu, mem = _controller_resource_requirements()
        assert cpu > 0
        assert mem > 0

    def test_returns_floats(self) -> None:
        cpu, mem = _controller_resource_requirements()
        assert isinstance(cpu, float)
        assert isinstance(mem, float)


# =============================================================================
# _evaluate_quotas
# =============================================================================


class TestEvaluateQuotas:
    """Verify quota-exceeded math and detail formatting."""

    def test_empty_quotas_list(self) -> None:
        evaluation = _evaluate_quotas([], required_cpu=10.0, required_mem=32.0)
        assert evaluation.would_exceed is False
        assert evaluation.details == []

    def test_under_quota_native_keys(self) -> None:
        """cpu/memory under the hard limit: no exceed, detail lines emitted."""
        quota = _build_quota(
            "ns-q", hard={"cpu": "100", "memory": "256Gi"}, used={"cpu": "2"}
        )
        evaluation = _evaluate_quotas([quota], required_cpu=4.0, required_mem=16.0)
        assert evaluation.would_exceed is False
        assert any("ResourceQuota 'ns-q'" in d for d in evaluation.details)
        assert any("cpu: 2 / 100" in d for d in evaluation.details)

    def test_exceeds_cpu_on_native_key(self) -> None:
        quota = _build_quota(
            "tight", hard={"cpu": "10", "memory": "64Gi"}, used={"cpu": "2"}
        )
        evaluation = _evaluate_quotas([quota], required_cpu=50.0, required_mem=8.0)
        assert evaluation.would_exceed is True
        assert any("CPU would exceed" in d for d in evaluation.details)

    def test_exceeds_memory_on_requests_key(self) -> None:
        """requests.cpu / requests.memory are accepted as fallback keys."""
        quota = _build_quota(
            "req",
            hard={"requests.cpu": "100", "requests.memory": "8Gi"},
            used={"requests.cpu": "1", "requests.memory": "1Gi"},
        )
        evaluation = _evaluate_quotas([quota], required_cpu=1.0, required_mem=50.0)
        assert evaluation.would_exceed is True
        assert any("Memory would exceed" in d for d in evaluation.details)

    def test_missing_status_is_safe(self) -> None:
        """A quota with no status should not crash and produces no exceed."""
        quota = V1ResourceQuota(metadata=V1ObjectMeta(name="new"), status=None)
        evaluation = _evaluate_quotas([quota], required_cpu=1.0, required_mem=1.0)
        assert evaluation.would_exceed is False

    def test_multiple_quotas_any_exceeds(self) -> None:
        ok = _build_quota("a", hard={"cpu": "100"}, used={"cpu": "1"})
        tight = _build_quota("b", hard={"cpu": "2"}, used={"cpu": "1"})
        evaluation = _evaluate_quotas([ok, tight], required_cpu=5.0, required_mem=0.0)
        assert evaluation.would_exceed is True


# =============================================================================
# _node_is_ready / _aggregate_ready_nodes / _any_node_fits
# =============================================================================


class TestNodeHelpers:
    """Verify node filtering primitives."""

    @pytest.mark.parametrize(
        "ready,expected",
        [
            param(True, True, id="ready"),
            param(False, False, id="not-ready"),
        ],
    )  # fmt: skip
    def test_node_is_ready(self, ready: bool, expected: bool) -> None:
        node = _build_node("n", "1", "1Gi", ready=ready)
        assert _node_is_ready(node) is expected

    def test_node_is_ready_no_status(self) -> None:
        node = _build_node("n", "1", "1Gi", has_status=False)
        assert _node_is_ready(node) is False

    def test_aggregate_ready_nodes_skips_not_ready(self) -> None:
        good = _build_node("good", "16", "64Gi", ready=True)
        bad = _build_node("bad", "32", "128Gi", ready=False)
        count, cpu, mem = _aggregate_ready_nodes([good, bad])
        assert count == 1
        assert cpu == 16
        assert mem == 64

    def test_aggregate_ready_nodes_skips_empty_allocatable(self) -> None:
        """Nodes with no allocatable field do not contribute to totals."""
        hollow = _build_node("hollow", "0", "0", has_allocatable=False)
        count, cpu, mem = _aggregate_ready_nodes([hollow])
        assert count == 0
        assert cpu == 0
        assert mem == 0

    def test_any_node_fits_true(self) -> None:
        small = _build_node("small", "2", "4Gi", ready=True)
        big = _build_node("big", "64", "256Gi", ready=True)
        assert _any_node_fits([small, big], max_pod_cpu=16.0, max_pod_mem=32.0) is True

    def test_any_node_fits_false(self) -> None:
        small = _build_node("small", "2", "4Gi", ready=True)
        assert _any_node_fits([small], max_pod_cpu=16.0, max_pod_mem=32.0) is False

    def test_any_node_fits_ignores_not_ready(self) -> None:
        """A huge but NotReady node cannot satisfy the fit check."""
        big_but_sick = _build_node("big", "64", "256Gi", ready=False)
        assert _any_node_fits([big_but_sick], max_pod_cpu=1.0, max_pod_mem=1.0) is False


# =============================================================================
# check_resource_quotas (free function direct)
# =============================================================================


class TestCheckResourceQuotasFreeFn:
    """Exercise branches the class-based suite under-covers."""

    @pytest.mark.asyncio
    async def test_quota_would_be_exceeded_returns_warn(self) -> None:
        """When required + used exceeds hard cpu, status is WARN with hints."""
        quota = _build_quota(
            "tight", hard={"cpu": "2", "memory": "4Gi"}, used={"cpu": "0"}
        )
        core = MagicMock(spec=CoreV1Api)
        core.list_namespaced_resource_quota = AsyncMock(
            return_value=V1ResourceQuotaList(items=[quota])
        )

        with _patch_core(core):
            result = await check_resource_quotas(
                _mock_api(), namespace="test-ns", workers=50
            )

        assert result.status == CheckStatus.WARN
        assert "exceed" in result.message
        assert result.hints
        assert any("Benchmark needs" in d for d in result.details)

    @pytest.mark.asyncio
    async def test_quota_api_exception_returns_warn_with_status(self) -> None:
        core = MagicMock(spec=CoreV1Api)
        core.list_namespaced_resource_quota = AsyncMock(
            side_effect=ApiException(status=418)
        )

        with _patch_core(core):
            result = await check_resource_quotas(_mock_api(), namespace="ns", workers=1)

        assert result.status == CheckStatus.WARN
        assert "HTTP 418" in result.message


# =============================================================================
# check_node_resources (free function direct)
# =============================================================================


class TestCheckNodeResourcesFreeFn:
    @pytest.mark.asyncio
    async def test_total_fits_but_no_single_node_fits_returns_fail(self) -> None:
        """Total cluster resources suffice, yet no individual node fits one pod."""
        tiny1 = _build_node("n1", "1", "1Gi", ready=True)
        tiny2 = _build_node("n2", "1", "1Gi", ready=True)
        tiny3 = _build_node("n3", "1", "1Gi", ready=True)
        tiny4 = _build_node("n4", "1", "1Gi", ready=True)

        core = MagicMock(spec=CoreV1Api)
        core.list_node = AsyncMock(
            return_value=V1NodeList(items=[tiny1, tiny2, tiny3, tiny4])
        )

        # Patch the per-pod worker requirement so a worker alone needs more than any node.
        with (
            _patch_core(core),
            patch(
                "aiperf.kubernetes.preflight_capacity_checks.K8sEnvironment"
            ) as mock_env,
            patch(
                "aiperf.kubernetes.preflight_capacity_checks."
                "_controller_resource_requirements",
                return_value=(0.1, 0.1),
            ),
        ):
            mock_env.WORKER_POD.CPU = "2"
            mock_env.WORKER_POD.MEMORY = "2Gi"
            result = await check_node_resources(_mock_api(), workers=1)

        assert result.status == CheckStatus.FAIL
        assert "No single node" in result.message
        assert result.hints


# =============================================================================
# check_secrets (free function direct)
# =============================================================================


class TestCheckSecretsFreeFn:
    @pytest.mark.asyncio
    async def test_mixed_results_prefers_missing_fail(self) -> None:
        """With both missing and permission_denied present, FAIL dominates."""
        core = MagicMock(spec=CoreV1Api)

        async def _read(name: str, _ns: str, **_: Any):
            if name == "ok":
                return V1Secret(metadata=V1ObjectMeta(name=name))
            if name == "gone":
                raise ApiException(status=404)
            if name == "denied":
                raise ApiException(status=403)
            raise ApiException(status=500)

        core.read_namespaced_secret = AsyncMock(side_effect=_read)

        with _patch_core(core):
            result = await check_secrets(
                _mock_api(),
                namespace="ns",
                image_pull_secrets=[],
                secrets=["ok", "gone", "denied"],
            )

        assert result.status == CheckStatus.FAIL
        assert "1 secret" in result.message
        detail_text = "\n".join(result.details)
        assert "✓ ok" in detail_text
        assert "gone" in detail_text and "not found" in detail_text
        assert "denied" in detail_text and "permission denied" in detail_text

    @pytest.mark.asyncio
    async def test_non_404_403_error_is_classified_missing(self) -> None:
        """HTTP 500 from read_namespaced_secret surfaces as a missing entry."""
        core = MagicMock(spec=CoreV1Api)
        core.read_namespaced_secret = AsyncMock(side_effect=ApiException(status=500))

        with _patch_core(core):
            result = await check_secrets(
                _mock_api(), namespace="ns", image_pull_secrets=[], secrets=["weird"]
            )

        assert result.status == CheckStatus.FAIL
        assert any("HTTP 500" in d for d in result.details)

    @pytest.mark.asyncio
    async def test_found_only_shows_checkmarks(self) -> None:
        core = MagicMock(spec=CoreV1Api)
        core.read_namespaced_secret = AsyncMock(
            return_value=V1Secret(metadata=V1ObjectMeta(name="x"))
        )
        with _patch_core(core):
            result = await check_secrets(
                _mock_api(),
                namespace="ns",
                image_pull_secrets=["pull"],
                secrets=["app"],
            )
        assert result.status == CheckStatus.PASS
        assert all(d.strip().startswith("✓") for d in result.details)


# =============================================================================
# check_image (free function direct)
# =============================================================================


class TestCheckImageFreeFn:
    """Exercise registry-classification branches."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "image",
        [
            param("nvcr.io/nvidia/aiperf:1.0", id="nvcr-io-tagged"),
            param("docker.io/library/python:3.12", id="dockerhub-tagged"),
            param("ghcr.io/org/repo:v2", id="ghcr-tagged"),
            param("quay.io/org/repo:dev", id="quay-tagged"),
        ],
    )  # fmt: skip
    async def test_public_registry_without_pull_secret_is_info(
        self, image: str
    ) -> None:
        """Known public registries land on INFO (not WARN)."""
        result = await check_image(_mock_api(), image=image, image_pull_secrets=[])
        assert result.status == CheckStatus.INFO
        assert "public registry" in result.message.lower()

    @pytest.mark.asyncio
    async def test_private_registry_without_pull_secret_is_warn(self) -> None:
        result = await check_image(
            _mock_api(),
            image="internal.corp.example.com/img:1",
            image_pull_secrets=[],
        )
        assert result.status == CheckStatus.WARN
        assert "pull secrets" in result.message.lower()
        assert any("may require authentication" in h for h in result.hints)

    @pytest.mark.asyncio
    async def test_image_without_tag_reports_latest_implicit(self) -> None:
        result = await check_image(
            _mock_api(), image="docker.io/library/busybox", image_pull_secrets=[]
        )
        assert "latest (implicit)" in "\n".join(result.details)

    @pytest.mark.asyncio
    async def test_image_with_pull_secrets_overrides_public_classification(
        self,
    ) -> None:
        """Pull secrets always PASS, regardless of registry reputation."""
        result = await check_image(
            _mock_api(),
            image="nvcr.io/nvidia/aiperf:1",
            image_pull_secrets=["ngc-secret"],
        )
        assert result.status == CheckStatus.PASS
        assert "ngc-secret" in "\n".join(result.details)

    @pytest.mark.asyncio
    async def test_no_image_skips(self) -> None:
        result = await check_image(_mock_api(), image=None, image_pull_secrets=[])
        assert result.status == CheckStatus.SKIP
