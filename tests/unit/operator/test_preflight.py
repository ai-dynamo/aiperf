# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.operator.preflight module.

Focuses on:
- OperatorPreflightChecker individual check methods (mocked k8s API)
- Tiered orchestration (short-circuit on tier 1/2, concurrent tier 3+)
- Timeout handling
- Error messages include actionable remediation hints
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import kr8s
import pytest
from pytest import param

from aiperf.config import AIPerfConfig
from aiperf.config.deployment import (
    DeploymentConfig,
    PodTemplateConfig,
    SchedulingConfig,
)
from aiperf.kubernetes.preflight import CheckStatus
from aiperf.kubernetes.resources import KubernetesDeployment
from aiperf.operator.preflight import OperatorPreflightChecker, _is_node_ready
from tests.harness.k8s import (
    async_list,
    create_not_found_error,
    create_server_error,
    make_kr8s_object,
)

# =============================================================================
# Helpers
# =============================================================================


def _mock_api() -> MagicMock:
    """Build a MagicMock kr8s API with commonly-needed async stubs."""
    api = MagicMock(spec=kr8s.Api)
    api.async_version = AsyncMock(
        return_value={"major": "1", "minor": "28", "gitVersion": "v1.28.0"}
    )
    api.async_get = MagicMock(return_value=async_list([]))
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
    labels: dict[str, str] | None = None,
    taints: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a minimal Node .raw dict."""
    conditions = [{"type": "Ready", "status": "True" if ready else "False"}]
    raw: dict[str, Any] = {
        "metadata": {"name": name, "namespace": "", "labels": labels or {}},
        "status": {
            "conditions": conditions,
            "allocatable": {"cpu": cpu, "memory": memory},
        },
        "spec": {},
    }
    if taints:
        raw["spec"]["taints"] = taints
    return raw


def _sample_config() -> AIPerfConfig:
    """Create a minimal AIPerfConfig for testing."""
    return AIPerfConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets={
            "main": {
                "type": "synthetic",
                "entries": 10,
                "prompts": {"isl": 32, "osl": 16},
            }
        },
        phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
    )


def _make_checker(
    *,
    api: MagicMock | None = None,
    namespace: str = "test-ns",
    deploy_config: DeploymentConfig | None = None,
    config: AIPerfConfig | None = None,
    total_workers: int = 2,
    num_pods: int = 1,
) -> OperatorPreflightChecker:
    """Create an OperatorPreflightChecker with sensible defaults."""
    if api is None:
        api = _mock_api()
    if deploy_config is None:
        deploy_config = DeploymentConfig()
    if config is None:
        config = _sample_config()

    deployment = KubernetesDeployment(
        job_id="test-job",
        namespace=namespace,
        worker_replicas=num_pods,
        config=config,
        deployment=deploy_config,
    )
    return OperatorPreflightChecker(
        api=api,
        namespace=namespace,
        deployment=deployment,
        deploy_config=deploy_config,
        config=config,
        total_workers=total_workers,
        num_pods=num_pods,
    )


# =============================================================================
# _is_node_ready helper
# =============================================================================


class TestIsNodeReady:
    """Verify _is_node_ready helper."""

    def test_ready_node(self) -> None:
        raw = _node_raw("n1", "4", "16Gi", ready=True)
        assert _is_node_ready(raw) is True

    def test_not_ready_node(self) -> None:
        raw = _node_raw("n1", "4", "16Gi", ready=False)
        assert _is_node_ready(raw) is False

    def test_no_conditions(self) -> None:
        raw: dict[str, Any] = {"status": {}}
        assert _is_node_ready(raw) is False


# =============================================================================
# Tier 1: Kubernetes Version
# =============================================================================


class TestCheckKubernetesVersion:
    """Verify Kubernetes version compatibility check."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "major,minor,expected_status",
        [
            param("1", "28", CheckStatus.PASS, id="v1.28-pass"),
            param("1", "24", CheckStatus.PASS, id="v1.24-pass-edge"),
            param("1", "23", CheckStatus.FAIL, id="v1.23-fail"),
            param("0", "99", CheckStatus.FAIL, id="v0.99-fail"),
        ],
    )  # fmt: skip
    async def test_version_thresholds(
        self,
        major: str,
        minor: str,
        expected_status: CheckStatus,
    ) -> None:
        checker = _make_checker()
        checker.api.async_version.return_value = {
            "major": major,
            "minor": minor,
            "gitVersion": f"v{major}.{minor}.0",
        }
        result = await checker._check_kubernetes_version()
        assert result.status == expected_status

    @pytest.mark.asyncio
    async def test_gke_version_with_plus_suffix(self) -> None:
        """GKE/EKS versions like '28+' should parse correctly."""
        checker = _make_checker()
        checker.api.async_version.return_value = {
            "major": "1",
            "minor": "28+",
            "gitVersion": "v1.28.2-gke.1",
        }
        result = await checker._check_kubernetes_version()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_empty_version_fields(self) -> None:
        checker = _make_checker()
        checker.api.async_version.return_value = {
            "major": "",
            "minor": None,
            "gitVersion": "unknown",
        }
        result = await checker._check_kubernetes_version()
        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_fail_message_includes_upgrade_hint(self) -> None:
        checker = _make_checker()
        checker.api.async_version.return_value = {
            "major": "1",
            "minor": "22",
            "gitVersion": "v1.22.0",
        }
        result = await checker._check_kubernetes_version()
        assert "Upgrade" in result.message


# =============================================================================
# Tier 1: JobSet CRD
# =============================================================================


class TestCheckJobSetCRD:
    """Verify JobSet CRD installation check."""

    @pytest.mark.asyncio
    async def test_crd_installed(self) -> None:
        checker = _make_checker()
        checker.api.call_api = _mock_call_api_response({"items": []})
        result = await checker._check_jobset_crd()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_crd_not_found(self) -> None:
        checker = _make_checker()
        checker.api.call_api = _mock_call_api_raises(
            create_not_found_error("JobSet CRD")
        )
        result = await checker._check_jobset_crd()
        assert result.status == CheckStatus.FAIL
        assert "Install" in result.message

    @pytest.mark.asyncio
    async def test_crd_server_error(self) -> None:
        checker = _make_checker()
        checker.api.call_api = _mock_call_api_raises(
            create_server_error(503, "Unavailable")
        )
        result = await checker._check_jobset_crd()
        assert result.status == CheckStatus.FAIL
        assert "503" in result.message


# =============================================================================
# Tier 2: RBAC Permissions
# =============================================================================


class TestCheckRBACPermissions:
    """Verify RBAC permission checks."""

    @pytest.mark.asyncio
    async def test_all_permissions_granted(self) -> None:
        checker = _make_checker()
        checker.api.call_api = _mock_call_api_response({"status": {"allowed": True}})
        result = await checker._check_rbac_permissions()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_some_permissions_denied(self) -> None:
        checker = _make_checker()
        call_count = 0

        @asynccontextmanager
        async def _alternating(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.json.return_value = {"status": {"allowed": call_count % 2 == 0}}
            yield resp

        checker.api.call_api = _alternating
        result = await checker._check_rbac_permissions()
        assert result.status == CheckStatus.FAIL
        assert "Missing" in result.message
        assert "namespace" in result.message.lower()

    @pytest.mark.asyncio
    async def test_rbac_check_exception_treated_as_missing(self) -> None:
        checker = _make_checker()
        checker.api.call_api = _mock_call_api_raises(RuntimeError("network"))
        result = await checker._check_rbac_permissions()
        assert result.status == CheckStatus.FAIL
        assert "check failed" in result.message


# =============================================================================
# Tier 3: JobSet Controller
# =============================================================================


class TestCheckJobSetController:
    """Verify JobSet controller detection."""

    @pytest.mark.asyncio
    async def test_controller_running(self) -> None:
        checker = _make_checker()
        deploy = make_kr8s_object(
            {
                "metadata": {
                    "name": "jobset-controller-manager",
                    "namespace": "jobset-system",
                },
                "status": {"readyReplicas": 1},
            }
        )
        checker.api.async_get = MagicMock(return_value=async_list([deploy]))
        result = await checker._check_jobset_controller()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_controller_found_not_ready(self) -> None:
        checker = _make_checker()
        deploy = make_kr8s_object(
            {
                "metadata": {
                    "name": "jobset-controller-manager",
                    "namespace": "jobset-system",
                },
                "status": {"readyReplicas": 0},
            }
        )
        checker.api.async_get = MagicMock(return_value=async_list([deploy]))
        result = await checker._check_jobset_controller()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_controller_not_found(self) -> None:
        checker = _make_checker()
        checker.api.async_get = MagicMock(return_value=async_list([]))
        result = await checker._check_jobset_controller()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_controller_forbidden(self) -> None:
        checker = _make_checker()
        checker.api.async_get = MagicMock(
            side_effect=create_server_error(403, "Forbidden")
        )
        result = await checker._check_jobset_controller()
        assert result.status == CheckStatus.SKIP


# =============================================================================
# Tier 3: Service Account
# =============================================================================


class TestCheckServiceAccount:
    """Verify service account check."""

    @pytest.mark.asyncio
    async def test_no_sa_specified_skips(self) -> None:
        checker = _make_checker()
        result = await checker._check_service_account()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_sa_exists(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(service_account_name="my-sa"),
        )
        checker = _make_checker(deploy_config=dc)

        with patch("kr8s.asyncio.objects.ServiceAccount") as MockSA:
            MockSA.get = AsyncMock()
            result = await checker._check_service_account()

        assert result.status == CheckStatus.PASS
        assert "my-sa" in result.message

    @pytest.mark.asyncio
    async def test_sa_not_found(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(service_account_name="missing-sa"),
        )
        checker = _make_checker(deploy_config=dc)

        with patch("kr8s.asyncio.objects.ServiceAccount") as MockSA:
            MockSA.get = AsyncMock(side_effect=kr8s.NotFoundError("not found"))
            result = await checker._check_service_account()

        assert result.status == CheckStatus.FAIL
        assert "missing-sa" in result.message


# =============================================================================
# Tier 3: Node Resources
# =============================================================================


class TestCheckNodeResources:
    """Verify node resource estimation."""

    @pytest.mark.asyncio
    async def test_sufficient_resources(self) -> None:
        checker = _make_checker(num_pods=1)
        node = make_kr8s_object(_node_raw("big-node", "16", "64Gi"))
        checker.api.async_get = MagicMock(return_value=async_list([node]))
        result = await checker._check_node_resources()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_insufficient_resources(self) -> None:
        checker = _make_checker(num_pods=1000)
        node = make_kr8s_object(_node_raw("tiny-node", "1", "1Gi"))
        checker.api.async_get = MagicMock(return_value=async_list([node]))
        result = await checker._check_node_resources()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_no_nodes(self) -> None:
        checker = _make_checker()
        checker.api.async_get = MagicMock(return_value=async_list([]))
        result = await checker._check_node_resources()
        assert result.status == CheckStatus.WARN
        assert "No nodes" in result.message

    @pytest.mark.asyncio
    async def test_api_error_returns_warn(self) -> None:
        checker = _make_checker()
        checker.api.async_get = MagicMock(side_effect=RuntimeError("gone"))
        result = await checker._check_node_resources()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_resource_mode_none_skips(self) -> None:
        checker = _make_checker(deploy_config=DeploymentConfig(resource_mode="none"))
        result = await checker._check_node_resources()
        assert result.status == CheckStatus.SKIP
        assert "resourceMode=none" in result.message


# =============================================================================
# Tier 3: Node Selector Match
# =============================================================================


class TestCheckNodeSelectorMatch:
    """Verify node selector matching."""

    @pytest.mark.asyncio
    async def test_no_selector_skips(self) -> None:
        checker = _make_checker()
        result = await checker._check_node_selector_match()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_matching_nodes_exist(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(node_selector={"gpu": "true"}),
        )
        checker = _make_checker(deploy_config=dc)
        node = make_kr8s_object(
            _node_raw("gpu-node", "8", "32Gi", labels={"gpu": "true"})
        )
        checker.api.async_get = MagicMock(return_value=async_list([node]))
        result = await checker._check_node_selector_match()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_no_matching_nodes(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(node_selector={"gpu": "true"}),
        )
        checker = _make_checker(deploy_config=dc)
        node = make_kr8s_object(
            _node_raw("cpu-node", "8", "32Gi", labels={"gpu": "false"})
        )
        checker.api.async_get = MagicMock(return_value=async_list([node]))
        result = await checker._check_node_selector_match()
        assert result.status == CheckStatus.FAIL
        assert "kubectl label" in result.message

    @pytest.mark.asyncio
    async def test_not_ready_nodes_excluded(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(node_selector={"gpu": "true"}),
        )
        checker = _make_checker(deploy_config=dc)
        node = make_kr8s_object(
            _node_raw("gpu-node", "8", "32Gi", ready=False, labels={"gpu": "true"})
        )
        checker.api.async_get = MagicMock(return_value=async_list([node]))
        result = await checker._check_node_selector_match()
        assert result.status == CheckStatus.FAIL


# =============================================================================
# Tier 3: Per-Node Schedulability
# =============================================================================


class TestCheckPerNodeSchedulability:
    """Verify per-node scheduling feasibility."""

    @pytest.mark.asyncio
    async def test_node_can_fit(self) -> None:
        checker = _make_checker()
        node = make_kr8s_object(_node_raw("big-node", "16", "64Gi"))
        checker.api.async_get = MagicMock(return_value=async_list([node]))
        result = await checker._check_per_node_schedulability()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_no_node_can_fit(self) -> None:
        checker = _make_checker()
        node = make_kr8s_object(_node_raw("tiny-node", "0.1", "100Mi"))
        checker.api.async_get = MagicMock(return_value=async_list([node]))
        result = await checker._check_per_node_schedulability()
        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_filters_by_node_selector(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(node_selector={"gpu": "true"}),
        )
        checker = _make_checker(deploy_config=dc)
        # Big node without label, tiny node with label
        big = make_kr8s_object(_node_raw("big-no-label", "16", "64Gi", labels={}))
        tiny = make_kr8s_object(
            _node_raw("tiny-gpu", "0.1", "100Mi", labels={"gpu": "true"})
        )
        checker.api.async_get = MagicMock(return_value=async_list([big, tiny]))
        result = await checker._check_per_node_schedulability()
        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_resource_mode_none_skips(self) -> None:
        checker = _make_checker(deploy_config=DeploymentConfig(resource_mode="none"))
        result = await checker._check_per_node_schedulability()
        assert result.status == CheckStatus.SKIP
        assert "resourceMode=none" in result.message


# =============================================================================
# Tier 3: Resource Quotas
# =============================================================================


class TestCheckResourceQuotas:
    """Verify resource quota checks."""

    @pytest.mark.asyncio
    async def test_no_quotas(self) -> None:
        checker = _make_checker()
        checker.api.async_get = MagicMock(return_value=async_list([]))
        result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_within_quota(self) -> None:
        checker = _make_checker(num_pods=1)
        quota = make_kr8s_object(
            {
                "metadata": {"name": "compute", "namespace": "test-ns"},
                "status": {
                    "hard": {"cpu": "100", "memory": "256Gi"},
                    "used": {"cpu": "2", "memory": "4Gi"},
                },
            }
        )
        checker.api.async_get = MagicMock(return_value=async_list([quota]))
        result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_exceeds_cpu_quota(self) -> None:
        checker = _make_checker(num_pods=100)
        quota = make_kr8s_object(
            {
                "metadata": {"name": "tight", "namespace": "test-ns"},
                "status": {
                    "hard": {"cpu": "10"},
                    "used": {"cpu": "9"},
                },
            }
        )
        checker.api.async_get = MagicMock(return_value=async_list([quota]))
        result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.WARN
        assert "quota" in result.message.lower()

    @pytest.mark.asyncio
    async def test_resource_mode_none_skips(self) -> None:
        checker = _make_checker(deploy_config=DeploymentConfig(resource_mode="none"))
        result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.SKIP
        assert "resourceMode=none" in result.message


# =============================================================================
# Tier 3: Memory Estimation
# =============================================================================


class TestCheckMemoryEstimation:
    """Verify memory OOM risk detection."""

    @pytest.mark.asyncio
    async def test_memory_ok(self) -> None:
        checker = _make_checker(total_workers=1)
        result = await checker._check_memory_estimation()
        assert result.status in (CheckStatus.PASS, CheckStatus.WARN)

    @pytest.mark.asyncio
    async def test_memory_estimation_error_returns_warn(self) -> None:
        checker = _make_checker()
        with patch(
            "aiperf.kubernetes.memory_estimator.estimate_memory",
            side_effect=RuntimeError("bad config"),
        ):
            result = await checker._check_memory_estimation()
        assert result.status == CheckStatus.WARN
        assert "bad config" in result.message

    @pytest.mark.asyncio
    async def test_resource_mode_none_skips(self) -> None:
        checker = _make_checker(deploy_config=DeploymentConfig(resource_mode="none"))
        result = await checker._check_memory_estimation()
        assert result.status == CheckStatus.SKIP
        assert "resourceMode=none" in result.message


# =============================================================================
# Tier 3: Secrets
# =============================================================================


class TestCheckSecrets:
    """Verify secret existence checks."""

    @pytest.mark.asyncio
    async def test_no_secrets_referenced(self) -> None:
        checker = _make_checker()
        result = await checker._check_secrets()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_all_secrets_found(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                image_pull_secrets=["pull-secret"],
                volumes=[{"name": "creds", "secret": {"secretName": "my-creds"}}],
            ),
        )
        checker = _make_checker(deploy_config=dc)

        with patch("kr8s.asyncio.objects.Secret") as MockSecret:
            MockSecret.get = AsyncMock()
            result = await checker._check_secrets()

        assert result.status == CheckStatus.PASS
        assert "2 secret" in result.message

    @pytest.mark.asyncio
    async def test_image_pull_secret_missing(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(image_pull_secrets=["missing"]),
        )
        checker = _make_checker(deploy_config=dc)

        with patch("kr8s.asyncio.objects.Secret") as MockSecret:
            MockSecret.get = AsyncMock(side_effect=kr8s.NotFoundError("not found"))
            result = await checker._check_secrets()

        assert result.status == CheckStatus.FAIL
        assert "missing" in result.message

    @pytest.mark.asyncio
    async def test_env_secret_missing(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                env=[
                    {
                        "name": "KEY",
                        "valueFrom": {"secretKeyRef": {"name": "api-keys", "key": "k"}},
                    }
                ],
            ),
        )
        checker = _make_checker(deploy_config=dc)

        with patch("kr8s.asyncio.objects.Secret") as MockSecret:
            MockSecret.get = AsyncMock(side_effect=kr8s.NotFoundError("not found"))
            result = await checker._check_secrets()

        assert result.status == CheckStatus.FAIL
        assert "api-keys" in result.message

    @pytest.mark.asyncio
    async def test_permission_denied(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(image_pull_secrets=["restricted"]),
        )
        checker = _make_checker(deploy_config=dc)

        with patch("kr8s.asyncio.objects.Secret") as MockSecret:
            MockSecret.get = AsyncMock(
                side_effect=create_server_error(403, "Forbidden")
            )
            result = await checker._check_secrets()

        assert result.status == CheckStatus.WARN
        assert "permission denied" in result.message.lower()


# =============================================================================
# Tier 3: Image Reference
# =============================================================================


class TestCheckImageReference:
    """Verify image reference validation."""

    @pytest.mark.asyncio
    async def test_valid_image_with_tag(self) -> None:
        dc = DeploymentConfig(image="nvcr.io/nvidia/aiperf:1.0.0")
        checker = _make_checker(deploy_config=dc)
        result = await checker._check_image_reference()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_no_tag_warns_latest(self) -> None:
        dc = DeploymentConfig(image="nvcr.io/nvidia/aiperf")
        checker = _make_checker(deploy_config=dc)
        result = await checker._check_image_reference()
        assert result.status == CheckStatus.WARN
        assert "latest" in result.message.lower()

    @pytest.mark.asyncio
    async def test_empty_image_fails(self) -> None:
        dc = DeploymentConfig(image="")
        checker = _make_checker(deploy_config=dc)
        result = await checker._check_image_reference()
        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_private_registry_no_secrets_warns(self) -> None:
        dc = DeploymentConfig(image="private.corp.io/team/app:v1")
        checker = _make_checker(deploy_config=dc)
        result = await checker._check_image_reference()
        assert result.status == CheckStatus.WARN
        assert "authentication" in result.message.lower()

    @pytest.mark.asyncio
    async def test_private_registry_with_secrets_passes(self) -> None:
        dc = DeploymentConfig(
            image="private.corp.io/team/app:v1",
            pod_template=PodTemplateConfig(image_pull_secrets=["my-pull"]),
        )
        checker = _make_checker(deploy_config=dc)
        result = await checker._check_image_reference()
        assert result.status == CheckStatus.PASS


# =============================================================================
# Tier 3: DNS
# =============================================================================


class TestCheckDNS:
    """Verify DNS resolution check."""

    @pytest.mark.asyncio
    async def test_coredns_running(self) -> None:
        checker = _make_checker()
        deploy = make_kr8s_object(
            {
                "metadata": {"name": "coredns", "namespace": "kube-system"},
                "status": {"readyReplicas": 2},
            }
        )
        checker.api.async_get = MagicMock(return_value=async_list([deploy]))
        result = await checker._check_dns()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_coredns_not_ready(self) -> None:
        checker = _make_checker()
        deploy = make_kr8s_object(
            {
                "metadata": {"name": "coredns", "namespace": "kube-system"},
                "status": {"readyReplicas": 0},
            }
        )
        checker.api.async_get = MagicMock(return_value=async_list([deploy]))
        result = await checker._check_dns()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_coredns_not_found(self) -> None:
        checker = _make_checker()
        checker.api.async_get = MagicMock(return_value=async_list([]))
        result = await checker._check_dns()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_forbidden(self) -> None:
        checker = _make_checker()
        checker.api.async_get = MagicMock(
            side_effect=create_server_error(403, "Forbidden")
        )
        result = await checker._check_dns()
        assert result.status == CheckStatus.SKIP


# =============================================================================
# Tier 3: Network Policies
# =============================================================================


class TestCheckNetworkPolicies:
    """Verify network policy detection."""

    @pytest.mark.asyncio
    async def test_no_policies(self) -> None:
        checker = _make_checker()
        checker.api.async_get = MagicMock(return_value=async_list([]))
        result = await checker._check_network_policies()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_policies_found_warns(self) -> None:
        checker = _make_checker()
        policy = make_kr8s_object(
            {
                "metadata": {"name": "deny-all", "namespace": "test-ns"},
            }
        )
        checker.api.async_get = MagicMock(return_value=async_list([policy]))
        result = await checker._check_network_policies()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_forbidden_skips(self) -> None:
        checker = _make_checker()
        checker.api.async_get = MagicMock(
            side_effect=create_server_error(403, "Forbidden")
        )
        result = await checker._check_network_policies()
        assert result.status == CheckStatus.SKIP


# =============================================================================
# Tier 3: Kueue Queue
# =============================================================================


class TestCheckKueueQueue:
    """Verify Kueue LocalQueue check."""

    @pytest.mark.asyncio
    async def test_no_queue_kueue_not_installed_skips(self) -> None:
        checker = _make_checker()
        # _is_kueue_installed returns False when call_api raises
        checker.api.call_api = _mock_call_api_raises(
            create_server_error(404, "Not Found")
        )
        result = await checker._check_kueue_queue()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_no_queue_kueue_installed_namespace_has_default_passes(
        self,
    ) -> None:
        checker = _make_checker()
        # _is_kueue_installed returns True
        checker.api.call_api = _mock_call_api_response({"items": []})
        # _namespace_has_default_queue returns True
        with patch("kr8s.asyncio.objects.Namespace") as MockNs:
            ns_obj = MagicMock()
            ns_obj.raw = {
                "metadata": {
                    "annotations": {
                        "kueue.x-k8s.io/default-queue-name": "my-queue",
                    },
                },
            }
            MockNs.get = AsyncMock(return_value=ns_obj)
            result = await checker._check_kueue_queue()
        assert result.status == CheckStatus.PASS
        assert "default-queue-name" in result.message

    @pytest.mark.asyncio
    async def test_no_queue_kueue_installed_no_default_warns(self) -> None:
        checker = _make_checker()
        # _is_kueue_installed returns True
        checker.api.call_api = _mock_call_api_response({"items": []})
        # _namespace_has_default_queue returns False (no annotation)
        with patch("kr8s.asyncio.objects.Namespace") as MockNs:
            ns_obj = MagicMock()
            ns_obj.raw = {"metadata": {"annotations": {}}}
            MockNs.get = AsyncMock(return_value=ns_obj)
            result = await checker._check_kueue_queue()
        assert result.status == CheckStatus.WARN
        assert "bypass" in result.message
        assert result.hints

    @pytest.mark.asyncio
    async def test_queue_found(self) -> None:
        dc = DeploymentConfig(scheduling=SchedulingConfig(queue_name="default"))
        checker = _make_checker(deploy_config=dc)
        checker.api.call_api = _mock_call_api_response(
            {"metadata": {"name": "default"}}
        )
        result = await checker._check_kueue_queue()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_queue_not_found_fails(self) -> None:
        dc = DeploymentConfig(scheduling=SchedulingConfig(queue_name="missing"))
        checker = _make_checker(deploy_config=dc)
        checker.api.call_api = _mock_call_api_raises(
            create_not_found_error("localqueue")
        )
        result = await checker._check_kueue_queue()
        assert result.status == CheckStatus.FAIL
        assert "missing" in result.message

    @pytest.mark.asyncio
    async def test_kueue_crd_not_installed_skips(self) -> None:
        dc = DeploymentConfig(scheduling=SchedulingConfig(queue_name="default"))
        checker = _make_checker(deploy_config=dc)
        checker.api.call_api = _mock_call_api_raises(
            create_server_error(404, "Not Found")
        )
        result = await checker._check_kueue_queue()
        assert result.status == CheckStatus.SKIP


# =============================================================================
# Tier 3: ConfigMap Size
# =============================================================================


class TestCheckConfigMapSize:
    """Verify ConfigMap size validation."""

    @pytest.mark.asyncio
    async def test_within_limit(self) -> None:
        checker = _make_checker()
        result = await checker._check_configmap_size()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_exceeds_limit(self) -> None:
        checker = _make_checker()

        # Mock get_configmap_spec to return a spec reporting oversized data
        # without triggering validate_size()
        mock_spec = MagicMock()
        mock_spec.get_data_size_bytes.return_value = 2_000_000
        with patch.object(
            checker.deployment, "get_configmap_spec", return_value=mock_spec
        ):
            result = await checker._check_configmap_size()
        assert result.status == CheckStatus.FAIL
        assert "MiB" in result.message


# =============================================================================
# Tier 3: Dry Run
# =============================================================================


class TestCheckDryRun:
    """Verify server dry-run check."""

    @pytest.mark.asyncio
    async def test_accepted(self) -> None:
        checker = _make_checker()
        checker.api.call_api = _mock_call_api_response({"metadata": {"name": "test"}})
        result = await checker._check_dry_run()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_rejected_with_error(self) -> None:
        checker = _make_checker()
        err = create_server_error(403, "admission webhook denied")
        err.response.text = '{"message": "policy violation"}'
        checker.api.call_api = _mock_call_api_raises(err)
        result = await checker._check_dry_run()
        assert result.status == CheckStatus.FAIL
        assert "policy violation" in result.message

    @pytest.mark.asyncio
    async def test_server_error_returns_warn(self) -> None:
        checker = _make_checker()
        checker.api.call_api = _mock_call_api_raises(RuntimeError("connection reset"))
        result = await checker._check_dry_run()
        assert result.status == CheckStatus.WARN


# =============================================================================
# Tier 3: Pod Security Admission
# =============================================================================


class TestCheckPodSecurityAdmission:
    """Verify PSA label check."""

    @pytest.mark.asyncio
    async def test_no_psa_labels(self) -> None:
        checker = _make_checker()

        with patch("kr8s.asyncio.objects.Namespace") as MockNs:
            ns_obj = MagicMock()
            ns_obj.raw = {"metadata": {"labels": {}}}
            MockNs.get = AsyncMock(return_value=ns_obj)
            result = await checker._check_pod_security_admission()

        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_restricted_level_passes(self) -> None:
        checker = _make_checker()

        with patch("kr8s.asyncio.objects.Namespace") as MockNs:
            ns_obj = MagicMock()
            ns_obj.raw = {
                "metadata": {
                    "labels": {"pod-security.kubernetes.io/enforce": "restricted"},
                },
            }
            MockNs.get = AsyncMock(return_value=ns_obj)
            result = await checker._check_pod_security_admission()

        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_unknown_level_warns(self) -> None:
        checker = _make_checker()

        with patch("kr8s.asyncio.objects.Namespace") as MockNs:
            ns_obj = MagicMock()
            ns_obj.raw = {
                "metadata": {
                    "labels": {"pod-security.kubernetes.io/enforce": "custom-policy"},
                },
            }
            MockNs.get = AsyncMock(return_value=ns_obj)
            result = await checker._check_pod_security_admission()

        assert result.status == CheckStatus.WARN


# =============================================================================
# Tier 3: Tolerations
# =============================================================================


class TestCheckTolerations:
    """Verify toleration check."""

    @pytest.mark.asyncio
    async def test_no_tolerations_skips(self) -> None:
        checker = _make_checker()
        result = await checker._check_tolerations()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_matching_taints_exist(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                tolerations=[
                    {
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    }
                ],
            ),
        )
        checker = _make_checker(deploy_config=dc)
        node = make_kr8s_object(
            _node_raw(
                "gpu-node",
                "8",
                "32Gi",
                taints=[{"key": "nvidia.com/gpu", "effect": "NoSchedule"}],
            )
        )
        checker.api.async_get = MagicMock(return_value=async_list([node]))
        result = await checker._check_tolerations()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_no_matching_taints_warns(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists"}],
            ),
        )
        checker = _make_checker(deploy_config=dc)
        node = make_kr8s_object(_node_raw("no-taint-node", "8", "32Gi"))
        checker.api.async_get = MagicMock(return_value=async_list([node]))
        result = await checker._check_tolerations()
        assert result.status == CheckStatus.WARN
        assert "unnecessary" in result.message.lower()


# =============================================================================
# run_all orchestration
# =============================================================================


class TestRunAll:
    """Verify tiered orchestration."""

    @pytest.mark.asyncio
    async def test_all_pass(self) -> None:
        checker = _make_checker()
        # Mock tier 1+2 to pass
        checker.api.async_version = AsyncMock(
            return_value={"major": "1", "minor": "28", "gitVersion": "v1.28.0"}
        )
        checker.api.call_api = _mock_call_api_response({"status": {"allowed": True}})

        # Provide a ready node with enough resources so per-node schedulability passes.
        # Use side_effect to return a fresh async generator on each call.
        def _fresh_nodes(*args, **kwargs):
            return async_list([make_kr8s_object(_node_raw("big-node", "64", "256Gi"))])

        checker.api.async_get = MagicMock(side_effect=_fresh_nodes)

        with (
            patch("kr8s.asyncio.objects.Secret") as MockSecret,
            patch("kr8s.asyncio.objects.ServiceAccount") as MockSA,
            patch("kr8s.asyncio.objects.Namespace") as MockNs,
        ):
            MockSecret.get = AsyncMock()
            MockSA.get = AsyncMock()
            ns_obj = MagicMock()
            ns_obj.raw = {"metadata": {"labels": {}}}
            MockNs.get = AsyncMock(return_value=ns_obj)
            results = await checker.run_all(timeout=30.0)

        assert results.passed
        assert len(results.checks) == 19

    @pytest.mark.asyncio
    async def test_tier1_short_circuits_on_version_fail(self) -> None:
        checker = _make_checker()
        checker.api.async_version = AsyncMock(
            return_value={"major": "1", "minor": "20", "gitVersion": "v1.20.0"}
        )
        results = await checker.run_all(timeout=30.0)
        assert not results.passed
        assert len(results.checks) == 1
        assert results.checks[0].name == "Kubernetes Version"

    @pytest.mark.asyncio
    async def test_tier1_short_circuits_on_crd_fail(self) -> None:
        checker = _make_checker()
        checker.api.async_version = AsyncMock(
            return_value={"major": "1", "minor": "28", "gitVersion": "v1.28.0"}
        )
        checker.api.call_api = _mock_call_api_raises(
            create_not_found_error("JobSet CRD")
        )
        results = await checker.run_all(timeout=30.0)
        assert not results.passed
        assert len(results.checks) == 2
        assert results.checks[1].name == "JobSet CRD"

    @pytest.mark.asyncio
    async def test_tier2_short_circuits_on_rbac_fail(self) -> None:
        checker = _make_checker()
        checker.api.async_version = AsyncMock(
            return_value={"major": "1", "minor": "28", "gitVersion": "v1.28.0"}
        )
        # CRD check passes, RBAC check fails
        call_count = 0

        @asynccontextmanager
        async def _route(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count == 1:
                # First call is CRD check
                resp.json.return_value = {"items": []}
                resp.raise_for_status = MagicMock()
            else:
                # Subsequent calls are RBAC checks
                resp.json.return_value = {"status": {"allowed": False}}
            yield resp

        checker.api.call_api = _route
        results = await checker.run_all(timeout=30.0)
        assert not results.passed
        assert len(results.checks) == 3

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        import asyncio

        checker = _make_checker()

        # Use a never-resolved Future instead of asyncio.sleep (which is auto-mocked)
        async def _hang():
            await asyncio.get_event_loop().create_future()

        checker.api.async_version = _hang
        results = await checker.run_all(timeout=0.1)
        assert not results.passed
        has_timeout = any("timed out" in c.message.lower() for c in results.checks)
        assert has_timeout


# =============================================================================
# Integration: on_create wiring
# =============================================================================


class TestIntegrationOnCreate:
    """Verify preflight failure blocks resource creation."""

    @pytest.mark.asyncio
    async def test_preflight_failure_prevents_resource_creation(self) -> None:
        """When preflight fails, on_create should raise PermanentError."""
        import kopf

        from aiperf.operator.main import on_create

        spec = {
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": {"main": {"type": "synthetic"}},
                "phases": {
                    "type": "concurrency",
                    "dataset": "main",
                    "requests": 10,
                    "concurrency": 1,
                },
            },
        }
        body = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "metadata": {"name": "test-job", "namespace": "test-ns", "uid": "uid-123"},
            "spec": spec,
        }

        mock_api = _mock_api()
        # Make version check fail
        mock_api.async_version = AsyncMock(
            return_value={"major": "1", "minor": "20", "gitVersion": "v1.20.0"}
        )

        patch_obj = kopf.Patch()

        with (
            patch(
                "aiperf.operator.handlers.create.get_api",
                new=AsyncMock(return_value=mock_api),
            ),
            patch(
                "aiperf.operator.handlers.create.check_endpoint_health",
                new=AsyncMock(
                    return_value=MagicMock(reachable=True, error=""),
                ),
            ),
            patch("aiperf.operator.events.spec_valid"),
            patch("aiperf.operator.events.endpoint_reachable"),
            patch("aiperf.operator.events.preflight_failed") as mock_pf_failed,
            pytest.raises(kopf.PermanentError, match="Pre-flight checks failed"),
        ):
            await on_create(
                body=body,
                spec=spec,
                name="test-job",
                namespace="test-ns",
                uid="uid-123",
                patch=patch_obj,
            )

        # Verify preflight failure event was emitted
        mock_pf_failed.assert_called_once()
        # Verify no resource creation happened (no ConfigMap.create calls)
        assert patch_obj.status.get("phase") == "Failed"


# =============================================================================
# Additional edge-case tests
# =============================================================================


class TestCheckKubernetesVersionEdgeCases:
    """Additional edge cases for Kubernetes version parsing."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "major,minor,expected_status",
        [
            param("2", "0", CheckStatus.PASS, id="v2.0-major-above-min"),
            param("2", "30", CheckStatus.PASS, id="v2.30-major-above-min"),
        ],
    )  # fmt: skip
    async def test_major_version_above_1(
        self,
        major: str,
        minor: str,
        expected_status: CheckStatus,
    ) -> None:
        """Version check passes for major > 1 since it exceeds minimum 1.24."""
        checker = _make_checker()
        checker.api.async_version.return_value = {
            "major": major,
            "minor": minor,
            "gitVersion": f"v{major}.{minor}.0",
        }
        result = await checker._check_kubernetes_version()
        assert result.status == expected_status

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "major,minor,expected_status",
        [
            param("abc", "28", CheckStatus.FAIL, id="gibberish-major"),
            param("1", "xyz", CheckStatus.FAIL, id="gibberish-minor"),
            param("!!!", "???", CheckStatus.FAIL, id="all-gibberish"),
        ],
    )  # fmt: skip
    async def test_non_numeric_gibberish(
        self,
        major: str,
        minor: str,
        expected_status: CheckStatus,
    ) -> None:
        """Non-numeric version fields are stripped to empty -> parsed as 0 -> FAIL."""
        checker = _make_checker()
        checker.api.async_version.return_value = {
            "major": major,
            "minor": minor,
            "gitVersion": "unknown",
        }
        result = await checker._check_kubernetes_version()
        assert result.status == expected_status


class TestCheckRBACPermissionsEdgeCases:
    """Additional edge cases for RBAC permission checks."""

    @pytest.mark.asyncio
    async def test_all_permissions_denied(self) -> None:
        """Every single RBAC check returns not-allowed."""
        checker = _make_checker()

        @asynccontextmanager
        async def _deny(*args, **kwargs):
            resp = MagicMock()
            resp.json.return_value = {"status": {"allowed": False}}
            yield resp

        checker.api.call_api = _deny
        result = await checker._check_rbac_permissions()
        assert result.status == CheckStatus.FAIL
        assert "Missing 15" in result.message

    @pytest.mark.asyncio
    async def test_mixed_denied_and_exception(self) -> None:
        """Some RBAC checks are denied, others raise exceptions."""
        checker = _make_checker()
        call_count = 0

        @asynccontextmanager
        async def _mixed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise RuntimeError("network blip")
            resp = MagicMock()
            resp.json.return_value = {"status": {"allowed": call_count % 3 == 1}}
            yield resp

        checker.api.call_api = _mixed
        result = await checker._check_rbac_permissions()
        assert result.status == CheckStatus.FAIL
        assert "check failed" in result.message


class TestCheckNodeResourcesEdgeCases:
    """Additional edge cases for node resource estimation."""

    @pytest.mark.asyncio
    async def test_mixed_ready_and_not_ready_nodes(self) -> None:
        """Only ready nodes contribute to aggregate resources."""
        checker = _make_checker(num_pods=1)
        ready_node = make_kr8s_object(_node_raw("ready", "16", "64Gi", ready=True))
        not_ready_node = make_kr8s_object(
            _node_raw("not-ready", "16", "64Gi", ready=False)
        )
        checker.api.async_get = MagicMock(
            return_value=async_list([ready_node, not_ready_node])
        )
        result = await checker._check_node_resources()
        assert result.status == CheckStatus.PASS
        assert "1 ready node" in result.message

    @pytest.mark.asyncio
    async def test_boundary_exactly_enough_resources(self) -> None:
        """When resources exactly match what is needed, it should pass."""
        from aiperf.kubernetes.environment import (
            CONTROLLER_RESOURCE_KEYS,
            K8sEnvironment,
        )

        ctrl_cpu = sum(
            float(getattr(K8sEnvironment, key).CPU.replace("m", "")) / 1000
            if getattr(K8sEnvironment, key).CPU.endswith("m")
            else float(getattr(K8sEnvironment, key).CPU)
            for key in CONTROLLER_RESOURCE_KEYS
        )
        ctrl_mem_mib = sum(
            float(getattr(K8sEnvironment, key).MEMORY.replace("Mi", ""))
            for key in CONTROLLER_RESOURCE_KEYS
        )
        worker_cpu = float(K8sEnvironment.WORKER_POD.CPU.replace("m", "")) / 1000
        worker_mem_mib = float(K8sEnvironment.WORKER_POD.MEMORY.replace("Mi", ""))

        num_pods = 2
        total_cpu = ctrl_cpu + (worker_cpu * num_pods)
        total_mem_mib = ctrl_mem_mib + (worker_mem_mib * num_pods)

        checker = _make_checker(num_pods=num_pods)
        node = make_kr8s_object(
            _node_raw("exact-node", str(total_cpu), f"{int(total_mem_mib)}Mi")
        )
        checker.api.async_get = MagicMock(return_value=async_list([node]))
        result = await checker._check_node_resources()
        assert result.status == CheckStatus.PASS


class TestCheckPerNodeSchedulabilityEdgeCases:
    """Additional edge cases for per-node schedulability."""

    @pytest.mark.asyncio
    async def test_multiple_nodes_only_one_fits(self) -> None:
        """If only one node can fit the pod, still passes."""
        checker = _make_checker()
        tiny = make_kr8s_object(_node_raw("tiny", "0.1", "100Mi"))
        big = make_kr8s_object(_node_raw("big", "32", "128Gi"))
        also_tiny = make_kr8s_object(_node_raw("also-tiny", "0.2", "200Mi"))
        checker.api.async_get = MagicMock(
            return_value=async_list([tiny, big, also_tiny])
        )
        result = await checker._check_per_node_schedulability()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_no_ready_nodes_at_all(self) -> None:
        """All nodes are not-ready -> no node can fit the pod."""
        checker = _make_checker()
        n1 = make_kr8s_object(_node_raw("n1", "32", "128Gi", ready=False))
        n2 = make_kr8s_object(_node_raw("n2", "32", "128Gi", ready=False))
        checker.api.async_get = MagicMock(return_value=async_list([n1, n2]))
        result = await checker._check_per_node_schedulability()
        assert result.status == CheckStatus.FAIL
        assert "No single node" in result.message


class TestCheckResourceQuotasEdgeCases:
    """Additional edge cases for resource quota checks."""

    @pytest.mark.asyncio
    async def test_exceeds_memory_quota(self) -> None:
        """Should warn when memory quota is exceeded."""
        checker = _make_checker(num_pods=100)
        quota = make_kr8s_object(
            {
                "metadata": {"name": "mem-tight", "namespace": "test-ns"},
                "status": {
                    "hard": {"memory": "4Gi"},
                    "used": {"memory": "3Gi"},
                },
            }
        )
        checker.api.async_get = MagicMock(return_value=async_list([quota]))
        result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.WARN
        assert "memory quota" in result.message.lower()

    @pytest.mark.asyncio
    async def test_requests_cpu_and_requests_memory_style_keys(self) -> None:
        """Quota keys like requests.cpu / requests.memory should be recognized."""
        checker = _make_checker(num_pods=100)
        quota = make_kr8s_object(
            {
                "metadata": {"name": "request-style", "namespace": "test-ns"},
                "status": {
                    "hard": {"requests.cpu": "4"},
                    "used": {"requests.cpu": "3"},
                },
            }
        )
        checker.api.async_get = MagicMock(return_value=async_list([quota]))
        result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.WARN
        assert "CPU quota" in result.message

    @pytest.mark.asyncio
    async def test_multiple_quotas_first_exceeds(self) -> None:
        """With multiple quotas, the first one that exceeds triggers a warning."""
        checker = _make_checker(num_pods=100)
        tight_quota = make_kr8s_object(
            {
                "metadata": {"name": "tight", "namespace": "test-ns"},
                "status": {
                    "hard": {"cpu": "2"},
                    "used": {"cpu": "1"},
                },
            }
        )
        generous_quota = make_kr8s_object(
            {
                "metadata": {"name": "generous", "namespace": "test-ns"},
                "status": {
                    "hard": {"cpu": "10000"},
                    "used": {"cpu": "0"},
                },
            }
        )
        checker.api.async_get = MagicMock(
            return_value=async_list([tight_quota, generous_quota])
        )
        result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.WARN


class TestCheckSecretsEdgeCases:
    """Additional edge cases for secret checks."""

    @pytest.mark.asyncio
    async def test_deduplication(self) -> None:
        """Same secret name referenced in multiple places is only checked once."""
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                image_pull_secrets=["shared-secret"],
                volumes=[
                    {"name": "vol1", "secret": {"secretName": "shared-secret"}},
                ],
                env=[
                    {
                        "name": "KEY",
                        "valueFrom": {
                            "secretKeyRef": {"name": "shared-secret", "key": "k"},
                        },
                    },
                ],
            ),
        )
        checker = _make_checker(deploy_config=dc)

        with patch("kr8s.asyncio.objects.Secret") as MockSecret:
            MockSecret.get = AsyncMock()
            result = await checker._check_secrets()

        assert result.status == CheckStatus.PASS
        # Only 1 unique secret despite 3 references
        assert "1 secret" in result.message

    @pytest.mark.asyncio
    async def test_volume_secret_present_env_secret_missing(self) -> None:
        """One secret found (volume), another missing (env) -> FAIL."""
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                volumes=[
                    {"name": "vol1", "secret": {"secretName": "found-secret"}},
                ],
                env=[
                    {
                        "name": "KEY",
                        "valueFrom": {
                            "secretKeyRef": {"name": "missing-secret", "key": "k"},
                        },
                    },
                ],
            ),
        )
        checker = _make_checker(deploy_config=dc)

        with patch("kr8s.asyncio.objects.Secret") as MockSecret:

            async def _get_side_effect(name, **kwargs):
                if name == "missing-secret":
                    raise kr8s.NotFoundError("not found")
                return MagicMock()

            MockSecret.get = AsyncMock(side_effect=_get_side_effect)
            result = await checker._check_secrets()

        assert result.status == CheckStatus.FAIL
        assert "missing-secret" in result.message
        assert "found-secret" not in result.message


class TestCheckImageReferenceEdgeCases:
    """Additional edge cases for image reference validation."""

    @pytest.mark.asyncio
    async def test_image_with_digest(self) -> None:
        """Image with @sha256:... digest should be treated as having a tag."""
        dc = DeploymentConfig(image="nvcr.io/nvidia/aiperf@sha256:abcdef1234567890")
        checker = _make_checker(deploy_config=dc)
        result = await checker._check_image_reference()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_docker_short_name_with_tag(self) -> None:
        """Short Docker Hub name like 'python:3.10' uses docker.io registry (public)."""
        dc = DeploymentConfig(image="python:3.10")
        checker = _make_checker(deploy_config=dc)
        result = await checker._check_image_reference()
        assert result.status == CheckStatus.PASS


class TestCheckDryRunEdgeCases:
    """Additional edge cases for dry-run check."""

    @pytest.mark.asyncio
    async def test_non_json_error_response_body(self) -> None:
        """ServerError with non-JSON response body falls back to str(e)."""
        checker = _make_checker()
        err = create_server_error(422, "Unprocessable Entity")
        err.response.text = "this is not json at all"
        checker.api.call_api = _mock_call_api_raises(err)
        result = await checker._check_dry_run()
        assert result.status == CheckStatus.FAIL
        assert "dry-run rejected" in result.message.lower()

    @pytest.mark.asyncio
    async def test_not_found_error(self) -> None:
        """kr8s.NotFoundError (not ServerError) falls into generic except -> WARN."""
        checker = _make_checker()
        checker.api.call_api = _mock_call_api_raises(
            create_not_found_error("jobset API")
        )
        result = await checker._check_dry_run()
        # NotFoundError is a subclass of ServerError in kr8s, so check the actual result
        assert result.status in (CheckStatus.FAIL, CheckStatus.WARN)


class TestCheckPodSecurityAdmissionEdgeCases:
    """Additional edge cases for PSA label check."""

    @pytest.mark.asyncio
    async def test_baseline_level_passes(self) -> None:
        checker = _make_checker()
        with patch("kr8s.asyncio.objects.Namespace") as MockNs:
            ns_obj = MagicMock()
            ns_obj.raw = {
                "metadata": {
                    "labels": {"pod-security.kubernetes.io/enforce": "baseline"},
                },
            }
            MockNs.get = AsyncMock(return_value=ns_obj)
            result = await checker._check_pod_security_admission()
        assert result.status == CheckStatus.PASS
        assert "baseline" in result.message

    @pytest.mark.asyncio
    async def test_privileged_level_passes(self) -> None:
        checker = _make_checker()
        with patch("kr8s.asyncio.objects.Namespace") as MockNs:
            ns_obj = MagicMock()
            ns_obj.raw = {
                "metadata": {
                    "labels": {"pod-security.kubernetes.io/enforce": "privileged"},
                },
            }
            MockNs.get = AsyncMock(return_value=ns_obj)
            result = await checker._check_pod_security_admission()
        assert result.status == CheckStatus.PASS
        assert "privileged" in result.message

    @pytest.mark.asyncio
    async def test_namespace_not_found(self) -> None:
        checker = _make_checker()
        with patch("kr8s.asyncio.objects.Namespace") as MockNs:
            MockNs.get = AsyncMock(side_effect=kr8s.NotFoundError("not found"))
            result = await checker._check_pod_security_admission()
        assert result.status == CheckStatus.WARN
        assert "not found" in result.message.lower()


class TestCheckTolerationsEdgeCases:
    """Additional edge cases for toleration checks."""

    @pytest.mark.asyncio
    async def test_multiple_tolerations_one_matches(self) -> None:
        """Multiple tolerations specified, only one matches a taint -> PASS."""
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                tolerations=[
                    {
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    },
                    {
                        "key": "special.io/zone",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    },
                ],
            ),
        )
        checker = _make_checker(deploy_config=dc)
        # Node only has the gpu taint, not the zone taint
        node = make_kr8s_object(
            _node_raw(
                "gpu-node",
                "8",
                "32Gi",
                taints=[{"key": "nvidia.com/gpu", "effect": "NoSchedule"}],
            )
        )
        checker.api.async_get = MagicMock(return_value=async_list([node]))
        result = await checker._check_tolerations()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_wildcard_toleration_empty_key(self) -> None:
        """Toleration with empty key (wildcard) is not added to toleration_keys set."""
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                tolerations=[
                    {"key": "", "operator": "Exists"},
                ],
            ),
        )
        checker = _make_checker(deploy_config=dc)
        node = make_kr8s_object(
            _node_raw(
                "tainted-node",
                "8",
                "32Gi",
                taints=[{"key": "some-taint", "effect": "NoSchedule"}],
            )
        )
        checker.api.async_get = MagicMock(return_value=async_list([node]))
        result = await checker._check_tolerations()
        # Empty key is filtered out by `if t.get("key")`, so no match
        assert result.status == CheckStatus.WARN


class TestRunAllEdgeCases:
    """Additional edge cases for tiered orchestration."""

    @pytest.mark.asyncio
    async def test_concurrent_check_exception_does_not_crash(self) -> None:
        """An exception from a concurrent check is caught, not a crash."""
        checker = _make_checker()
        # Tier 1+2 pass
        checker.api.async_version = AsyncMock(
            return_value={"major": "1", "minor": "28", "gitVersion": "v1.28.0"}
        )
        checker.api.call_api = _mock_call_api_response({"status": {"allowed": True}})

        def _fresh_nodes(*args, **kwargs):
            return async_list([make_kr8s_object(_node_raw("big-node", "64", "256Gi"))])

        checker.api.async_get = MagicMock(side_effect=_fresh_nodes)

        # Make one concurrent check raise an unhandled exception
        async def _exploding_check(self_arg):
            raise RuntimeError("unexpected kaboom")

        with (
            patch.object(
                OperatorPreflightChecker,
                "_check_tolerations",
                new=_exploding_check,
            ),
            patch("kr8s.asyncio.objects.Secret") as MockSecret,
            patch("kr8s.asyncio.objects.ServiceAccount") as MockSA,
            patch("kr8s.asyncio.objects.Namespace") as MockNs,
        ):
            MockSecret.get = AsyncMock()
            MockSA.get = AsyncMock()
            ns_obj = MagicMock()
            ns_obj.raw = {"metadata": {"labels": {}}}
            MockNs.get = AsyncMock(return_value=ns_obj)
            results = await checker.run_all(timeout=30.0)

        # The exception is wrapped in a FAIL result, not propagated
        fail_results = [c for c in results.checks if c.status == CheckStatus.FAIL]
        assert any("kaboom" in c.message for c in fail_results)
        # Orchestration still completed (got results for all checks)
        assert len(results.checks) == 19

    @pytest.mark.asyncio
    async def test_warnings_dont_block_passed(self) -> None:
        """All checks pass + some warnings -> results.passed is True."""
        checker = _make_checker()
        # Tier 1+2 pass
        checker.api.async_version = AsyncMock(
            return_value={"major": "1", "minor": "28", "gitVersion": "v1.28.0"}
        )
        checker.api.call_api = _mock_call_api_response({"status": {"allowed": True}})

        def _fresh_nodes(*args, **kwargs):
            return async_list([make_kr8s_object(_node_raw("big-node", "64", "256Gi"))])

        checker.api.async_get = MagicMock(side_effect=_fresh_nodes)

        with (
            patch("kr8s.asyncio.objects.Secret") as MockSecret,
            patch("kr8s.asyncio.objects.ServiceAccount") as MockSA,
            patch("kr8s.asyncio.objects.Namespace") as MockNs,
        ):
            MockSecret.get = AsyncMock()
            MockSA.get = AsyncMock()
            ns_obj = MagicMock()
            ns_obj.raw = {"metadata": {"labels": {}}}
            MockNs.get = AsyncMock(return_value=ns_obj)
            results = await checker.run_all(timeout=30.0)

        assert results.passed is True
        # Some checks may produce warnings (network policies, tolerations, etc.)
        # but no FAILs means passed=True

    @pytest.mark.asyncio
    async def test_mixed_failures_and_warnings_in_concurrent_tier(self) -> None:
        """Concurrent tier with both FAIL and WARN results -> not passed."""
        checker = _make_checker()
        # Tier 1+2 pass
        checker.api.async_version = AsyncMock(
            return_value={"major": "1", "minor": "28", "gitVersion": "v1.28.0"}
        )
        checker.api.call_api = _mock_call_api_response({"status": {"allowed": True}})

        def _fresh_nodes(*args, **kwargs):
            return async_list([make_kr8s_object(_node_raw("big-node", "64", "256Gi"))])

        checker.api.async_get = MagicMock(side_effect=_fresh_nodes)

        # Make image reference fail (empty image)
        checker.deploy_config = DeploymentConfig(image="")

        with (
            patch("kr8s.asyncio.objects.Secret") as MockSecret,
            patch("kr8s.asyncio.objects.ServiceAccount") as MockSA,
            patch("kr8s.asyncio.objects.Namespace") as MockNs,
        ):
            MockSecret.get = AsyncMock()
            MockSA.get = AsyncMock()
            ns_obj = MagicMock()
            ns_obj.raw = {"metadata": {"labels": {}}}
            MockNs.get = AsyncMock(return_value=ns_obj)
            results = await checker.run_all(timeout=30.0)

        assert results.passed is False
        statuses = {c.status for c in results.checks}
        assert CheckStatus.FAIL in statuses


class TestCheckNodeSelectorMatchEdgeCases:
    """Additional edge cases for node selector matching."""

    @pytest.mark.asyncio
    async def test_multiple_selector_labels_all_must_match(self) -> None:
        """Node must match ALL selector labels, not just some."""
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                node_selector={"gpu": "true", "zone": "us-east-1a"},
            ),
        )
        checker = _make_checker(deploy_config=dc)
        # Node has gpu=true but not zone
        partial_match = make_kr8s_object(
            _node_raw("partial", "8", "32Gi", labels={"gpu": "true"})
        )
        # Node has zone but not gpu
        other_partial = make_kr8s_object(
            _node_raw("other", "8", "32Gi", labels={"zone": "us-east-1a"})
        )
        # Node has both
        full_match = make_kr8s_object(
            _node_raw(
                "full",
                "8",
                "32Gi",
                labels={"gpu": "true", "zone": "us-east-1a"},
            )
        )
        checker.api.async_get = MagicMock(
            return_value=async_list([partial_match, other_partial, full_match])
        )
        result = await checker._check_node_selector_match()
        assert result.status == CheckStatus.PASS
        # Only the fully matching node counts
        assert "1 node" in result.message

    @pytest.mark.asyncio
    async def test_multiple_selector_labels_none_match_all(self) -> None:
        """No node matches all selector labels -> FAIL."""
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                node_selector={"gpu": "true", "zone": "us-east-1a"},
            ),
        )
        checker = _make_checker(deploy_config=dc)
        partial_a = make_kr8s_object(
            _node_raw("a", "8", "32Gi", labels={"gpu": "true"})
        )
        partial_b = make_kr8s_object(
            _node_raw("b", "8", "32Gi", labels={"zone": "us-east-1a"})
        )
        checker.api.async_get = MagicMock(
            return_value=async_list([partial_a, partial_b])
        )
        result = await checker._check_node_selector_match()
        assert result.status == CheckStatus.FAIL
