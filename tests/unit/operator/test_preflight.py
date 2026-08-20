# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.operator.preflight module.

Focuses on:
- OperatorPreflightChecker individual check methods (mocked kubernetes_asyncio API)
- Tiered orchestration (short-circuit on tier 1/2, concurrent tier 3+)
- Timeout handling
- Error messages include actionable remediation hints
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.config import AIPerfConfig
from aiperf.config.deployment import (
    DeploymentConfig,
    PodTemplateConfig,
    SchedulingConfig,
)
from aiperf.kubernetes.preflight import CheckStatus
from aiperf.kubernetes.resources import KubernetesDeployment
from aiperf.operator.preflight import (
    OperatorPreflightChecker,
    _is_node_ready_typed,
)

# =============================================================================
# Helpers — build typed-ish objects
# =============================================================================


def _mock_api() -> MagicMock:
    """Build a MagicMock ApiClient (we patch the API constructors per-test)."""
    return MagicMock()


def _version_obj(major: str = "1", minor: str = "28", git_version: str | None = None):
    obj = MagicMock()
    obj.major = major
    obj.minor = minor
    obj.git_version = git_version or f"v{major}.{minor}.0"
    return obj


def _node(
    name: str,
    cpu: str,
    memory: str,
    ready: bool = True,
    labels: dict[str, str] | None = None,
    taints: list[dict[str, Any]] | None = None,
):
    """Build a typed-ish V1Node mock."""
    node = MagicMock()
    node.metadata = MagicMock()
    node.metadata.name = name
    node.metadata.labels = labels or {}
    status = MagicMock()
    status.conditions = [MagicMock(type="Ready", status=("True" if ready else "False"))]
    status.allocatable = {"cpu": cpu, "memory": memory}
    node.status = status
    spec = MagicMock()
    taint_objs = []
    for t in taints or []:
        to = MagicMock()
        to.key = t.get("key")
        to.value = t.get("value")
        to.effect = t.get("effect")
        taint_objs.append(to)
    spec.taints = taint_objs or None
    node.spec = spec
    return node


def _deploy(name: str, namespace: str, ready_replicas: int):
    d = MagicMock()
    d.metadata = MagicMock()
    d.metadata.name = name
    d.metadata.namespace = namespace
    d.status = MagicMock()
    d.status.ready_replicas = ready_replicas
    return d


def _list(items: list) -> MagicMock:
    res = MagicMock()
    res.items = items
    return res


def _sample_config() -> AIPerfConfig:
    """Create a minimal AIPerfConfig for testing."""
    return AIPerfConfig(
        benchmark={
            "models": ["test-model"],
            "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
            "datasets": [
                {
                    "name": "main",
                    "type": "synthetic",
                    "entries": 10,
                    "prompts": {"isl": 32, "osl": 16},
                }
            ],
            "phases": [
                {
                    "name": "default",
                    "type": "concurrency",
                    "kind": "profiling",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        }
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


def _patch_version(major: str = "1", minor: str = "28", git_version: str | None = None):
    """Patch ``client.VersionApi`` to return a pinned version."""
    mock_v = MagicMock()
    mock_v.get_code = AsyncMock(
        return_value=_version_obj(major=major, minor=minor, git_version=git_version)
    )
    return patch(
        "aiperf.operator.preflight.client.VersionApi",
        return_value=mock_v,
    )


def _patch_core_v1(mock_core: MagicMock):
    return patch(
        "aiperf.operator.preflight.client.CoreV1Api",
        return_value=mock_core,
    )


def _patch_apps_v1(mock_apps: MagicMock):
    return patch(
        "aiperf.operator.preflight.client.AppsV1Api",
        return_value=mock_apps,
    )


def _patch_networking_v1(mock_net: MagicMock):
    return patch(
        "aiperf.operator.preflight.client.NetworkingV1Api",
        return_value=mock_net,
    )


def _patch_custom_objects(mock_custom: MagicMock):
    return patch(
        "aiperf.operator.preflight.client.CustomObjectsApi",
        return_value=mock_custom,
    )


def _patch_auth_v1(mock_auth: MagicMock):
    return patch(
        "aiperf.operator.preflight.client.AuthorizationV1Api",
        return_value=mock_auth,
    )


# =============================================================================
# _is_node_ready_typed helper
# =============================================================================


class TestIsNodeReady:
    """Verify _is_node_ready_typed helper."""

    def test_ready_node(self) -> None:
        n = _node("n1", "4", "16Gi", ready=True)
        assert _is_node_ready_typed(n) is True

    def test_not_ready_node(self) -> None:
        n = _node("n1", "4", "16Gi", ready=False)
        assert _is_node_ready_typed(n) is False

    def test_no_conditions(self) -> None:
        n = MagicMock()
        n.status = MagicMock()
        n.status.conditions = []
        assert _is_node_ready_typed(n) is False


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
        with _patch_version(major=major, minor=minor):
            result = await checker._check_kubernetes_version()
        assert result.status == expected_status

    @pytest.mark.asyncio
    async def test_gke_version_with_plus_suffix(self) -> None:
        """GKE/EKS versions like '28+' should parse correctly."""
        checker = _make_checker()
        with _patch_version(major="1", minor="28+", git_version="v1.28.2-gke.1"):
            result = await checker._check_kubernetes_version()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_empty_version_fields(self) -> None:
        checker = _make_checker()
        with _patch_version(major="", minor=None, git_version="unknown"):
            result = await checker._check_kubernetes_version()
        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_fail_message_includes_upgrade_hint(self) -> None:
        checker = _make_checker()
        with _patch_version(major="1", minor="22"):
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
        mock_custom = MagicMock(
            list_cluster_custom_object=AsyncMock(return_value={"items": []})
        )
        with _patch_custom_objects(mock_custom):
            result = await checker._check_jobset_crd()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_crd_not_found(self) -> None:
        checker = _make_checker()
        mock_custom = MagicMock(
            list_cluster_custom_object=AsyncMock(
                side_effect=ApiException(status=404, reason="NotFound")
            )
        )
        with _patch_custom_objects(mock_custom):
            result = await checker._check_jobset_crd()
        assert result.status == CheckStatus.FAIL
        assert "Install" in result.message

    @pytest.mark.asyncio
    async def test_crd_server_error(self) -> None:
        checker = _make_checker()
        mock_custom = MagicMock(
            list_cluster_custom_object=AsyncMock(
                side_effect=ApiException(status=503, reason="Unavailable")
            )
        )
        with _patch_custom_objects(mock_custom):
            result = await checker._check_jobset_crd()
        assert result.status == CheckStatus.FAIL
        assert "503" in result.message


# =============================================================================
# Tier 2: RBAC Permissions
# =============================================================================


def _make_review(allowed: bool):
    review = MagicMock()
    review.status = MagicMock()
    review.status.allowed = allowed
    return review


class TestCheckRBACPermissions:
    """Verify RBAC permission checks."""

    @pytest.mark.asyncio
    async def test_all_permissions_granted(self) -> None:
        checker = _make_checker()
        mock_auth = MagicMock(
            create_self_subject_access_review=AsyncMock(return_value=_make_review(True))
        )
        with _patch_auth_v1(mock_auth):
            result = await checker._check_rbac_permissions()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_some_permissions_denied(self) -> None:
        checker = _make_checker()
        call_count = 0

        async def _alternating(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_review(call_count % 2 == 0)

        mock_auth = MagicMock(
            create_self_subject_access_review=AsyncMock(side_effect=_alternating)
        )
        with _patch_auth_v1(mock_auth):
            result = await checker._check_rbac_permissions()
        assert result.status == CheckStatus.FAIL
        assert "Missing" in result.message
        assert "namespace" in result.message.lower()

    @pytest.mark.asyncio
    async def test_rbac_check_exception_treated_as_transient_warn(self) -> None:
        """A RuntimeError from the apiserver is transient — WARN, not FAIL.

        ``RuntimeError`` is in ``_CLUSTER_API_ERRORS`` (network/aiohttp glue
        sometimes wraps connection issues as RuntimeError). We can't say the
        permission is missing because we never got an answer.
        """
        checker = _make_checker()
        mock_auth = MagicMock(
            create_self_subject_access_review=AsyncMock(
                side_effect=RuntimeError("network")
            )
        )
        with _patch_auth_v1(mock_auth):
            result = await checker._check_rbac_permissions()
        assert result.status == CheckStatus.WARN
        assert "transient" in result.message.lower()


# =============================================================================
# Tier 3: JobSet Controller
# =============================================================================


class TestCheckJobSetController:
    """Verify JobSet controller detection."""

    @pytest.mark.asyncio
    async def test_controller_running(self) -> None:
        checker = _make_checker()
        deploy = _deploy("jobset-controller-manager", "jobset-system", 1)
        mock_apps = MagicMock(
            list_namespaced_deployment=AsyncMock(return_value=_list([deploy]))
        )
        with _patch_apps_v1(mock_apps):
            result = await checker._check_jobset_controller()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_controller_found_not_ready(self) -> None:
        checker = _make_checker()
        deploy = _deploy("jobset-controller-manager", "jobset-system", 0)
        mock_apps = MagicMock(
            list_namespaced_deployment=AsyncMock(return_value=_list([deploy]))
        )
        with _patch_apps_v1(mock_apps):
            result = await checker._check_jobset_controller()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_controller_not_found(self) -> None:
        checker = _make_checker()
        mock_apps = MagicMock(
            list_namespaced_deployment=AsyncMock(return_value=_list([]))
        )
        with _patch_apps_v1(mock_apps):
            result = await checker._check_jobset_controller()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_controller_forbidden(self) -> None:
        checker = _make_checker()
        mock_apps = MagicMock(
            list_namespaced_deployment=AsyncMock(
                side_effect=ApiException(status=403, reason="Forbidden")
            )
        )
        with _patch_apps_v1(mock_apps):
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
        mock_core = MagicMock(
            read_namespaced_service_account=AsyncMock(return_value=MagicMock())
        )
        with _patch_core_v1(mock_core):
            result = await checker._check_service_account()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_sa_not_found(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(service_account_name="my-sa"),
        )
        checker = _make_checker(deploy_config=dc)
        mock_core = MagicMock(
            read_namespaced_service_account=AsyncMock(
                side_effect=ApiException(status=404, reason="NotFound")
            )
        )
        with _patch_core_v1(mock_core):
            result = await checker._check_service_account()
        assert result.status == CheckStatus.FAIL


# =============================================================================
# Tier 3: Node Resources / Node Selector / Per-Node Schedulability
# =============================================================================


class TestCheckNodeResources:
    @pytest.mark.asyncio
    async def test_enough_resources(self) -> None:
        checker = _make_checker()
        n = _node("big-node", "16", "64Gi")
        mock_core = MagicMock(list_node=AsyncMock(return_value=_list([n])))
        with _patch_core_v1(mock_core):
            result = await checker._check_node_resources()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_not_enough_resources(self) -> None:
        checker = _make_checker()
        n = _node("tiny-node", "1", "1Gi")
        mock_core = MagicMock(list_node=AsyncMock(return_value=_list([n])))
        with _patch_core_v1(mock_core):
            result = await checker._check_node_resources()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_no_nodes(self) -> None:
        checker = _make_checker()
        mock_core = MagicMock(list_node=AsyncMock(return_value=_list([])))
        with _patch_core_v1(mock_core):
            result = await checker._check_node_resources()
        assert result.status == CheckStatus.WARN


class TestCheckNodeSelectorMatch:
    @pytest.mark.asyncio
    async def test_no_selector_skips(self) -> None:
        checker = _make_checker()
        result = await checker._check_node_selector_match()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_matching_nodes_found(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(node_selector={"gpu": "true"}),
        )
        checker = _make_checker(deploy_config=dc)
        n = _node("big-node", "16", "64Gi", labels={"gpu": "true"})
        mock_core = MagicMock(list_node=AsyncMock(return_value=_list([n])))
        with _patch_core_v1(mock_core):
            result = await checker._check_node_selector_match()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_no_matching_nodes_fails(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(node_selector={"gpu": "true"}),
        )
        checker = _make_checker(deploy_config=dc)
        n = _node("big-node", "16", "64Gi", labels={"gpu": "false"})
        mock_core = MagicMock(list_node=AsyncMock(return_value=_list([n])))
        with _patch_core_v1(mock_core):
            result = await checker._check_node_selector_match()
        assert result.status == CheckStatus.FAIL


class TestCheckPerNodeSchedulability:
    @pytest.mark.asyncio
    async def test_fits(self) -> None:
        checker = _make_checker()
        n = _node("big-node", "16", "64Gi")
        mock_core = MagicMock(list_node=AsyncMock(return_value=_list([n])))
        with _patch_core_v1(mock_core):
            result = await checker._check_per_node_schedulability()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_does_not_fit(self) -> None:
        checker = _make_checker()
        n = _node("tiny-node", "0.1", "100Mi")
        mock_core = MagicMock(list_node=AsyncMock(return_value=_list([n])))
        with _patch_core_v1(mock_core):
            result = await checker._check_per_node_schedulability()
        assert result.status == CheckStatus.FAIL


# =============================================================================
# Tier 3: Resource Quotas
# =============================================================================


class TestCheckResourceQuotas:
    @pytest.mark.asyncio
    async def test_no_quotas_passes(self) -> None:
        checker = _make_checker()
        mock_core = MagicMock(
            list_namespaced_resource_quota=AsyncMock(return_value=_list([]))
        )
        with _patch_core_v1(mock_core):
            result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_quota_error_warns(self) -> None:
        checker = _make_checker()
        mock_core = MagicMock(
            list_namespaced_resource_quota=AsyncMock(
                side_effect=ApiException(status=403, reason="Forbidden")
            )
        )
        with _patch_core_v1(mock_core):
            result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_cpu_quota_overcommit_fails(self) -> None:
        checker = _make_checker(num_pods=10)
        quota = MagicMock()
        quota.status = MagicMock()
        quota.status.hard = {"requests.cpu": "1"}
        quota.status.used = {"requests.cpu": "0"}
        mock_core = MagicMock(
            list_namespaced_resource_quota=AsyncMock(return_value=_list([quota]))
        )
        with _patch_core_v1(mock_core):
            result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.FAIL
        assert "CPU quota" in result.message

    @pytest.mark.asyncio
    async def test_memory_quota_overcommit_fails(self) -> None:
        checker = _make_checker()
        quota = MagicMock()
        quota.status = MagicMock()
        quota.status.hard = {"requests.memory": "256Mi"}
        quota.status.used = {"requests.memory": "0"}
        mock_core = MagicMock(
            list_namespaced_resource_quota=AsyncMock(return_value=_list([quota]))
        )
        with _patch_core_v1(mock_core):
            result = await checker._check_resource_quotas()
        assert result.status == CheckStatus.FAIL
        assert "memory quota" in result.message


# =============================================================================
# Tier 3: Secrets
# =============================================================================


class TestCheckSecrets:
    @pytest.mark.asyncio
    async def test_no_secrets_referenced_skips(self) -> None:
        checker = _make_checker()
        result = await checker._check_secrets()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_all_secrets_exist(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(image_pull_secrets=[{"name": "regcred"}]),
        )
        checker = _make_checker(deploy_config=dc)
        mock_core = MagicMock(
            read_namespaced_secret=AsyncMock(return_value=MagicMock())
        )
        with _patch_core_v1(mock_core):
            result = await checker._check_secrets()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_present_secret_missing_referenced_key_fails(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                env=[
                    {
                        "name": "AIPERF_INJECTED_API_KEY",
                        "valueFrom": {
                            "secretKeyRef": {"name": "endpoint", "key": "api-key"}
                        },
                    }
                ]
            )
        )
        checker = _make_checker(deploy_config=dc)
        secret = MagicMock()
        secret.data = {"different-key": "dmFsdWU="}
        mock_core = MagicMock(read_namespaced_secret=AsyncMock(return_value=secret))
        with _patch_core_v1(mock_core):
            result = await checker._check_secrets()
        assert result.status == CheckStatus.FAIL
        assert "endpoint/api-key" in result.message

    @pytest.mark.asyncio
    async def test_optional_secret_key_reference_does_not_fail_preflight(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                env=[
                    {
                        "name": "OPTIONAL_TOKEN",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "optional-endpoint",
                                "key": "token",
                                "optional": True,
                            }
                        },
                    }
                ]
            )
        )
        checker = _make_checker(deploy_config=dc)
        mock_core = MagicMock(
            read_namespaced_secret=AsyncMock(
                side_effect=ApiException(status=404, reason="NotFound")
            )
        )
        with _patch_core_v1(mock_core):
            result = await checker._check_secrets()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_missing_secret_fails(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(image_pull_secrets=[{"name": "regcred"}]),
        )
        checker = _make_checker(deploy_config=dc)
        mock_core = MagicMock(
            read_namespaced_secret=AsyncMock(
                side_effect=ApiException(status=404, reason="NotFound")
            )
        )
        with _patch_core_v1(mock_core):
            result = await checker._check_secrets()
        assert result.status == CheckStatus.FAIL
        assert "regcred" in result.message

    @pytest.mark.asyncio
    async def test_forbidden_secret_warns(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(image_pull_secrets=[{"name": "regcred"}]),
        )
        checker = _make_checker(deploy_config=dc)
        mock_core = MagicMock(
            read_namespaced_secret=AsyncMock(
                side_effect=ApiException(status=403, reason="Forbidden")
            )
        )
        with _patch_core_v1(mock_core):
            result = await checker._check_secrets()
        assert result.status == CheckStatus.WARN


# =============================================================================
# Tier 3: Image Reference
# =============================================================================


class TestCheckImageReference:
    @pytest.mark.asyncio
    async def test_no_image_fails(self) -> None:
        dc = DeploymentConfig()
        dc.image = ""
        checker = _make_checker(deploy_config=dc)
        result = await checker._check_image_reference()
        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_public_registry_with_tag_passes(self) -> None:
        dc = DeploymentConfig()
        dc.image = "nvcr.io/org/image:1.0"
        checker = _make_checker(deploy_config=dc)
        result = await checker._check_image_reference()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_implicit_latest_warns(self) -> None:
        dc = DeploymentConfig()
        dc.image = "nvcr.io/org/image"
        checker = _make_checker(deploy_config=dc)
        result = await checker._check_image_reference()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_digest_reference_does_not_warn_implicit_latest(self) -> None:
        """Digest refs are immutable; should not trigger implicit-'latest' WARN."""
        dc = DeploymentConfig()
        dc.image = "nvcr.io/org/image@sha256:abcdef"
        checker = _make_checker(deploy_config=dc)
        result = await checker._check_image_reference()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_private_registry_no_secret_warns(self) -> None:
        dc = DeploymentConfig()
        dc.image = "private.example.com/org/image:1.0"
        checker = _make_checker(deploy_config=dc)
        result = await checker._check_image_reference()
        assert result.status == CheckStatus.WARN


# =============================================================================
# Tier 3: DNS
# =============================================================================


class TestCheckDNS:
    @pytest.mark.asyncio
    async def test_coredns_running(self) -> None:
        checker = _make_checker()
        d = _deploy("coredns", "kube-system", 2)
        mock_apps = MagicMock(
            list_namespaced_deployment=AsyncMock(return_value=_list([d]))
        )
        with _patch_apps_v1(mock_apps):
            result = await checker._check_dns()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_coredns_not_ready(self) -> None:
        checker = _make_checker()
        d = _deploy("coredns", "kube-system", 0)
        mock_apps = MagicMock(
            list_namespaced_deployment=AsyncMock(return_value=_list([d]))
        )
        with _patch_apps_v1(mock_apps):
            result = await checker._check_dns()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_coredns_missing(self) -> None:
        checker = _make_checker()
        mock_apps = MagicMock(
            list_namespaced_deployment=AsyncMock(return_value=_list([]))
        )
        with _patch_apps_v1(mock_apps):
            result = await checker._check_dns()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_forbidden_skips(self) -> None:
        checker = _make_checker()
        mock_apps = MagicMock(
            list_namespaced_deployment=AsyncMock(
                side_effect=ApiException(status=403, reason="Forbidden")
            )
        )
        with _patch_apps_v1(mock_apps):
            result = await checker._check_dns()
        assert result.status == CheckStatus.SKIP


# =============================================================================
# Tier 3: Network Policies
# =============================================================================


class TestCheckNetworkPolicies:
    @pytest.mark.asyncio
    async def test_no_policies_passes(self) -> None:
        checker = _make_checker()
        mock_net = MagicMock(
            list_namespaced_network_policy=AsyncMock(return_value=_list([]))
        )
        with _patch_networking_v1(mock_net):
            result = await checker._check_network_policies()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_policy_present_warns(self) -> None:
        checker = _make_checker()
        pol = MagicMock()
        pol.metadata = MagicMock(name="deny-all")
        pol.metadata.name = "deny-all"
        mock_net = MagicMock(
            list_namespaced_network_policy=AsyncMock(return_value=_list([pol]))
        )
        with _patch_networking_v1(mock_net):
            result = await checker._check_network_policies()
        assert result.status == CheckStatus.WARN
        assert "deny-all" in result.message

    @pytest.mark.asyncio
    async def test_forbidden_skips(self) -> None:
        checker = _make_checker()
        mock_net = MagicMock(
            list_namespaced_network_policy=AsyncMock(
                side_effect=ApiException(status=403, reason="Forbidden")
            )
        )
        with _patch_networking_v1(mock_net):
            result = await checker._check_network_policies()
        assert result.status == CheckStatus.SKIP


# =============================================================================
# Tier 3: Kueue Queue
# =============================================================================


class TestCheckKueueQueue:
    @pytest.mark.asyncio
    async def test_no_queue_kueue_not_installed_skips(self) -> None:
        dc = DeploymentConfig()
        dc.scheduling = SchedulingConfig()
        checker = _make_checker(deploy_config=dc)

        # Kueue not installed → list_namespaced_custom_object raises / returns
        mock_custom = MagicMock(
            list_namespaced_custom_object=AsyncMock(
                side_effect=ApiException(status=404, reason="NotFound")
            )
        )
        with _patch_custom_objects(mock_custom):
            result = await checker._check_kueue_queue()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_queue_exists(self) -> None:
        dc = DeploymentConfig()
        dc.scheduling = SchedulingConfig(queue_name="my-queue")
        checker = _make_checker(deploy_config=dc)

        mock_custom = MagicMock(
            get_namespaced_custom_object=AsyncMock(return_value={"metadata": {}}),
            list_namespaced_custom_object=AsyncMock(return_value={"items": []}),
        )
        with _patch_custom_objects(mock_custom):
            result = await checker._check_kueue_queue()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_queue_missing_kueue_installed(self) -> None:
        dc = DeploymentConfig()
        dc.scheduling = SchedulingConfig(queue_name="my-queue")
        checker = _make_checker(deploy_config=dc)

        # Queue lookup 404, but CRD list succeeds → FAIL
        mock_custom = MagicMock(
            get_namespaced_custom_object=AsyncMock(
                side_effect=ApiException(status=404)
            ),
            list_namespaced_custom_object=AsyncMock(return_value={"items": []}),
        )
        with _patch_custom_objects(mock_custom):
            result = await checker._check_kueue_queue()
        assert result.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_explicit_queue_kueue_not_installed_fails(self) -> None:
        dc = DeploymentConfig()
        dc.scheduling = SchedulingConfig(queue_name="my-queue")
        checker = _make_checker(deploy_config=dc)

        mock_custom = MagicMock(
            get_namespaced_custom_object=AsyncMock(
                side_effect=ApiException(status=404)
            ),
            list_namespaced_custom_object=AsyncMock(
                side_effect=ApiException(status=404)
            ),
        )
        with _patch_custom_objects(mock_custom):
            result = await checker._check_kueue_queue()

        assert result.status == CheckStatus.FAIL
        assert "not installed" in result.message
        assert "my-queue" in result.message


# =============================================================================
# Tier 3: ConfigMap Size
# =============================================================================


class TestCheckConfigMapSize:
    @pytest.mark.asyncio
    async def test_size_ok(self) -> None:
        checker = _make_checker()
        result = await checker._check_configmap_size()
        assert result.status == CheckStatus.PASS


# =============================================================================
# Tier 3: Dry Run
# =============================================================================


class TestCheckDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_passes(self) -> None:
        checker = _make_checker()
        mock_custom = MagicMock(
            create_namespaced_custom_object=AsyncMock(return_value={})
        )
        with _patch_custom_objects(mock_custom):
            result = await checker._check_dry_run()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_dry_run_fails(self) -> None:
        checker = _make_checker()
        mock_custom = MagicMock(
            create_namespaced_custom_object=AsyncMock(
                side_effect=ApiException(status=400, reason="invalid")
            )
        )
        with _patch_custom_objects(mock_custom):
            result = await checker._check_dry_run()
        assert result.status == CheckStatus.FAIL


# =============================================================================
# Tier 3: Pod Security Admission
# =============================================================================


class TestCheckPodSecurityAdmission:
    @pytest.mark.asyncio
    async def test_no_psa_label_passes(self) -> None:
        checker = _make_checker()
        ns = MagicMock()
        ns.metadata = MagicMock()
        ns.metadata.labels = {}
        mock_core = MagicMock(read_namespace=AsyncMock(return_value=ns))
        with _patch_core_v1(mock_core):
            result = await checker._check_pod_security_admission()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_restricted_psa_warns(self) -> None:
        """PSA enforce=restricted is now WARN, not PASS — workload not yet
        verified against full restricted constraints. See Bug 4.
        """
        checker = _make_checker()
        ns = MagicMock()
        ns.metadata = MagicMock()
        ns.metadata.labels = {"pod-security.kubernetes.io/enforce": "restricted"}
        mock_core = MagicMock(read_namespace=AsyncMock(return_value=ns))
        with _patch_core_v1(mock_core):
            result = await checker._check_pod_security_admission()
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_ns_not_found_warns(self) -> None:
        checker = _make_checker()
        mock_core = MagicMock(
            read_namespace=AsyncMock(
                side_effect=ApiException(status=404, reason="NotFound")
            )
        )
        with _patch_core_v1(mock_core):
            result = await checker._check_pod_security_admission()
        assert result.status == CheckStatus.WARN


# =============================================================================
# Tier 3: Tolerations
# =============================================================================


class TestCheckTolerations:
    @pytest.mark.asyncio
    async def test_no_tolerations_skips(self) -> None:
        checker = _make_checker()
        result = await checker._check_tolerations()
        assert result.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_tainted_match_passes(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                tolerations=[
                    {
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    }
                ]
            )
        )
        checker = _make_checker(deploy_config=dc)
        n = _node(
            "gpu-node",
            "16",
            "64Gi",
            taints=[{"key": "nvidia.com/gpu", "value": "", "effect": "NoSchedule"}],
        )
        mock_core = MagicMock(list_node=AsyncMock(return_value=_list([n])))
        with _patch_core_v1(mock_core):
            result = await checker._check_tolerations()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_no_matching_taints_warns(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                tolerations=[
                    {
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    }
                ]
            )
        )
        checker = _make_checker(deploy_config=dc)
        n = _node("no-taint-node", "8", "32Gi")
        mock_core = MagicMock(list_node=AsyncMock(return_value=_list([n])))
        with _patch_core_v1(mock_core):
            result = await checker._check_tolerations()
        assert result.status == CheckStatus.WARN


# =============================================================================
# Orchestration
# =============================================================================


class TestRunAll:
    @pytest.mark.asyncio
    async def test_short_circuits_on_tier_1_fail(self) -> None:
        """When Kubernetes version fails, subsequent tiers should not run."""
        checker = _make_checker()
        mock_tier3 = AsyncMock()
        mock_tier3.assert_not_called()

        with _patch_version(major="0", minor="99"):
            results = await checker.run_all(timeout=5.0)

        assert not results.passed
        names = [c.name for c in results.checks]
        assert "Kubernetes Version" in names
        # No tier 3 checks ran
        assert "JobSet Controller" not in names

    @pytest.mark.asyncio
    async def test_short_circuits_on_rbac_fail(self) -> None:
        """When RBAC fails, tier 3+ checks should not run."""
        checker = _make_checker()

        mock_custom = MagicMock(
            list_cluster_custom_object=AsyncMock(return_value={"items": []})
        )
        mock_auth = MagicMock(
            create_self_subject_access_review=AsyncMock(
                return_value=_make_review(False)
            )
        )
        with (
            _patch_version(),
            _patch_custom_objects(mock_custom),
            _patch_auth_v1(mock_auth),
        ):
            results = await checker.run_all(timeout=5.0)

        assert not results.passed
        names = [c.name for c in results.checks]
        assert "RBAC Permissions" in names
        assert "JobSet Controller" not in names

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        """When a blocking check never returns, Preflight Timeout is reported."""
        checker = _make_checker()

        # Use an asyncio.Event that is never set so the check blocks forever.
        # The global autouse fake-sleep fixture makes asyncio.sleep instant,
        # so we can't rely on sleep for timeout — use a never-settled event.
        import asyncio

        never = asyncio.Event()

        async def _block(*args, **kwargs):
            await never.wait()
            return _version_obj()

        mock_v = MagicMock(get_code=AsyncMock(side_effect=_block))
        with patch(
            "aiperf.operator.preflight.client.VersionApi",
            return_value=mock_v,
        ):
            results = await checker.run_all(timeout=0.01)

        timeout_check = [c for c in results.checks if c.name == "Preflight Timeout"]
        assert len(timeout_check) == 1
        # WARN, not FAIL: _is_transient_error treats every per-check
        # TimeoutError as transient and retryable, so the aggregate deadline
        # firing must not permanently fail the job on a merely slow apiserver.
        assert timeout_check[0].status is CheckStatus.WARN
        assert results.passed


# =============================================================================
# _run_check exception handling (Bugs 1 + 2)
# =============================================================================


class TestRunCheckExceptionHandling:
    """Cover the operator-side _run_check fail-closed + transient-classification.

    Bug 1: ``_run_check`` previously caught only a narrow tuple
    (``ApiException`` + ``aiohttp.ClientError`` + ``TimeoutError`` +
    ``OSError``). A ``RuntimeError`` from a misbehaving check propagated and
    aborted the tier 3+ ``asyncio.gather``, swallowing every other check's
    result.

    Bug 2: Transient classification used ``"connect" in str(e).lower()``,
    so an admission-webhook FAIL message containing the word "connect"
    silently downgraded to a WARN. Classification is now exception-type-based.
    """

    @pytest.mark.asyncio
    async def test_runtime_error_caught_and_reported_as_fail(self) -> None:
        """A RuntimeError raised inside a check must NOT abort preflight."""
        checker = _make_checker()

        async def _broken_check():
            raise RuntimeError("unexpected null pointer in foo")

        result = await checker._run_check(_broken_check)
        assert result.status == CheckStatus.FAIL
        assert "unexpected null pointer" in result.message

    @pytest.mark.asyncio
    async def test_connect_string_in_runtime_error_does_not_warn(self) -> None:
        """The substring 'connect' must not downgrade a RuntimeError to WARN.

        A real cluster admission webhook may say 'cannot connect to validation
        service'; previously this was misread as a transient network blip.
        """
        checker = _make_checker()

        async def _bad_check():
            raise RuntimeError("admission webhook cannot connect to validator")

        result = await checker._run_check(_bad_check)
        assert result.status == CheckStatus.FAIL
        assert "connect" in result.message  # message preserved

    @pytest.mark.asyncio
    async def test_asyncio_timeout_classified_as_transient_warn(self) -> None:
        """TimeoutError -> WARN (transient)."""

        checker = _make_checker()

        async def _timed_out():
            raise TimeoutError("api timed out")

        result = await checker._run_check(_timed_out)
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_5xx_api_exception_classified_as_transient_warn(self) -> None:
        """ApiException with HTTP 5xx -> WARN (transient apiserver error)."""
        checker = _make_checker()

        async def _five_hundred():
            raise ApiException(status=503, reason="ServiceUnavailable")

        result = await checker._run_check(_five_hundred)
        assert result.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_4xx_api_exception_classified_as_permanent_fail(self) -> None:
        """ApiException with HTTP 4xx -> FAIL (permanent — bad request, forbidden)."""
        checker = _make_checker()

        async def _forbidden():
            raise ApiException(status=403, reason="Forbidden")

        result = await checker._run_check(_forbidden)
        assert result.status == CheckStatus.FAIL


# =============================================================================
# Node resources taint awareness (Bug 3)
# =============================================================================


class TestCheckNodeResourcesTaintAwareness:
    """A 50-node cluster of NoSchedule-tainted GPU nodes must NOT be reported
    as 'sufficient resources' for a CPU-only workload that has no matching
    toleration.
    """

    @pytest.mark.asyncio
    async def test_untolerated_tainted_nodes_excluded(self) -> None:
        checker = _make_checker()
        big_tainted = _node(
            "gpu-node",
            "64",
            "256Gi",
            taints=[
                {"key": "nvidia.com/gpu", "value": "", "effect": "NoSchedule"},
            ],
        )
        mock_core = MagicMock(list_node=AsyncMock(return_value=_list([big_tainted])))
        with _patch_core_v1(mock_core):
            result = await checker._check_node_resources()
        # No usable nodes -> WARN, not PASS.
        assert result.status == CheckStatus.WARN
        assert (
            "tainted" in result.message.lower()
            or "schedulable" in result.message.lower()
        )

    @pytest.mark.asyncio
    async def test_tolerated_tainted_nodes_included(self) -> None:
        dc = DeploymentConfig(
            pod_template=PodTemplateConfig(
                tolerations=[
                    {
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    }
                ]
            )
        )
        checker = _make_checker(deploy_config=dc)
        big_tainted = _node(
            "gpu-node",
            "64",
            "256Gi",
            taints=[
                {"key": "nvidia.com/gpu", "value": "", "effect": "NoSchedule"},
            ],
        )
        mock_core = MagicMock(list_node=AsyncMock(return_value=_list([big_tainted])))
        with _patch_core_v1(mock_core):
            result = await checker._check_node_resources()
        assert result.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_prefer_no_schedule_taint_does_not_exclude(self) -> None:
        """PreferNoSchedule is a soft preference — node is still schedulable."""
        checker = _make_checker()
        node = _node(
            "soft-tainted",
            "64",
            "256Gi",
            taints=[
                {
                    "key": "some.taint",
                    "value": "",
                    "effect": "PreferNoSchedule",
                }
            ],
        )
        mock_core = MagicMock(list_node=AsyncMock(return_value=_list([node])))
        with _patch_core_v1(mock_core):
            result = await checker._check_node_resources()
        assert result.status == CheckStatus.PASS


# =============================================================================
# PSA restricted should WARN, not PASS (Bug 4)
# =============================================================================


class TestCheckPodSecurityAdmissionRestricted:
    """The previous code rubber-stamped every standard PSA level. 'restricted'
    enforces constraints (runAsNonRoot, allowPrivilegeEscalation=false, ...)
    that the AIPerf pod template has not been audited against — surface as
    WARN until that audit lands.
    """

    @pytest.mark.asyncio
    async def test_restricted_warns(self) -> None:
        checker = _make_checker()
        ns = MagicMock()
        ns.metadata = MagicMock()
        ns.metadata.labels = {"pod-security.kubernetes.io/enforce": "restricted"}
        mock_core = MagicMock(read_namespace=AsyncMock(return_value=ns))
        with _patch_core_v1(mock_core):
            result = await checker._check_pod_security_admission()
        assert result.status == CheckStatus.WARN
        assert "restricted" in result.message.lower()

    @pytest.mark.asyncio
    async def test_baseline_still_passes(self) -> None:
        checker = _make_checker()
        ns = MagicMock()
        ns.metadata = MagicMock()
        ns.metadata.labels = {"pod-security.kubernetes.io/enforce": "baseline"}
        mock_core = MagicMock(read_namespace=AsyncMock(return_value=ns))
        with _patch_core_v1(mock_core):
            result = await checker._check_pod_security_admission()
        assert result.status == CheckStatus.PASS


class TestDryRunTransientClassification:
    """A 5xx on the dry-run POST is retryable, not a manifest rejection.

    _check_dry_run caught every ApiException and returned FAIL with a message
    blaming OPA/Gatekeeper, which preempted the dispatcher's own transient
    classification. A 503 or 429 from a busy apiserver permanently failed the
    job.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,expected",
        [
            param(503, CheckStatus.WARN, id="service-unavailable-is-transient"),
            param(429, CheckStatus.WARN, id="too-many-requests-is-transient"),
            param(422, CheckStatus.FAIL, id="unprocessable-is-permanent"),
            param(403, CheckStatus.FAIL, id="forbidden-is-permanent"),
        ],
    )  # fmt: skip
    async def test_dry_run_status_classification(
        self, status: int, expected: CheckStatus
    ) -> None:
        checker = _make_checker()
        exc = ApiException(status=status, reason="boom")
        mock_api = MagicMock(create_namespaced_custom_object=AsyncMock(side_effect=exc))
        with patch(
            "aiperf.operator.preflight.client.CustomObjectsApi", return_value=mock_api
        ):
            result = await checker._run_check(checker._check_dry_run)
        assert result.status is expected
