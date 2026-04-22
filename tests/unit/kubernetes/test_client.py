# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.client (free functions + facade).

Focuses on:
- k8s_client context manager (incluster/kubeconfig/close).
- AIPerfJob CR helpers (list/find/get_status/cancel).
- JobSet helpers (list/find/delete).
- Namespace deletion tolerance.
- Pod summary + controller/operator/retrievable lookups.
- wait_for_controller_pod_ready polling + timeout.
- cluster_version.
- Facade delegations.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from kubernetes_asyncio.client.models import (
    V1ContainerStatus,
    V1ObjectMeta,
    V1Pod,
    V1PodList,
    V1PodStatus,
)
from pytest import param

from aiperf.kubernetes.client import (
    AIPerfKubeClient,
    cancel_aiperf_job,
    cluster_version,
    controller_selector,
    delete_jobset,
    delete_namespace,
    find_aiperf_job,
    find_controller_pod,
    find_jobset,
    find_operator_pod,
    find_retrievable_pod,
    get_pod_summary,
    get_raw_aiperfjob_status,
    job_selector,
    k8s_client,
    list_aiperf_jobs,
    list_jobsets,
    wait_for_controller_pod_ready,
)
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.models import PodSummary

# ============================================================
# Helpers
# ============================================================


def _make_v1pod(
    name: str = "pod-0",
    namespace: str = "default",
    phase: str = "Running",
    container_statuses: list[dict[str, Any]] | None = None,
) -> V1Pod:
    """Build a V1Pod for list_namespaced_pod mocking."""
    if container_statuses is None:
        container_statuses = [{"name": "c", "ready": True, "restart_count": 0}]
    css = [
        V1ContainerStatus(
            name=cs["name"],
            ready=cs["ready"],
            restart_count=cs.get("restart_count", 0),
            image="x",
            image_id="y",
            state={},
        )
        for cs in container_statuses
    ]
    return V1Pod(
        metadata=V1ObjectMeta(name=name, namespace=namespace),
        status=V1PodStatus(phase=phase, container_statuses=css or None),
    )


def _pod_list(pods: list[V1Pod]) -> V1PodList:
    """Wrap pods in a V1PodList."""
    return V1PodList(items=pods)


def _raw_aiperfjob(
    name: str = "test-job",
    namespace: str = "default",
    phase: str = "Running",
    job_id: str = "job-abc",
    created: str = "2026-01-15T10:30:00Z",
) -> dict[str, Any]:
    """Raw AIPerfJob dict as returned by the API."""
    return {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": created,
        },
        "spec": {
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"url": "http://localhost:8000"},
            },
        },
        "status": {"phase": phase, "jobId": job_id},
    }


def _raw_jobset(
    name: str = "js-1",
    namespace: str = "default",
    created: str = "2026-01-15T10:30:00Z",
    labels: dict[str, str] | None = None,
    status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Raw JobSet dict as returned by the API."""
    meta: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "creationTimestamp": created,
    }
    if labels is not None:
        meta["labels"] = labels
    else:
        meta["labels"] = {"app": "aiperf"}
    return {"metadata": meta, "status": status or {}}


def _api_exception(status: int, message: str = "err") -> ApiException:
    """Construct an ApiException with a given status code."""
    exc = ApiException(status=status, reason=message)
    return exc


# ============================================================
# k8s_client context manager
# ============================================================


class TestK8sClient:
    """Verify the k8s_client async context manager."""

    @pytest.mark.asyncio
    async def test_k8s_client_uses_incluster_first(self) -> None:
        """load_incluster_config is tried first; load_kube_config is NOT called."""
        fake_api = MagicMock()
        fake_api.close = AsyncMock()
        with (
            patch("aiperf.kubernetes.client.suppress_noisy_http_loggers"),
            patch("aiperf.kubernetes.client.config.load_incluster_config"),
            patch(
                "aiperf.kubernetes.client.config.load_kube_config",
                new_callable=AsyncMock,
            ) as mock_kube,
            patch("aiperf.kubernetes.client.ApiClient", return_value=fake_api),
        ):
            async with k8s_client() as api:
                assert api is fake_api
            mock_kube.assert_not_called()

    @pytest.mark.asyncio
    async def test_k8s_client_falls_back_to_kubeconfig(self) -> None:
        """When load_incluster_config raises ConfigException, load_kube_config is called."""
        from kubernetes_asyncio import config as k8s_config

        fake_api = MagicMock()
        fake_api.close = AsyncMock()
        with (
            patch("aiperf.kubernetes.client.suppress_noisy_http_loggers"),
            patch(
                "aiperf.kubernetes.client.config.load_incluster_config",
                side_effect=k8s_config.ConfigException("no incluster"),
            ),
            patch(
                "aiperf.kubernetes.client.config.load_kube_config",
                new_callable=AsyncMock,
            ) as mock_kube,
            patch("aiperf.kubernetes.client.ApiClient", return_value=fake_api),
        ):
            async with k8s_client(kubeconfig="/cfg", context="ctx"):
                pass
        mock_kube.assert_awaited_once_with(config_file="/cfg", context="ctx")

    @pytest.mark.asyncio
    async def test_k8s_client_closes_api_on_exit(self) -> None:
        """ApiClient.close() is awaited on context exit."""
        fake_api = MagicMock()
        fake_api.close = AsyncMock()
        with (
            patch("aiperf.kubernetes.client.suppress_noisy_http_loggers"),
            patch("aiperf.kubernetes.client.config.load_incluster_config"),
            patch("aiperf.kubernetes.client.ApiClient", return_value=fake_api),
        ):
            async with k8s_client():
                pass
        fake_api.close.assert_awaited_once()


# ============================================================
# Label selectors
# ============================================================


class TestSelectors:
    """Verify pure string selector helpers."""

    def test_job_selector(self) -> None:
        assert job_selector("j-1") == "app=aiperf,aiperf.nvidia.com/job-id=j-1"

    def test_controller_selector(self) -> None:
        result = controller_selector("j-1")
        assert "app=aiperf" in result
        assert "aiperf.nvidia.com/job-id=j-1" in result
        assert "jobset.sigs.k8s.io/replicatedjob-name=controller" in result


# ============================================================
# list_aiperf_jobs
# ============================================================


class TestListAIPerfJobs:
    """Verify AIPerfJob CR listing, filtering, sorting."""

    @pytest.mark.asyncio
    async def test_list_aiperf_jobs_returns_sorted_infos(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={
                "items": [
                    _raw_aiperfjob(name="older", created="2026-01-01T00:00:00Z"),
                    _raw_aiperfjob(name="newer", created="2026-01-15T00:00:00Z"),
                ],
            }
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await list_aiperf_jobs(api, namespace="default")
        assert [r.name for r in result] == ["newer", "older"]

    @pytest.mark.asyncio
    async def test_list_aiperf_jobs_404_returns_empty(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await list_aiperf_jobs(api, namespace="default")
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_filter,expected_names",
        [
            param("Running", ["run"], id="filter_running"),
            param("Failed", ["fail"], id="filter_failed"),
            param("Completed", ["done"], id="filter_completed"),
        ],
    )  # fmt: skip
    async def test_list_aiperf_jobs_filters_by_phase(
        self,
        status_filter: str,
        expected_names: list[str],
    ) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={
                "items": [
                    _raw_aiperfjob(name="run", phase="Running"),
                    _raw_aiperfjob(name="fail", phase="Failed"),
                    _raw_aiperfjob(name="done", phase="Completed"),
                ],
            }
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await list_aiperf_jobs(
                api, namespace="default", status_filter=status_filter
            )
        assert [r.name for r in result] == expected_names

    @pytest.mark.asyncio
    async def test_list_aiperf_jobs_all_namespaces_uses_list_cluster(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object = AsyncMock(
            return_value={"items": [_raw_aiperfjob(namespace="ns-x")]}
        )
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": []}
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await list_aiperf_jobs(api, all_namespaces=True)
        mock_custom.list_cluster_custom_object.assert_awaited_once()
        mock_custom.list_namespaced_custom_object.assert_not_called()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_aiperf_jobs_non_404_raises(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(500)
        )
        with (
            patch(
                "aiperf.kubernetes.client.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            pytest.raises(ApiException),
        ):
            await list_aiperf_jobs(api, namespace="default")


# ============================================================
# find_aiperf_job
# ============================================================


class TestFindAIPerfJob:
    """Verify AIPerfJob resolution by name and jobId fallback."""

    @pytest.mark.asyncio
    async def test_find_aiperf_job_by_name_namespaced(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            return_value=_raw_aiperfjob(name="n", job_id="j")
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_job(api, "n", namespace="default")
        assert result is not None
        assert result.name == "n"
        assert result.job_id == "j"

    @pytest.mark.asyncio
    async def test_find_aiperf_job_falls_back_to_job_id(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        # Direct lookup 404
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
        # Fallback finds by status.jobId
        mock_custom.list_cluster_custom_object = AsyncMock(
            return_value={
                "items": [
                    _raw_aiperfjob(name="cr-x", job_id="other"),
                    _raw_aiperfjob(name="cr-target", job_id="target-id"),
                ],
            }
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_job(api, "target-id", namespace="default")
        assert result is not None
        assert result.name == "cr-target"
        assert result.job_id == "target-id"

    @pytest.mark.asyncio
    async def test_find_aiperf_job_not_found_returns_none(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
        mock_custom.list_cluster_custom_object = AsyncMock(return_value={"items": []})
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_job(api, "nope", namespace="default")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_aiperf_job_direct_lookup_non_404_raises(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(500)
        )
        with (
            patch(
                "aiperf.kubernetes.client.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            pytest.raises(ApiException),
        ):
            await find_aiperf_job(api, "x", namespace="default")


# ============================================================
# get_raw_aiperfjob_status
# ============================================================


class TestGetRawStatus:
    """Verify raw status dict extraction."""

    @pytest.mark.asyncio
    async def test_returns_status_dict(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            return_value={
                "status": {"phase": "Completed", "extra": 123},
            }
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await get_raw_aiperfjob_status(api, "n", "default")
        assert result == {"phase": "Completed", "extra": 123}

    @pytest.mark.asyncio
    async def test_missing_returns_empty_dict(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await get_raw_aiperfjob_status(api, "n", "default")
        assert result == {}

    @pytest.mark.asyncio
    async def test_null_status_returns_empty_dict(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            return_value={"status": None}
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await get_raw_aiperfjob_status(api, "n", "default")
        assert result == {}


# ============================================================
# cancel_aiperf_job
# ============================================================


class TestCancelAIPerfJob:
    """Verify cancel issues spec.cancel=true merge patch."""

    @pytest.mark.asyncio
    async def test_applies_spec_cancel_true(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.patch_namespaced_custom_object = AsyncMock(return_value={})
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            await cancel_aiperf_job(api, "n", "default")
        mock_custom.patch_namespaced_custom_object.assert_awaited_once()
        call_kwargs = mock_custom.patch_namespaced_custom_object.call_args.kwargs
        assert call_kwargs["body"] == {"spec": {"cancel": True}}
        assert call_kwargs["name"] == "n"
        assert call_kwargs["namespace"] == "default"


# ============================================================
# list_jobsets
# ============================================================


class TestListJobsets:
    """Verify JobSet listing."""

    @pytest.mark.asyncio
    async def test_by_namespace(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": [_raw_jobset()]}
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await list_jobsets(api, namespace="default")
        assert len(result) == 1
        mock_custom.list_namespaced_custom_object.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_all_namespaces_uses_list_cluster(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object = AsyncMock(
            return_value={"items": [_raw_jobset()]}
        )
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": []}
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            await list_jobsets(api, all_namespaces=True)
        mock_custom.list_cluster_custom_object.assert_awaited_once()
        mock_custom.list_namespaced_custom_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_filters_by_status(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={
                "items": [
                    # Running (no conditions)
                    _raw_jobset(name="a"),
                    # Completed
                    _raw_jobset(
                        name="b",
                        status={
                            "conditions": [{"type": "Completed", "status": "True"}]
                        },
                    ),
                ],
            }
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await list_jobsets(
                api, namespace="default", status_filter="Running"
            )
        assert len(result) == 1
        assert result[0].name == "a"

    @pytest.mark.asyncio
    async def test_404_returns_empty(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await list_jobsets(api, namespace="default")
        assert result == []

    @pytest.mark.asyncio
    async def test_with_job_id_in_label_selector(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": []}
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            await list_jobsets(api, namespace="default", job_id="abc")
        call_kwargs = mock_custom.list_namespaced_custom_object.call_args.kwargs
        assert "aiperf.nvidia.com/job-id=abc" in call_kwargs["label_selector"]


# ============================================================
# find_jobset
# ============================================================


class TestFindJobset:
    """Verify JobSet resolution by label and name fallback."""

    @pytest.mark.asyncio
    async def test_by_label(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": [_raw_jobset(name="found")]}
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_jobset(api, "abc", namespace="default")
        assert result is not None
        assert result.name == "found"

    @pytest.mark.asyncio
    async def test_by_name_fallback(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        # First call (by label): empty. Second call (by name): hit.
        mock_custom.list_namespaced_custom_object = AsyncMock(
            side_effect=[
                {"items": []},
                {"items": [_raw_jobset(name="by-name")]},
            ]
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_jobset(api, "abc", namespace="default")
        assert result is not None
        assert result.name == "by-name"
        # Verify second call used field_selector
        second_call_kwargs = mock_custom.list_namespaced_custom_object.call_args_list[
            1
        ].kwargs
        assert "field_selector" in second_call_kwargs

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": []}
        )
        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_jobset(api, "nope", namespace="default")
        assert result is None


# ============================================================
# delete_jobset
# ============================================================


class TestDeleteJobset:
    """Verify JobSet + aux resource deletion."""

    @pytest.mark.asyncio
    async def test_deletes_jobset_and_aux_resources(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.delete_namespaced_custom_object = AsyncMock(return_value={})
        mock_core = MagicMock()
        mock_core.delete_namespaced_config_map = AsyncMock(return_value={})
        mock_rbac = MagicMock()
        mock_rbac.delete_namespaced_role = AsyncMock(return_value={})
        mock_rbac.delete_namespaced_role_binding = AsyncMock(return_value={})

        with (
            patch(
                "aiperf.kubernetes.client.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            patch(
                "aiperf.kubernetes.client.client.CoreV1Api",
                return_value=mock_core,
            ),
            patch(
                "aiperf.kubernetes.client.client.RbacAuthorizationV1Api",
                return_value=mock_rbac,
            ),
        ):
            await delete_jobset(api, "my-js", "default")

        mock_custom.delete_namespaced_custom_object.assert_awaited_once()
        mock_core.delete_namespaced_config_map.assert_awaited_once()
        mock_rbac.delete_namespaced_role.assert_awaited_once()
        mock_rbac.delete_namespaced_role_binding.assert_awaited_once()

        # Check resource names
        cm_kwargs = mock_core.delete_namespaced_config_map.call_args.kwargs
        assert cm_kwargs["name"] == "my-js-config"
        role_kwargs = mock_rbac.delete_namespaced_role.call_args.kwargs
        assert role_kwargs["name"] == "my-js-role"
        rb_kwargs = mock_rbac.delete_namespaced_role_binding.call_args.kwargs
        assert rb_kwargs["name"] == "my-js-binding"

    @pytest.mark.asyncio
    async def test_tolerates_404_on_jobset(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.delete_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
        mock_core = MagicMock()
        mock_core.delete_namespaced_config_map = AsyncMock(return_value={})
        mock_rbac = MagicMock()
        mock_rbac.delete_namespaced_role = AsyncMock(return_value={})
        mock_rbac.delete_namespaced_role_binding = AsyncMock(return_value={})

        with (
            patch(
                "aiperf.kubernetes.client.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            patch(
                "aiperf.kubernetes.client.client.CoreV1Api",
                return_value=mock_core,
            ),
            patch(
                "aiperf.kubernetes.client.client.RbacAuthorizationV1Api",
                return_value=mock_rbac,
            ),
        ):
            # Should NOT raise
            await delete_jobset(api, "my-js", "default")

        # Aux deletions should still have been attempted
        mock_core.delete_namespaced_config_map.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "aux_status",
        [
            param(404, id="already_gone"),
            param(409, id="namespace_terminating"),
        ],
    )  # fmt: skip
    async def test_tolerates_404_409_on_aux_resources(self, aux_status: int) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.delete_namespaced_custom_object = AsyncMock(return_value={})
        mock_core = MagicMock()
        mock_core.delete_namespaced_config_map = AsyncMock(
            side_effect=_api_exception(aux_status)
        )
        mock_rbac = MagicMock()
        mock_rbac.delete_namespaced_role = AsyncMock(
            side_effect=_api_exception(aux_status)
        )
        mock_rbac.delete_namespaced_role_binding = AsyncMock(
            side_effect=_api_exception(aux_status)
        )

        with (
            patch(
                "aiperf.kubernetes.client.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            patch(
                "aiperf.kubernetes.client.client.CoreV1Api",
                return_value=mock_core,
            ),
            patch(
                "aiperf.kubernetes.client.client.RbacAuthorizationV1Api",
                return_value=mock_rbac,
            ),
        ):
            await delete_jobset(api, "my-js", "default")

    @pytest.mark.asyncio
    async def test_non_404_on_jobset_raises(self) -> None:
        api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.delete_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(500)
        )
        mock_core = MagicMock()
        mock_rbac = MagicMock()

        with (
            patch(
                "aiperf.kubernetes.client.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            patch(
                "aiperf.kubernetes.client.client.CoreV1Api",
                return_value=mock_core,
            ),
            patch(
                "aiperf.kubernetes.client.client.RbacAuthorizationV1Api",
                return_value=mock_rbac,
            ),
            pytest.raises(ApiException),
        ):
            await delete_jobset(api, "my-js", "default")


# ============================================================
# delete_namespace
# ============================================================


class TestDeleteNamespace:
    """Verify namespace deletion tolerance."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.delete_namespace = AsyncMock(return_value={})
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            await delete_namespace(api, "ns")
        mock_core.delete_namespace.assert_awaited_once_with(name="ns")

    @pytest.mark.asyncio
    async def test_tolerates_404(self) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.delete_namespace = AsyncMock(side_effect=_api_exception(404))
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            # Should not raise
            await delete_namespace(api, "ns")

    @pytest.mark.asyncio
    async def test_logs_warning_on_other_errors(self) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.delete_namespace = AsyncMock(side_effect=_api_exception(500))
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            # Does not raise — just logs warning
            await delete_namespace(api, "ns")


# ============================================================
# get_pod_summary
# ============================================================


class TestGetPodSummary:
    """Verify pod readiness summary."""

    @pytest.mark.asyncio
    async def test_counts_ready_and_restarts(self) -> None:
        api = MagicMock()
        pods = [
            _make_v1pod(
                name="p1",
                phase="Running",
                container_statuses=[
                    {"name": "c1", "ready": True, "restart_count": 1},
                    {"name": "c2", "ready": True, "restart_count": 2},
                ],
            ),
            _make_v1pod(
                name="p2",
                phase="Pending",
                container_statuses=[
                    {"name": "c1", "ready": False, "restart_count": 0},
                ],
            ),
        ]
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list(pods))
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await get_pod_summary(api, "js", "default")
        assert result == PodSummary(ready=1, total=2, restarts=3)

    @pytest.mark.asyncio
    async def test_returns_zero_summary_on_api_exception(self) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(side_effect=_api_exception(500))
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await get_pod_summary(api, "js", "default")
        assert result == PodSummary(ready=0, total=0, restarts=0)

    @pytest.mark.asyncio
    async def test_no_pods(self) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([]))
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await get_pod_summary(api, "js", "default")
        assert result == PodSummary(ready=0, total=0, restarts=0)


# ============================================================
# find_operator_pod
# ============================================================


class TestFindOperatorPod:
    """Verify operator pod discovery."""

    @pytest.mark.asyncio
    async def test_returns_name_and_phase(self) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            return_value=_pod_list([_make_v1pod(name="op-0", phase="Running")])
        )
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_operator_pod(api)
        assert result == ("op-0", PodPhase.RUNNING)

    @pytest.mark.asyncio
    async def test_returns_none_when_empty(self) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([]))
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_operator_pod(api)
        assert result is None


# ============================================================
# find_controller_pod
# ============================================================


class TestFindControllerPod:
    """Verify controller pod discovery."""

    @pytest.mark.asyncio
    async def test_returns_name_and_phase(self) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            return_value=_pod_list([_make_v1pod(name="ctrl-0", phase="Running")])
        )
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_controller_pod(api, "default", "j-1")
        assert result == ("ctrl-0", PodPhase.RUNNING)

    @pytest.mark.asyncio
    async def test_returns_none_when_empty(self) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([]))
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_controller_pod(api, "default", "j-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_phase_defaults_to_unknown(self) -> None:
        api = MagicMock()
        # V1PodStatus with no phase -> None
        pod = V1Pod(
            metadata=V1ObjectMeta(name="ctrl-0", namespace="default"),
            status=V1PodStatus(),
        )
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([pod]))
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_controller_pod(api, "default", "j-1")
        assert result is not None
        assert result[1] == PodPhase.UNKNOWN


# ============================================================
# find_retrievable_pod
# ============================================================


class TestFindRetrievablePod:
    """Verify phase-based filtering."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phase,require_running,expected_none",
        [
            param("Running", False, False, id="running_retrievable"),
            param("Succeeded", False, False, id="succeeded_retrievable"),
            param("Failed", False, True, id="failed_not_retrievable"),
            param("Pending", False, True, id="pending_not_retrievable"),
            param("Running", True, False, id="running_with_require_running"),
            param("Succeeded", True, True, id="succeeded_excluded_by_require_running"),
        ],
    )  # fmt: skip
    async def test_requires_running_phase(
        self, phase: str, require_running: bool, expected_none: bool
    ) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            return_value=_pod_list([_make_v1pod(name="ctrl-0", phase=phase)])
        )
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_retrievable_pod(
                api, "default", "j-1", require_running=require_running
            )
        if expected_none:
            assert result is None
        else:
            assert result is not None

    @pytest.mark.asyncio
    async def test_no_pod_returns_none(self) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_pod_list([]))
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await find_retrievable_pod(api, "default", "j-1")
        assert result is None


# ============================================================
# wait_for_controller_pod_ready
# ============================================================


class TestWaitForControllerPodReady:
    """Verify polling loop and timeout behavior."""

    @pytest.mark.asyncio
    async def test_immediately_ready(self) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            return_value=_pod_list([_make_v1pod(name="ctrl-0", phase="Running")])
        )
        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await wait_for_controller_pod_ready(
                api, "default", "j-1", timeout=10
            )
        assert result == "ctrl-0"

    @pytest.mark.asyncio
    async def test_succeeds_after_polling(self, monkeypatch) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            side_effect=[
                _pod_list([_make_v1pod(name="ctrl-0", phase="Pending")]),
                _pod_list([_make_v1pod(name="ctrl-0", phase="Running")]),
            ]
        )

        # Fast-forward sleeps
        async def _fast_sleep(_: float) -> None:
            return None

        monkeypatch.setattr("aiperf.kubernetes.client.asyncio.sleep", _fast_sleep)

        with patch(
            "aiperf.kubernetes.client.client.CoreV1Api",
            return_value=mock_core,
        ):
            result = await wait_for_controller_pod_ready(
                api, "default", "j-1", timeout=300
            )
        assert result == "ctrl-0"

    @pytest.mark.asyncio
    async def test_times_out(self, monkeypatch) -> None:
        api = MagicMock()
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(
            return_value=_pod_list([_make_v1pod(name="ctrl-0", phase="Pending")])
        )

        # Fast-forward: advance the loop time by the sleep delay each call
        base_time = [0.0]

        class FakeLoop:
            def time(self) -> float:
                return base_time[0]

        fake_loop = FakeLoop()

        def fake_get_loop():
            return fake_loop

        async def _advance_sleep(delay: float) -> None:
            base_time[0] += delay

        monkeypatch.setattr(
            "aiperf.kubernetes.client.asyncio.get_running_loop", fake_get_loop
        )
        monkeypatch.setattr("aiperf.kubernetes.client.asyncio.sleep", _advance_sleep)

        with (
            patch(
                "aiperf.kubernetes.client.client.CoreV1Api",
                return_value=mock_core,
            ),
            pytest.raises(TimeoutError, match="Controller pod not ready"),
        ):
            await wait_for_controller_pod_ready(api, "default", "j-1", timeout=5)


# ============================================================
# cluster_version
# ============================================================


class TestClusterVersion:
    """Verify cluster_version returns a plain dict."""

    @pytest.mark.asyncio
    async def test_returns_dict(self) -> None:
        api = MagicMock()
        vinfo = MagicMock()
        vinfo.major = "1"
        vinfo.minor = "28"
        vinfo.git_version = "v1.28.0"
        vinfo.git_commit = "abc"
        vinfo.platform = "linux/amd64"
        mock_version_api = MagicMock()
        mock_version_api.get_code = AsyncMock(return_value=vinfo)
        with patch(
            "aiperf.kubernetes.client.client.VersionApi",
            return_value=mock_version_api,
        ):
            result = await cluster_version(api)
        assert result == {
            "major": "1",
            "minor": "28",
            "gitVersion": "v1.28.0",
            "gitCommit": "abc",
            "platform": "linux/amd64",
        }


# ============================================================
# AIPerfKubeClient facade
# ============================================================


class TestFacade:
    """Verify the AIPerfKubeClient facade delegates to free functions."""

    @pytest.mark.asyncio
    async def test_list_jobs_delegates(self) -> None:
        api = MagicMock()
        facade = AIPerfKubeClient(api)
        with patch(
            "aiperf.kubernetes.client.list_aiperf_jobs",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_fn:
            result = await facade.list_jobs(namespace="default")
        mock_fn.assert_awaited_once_with(api, namespace="default")
        assert result == []

    @pytest.mark.asyncio
    async def test_cancel_job_delegates(self) -> None:
        api = MagicMock()
        facade = AIPerfKubeClient(api)
        with patch(
            "aiperf.kubernetes.client.cancel_aiperf_job",
            new_callable=AsyncMock,
        ) as mock_fn:
            await facade.cancel_job("n", "default")
        mock_fn.assert_awaited_once_with(api, "n", "default")

    @pytest.mark.asyncio
    async def test_close_closes_api_client(self) -> None:
        api = MagicMock()
        api.close = AsyncMock()
        facade = AIPerfKubeClient(api)
        await facade.close()
        api.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_loads_incluster_first(self) -> None:
        fake_api = MagicMock()
        with (
            patch("aiperf.kubernetes.client.suppress_noisy_http_loggers"),
            patch(
                "aiperf.kubernetes.client.config.load_incluster_config"
            ) as mock_incluster,
            patch(
                "aiperf.kubernetes.client.config.load_kube_config",
                new_callable=AsyncMock,
            ) as mock_kube,
            patch("aiperf.kubernetes.client.ApiClient", return_value=fake_api),
        ):
            result = await AIPerfKubeClient.create()
        mock_incluster.assert_called_once()
        mock_kube.assert_not_called()
        assert isinstance(result, AIPerfKubeClient)
        assert result.api is fake_api

    @pytest.mark.asyncio
    async def test_create_falls_back_to_kubeconfig(self) -> None:
        from kubernetes_asyncio import config as k8s_config

        fake_api = MagicMock()
        with (
            patch("aiperf.kubernetes.client.suppress_noisy_http_loggers"),
            patch(
                "aiperf.kubernetes.client.config.load_incluster_config",
                side_effect=k8s_config.ConfigException("no incluster"),
            ),
            patch(
                "aiperf.kubernetes.client.config.load_kube_config",
                new_callable=AsyncMock,
            ) as mock_kube,
            patch("aiperf.kubernetes.client.ApiClient", return_value=fake_api),
        ):
            await AIPerfKubeClient.create(kubeconfig="/cfg", kube_context="ctx")
        mock_kube.assert_awaited_once_with(config_file="/cfg", context="ctx")

    def test_api_property(self) -> None:
        api = MagicMock()
        facade = AIPerfKubeClient(api)
        assert facade.api is api

    def test_selector_static_methods(self) -> None:
        """The facade preserves kr8s-era access to selector functions."""
        assert (
            AIPerfKubeClient.job_selector("x")
            == "app=aiperf,aiperf.nvidia.com/job-id=x"
        )
