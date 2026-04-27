# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.client_jobs — edge cases not covered by test_client.py.

The test_client facade tests already exercise the happy paths via patches on the
facade module. This file focuses on:

- namespace=None default-resolution behaviour
- find_aiperf_job fallback-list error paths (404 suppressed, non-404 re-raises)
- find_aiperf_job name-match branch in fallback
- cluster-wide find (no namespace -> no direct get, list cluster with field_selector)
- get_raw_aiperfjob_status when "status" key is absent
- cancel_aiperf_job surfaces all ApiException statuses (nothing suppressed)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.client_jobs import (
    cancel_aiperf_job,
    find_aiperf_job,
    find_aiperf_sweep,
    get_raw_aiperfjob_status,
    list_aiperf_jobs,
)


def _raw_aiperfjob(
    name: str = "test-job",
    namespace: str = "default",
    phase: str = "Running",
    job_id: str = "job-abc",
    created: str = "2026-01-15T10:30:00Z",
) -> dict[str, Any]:
    """Build a minimal raw AIPerfJob CR dict."""
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


def _api_exception(status: int) -> ApiException:
    """Construct an ApiException with the given HTTP status code."""
    return ApiException(status=status, reason=f"err-{status}")


class TestListAIPerfJobsNamespaceResolution:
    """Verify namespace=None fallback to 'default'."""

    @pytest.mark.asyncio
    async def test_none_namespace_resolves_to_default(self) -> None:
        """Passing namespace=None with all_namespaces=False uses 'default'."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": []}
        )
        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            await list_aiperf_jobs(api, namespace=None)
        kwargs = mock_custom.list_namespaced_custom_object.call_args.kwargs
        assert kwargs["namespace"] == "default"

    @pytest.mark.asyncio
    async def test_empty_items_returns_empty_list(self) -> None:
        """Missing or empty items key yields []."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(return_value={})
        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await list_aiperf_jobs(api, namespace="ns")
        assert result == []


class TestFindAIPerfJobClusterWide:
    """Verify cluster-wide fallback path when namespace is None."""

    @pytest.mark.asyncio
    async def test_cluster_wide_adds_field_selector(self) -> None:
        """namespace=None -> skip get, use list_cluster with metadata.name field selector."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock()
        mock_custom.list_cluster_custom_object = AsyncMock(
            return_value={"items": [_raw_aiperfjob(name="hit", job_id="j")]}
        )
        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_job(api, "hit")
        mock_custom.get_namespaced_custom_object.assert_not_called()
        kwargs = mock_custom.list_cluster_custom_object.call_args.kwargs
        assert kwargs["field_selector"] == "metadata.name=hit"
        assert result is not None
        assert result.name == "hit"

    @pytest.mark.asyncio
    async def test_match_by_metadata_name(self) -> None:
        """Fallback list result that matches metadata.name (not jobId) still resolves
        when no namespace is supplied."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object = AsyncMock(
            return_value={
                "items": [
                    _raw_aiperfjob(name="other", job_id="other-id"),
                    _raw_aiperfjob(name="target-name", job_id="unrelated"),
                ]
            }
        )
        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_job(api, "target-name", namespace=None)
        assert result is not None
        assert result.name == "target-name"

    @pytest.mark.asyncio
    async def test_fallback_list_404_returns_none(self) -> None:
        """404 on the fallback list is suppressed to None (CRD not installed)."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_job(api, "nope", namespace=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_list_non_404_raises(self) -> None:
        """A 500 on the cluster-wide fallback list propagates to the caller."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object = AsyncMock(
            side_effect=_api_exception(500)
        )
        with (
            patch(
                "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            pytest.raises(ApiException),
        ):
            await find_aiperf_job(api, "x", namespace=None)

    @pytest.mark.asyncio
    async def test_namespaced_404_does_not_call_cluster_wide(self) -> None:
        """Cross-namespace leak guard: a 404 on the namespaced GET must NOT
        fall back to a cluster-wide scan, because a same-named CR in another
        namespace is a different resource.
        """
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
        mock_custom.list_cluster_custom_object = AsyncMock(return_value={"items": []})
        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_job(api, "x", namespace="ns")
        assert result is None
        mock_custom.list_cluster_custom_object.assert_not_called()


class TestGetRawAIPerfJobStatusEdges:
    """Verify raw-status helper edge cases not covered elsewhere."""

    @pytest.mark.asyncio
    async def test_missing_status_key_returns_empty(self) -> None:
        """A CR object lacking the status key entirely returns {}."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            return_value={"metadata": {"name": "x"}, "spec": {}}
        )
        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await get_raw_aiperfjob_status(api, "x", "ns")
        assert result == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status",
        [
            param(500, id="server_error"),
            param(403, id="forbidden"),
            param(400, id="bad_request"),
        ],
    )  # fmt: skip
    async def test_any_api_error_returns_empty(self, status: int) -> None:
        """Any ApiException (not just 404) is swallowed and returns {}."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(status)
        )
        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await get_raw_aiperfjob_status(api, "x", "ns")
        assert result == {}


class TestCancelAIPerfJobPropagatesErrors:
    """Verify cancel surfaces every ApiException (nothing suppressed)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status",
        [
            param(404, id="not_found"),
            param(403, id="forbidden"),
            param(409, id="conflict"),
            param(500, id="server_error"),
        ],
    )  # fmt: skip
    async def test_propagates_api_exception(self, status: int) -> None:
        """Every ApiException status reaches the caller unchanged."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(status)
        )
        with (
            patch(
                "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            pytest.raises(ApiException) as exc_info,
        ):
            await cancel_aiperf_job(api, "n", "default")
        assert exc_info.value.status == status


def _raw_aiperfsweep(
    name: str = "test-sweep",
    namespace: str = "default",
    phase: str = "Running",
    total_variations: int = 4,
    max_total_runs: int = 12,
    completed_runs: int = 1,
    failed_runs: int = 0,
    run_epoch: int = 1700000000,
    created: str = "2026-01-15T10:30:00Z",
) -> dict[str, Any]:
    """Build a minimal raw AIPerfSweep CR dict."""
    return {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": created,
        },
        "status": {
            "phase": phase,
            "totalVariations": total_variations,
            "maxTotalRuns": max_total_runs,
            "completedRuns": completed_runs,
            "failedRuns": failed_runs,
            "runEpoch": run_epoch,
        },
    }


class TestFindAIPerfSweep:
    """Tests for find_aiperf_sweep — namespaced + cluster-wide fallback."""

    @pytest.mark.asyncio
    async def test_find_aiperf_sweep_namespaced_returns_typed_info(self) -> None:
        """Direct namespaced lookup returns a populated AIPerfSweepInfo."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            return_value=_raw_aiperfsweep(
                name="my-sweep",
                namespace="bench-ns",
                phase="Running",
                total_variations=6,
                max_total_runs=18,
                completed_runs=3,
                failed_runs=1,
            )
        )
        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_sweep(api, "my-sweep", namespace="bench-ns")

        assert result is not None
        assert result.name == "my-sweep"
        assert result.namespace == "bench-ns"
        assert result.phase == "Running"
        assert result.total_variations == 6
        assert result.max_total_runs == 18
        assert result.completed_runs == 3
        assert result.failed_runs == 1
        kwargs = mock_custom.get_namespaced_custom_object.call_args.kwargs
        assert kwargs["plural"] == "aiperfsweeps"
        assert kwargs["group"] == "aiperf.nvidia.com"
        assert kwargs["version"] == "v1alpha1"

    @pytest.mark.asyncio
    async def test_find_aiperf_sweep_404_returns_none(self) -> None:
        """A 404 in both namespaced get and cluster-wide list returns None."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
        mock_custom.list_cluster_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_sweep(api, "missing", namespace="ns")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_aiperf_sweep_cluster_wide_fallback_when_namespace_none(
        self,
    ) -> None:
        """namespace=None skips direct get and uses cluster list with field selector."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock()
        mock_custom.list_cluster_custom_object = AsyncMock(
            return_value={
                "items": [
                    _raw_aiperfsweep(name="found-sweep", namespace="other-ns"),
                ]
            }
        )
        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_sweep(api, "found-sweep")

        mock_custom.get_namespaced_custom_object.assert_not_called()
        kwargs = mock_custom.list_cluster_custom_object.call_args.kwargs
        assert kwargs["field_selector"] == "metadata.name=found-sweep"
        assert kwargs["plural"] == "aiperfsweeps"
        assert result is not None
        assert result.name == "found-sweep"
        assert result.namespace == "other-ns"

    @pytest.mark.asyncio
    async def test_find_aiperf_sweep_other_api_exception_propagates(self) -> None:
        """A non-404 ApiException on the namespaced get propagates."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(500)
        )
        with (
            patch(
                "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            pytest.raises(ApiException) as exc_info,
        ):
            await find_aiperf_sweep(api, "x", namespace="ns")
        assert exc_info.value.status == 500

    @pytest.mark.asyncio
    async def test_find_aiperf_sweep_namespaced_404_does_not_fall_back_to_cluster(
        self,
    ) -> None:
        """Cross-namespace leak guard: 404 in namespaced GET must NOT trigger
        a cluster-wide scan that would return a same-named sweep from another
        namespace.
        """
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
        # Cluster-wide list would return a same-named sweep from another ns;
        # we must not call it.
        mock_custom.list_cluster_custom_object = AsyncMock(
            return_value={
                "items": [_raw_aiperfsweep(name="x", namespace="other-ns")],
            }
        )
        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_sweep(api, "x", namespace="ns")
        assert result is None
        mock_custom.list_cluster_custom_object.assert_not_called()
