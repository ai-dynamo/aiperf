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
        """Fallback list result that matches metadata.name (not jobId) still resolves."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
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
            result = await find_aiperf_job(api, "target-name", namespace="ns")
        assert result is not None
        assert result.name == "target-name"

    @pytest.mark.asyncio
    async def test_fallback_list_404_returns_none(self) -> None:
        """404 on the fallback list is suppressed to None (CRD not installed)."""
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
            result = await find_aiperf_job(api, "nope", namespace="ns")
        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_list_non_404_raises(self) -> None:
        """A 500 on the fallback list propagates to the caller."""
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(404)
        )
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
            await find_aiperf_job(api, "x", namespace="ns")

    @pytest.mark.asyncio
    async def test_cluster_wide_without_namespace_no_field_selector_when_ns_given(
        self,
    ) -> None:
        """When a namespace is provided, the fallback list omits the field selector."""
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
            await find_aiperf_job(api, "x", namespace="ns")
        kwargs = mock_custom.list_cluster_custom_object.call_args.kwargs
        assert kwargs["field_selector"] is None


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
