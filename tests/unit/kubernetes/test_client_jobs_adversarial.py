# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes client job, sweep, and JobSet lookup helpers.

Focuses on:
- namespace fallback and all-namespaces routing for CR and JobSet list calls
- 404 suppression versus non-404 propagation at Kubernetes trust boundaries
- empty, stale, and malformed Kubernetes API response shapes
- exact field/label selector construction for fallback lookup paths
- cancellation conflict behavior and merge-patch wire shape

Out of scope: manifest construction and pod/resource helpers; see sibling
``test_jobset_resources.py`` and ``test_jobset_builder.py`` for those contracts.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.client_jobs import (
    cancel_aiperf_job,
    find_aiperf_job,
    find_aiperf_sweep,
    get_raw_aiperfjob,
    list_aiperf_jobs,
)
from aiperf.kubernetes.client_jobsets import find_jobset, list_jobsets

# =============================================================================
# Helpers
# =============================================================================


def _api_exception(status: int) -> ApiException:
    """Build a Kubernetes ApiException with a stable status for assertions."""
    return ApiException(status=status, reason=f"aiperf-api-{status}")


def _raw_aiperfjob(
    *,
    name: str = "llama3-throughput-v07",
    namespace: str = "aiperf-benchmarks",
    phase: str | None = "Running",
    job_id: str = "job-2026-05-18-9c3a",
    created: str = "2026-05-18T10:30:00Z",
    phases: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build a minimal raw AIPerfJob CR response from the apiserver."""
    status: dict[str, object] = {"phase": phase, "jobId": job_id}
    if phases is not None:
        status["phases"] = phases
    return {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": created,
        },
        "spec": {
            "benchmark": {
                "models": ["meta-llama/Llama-3-8B"],
                "endpoint": {"url": "http://localhost:8000"},
            }
        },
        "status": status,
    }


def _raw_aiperfsweep(
    *,
    name: str = "latency-grid-search",
    namespace: str = "aiperf-benchmarks",
    phase: str | None = "Running",
    created: str = "2026-05-18T10:30:00Z",
) -> dict[str, object]:
    """Build a minimal raw AIPerfSweep CR response from the apiserver."""
    return {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": created,
        },
        "status": {
            "phase": phase,
            "runEpoch": 1779129000,
            "totalVariations": 4,
            "maxTotalRuns": 12,
            "completedRuns": 2,
            "failedRuns": 0,
        },
    }


def _raw_jobset(
    *,
    name: str = "llama3-throughput-v07",
    namespace: str = "aiperf-benchmarks",
    labels: dict[str, str] | None = None,
    conditions: list[dict[str, str]] | None = None,
    created: str = "2026-05-18T10:30:00Z",
) -> dict[str, object]:
    """Build a minimal raw JobSet response from the apiserver."""
    return {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": created,
            "labels": labels or {"app": "aiperf"},
        },
        "status": {"conditions": conditions or []},
    }


# =============================================================================
# AIPerfJob list and lookup adversaries
# =============================================================================


class TestAIPerfJobListAdversarial:
    """List path selector, namespace, and API-error edge contracts."""

    @pytest.mark.asyncio
    async def test_list_aiperf_jobs_all_namespaces_ignores_namespace_uses_cluster_scope(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object = AsyncMock(return_value={"items": []})
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": []}
        )

        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await list_aiperf_jobs(
                api,
                namespace="wrong-tenant-namespace",
                all_namespaces=True,
            )

        assert result == []
        mock_custom.list_cluster_custom_object.assert_awaited_once()
        mock_custom.list_namespaced_custom_object.assert_not_called()
        assert (
            "namespace" not in mock_custom.list_cluster_custom_object.call_args.kwargs
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,raises",
        [
            param(404, False, id="crd-not-installed-suppressed"),
            param(403, True, id="forbidden-propagates"),
            param(500, True, id="apiserver-error-propagates"),
        ],
    )  # fmt: skip
    async def test_list_aiperf_jobs_api_error_status_follows_contract(
        self,
        status: int,
        raises: bool,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(status)
        )

        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            if raises:
                with pytest.raises(ApiException) as exc_info:
                    await list_aiperf_jobs(api, namespace="aiperf-benchmarks")
                assert exc_info.value.status == status
            else:
                assert await list_aiperf_jobs(api, namespace="aiperf-benchmarks") == []


class TestFindAIPerfJobAdversarial:
    """Find path stale-response and selector contracts."""

    @pytest.mark.asyncio
    async def test_find_aiperf_job_cluster_empty_items_returns_none(self) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object = AsyncMock(return_value={"items": []})

        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_job(api, "llama3-throughput-v07")

        assert result is None
        assert (
            mock_custom.list_cluster_custom_object.call_args.kwargs["field_selector"]
            == "metadata.name=llama3-throughput-v07"
        )

    @pytest.mark.asyncio
    async def test_find_aiperf_job_cluster_stale_name_and_job_id_mismatch_ignored(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object = AsyncMock(
            return_value={
                "items": [
                    _raw_aiperfjob(
                        name="stale-llama3-throughput-v06",
                        job_id="job-previous-run-1d8f",
                    )
                ]
            }
        )

        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_job(api, "llama3-throughput-v07")

        assert result is None

    @pytest.mark.asyncio
    async def test_find_aiperf_job_namespaced_missing_status_defaults_to_pending(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        raw = _raw_aiperfjob(name="llama3-throughput-v07")
        raw.pop("status")
        mock_custom.get_namespaced_custom_object = AsyncMock(return_value=raw)

        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_job(
                api,
                "llama3-throughput-v07",
                namespace="aiperf-benchmarks",
            )

        assert result is not None
        assert result.phase == "Pending"
        assert result.job_id == "llama3-throughput-v07"

    @pytest.mark.asyncio
    async def test_find_aiperf_job_malformed_progress_percent_raises_value_error(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            return_value=_raw_aiperfjob(
                phases={"profiling": {"requestsProgressPercent": "many"}}
            )
        )

        with (
            patch(
                "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            pytest.raises(ValueError, match="many"),
        ):
            await find_aiperf_job(
                api,
                "llama3-throughput-v07",
                namespace="aiperf-benchmarks",
            )


# =============================================================================
# AIPerfSweep lookup adversaries
# =============================================================================


class TestFindAIPerfSweepAdversarial:
    """Sweep lookup selector and stale-response contracts."""

    @pytest.mark.asyncio
    async def test_find_aiperf_sweep_cluster_empty_items_returns_none(self) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object = AsyncMock(return_value={"items": []})

        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_sweep(api, "latency-grid-search")

        assert result is None
        kwargs = mock_custom.list_cluster_custom_object.call_args.kwargs
        assert kwargs["plural"] == "aiperfsweeps"
        assert kwargs["field_selector"] == "metadata.name=latency-grid-search"

    @pytest.mark.asyncio
    async def test_find_aiperf_sweep_cluster_stale_name_mismatch_ignored(self) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object = AsyncMock(
            return_value={"items": [_raw_aiperfsweep(name="old-latency-grid-search")]}
        )

        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_sweep(api, "latency-grid-search")

        assert result is None

    @pytest.mark.asyncio
    async def test_find_aiperf_sweep_namespaced_empty_phase_defaults_to_pending(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            return_value=_raw_aiperfsweep(phase="")
        )

        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_aiperf_sweep(
                api,
                "latency-grid-search",
                namespace="aiperf-benchmarks",
            )

        assert result is not None
        assert result.phase == "Pending"


# =============================================================================
# JobSet list and lookup adversaries
# =============================================================================


class TestJobSetLookupAdversarial:
    """JobSet selector construction and fallback lookup contracts."""

    @pytest.mark.asyncio
    async def test_list_jobsets_all_namespaces_ignores_namespace_uses_cluster_scope(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object = AsyncMock(return_value={"items": []})
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": []}
        )

        with patch(
            "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await list_jobsets(
                api,
                namespace="wrong-tenant-namespace",
                all_namespaces=True,
                job_id="job-2026-05-18-9c3a",
            )

        assert result == []
        mock_custom.list_cluster_custom_object.assert_awaited_once()
        mock_custom.list_namespaced_custom_object.assert_not_called()
        kwargs = mock_custom.list_cluster_custom_object.call_args.kwargs
        assert kwargs["label_selector"] == (
            "app=aiperf,aiperf.nvidia.com/job-id=job-2026-05-18-9c3a"
        )
        assert "namespace" not in kwargs

    @pytest.mark.asyncio
    async def test_find_jobset_label_hit_does_not_execute_name_fallback(self) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": [_raw_jobset(name="llama3-throughput-v07")]}
        )

        with patch(
            "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_jobset(
                api,
                "job-2026-05-18-9c3a",
                namespace="aiperf-benchmarks",
            )

        assert result is not None
        assert result.name == "llama3-throughput-v07"
        mock_custom.list_namespaced_custom_object.assert_awaited_once()
        assert "field_selector" not in (
            mock_custom.list_namespaced_custom_object.call_args.kwargs
        )

    @pytest.mark.asyncio
    async def test_find_jobset_empty_first_pass_uses_name_field_selector_then_none(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            side_effect=[{"items": []}, {"items": []}]
        )

        with patch(
            "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await find_jobset(
                api,
                "llama3-throughput-v07",
                namespace="aiperf-benchmarks",
            )

        assert result is None
        first_kwargs = mock_custom.list_namespaced_custom_object.call_args_list[
            0
        ].kwargs
        second_kwargs = mock_custom.list_namespaced_custom_object.call_args_list[
            1
        ].kwargs
        assert first_kwargs["label_selector"] == (
            "app=aiperf,aiperf.nvidia.com/job-id=llama3-throughput-v07"
        )
        assert "field_selector" not in first_kwargs
        assert second_kwargs["label_selector"] == "app=aiperf"
        assert second_kwargs["field_selector"] == "metadata.name=llama3-throughput-v07"

    @pytest.mark.asyncio
    async def test_list_jobsets_completed_filter_drops_stale_running_status(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.list_namespaced_custom_object = AsyncMock(
            return_value={"items": [_raw_jobset(name="llama3-throughput-v07")]}
        )

        with patch(
            "aiperf.kubernetes.client_jobsets.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await list_jobsets(
                api,
                namespace="aiperf-benchmarks",
                status_filter="Completed",
            )

        assert result == []


# =============================================================================
# Raw status and cancellation adversaries
# =============================================================================


class TestRawAndCancelAdversarial:
    """Best-effort raw lookup and cancellation conflict contracts."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status",
        [
            param(404, id="not-found"),
            param(403, id="forbidden"),
            param(500, id="server-error"),
        ],
    )  # fmt: skip
    async def test_get_raw_aiperfjob_any_api_error_returns_none(
        self,
        status: int,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(status)
        )

        with patch(
            "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            result = await get_raw_aiperfjob(
                api,
                namespace="aiperf-benchmarks",
                name="llama3-throughput-v07",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_aiperf_job_conflict_propagates_and_preserves_merge_patch(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        mock_custom = MagicMock()
        mock_custom.patch_namespaced_custom_object = AsyncMock(
            side_effect=_api_exception(409)
        )

        with (
            patch(
                "aiperf.kubernetes.client_jobs.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            pytest.raises(ApiException) as exc_info,
        ):
            await cancel_aiperf_job(
                api,
                "llama3-throughput-v07",
                "aiperf-benchmarks",
            )

        assert exc_info.value.status == 409
        mock_custom.patch_namespaced_custom_object.assert_awaited_once()
        kwargs = mock_custom.patch_namespaced_custom_object.call_args.kwargs
        assert kwargs["name"] == "llama3-throughput-v07"
        assert kwargs["namespace"] == "aiperf-benchmarks"
        assert kwargs["body"] == {"spec": {"cancel": True}}
        assert kwargs["_content_type"] == "application/merge-patch+json"
