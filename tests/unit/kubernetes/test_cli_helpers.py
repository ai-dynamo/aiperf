# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.cli_helpers and AIPerfKubeClient."""

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import kr8s
import pytest

from aiperf.kubernetes.cli_helpers import ResolvedJob, format_age, resolve_job
from aiperf.kubernetes.client import AIPerfKubeClient
from aiperf.kubernetes.constants import Labels
from aiperf.kubernetes.models import AIPerfJobInfo, JobSetInfo
from tests.harness.k8s import (
    async_list,
    create_server_error,
    make_kr8s_object,
)


def _raw_jobset(status_obj: dict | None = None) -> dict[str, Any]:
    """Build a minimal raw JobSet dict for testing."""
    raw: dict[str, Any] = {
        "metadata": {"name": "test", "namespace": "default"},
    }
    if status_obj is not None:
        raw["status"] = status_obj
    return raw


class TestJobSetInfoStatus:
    """Tests for status parsing via JobSetInfo.from_raw."""

    @pytest.mark.parametrize(
        "status_obj,expected",
        [
            (
                {"conditions": [{"type": "Completed", "status": "True"}]},
                "Completed",
            ),
            (
                {
                    "conditions": [{"type": "Failed", "status": "True"}],
                    "replicatedJobsStatus": [
                        {"name": "controller", "failed": 1, "ready": 0, "succeeded": 0}
                    ],
                },
                "Failed",
            ),
            ({"conditions": [], "ready": 1}, "Running"),
            ({"conditions": [], "ready": 0}, "Running"),
            ({}, "Running"),
            (
                {"conditions": [{"type": "Completed", "status": "False"}]},
                "Running",
            ),
        ],  # fmt: skip
    )
    def test_status_from_raw(self, status_obj: dict, expected: str) -> None:
        """Test extracting status from various JobSet objects."""
        info = JobSetInfo.from_raw(_raw_jobset(status_obj))
        assert info.status == expected

    def test_status_no_status_key(self) -> None:
        """Test JobSet without status key defaults to Running."""
        info = JobSetInfo.from_raw(_raw_jobset())
        assert info.status == "Running"

    def test_status_completed_takes_priority(self) -> None:
        """Test JobSet with multiple conditions - Completed takes priority."""
        info = JobSetInfo.from_raw(
            _raw_jobset(
                {
                    "conditions": [
                        {"type": "Running", "status": "True"},
                        {"type": "Completed", "status": "True"},
                    ]
                }
            )
        )
        assert info.status == "Completed"


class TestFormatAge:
    """Tests for format_age function."""

    def test_format_age_seconds(self) -> None:
        """Test formatting age in seconds."""
        now = datetime.now(timezone.utc)
        timestamp = (now - timedelta(seconds=30)).isoformat().replace("+00:00", "Z")
        result = format_age(timestamp)
        # Allow some tolerance for test execution time
        assert result.endswith("s")
        assert int(result[:-1]) >= 29

    def test_format_age_minutes(self) -> None:
        """Test formatting age in minutes."""
        now = datetime.now(timezone.utc)
        timestamp = (now - timedelta(minutes=15)).isoformat().replace("+00:00", "Z")
        result = format_age(timestamp)
        assert result == "15m"

    def test_format_age_hours(self) -> None:
        """Test formatting age in hours."""
        now = datetime.now(timezone.utc)
        timestamp = (now - timedelta(hours=3)).isoformat().replace("+00:00", "Z")
        result = format_age(timestamp)
        assert result == "3h"

    def test_format_age_empty_string(self) -> None:
        """Test formatting with empty string."""
        assert format_age("") == "Unknown"

    def test_format_age_boundary_59_seconds(self) -> None:
        """Test boundary at 59 seconds."""
        now = datetime.now(timezone.utc)
        timestamp = (now - timedelta(seconds=59)).isoformat().replace("+00:00", "Z")
        result = format_age(timestamp)
        assert result.endswith("s")

    def test_format_age_boundary_60_seconds(self) -> None:
        """Test boundary at exactly 60 seconds becomes 1m."""
        now = datetime.now(timezone.utc)
        timestamp = (now - timedelta(seconds=60)).isoformat().replace("+00:00", "Z")
        result = format_age(timestamp)
        assert result == "1m"


class TestAIPerfKubeClientCreate:
    """Tests for AIPerfKubeClient.create classmethod."""

    async def test_create_returns_client(self) -> None:
        """Test that create() returns an AIPerfKubeClient."""
        mock_api = MagicMock(spec=kr8s.Api)
        with patch(
            "aiperf.kubernetes.client.get_api", new_callable=AsyncMock
        ) as mock_get_api:
            mock_get_api.return_value = mock_api
            result = await AIPerfKubeClient.create()
            mock_get_api.assert_called_once_with(kubeconfig=None, kube_context=None)
            assert isinstance(result, AIPerfKubeClient)
            assert result.api is mock_api

    async def test_create_with_kubeconfig(self) -> None:
        """Test that create() passes kubeconfig and kube_context."""
        mock_api = MagicMock(spec=kr8s.Api)
        with patch(
            "aiperf.kubernetes.client.get_api", new_callable=AsyncMock
        ) as mock_get_api:
            mock_get_api.return_value = mock_api
            result = await AIPerfKubeClient.create(
                kubeconfig="/custom/kubeconfig", kube_context="my-context"
            )
            mock_get_api.assert_called_once_with(
                kubeconfig="/custom/kubeconfig", kube_context="my-context"
            )
            assert result.api is mock_api


class TestAIPerfKubeClientLabelSelectors:
    """Tests for label selector methods."""

    def test_job_selector(self) -> None:
        """Test job_selector builds correct label string."""
        selector = AIPerfKubeClient.job_selector("abc123")
        assert selector == "app=aiperf,aiperf.nvidia.com/job-id=abc123"

    def test_controller_selector(self) -> None:
        """Test controller_selector builds correct label string."""
        selector = AIPerfKubeClient.controller_selector("abc123")
        assert "app=aiperf" in selector
        assert "aiperf.nvidia.com/job-id=abc123" in selector
        assert "jobset.sigs.k8s.io/replicatedjob-name=controller" in selector


class TestFindJobset:
    """Tests for AIPerfKubeClient.find_jobset method."""

    async def test_find_jobset_found_by_label(
        self, mock_kube_client, mock_kr8s_api, sample_running_jobset
    ) -> None:
        """Test finding JobSet by job ID label."""
        kr8s_obj = make_kr8s_object(sample_running_jobset)

        mock_kr8s_api.async_get = MagicMock(side_effect=[async_list([kr8s_obj])])

        result = await mock_kube_client.find_jobset("test-job-123", namespace="default")

        assert result is not None
        assert isinstance(result, JobSetInfo)
        assert result.name == "aiperf-test-job"
        assert result.namespace == "default"
        assert result.status == "Running"

    async def test_find_jobset_found_cluster_wide(
        self, mock_kube_client, mock_kr8s_api, sample_completed_jobset
    ) -> None:
        """Test finding JobSet across all namespaces."""
        kr8s_obj = make_kr8s_object(sample_completed_jobset)

        mock_kr8s_api.async_get = MagicMock(side_effect=[async_list([kr8s_obj])])

        result = await mock_kube_client.find_jobset("test-job-123", namespace=None)

        assert result is not None
        assert result.status == "Completed"

    async def test_find_jobset_fallback_to_name(
        self, mock_kube_client, mock_kr8s_api, sample_running_jobset
    ) -> None:
        """Test fallback to matching by JobSet name when label search fails."""
        kr8s_obj = make_kr8s_object(sample_running_jobset)

        mock_kr8s_api.async_get = MagicMock(
            side_effect=[async_list([]), async_list([kr8s_obj])]
        )

        result = await mock_kube_client.find_jobset(
            "aiperf-test-job", namespace="default"
        )

        assert result is not None
        assert result.name == "aiperf-test-job"
        assert mock_kr8s_api.async_get.call_count == 2

    async def test_find_jobset_not_found(self, mock_kube_client, mock_kr8s_api) -> None:
        """Test finding JobSet that doesn't exist."""
        mock_kr8s_api.async_get = MagicMock(
            side_effect=[async_list([]), async_list([])]
        )

        result = await mock_kube_client.find_jobset("nonexistent", namespace="default")

        assert result is None


class TestListJobsets:
    """Tests for AIPerfKubeClient.list_jobsets method."""

    async def test_list_jobsets_default_namespace(
        self, mock_kube_client, mock_kr8s_api, sample_jobset
    ) -> None:
        """Test listing JobSets in default namespace."""
        kr8s_obj = make_kr8s_object(sample_jobset)
        mock_kr8s_api.async_get = MagicMock(return_value=async_list([kr8s_obj]))

        result = await mock_kube_client.list_jobsets()

        assert len(result) == 1
        mock_kr8s_api.async_get.assert_called_once()

    async def test_list_jobsets_all_namespaces(
        self, mock_kube_client, mock_kr8s_api, sample_jobset
    ) -> None:
        """Test listing JobSets across all namespaces."""
        kr8s_obj = make_kr8s_object(sample_jobset)
        mock_kr8s_api.async_get = MagicMock(return_value=async_list([kr8s_obj]))

        result = await mock_kube_client.list_jobsets(all_namespaces=True)

        assert len(result) == 1
        call_kwargs = mock_kr8s_api.async_get.call_args.kwargs
        assert call_kwargs["namespace"] == kr8s.ALL

    async def test_list_jobsets_with_job_id_filter(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        """Test listing JobSets filtered by job_id."""
        mock_kr8s_api.async_get = MagicMock(return_value=async_list([]))

        await mock_kube_client.list_jobsets(job_id="specific-job")

        call_kwargs = mock_kr8s_api.async_get.call_args.kwargs
        assert f"{Labels.JOB_ID}=specific-job" in call_kwargs["label_selector"]

    async def test_list_jobsets_404_returns_empty(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        """Test that 404 error returns empty list."""
        mock_kr8s_api.async_get = MagicMock(
            side_effect=create_server_error(404, "Not Found")
        )

        result = await mock_kube_client.list_jobsets()

        assert result == []

    async def test_list_jobsets_other_error_raises(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        """Test that non-404 errors are raised."""
        mock_kr8s_api.async_get = MagicMock(
            side_effect=create_server_error(500, "Internal Server Error")
        )

        with pytest.raises(kr8s.ServerError):
            await mock_kube_client.list_jobsets()

    async def test_list_jobsets_with_status_filter(
        self,
        mock_kube_client,
        mock_kr8s_api,
        sample_running_jobset,
        sample_completed_jobset,
    ) -> None:
        """Test listing JobSets filtered by status."""
        objs = [
            make_kr8s_object(sample_running_jobset),
            make_kr8s_object(sample_completed_jobset),
        ]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(objs))

        result = await mock_kube_client.list_jobsets(status_filter="Running")

        assert len(result) == 1
        assert result[0].status == "Running"

    async def test_list_jobsets_specific_namespace(
        self, mock_kube_client, mock_kr8s_api, sample_jobset
    ) -> None:
        """Test listing JobSets in a specific namespace."""
        kr8s_obj = make_kr8s_object(sample_jobset)
        mock_kr8s_api.async_get = MagicMock(return_value=async_list([kr8s_obj]))

        await mock_kube_client.list_jobsets(namespace="custom-namespace")

        call_kwargs = mock_kr8s_api.async_get.call_args.kwargs
        assert call_kwargs["namespace"] == "custom-namespace"

    async def test_list_jobsets_sorted_by_creation_time(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        """Test that JobSets are sorted by creation time (newest first)."""
        older_jobset: dict[str, Any] = {
            "metadata": {
                "name": "older-job",
                "namespace": "default",
                "creationTimestamp": "2026-01-01T10:00:00Z",
                "labels": {"app": "aiperf"},
            },
            "status": {"conditions": [], "ready": 0},
        }
        newer_jobset: dict[str, Any] = {
            "metadata": {
                "name": "newer-job",
                "namespace": "default",
                "creationTimestamp": "2026-01-15T10:00:00Z",
                "labels": {"app": "aiperf"},
            },
            "status": {"conditions": [], "ready": 0},
        }
        objs = [make_kr8s_object(older_jobset), make_kr8s_object(newer_jobset)]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(objs))

        result = await mock_kube_client.list_jobsets()

        assert len(result) == 2
        assert result[0].name == "newer-job"
        assert result[1].name == "older-job"


class TestDeleteJobset:
    """Tests for AIPerfKubeClient.delete_jobset method."""

    @staticmethod
    def _patch_resource_classes(
        jobset_get_effect=None,
        configmap_get_effect=None,
        role_get_effect=None,
        rolebinding_get_effect=None,
    ):
        """Build a context manager that patches the four resource class .get methods."""

        def _make_patch(target, effect):
            if isinstance(effect, BaseException):
                return patch(target, new_callable=AsyncMock, side_effect=effect)
            return patch(target, new_callable=AsyncMock, return_value=effect)

        stack = ExitStack()
        patches = {
            "aiperf.kubernetes.kr8s_resources.AsyncJobSet.get": jobset_get_effect,
            "kr8s.asyncio.objects.ConfigMap.get": configmap_get_effect,
            "kr8s.asyncio.objects.Role.get": role_get_effect,
            "kr8s.asyncio.objects.RoleBinding.get": rolebinding_get_effect,
        }
        for target, effect in patches.items():
            stack.enter_context(_make_patch(target, effect))
        return stack

    async def test_delete_jobset_success(self, mock_kube_client, capsys) -> None:
        """Test successful JobSet deletion."""
        mock_jobset = AsyncMock()
        mock_configmap = AsyncMock()
        mock_role = AsyncMock()
        mock_rolebinding = AsyncMock()

        with self._patch_resource_classes(
            jobset_get_effect=mock_jobset,
            configmap_get_effect=mock_configmap,
            role_get_effect=mock_role,
            rolebinding_get_effect=mock_rolebinding,
        ):
            await mock_kube_client.delete_jobset("test-job", "default")

        mock_jobset.delete.assert_awaited_once()
        mock_configmap.delete.assert_awaited_once()
        mock_role.delete.assert_awaited_once()
        mock_rolebinding.delete.assert_awaited_once()

        captured = capsys.readouterr()
        assert "Deleted JobSet/test-job" in captured.out
        assert "Deleted ConfigMap/test-job-config" in captured.out
        assert "Deleted Role/test-job-role" in captured.out
        assert "Deleted RoleBinding/test-job-binding" in captured.out

    async def test_delete_jobset_not_found(self, mock_kube_client, capsys) -> None:
        """Test deletion when JobSet doesn't exist."""
        not_found = kr8s.NotFoundError("not found")
        with self._patch_resource_classes(
            jobset_get_effect=not_found,
            configmap_get_effect=not_found,
            role_get_effect=not_found,
            rolebinding_get_effect=not_found,
        ):
            await mock_kube_client.delete_jobset("test-job", "default")

        captured = capsys.readouterr()
        assert "JobSet/test-job not found" in captured.out

    async def test_delete_jobset_associated_resource_server_error(
        self, mock_kube_client, capsys
    ) -> None:
        """Test deletion when associated resource fails with non-404 ServerError."""
        mock_jobset = AsyncMock()
        err = create_server_error(500, "Internal Server Error")
        not_found = kr8s.NotFoundError("not found")

        with self._patch_resource_classes(
            jobset_get_effect=mock_jobset,
            configmap_get_effect=err,
            role_get_effect=not_found,
            rolebinding_get_effect=not_found,
        ):
            await mock_kube_client.delete_jobset("test-job", "default")

        captured = capsys.readouterr()
        assert "Failed to delete ConfigMap" in captured.out


class TestLabelConstants:
    """Tests for label constants."""

    def test_aiperf_label_format(self) -> None:
        """Test Labels.SELECTOR constant format."""
        assert Labels.SELECTOR == "app=aiperf"

    def test_aiperf_job_id_label_format(self) -> None:
        """Test Labels.JOB_ID constant format."""
        assert Labels.JOB_ID == "aiperf.nvidia.com/job-id"


class TestJobSetInfo:
    """Tests for JobSetInfo dataclass."""

    def test_jobset_info_creation(self, sample_jobset) -> None:
        """Test creating JobSetInfo instance."""
        info = JobSetInfo(
            name="test",
            namespace="default",
            jobset=sample_jobset,
            status="Running",
        )
        assert info.name == "test"
        assert info.namespace == "default"
        assert info.status == "Running"
        assert info.jobset == sample_jobset


# ============================================================
# ResolvedJob
# ============================================================


def _make_job_info(**overrides: Any) -> AIPerfJobInfo:
    """Build an AIPerfJobInfo with sensible defaults, overridden by kwargs."""
    defaults: dict[str, Any] = {
        "name": "my-job",
        "namespace": "bench-ns",
        "phase": "Running",
        "job_id": "abc123",
        "jobset_name": "aiperf-abc123",
        "created": "2026-01-15T10:30:00Z",
    }
    defaults.update(overrides)
    return AIPerfJobInfo(**defaults)


class TestResolvedJob:
    """Tests for the ResolvedJob wrapper class."""

    def test_jobset_name_delegates_to_job_info(self) -> None:
        """Test that jobset_name property reads from job_info."""
        info = _make_job_info(jobset_name="js-name")
        resolved = ResolvedJob(name="n", job_info=info, client=MagicMock())
        assert resolved.jobset_name == "js-name"

    def test_jobset_name_none_when_absent(self) -> None:
        """Test that jobset_name is None when job_info has no jobset_name."""
        info = _make_job_info(jobset_name=None)
        resolved = ResolvedJob(name="n", job_info=info, client=MagicMock())
        assert resolved.jobset_name is None

    def test_namespace_delegates_to_job_info(self) -> None:
        """Test that namespace property reads from job_info."""
        info = _make_job_info(namespace="prod")
        resolved = ResolvedJob(name="n", job_info=info, client=MagicMock())
        assert resolved.namespace == "prod"

    def test_job_id_delegates_to_job_info(self) -> None:
        """Test that job_id property reads from job_info."""
        info = _make_job_info(job_id="xyz789")
        resolved = ResolvedJob(name="n", job_info=info, client=MagicMock())
        assert resolved.job_id == "xyz789"

    def test_name_stored_directly(self) -> None:
        """Test that the name attribute is stored directly on ResolvedJob."""
        info = _make_job_info()
        resolved = ResolvedJob(name="lookup-name", job_info=info, client=MagicMock())
        assert resolved.name == "lookup-name"

    def test_client_stored_directly(self) -> None:
        """Test that the client attribute is stored directly on ResolvedJob."""
        client = MagicMock()
        resolved = ResolvedJob(name="n", job_info=_make_job_info(), client=client)
        assert resolved.client is client


# ============================================================
# resolve_job
# ============================================================


class TestResolveJob:
    """Tests for the resolve_job async helper."""

    async def test_resolve_job_found_via_aiperfjob_cr(self) -> None:
        """Test that resolve_job returns ResolvedJob when AIPerfJob CR is found."""
        mock_client = AsyncMock(spec=AIPerfKubeClient)
        job_info = _make_job_info(name="bench-1", namespace="ns-1", job_id="bench-1")
        mock_client.find_job = AsyncMock(return_value=job_info)
        mock_client.find_jobset = AsyncMock()

        with patch(
            "aiperf.kubernetes.client.AIPerfKubeClient.create",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            result = await resolve_job("bench-1", namespace="ns-1")

        assert result is not None
        assert isinstance(result, ResolvedJob)
        assert result.name == "bench-1"
        assert result.job_info is job_info
        assert result.client is mock_client
        mock_client.find_jobset.assert_not_awaited()

    async def test_resolve_job_fallback_to_jobset(self) -> None:
        """Test that resolve_job falls back to JobSet when no AIPerfJob CR exists."""
        mock_client = AsyncMock(spec=AIPerfKubeClient)
        mock_client.find_job = AsyncMock(return_value=None)

        jobset_info = JobSetInfo(
            name="aiperf-fallback",
            namespace="default",
            jobset={
                "metadata": {
                    "name": "aiperf-fallback",
                    "namespace": "default",
                    "creationTimestamp": "2026-01-10T08:00:00Z",
                    "labels": {"aiperf.nvidia.com/job-id": "fallback-id"},
                    "annotations": {},
                },
                "status": {},
            },
            status="Running",
            model="llama-3",
            endpoint="http://llm:8000",
        )
        mock_client.find_jobset = AsyncMock(return_value=jobset_info)

        with patch(
            "aiperf.kubernetes.client.AIPerfKubeClient.create",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            result = await resolve_job("fallback-id", namespace="default")

        assert result is not None
        assert isinstance(result, ResolvedJob)
        assert result.job_info.namespace == "default"
        assert result.job_info.model == "llama-3"
        assert result.job_info.jobset_name == "aiperf-fallback"

    async def test_resolve_job_not_found_returns_none(self) -> None:
        """Test that resolve_job returns None when neither CR nor JobSet exists."""
        mock_client = AsyncMock(spec=AIPerfKubeClient)
        mock_client.find_job = AsyncMock(return_value=None)
        mock_client.find_jobset = AsyncMock(return_value=None)

        with patch(
            "aiperf.kubernetes.client.AIPerfKubeClient.create",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            result = await resolve_job("nonexistent", namespace="default")

        assert result is None

    async def test_resolve_job_no_job_id_no_last_benchmark_returns_none(self) -> None:
        """Test that resolve_job returns None when job_id is None and no last benchmark."""
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark", return_value=None
        ):
            result = await resolve_job(None)

        assert result is None

    async def test_resolve_job_passes_kubeconfig_and_context(self) -> None:
        """Test that kubeconfig and kube_context are forwarded to client creation."""
        mock_client = AsyncMock(spec=AIPerfKubeClient)
        mock_client.find_job = AsyncMock(
            return_value=_make_job_info(name="j", job_id="j")
        )

        with patch(
            "aiperf.kubernetes.client.AIPerfKubeClient.create",
            new_callable=AsyncMock,
            return_value=mock_client,
        ) as mock_create:
            await resolve_job(
                "j",
                namespace="ns",
                kubeconfig="/my/kube",
                kube_context="ctx",
            )

        mock_create.assert_awaited_once_with(kubeconfig="/my/kube", kube_context="ctx")

    async def test_resolve_job_not_found_prints_namespace_hint(self, capsys) -> None:
        """Test that resolve_job prints searched namespace when not found."""
        mock_client = AsyncMock(spec=AIPerfKubeClient)
        mock_client.find_job = AsyncMock(return_value=None)
        mock_client.find_jobset = AsyncMock(return_value=None)

        with patch(
            "aiperf.kubernetes.client.AIPerfKubeClient.create",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            await resolve_job("missing", namespace="prod")

        captured = capsys.readouterr()
        assert "prod" in captured.out

    async def test_resolve_job_not_found_defaults_to_benchmark_namespace(
        self, capsys
    ) -> None:
        """Test that resolve_job defaults to aiperf-benchmarks when namespace is None."""
        mock_client = AsyncMock(spec=AIPerfKubeClient)
        mock_client.find_job = AsyncMock(return_value=None)
        mock_client.find_jobset = AsyncMock(return_value=None)

        with patch(
            "aiperf.kubernetes.client.AIPerfKubeClient.create",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            await resolve_job("missing", namespace=None)

        captured = capsys.readouterr()
        assert "aiperf-benchmarks" in captured.out
