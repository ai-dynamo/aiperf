# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.client module.

Focuses on:
- _kr8s_kwargs helper for kr8s workaround
- get_api factory function
- Pod operations (get_pod_summary, find_controller_pod, find_retrievable_pod)
- wait_for_controller_pod_ready timeout behavior
- Resource management (get_pods, delete_namespace, version)

Label selectors, find_jobset, list_jobsets, and delete_jobset are tested
in test_cli_helpers.py.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import kr8s
import pytest

from aiperf.kubernetes.client import _kr8s_kwargs, get_api
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.models import PodSummary
from tests.harness.k8s import (
    async_list,
    create_server_error,
    make_kr8s_object,
)

# ============================================================
# Helpers
# ============================================================


def _make_pod_raw(
    name: str = "test-pod-0",
    namespace: str = "default",
    phase: str = "Running",
    container_statuses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal raw pod dict for testing."""
    if container_statuses is None:
        container_statuses = [{"name": "main", "ready": True, "restartCount": 0}]
    return {
        "metadata": {"name": name, "namespace": namespace},
        "status": {"phase": phase, "containerStatuses": container_statuses},
    }


# ============================================================
# _kr8s_kwargs
# ============================================================


class TestKr8sKwargs:
    """Verify kwarg construction for kr8s api() calls."""

    def test_no_args_returns_empty(self) -> None:
        result = _kr8s_kwargs(kubeconfig=None, kube_context=None)
        assert result == {}

    def test_kubeconfig_only(self) -> None:
        result = _kr8s_kwargs(kubeconfig="/path/to/config", kube_context=None)
        assert result == {"kubeconfig": "/path/to/config"}
        assert "namespace" not in result

    def test_context_only_adds_namespace_default(self) -> None:
        result = _kr8s_kwargs(kubeconfig=None, kube_context="my-ctx")
        assert result == {"context": "my-ctx", "namespace": "default"}

    def test_both_args(self) -> None:
        result = _kr8s_kwargs(kubeconfig="/cfg", kube_context="ctx")
        assert result == {
            "kubeconfig": "/cfg",
            "context": "ctx",
            "namespace": "default",
        }


# ============================================================
# get_api
# ============================================================


class TestGetApi:
    """Verify get_api delegates to kr8s.asyncio.api correctly."""

    @pytest.mark.asyncio
    async def test_get_api_default(self) -> None:
        mock_api = MagicMock(spec=kr8s.Api)
        with (
            patch(
                "aiperf.kubernetes.client.suppress_noisy_http_loggers"
            ) as suppress_mock,
            patch("kr8s.asyncio.api", new_callable=AsyncMock, return_value=mock_api),
        ):
            result = await get_api()
        suppress_mock.assert_called_once_with()
        assert result is mock_api

    @pytest.mark.asyncio
    async def test_get_api_passes_kwargs(self) -> None:
        mock_api = MagicMock(spec=kr8s.Api)
        with (
            patch("aiperf.kubernetes.client.suppress_noisy_http_loggers"),
            patch(
                "kr8s.asyncio.api", new_callable=AsyncMock, return_value=mock_api
            ) as patched,
        ):
            await get_api(kubeconfig="/cfg", kube_context="ctx")
        patched.assert_awaited_once_with(
            kubeconfig="/cfg", context="ctx", namespace="default"
        )


# ============================================================
# get_pod_summary
# ============================================================


class TestGetPodSummary:
    """Verify pod readiness summary calculation."""

    @pytest.mark.asyncio
    async def test_all_pods_ready(self, mock_kube_client, mock_kr8s_api) -> None:
        pods = [
            make_kr8s_object(_make_pod_raw("pod-0", phase="Running")),
            make_kr8s_object(_make_pod_raw("pod-1", phase="Running")),
        ]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.get_pod_summary("my-jobset", "default")

        assert result == PodSummary(ready=2, total=2, restarts=0)

    @pytest.mark.asyncio
    async def test_no_pods(self, mock_kube_client, mock_kr8s_api) -> None:
        mock_kr8s_api.async_get = MagicMock(return_value=async_list([]))

        result = await mock_kube_client.get_pod_summary("my-jobset", "default")

        assert result == PodSummary(ready=0, total=0, restarts=0)

    @pytest.mark.asyncio
    async def test_pending_pod_not_ready(self, mock_kube_client, mock_kr8s_api) -> None:
        pods = [
            make_kr8s_object(
                _make_pod_raw(
                    "pod-0",
                    phase="Pending",
                    container_statuses=[
                        {"name": "main", "ready": False, "restartCount": 0}
                    ],
                )
            ),
        ]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.get_pod_summary("my-jobset", "default")

        assert result == PodSummary(ready=0, total=1, restarts=0)

    @pytest.mark.asyncio
    async def test_restarts_accumulated(self, mock_kube_client, mock_kr8s_api) -> None:
        pods = [
            make_kr8s_object(
                _make_pod_raw(
                    "pod-0",
                    phase="Running",
                    container_statuses=[
                        {"name": "c1", "ready": True, "restartCount": 3},
                        {"name": "c2", "ready": True, "restartCount": 2},
                    ],
                )
            ),
            make_kr8s_object(
                _make_pod_raw(
                    "pod-1",
                    phase="Running",
                    container_statuses=[
                        {"name": "c1", "ready": True, "restartCount": 1},
                    ],
                )
            ),
        ]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.get_pod_summary("my-jobset", "default")

        assert result == PodSummary(ready=2, total=2, restarts=6)

    @pytest.mark.asyncio
    async def test_running_but_container_not_ready(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        """Pod phase is Running but container readiness probe has not passed."""
        pods = [
            make_kr8s_object(
                _make_pod_raw(
                    "pod-0",
                    phase="Running",
                    container_statuses=[
                        {"name": "main", "ready": False, "restartCount": 0}
                    ],
                )
            ),
        ]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.get_pod_summary("my-jobset", "default")

        assert result.ready == 0
        assert result.total == 1

    @pytest.mark.asyncio
    async def test_empty_container_statuses(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        """Pod with empty containerStatuses is not counted as ready."""
        pods = [
            make_kr8s_object(
                _make_pod_raw("pod-0", phase="Running", container_statuses=[])
            ),
        ]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.get_pod_summary("my-jobset", "default")

        assert result.ready == 0
        assert result.total == 1

    @pytest.mark.asyncio
    async def test_no_container_statuses_key(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        """Pod raw dict missing containerStatuses entirely."""
        raw = {
            "metadata": {"name": "pod-0", "namespace": "default"},
            "status": {"phase": "Pending"},
        }
        pods = [make_kr8s_object(raw)]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.get_pod_summary("my-jobset", "default")

        assert result == PodSummary(ready=0, total=1, restarts=0)

    @pytest.mark.asyncio
    async def test_server_error_returns_empty_summary(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        mock_kr8s_api.async_get = MagicMock(side_effect=kr8s.ServerError("fail"))

        result = await mock_kube_client.get_pod_summary("my-jobset", "default")

        assert result == PodSummary(ready=0, total=0, restarts=0)


# ============================================================
# find_controller_pod
# ============================================================


class TestFindControllerPod:
    """Verify controller pod lookup and phase parsing."""

    @pytest.mark.asyncio
    async def test_found_running(self, mock_kube_client, mock_kr8s_api) -> None:
        pods = [make_kr8s_object(_make_pod_raw("ctrl-0", phase="Running"))]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_controller_pod("default", "job-123")

        assert result == ("ctrl-0", PodPhase.RUNNING)

    @pytest.mark.asyncio
    async def test_found_succeeded(self, mock_kube_client, mock_kr8s_api) -> None:
        pods = [make_kr8s_object(_make_pod_raw("ctrl-0", phase="Succeeded"))]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_controller_pod("default", "job-123")

        assert result == ("ctrl-0", PodPhase.SUCCEEDED)

    @pytest.mark.asyncio
    async def test_not_found(self, mock_kube_client, mock_kr8s_api) -> None:
        mock_kr8s_api.async_get = MagicMock(return_value=async_list([]))

        result = await mock_kube_client.find_controller_pod("default", "job-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_missing_phase_defaults_to_unknown(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        raw = {"metadata": {"name": "ctrl-0", "namespace": "default"}, "status": {}}
        pods = [make_kr8s_object(raw)]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_controller_pod("default", "job-123")

        assert result is not None
        assert result[1] == PodPhase.UNKNOWN

    @pytest.mark.asyncio
    async def test_no_status_key_defaults_to_unknown(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        raw = {"metadata": {"name": "ctrl-0", "namespace": "default"}}
        pods = [make_kr8s_object(raw)]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_controller_pod("default", "job-123")

        assert result is not None
        assert result[1] == PodPhase.UNKNOWN


# ============================================================
# find_retrievable_pod
# ============================================================


class TestFindRetrievablePod:
    """Verify phase-based filtering for file retrieval."""

    @pytest.mark.asyncio
    async def test_running_pod_retrievable(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        pods = [make_kr8s_object(_make_pod_raw("ctrl-0", phase="Running"))]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_retrievable_pod("default", "job-123")

        assert result == ("ctrl-0", PodPhase.RUNNING)

    @pytest.mark.asyncio
    async def test_succeeded_pod_retrievable(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        pods = [make_kr8s_object(_make_pod_raw("ctrl-0", phase="Succeeded"))]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_retrievable_pod("default", "job-123")

        assert result == ("ctrl-0", PodPhase.SUCCEEDED)

    @pytest.mark.asyncio
    async def test_failed_pod_not_retrievable(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        pods = [make_kr8s_object(_make_pod_raw("ctrl-0", phase="Failed"))]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_retrievable_pod("default", "job-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_pending_pod_not_retrievable(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        pods = [make_kr8s_object(_make_pod_raw("ctrl-0", phase="Pending"))]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_retrievable_pod("default", "job-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_require_running_rejects_succeeded(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        pods = [make_kr8s_object(_make_pod_raw("ctrl-0", phase="Succeeded"))]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_retrievable_pod(
            "default", "job-123", require_running=True
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_require_running_accepts_running(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        pods = [make_kr8s_object(_make_pod_raw("ctrl-0", phase="Running"))]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_retrievable_pod(
            "default", "job-123", require_running=True
        )

        assert result == ("ctrl-0", PodPhase.RUNNING)

    @pytest.mark.asyncio
    async def test_no_pod_returns_none(self, mock_kube_client, mock_kr8s_api) -> None:
        mock_kr8s_api.async_get = MagicMock(return_value=async_list([]))

        result = await mock_kube_client.find_retrievable_pod("default", "job-123")

        assert result is None

    @pytest.mark.parametrize(
        "phase,require_running,expected_none",
        [
            ("Running", False, False),
            ("Succeeded", False, False),
            ("Failed", False, True),
            ("Pending", False, True),
            ("Unknown", False, True),
            ("Running", True, False),
            ("Succeeded", True, True),
            ("Failed", True, True),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_phase_matrix(
        self,
        mock_kube_client,
        mock_kr8s_api,
        phase: str,
        require_running: bool,
        expected_none: bool,
    ) -> None:
        pods = [make_kr8s_object(_make_pod_raw("ctrl-0", phase=phase))]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_retrievable_pod(
            "default", "job-123", require_running=require_running
        )

        if expected_none:
            assert result is None
        else:
            assert result is not None


# ============================================================
# wait_for_controller_pod_ready
# ============================================================


class TestWaitForControllerPodReady:
    """Verify polling and timeout for controller pod readiness.

    Uses the time_traveler fixture so asyncio.sleep and loop.time() advance
    virtually without real delays.
    """

    @pytest.mark.asyncio
    async def test_immediately_ready(self, mock_kube_client, mock_kr8s_api) -> None:
        pods = [make_kr8s_object(_make_pod_raw("ctrl-0", phase="Running"))]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.wait_for_controller_pod_ready(
            "default", "job-123", timeout=10
        )

        assert result == "ctrl-0"

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_ready_after_polling(
        self, mock_kube_client, mock_kr8s_api, time_traveler_no_patch_sleep
    ) -> None:
        """Pod is Pending first, then becomes Running on second poll."""
        pending_pod = make_kr8s_object(_make_pod_raw("ctrl-0", phase="Pending"))
        running_pod = make_kr8s_object(_make_pod_raw("ctrl-0", phase="Running"))

        mock_kr8s_api.async_get = MagicMock(
            side_effect=[async_list([pending_pod]), async_list([running_pod])]
        )

        result = await mock_kube_client.wait_for_controller_pod_ready(
            "default", "job-123", timeout=300
        )

        assert result == "ctrl-0"

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_timeout_raises(
        self, mock_kube_client, mock_kr8s_api, time_traveler_no_patch_sleep
    ) -> None:
        """Raises TimeoutError when pod never becomes Running."""
        pending_raw = _make_pod_raw("ctrl-0", phase="Pending")
        mock_kr8s_api.async_get = MagicMock(
            side_effect=lambda *a, **kw: async_list([make_kr8s_object(pending_raw)])
        )

        with pytest.raises(TimeoutError, match="Controller pod not ready after 5s"):
            await mock_kube_client.wait_for_controller_pod_ready(
                "default", "job-123", timeout=5
            )

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_timeout_no_pod_found(
        self, mock_kube_client, mock_kr8s_api, time_traveler_no_patch_sleep
    ) -> None:
        """Raises TimeoutError when no pod exists at all."""
        mock_kr8s_api.async_get = MagicMock(side_effect=lambda *a, **kw: async_list([]))

        with pytest.raises(TimeoutError, match="Controller pod not ready"):
            await mock_kube_client.wait_for_controller_pod_ready(
                "default", "job-123", timeout=5
            )

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_timeout_message_includes_namespace(
        self, mock_kube_client, mock_kr8s_api, time_traveler_no_patch_sleep
    ) -> None:
        mock_kr8s_api.async_get = MagicMock(side_effect=lambda *a, **kw: async_list([]))

        with pytest.raises(TimeoutError, match="kubectl get pods -n my-ns"):
            await mock_kube_client.wait_for_controller_pod_ready(
                "my-ns", "job-123", timeout=5
            )


# ============================================================
# get_pods
# ============================================================


class TestGetPods:
    """Verify raw pod listing."""

    @pytest.mark.asyncio
    async def test_returns_pods(self, mock_kube_client, mock_kr8s_api) -> None:
        pods = [
            make_kr8s_object(_make_pod_raw("pod-0")),
            make_kr8s_object(_make_pod_raw("pod-1")),
        ]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.get_pods("default", "app=aiperf")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_result(self, mock_kube_client, mock_kr8s_api) -> None:
        mock_kr8s_api.async_get = MagicMock(return_value=async_list([]))

        result = await mock_kube_client.get_pods("default", "app=aiperf")

        assert result == []


# ============================================================
# delete_namespace
# ============================================================


class TestDeleteNamespace:
    """Verify namespace deletion behavior."""

    @pytest.mark.asyncio
    async def test_success(self, mock_kube_client, capsys) -> None:
        mock_ns = AsyncMock()
        with patch(
            "kr8s.asyncio.objects.Namespace.get",
            new_callable=AsyncMock,
            return_value=mock_ns,
        ):
            await mock_kube_client.delete_namespace("test-ns")

        mock_ns.delete.assert_awaited_once()
        captured = capsys.readouterr()
        assert "Deleted Namespace/test-ns" in captured.out

    @pytest.mark.asyncio
    async def test_not_found(self, mock_kube_client, capsys) -> None:
        with patch(
            "kr8s.asyncio.objects.Namespace.get",
            new_callable=AsyncMock,
            side_effect=kr8s.NotFoundError("not found"),
        ):
            await mock_kube_client.delete_namespace("test-ns")

        captured = capsys.readouterr()
        assert (
            "not found" in captured.out.lower() or "already be deleted" in captured.out
        )

    @pytest.mark.asyncio
    async def test_server_error(self, mock_kube_client, capsys) -> None:
        with patch(
            "kr8s.asyncio.objects.Namespace.get",
            new_callable=AsyncMock,
            side_effect=kr8s.ServerError("server error"),
        ):
            await mock_kube_client.delete_namespace("test-ns")

        captured = capsys.readouterr()
        assert "Failed to delete namespace" in captured.out


# ============================================================
# version
# ============================================================


class TestVersion:
    """Verify cluster version info retrieval."""

    @pytest.mark.asyncio
    async def test_returns_version_dict(self, mock_kube_client, mock_kr8s_api) -> None:
        mock_kr8s_api.async_version = AsyncMock(
            return_value={"major": "1", "minor": "28"}
        )

        result = await mock_kube_client.version()

        assert result == {"major": "1", "minor": "28"}
        mock_kr8s_api.async_version.assert_awaited_once()


# ============================================================
# AIPerfKubeClient.api property
# ============================================================


class TestApiProperty:
    """Verify the api property exposes the underlying kr8s client."""

    def test_api_returns_wrapped_client(self, mock_kube_client, mock_kr8s_api) -> None:
        assert mock_kube_client.api is mock_kr8s_api


# ============================================================
# Helpers for AIPerfJob tests
# ============================================================


def _make_raw_aiperfjob(
    name: str = "test-job",
    namespace: str = "default",
    phase: str = "Running",
    job_id: str = "test-123",
    created: str = "2026-01-15T10:30:00Z",
) -> dict[str, Any]:
    """Build a minimal raw AIPerfJob dict for testing."""
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


# ============================================================
# list_jobs
# ============================================================


class TestListJobs:
    """Verify AIPerfJob CR listing, filtering, and sorting."""

    @pytest.mark.asyncio
    async def test_list_jobs_returns_aiperfjob_info_list(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        raw = _make_raw_aiperfjob(name="job-a")
        mock_kr8s_api.async_get = MagicMock(
            return_value=async_list([make_kr8s_object(raw)])
        )

        result = await mock_kube_client.list_jobs(namespace="default")

        assert len(result) == 1
        assert result[0].name == "job-a"
        assert result[0].phase == "Running"

    @pytest.mark.asyncio
    async def test_list_jobs_all_namespaces(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        jobs = [
            make_kr8s_object(_make_raw_aiperfjob(name="job-ns1", namespace="ns1")),
            make_kr8s_object(_make_raw_aiperfjob(name="job-ns2", namespace="ns2")),
        ]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(jobs))

        result = await mock_kube_client.list_jobs(all_namespaces=True)

        assert len(result) == 2
        namespaces = {r.namespace for r in result}
        assert namespaces == {"ns1", "ns2"}

    @pytest.mark.asyncio
    async def test_list_jobs_status_filter(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        jobs = [
            make_kr8s_object(_make_raw_aiperfjob(name="running-job", phase="Running")),
            make_kr8s_object(
                _make_raw_aiperfjob(name="completed-job", phase="Completed")
            ),
            make_kr8s_object(_make_raw_aiperfjob(name="failed-job", phase="Failed")),
        ]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(jobs))

        result = await mock_kube_client.list_jobs(
            namespace="default", status_filter="Running"
        )

        assert len(result) == 1
        assert result[0].name == "running-job"

    @pytest.mark.asyncio
    async def test_list_jobs_404_returns_empty(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        mock_kr8s_api.async_get = MagicMock(
            side_effect=create_server_error(404, "Not Found")
        )

        result = await mock_kube_client.list_jobs(namespace="default")

        assert result == []

    @pytest.mark.asyncio
    async def test_list_jobs_non_404_server_error_raises(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        mock_kr8s_api.async_get = MagicMock(
            side_effect=create_server_error(500, "Internal Server Error")
        )

        with pytest.raises(kr8s.ServerError):
            await mock_kube_client.list_jobs(namespace="default")

    @pytest.mark.asyncio
    async def test_list_jobs_sorted_newest_first(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        jobs = [
            make_kr8s_object(
                _make_raw_aiperfjob(name="oldest", created="2026-01-01T00:00:00Z")
            ),
            make_kr8s_object(
                _make_raw_aiperfjob(name="newest", created="2026-01-03T00:00:00Z")
            ),
            make_kr8s_object(
                _make_raw_aiperfjob(name="middle", created="2026-01-02T00:00:00Z")
            ),
        ]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(jobs))

        result = await mock_kube_client.list_jobs(namespace="default")

        assert [r.name for r in result] == ["newest", "middle", "oldest"]


# ============================================================
# find_job
# ============================================================


class TestFindJob:
    """Verify AIPerfJob lookup by name and jobId fallback."""

    @pytest.mark.asyncio
    async def test_find_job_found_by_name(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        raw = _make_raw_aiperfjob(name="my-job", job_id="job-abc")
        mock_kr8s_api.async_get = MagicMock(
            return_value=async_list([make_kr8s_object(raw)])
        )

        result = await mock_kube_client.find_job("my-job", namespace="default")

        assert result is not None
        assert result.name == "my-job"
        assert result.job_id == "job-abc"

    @pytest.mark.asyncio
    async def test_find_job_not_found_by_name_found_by_job_id(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        target_raw = _make_raw_aiperfjob(
            name="cr-name", job_id="target-id", phase="Running"
        )
        other_raw = _make_raw_aiperfjob(
            name="other-cr", job_id="other-id", phase="Running"
        )
        mock_kr8s_api.async_get = MagicMock(
            side_effect=[
                # First call: field_selector lookup returns empty
                async_list([]),
                # Second call: full scan returns all jobs
                async_list(
                    [
                        make_kr8s_object(other_raw),
                        make_kr8s_object(target_raw),
                    ]
                ),
            ]
        )

        result = await mock_kube_client.find_job("target-id", namespace="default")

        assert result is not None
        assert result.name == "cr-name"
        assert result.job_id == "target-id"

    @pytest.mark.asyncio
    async def test_find_job_not_found_returns_none(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        mock_kr8s_api.async_get = MagicMock(
            side_effect=[
                # field_selector lookup: empty
                async_list([]),
                # full scan: empty
                async_list([]),
            ]
        )

        result = await mock_kube_client.find_job("nonexistent", namespace="default")

        assert result is None

    @pytest.mark.asyncio
    async def test_find_job_404_on_field_selector_falls_through_to_scan(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        target_raw = _make_raw_aiperfjob(name="cr-x", job_id="scan-target")
        mock_kr8s_api.async_get = MagicMock(
            side_effect=[
                # First call: field_selector gets 404
                create_server_error(404, "Not Found"),
                # Second call: full scan succeeds
                async_list([make_kr8s_object(target_raw)]),
            ]
        )

        result = await mock_kube_client.find_job("scan-target", namespace="default")

        assert result is not None
        assert result.job_id == "scan-target"

    @pytest.mark.asyncio
    async def test_find_job_namespace_filtering(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        raw = _make_raw_aiperfjob(name="ns-job", namespace="custom-ns", job_id="ns-123")
        mock_kr8s_api.async_get = MagicMock(
            return_value=async_list([make_kr8s_object(raw)])
        )

        result = await mock_kube_client.find_job("ns-job", namespace="custom-ns")

        assert result is not None
        assert result.namespace == "custom-ns"

    @pytest.mark.asyncio
    async def test_find_job_no_namespace_searches_all(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        raw = _make_raw_aiperfjob(name="any-job", namespace="some-ns")
        mock_kr8s_api.async_get = MagicMock(
            return_value=async_list([make_kr8s_object(raw)])
        )

        result = await mock_kube_client.find_job("any-job")

        assert result is not None
        assert result.name == "any-job"


# ============================================================
# find_operator_pod
# ============================================================


class TestFindOperatorPod:
    """Verify operator pod discovery and phase parsing."""

    @pytest.mark.asyncio
    async def test_found_running(self, mock_kube_client, mock_kr8s_api) -> None:
        pods = [make_kr8s_object(_make_pod_raw("aiperf-operator-0", phase="Running"))]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_operator_pod()

        assert result == ("aiperf-operator-0", PodPhase.RUNNING)

    @pytest.mark.asyncio
    async def test_not_found_returns_none(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        mock_kr8s_api.async_get = MagicMock(return_value=async_list([]))

        result = await mock_kube_client.find_operator_pod()

        assert result is None

    @pytest.mark.asyncio
    async def test_custom_namespace(self, mock_kube_client, mock_kr8s_api) -> None:
        pods = [make_kr8s_object(_make_pod_raw("op-0", phase="Running"))]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_operator_pod(namespace="custom-ns")

        assert result is not None
        mock_kr8s_api.async_get.assert_called_once()
        call_kwargs = mock_kr8s_api.async_get.call_args[1]
        assert call_kwargs["namespace"] == "custom-ns"

    @pytest.mark.asyncio
    async def test_custom_label_selector(self, mock_kube_client, mock_kr8s_api) -> None:
        pods = [make_kr8s_object(_make_pod_raw("op-0", phase="Running"))]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_operator_pod(label_selector="custom=label")

        assert result is not None
        call_kwargs = mock_kr8s_api.async_get.call_args[1]
        assert call_kwargs["label_selector"] == "custom=label"

    @pytest.mark.asyncio
    async def test_missing_phase_defaults_to_unknown(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        raw = {"metadata": {"name": "op-0", "namespace": "aiperf-system"}, "status": {}}
        pods = [make_kr8s_object(raw)]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_operator_pod()

        assert result is not None
        assert result[1] == PodPhase.UNKNOWN

    @pytest.mark.asyncio
    async def test_multiple_pods_returns_first(
        self, mock_kube_client, mock_kr8s_api
    ) -> None:
        pods = [
            make_kr8s_object(_make_pod_raw("op-0", phase="Running")),
            make_kr8s_object(_make_pod_raw("op-1", phase="Pending")),
        ]
        mock_kr8s_api.async_get = MagicMock(return_value=async_list(pods))

        result = await mock_kube_client.find_operator_pod()

        assert result is not None
        assert result[0] == "op-0"
