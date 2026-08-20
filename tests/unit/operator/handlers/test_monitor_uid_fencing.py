# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable-identity tests for delayed AIPerfJob monitor callbacks."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.kubernetes.cr_refs import AIPERF_JOB_API_VERSION
from aiperf.kubernetes.phase import Phase
from aiperf.operator.handlers import monitor
from aiperf.operator.handlers._job_identity import StaleAIPerfJobCallback
from aiperf.operator.status import StatusBuilder


def _body(*, uid: str = "old-job-uid") -> dict[str, Any]:
    return {
        "metadata": {
            "name": "benchmark",
            "namespace": "ns",
            "uid": uid,
            "creationTimestamp": "2026-08-04T12:00:00Z",
        }
    }


def _status_builder(
    status: dict[str, Any] | None = None,
) -> tuple[StatusBuilder, MagicMock]:
    result = MagicMock()
    result.status = {}
    result.metadata = {}
    return StatusBuilder(result, status or {}), result


@asynccontextmanager
async def _api_context() -> Any:
    yield MagicMock()


@pytest.mark.asyncio
async def test_fetch_jobset_rejects_replacement_owner_without_patch() -> None:
    """A same-name JobSet owned by a replacement parent is not interpreted."""
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={
            "metadata": {
                "name": "aiperf-benchmark",
                "uid": "replacement-jobset-uid",
                "ownerReferences": [
                    {
                        "apiVersion": AIPERF_JOB_API_VERSION,
                        "kind": "AIPerfJob",
                        "name": "benchmark",
                        "uid": "replacement-job-uid",
                        "controller": True,
                    }
                ],
            },
            "status": {"conditions": [{"type": "Failed", "status": "True"}]},
        }
    )
    sb, result = _status_builder()
    with patch.object(monitor, "close_progress_client", new=AsyncMock()) as close:
        jobset = await monitor._fetch_jobset_or_reconcile(
            custom,
            body=_body(),
            namespace="ns",
            name="benchmark",
            jobset_name="aiperf-benchmark",
            current_phase=Phase.RUNNING,
            key="ns/benchmark@old-job-uid",
            sb=sb,
        )

    assert jobset is None
    assert result.status == {}
    close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stale_monitor_callback_has_no_status_or_event_side_effects() -> None:
    """A replacement parent UID stops the tick before any JobSet or event work."""
    _, result = _status_builder()
    body = _body()
    status = {
        "phase": Phase.RUNNING,
        "jobSetName": "aiperf-benchmark",
        "jobId": "benchmark",
    }
    with (
        patch.object(
            monitor,
            "current_aiperfjob_resource_version",
            new=AsyncMock(side_effect=StaleAIPerfJobCallback("replacement parent")),
        ),
        patch.object(monitor, "k8s_client") as k8s,
        patch.object(monitor.events, "job_timeout") as timeout_event,
        patch.object(monitor.events, "failed") as failed_event,
        patch.object(monitor.events, "pod_startup_blocked") as startup_event,
    ):
        await monitor.monitor_progress(
            body,
            status,
            {},
            "benchmark",
            "ns",
            result,
        )

    assert result.status == {}
    assert result.metadata == {}
    k8s.assert_not_called()
    timeout_event.assert_not_called()
    failed_event.assert_not_called()
    startup_event.assert_not_called()


@pytest.mark.asyncio
async def test_monitor_status_patch_is_resource_version_fenced() -> None:
    """A live callback pins Kopf's eventual merge patch to its parent version."""
    _, result = _status_builder()
    with (
        patch.object(
            monitor,
            "current_aiperfjob_resource_version",
            new=AsyncMock(return_value="73"),
        ),
        patch.object(monitor, "k8s_client", return_value=_api_context()),
        patch.object(monitor, "_monitor_tick", new=AsyncMock()),
    ):
        await monitor.monitor_progress(
            _body(),
            {
                "phase": Phase.RUNNING,
                "jobSetName": "aiperf-benchmark",
                "jobId": "benchmark",
            },
            {},
            "benchmark",
            "ns",
            result,
        )

    assert result.metadata == {"resourceVersion": "73"}


@pytest.mark.asyncio
async def test_timeout_foreign_jobset_leaves_patch_and_events_empty() -> None:
    """Timeout finalization is abandoned when exact JobSet deletion is refused."""
    status = {
        "startTime": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
    }
    sb, result = _status_builder(status)
    with (
        patch.object(
            monitor, "_delete_jobset_or_retry", new=AsyncMock(return_value=False)
        ),
        patch.object(monitor.events, "job_timeout") as timeout_event,
        patch.object(monitor, "close_progress_client", new=AsyncMock()) as close,
    ):
        handled = await monitor._check_job_timeout(
            MagicMock(),
            body=_body(),
            status=status,
            spec={"timeoutSeconds": 1},
            namespace="ns",
            jobset_name="aiperf-benchmark",
            job_id="benchmark",
            key="ns/benchmark@old-job-uid",
            sb=sb,
        )

    assert handled is True
    assert result.status == {}
    timeout_event.assert_not_called()
    close.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_failure_defers_foreign_jobset_cleanup_to_claim_timer() -> None:
    """Critical observation stays diagnostic until the claim timer checks ownership."""
    issue = monitor.PodStartupIssue(
        pod_name="controller-0",
        container_name="controller",
        reason="CrashLoopBackOff",
        message="restarting",
        category="CrashLoop",
        terminal_after_threshold=True,
    )
    core = MagicMock()
    core.list_namespaced_pod = AsyncMock(return_value=SimpleNamespace(items=[]))
    sb, result = _status_builder()
    with (
        patch.object(monitor.client, "CoreV1Api", return_value=core),
        patch.object(monitor, "_get_pod_startup_issue", return_value=issue),
        patch.object(
            monitor,
            "_startup_issue_state",
            return_value=(
                {"warningEmitted": False},
                monitor.K8sEnvironment.WATCHDOG.PENDING_CRITICAL_THRESHOLD_SECONDS + 1,
            ),
        ),
        patch.object(
            monitor, "_delete_jobset_or_retry", new=AsyncMock(return_value=False)
        ) as delete_jobset,
        patch.object(monitor.events, "pod_startup_blocked") as startup_event,
        patch.object(monitor.events, "failed") as failed_event,
    ):
        handled = await monitor._reconcile_pod_startup_issue(
            MagicMock(),
            body=_body(),
            status={},
            patch=result,
            namespace="ns",
            name="benchmark",
            jobset_name="aiperf-benchmark",
            job_id="benchmark",
            key="ns/benchmark@old-job-uid",
            sb=sb,
        )

    assert handled is True
    assert "phase" not in result.status
    assert "error" not in result.status
    assert result.status["startupIssue"] == {"warningEmitted": False}
    delete_jobset.assert_not_awaited()
    startup_event.assert_not_called()
    failed_event.assert_not_called()


@pytest.mark.asyncio
async def test_checkpoint_salvage_foreign_jobset_leaves_patch_and_events_empty() -> (
    None
):
    """A stale salvage callback does not parse or publish partial artifacts."""
    sb, result = _status_builder()
    parse = MagicMock()
    with (
        patch.object(
            monitor, "_delete_jobset_or_retry", new=AsyncMock(return_value=False)
        ),
        patch.object(monitor, "_parse_metrics_from_files", parse),
        patch.object(monitor.events, "results_stored") as stored_event,
        patch.object(monitor.events, "failed") as failed_event,
    ):
        await monitor._recover_from_partial_checkpoints(
            body=_body(),
            result=SimpleNamespace(checkpoints=["checkpoint.json"]),
            namespace="ns",
            jobset_name="aiperf-benchmark",
            job_id="benchmark",
            sb=sb,
            custom=MagicMock(),
        )

    assert result.status == {}
    parse.assert_not_called()
    stored_event.assert_not_called()
    failed_event.assert_not_called()
