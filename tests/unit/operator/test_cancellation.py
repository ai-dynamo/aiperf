# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cooperative cancellation on CR delete (M4).

on_delete sets a per-job cancellation event. Long-running handler paths
(monitor_progress, handle_completion, fetch retries, JobSet delete) check
``is_cancellation_requested`` at await boundaries and short-circuit so
kopf's per-object serialization doesn't make delete wait on fetch
backoff.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.kubernetes.phase import Phase
from aiperf.operator.client_cache import (
    _reset_for_testing,
    is_cancellation_requested,
    job_key,
    request_cancellation,
)
from aiperf.operator.progress_models import JobProgress

# Minimal CR body with a deterministic creationTimestamp so
# epoch_key_from_body() resolves to a real epoch when fetch_results_with_retry
# derives the run directory. No uid: the identity fence is a no-op for bodies
# that carry none, which keeps the cancellation tests off the cluster.
_FIXTURE_BODY: dict = {"metadata": {"creationTimestamp": "2024-04-25T18:22:03Z"}}

# Completion refuses an unfenced JobSet delete, so tests that assert the delete
# actually happens need a body with the immutable uid a real callback carries.
_FIXTURE_UID = "job-uid-4c1b"
_FIXTURE_BODY_WITH_UID: dict = {
    "metadata": {
        "creationTimestamp": "2024-04-25T18:22:03Z",
        "uid": _FIXTURE_UID,
    }
}


@contextlib.contextmanager
def _stub_completion_fence(resource_version: str = "1"):
    """Answer the live-CR read completion makes before it mutates status."""
    with mock_patch(
        "aiperf.operator.handlers.completion.current_aiperfjob_resource_version",
        new=AsyncMock(return_value=resource_version),
    ) as fence:
        yield fence


@pytest.fixture(autouse=True)
def _reset_state():
    _reset_for_testing()
    yield
    _reset_for_testing()


def test_request_cancellation_sets_event_idempotently() -> None:
    """Calling request_cancellation twice is safe and leaves the event set."""
    key = "ns/job-1"
    assert not is_cancellation_requested(key)
    request_cancellation(key)
    assert is_cancellation_requested(key)
    request_cancellation(key)
    assert is_cancellation_requested(key)


def test_is_cancellation_requested_per_key_isolation() -> None:
    """Cancellation for one key does not leak to another."""
    request_cancellation("ns/job-a")
    assert is_cancellation_requested("ns/job-a")
    assert not is_cancellation_requested("ns/job-b")


@pytest.mark.asyncio
async def test_final_phase_refresh_cancellation_interrupts_client_creation() -> None:
    """Cancellation must not wait for progress-client cache contention."""
    from aiperf.operator.handlers.completion import _refresh_final_phase_progress

    creation_started = asyncio.Event()
    creation_cancelled = asyncio.Event()
    never = asyncio.Event()
    patch = MagicMock()
    patch.status = {}

    async def blocking_client_creation(_key: str) -> None:
        creation_started.set()
        try:
            await never.wait()
        finally:
            creation_cancelled.set()

    with mock_patch(
        "aiperf.operator.handlers.completion.get_or_create_progress_client",
        side_effect=blocking_client_creation,
    ):
        task = asyncio.create_task(
            _refresh_final_phase_progress(
                namespace="ns", jobset_name="js", job_id="j", patch=patch
            )
        )
        await asyncio.wait_for(creation_started.wait(), timeout=1)
        request_cancellation("ns/j")
        await asyncio.wait_for(task, timeout=1)

    assert creation_cancelled.is_set()
    assert patch.status == {}


@pytest.mark.asyncio
async def test_final_phase_refresh_cancellation_interrupts_get_progress() -> None:
    """Cancellation must interrupt the controller progress HTTP await."""
    from aiperf.operator.handlers.completion import _refresh_final_phase_progress

    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()
    never = asyncio.Event()
    progress_client = MagicMock()
    patch = MagicMock()
    patch.status = {}

    async def blocking_get_progress(_host: str) -> None:
        request_started.set()
        try:
            await never.wait()
        finally:
            request_cancelled.set()

    progress_client.get_progress = AsyncMock(side_effect=blocking_get_progress)
    with mock_patch(
        "aiperf.operator.handlers.completion.get_or_create_progress_client",
        new=AsyncMock(return_value=progress_client),
    ):
        task = asyncio.create_task(
            _refresh_final_phase_progress(
                namespace="ns", jobset_name="js", job_id="j", patch=patch
            )
        )
        await asyncio.wait_for(request_started.wait(), timeout=1)
        request_cancellation("ns/j")
        await asyncio.wait_for(task, timeout=1)

    assert request_cancelled.is_set()
    assert patch.status == {}


@pytest.mark.asyncio
async def test_final_phase_refresh_cancellation_interrupts_settle_delay() -> None:
    """Cancellation must not wait for the cosmetic phase settle delay."""
    from aiperf.operator.handlers.completion import _refresh_final_phase_progress

    settle_started = asyncio.Event()
    settle_cancelled = asyncio.Event()
    never = asyncio.Event()
    patch = MagicMock()
    patch.status = {}
    progress_client = MagicMock()
    progress_client.get_progress = AsyncMock(
        return_value=JobProgress(
            phases={
                "profiling": CombinedPhaseStats(
                    phase=CreditPhase.PROFILING,
                    phase_name="profiling",
                    phase_kind="profiling",
                    start_ns=1,
                    sent_end_ns=2,
                    requests_end_ns=3,
                    records_end_ns=None,
                    requests_sent=1,
                    requests_completed=1,
                    success_records=0,
                    total_expected_requests=1,
                )
            }
        )
    )

    async def blocking_sleep(_delay: float) -> None:
        settle_started.set()
        try:
            await never.wait()
        finally:
            settle_cancelled.set()

    with (
        mock_patch(
            "aiperf.operator.handlers.completion.get_or_create_progress_client",
            new=AsyncMock(return_value=progress_client),
        ),
        mock_patch(
            "aiperf.operator.handlers.completion.asyncio.sleep",
            new=blocking_sleep,
        ),
    ):
        task = asyncio.create_task(
            _refresh_final_phase_progress(
                namespace="ns", jobset_name="js", job_id="j", patch=patch
            )
        )
        await asyncio.wait_for(settle_started.wait(), timeout=1)
        request_cancellation("ns/j")
        await asyncio.wait_for(task, timeout=1)

    assert settle_cancelled.is_set()
    assert progress_client.get_progress.await_count == 1
    assert patch.status == {}


@pytest.mark.asyncio
async def test_on_delete_requests_cancellation_before_closing_client() -> None:
    """on_delete must signal cancellation before freeing the client so any
    concurrent handler still holding the client sees the flag."""
    from aiperf.operator.handlers.lifecycle import on_delete

    call_order: list[str] = []

    def observer(key: str) -> None:
        call_order.append(f"cancel:{key}")

    async def fake_close(key: str) -> None:
        call_order.append(f"close:{key}")

    with (
        mock_patch(
            "aiperf.operator.handlers.lifecycle.request_cancellation",
            side_effect=observer,
        ),
        mock_patch(
            "aiperf.operator.handlers.lifecycle.close_progress_client",
            side_effect=fake_close,
        ),
    ):
        await on_delete(name="j", namespace="ns", status={"jobId": "j"})

    assert call_order == ["cancel:ns/j", "close:ns/j"]


@pytest.mark.asyncio
async def test_handle_completion_short_circuits_on_cancellation() -> None:
    """handle_completion must return early on cancellation without calling
    fetch_results_with_retry, JobSet delete, or events.completed."""
    from aiperf.operator.handlers.completion import handle_completion
    from aiperf.operator.status import StatusBuilder

    request_cancellation("ns/j")

    patch = MagicMock()
    patch.status = {}
    sb = StatusBuilder(patch, {"workers": {"total": 1}})

    with (
        mock_patch(
            "aiperf.operator.handlers.completion.fetch_results_with_retry",
            new=AsyncMock(),
        ) as mock_fetch,
        # Every cluster round-trip completion makes now goes through the shared
        # identity helpers, so that is where "no I/O happened" is observable.
        mock_patch(
            "aiperf.operator.handlers._job_identity.k8s_client",
        ) as mock_client_cm,
        mock_patch(
            "aiperf.operator.handlers.completion.events.completed"
        ) as mock_completed,
    ):
        await handle_completion(
            body={},
            namespace="ns",
            jobset_name="js",
            job_id="j",
            status={},
            sb=sb,
        )

    mock_fetch.assert_not_awaited()
    mock_client_cm.assert_not_called()
    mock_completed.assert_not_called()


@pytest.mark.asyncio
async def test_handle_completion_short_circuits_when_cancelled_after_fetch() -> None:
    """In-flight cancellation after fetch must not stamp Failed/Completed."""
    from aiperf.kubernetes.crd_models import ControllerFetchResult
    from aiperf.operator.handlers.completion import handle_completion
    from aiperf.operator.status import StatusBuilder

    patch = MagicMock()
    patch.status = {}
    sb = StatusBuilder(patch, {"workers": {"total": 1}})

    async def fake_fetch(*_args, **_kwargs):
        request_cancellation("ns/j")
        return ControllerFetchResult(
            metrics=None,
            downloaded=[],
            checkpoints=[],
            error="Cancelled by CR deletion",
        )

    with (
        mock_patch(
            "aiperf.operator.handlers.completion.fetch_results_with_retry",
            side_effect=fake_fetch,
        ) as mock_fetch,
        mock_patch(
            "aiperf.operator.handlers.completion._apply_completion_results",
            new_callable=AsyncMock,
        ) as mock_apply,
        mock_patch(
            "aiperf.operator.handlers.completion._maybe_delete_jobset_after_success",
            new_callable=AsyncMock,
        ) as mock_delete,
    ):
        await handle_completion(
            body=_FIXTURE_BODY,
            namespace="ns",
            jobset_name="js",
            job_id="j",
            status={},
            sb=sb,
        )

    mock_fetch.assert_awaited_once()
    mock_apply.assert_not_awaited()
    mock_delete.assert_not_awaited()
    assert patch.status.get("phase") is None


@pytest.mark.asyncio
async def test_handle_completion_cancellation_during_jobset_delete_aborts_commit() -> (
    None
):
    """The final cluster await remains before every completion commit side effect."""
    from aiperf.kubernetes.crd_models import ControllerFetchResult
    from aiperf.operator.handlers.completion import handle_completion
    from aiperf.operator.status import StatusBuilder

    patch = MagicMock()
    patch.status = {}
    sb = StatusBuilder(patch, {"workers": {"total": 1}})
    result = ControllerFetchResult(
        metrics={"metrics": {"request_count": {"avg": 1}}},
        downloaded=["profile_export_aiperf.json"],
    )
    call_order: list[str] = []

    async def cancel_during_delete(*_args: object, **_kwargs: object) -> bool:
        call_order.append("delete")
        # Cancellation keys are uid-scoped so a delayed callback cannot cancel a
        # same-name replacement; request it under the key completion will read.
        request_cancellation(job_key("ns", "j", _FIXTURE_UID))
        # Report a successful delete: the abort under test is the cancellation
        # gate that follows it, not a failed delete.
        return True

    async def apply_results(**_kwargs: object) -> None:
        call_order.append("apply")

    with (
        mock_patch(
            "aiperf.operator.handlers.completion._refresh_final_phase_progress",
            new_callable=AsyncMock,
        ),
        mock_patch(
            "aiperf.operator.handlers.completion._key_files_materialized",
            return_value=True,
        ),
        _stub_completion_fence(),
        mock_patch(
            "aiperf.operator.handlers.completion._delete_backing_jobset",
            side_effect=cancel_during_delete,
        ) as delete_jobset,
        mock_patch(
            "aiperf.operator.handlers.completion._apply_completion_results",
            side_effect=apply_results,
        ) as apply_completion,
        mock_patch(
            "aiperf.operator.handlers.completion.events.results_stored"
        ) as results_stored,
        mock_patch("aiperf.operator.handlers.completion.events.completed") as completed,
    ):
        await handle_completion(
            body=_FIXTURE_BODY_WITH_UID,
            namespace="ns",
            jobset_name="js",
            job_id="j",
            status={},
            sb=sb,
            result=result,
        )

    assert call_order == ["delete"]
    delete_jobset.assert_awaited_once_with(
        "ns", "js", parent_name="j", parent_uid=_FIXTURE_UID
    )
    apply_completion.assert_not_awaited()
    results_stored.assert_not_called()
    completed.assert_not_called()
    assert patch.status == {}


@pytest.mark.asyncio
async def test_handle_completion_success_deletes_before_committing_status() -> None:
    """A normal success still commits exactly once after JobSet deletion."""
    from aiperf.kubernetes.crd_models import ControllerFetchResult
    from aiperf.operator.handlers.completion import handle_completion
    from aiperf.operator.status import ConditionType, StatusBuilder

    patch = MagicMock()
    patch.status = {}
    sb = StatusBuilder(patch, {"workers": {"total": 1}})
    result = ControllerFetchResult(
        metrics={"metrics": {"request_count": {"avg": 1}}},
        downloaded=["profile_export_aiperf.json"],
    )
    call_order: list[str] = []

    async def delete_jobset(*_args: object, **_kwargs: object) -> bool:
        call_order.append("delete")
        # _maybe_delete_jobset_after_success now propagates this result, and a
        # falsy one means "the JobSet was not mine" -- completion abandons the
        # commit rather than publishing terminal status.
        return True

    async def apply_results(
        *,
        sb: StatusBuilder,
        flags: object,
        **_kwargs: object,
    ) -> tuple[object, tuple[object, ...]]:
        call_order.append("apply")
        sb.set_phase(Phase.COMPLETED)
        sb.conditions.set_true(
            ConditionType.RESULTS_AVAILABLE,
            "ResultsStored",
            "Results stored",
        )
        return flags, ()

    with (
        mock_patch(
            "aiperf.operator.handlers.completion._refresh_final_phase_progress",
            new_callable=AsyncMock,
        ),
        mock_patch(
            "aiperf.operator.handlers.completion._key_files_materialized",
            return_value=True,
        ),
        _stub_completion_fence(),
        mock_patch(
            "aiperf.operator.handlers.completion._delete_backing_jobset",
            side_effect=delete_jobset,
        ) as delete_backing_jobset,
        mock_patch(
            "aiperf.operator.handlers.completion._apply_completion_results",
            side_effect=apply_results,
        ) as apply_completion,
        mock_patch(
            "aiperf.operator.handlers.completion._verify_final_artifact_publication",
            new=AsyncMock(side_effect=lambda *, flags, **_kwargs: (flags, True)),
        ),
        mock_patch("aiperf.operator.handlers.completion.events.completed") as completed,
    ):
        await handle_completion(
            body=_FIXTURE_BODY_WITH_UID,
            namespace="ns",
            jobset_name="js",
            job_id="j",
            status={},
            sb=sb,
            result=result,
        )

    assert call_order == ["delete", "apply"]
    delete_backing_jobset.assert_awaited_once_with(
        "ns", "js", parent_name="j", parent_uid=_FIXTURE_UID
    )
    apply_completion.assert_awaited_once()
    completed.assert_called_once()
    assert patch.status["phase"] == Phase.COMPLETED
    conditions = {
        condition["type"]: condition for condition in patch.status["conditions"]
    }
    assert conditions[ConditionType.COMPLETE.value]["status"] == "True"


@pytest.mark.asyncio
async def test_handle_completion_leaves_no_terminal_patch_when_cancelled_during_index_update() -> (
    None
):
    """Cancellation during the index await must not leave a terminal status patch."""
    from aiperf.kubernetes.crd_models import ControllerFetchResult
    from aiperf.operator.handlers.completion import handle_completion
    from aiperf.operator.status import StatusBuilder

    patch = MagicMock()
    patch.status = {}
    sb = StatusBuilder(patch, {"workers": {"total": 1}})

    async def fake_update_index(*_args, **_kwargs) -> None:
        await asyncio.sleep(0)
        request_cancellation("ns/j")

    result = ControllerFetchResult(
        metrics=None,
        downloaded=[],
        checkpoints=[],
        error="controller result fetch failed",
    )

    with (
        mock_patch(
            "aiperf.operator.handlers.completion._update_job_index_safe",
            side_effect=fake_update_index,
        ) as mock_update_index,
        mock_patch(
            "aiperf.operator.handlers.completion._maybe_delete_jobset_after_success",
            new_callable=AsyncMock,
        ) as mock_delete,
    ):
        await handle_completion(
            body=_FIXTURE_BODY,
            namespace="ns",
            jobset_name="js",
            job_id="j",
            status={},
            sb=sb,
            result=result,
        )

    mock_update_index.assert_awaited_once()
    mock_delete.assert_awaited_once()
    assert "phase" not in patch.status
    assert "currentPhase" not in patch.status
    assert "subPhase" not in patch.status
    assert "completionTime" not in patch.status
    assert "conditions" not in patch.status


@pytest.mark.asyncio
async def test_apply_completion_stops_before_status_write_when_cancelled_during_manifest_read() -> (
    None
):
    """The off-thread manifest read must remain a cancellation boundary."""
    from aiperf.kubernetes.crd_models import ControllerFetchResult
    from aiperf.operator.handlers import completion

    async def cancel_during_manifest_read(*_args: object) -> None:
        request_cancellation("ns/j")

    result = ControllerFetchResult(
        metrics={"metrics": {"request_count": {"avg": 1}}},
        downloaded=["profile_export_aiperf.json"],
        checkpoints=[],
        error="",
    )
    flags = completion._ResultFlags(
        has_metrics=True,
        has_files=True,
        has_error=False,
        success=True,
    )

    with (
        mock_patch(
            "aiperf.operator.handlers.completion.asyncio.to_thread",
            new=AsyncMock(side_effect=cancel_during_manifest_read),
        ),
        mock_patch(
            "aiperf.operator.handlers.completion._demote_missing_publication_artifacts",
            side_effect=lambda *, flags, **_kwargs: flags,
        ),
        mock_patch(
            "aiperf.operator.handlers.completion._record_results_on_status"
        ) as record_results,
        mock_patch(
            "aiperf.operator.handlers.completion._run_retention_pass",
            new_callable=AsyncMock,
        ) as retention,
        mock_patch(
            "aiperf.operator.handlers.completion._update_job_index_safe",
            new_callable=AsyncMock,
        ) as update_index,
    ):
        await completion._apply_completion_results(
            body=_FIXTURE_BODY,
            namespace="ns",
            jobset_name="js",
            job_id="j",
            result=result,
            sb=MagicMock(),
            status={},
            flags=flags,
        )

    record_results.assert_not_called()
    retention.assert_not_awaited()
    update_index.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_completion_stops_before_upsert_when_cancelled_during_retention() -> (
    None
):
    """A delete arriving during the retention await prevents a later upsert."""
    from aiperf.kubernetes.crd_models import ControllerFetchResult
    from aiperf.operator.handlers import completion
    from aiperf.operator.status import StatusBuilder

    staged_patch = MagicMock()
    staged_patch.status = {}
    sb = StatusBuilder(staged_patch, {"workers": {"total": 1}})
    result = ControllerFetchResult(
        metrics={"metrics": {"request_count": {"avg": 1}}},
        downloaded=["profile_export_aiperf.json"],
        checkpoints=[],
        error="",
    )
    flags = completion._ResultFlags(
        has_metrics=True,
        has_files=True,
        has_error=False,
        success=True,
    )

    async def cancel_during_retention(*_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(0)
        request_cancellation("ns/j")

    with (
        mock_patch("aiperf.operator.handlers.completion._record_results_on_status"),
        mock_patch(
            "aiperf.operator.handlers.completion._demote_missing_publication_artifacts",
            side_effect=lambda *, flags, **_kwargs: flags,
        ),
        mock_patch(
            "aiperf.operator.handlers.completion._key_files_materialized",
            return_value=True,
        ),
        mock_patch(
            "aiperf.operator.handlers.completion._capture_publication_artifacts",
            side_effect=lambda *, flags, **_kwargs: (flags, True, ()),
        ),
        mock_patch(
            "aiperf.operator.handlers.completion._run_retention_pass",
            side_effect=cancel_during_retention,
        ) as retention,
        mock_patch(
            "aiperf.operator.handlers.completion._update_job_index_safe",
            new_callable=AsyncMock,
        ) as update_index,
    ):
        await completion._apply_completion_results(
            body=_FIXTURE_BODY,
            namespace="ns",
            jobset_name="js",
            job_id="j",
            result=result,
            sb=sb,
            status={},
            flags=flags,
        )

    retention.assert_awaited_once()
    update_index.assert_not_awaited()


@pytest.mark.asyncio
async def test_index_upsert_racing_delete_compensates_after_late_write() -> None:
    """A late upsert cannot recreate a row after delete cleanup has finished."""
    from aiperf.operator.handlers import completion
    from aiperf.operator.status import StatusBuilder

    row_exists = False
    call_order: list[str] = []

    async def delete_run(*_args: object, **_kwargs: object) -> None:
        nonlocal row_exists
        call_order.append("delete")
        row_exists = False

    async def late_upsert(*_args: object, **_kwargs: object) -> None:
        nonlocal row_exists
        call_order.append("upsert-start")
        request_cancellation("ns/j")
        await delete_run("ns", "j", "1714069323")
        call_order.append("upsert-landed")
        row_exists = True

    fake_index = MagicMock()
    fake_index.upsert_run_completed = AsyncMock(side_effect=late_upsert)
    fake_index.upsert_run_failed = AsyncMock()
    fake_index.set_latest = AsyncMock()
    fake_index.delete_run = AsyncMock(side_effect=delete_run)

    patch = MagicMock()
    patch.status = {}
    sb = StatusBuilder(patch, {})

    with (
        mock_patch("aiperf.operator.handlers.completion.runs_index", fake_index),
        mock_patch(
            "aiperf.operator.handlers.completion._key_files_materialized",
            return_value=False,
        ),
    ):
        await completion._update_job_index_safe(
            namespace="ns",
            job_id="j",
            epoch="1714069323",
            body=_FIXTURE_BODY,
            sb=sb,
            phase="Succeeded",
            summary_blob=b"summary",
            metrics={},
            downloaded_files=[],
            error=None,
            mtime_epoch=0,
            end_time=None,
            total_size_bytes=0,
        )

    assert call_order == ["upsert-start", "delete", "upsert-landed", "delete"]
    assert row_exists is False
    fake_index.delete_run.assert_awaited_once_with("ns", "j", "1714069323")
    fake_index.set_latest.assert_not_awaited()
    assert "conditions" not in patch.status


@pytest.mark.asyncio
async def test_monitor_progress_short_circuits_on_cancellation() -> None:
    """monitor_progress must return early on cancellation without fetching
    the JobSet (the CR is about to disappear)."""
    from aiperf.operator.main import monitor_progress

    request_cancellation("ns/job-123")

    patch = MagicMock()
    patch.status = {}

    with mock_patch(
        "aiperf.operator.handlers.monitor.k8s_client",
    ) as mock_client_cm:
        await monitor_progress(
            body={},
            status={
                "phase": Phase.RUNNING,
                "jobSetName": "jobset",
                "jobId": "job-123",
            },
            spec={},
            name="j",
            namespace="ns",
            patch=patch,
        )

    # k8s_client must not have been entered (cancellation short-circuits before)
    mock_client_cm.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_results_returns_cancellation_error_when_flag_set() -> None:
    """When cancellation is requested, _fetch_once returns a FetchResult
    with error set; retry_with_backoff stops (returning is not an error)."""
    from aiperf.operator.handlers.completion import fetch_results_with_retry

    request_cancellation("ns/j")

    mock_client = MagicMock()
    mock_client.get_metrics = AsyncMock(
        return_value={"metrics": "SHOULD_NOT_BE_CALLED"}
    )
    mock_client.download_all_results = AsyncMock(return_value=[])

    with mock_patch(
        "aiperf.operator.handlers.completion.get_or_create_progress_client",
        new=AsyncMock(return_value=mock_client),
    ):
        result = await fetch_results_with_retry(
            controller_host="host",
            namespace="ns",
            job_id="j",
            body=_FIXTURE_BODY,
        )

    assert result.error == "Cancelled by CR deletion"
    mock_client.get_metrics.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_persists_after_close_progress_client() -> None:
    """close_progress_client must NOT clear the cancellation event: observers
    may still need to see the flag after the client is freed (e.g. the
    fetch-retry loop yielding across the close). The event is only cleared
    by _reset_for_testing or process exit."""
    from aiperf.operator.client_cache import close_progress_client

    key = "ns/j"
    request_cancellation(key)
    assert is_cancellation_requested(key)

    await close_progress_client(key)

    assert is_cancellation_requested(key), (
        "Cancellation flag must survive close_progress_client so observers "
        "yielding across the close still see the request."
    )


@pytest.mark.asyncio
async def test_delete_unblocks_concurrent_fetch_loop() -> None:
    """End-to-end: a fetch loop that would otherwise retry for tens of
    seconds exits promptly once on_delete fires."""
    from aiperf.operator.handlers.completion import fetch_results_with_retry
    from aiperf.operator.handlers.lifecycle import on_delete

    # Client returns empty downloads forever so retry loop keeps going
    # until cancellation flips.
    mock_client = MagicMock()
    mock_client.get_metrics = AsyncMock(return_value=None)
    mock_client.download_all_results = AsyncMock(return_value=[])

    with (
        mock_patch(
            "aiperf.operator.handlers.completion.get_or_create_progress_client",
            new=AsyncMock(return_value=mock_client),
        ),
        mock_patch(
            "aiperf.operator.handlers.lifecycle.close_progress_client",
            new_callable=AsyncMock,
        ),
    ):
        fetch_task = asyncio.create_task(
            fetch_results_with_retry(
                controller_host="host",
                namespace="ns",
                job_id="j",
                max_retries=50,
                retry_delay=10.0,
                body=_FIXTURE_BODY,
            )
        )

        # Let the fetch loop get going, then fire on_delete.
        for _ in range(10):
            await asyncio.sleep(0)
        await on_delete(name="j", namespace="ns", status={"jobId": "j"})

        result = await asyncio.wait_for(fetch_task, timeout=2.0)

    assert result.error and "Cancel" in result.error
