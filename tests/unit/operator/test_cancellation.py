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
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import pytest

from aiperf.operator.client_cache import (
    _reset_for_testing,
    is_cancellation_requested,
    request_cancellation,
)
from aiperf.operator.status import Phase


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
        mock_patch(
            "aiperf.operator.handlers.completion.k8s_client",
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
            )
        )

        # Let the fetch loop get going, then fire on_delete.
        for _ in range(10):
            await asyncio.sleep(0)
        await on_delete(name="j", namespace="ns", status={"jobId": "j"})

        result = await asyncio.wait_for(fetch_task, timeout=2.0)

    assert result.error and "Cancel" in result.error
