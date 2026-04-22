# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.operator.handlers.completion import handle_completion
from aiperf.operator.models import FetchResult
from aiperf.operator.status import ConditionType, Phase, StatusBuilder


@pytest.mark.asyncio
async def test_handle_completion_without_result_files_marks_failed() -> None:
    patch_obj = MagicMock()
    patch_obj.status = {}
    sb = StatusBuilder(patch_obj, existing_status={"workers": {"total": 90}})

    result = FetchResult(
        metrics={
            "aiperf_version": "0.6.0",
            "benchmark_id": "bench-1",
            "model": "mock",
            "endpoint_type": "chat",
            "streaming": True,
            "concurrency": 450000,
            "request_rate": None,
            "metrics": {},
        },
        downloaded=[],
        error="controller terminated before results were recoverable",
    )

    with (
        patch("aiperf.operator.handlers.completion.events.results_failed"),
        patch("aiperf.operator.handlers.completion.events.completed"),
        patch(
            "aiperf.operator.handlers.completion.index_job_completed", new=AsyncMock()
        ),
    ):
        await handle_completion(
            body={},
            namespace="test-ns",
            jobset_name="test-jobset",
            job_id="test-job",
            status={"workers": {"total": 90}, "startTime": "2026-03-26T00:00:00Z"},
            sb=sb,
            result=result,
        )

    assert patch_obj.status["phase"] == Phase.FAILED
    assert patch_obj.status["conditions"][-1]["type"] == "ResultsAvailable"
    assert patch_obj.status["conditions"][-1]["status"] == "False"


@pytest.mark.asyncio
async def test_handle_completion_has_files_with_error_marks_failed() -> None:
    """M3: FetchResult with has_files=True AND error set should NOT be Completed.

    A partial fetch can download some files (e.g. checkpoints) while still
    missing the key export artifacts. The error field is authoritative.
    """
    patch_obj = MagicMock()
    patch_obj.status = {}
    sb = StatusBuilder(patch_obj, existing_status={"workers": {"total": 1}})

    result = FetchResult(
        metrics=None,
        downloaded=["profile_export_aiperf.json", "checkpoints/aggregator-0.parquet"],
        error="key artifact write interrupted; retry needed",
    )

    results_failed_mock = MagicMock()
    completed_mock = MagicMock()
    with (
        patch(
            "aiperf.operator.handlers.completion.events.results_failed",
            results_failed_mock,
        ),
        patch(
            "aiperf.operator.handlers.completion.events.completed",
            completed_mock,
        ),
        patch(
            "aiperf.operator.handlers.completion.events.results_stored",
        ),
        patch(
            "aiperf.operator.handlers.completion.index_job_completed", new=AsyncMock()
        ),
    ):
        await handle_completion(
            body={},
            namespace="ns",
            jobset_name="js",
            job_id="j1",
            status={"workers": {"total": 1}, "startTime": "2026-03-26T00:00:00Z"},
            sb=sb,
            result=result,
        )

    assert patch_obj.status["phase"] == Phase.FAILED
    # completed event must NOT fire on partial/errored result
    completed_mock.assert_not_called()
    # results_failed event must fire with the authoritative error message
    results_failed_mock.assert_called_once()
    _, kwargs = results_failed_mock.call_args
    args = results_failed_mock.call_args.args
    assert "key artifact write interrupted" in args[1]
    ra = next(
        c for c in patch_obj.status["conditions"] if c["type"] == "ResultsAvailable"
    )
    assert ra["status"] == "False"


@pytest.mark.asyncio
async def test_handle_completion_index_failure_sets_condition_and_event() -> None:
    """M1: index_job_completed failure should set INDEX_UPDATED=False and warn.

    Results are already on disk, so we must not retry the completion handler;
    instead surface the failure via a condition + Warning event.
    """
    patch_obj = MagicMock()
    patch_obj.status = {}
    sb = StatusBuilder(patch_obj, existing_status={"workers": {"total": 1}})

    result = FetchResult(
        metrics={"metrics": {"latency": 1.0}},
        downloaded=["profile_export_aiperf.json"],
        error="",
    )

    kopf_event_mock = MagicMock()
    with (
        patch("aiperf.operator.handlers.completion.events.results_stored"),
        patch("aiperf.operator.handlers.completion.events.completed"),
        patch(
            "aiperf.operator.handlers.completion.index_job_completed",
            new=AsyncMock(side_effect=RuntimeError("disk full")),
        ),
        patch(
            "aiperf.operator.handlers.completion.kopf.event",
            kopf_event_mock,
        ),
        patch(
            "aiperf.operator.handlers.completion.k8s_client",
        ),
        patch(
            "aiperf.operator.handlers.completion.client.CustomObjectsApi",
            return_value=MagicMock(
                delete_namespaced_custom_object=AsyncMock(return_value={})
            ),
        ),
    ):
        await handle_completion(
            body={"metadata": {"name": "j1", "namespace": "ns"}},
            namespace="ns",
            jobset_name="js",
            job_id="j1",
            status={"workers": {"total": 1}, "startTime": "2026-03-26T00:00:00Z"},
            sb=sb,
            result=result,
        )

    # Phase still reflects the actual result (Completed) — index failure
    # doesn't flip the job to FAILED, results are on disk.
    assert patch_obj.status["phase"] == Phase.COMPLETED
    # INDEX_UPDATED condition was set to False
    index_cond = next(
        (c for c in patch_obj.status["conditions"] if c["type"] == "IndexUpdated"),
        None,
    )
    assert index_cond is not None
    assert index_cond["status"] == "False"
    assert index_cond["reason"] == "IndexUpdateFailed"
    # Warning event was emitted
    kopf_event_mock.assert_called_once()
    assert kopf_event_mock.call_args.kwargs["type"] == "Warning"
    assert kopf_event_mock.call_args.kwargs["reason"] == "IndexUpdateFailed"


@pytest.mark.asyncio
async def test_condition_type_index_updated_exists() -> None:
    """Sanity: ConditionType.INDEX_UPDATED enum value exists and is spelled right."""
    assert ConditionType.INDEX_UPDATED == "IndexUpdated"
