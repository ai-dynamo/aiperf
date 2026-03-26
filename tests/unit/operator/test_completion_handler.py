# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.operator.handlers.completion import handle_completion
from aiperf.operator.models import FetchResult
from aiperf.operator.status import Phase, StatusBuilder


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
