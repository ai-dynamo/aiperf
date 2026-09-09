# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Progress-client lifetime tests for the results-fetch retry loop.

The client cache is bounded, so a long fetch loop can have its cached
``ProgressClient`` evicted by another job's monitor tick. An evicted client
drops its aiohttp session and raises ``RuntimeError`` on every later call, so
the fetch loop must re-resolve its client per attempt rather than holding one
reference for the lifetime of the run.

Out of scope: the cache's own eviction ordering, covered by
``tests/unit/operator/test_client_cache.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest

from aiperf.kubernetes.crd_models import ControllerFetchResult
from aiperf.operator.client_cache import _reset_for_testing
from aiperf.operator.handlers import _completion_fetch as completion_fetch


def _body() -> dict[str, Any]:
    return {
        "metadata": {
            "name": "bench",
            "namespace": "ns",
            "uid": "job-uid",
            "creationTimestamp": "2026-05-17T00:00:00Z",
        },
        "spec": {},
    }


@pytest.fixture(autouse=True)
def _reset_client_cache() -> Any:
    _reset_for_testing()
    yield
    _reset_for_testing()


@pytest.mark.asyncio
async def test_fetch_results_with_retry_evicted_client_reresolves_and_completes(
    tmp_path: Path,
) -> None:
    """An evicted client must not poison every remaining fetch attempt.

    Holding one client for the whole loop meant a mid-run eviction made every
    later attempt raise the same RuntimeError, so the run burned through its
    stagnation limit and was stamped failed with its results sitting ready.
    """
    evicted = MagicMock(name="evicted-client")
    live = MagicMock(name="live-client")
    handed_out = [evicted, live]
    used: list[MagicMock] = []
    expected = ControllerFetchResult(
        metrics={"metrics": {"request_throughput": {"avg": 12.5}}},
        downloaded=["profile_export_aiperf.json"],
    )

    released: list[str] = []

    async def _acquire_client(_key: str) -> MagicMock:
        return handed_out.pop(0) if handed_out else live

    async def _release_client(key: str) -> None:
        released.append(key)

    async def _fetch_once_into_state(
        *, progress_client: MagicMock, state: dict[str, Any], **_kwargs: Any
    ) -> ControllerFetchResult:
        used.append(progress_client)
        if progress_client is evicted:
            raise RuntimeError(
                "ProgressClient.get_metrics() called outside async context; "
                "wrap in 'async with ProgressClient(...) as pc:'"
            )
        state["metrics"] = expected.metrics
        state["downloaded"] = expected.downloaded
        return expected

    with (
        mock_patch.object(
            completion_fetch, "acquire_progress_client", new=_acquire_client
        ),
        mock_patch.object(
            completion_fetch, "release_progress_client", new=_release_client
        ),
        mock_patch.object(
            completion_fetch, "_fetch_once_into_state", new=_fetch_once_into_state
        ),
    ):
        fetched = await completion_fetch.fetch_results_with_retry(
            "controller.ns.svc",
            "ns",
            "bench",
            dest_dir=tmp_path,
            body=_body(),
            max_retries=3,
            retry_delay=0.0,
        )

    assert used == [evicted, live]
    assert released == ["ns/bench@job-uid", "ns/bench@job-uid"]
    assert not fetched.error
    assert fetched.downloaded == ["profile_export_aiperf.json"]
