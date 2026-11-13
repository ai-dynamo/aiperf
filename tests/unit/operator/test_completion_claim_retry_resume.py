# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for kopf retries of ``on_benchmark_complete``.

When ``handle_completion`` raised the transient-fetch ``TemporaryError`` under
a claim this process won, the kopf retry used to lose ``try_claim_completion``
at the in-process fast path and return success, so completion never re-ran
and the CR stayed ``Running`` with a live controller. The retry must resume
completion for the claim owner only: never for a process that lost the race,
and never blindly after an operator restart.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import kopf
import pytest
from pytest import param

from aiperf.kubernetes.constants import Annotations
from aiperf.kubernetes.phase import Phase
from aiperf.operator import client_cache
from aiperf.operator.client_cache import _reset_for_testing, job_key
from aiperf.operator.handlers import lifecycle
from aiperf.operator.handlers._completion_retry import TransientCompletionFetchError

_UID = "6f1d2c3b-9a8e-4d7f-b6c5-0e1f2a3b4c5d"
_NAMESPACE = "ns"
_NAME = "j"
_KEY = job_key(_NAMESPACE, _NAME, _UID)
_STATUS = {"phase": Phase.RUNNING, "jobId": _NAME, "jobSetName": "js"}


def _body(claim: str | None = None) -> dict[str, Any]:
    annotations = {Annotations.BENCHMARK_COMPLETE: "true"}
    if claim is not None:
        annotations[Annotations.COMPLETION_CLAIMED] = claim
    return {
        "metadata": {
            "name": _NAME,
            "namespace": _NAMESPACE,
            "uid": _UID,
            "annotations": annotations,
        }
    }


def _transient() -> TransientCompletionFetchError:
    return TransientCompletionFetchError("transient fetch failure", delay=5)


async def _invoke(body: dict[str, Any]) -> None:
    patch = MagicMock()
    patch.status = {}
    await lifecycle.on_benchmark_complete(
        body=body,
        status=_STATUS,
        name=_NAME,
        namespace=_NAMESPACE,
        patch=patch,
        expected_parent_uid=_UID,
    )


def _live_claim_is_ours() -> AsyncMock:
    """Live CR annotations echo whatever claim timestamp this process latched."""

    async def _read(*_: Any, **__: Any) -> dict[str, str]:
        return {Annotations.COMPLETION_CLAIMED: client_cache._claim_timestamps[_KEY]}

    return AsyncMock(side_effect=_read)


@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    _reset_for_testing()
    yield
    _reset_for_testing()


@pytest.fixture
def progress_client() -> Iterator[AsyncMock]:
    client = AsyncMock()
    with (
        mock_patch.object(lifecycle, "current_aiperfjob_resource_version", AsyncMock()),
        mock_patch.object(
            lifecycle, "acquire_progress_client", AsyncMock(return_value=client)
        ),
        mock_patch.object(lifecycle, "release_progress_client", AsyncMock()),
        mock_patch.object(client_cache, "_post_dashboard_refresh", AsyncMock()),
    ):
        yield client


class TestOnBenchmarkCompleteRetryResume:
    @pytest.mark.asyncio
    async def test_on_benchmark_complete_retry_after_transient_error_reruns_and_shuts_down(
        self, progress_client: AsyncMock
    ) -> None:
        handle = AsyncMock(side_effect=[_transient(), None])
        with (
            mock_patch.object(
                client_cache, "_submit_claim_patch", AsyncMock(return_value=True)
            ),
            mock_patch.object(
                client_cache,
                "_read_live_completion_claim_annotations",
                _live_claim_is_ours(),
            ),
            mock_patch.object(lifecycle, "handle_completion", handle),
        ):
            with pytest.raises(kopf.TemporaryError):
                await _invoke(_body())
            progress_client.send_shutdown.assert_not_called()

            # kopf retry: the snapshot now carries the claim this process wrote.
            await _invoke(_body(claim=client_cache._claim_timestamps[_KEY]))

        assert handle.await_count == 2
        progress_client.send_shutdown.assert_awaited_once()
        assert _KEY not in client_cache._shutdown_sent
        assert _KEY not in client_cache._completion_retry_pending

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_repeated_transient_errors_keep_retrying(
        self, progress_client: AsyncMock
    ) -> None:
        handle = AsyncMock(side_effect=[_transient(), _transient(), None])
        with (
            mock_patch.object(
                client_cache, "_submit_claim_patch", AsyncMock(return_value=True)
            ),
            mock_patch.object(
                client_cache,
                "_read_live_completion_claim_annotations",
                _live_claim_is_ours(),
            ),
            mock_patch.object(lifecycle, "handle_completion", handle),
        ):
            with pytest.raises(kopf.TemporaryError):
                await _invoke(_body())
            claimed = _body(claim=client_cache._claim_timestamps[_KEY])
            with pytest.raises(kopf.TemporaryError):
                await _invoke(claimed)
            await _invoke(claimed)

        assert handle.await_count == 3
        progress_client.send_shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_lost_claim_race_never_runs_handle_completion(
        self, progress_client: AsyncMock
    ) -> None:
        handle = AsyncMock()
        live_read = AsyncMock(
            return_value={Annotations.COMPLETION_CLAIMED: "2026-01-01T00:00:00Z"}
        )
        with (
            mock_patch.object(
                client_cache, "_submit_claim_patch", AsyncMock(return_value=False)
            ),
            mock_patch.object(
                client_cache, "_read_live_completion_claim_annotations", live_read
            ),
            mock_patch.object(lifecycle, "handle_completion", handle),
        ):
            await _invoke(_body())
            await _invoke(_body(claim="2026-01-01T00:00:00Z"))

        handle.assert_not_awaited()
        progress_client.send_shutdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_after_restart_does_not_reenter(
        self, progress_client: AsyncMock
    ) -> None:
        handle = AsyncMock(side_effect=[_transient(), None])
        with (
            mock_patch.object(
                client_cache, "_submit_claim_patch", AsyncMock(return_value=True)
            ),
            mock_patch.object(
                client_cache,
                "_read_live_completion_claimed",
                AsyncMock(return_value=True),
            ),
            mock_patch.object(lifecycle, "handle_completion", handle),
        ):
            with pytest.raises(kopf.TemporaryError):
                await _invoke(_body())
            claim = client_cache._claim_timestamps[_KEY]

            _reset_for_testing()  # operator restart: in-process caches are empty
            await _invoke(_body(claim=claim))

        assert handle.await_count == 1
        progress_client.send_shutdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_non_transient_temporary_error_does_not_arm_resume(
        self, progress_client: AsyncMock
    ) -> None:
        handle = AsyncMock(side_effect=[kopf.TemporaryError("index blip", delay=5)])
        with (
            mock_patch.object(
                client_cache, "_submit_claim_patch", AsyncMock(return_value=True)
            ),
            mock_patch.object(
                client_cache,
                "_read_live_completion_claim_annotations",
                _live_claim_is_ours(),
            ),
            mock_patch.object(lifecycle, "handle_completion", handle),
        ):
            with pytest.raises(kopf.TemporaryError):
                await _invoke(_body())
            await _invoke(_body(claim=client_cache._claim_timestamps[_KEY]))

        assert handle.await_count == 1
        assert _KEY not in client_cache._completion_retry_pending

    @pytest.mark.parametrize(
        "live_annotations",
        [
            param({Annotations.COMPLETION_CLAIMED: "1999-01-01T00:00:00Z"}, id="foreign-claim"),
            param({}, id="claim-removed"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_on_benchmark_complete_live_claim_not_ours_does_not_reenter(
        self, progress_client: AsyncMock, live_annotations: dict[str, str]
    ) -> None:
        handle = AsyncMock(side_effect=[_transient(), None])
        with (
            mock_patch.object(
                client_cache, "_submit_claim_patch", AsyncMock(return_value=True)
            ),
            mock_patch.object(
                client_cache,
                "_read_live_completion_claim_annotations",
                AsyncMock(return_value=live_annotations),
            ),
            mock_patch.object(lifecycle, "handle_completion", handle),
        ):
            with pytest.raises(kopf.TemporaryError):
                await _invoke(_body())
            await _invoke(_body(claim=client_cache._claim_timestamps[_KEY]))

        assert handle.await_count == 1
        progress_client.send_shutdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_unreadable_live_claim_raises_and_keeps_mark(
        self, progress_client: AsyncMock
    ) -> None:
        handle = AsyncMock(side_effect=[_transient(), None])
        live_read = AsyncMock(return_value=None)
        with (
            mock_patch.object(
                client_cache, "_submit_claim_patch", AsyncMock(return_value=True)
            ),
            mock_patch.object(
                client_cache, "_read_live_completion_claim_annotations", live_read
            ),
            mock_patch.object(lifecycle, "handle_completion", handle),
        ):
            with pytest.raises(kopf.TemporaryError):
                await _invoke(_body())
            claimed = _body(claim=client_cache._claim_timestamps[_KEY])
            with pytest.raises(kopf.TemporaryError, match="claim ownership"):
                await _invoke(claimed)
            assert _KEY in client_cache._completion_retry_pending

            live_read.side_effect = _live_claim_is_ours().side_effect
            await _invoke(claimed)

        assert handle.await_count == 2
        progress_client.send_shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_cancellation_before_retry_skips_resume(
        self, progress_client: AsyncMock
    ) -> None:
        handle = AsyncMock(side_effect=[_transient(), None])
        with (
            mock_patch.object(
                client_cache, "_submit_claim_patch", AsyncMock(return_value=True)
            ),
            mock_patch.object(
                client_cache,
                "_read_live_completion_claim_annotations",
                _live_claim_is_ours(),
            ),
            mock_patch.object(lifecycle, "handle_completion", handle),
        ):
            with pytest.raises(kopf.TemporaryError):
                await _invoke(_body())
            client_cache.request_cancellation(_KEY)
            await _invoke(_body(claim=client_cache._claim_timestamps[_KEY]))

        assert handle.await_count == 1
