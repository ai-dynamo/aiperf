# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the durable completion-claim dedup mechanism.

The operator replaces the in-memory ``_shutdown_sent`` set with a CR-level
annotation (``Annotations.COMPLETION_CLAIMED``). These tests cover the
``try_claim_completion`` / ``is_completion_claimed`` helpers and verify
that the call-site handlers skip ``handle_completion`` when the claim
already exists.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import kr8s
import pytest

from aiperf.kubernetes.constants import Annotations
from aiperf.operator.client_cache import (
    _reset_for_testing,
    _shutdown_sent,
    is_completion_claimed,
    try_claim_completion,
)
from aiperf.operator.status import Phase


@pytest.fixture(autouse=True)
def _clean_state():
    _reset_for_testing()
    yield
    _reset_for_testing()


def _body_with_annotation() -> dict:
    return {
        "metadata": {
            "annotations": {Annotations.COMPLETION_CLAIMED: "2026-01-01T00:00:00Z"},
        },
    }


def _body_without_annotation() -> dict:
    return {"metadata": {"annotations": {}}}


class TestIsCompletionClaimed:
    def test_returns_true_when_annotation_set(self) -> None:
        assert is_completion_claimed(_body_with_annotation()) is True

    def test_returns_false_when_annotation_missing(self) -> None:
        assert is_completion_claimed(_body_without_annotation()) is False

    def test_handles_missing_annotations_dict(self) -> None:
        assert is_completion_claimed({"metadata": {}}) is False

    def test_handles_null_annotations(self) -> None:
        assert is_completion_claimed({"metadata": {"annotations": None}}) is False


class TestTryClaimCompletion:
    @pytest.mark.asyncio
    async def test_fast_path_annotation_already_set(self) -> None:
        """If annotation is already on body, skip the API call and return False."""
        body = _body_with_annotation()

        # Patch get_api to blow up if called — we should not hit it.
        with mock_patch(
            "aiperf.kubernetes.client.get_api",
            side_effect=AssertionError("should not call API"),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is False
        # Fast-path also populates the in-process cache.
        assert "ns/j" in _shutdown_sent

    @pytest.mark.asyncio
    async def test_fast_path_in_process_cache(self) -> None:
        """If key already in _shutdown_sent this process, return False without touching body/API."""
        _shutdown_sent.add("ns/j")

        with mock_patch(
            "aiperf.kubernetes.client.get_api",
            side_effect=AssertionError("should not call API"),
        ):
            result = await try_claim_completion("ns", "j", _body_without_annotation())

        assert result is False

    @pytest.mark.asyncio
    async def test_successful_claim_patches_annotation(self) -> None:
        """Slow path: mock get_api + AsyncAIPerfJob.get, verify patch called and True returned."""
        body = _body_without_annotation()
        mock_obj = AsyncMock()

        with (
            mock_patch(
                "aiperf.kubernetes.client.get_api",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            mock_patch(
                "aiperf.kubernetes.kr8s_resources.AsyncAIPerfJob.get",
                new_callable=AsyncMock,
                return_value=mock_obj,
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is True
        assert "ns/j" in _shutdown_sent
        mock_obj.patch.assert_awaited_once()
        args, kwargs = mock_obj.patch.call_args
        patch_ops = args[0]
        assert kwargs.get("type") == "json"
        assert any(op.get("op") == "test" for op in patch_ops)
        assert any(
            op.get("op") == "add" and "completion-claimed" in op.get("path", "")
            for op in patch_ops
        )

    @pytest.mark.asyncio
    async def test_successful_claim_with_null_annotations(self) -> None:
        """If metadata.annotations is missing, patch adds the dict first."""
        body = {"metadata": {}}
        mock_obj = AsyncMock()

        with (
            mock_patch(
                "aiperf.kubernetes.client.get_api",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            mock_patch(
                "aiperf.kubernetes.kr8s_resources.AsyncAIPerfJob.get",
                new_callable=AsyncMock,
                return_value=mock_obj,
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is True
        patch_ops = mock_obj.patch.call_args[0][0]
        # Three-op sequence: test null, add dict, add annotation.
        assert len(patch_ops) == 3
        assert patch_ops[0]["op"] == "test"
        assert patch_ops[0]["value"] is None
        assert patch_ops[1]["op"] == "add"
        assert patch_ops[1]["path"] == "/metadata/annotations"

    @pytest.mark.asyncio
    async def test_conflict_409_returns_false(self) -> None:
        """On 409 the patch lost a race: return False, mark in-process cache."""
        body = _body_without_annotation()
        response = MagicMock()
        response.status_code = 409
        err = kr8s.ServerError("conflict", response=response)
        mock_obj = AsyncMock()
        mock_obj.patch.side_effect = err

        with (
            mock_patch(
                "aiperf.kubernetes.client.get_api",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            mock_patch(
                "aiperf.kubernetes.kr8s_resources.AsyncAIPerfJob.get",
                new_callable=AsyncMock,
                return_value=mock_obj,
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is False
        assert "ns/j" in _shutdown_sent

    @pytest.mark.asyncio
    async def test_unprocessable_422_returns_false(self) -> None:
        """On 422 (JSON-patch test op failed) return False."""
        body = _body_without_annotation()
        response = MagicMock()
        response.status_code = 422
        err = kr8s.ServerError("test op failed", response=response)
        mock_obj = AsyncMock()
        mock_obj.patch.side_effect = err

        with (
            mock_patch(
                "aiperf.kubernetes.client.get_api",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            mock_patch(
                "aiperf.kubernetes.kr8s_resources.AsyncAIPerfJob.get",
                new_callable=AsyncMock,
                return_value=mock_obj,
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is False
        assert "ns/j" in _shutdown_sent

    @pytest.mark.asyncio
    async def test_other_server_error_returns_false_without_marking(self) -> None:
        """Non-409/422 errors return False but do not poison the in-process cache.

        We must be able to retry later; a transient 500 should not latch
        the claim off for the lifetime of the operator process.
        """
        body = _body_without_annotation()
        response = MagicMock()
        response.status_code = 500
        err = kr8s.ServerError("server error", response=response)
        mock_obj = AsyncMock()
        mock_obj.patch.side_effect = err

        with (
            mock_patch(
                "aiperf.kubernetes.client.get_api",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            mock_patch(
                "aiperf.kubernetes.kr8s_resources.AsyncAIPerfJob.get",
                new_callable=AsyncMock,
                return_value=mock_obj,
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is False
        assert "ns/j" not in _shutdown_sent

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_false(self) -> None:
        """Any unexpected error returns False fail-safe (don't double-complete)."""
        body = _body_without_annotation()

        with (
            mock_patch(
                "aiperf.kubernetes.client.get_api",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
        ):
            result = await try_claim_completion("ns", "j", body)

        assert result is False
        assert "ns/j" not in _shutdown_sent


class TestLifecycleIntegration:
    """Verify on_benchmark_complete honours the durable claim annotation."""

    @pytest.mark.asyncio
    async def test_annotation_preset_on_body_skips_handle_completion(self) -> None:
        """If body already carries the annotation, handler returns early."""
        from aiperf.operator.handlers.lifecycle import on_benchmark_complete

        patch = MagicMock()
        patch.status = {}

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new_callable=AsyncMock,
            ) as mock_handle,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.get_or_create_progress_client",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                new_callable=AsyncMock,
            ),
        ):
            await on_benchmark_complete(
                body=_body_with_annotation(),
                status={"phase": Phase.RUNNING, "jobId": "j", "jobSetName": "js"},
                name="j",
                namespace="ns",
                patch=patch,
            )

        mock_handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_claim_failure_skips_handle_completion(self) -> None:
        """If try_claim_completion returns False, handle_completion is not called."""
        from aiperf.operator.handlers.lifecycle import on_benchmark_complete

        patch = MagicMock()
        patch.status = {}

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.try_claim_completion",
                new_callable=AsyncMock,
                return_value=False,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            await on_benchmark_complete(
                body=_body_without_annotation(),
                status={"phase": Phase.RUNNING, "jobId": "j", "jobSetName": "js"},
                name="j",
                namespace="ns",
                patch=patch,
            )

        mock_handle.assert_not_called()


class TestMonitorIntegration:
    """Verify monitor_progress' JobSet-completed path honours the durable claim."""

    @pytest.mark.asyncio
    async def test_jobset_completed_annotation_preset_skips_handle_completion(
        self,
    ) -> None:
        """When the CR body has the annotation, JobSet Completed should not re-enter handle_completion."""
        from aiperf.operator.handlers.monitor import monitor_progress

        # Minimal kopf.Patch-like object
        patch = MagicMock()
        patch.status = {}

        jobset_raw = {
            "status": {
                "conditions": [{"type": "Completed", "status": "True"}],
                "replicatedJobsStatus": [],
            },
            "metadata": {"labels": {}},
            "spec": {},
        }
        mock_jobset_obj = MagicMock()
        mock_jobset_obj.raw = jobset_raw

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.get_api",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.AsyncJobSet.get",
                new_callable=AsyncMock,
                return_value=mock_jobset_obj,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new_callable=AsyncMock,
            ) as mock_handle,
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new_callable=AsyncMock,
            ),
        ):
            await monitor_progress(
                body=_body_with_annotation(),
                status={
                    "phase": Phase.RUNNING,
                    "jobId": "j",
                    "jobSetName": "js",
                },
                spec={},
                name="j",
                namespace="ns",
                patch=patch,
            )

        mock_handle.assert_not_called()
