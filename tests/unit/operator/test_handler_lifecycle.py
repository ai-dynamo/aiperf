# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator lifecycle handlers (on_delete, on_cancel, on_benchmark_complete)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import pytest
from pytest import param

from aiperf.operator.client_cache import (
    _reset_for_testing,
    _shutdown_sent,
)
from aiperf.operator.status import Phase


@pytest.fixture(autouse=True)
def _clean_state():
    _reset_for_testing()
    yield
    _reset_for_testing()


class TestOnDelete:
    """Tests for on_delete handler."""

    @pytest.mark.asyncio
    async def test_closes_progress_client(self) -> None:
        from aiperf.operator.handlers.lifecycle import on_delete

        with mock_patch(
            "aiperf.operator.handlers.lifecycle.close_progress_client",
            new_callable=AsyncMock,
        ) as mock_close:
            await on_delete(
                name="test-job", namespace="default", status={"jobId": "test-job"}
            )

        mock_close.assert_called_once_with("default/test-job")


class TestOnCancel:
    """Tests for on_cancel handler."""

    @pytest.fixture
    def mock_events(self):
        with mock_patch("aiperf.operator.events.cancelled"):
            yield

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "spec,status",
        [
            param({"cancel": False}, {"phase": Phase.RUNNING}, id="cancel_false"),
            param({"cancel": True}, {"phase": Phase.COMPLETED}, id="already_completed"),
            param({"cancel": True}, {"phase": Phase.FAILED}, id="already_failed"),
            param({"cancel": True}, {"phase": Phase.CANCELLED}, id="already_cancelled"),
        ],
    )  # fmt: skip
    async def test_noop_when_not_applicable(self, spec: dict, status: dict) -> None:
        from aiperf.operator.handlers.lifecycle import on_cancel

        patch = MagicMock()
        patch.status = {}

        await on_cancel(
            body={}, spec=spec, status=status, name="j", namespace="ns", patch=patch
        )
        assert patch.status.get("phase") != Phase.CANCELLED

    @pytest.mark.asyncio
    async def test_cancels_single_job(self, mock_events: None) -> None:
        from aiperf.operator.handlers.lifecycle import on_cancel

        mock_js = AsyncMock()
        patch = MagicMock()
        patch.status = {}

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.get_api",
                new_callable=AsyncMock,
                return_value=AsyncMock(),
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.AsyncJobSet.get",
                new_callable=AsyncMock,
                return_value=mock_js,
            ),
        ):
            await on_cancel(
                body={},
                spec={"cancel": True},
                status={"phase": Phase.RUNNING, "jobId": "j1", "jobSetName": "js1"},
                name="j1",
                namespace="ns",
                patch=patch,
            )

        mock_js.delete.assert_called_once()
        assert patch.status["phase"] == Phase.CANCELLED


class TestOnBenchmarkComplete:
    """Tests for on_benchmark_complete handler."""

    @pytest.mark.asyncio
    async def test_skips_terminal_phases(self) -> None:
        from aiperf.operator.handlers.lifecycle import on_benchmark_complete

        patch = MagicMock()
        patch.status = {}

        await on_benchmark_complete(
            body={},
            status={"phase": Phase.COMPLETED},
            name="j",
            namespace="ns",
            patch=patch,
        )
        # No error means it returned early

    @pytest.mark.asyncio
    async def test_skips_duplicate_shutdown(self) -> None:
        from aiperf.operator.handlers.lifecycle import on_benchmark_complete

        _shutdown_sent.add("ns/j")
        patch = MagicMock()
        patch.status = {}

        with mock_patch(
            "aiperf.operator.handlers.lifecycle.handle_completion",
            new_callable=AsyncMock,
        ) as mock_handle:
            await on_benchmark_complete(
                body={},
                status={"phase": Phase.RUNNING, "jobId": "j", "jobSetName": "js"},
                name="j",
                namespace="ns",
                patch=patch,
            )

        mock_handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetches_results_and_shuts_down(self) -> None:
        from aiperf.operator.handlers.lifecycle import on_benchmark_complete

        mock_client = AsyncMock()
        patch = MagicMock()
        patch.status = {}

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.try_claim_completion",
                new_callable=AsyncMock,
                return_value=True,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.get_or_create_progress_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                new_callable=AsyncMock,
            ),
        ):
            await on_benchmark_complete(
                body={},
                status={"phase": Phase.RUNNING, "jobId": "j", "jobSetName": "js"},
                name="j",
                namespace="ns",
                patch=patch,
            )

        mock_client.send_shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_failure_logs_exception_and_emits_kopf_event(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When send_shutdown raises, the handler must preserve the traceback
        via logger.exception and surface the failure to cluster operators
        via a kopf Warning event."""
        import logging

        from aiperf.operator.handlers.lifecycle import on_benchmark_complete

        mock_client = AsyncMock()
        mock_client.send_shutdown = AsyncMock(side_effect=RuntimeError("boom"))
        patch = MagicMock()
        patch.status = {}
        body = {"kind": "AIPerfJob", "metadata": {"name": "j", "namespace": "ns"}}

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.try_claim_completion",
                new_callable=AsyncMock,
                return_value=True,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.get_or_create_progress_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.kopf.event"
            ) as mock_kopf_event,
            caplog.at_level(logging.ERROR, logger="aiperf.operator.handlers.lifecycle"),
        ):
            await on_benchmark_complete(
                body=body,
                status={"phase": Phase.RUNNING, "jobId": "j", "jobSetName": "js"},
                name="j",
                namespace="ns",
                patch=patch,
            )

        # Exception is logged with traceback
        assert any(
            "Failed to send shutdown" in rec.message and rec.exc_info is not None
            for rec in caplog.records
        ), (
            f"Expected exception log with traceback, got: {[r.message for r in caplog.records]}"
        )

        # kopf Warning event is emitted
        mock_kopf_event.assert_called_once()
        call_kwargs = mock_kopf_event.call_args.kwargs
        assert call_kwargs["type"] == "Warning"
        assert call_kwargs["reason"] == "ShutdownSignalFailed"
        assert "boom" in call_kwargs["message"]
        # Positional body arg
        assert mock_kopf_event.call_args.args[0] is body
