# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator lifecycle handlers (on_delete, on_cancel, on_benchmark_complete)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import kopf
import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.cr_refs import AIPERF_JOB_API_VERSION
from aiperf.kubernetes.phase import Phase
from aiperf.operator.client_cache import (
    _cancellation_events,
    _reset_for_testing,
    _shutdown_sent,
    is_cancellation_requested,
)

_JOB_UID = "b0c9d2f4-7a18-4e63-9c05-1d8f3a6b2e47"
_JOBSET_UID = "3e7a1c85-64bf-4d02-8a19-5c2e9f0b7d31"
# Cancellation and progress-client state is keyed by immutable CR identity.
_JOB_KEY = f"ns/j1@{_JOB_UID}"


def _job_body() -> dict:
    """Build an AIPerfJob body carrying the immutable identity kopf supplies."""
    return {
        "kind": "AIPerfJob",
        "metadata": {"name": "j1", "namespace": "ns", "uid": _JOB_UID},
    }


def _owned_jobset() -> dict:
    """Build a JobSet snapshot controlled by ``_job_body``'s identity."""
    return {
        "metadata": {
            "name": "js1",
            "uid": _JOBSET_UID,
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_JOB_API_VERSION,
                    "kind": "AIPerfJob",
                    "name": "j1",
                    "uid": _JOB_UID,
                    "controller": True,
                }
            ],
        }
    }


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
        from contextlib import asynccontextmanager

        from aiperf.operator.handlers.lifecycle import on_cancel

        mock_delete = AsyncMock(return_value={})
        mock_custom = MagicMock(
            get_namespaced_custom_object=AsyncMock(return_value=_owned_jobset()),
            delete_namespaced_custom_object=mock_delete,
        )
        patch = MagicMock()
        patch.status = {}

        @asynccontextmanager
        async def _fake_client(*_args, **_kwargs):
            yield MagicMock()

        with (
            # The JobSet delete runs through the shared identity helper, which
            # opens its own client to prove controller ownership before issuing
            # a UID-preconditioned delete. Patch it there, not on lifecycle.
            mock_patch(
                "aiperf.operator.handlers._job_identity.k8s_client",
                side_effect=_fake_client,
            ),
            mock_patch(
                "aiperf.operator.handlers._job_identity.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.current_aiperfjob_resource_version",
                new=AsyncMock(return_value="11"),
            ),
        ):
            await on_cancel(
                body=_job_body(),
                spec={"cancel": True},
                status={"phase": Phase.RUNNING, "jobId": "j1", "jobSetName": "js1"},
                name="j1",
                namespace="ns",
                patch=patch,
            )

        mock_delete.assert_awaited_once()
        assert patch.status["phase"] == Phase.CANCELLED
        assert is_cancellation_requested(_JOB_KEY) is True

    @pytest.mark.asyncio
    async def test_delete_failure_retries_without_terminalizing(
        self, mock_events: None
    ) -> None:
        """Non-404 delete failure must not mark Cancelled while pods may run."""
        from contextlib import asynccontextmanager

        from aiperf.operator.handlers.lifecycle import on_cancel

        mock_delete = AsyncMock(
            side_effect=ApiException(status=500, reason="apiserver unavailable")
        )
        mock_custom = MagicMock(
            get_namespaced_custom_object=AsyncMock(return_value=_owned_jobset()),
            delete_namespaced_custom_object=mock_delete,
        )
        patch = MagicMock()
        patch.status = {}

        @asynccontextmanager
        async def _fake_client(*_args, **_kwargs):
            yield MagicMock()

        with (
            mock_patch(
                "aiperf.operator.handlers._job_identity.k8s_client",
                side_effect=_fake_client,
            ),
            mock_patch(
                "aiperf.operator.handlers._job_identity.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.current_aiperfjob_resource_version",
                new=AsyncMock(return_value="11"),
            ),
            pytest.raises(kopf.TemporaryError),
        ):
            await on_cancel(
                body=_job_body(),
                spec={"cancel": True},
                status={"phase": Phase.RUNNING, "jobId": "j1", "jobSetName": "js1"},
                name="j1",
                namespace="ns",
                patch=patch,
            )

        mock_delete.assert_awaited_once()
        assert patch.status.get("phase") != Phase.CANCELLED
        assert is_cancellation_requested(_JOB_KEY) is True


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
    async def test_skips_when_cancellation_requested_before_claim(self) -> None:
        """If on_delete fired and stamped the cancellation flag, the
        completion handler MUST NOT call try_claim_completion. Otherwise
        the durable claim annotation is written to the CR right before
        ownerRef GC removes it, leaking the claim to no recovery path.
        Pairs with the same gate at the top of monitor.py's tick.
        """
        from aiperf.operator.client_cache import request_cancellation
        from aiperf.operator.handlers.lifecycle import on_benchmark_complete

        request_cancellation("ns/j")
        assert "ns/j" in _cancellation_events

        patch = MagicMock()
        patch.status = {}

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.try_claim_completion",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_claim,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new_callable=AsyncMock,
            ) as mock_handle,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.get_or_create_progress_client",
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

        # Critical: claim was NEVER attempted.
        mock_claim.assert_not_called()
        mock_handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_failure_narrow_exception_propagates_unrelated(
        self,
    ) -> None:
        """The shutdown except clause caught everything; narrow it to
        (aiohttp.ClientError, TimeoutError, OSError) so programmer
        errors (TypeError/AttributeError/etc.) propagate instead of being
        silenced as ShutdownSignalFailed.
        """
        from aiperf.operator.handlers.lifecycle import on_benchmark_complete

        mock_client = AsyncMock()
        # AttributeError is a programmer bug — must NOT be silently swallowed.
        mock_client.send_shutdown = AsyncMock(side_effect=AttributeError("no attr"))
        patch = MagicMock()
        patch.status = {}
        body = {"kind": "AIPerfJob"}

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
            pytest.raises(AttributeError, match="no attr"),
        ):
            await on_benchmark_complete(
                body=body,
                status={"phase": Phase.RUNNING, "jobId": "j", "jobSetName": "js"},
                name="j",
                namespace="ns",
                patch=patch,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_factory",
        [
            param(lambda: TimeoutError("slow"), id="asyncio_timeout"),
            param(lambda: OSError("io"), id="os_error"),
        ],
    )  # fmt: skip
    async def test_shutdown_failure_narrow_exception_caught_for_transport(
        self, exc_factory
    ) -> None:
        """TimeoutError and OSError must still be caught and emit a
        ShutdownSignalFailed event (the narrowed handler covers transport).
        """
        from aiperf.operator.handlers.lifecycle import on_benchmark_complete

        mock_client = AsyncMock()
        mock_client.send_shutdown = AsyncMock(side_effect=exc_factory())
        patch = MagicMock()
        patch.status = {}
        body = {"kind": "AIPerfJob"}

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
        ):
            await on_benchmark_complete(
                body=body,
                status={"phase": Phase.RUNNING, "jobId": "j", "jobSetName": "js"},
                name="j",
                namespace="ns",
                patch=patch,
            )

        mock_kopf_event.assert_called_once()
        assert mock_kopf_event.call_args.kwargs["reason"] == "ShutdownSignalFailed"

    @pytest.mark.asyncio
    async def test_shutdown_transport_failure_logs_exception_and_emits_kopf_event(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When send_shutdown raises a transport error, the handler must
        preserve the traceback via logger.exception and surface the failure
        to cluster operators via a kopf Warning event.
        """
        import logging

        from aiperf.operator.handlers.lifecycle import on_benchmark_complete

        mock_client = AsyncMock()
        mock_client.send_shutdown = AsyncMock(side_effect=OSError("boom"))
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
