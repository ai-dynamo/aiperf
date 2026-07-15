# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for AIPerfJob lifecycle handlers.

Focuses on:
- cancel-path terminalization, cancellation flags, and phase-column cleanup
- benchmark-complete claim gates and shutdown side effects
- observedGeneration stamping only on successful reconcile paths
- malformed or incomplete body/spec/status inputs at the kopf trust boundary

Out of scope: result artifact parsing and JobSet monitor state-machine edges;
see sibling files ``test_completion_handler.py`` and
``test_monitor_state_machine_edges.py`` for those contracts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import kopf
import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.operator.client_cache import _reset_for_testing, is_cancellation_requested
from aiperf.operator.handlers.lifecycle import (
    on_benchmark_complete,
    on_cancel,
    on_delete,
)
from aiperf.operator.status import Phase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch() -> MagicMock:
    """Build a kopf-like patch object with an observable status dict."""
    patch = MagicMock(name="kopf_patch")
    patch.status = {}
    return patch


def _body(*, generation: object = 7) -> dict[str, Any]:
    """Build a minimal AIPerfJob body with realistic metadata."""
    return {
        "kind": "AIPerfJob",
        "metadata": {
            "name": "aiperf-bench-7f2a",
            "namespace": "perf-lab",
            "generation": generation,
        },
    }


def _running_status(**overrides: Any) -> dict[str, Any]:
    """Build a running status snapshot with stale phase-column values."""
    status: dict[str, Any] = {
        "phase": Phase.RUNNING,
        "jobId": "aiperf-bench-7f2a",
        "jobSetName": "aiperf-bench-7f2a-js",
        "currentPhase": "profile",
        "subPhase": "profiling",
    }
    status.update(overrides)
    return status


@asynccontextmanager
async def _fake_k8s_delete(
    delete_result: object | BaseException = None,
) -> AsyncIterator[SimpleNamespace]:
    """Install a fake JobSet delete client and expose the delete mock."""
    delete = AsyncMock(name="delete_namespaced_custom_object")
    if isinstance(delete_result, BaseException):
        delete.side_effect = delete_result
    else:
        delete.return_value = delete_result

    custom = MagicMock(name="CustomObjectsApi")
    custom.delete_namespaced_custom_object = delete

    @asynccontextmanager
    async def fake_client() -> AsyncIterator[MagicMock]:
        yield MagicMock(name="ApiClient")

    with (
        mock_patch(
            "aiperf.operator.handlers.lifecycle.k8s_client",
            return_value=fake_client(),
        ),
        mock_patch(
            "aiperf.operator.handlers.lifecycle.client.CustomObjectsApi",
            return_value=custom,
        ),
    ):
        yield SimpleNamespace(delete=delete)


@pytest.fixture(autouse=True)
def _reset_client_cache() -> AsyncIterator[None]:
    """Reset lifecycle singleton state around every adversarial case."""
    _reset_for_testing()
    yield
    _reset_for_testing()


# =============================================================================
# Cancel path
# =============================================================================


class TestOnCancelAdversarial:
    """Cancel-path edge cases around terminalization and retry behavior."""

    @pytest.mark.asyncio
    async def test_on_cancel_jobset_404_terminalizes_and_clears_phase_columns(
        self,
    ) -> None:
        """A missing JobSet means cleanup already won; the CR still becomes Cancelled."""
        patch = _patch()

        async with _fake_k8s_delete(ApiException(status=404, reason="Not Found")):
            with mock_patch("aiperf.operator.handlers.lifecycle.events.cancelled"):
                await on_cancel(
                    body=_body(generation="42"),
                    spec={"cancel": True},
                    status=_running_status(),
                    name="aiperf-bench-7f2a",
                    namespace="perf-lab",
                    patch=patch,
                )

        assert patch.status["phase"] == Phase.CANCELLED
        assert patch.status["currentPhase"] is None
        assert patch.status["subPhase"] is None
        assert patch.status["observedGeneration"] == 42
        assert is_cancellation_requested("perf-lab/aiperf-bench-7f2a") is True

    @pytest.mark.asyncio
    async def test_on_cancel_delete_temporary_error_does_not_stamp_observed_generation(
        self,
    ) -> None:
        """Apiserver delete failures retry without acknowledging the spec edit."""
        patch = _patch()

        async with _fake_k8s_delete(ApiException(status=500, reason="apiserver down")):
            with pytest.raises(kopf.TemporaryError, match=r"Failed to delete JobSet"):
                await on_cancel(
                    body=_body(generation=13),
                    spec={"cancel": True},
                    status=_running_status(),
                    name="aiperf-bench-7f2a",
                    namespace="perf-lab",
                    patch=patch,
                )

        assert "phase" not in patch.status
        assert "observedGeneration" not in patch.status
        assert is_cancellation_requested("perf-lab/aiperf-bench-7f2a") is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "spec,status",
        [
            param({}, _running_status(), id="missing-cancel-field"),
            param({"cancel": False}, _running_status(), id="cancel-false"),
            param({"cancel": True}, _running_status(phase=Phase.COMPLETED), id="completed"),
            param({"cancel": True}, _running_status(phase=Phase.FAILED), id="failed"),
            param({"cancel": True}, _running_status(phase=Phase.CANCELLED), id="cancelled"),
        ],
    )  # fmt: skip
    async def test_on_cancel_inapplicable_inputs_leave_status_and_clients_untouched(
        self, spec: dict[str, Any], status: dict[str, Any]
    ) -> None:
        """No-op cancel updates must not delete JobSets or acknowledge generation."""
        patch = _patch()

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.k8s_client",
            ) as mock_k8s_client,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                new_callable=AsyncMock,
            ) as mock_close,
        ):
            await on_cancel(
                body=_body(generation=21),
                spec=spec,
                status=status,
                name="aiperf-bench-7f2a",
                namespace="perf-lab",
                patch=patch,
            )

        assert patch.status == {}
        mock_k8s_client.assert_not_called()
        mock_close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_cancel_missing_jobset_name_still_sets_cancellation_flag_and_status(
        self,
    ) -> None:
        """Malformed status without jobSetName still records the user cancellation."""
        patch = _patch()

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.k8s_client"
            ) as mock_k8s_client,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.events.cancelled"
            ) as mock_event,
        ):
            await on_cancel(
                body=_body(generation=3),
                spec={"cancel": True},
                status=_running_status(jobSetName=None),
                name="aiperf-bench-7f2a",
                namespace="perf-lab",
                patch=patch,
            )

        assert patch.status["phase"] == Phase.CANCELLED
        assert patch.status["observedGeneration"] == 3
        assert is_cancellation_requested("perf-lab/aiperf-bench-7f2a") is True
        mock_k8s_client.assert_not_called()
        mock_event.assert_called_once()


# =============================================================================
# Benchmark-complete path
# =============================================================================


class TestOnBenchmarkCompleteAdversarial:
    """Completion-signal edge cases around claims, retries, and shutdown."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status",
        [
            param(_running_status(jobSetName=None), id="missing-jobset-name"),
            param(_running_status(jobSetName=""), id="empty-jobset-name"),
            param(_running_status(phase=Phase.COMPLETED), id="completed"),
            param(_running_status(phase=Phase.FAILED), id="failed"),
            param(_running_status(phase=Phase.CANCELLED), id="cancelled"),
        ],
    )  # fmt: skip
    async def test_on_benchmark_complete_inapplicable_status_does_not_claim_or_stamp(
        self, status: dict[str, Any]
    ) -> None:
        """Terminal or incomplete status snapshots must be pure no-ops."""
        patch = _patch()

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.try_claim_completion",
                new_callable=AsyncMock,
            ) as mock_claim,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            await on_benchmark_complete(
                body=_body(generation=55),
                status=status,
                name="aiperf-bench-7f2a",
                namespace="perf-lab",
                patch=patch,
            )

        assert patch.status == {}
        mock_claim.assert_not_awaited()
        mock_handle.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_lost_claim_does_not_stamp_observed_generation(
        self,
    ) -> None:
        """A peer handler winning the durable claim is not a successful reconcile."""
        patch = _patch()

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.try_claim_completion",
                new_callable=AsyncMock,
                return_value=False,
            ) as mock_claim,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            await on_benchmark_complete(
                body=_body(generation=56),
                status=_running_status(),
                name="aiperf-bench-7f2a",
                namespace="perf-lab",
                patch=patch,
            )

        mock_claim.assert_awaited_once()
        mock_handle.assert_not_awaited()
        assert patch.status == {}

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_temporary_error_does_not_stamp_observed_generation(
        self,
    ) -> None:
        """Transient completion failures must retry without acknowledging generation."""
        patch = _patch()

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.try_claim_completion",
                new_callable=AsyncMock,
                return_value=True,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new_callable=AsyncMock,
                side_effect=kopf.TemporaryError("results fetch retry", delay=5),
            ),
            pytest.raises(kopf.TemporaryError, match="results fetch retry"),
        ):
            await on_benchmark_complete(
                body=_body(generation=57),
                status=_running_status(),
                name="aiperf-bench-7f2a",
                namespace="perf-lab",
                patch=patch,
            )

        assert "observedGeneration" not in patch.status

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_mid_completion_cancel_does_not_stamp_observed_generation(
        self,
    ) -> None:
        """A cancellation that lands between the claim and handle_completion's
        own guards leaves sb without a phase; absence of COMPLETED must not be
        read as success and stamp generation."""
        progress_client = AsyncMock(name="ProgressClient")
        patch = _patch()

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
                return_value=progress_client,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                new_callable=AsyncMock,
            ),
        ):
            await on_benchmark_complete(
                body=_body(generation=58),
                status=_running_status(),
                name="aiperf-bench-7f2a",
                namespace="perf-lab",
                patch=patch,
            )

        assert "observedGeneration" not in patch.status

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_permanent_error_does_not_send_shutdown(
        self,
    ) -> None:
        """Permanent completion failures propagate before controller shutdown."""
        progress_client = AsyncMock(name="ProgressClient")
        patch = _patch()

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.try_claim_completion",
                new_callable=AsyncMock,
                return_value=True,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new_callable=AsyncMock,
                side_effect=kopf.PermanentError("malformed completion bundle"),
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.get_or_create_progress_client",
                new_callable=AsyncMock,
                return_value=progress_client,
            ) as mock_get_client,
            pytest.raises(kopf.PermanentError, match="malformed completion bundle"),
        ):
            await on_benchmark_complete(
                body=_body(generation=58),
                status=_running_status(),
                name="aiperf-bench-7f2a",
                namespace="perf-lab",
                patch=patch,
            )

        mock_get_client.assert_not_awaited()
        progress_client.send_shutdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_success_stamps_generation_after_completion_and_closes_client(
        self,
    ) -> None:
        """The successful fast path stamps generation and releases the progress client."""
        progress_client = AsyncMock(name="ProgressClient")
        patch = _patch()

        async def _complete(*_args: Any, sb: Any, **_kwargs: Any) -> None:
            sb.set_phase(Phase.COMPLETED)

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.try_claim_completion",
                new_callable=AsyncMock,
                return_value=True,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new_callable=AsyncMock,
                side_effect=_complete,
            ) as mock_handle,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.get_or_create_progress_client",
                new_callable=AsyncMock,
                return_value=progress_client,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                new_callable=AsyncMock,
            ) as mock_close,
        ):
            await on_benchmark_complete(
                body=_body(generation="59"),
                status=_running_status(),
                name="aiperf-bench-7f2a",
                namespace="perf-lab",
                patch=patch,
            )

        assert patch.status["observedGeneration"] == 59
        mock_handle.assert_awaited_once()
        progress_client.send_shutdown.assert_awaited_once()
        mock_close.assert_awaited_once_with("perf-lab/aiperf-bench-7f2a")


# =============================================================================
# Delete path
# =============================================================================


class TestOnDeleteAdversarial:
    """Deletion side effects that protect concurrent lifecycle handlers."""

    @pytest.mark.asyncio
    async def test_on_delete_missing_job_id_uses_resource_name_for_cancellation_and_cleanup(
        self,
    ) -> None:
        """A pre-controller-delete status still cancels and cleans the named job."""
        call_order: list[str] = []

        async def fake_close(key: str) -> None:
            call_order.append(f"close:{key}")

        async def fake_cleanup(
            namespace: str, name: str, status: dict[str, Any]
        ) -> None:
            call_order.append(f"cleanup:{namespace}/{name}:{bool(status)}")

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                side_effect=fake_close,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.on_aiperfjob_delete_index_cleanup",
                side_effect=fake_cleanup,
            ),
        ):
            await on_delete(
                name="aiperf-bench-7f2a",
                namespace="perf-lab",
                status={},
            )

        assert is_cancellation_requested("perf-lab/aiperf-bench-7f2a") is True
        assert call_order == [
            "close:perf-lab/aiperf-bench-7f2a",
            "cleanup:perf-lab/aiperf-bench-7f2a:False",
        ]
