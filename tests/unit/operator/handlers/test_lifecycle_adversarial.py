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

import contextlib
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import kopf
import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.cr_refs import AIPERF_JOB_API_VERSION
from aiperf.kubernetes.phase import Phase
from aiperf.operator.client_cache import (
    _reset_for_testing,
    is_cancellation_requested,
    request_cancellation,
)
from aiperf.operator.handlers.lifecycle import (
    acknowledge_timeout_update,
    on_benchmark_complete,
    on_cancel,
    on_delete,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NAMESPACE = "perf-lab"
_JOB_NAME = "aiperf-bench-7f2a"
_JOB_UID = "cf3f1f5e-0d21-4c0c-9d2a-6f4b1c2f7a10"
_JOBSET_NAME = "aiperf-bench-7f2a-js"
_JOBSET_UID = "0b7f4a2c-9e13-4a55-8f7d-2c1f9a4e6b33"
# Process-local lifecycle state is keyed by immutable CR identity, so every
# cancellation assertion must address ``ns/job@uid`` rather than ``ns/job``.
# A bare-key assertion silently stops observing the flag the handler sets.
_JOB_KEY = f"{_NAMESPACE}/{_JOB_NAME}@{_JOB_UID}"


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
            "name": _JOB_NAME,
            "namespace": _NAMESPACE,
            "uid": _JOB_UID,
            "generation": generation,
        },
    }


def _running_status(**overrides: Any) -> dict[str, Any]:
    """Build a running status snapshot with stale phase-column values."""
    status: dict[str, Any] = {
        "phase": Phase.RUNNING,
        "jobId": _JOB_NAME,
        "jobSetName": _JOBSET_NAME,
        "currentPhase": "profile",
        "subPhase": "profiling",
    }
    status.update(overrides)
    return status


@contextlib.contextmanager
def _stub_parent_fence(resource_version: str = "1") -> Iterator[AsyncMock]:
    """Answer the live-CR identity fence every lifecycle handler crosses.

    ``on_cancel`` and ``on_benchmark_complete`` re-read the parent AIPerfJob and
    compare its immutable UID before touching status. Without this stub the read
    reaches a real apiserver whenever a kubeconfig exists, so the handler's
    behaviour would depend on cluster state. The fence itself is covered
    directly in ``test_job_identity_fences.py``.
    """
    fence = AsyncMock(
        name="current_aiperfjob_resource_version", return_value=resource_version
    )
    with mock_patch(
        "aiperf.operator.handlers.lifecycle.current_aiperfjob_resource_version",
        new=fence,
    ):
        yield fence


def _owned_jobset() -> dict[str, Any]:
    """Build a JobSet snapshot controlled by the fixture AIPerfJob identity."""
    return {
        "metadata": {
            "name": _JOBSET_NAME,
            "uid": _JOBSET_UID,
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_JOB_API_VERSION,
                    "kind": "AIPerfJob",
                    "name": _JOB_NAME,
                    "uid": _JOB_UID,
                    "controller": True,
                }
            ],
        }
    }


@asynccontextmanager
async def _fake_k8s_delete(
    delete_result: object | BaseException = None,
) -> AsyncIterator[SimpleNamespace]:
    """Install a fake owned-JobSet API and expose the delete mock.

    The delete now runs through ``_job_identity.delete_owned_aiperfjob_jobset``,
    which reads the JobSet first to prove exact controller ownership and only
    then issues a UID-preconditioned delete. Both round-trips are stubbed here
    so the real helper -- including its 404/409/5xx mapping -- executes without
    a cluster.
    """
    delete = AsyncMock(name="delete_namespaced_custom_object")
    if isinstance(delete_result, BaseException):
        delete.side_effect = delete_result
    else:
        delete.return_value = delete_result

    custom = MagicMock(name="CustomObjectsApi")
    custom.get_namespaced_custom_object = AsyncMock(return_value=_owned_jobset())
    custom.delete_namespaced_custom_object = delete

    @asynccontextmanager
    async def fake_client() -> AsyncIterator[MagicMock]:
        yield MagicMock(name="ApiClient")

    with (
        mock_patch(
            "aiperf.operator.handlers._job_identity.k8s_client",
            side_effect=fake_client,
        ),
        mock_patch(
            "aiperf.operator.handlers._job_identity.client.CustomObjectsApi",
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
            with (
                _stub_parent_fence(),
                mock_patch("aiperf.operator.handlers.lifecycle.events.cancelled"),
            ):
                await on_cancel(
                    body=_body(generation="42"),
                    spec={"cancel": True},
                    status=_running_status(),
                    name=_JOB_NAME,
                    namespace=_NAMESPACE,
                    patch=patch,
                )

        assert patch.status["phase"] == Phase.CANCELLED
        assert patch.status["currentPhase"] is None
        assert patch.status["subPhase"] is None
        assert patch.status["observedGeneration"] == 42
        assert is_cancellation_requested(_JOB_KEY) is True

    @pytest.mark.asyncio
    async def test_on_cancel_delete_temporary_error_does_not_stamp_observed_generation(
        self,
    ) -> None:
        """Apiserver delete failures retry without acknowledging the spec edit."""
        patch = _patch()

        async with _fake_k8s_delete(ApiException(status=500, reason="apiserver down")):
            with (
                _stub_parent_fence(),
                pytest.raises(
                    kopf.TemporaryError, match=r"delete failed after cancel \(500\)"
                ),
            ):
                await on_cancel(
                    body=_body(generation=13),
                    spec={"cancel": True},
                    status=_running_status(),
                    name=_JOB_NAME,
                    namespace=_NAMESPACE,
                    patch=patch,
                )

        assert "phase" not in patch.status
        assert "observedGeneration" not in patch.status
        assert is_cancellation_requested(_JOB_KEY) is True

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
    async def test_on_cancel_inapplicable_inputs_acknowledge_without_side_effects(
        self, spec: dict[str, Any], status: dict[str, Any]
    ) -> None:
        """No-op cancel updates acknowledge the edit without runtime side effects."""
        patch = _patch()

        with (
            _stub_parent_fence(),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.delete_owned_aiperfjob_jobset",
                new_callable=AsyncMock,
            ) as mock_delete_jobset,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                new_callable=AsyncMock,
            ) as mock_close,
        ):
            await on_cancel(
                body=_body(generation=21),
                spec=spec,
                status=status,
                name=_JOB_NAME,
                namespace=_NAMESPACE,
                patch=patch,
            )

        assert patch.status == {"observedGeneration": 21}
        mock_delete_jobset.assert_not_awaited()
        mock_close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_cancel_missing_jobset_name_still_sets_cancellation_flag_and_status(
        self,
    ) -> None:
        """Malformed status without jobSetName still records the user cancellation."""
        patch = _patch()

        with (
            _stub_parent_fence(),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.delete_owned_aiperfjob_jobset",
                new_callable=AsyncMock,
            ) as mock_delete_jobset,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.events.cancelled"
            ) as mock_event,
        ):
            await on_cancel(
                body=_body(generation=3),
                spec={"cancel": True},
                status=_running_status(jobSetName=None),
                name=_JOB_NAME,
                namespace=_NAMESPACE,
                patch=patch,
            )

        assert patch.status["phase"] == Phase.CANCELLED
        assert patch.status["observedGeneration"] == 3
        assert is_cancellation_requested(_JOB_KEY) is True
        mock_delete_jobset.assert_not_awaited()
        mock_event.assert_called_once()


@pytest.mark.asyncio
async def test_acknowledge_timeout_update_stamps_observed_generation() -> None:
    """The monitor consumes timeout edits without recreating the JobSet."""
    patch = _patch()

    await acknowledge_timeout_update(body=_body(generation=31), patch=patch)

    assert patch.status == {"observedGeneration": 31}


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
            _stub_parent_fence() as fence,
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
                name=_JOB_NAME,
                namespace=_NAMESPACE,
                patch=patch,
            )

        assert patch.status == {}
        # A pure no-op must not even pay the identity round-trip.
        fence.assert_not_awaited()
        mock_claim.assert_not_awaited()
        mock_handle.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_lost_claim_does_not_stamp_observed_generation(
        self,
    ) -> None:
        """A peer handler winning the durable claim is not a successful reconcile."""
        patch = _patch()

        with (
            _stub_parent_fence(),
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
                name=_JOB_NAME,
                namespace=_NAMESPACE,
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
            _stub_parent_fence(),
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
                name=_JOB_NAME,
                namespace=_NAMESPACE,
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

        async def _cancel_during_completion(*_args: Any, **_kwargs: Any) -> None:
            # Must be the UID-scoped key: the handler's own cancellation gate
            # reads ``ns/job@uid``, so a bare-key request would leave the gate
            # closed and the test would assert nothing.
            request_cancellation(_JOB_KEY)

        with (
            _stub_parent_fence(),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.try_claim_completion",
                new_callable=AsyncMock,
                return_value=True,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new_callable=AsyncMock,
                side_effect=_cancel_during_completion,
            ) as mock_handle,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.get_or_create_progress_client",
                new_callable=AsyncMock,
                return_value=progress_client,
            ) as mock_get_client,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                new_callable=AsyncMock,
            ) as mock_close,
        ):
            await on_benchmark_complete(
                body=_body(generation=58),
                status=_running_status(),
                name=_JOB_NAME,
                namespace=_NAMESPACE,
                patch=patch,
            )

        assert "observedGeneration" not in patch.status
        mock_handle.assert_awaited_once()
        mock_get_client.assert_not_awaited()
        progress_client.send_shutdown.assert_not_awaited()
        mock_close.assert_awaited_once_with(_JOB_KEY)

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_permanent_error_does_not_send_shutdown(
        self,
    ) -> None:
        """Permanent completion failures propagate before controller shutdown."""
        progress_client = AsyncMock(name="ProgressClient")
        patch = _patch()

        with (
            _stub_parent_fence(),
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
                name=_JOB_NAME,
                namespace=_NAMESPACE,
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
            _stub_parent_fence(),
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
                name=_JOB_NAME,
                namespace=_NAMESPACE,
                patch=patch,
            )

        assert patch.status["observedGeneration"] == 59
        mock_handle.assert_awaited_once()
        progress_client.send_shutdown.assert_awaited_once()
        mock_close.assert_awaited_once_with(_JOB_KEY)


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
                name=_JOB_NAME,
                namespace=_NAMESPACE,
                status={},
                uid=_JOB_UID,
            )

        assert is_cancellation_requested(_JOB_KEY) is True
        assert call_order == [
            f"close:{_JOB_KEY}",
            f"cleanup:{_NAMESPACE}/{_JOB_NAME}:False",
        ]
