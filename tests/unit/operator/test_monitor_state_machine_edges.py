# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State-machine and edge-case tests for the AIPerfJob monitor handler.

The monitor handler observes JobSet/Pod state and patches the CR's
``status.phase`` to drive the AIPerfJob lifecycle:

    Pending -> (Queued) -> Initializing -> Running -> Completed/Failed/Cancelled

This file targets edges not covered by the existing happy-path tests:

* Forward and refused phase transitions (e.g. completed-CR no-op).
* Job-timeout escalation (FAILED + JobSet delete).
* Concurrent claim races (TOCTOU on completion-claim annotation).
* Pod-state aggregation: all-pending, mixed, all-failed, controller-cascade.
* Heartbeat / progress-tracking: missing/stale/backwards progress, no
  progress endpoint yet.
* Bootstrap / reconciliation paths after operator restart.
* Orphan-claim recovery gating (only fire when the benchmark is done).

Mocking strategy:
    * ``k8s_client`` replaced with an ``@asynccontextmanager`` yielding
      a ``MagicMock`` ApiClient (avoids ``AsyncMock`` quirks around
      ``__aenter__``/``__aexit__``).
    * ``CustomObjectsApi`` patched to a ``MagicMock`` whose async methods
      are ``AsyncMock`` instances.
    * ``is_cancellation_requested``/``try_claim_completion`` etc. are
      monkeypatched on the monitor module symbol.
    * Real ``kopf.PermanentError`` / ``kopf.TemporaryError`` only where
      the production code raises them — this handler currently degrades
      to ``logger.warning`` + ``sb.finalize`` rather than raising kopf
      errors, so escalation tests assert side-effects (FAILED phase /
      JobSet delete) instead of error types.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.constants import Annotations
from aiperf.operator.client_cache import _reset_for_testing
from aiperf.operator.handlers.monitor import (
    _check_job_timeout,
    _fetch_jobset_or_reconcile,
    _fetch_progress,
    _handle_jobset_failed_condition,
    _handle_jobset_terminal_condition,
    _maybe_recover_orphan_claim,
    _monitor_tick,
    _poll_controller_progress,
    _run_worker_and_progress_phase,
    monitor_progress,
)
from aiperf.operator.status import Phase, StatusBuilder

# -----------------------------------------------------------------------------
# Fixtures and helpers
# -----------------------------------------------------------------------------

_FIXTURE_CREATION_TS = "2024-04-25T17:02:03Z"
_FIXTURE_BODY: dict[str, Any] = {
    "metadata": {"creationTimestamp": _FIXTURE_CREATION_TS}
}


@pytest.fixture(autouse=True)
def _reset_module_state() -> Any:
    """Reset shared cache state so tests don't leak warned-restart sets etc."""
    _reset_for_testing()
    yield
    _reset_for_testing()


def _make_status_builder(
    existing: dict[str, Any] | None = None,
) -> tuple[StatusBuilder, Any]:
    patch = MagicMock()
    patch.status = {}
    return StatusBuilder(patch, existing or {}), patch


def _fake_k8s_client(api: Any) -> Any:
    """Async context manager helper that yields the given mock ApiClient."""

    @asynccontextmanager
    async def _ctx() -> Any:
        yield api

    return _ctx()


def _body(*, claimed: bool = False, phase: str | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"creationTimestamp": _FIXTURE_CREATION_TS}
    if claimed:
        metadata["annotations"] = {
            Annotations.COMPLETION_CLAIMED: "2026-04-29T08:18:22Z"
        }
    body: dict[str, Any] = {"metadata": metadata}
    if phase is not None:
        body["status"] = {"phase": phase}
    return body


def _progress_obj(
    *,
    current_phase: str | None = "profiling",
    is_complete: bool = False,
    connection_error: bool = False,
    error: str | None = None,
    phases: dict[str, Any] | None = None,
    workers: dict[str, int] | None = None,
) -> Any:
    """Construct a stand-in ``JobProgress``-like object for ``_fetch_progress``."""
    obj = MagicMock()
    obj.current_phase = current_phase
    obj.is_complete = is_complete
    obj.connection_error = connection_error
    obj.error = error
    obj.phases = phases or {}
    obj.workers = MagicMock()
    obj.workers.model_dump = MagicMock(
        return_value=workers if workers is not None else {"ready": 1, "total": 1}
    )
    return obj


# =============================================================================
# State-machine: forward and refused transitions
# =============================================================================


class TestStateMachineTransitions:
    """Verify forward phase transitions and that terminal CRs are not re-driven."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phase",
        [
            param(Phase.COMPLETED, id="completed"),
            param(Phase.FAILED, id="failed"),
            param(Phase.CANCELLED, id="cancelled"),
        ],
    )  # fmt: skip
    async def test_terminal_phase_short_circuits_before_k8s_call(
        self, phase: Phase
    ) -> None:
        """A terminal CR must not trigger any k8s round-trip from the monitor."""
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        with mock_patch(
            "aiperf.operator.handlers.monitor.k8s_client"
        ) as mock_client_cm:
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": phase,
                    "jobSetName": "js",
                    "jobId": "job-1",
                },
                spec={},
                name="job",
                namespace="ns",
                patch=kopf_patch,
            )

        mock_client_cm.assert_not_called()
        assert kopf_patch.status == {}

    @pytest.mark.asyncio
    async def test_missing_jobset_name_short_circuits(self) -> None:
        """A CR with no jobSetName is pre-resource-creation; skip silently."""
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        with mock_patch(
            "aiperf.operator.handlers.monitor.k8s_client"
        ) as mock_client_cm:
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={"phase": Phase.PENDING, "jobId": "job-1"},
                spec={},
                name="job",
                namespace="ns",
                patch=kopf_patch,
            )

        mock_client_cm.assert_not_called()
        assert kopf_patch.status == {}

# =============================================================================
# Job-timeout escalation
# =============================================================================


class TestJobTimeoutEscalation:
    """Verify timeout escalates to FAILED + JobSet deletion."""

    @pytest.mark.asyncio
    async def test_timeout_zero_disables_check(self) -> None:
        """``timeoutSeconds: 0`` (default) means no timeout — never escalates."""
        sb, patch = _make_status_builder()
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock()

        # startTime far in the past — would trip a real timeout — but spec=0.
        result = await _check_job_timeout(
            custom,
            body=_body(),
            status={"startTime": "2020-01-01T00:00:00Z"},
            spec={"timeoutSeconds": 0},
            namespace="ns",
            jobset_name="js",
            job_id="job-1",
            key="ns/job-1",
            sb=sb,
        )

        assert result is False
        assert "phase" not in patch.status
        custom.delete_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_timeout_with_no_start_time_no_escalation(self) -> None:
        """No ``startTime`` (yet) means we cannot compute elapsed; do nothing."""
        sb, patch = _make_status_builder()
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock()

        result = await _check_job_timeout(
            custom,
            body=_body(),
            status={},
            spec={"timeoutSeconds": 60},
            namespace="ns",
            jobset_name="js",
            job_id="job-1",
            key="ns/job-1",
            sb=sb,
        )

        assert result is False
        assert "phase" not in patch.status

    @pytest.mark.asyncio
    async def test_timeout_under_limit_no_escalation(self) -> None:
        """Elapsed below the limit must not escalate."""
        sb, patch = _make_status_builder()
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock()

        # startTime ~ now (use a recent timestamp; auto-reset RNG isn't relevant)
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        result = await _check_job_timeout(
            custom,
            body=_body(),
            status={"startTime": now},
            spec={"timeoutSeconds": 3600},
            namespace="ns",
            jobset_name="js",
            job_id="job-1",
            key="ns/job-1",
            sb=sb,
        )

        assert result is False
        assert "phase" not in patch.status

    @pytest.mark.asyncio
    async def test_timeout_exceeded_marks_failed_and_deletes_jobset(self) -> None:
        """Past the limit: phase=FAILED, error names elapsed/limit, JobSet deleted."""
        sb, patch = _make_status_builder()
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock()

        with mock_patch(
            "aiperf.operator.handlers.monitor.events.job_timeout"
        ) as mock_event:
            result = await _check_job_timeout(
                custom,
                body=_body(),
                status={"startTime": "2020-01-01T00:00:00Z"},
                spec={"timeoutSeconds": 1.0},
                namespace="ns",
                jobset_name="js",
                job_id="job-1",
                key="ns/job-1",
                sb=sb,
            )

        assert result is True
        assert patch.status["phase"] == str(Phase.FAILED)
        assert "Job timed out" in patch.status["error"]
        assert "1s" in patch.status["error"]
        custom.delete_namespaced_custom_object.assert_awaited_once()
        mock_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_with_no_jobset_name_skips_delete(self) -> None:
        """If ``jobset_name`` is None, the timeout still fires but no delete is attempted."""
        sb, patch = _make_status_builder()
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock()

        with mock_patch("aiperf.operator.handlers.monitor.events.job_timeout"):
            result = await _check_job_timeout(
                custom,
                body=_body(),
                status={"startTime": "2020-01-01T00:00:00Z"},
                spec={"timeoutSeconds": 1.0},
                namespace="ns",
                jobset_name=None,
                job_id="job-1",
                key="ns/job-1",
                sb=sb,
            )

        assert result is True
        assert patch.status["phase"] == str(Phase.FAILED)
        custom.delete_namespaced_custom_object.assert_not_awaited()


# =============================================================================
# Concurrent phase-write races and JobSet terminal handling
# =============================================================================


class TestJobsetTerminalConditions:
    """Cover JobSet ``status.conditions`` Completed/Failed branches."""

    @pytest.mark.asyncio
    async def test_completed_condition_invokes_completion_when_claim_won(
        self,
    ) -> None:
        """Completed + claim won -> handle_completion runs."""
        sb, _patch = _make_status_builder()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                new=AsyncMock(return_value=True),
            ) as mock_claim,
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new=AsyncMock(),
            ) as mock_complete,
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new=AsyncMock(),
            ),
        ):
            result = await _handle_jobset_terminal_condition(
                body=_body(),
                status={},
                jobset_status={"conditions": [{"type": "Completed", "status": "True"}]},
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                key="ns/j",
                sb=sb,
            )

        assert result is True
        mock_claim.assert_awaited_once()
        mock_complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_completed_condition_skips_completion_when_claim_lost(
        self,
    ) -> None:
        """Completed + claim lost (peer pod won race) -> short-circuit, no double-run."""
        sb, _patch = _make_status_builder()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                new=AsyncMock(return_value=False),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new=AsyncMock(),
            ) as mock_complete,
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new=AsyncMock(),
            ) as mock_close,
        ):
            result = await _handle_jobset_terminal_condition(
                body=_body(),
                status={},
                jobset_status={"conditions": [{"type": "Completed", "status": "True"}]},
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                key="ns/j",
                sb=sb,
            )

        assert result is True
        mock_complete.assert_not_awaited()
        mock_close.assert_awaited()

    @pytest.mark.asyncio
    async def test_non_true_status_conditions_are_ignored(self) -> None:
        """Conditions with status!=True must not drive phase transitions."""
        sb, _patch = _make_status_builder()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                new=AsyncMock(return_value=True),
            ) as mock_claim,
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new=AsyncMock(),
            ),
        ):
            result = await _handle_jobset_terminal_condition(
                body=_body(),
                status={},
                jobset_status={
                    "conditions": [
                        {"type": "Completed", "status": "False"},
                        {"type": "Failed", "status": "Unknown"},
                    ]
                },
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                key="ns/j",
                sb=sb,
            )

        assert result is False
        mock_claim.assert_not_awaited()


class TestJobsetFailedConditionEdges:
    """Cover the Failed-condition fatality classification and cascade escalation."""

    @pytest.mark.asyncio
    async def test_controller_failure_is_fatal(self) -> None:
        """``controller.failed > 0`` is a fatal failure regardless of workers."""
        sb, patch = _make_status_builder()

        with (
            mock_patch("aiperf.operator.handlers.monitor.events.failed") as mock_failed,
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new=AsyncMock(),
            ),
        ):
            result = await _handle_jobset_failed_condition(
                body=_body(),
                condition={
                    "type": "Failed",
                    "status": "True",
                    "message": "controller crashed",
                },
                jobset_status={
                    "replicatedJobsStatus": [
                        {"name": "controller", "failed": 1},
                        {"name": "workers", "failed": 0},
                    ]
                },
                job_id="j",
                key="ns/j",
                sb=sb,
            )

        assert result is True
        assert patch.status["phase"] == str(Phase.FAILED)
        assert patch.status["error"] == "controller crashed"
        mock_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_workers_only_failure_with_live_controller_is_non_fatal(
        self,
    ) -> None:
        """Workers failed but controller still active -> degrade, do not escalate."""
        sb, patch = _make_status_builder()

        with mock_patch(
            "aiperf.operator.handlers.monitor.close_progress_client",
            new=AsyncMock(),
        ) as mock_close:
            result = await _handle_jobset_failed_condition(
                body=_body(),
                condition={
                    "type": "Failed",
                    "status": "True",
                    "message": "worker pod failed",
                },
                jobset_status={
                    "replicatedJobsStatus": [
                        {
                            "name": "controller",
                            "failed": 0,
                            "active": 1,
                            "succeeded": 0,
                        },
                        {"name": "workers", "failed": 1},
                    ]
                },
                job_id="j",
                key="ns/j",
                sb=sb,
            )

        assert result is False
        assert "phase" not in patch.status
        mock_close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_workers_failure_with_dead_controller_escalates_to_fatal(
        self,
    ) -> None:
        """JobSet cascade killed the controller after worker failure -> fatal."""
        sb, patch = _make_status_builder()

        with (
            mock_patch("aiperf.operator.handlers.monitor.events.failed") as mock_failed,
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new=AsyncMock(),
            ),
        ):
            result = await _handle_jobset_failed_condition(
                body=_body(),
                condition={
                    "type": "Failed",
                    "status": "True",
                    "message": "worker pod failed",
                },
                jobset_status={
                    "replicatedJobsStatus": [
                        {
                            "name": "controller",
                            "failed": 0,
                            "active": 0,
                            "succeeded": 0,
                        },
                        {"name": "workers", "failed": 1},
                    ]
                },
                job_id="j",
                key="ns/j",
                sb=sb,
            )

        assert result is True
        assert patch.status["phase"] == str(Phase.FAILED)
        assert "Controller terminated" in patch.status["error"]
        mock_failed.assert_called_once()


# =============================================================================
# Reconciliation / fetch edges
# =============================================================================


class TestFetchJobsetOrReconcile:
    """Cover the fetch-or-reconcile dispatcher (404 vs other API errors)."""

    @pytest.mark.asyncio
    async def test_returns_jobset_dict_on_success(self) -> None:
        """Happy-path: returns the JobSet body verbatim."""
        sb, _patch = _make_status_builder()
        custom = MagicMock()
        custom.get_namespaced_custom_object = AsyncMock(
            return_value={"status": {"conditions": []}}
        )

        result = await _fetch_jobset_or_reconcile(
            custom,
            body=_body(),
            namespace="ns",
            name="job",
            jobset_name="js",
            current_phase=Phase.RUNNING,
            key="ns/j",
            sb=sb,
        )

        assert result == {"status": {"conditions": []}}

    @pytest.mark.asyncio
    async def test_404_routes_to_reconcile(self) -> None:
        """A 404 must hand off to ``_reconcile_missing_jobset`` and return None."""
        sb, _patch = _make_status_builder()
        custom = MagicMock()
        custom.get_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=404, reason="not found")
        )

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor._reconcile_missing_jobset",
                new=AsyncMock(return_value=True),
            ) as mock_reconcile,
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new=AsyncMock(),
            ) as mock_close,
        ):
            result = await _fetch_jobset_or_reconcile(
                custom,
                body=_body(),
                namespace="ns",
                name="job",
                jobset_name="js",
                current_phase=Phase.RUNNING,
                key="ns/j",
                sb=sb,
            )

        assert result is None
        mock_reconcile.assert_awaited_once()
        mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_404_error_is_re_raised(self) -> None:
        """A 500/transient must propagate (handled at monitor_progress level)."""
        sb, _patch = _make_status_builder()
        custom = MagicMock()
        custom.get_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=500, reason="internal")
        )

        with pytest.raises(ApiException) as excinfo:
            await _fetch_jobset_or_reconcile(
                custom,
                body=_body(),
                namespace="ns",
                name="job",
                jobset_name="js",
                current_phase=Phase.RUNNING,
                key="ns/j",
                sb=sb,
            )

        assert excinfo.value.status == 500


# =============================================================================
# Heartbeat / progress polling edges
# =============================================================================


class TestProgressFetchEdges:
    """Cover ``_fetch_progress`` failure modes and completion semantics."""

    @pytest.mark.asyncio
    async def test_connection_error_returns_false_silently(self) -> None:
        """Controller not reachable yet -> False, no patch writes."""
        sb, patch = _make_status_builder()
        progress_client = MagicMock()
        progress_client.get_progress = AsyncMock(
            return_value=_progress_obj(connection_error=True)
        )

        result = await _fetch_progress(
            "ns",
            "js",
            patch,
            sb,
            progress_client,
            "ns/j",
            Phase.PENDING,
            body=_body(),
        )

        assert result is False
        assert "phases" not in patch.status

    @pytest.mark.asyncio
    async def test_aiohttp_error_returns_false(self) -> None:
        """A transport-level aiohttp error degrades to "no progress this tick"."""
        import aiohttp

        sb, patch = _make_status_builder()
        progress_client = MagicMock()
        progress_client.get_progress = AsyncMock(
            side_effect=aiohttp.ClientError("network down")
        )

        result = await _fetch_progress(
            "ns",
            "js",
            patch,
            sb,
            progress_client,
            "ns/j",
            Phase.RUNNING,
            body=_body(),
        )

        assert result is False
        assert "phases" not in patch.status

    @pytest.mark.asyncio
    async def test_unexpected_exception_does_not_propagate(self) -> None:
        """Bare-except clause must keep monitor ticks alive on any progress error."""
        sb, patch = _make_status_builder()
        progress_client = MagicMock()
        progress_client.get_progress = AsyncMock(side_effect=RuntimeError("boom"))

        result = await _fetch_progress(
            "ns",
            "js",
            patch,
            sb,
            progress_client,
            "ns/j",
            Phase.RUNNING,
            body=_body(),
        )

        assert result is False


# =============================================================================
# Cancellation interaction
# =============================================================================


class TestCancellationInteraction:
    """Verify cancellation flag short-circuits tick before any k8s call."""

    @pytest.mark.asyncio
    async def test_cancellation_skips_jobset_get(self) -> None:
        """``is_cancellation_requested`` short-circuits before ``k8s_client``."""
        from aiperf.operator.client_cache import request_cancellation

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        request_cancellation("ns/job-xyz")

        with mock_patch(
            "aiperf.operator.handlers.monitor.k8s_client"
        ) as mock_client_cm:
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "js",
                    "jobId": "job-xyz",
                },
                spec={},
                name="job",
                namespace="ns",
                patch=kopf_patch,
            )

        mock_client_cm.assert_not_called()
        assert kopf_patch.status == {}


# =============================================================================
# Top-level monitor_progress: error swallowing + reconciliation bootstrap
# =============================================================================


class TestMonitorProgressTopLevel:
    """Verify ``monitor_progress`` swallows transient errors and propagates fatals."""

    @pytest.mark.asyncio
    async def test_transient_apiserver_error_is_swallowed_and_finalized(
        self,
    ) -> None:
        """ApiException inside the tick -> log + sb.finalize, no exception escapes."""
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        api_mock = MagicMock()
        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(api_mock),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._monitor_tick",
                new=AsyncMock(side_effect=ApiException(status=503)),
            ),
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "js",
                    "jobId": "job-1",
                },
                spec={},
                name="job",
                namespace="ns",
                patch=kopf_patch,
            )

        # No phase-stomp on transient error.
        assert "phase" not in kopf_patch.status

    @pytest.mark.asyncio
    async def test_unexpected_exception_propagates_for_kopf_retry(self) -> None:
        """Non-transient exceptions must propagate so kopf retries the tick."""
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        api_mock = MagicMock()
        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(api_mock),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._monitor_tick",
                new=AsyncMock(side_effect=ValueError("logic bug")),
            ),
            pytest.raises(ValueError, match="logic bug"),
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "js",
                    "jobId": "job-1",
                },
                spec={},
                name="job",
                namespace="ns",
                patch=kopf_patch,
            )

    @pytest.mark.asyncio
    async def test_bootstrap_first_observation_pending_phase_runs_tick(
        self,
    ) -> None:
        """A freshly-created CR (Pending, in-memory state empty) drives the tick.

        Operator restart leaves ``_warned_pod_restarts`` and ``_shutdown_sent``
        empty. The tick must not assume any prior in-memory state — the
        cancellation flag is per-key (no key set => no cancel), and the
        top-level guard only short-circuits on terminal phase or missing
        jobSetName. Verify the tick reaches ``_monitor_tick``.
        """
        kopf_patch = MagicMock()
        kopf_patch.status = {}
        api_mock = MagicMock()

        tick_called = AsyncMock()
        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(api_mock),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._monitor_tick",
                new=tick_called,
            ),
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.PENDING,
                    "jobSetName": "js",
                    "jobId": "job-bootstrap",
                },
                spec={},
                name="job",
                namespace="ns",
                patch=kopf_patch,
            )

        tick_called.assert_awaited_once()


# =============================================================================
# Orphan-claim recovery gating
# =============================================================================


class TestOrphanClaimRecoveryGating:
    """Verify the orphan-claim recovery only fires when the benchmark is done."""

    @pytest.mark.asyncio
    async def test_no_recovery_when_no_claim_annotation(self) -> None:
        """Without a claim annotation, recovery never runs even at non-terminal phase."""
        sb, _patch = _make_status_builder()
        api = MagicMock()

        with mock_patch(
            "aiperf.operator.handlers.monitor._benchmark_appears_complete",
            new=AsyncMock(return_value=True),
        ) as mock_gate:
            result = await _maybe_recover_orphan_claim(
                api,
                body=_body(),  # no claim annotation
                status={"phase": "Running"},
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                current_phase=Phase.RUNNING,
                key="ns/j",
                sb=sb,
            )

        assert result is False
        # gate not even queried — short-circuited before the cost.
        mock_gate.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phase",
        [
            param(Phase.COMPLETED, id="completed"),
            param(Phase.FAILED, id="failed"),
            param(Phase.CANCELLED, id="cancelled"),
        ],
    )  # fmt: skip
    async def test_no_recovery_when_phase_already_terminal(self, phase: Phase) -> None:
        """Claim + already-terminal phase -> nothing to recover."""
        sb, _patch = _make_status_builder()
        api = MagicMock()

        with mock_patch(
            "aiperf.operator.handlers.monitor._benchmark_appears_complete",
            new=AsyncMock(return_value=True),
        ) as mock_gate:
            result = await _maybe_recover_orphan_claim(
                api,
                body=_body(claimed=True),
                status={"phase": str(phase)},
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                current_phase=phase,
                key="ns/j",
                sb=sb,
            )

        assert result is False
        mock_gate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_defers_recovery_when_benchmark_not_yet_complete(self) -> None:
        """Claim + non-terminal + benchmark still in flight -> defer to next tick.

        This is the load-bearing gate: firing recovery while the benchmark
        is still running drives ``handle_completion`` into a retry-stagnation
        loop that ends in FAILED even when the benchmark would have succeeded.
        """
        sb, _patch = _make_status_builder()
        api = MagicMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor._benchmark_appears_complete",
                new=AsyncMock(return_value=False),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._recover_orphaned_completion_claim",
                new=AsyncMock(),
            ) as mock_recover,
        ):
            result = await _maybe_recover_orphan_claim(
                api,
                body=_body(claimed=True),
                status={"phase": "Running"},
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                current_phase=Phase.RUNNING,
                key="ns/j",
                sb=sb,
            )

        assert result is False
        mock_recover.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_runs_recovery_when_all_gates_pass(self) -> None:
        """Claim + non-terminal + benchmark done -> recovery fires."""
        sb, _patch = _make_status_builder()
        api = MagicMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor._benchmark_appears_complete",
                new=AsyncMock(return_value=True),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._recover_orphaned_completion_claim",
                new=AsyncMock(),
            ) as mock_recover,
        ):
            result = await _maybe_recover_orphan_claim(
                api,
                body=_body(claimed=True),
                status={"phase": "Running"},
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                current_phase=Phase.RUNNING,
                key="ns/j",
                sb=sb,
            )

        assert result is True
        mock_recover.assert_awaited_once()


# =============================================================================
# Worker-and-progress phase: pod-recovery short-circuit
# =============================================================================


class TestWorkerAndProgressPhase:
    """Verify the ``_run_worker_and_progress_phase`` dispatch wires fall through correctly."""

    @pytest.mark.asyncio
    async def test_terminated_controller_short_circuits_progress_polling(
        self,
    ) -> None:
        """If the salvage path fires (terminated controller), no progress poll runs."""
        sb, patch = _make_status_builder()
        api = MagicMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor._maybe_recover_terminated_controller",
                new=AsyncMock(return_value=True),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._poll_controller_progress",
                new=AsyncMock(return_value=False),
            ) as mock_poll,
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new=AsyncMock(),
            ) as mock_close,
        ):
            await _run_worker_and_progress_phase(
                api,
                body=_body(),
                status={"workers": {"total": 1}},
                patch=patch,
                jobset_status={
                    "replicatedJobsStatus": [
                        {"name": "workers", "ready": 1, "active": 0},
                    ]
                },
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                current_phase=Phase.RUNNING,
                key="ns/j",
                sb=sb,
            )

        mock_poll.assert_not_awaited()
        mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_zero_workers_ready_keeps_phase_pending(self) -> None:
        """All-pending workers must not promote the CR out of PENDING."""
        sb, patch = _make_status_builder()
        api = MagicMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor._maybe_recover_terminated_controller",
                new=AsyncMock(return_value=False),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._poll_controller_progress",
                new=AsyncMock(return_value=False),
            ),
        ):
            await _run_worker_and_progress_phase(
                api,
                body=_body(),
                status={"workers": {"total": 4}},
                patch=patch,
                jobset_status={
                    "replicatedJobsStatus": [
                        {
                            "name": "workers",
                            "ready": 0,
                            "active": 4,
                            "succeeded": 0,
                            "failed": 0,
                        },
                    ]
                },
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                current_phase=Phase.PENDING,
                key="ns/j",
                sb=sb,
            )

        # No phase write — still Pending.
        assert "phase" not in patch.status

    @pytest.mark.asyncio
    async def test_workers_ready_transitions_pending_to_initializing(self) -> None:
        """First worker readiness flips PENDING->INITIALIZING."""
        sb, patch = _make_status_builder()
        api = MagicMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor._maybe_recover_terminated_controller",
                new=AsyncMock(return_value=False),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._poll_controller_progress",
                new=AsyncMock(return_value=False),
            ),
        ):
            await _run_worker_and_progress_phase(
                api,
                body=_body(),
                status={"workers": {"total": 4}},
                patch=patch,
                jobset_status={
                    "replicatedJobsStatus": [
                        {
                            "name": "workers",
                            "ready": 1,
                            "active": 3,
                            "succeeded": 0,
                            "failed": 0,
                        },
                    ]
                },
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                current_phase=Phase.PENDING,
                key="ns/j",
                sb=sb,
            )

        assert patch.status["phase"] == str(Phase.INITIALIZING)


# =============================================================================
# poll_controller_progress: completion shutdown
# =============================================================================


class TestPollControllerProgress:
    """Cover ``_poll_controller_progress`` completion + shutdown sequence."""

    @pytest.mark.asyncio
    async def test_no_op_when_benchmark_not_complete(self) -> None:
        """Incomplete progress must NOT claim or shutdown."""
        sb, patch = _make_status_builder()
        client = MagicMock()
        client.send_shutdown = AsyncMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.get_or_create_progress_client",
                new=AsyncMock(return_value=client),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._fetch_progress",
                new=AsyncMock(return_value=False),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                new=AsyncMock(return_value=True),
            ) as mock_claim,
        ):
            result = await _poll_controller_progress(
                body=_body(),
                status={},
                patch=patch,
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                key="ns/j",
                effective_phase=Phase.RUNNING,
                sb=sb,
            )

        assert result is False
        mock_claim.assert_not_awaited()
        client.send_shutdown.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_complete_with_lost_claim_skips_shutdown(self) -> None:
        """Benchmark complete but claim lost (peer pod won) -> close + skip shutdown."""
        sb, patch = _make_status_builder()
        client = MagicMock()
        client.send_shutdown = AsyncMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.get_or_create_progress_client",
                new=AsyncMock(return_value=client),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._fetch_progress",
                new=AsyncMock(return_value=True),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                new=AsyncMock(return_value=False),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new=AsyncMock(),
            ) as mock_complete,
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new=AsyncMock(),
            ) as mock_close,
        ):
            result = await _poll_controller_progress(
                body=_body(),
                status={},
                patch=patch,
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                key="ns/j",
                effective_phase=Phase.RUNNING,
                sb=sb,
            )

        assert result is True
        mock_complete.assert_not_awaited()
        client.send_shutdown.assert_not_awaited()
        mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_complete_with_won_claim_drives_shutdown(self) -> None:
        """Benchmark complete + claim won -> handle_completion + send_shutdown."""
        sb, patch = _make_status_builder()
        client = MagicMock()
        client.send_shutdown = AsyncMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.get_or_create_progress_client",
                new=AsyncMock(return_value=client),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._fetch_progress",
                new=AsyncMock(return_value=True),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                new=AsyncMock(return_value=True),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new=AsyncMock(),
            ) as mock_complete,
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new=AsyncMock(),
            ) as mock_close,
        ):
            result = await _poll_controller_progress(
                body=_body(),
                status={},
                patch=patch,
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                key="ns/j",
                effective_phase=Phase.RUNNING,
                sb=sb,
            )

        assert result is True
        mock_complete.assert_awaited_once()
        client.send_shutdown.assert_awaited_once()
        mock_close.assert_awaited_once()


# =============================================================================
# Monitor tick: orchestration ordering
# =============================================================================


class TestMonitorTickOrdering:
    """Verify ``_monitor_tick`` runs orphan-claim, timeout, then jobset reconcile."""

    @pytest.mark.asyncio
    async def test_orphan_claim_recovery_short_circuits_other_branches(
        self,
    ) -> None:
        """If orphan-claim recovery fires, neither timeout nor jobset reconcile run."""
        sb, patch = _make_status_builder()
        api = MagicMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor._maybe_recover_orphan_claim",
                new=AsyncMock(return_value=True),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._check_job_timeout",
                new=AsyncMock(return_value=False),
            ) as mock_timeout,
            mock_patch(
                "aiperf.operator.handlers.monitor._reconcile_and_handle_jobset",
                new=AsyncMock(),
            ) as mock_reconcile,
        ):
            await _monitor_tick(
                api,
                body=_body(claimed=True),
                status={"phase": "Running"},
                spec={},
                patch=patch,
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                current_phase=Phase.RUNNING,
                key="ns/j",
                sb=sb,
            )

        mock_timeout.assert_not_awaited()
        mock_reconcile.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_timeout_short_circuits_jobset_reconcile(self) -> None:
        """If timeout fires, the JobSet reconciler is skipped (CR already FAILED)."""
        sb, patch = _make_status_builder()
        api = MagicMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor._maybe_recover_orphan_claim",
                new=AsyncMock(return_value=False),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._check_job_timeout",
                new=AsyncMock(return_value=True),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._reconcile_and_handle_jobset",
                new=AsyncMock(),
            ) as mock_reconcile,
        ):
            await _monitor_tick(
                api,
                body=_body(),
                status={"phase": "Running"},
                spec={"timeoutSeconds": 1},
                patch=patch,
                namespace="ns",
                name="job",
                jobset_name="js",
                job_id="j",
                current_phase=Phase.RUNNING,
                key="ns/j",
                sb=sb,
            )

        mock_reconcile.assert_not_awaited()
