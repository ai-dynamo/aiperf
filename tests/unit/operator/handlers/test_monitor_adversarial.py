# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial monitor/progress tests for Kubernetes operator handlers.

Focuses on production-hostile monitor-tick contracts:
- Missing JobSet reconciliation must not clobber terminal or claimed CRs.
- Transient API failures in best-effort pod scans degrade to no-op evidence.
- Phase transitions must clear stale stage labels and gate Complete on results.
- Controller SystemState must propagate even before a CreditPhase exists.
- Pod-restart shortcuts must emit only observable warning events.
- Cancellation checks must stop late salvage side effects after awaited fetches.
- Retryable cleanup failures must preserve the retry delay contract.

Out of scope: happy-path completion fetch/storage; see sibling files
``test_completion_handler.py`` and ``test_monitor_state_machine_edges.py``.
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

from aiperf.common.enums import SystemState
from aiperf.kubernetes.constants import Annotations, Containers, JobSetLabels
from aiperf.kubernetes.cr_refs import AIPERF_JOB_API_VERSION
from aiperf.kubernetes.phase import Phase
from aiperf.operator.client_cache import _reset_for_testing, request_cancellation
from aiperf.operator.handlers.monitor import (
    _benchmark_appears_complete,
    _delete_jobset_or_retry,
    _fetch_jobset_or_reconcile,
    _handle_jobset_failed_condition,
    _maybe_recover_terminated_controller,
    _reconcile_missing_jobset,
)
from aiperf.operator.handlers.pod_restarts import handle_pod_restart
from aiperf.operator.status import ConditionType, StatusBuilder

# =============================================================================
# Helpers
# =============================================================================

_FIXTURE_CREATION_TS = "2026-05-18T09:41:03Z"


@pytest.fixture(autouse=True)
def _reset_operator_caches() -> None:
    """Reset monitor/client-cache singleton state across adversarial cases."""
    _reset_for_testing()
    yield
    _reset_for_testing()


def _status_builder(
    existing_status: dict[str, Any] | None = None,
) -> tuple[StatusBuilder, MagicMock]:
    """Return a StatusBuilder backed by a kopf-like patch mock."""
    patch = MagicMock()
    patch.status = {}
    return StatusBuilder(patch, existing_status or {}), patch


_FIXTURE_UID = "6d0a9f1b-51c4-4a7e-8e2c-b3f0d7c6a915"
_JOBSET_NAME = "llama3-8b-throughput-js"
_JOBSET_UID = "1f8e2b47-3a6d-4c19-9b05-7e4d8a1c3f62"


def _body(
    *,
    claimed: bool = False,
    generation: int | None = 7,
    uid: str | None = None,
) -> dict[str, Any]:
    """Build a realistic AIPerfJob body for monitor helper tests.

    ``uid`` is opt-in: the identity fences are inert without it, which keeps
    helpers whose contract predates fencing on their original code path. Tests
    that exercise a fenced cluster round-trip pass ``uid=_FIXTURE_UID`` and
    install ``_fenced_jobset_api``.
    """
    metadata: dict[str, Any] = {
        "name": "llama3-8b-throughput",
        "namespace": "bench-prod",
        "creationTimestamp": _FIXTURE_CREATION_TS,
    }
    if uid is not None:
        metadata["uid"] = uid
    if generation is not None:
        metadata["generation"] = generation
    if claimed:
        metadata["annotations"] = {
            Annotations.COMPLETION_CLAIMED: "2026-05-18T09:42:11Z"
        }
    return {"kind": "AIPerfJob", "metadata": metadata}


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
                    "name": "llama3-8b-throughput",
                    "uid": _FIXTURE_UID,
                    "controller": True,
                }
            ],
        }
    }


@contextlib.contextmanager
def _fenced_jobset_api(custom: MagicMock) -> Iterator[None]:
    """Route the shared identity helper's cluster reads at one fake API object.

    ``_delete_jobset_or_retry`` no longer uses the ``CustomObjectsApi`` handed
    to it; it delegates to ``_job_identity.delete_owned_aiperfjob_jobset``,
    which opens its own short-lived client, proves controller ownership, and
    only then issues a UID-preconditioned delete. Patching ``k8s_client`` here
    is what keeps the test off a real apiserver -- without it a machine with a
    live kubeconfig answers the ownership read for real.
    """
    custom.get_namespaced_custom_object = AsyncMock(return_value=_owned_jobset())

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
        yield


def _progress(
    *,
    current_phase: str | None = "profiling",
    system_state: SystemState = SystemState.PROFILING,
    is_complete: bool = False,
    error: str | None = None,
) -> MagicMock:
    """Build a JobProgress-like object returned by the controller API."""
    progress = MagicMock()
    progress.current_phase = current_phase
    progress.system_state = system_state
    progress.is_complete = is_complete
    progress.connection_error = False
    progress.error = error
    progress.phases = {}
    progress.workers = MagicMock()
    progress.workers.model_dump = MagicMock(
        return_value={"ready": 4, "total": 4, "degraded": 0}
    )
    return progress


def _controller_pod(
    *,
    exit_code: int = 137,
    reason: str = "OOMKilled",
) -> SimpleNamespace:
    """Build a controller pod whose control-plane container has terminated."""
    terminated = SimpleNamespace(exit_code=exit_code, reason=reason)
    controller = SimpleNamespace(
        name=Containers.CONTROL_PLANE,
        state=SimpleNamespace(terminated=terminated),
    )
    sidecar = SimpleNamespace(
        name=Containers.RESULTS_SIDECAR,
        state=SimpleNamespace(terminated=None),
    )
    return SimpleNamespace(
        metadata=SimpleNamespace(name="llama3-8b-throughput-controller-0"),
        status=SimpleNamespace(container_statuses=[controller, sidecar]),
    )


# =============================================================================
# Missing JobSet and API-error reconciliation
# =============================================================================


class TestMissingJobSetReconciliation:
    """Missing JobSet races must not overwrite successful completion paths."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fresh_phase",
        [
            param(Phase.COMPLETED, id="fresh-completed"),
            param(Phase.FAILED, id="fresh-failed"),
            param(Phase.CANCELLED, id="fresh-cancelled"),
        ],
    )  # fmt: skip
    async def test_reconcile_missing_jobset_fresh_terminal_phase_skips_failed_stamp(
        self, fresh_phase: Phase
    ) -> None:
        """404 + fresh terminal CR means another path already converged."""
        sb, patch = _status_builder()
        custom = MagicMock()
        custom.get_namespaced_custom_object = AsyncMock(
            return_value={"status": {"phase": str(fresh_phase)}}
        )

        result = await _reconcile_missing_jobset(
            custom,
            body=_body(),
            namespace="bench-prod",
            name="llama3-8b-throughput",
            jobset_name="llama3-8b-throughput-js",
            current_phase=Phase.RUNNING,
            sb=sb,
        )

        assert result is True
        assert "phase" not in patch.status

    @pytest.mark.asyncio
    async def test_fetch_jobset_404_with_claim_closes_progress_without_failed_stamp(
        self,
    ) -> None:
        """404 + completion claim is success evidence, not a Failed signal."""
        sb, patch = _status_builder()
        custom = MagicMock()
        custom.get_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=404, reason="not found")
        )

        with mock_patch(
            "aiperf.operator.handlers.monitor.close_progress_client",
            new=AsyncMock(),
        ) as close_progress:
            result = await _fetch_jobset_or_reconcile(
                custom,
                body=_body(claimed=True),
                namespace="bench-prod",
                name="llama3-8b-throughput",
                jobset_name="llama3-8b-throughput-js",
                current_phase=Phase.RUNNING,
                key="bench-prod/llama3-8b-throughput",
                sb=sb,
            )

        assert result is None
        assert "phase" not in patch.status
        close_progress.assert_awaited_once_with("bench-prod/llama3-8b-throughput")

    @pytest.mark.asyncio
    async def test_benchmark_appears_complete_api_error_returns_false(self) -> None:
        """Progress and pod-scan API failures are not completion evidence."""
        api = MagicMock()
        progress_client = MagicMock()
        progress_client.get_progress = AsyncMock(side_effect=OSError("route reset"))
        core_api = MagicMock()
        core_api.list_namespaced_pod = AsyncMock(
            side_effect=ApiException(status=503, reason="apiserver")
        )

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.get_or_create_progress_client",
                new=AsyncMock(return_value=progress_client),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CoreV1Api",
                return_value=core_api,
            ),
        ):
            result = await _benchmark_appears_complete(
                api=api,
                namespace="bench-prod",
                jobset_name="llama3-8b-throughput-js",
                key="bench-prod/llama3-8b-throughput",
            )

        assert result is False


# =============================================================================
# Status phase, subPhase, and ResultsAvailable contracts
# =============================================================================


class TestStatusProgressContracts:
    """Phase/status fields are the public Kubernetes contract."""

    @pytest.mark.parametrize(
        "terminal_phase",
        [
            param(Phase.COMPLETED, id="completed-clears-stage-labels"),
            param(Phase.FAILED, id="failed-clears-stage-labels"),
            param(Phase.CANCELLED, id="cancelled-clears-stage-labels"),
        ],
    )  # fmt: skip
    def test_status_builder_terminal_phase_clears_current_phase_and_subphase(
        self, terminal_phase: Phase
    ) -> None:
        """Terminal status must not leave stale profiling/processing labels."""
        sb, patch = _status_builder(
            {
                "currentPhase": "profiling",
                "subPhase": "processing",
            }
        )

        sb.set_phase(terminal_phase)

        assert patch.status["phase"] == str(terminal_phase)
        assert patch.status["currentPhase"] is None
        assert patch.status["subPhase"] is None

    def test_status_builder_completed_without_results_available_does_not_set_complete(
        self,
    ) -> None:
        """Completed phase alone is insufficient for batchv1 Complete=True."""
        sb, patch = _status_builder()

        sb.set_phase(Phase.COMPLETED)
        sb.finalize()

        assert "conditions" not in patch.status

    def test_status_builder_completed_with_results_available_sets_complete_true(
        self,
    ) -> None:
        """Complete=True is gated on ResultsAvailable=True."""
        sb, patch = _status_builder()

        sb.conditions.set_true(
            ConditionType.RESULTS_AVAILABLE,
            "ResultsStored",
            "Results for llama3-8b-throughput are stored",
        )
        sb.set_phase(Phase.COMPLETED)
        sb.finalize()

        conditions = {cond["type"]: cond for cond in patch.status["conditions"]}
        assert conditions["Complete"]["status"] == "True"
        assert conditions["Failed"]["status"] == "False"

    @pytest.mark.asyncio
    async def test_jobset_failed_condition_clears_stage_labels_on_fatal_failure(
        self,
    ) -> None:
        """Fatal JobSet failure must clear currentPhase/subPhase in the same patch."""
        sb, patch = _status_builder(
            {"currentPhase": "profiling", "subPhase": "profiling"}
        )

        with (
            mock_patch("aiperf.operator.handlers.monitor.events.failed"),
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
                    "message": "controller pod OOMKilled",
                },
                jobset_status={
                    "replicatedJobsStatus": [
                        {"name": "controller", "failed": 1},
                        {"name": "workers", "failed": 0},
                    ]
                },
                job_id="llama3-8b-throughput",
                key="bench-prod/llama3-8b-throughput",
                sb=sb,
            )

        assert result is True
        assert patch.status["phase"] == str(Phase.FAILED)
        assert patch.status["currentPhase"] is None
        assert patch.status["subPhase"] is None


# =============================================================================
# Pod-restart shortcut events
# =============================================================================


class TestPodRestartShortcutEvents:
    """Pod watch shortcuts report restart spikes without monitor polling."""

    @pytest.mark.asyncio
    async def test_pod_restart_at_threshold_emits_event_with_jobset_identity(
        self,
    ) -> None:
        """restartCount == threshold is actionable and emits exactly one event."""
        meta = {
            "name": "llama3-8b-throughput-controller-0",
            "namespace": "bench-prod",
            "labels": {JobSetLabels.JOBSET_NAME: "llama3-8b-throughput-js"},
        }
        job_body = _body()

        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
                return_value=job_body,
            ) as lookup_job,
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as emit_restart,
        ):
            await handle_pod_restart(
                old=[{"name": "controller", "restartCount": 2}],
                new=[
                    {
                        "name": "controller",
                        "restartCount": 3,
                        "lastState": {"terminated": {"reason": "OOMKilled"}},
                    }
                ],
                body={"metadata": meta},
                meta=meta,
                namespace="bench-prod",
                name="llama3-8b-throughput-controller-0",
                threshold=3,
            )

        # The lookup now walks Pod -> Job -> JobSet -> AIPerfJob by immutable
        # controller-owner UID, so it needs the observed Pod body -- the JobSet
        # label alone only selects a candidate.
        lookup_job.assert_called_once_with(
            "bench-prod", "llama3-8b-throughput-js", {"metadata": meta}
        )
        emit_restart.assert_called_once()
        assert emit_restart.call_args.args == (
            job_body,
            "llama3-8b-throughput-controller-0",
            3,
            "OOMKilled",
        )

    @pytest.mark.asyncio
    async def test_pod_restart_missing_jobset_label_skips_lookup_and_event(
        self,
    ) -> None:
        """Pods outside JobSet ownership must not trigger AIPerfJob lookups."""
        meta = {"name": "unrelated-controller-0", "namespace": "bench-prod"}

        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body"
            ) as lookup_job,
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as emit_restart,
        ):
            await handle_pod_restart(
                old=[],
                new=[{"name": "controller", "restartCount": 5}],
                body={"metadata": meta},
                meta=meta,
                namespace="bench-prod",
                name="unrelated-controller-0",
                threshold=3,
            )

        lookup_job.assert_not_called()
        emit_restart.assert_not_called()


# =============================================================================
# Cancellation and retry delay semantics
# =============================================================================


class TestCancellationAndRetrySemantics:
    """Late cancellation and cleanup retries must preserve operator safety."""

    @pytest.mark.asyncio
    async def test_terminated_controller_salvage_honors_cancellation_before_claim(
        self,
    ) -> None:
        """Cancellation prevents the recovery path from claiming or fetching."""
        sb, patch = _status_builder()
        key = "bench-prod/llama3-8b-throughput"
        request_cancellation(key)
        result = SimpleNamespace(downloaded=True, checkpoints=[])

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor._get_controller_pod",
                new=AsyncMock(return_value=_controller_pod()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.fetch_results_with_retry",
                new=AsyncMock(return_value=result),
            ) as fetch_results,
            mock_patch(
                "aiperf.operator.handlers.monitor._recover_result_from_disk",
                return_value=result,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                new=AsyncMock(return_value=True),
            ) as claim_completion,
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new=AsyncMock(),
            ) as handle_completion,
            mock_patch(
                "aiperf.operator.handlers.monitor._recover_from_partial_checkpoints",
                new=AsyncMock(),
            ) as recover_checkpoints,
            mock_patch(
                "aiperf.operator.handlers.monitor._fail_unrecoverable_controller",
                new=AsyncMock(),
            ) as fail_unrecoverable,
        ):
            handled = await _maybe_recover_terminated_controller(
                MagicMock(),
                _body(),
                "bench-prod",
                "llama3-8b-throughput-js",
                "llama3-8b-throughput",
                status={"phase": str(Phase.RUNNING)},
                sb=sb,
                key=key,
                name="llama3-8b-throughput",
            )

        assert handled is True
        fetch_results.assert_not_awaited()
        claim_completion.assert_not_awaited()
        handle_completion.assert_not_awaited()
        recover_checkpoints.assert_not_awaited()
        fail_unrecoverable.assert_not_awaited()
        assert patch.status == {}

    @pytest.mark.asyncio
    async def test_delete_jobset_non_404_raises_temporary_error_with_15s_delay(
        self,
    ) -> None:
        """Cleanup failures must ask kopf to retry soon, not fail permanently."""
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=503, reason="apiserver unavailable")
        )

        with _fenced_jobset_api(custom), pytest.raises(kopf.TemporaryError) as excinfo:
            await _delete_jobset_or_retry(
                custom,
                "bench-prod",
                _JOBSET_NAME,
                body=_body(uid=_FIXTURE_UID),
                context="timeout",
            )

        assert excinfo.value.delay == 15
        assert f"bench-prod/{_JOBSET_NAME}" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_delete_jobset_404_is_idempotent_success(self) -> None:
        """A deleted JobSet means cleanup already converged; no retry needed."""
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=404, reason="not found")
        )

        with _fenced_jobset_api(custom):
            deleted = await _delete_jobset_or_retry(
                custom,
                "bench-prod",
                _JOBSET_NAME,
                body=_body(uid=_FIXTURE_UID),
                context="success cleanup",
            )

        # A converged delete must report success, or every caller that gates
        # its terminal status patch on the return value abandons the commit.
        assert deleted is True
        custom.delete_namespaced_custom_object.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_jobset_without_parent_uid_fails_closed(self) -> None:
        """An unfenced caller must not delete a JobSet it cannot prove it owns."""
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock()

        with _fenced_jobset_api(custom):
            deleted = await _delete_jobset_or_retry(
                custom,
                "bench-prod",
                _JOBSET_NAME,
                body=_body(),
                context="timeout",
            )

        assert deleted is False
        custom.delete_namespaced_custom_object.assert_not_awaited()
