# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pure helpers in ``aiperf.operator.handlers.monitor``.

The end-to-end ``monitor_progress`` flow is exercised by ``test_main.py`` and
``test_cancellation.py``. This file targets the small helper functions
(``_classify_jobset_failure``, ``_should_poll_progress``,
``_handle_kueue_suspension``, ``_container_status_by_name``,
``_get_terminated_controller_info``, ``_update_worker_counts``) which have
no direct unit tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import kopf
import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.common.enums.lifecycle_enums import SystemState
from aiperf.operator.handlers.monitor import (
    _check_job_timeout,
    _classify_jobset_failure,
    _container_status_by_name,
    _fail_on_fatal_pod_waiting_reason,
    _fail_unrecoverable_controller,
    _fatal_pod_waiting_message,
    _get_fatal_pod_waiting_reason,
    _get_terminated_controller_info,
    _handle_kueue_suspension,
    _maybe_recover_exported_results_from_sidecar,
    _recover_from_live_status,
    _recover_from_partial_checkpoints,
    _should_poll_progress,
    _update_worker_counts,
)
from aiperf.operator.status import Phase, StatusBuilder


def _make_status_builder() -> tuple[StatusBuilder, Any]:
    """Return a StatusBuilder wrapping a MagicMock-backed patch with .status={}."""
    patch = MagicMock()
    patch.status = {}
    return StatusBuilder(patch, {}), patch


class TestClassifyJobsetFailure:
    """Tests for ``_classify_jobset_failure``."""

    @pytest.mark.parametrize(
        "replicated,expected",
        [
            param(
                [{"name": "controller", "failed": 1}, {"name": "workers", "failed": 0}],
                (True, "controller"),
                id="controller_failed",
            ),
            param(
                [{"name": "controller", "failed": 0}, {"name": "workers", "failed": 2}],
                (False, "workers"),
                id="workers_only",
            ),
            param(
                [{"name": "controller", "failed": 0}, {"name": "workers", "failed": 0}],
                (True, None),
                id="no_identified_failure",
            ),
            param([], (True, None), id="empty_status"),
        ],
    )  # fmt: skip
    def test_classifies_fatal_vs_non_fatal(
        self, replicated: list[dict[str, Any]], expected: tuple[bool, str | None]
    ) -> None:
        """Verify fatal/non-fatal classification per replicated-job role."""
        jobset_status = {"replicatedJobsStatus": replicated}
        assert _classify_jobset_failure(jobset_status) == expected


class TestShouldPollProgress:
    """Tests for ``_should_poll_progress``."""

    @pytest.mark.parametrize(
        "phase,succeeded,total,expected",
        [
            param(Phase.PENDING, 0, 0, True, id="pending_always"),
            param(Phase.INITIALIZING, 0, 0, True, id="initializing_always"),
            param(Phase.RUNNING, 0, 0, True, id="running_always"),
            param(Phase.QUEUED, 0, 2, False, id="queued_no_progress"),
            param(Phase.QUEUED, 2, 2, True, id="queued_all_succeeded"),
            param(Phase.COMPLETED, 1, 0, True, id="completed_short_circuit_positive"),
            param(Phase.COMPLETED, 0, 1, False, id="completed_no_succeeded"),
        ],
    )  # fmt: skip
    def test_decision_matrix(
        self, phase: Phase, succeeded: int, total: int, expected: bool
    ) -> None:
        """Verify the poll-decision truth table."""
        assert _should_poll_progress(phase, succeeded, total) is expected


class TestHandleKueueSuspension:
    """Tests for ``_handle_kueue_suspension``."""

    def test_detects_suspension_and_sets_queued_phase(self) -> None:
        """Verify a kueue-managed suspended JobSet is marked QUEUED."""
        sb, patch = _make_status_builder()
        jobset = {
            "metadata": {"labels": {"kueue.x-k8s.io/queue-name": "default"}},
            "spec": {"suspend": True},
        }

        result = _handle_kueue_suspension(
            jobset=jobset, current_phase=Phase.PENDING, sb=sb
        )

        assert result is True
        assert patch.status["phase"] == str(Phase.QUEUED)

    def test_ignores_suspended_but_not_kueue_managed(self) -> None:
        """Verify a non-kueue-managed suspension is not treated as QUEUED."""
        sb, patch = _make_status_builder()
        jobset = {
            "metadata": {"labels": {}},
            "spec": {"suspend": True},
        }

        result = _handle_kueue_suspension(
            jobset=jobset, current_phase=Phase.PENDING, sb=sb
        )

        assert result is False
        assert "phase" not in patch.status

    def test_ignores_kueue_managed_but_not_suspended(self) -> None:
        """Verify a running kueue-managed JobSet is not marked QUEUED."""
        sb, _patch = _make_status_builder()
        jobset = {
            "metadata": {"labels": {"kueue.x-k8s.io/queue-name": "default"}},
            "spec": {"suspend": False},
        }

        result = _handle_kueue_suspension(
            jobset=jobset, current_phase=Phase.PENDING, sb=sb
        )

        assert result is False

    def test_ignores_suspension_when_phase_is_running(self) -> None:
        """Verify post-admission suspension is not demoted to QUEUED."""
        sb, _patch = _make_status_builder()
        jobset = {
            "metadata": {"labels": {"kueue.x-k8s.io/queue-name": "default"}},
            "spec": {"suspend": True},
        }

        result = _handle_kueue_suspension(
            jobset=jobset, current_phase=Phase.RUNNING, sb=sb
        )

        assert result is False


class TestFatalPodWaitingReason:
    """Tests for fatal container waiting-state detection."""

    def _pod(
        self,
        *,
        pod_name: str = "aiperf-bench-controller-0",
        container_name: str = "control-plane",
        reason: str = "ImagePullBackOff",
        message: str = "Back-off pulling image 'missing:latest'",
    ) -> SimpleNamespace:
        waiting = SimpleNamespace(reason=reason, message=message)
        state = SimpleNamespace(waiting=waiting)
        status = SimpleNamespace(name=container_name, state=state)
        return SimpleNamespace(
            metadata=SimpleNamespace(name=pod_name),
            status=SimpleNamespace(container_statuses=[status]),
        )

    @pytest.mark.parametrize(
        "reason",
        [
            param("ErrImagePull", id="err_image_pull"),
            param("ImagePullBackOff", id="image_pull_backoff"),
            param("CreateContainerConfigError", id="create_container_config_error"),
        ],
    )  # fmt: skip
    def test_detects_fatal_waiting_reasons(self, reason: str) -> None:
        """Fatal pod startup waiting reasons surface the offending container."""
        waiting = _get_fatal_pod_waiting_reason([self._pod(reason=reason)])

        assert waiting is not None
        assert waiting.pod_name == "aiperf-bench-controller-0"
        assert waiting.container_name == "control-plane"
        assert waiting.reason == reason
        assert waiting.message == "Back-off pulling image 'missing:latest'"

    def test_ignores_container_creating(self) -> None:
        """Normal ContainerCreating startup is not fatal."""
        assert (
            _get_fatal_pod_waiting_reason(
                [self._pod(reason="ContainerCreating", message="creating container")]
            )
            is None
        )

    def test_message_names_jobset_reason_and_image_detail(self) -> None:
        """Formatted operator error includes all user-actionable context."""
        waiting = _get_fatal_pod_waiting_reason([self._pod()])
        assert waiting is not None

        message = _fatal_pod_waiting_message("aiperf-bench", "aiperf-bench-js", waiting)

        assert "aiperf-bench" in message
        assert "aiperf-bench-js" in message
        assert "ImagePullBackOff" in message
        assert "missing:latest" in message
        assert "Back-off pulling image" in message


class TestFailOnFatalPodWaitingReason:
    """Tests for terminalizing active jobs on fatal pod waiting states."""

    @pytest.mark.asyncio
    async def test_marks_failed_and_deletes_jobset(self) -> None:
        """Fatal image-pull backoff deletes the JobSet and marks the CR Failed."""
        sb, status_patch = _make_status_builder()
        pod = SimpleNamespace(
            metadata=SimpleNamespace(name="aiperf-bench-worker-0"),
            status=SimpleNamespace(
                container_statuses=[
                    SimpleNamespace(
                        name="worker",
                        state=SimpleNamespace(
                            waiting=SimpleNamespace(
                                reason="ImagePullBackOff",
                                message="Back-off pulling image 'missing:latest'",
                            )
                        ),
                    )
                ]
            ),
        )
        core = MagicMock()
        core.list_namespaced_pod = AsyncMock(return_value=SimpleNamespace(items=[pod]))
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock()
        api = MagicMock()

        with (
            patch(
                "aiperf.operator.handlers.monitor.client.CoreV1Api",
                return_value=core,
            ),
            patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=custom,
            ),
        ):
            handled = await _fail_on_fatal_pod_waiting_reason(
                api,
                body={"kind": "AIPerfJob", "metadata": {"name": "aiperf-bench"}},
                namespace="ns",
                name="aiperf-bench",
                jobset_name="aiperf-bench-js",
                job_id="aiperf-bench",
                key="ns/aiperf-bench",
                sb=sb,
            )

        assert handled is True
        assert status_patch.status["phase"] == str(Phase.FAILED)
        assert "ImagePullBackOff" in status_patch.status["error"]
        assert status_patch.status["completionTime"]
        failed_condition = next(
            condition
            for condition in status_patch.status["conditions"]
            if condition["type"] == "Failed"
        )
        assert failed_condition["status"] == "True"
        custom.delete_namespaced_custom_object.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pod_inspection_type_error_degrades_to_no_fatal_reason(self) -> None:
        """A malformed pod-list client must not abort the monitor tick."""
        sb, patch_status = _make_status_builder()
        core = MagicMock()
        core.list_namespaced_pod = MagicMock(return_value=MagicMock())

        with patch(
            "aiperf.operator.handlers.monitor.client.CoreV1Api",
            return_value=core,
        ):
            handled = await _fail_on_fatal_pod_waiting_reason(
                MagicMock(),
                body={"kind": "AIPerfJob", "metadata": {"name": "aiperf-bench"}},
                namespace="ns",
                name="aiperf-bench",
                jobset_name="aiperf-bench-js",
                job_id="aiperf-bench",
                key="ns/aiperf-bench",
                sb=sb,
            )

        assert handled is False
        assert patch_status.status == {}


class TestContainerStatusByName:
    """Tests for ``_container_status_by_name``."""

    def test_returns_matching_status(self) -> None:
        """Verify returns the first container-status matching name."""
        a = SimpleNamespace(name="controller", restart_count=2)
        b = SimpleNamespace(name="sidecar", restart_count=0)

        assert _container_status_by_name([a, b], "controller") is a
        assert _container_status_by_name([a, b], "sidecar") is b

    def test_returns_none_when_not_found(self) -> None:
        """Verify returns None when no match exists."""
        a = SimpleNamespace(name="controller")
        assert _container_status_by_name([a], "missing") is None
        assert _container_status_by_name([], "anything") is None


class TestGetTerminatedControllerInfo:
    """Tests for ``_get_terminated_controller_info``."""

    def test_returns_none_when_status_missing(self) -> None:
        """Verify returns None when the pod has no container statuses."""
        pod = SimpleNamespace(status=SimpleNamespace(container_statuses=None))
        assert _get_terminated_controller_info(pod) is None

    def test_returns_none_when_controller_missing(self) -> None:
        """Verify returns None when the controller container status is absent."""
        sidecar = SimpleNamespace(name="results-sidecar", state=None)
        pod = SimpleNamespace(status=SimpleNamespace(container_statuses=[sidecar]))
        assert _get_terminated_controller_info(pod) is None

    def test_returns_none_when_controller_still_running(self) -> None:
        """Verify returns None when the controller is not terminated."""
        controller = SimpleNamespace(
            name="control-plane", state=SimpleNamespace(terminated=None)
        )
        sidecar = SimpleNamespace(name="results-sidecar", state=None)
        pod = SimpleNamespace(
            status=SimpleNamespace(container_statuses=[controller, sidecar])
        )
        assert _get_terminated_controller_info(pod) is None

    def test_returns_none_on_zero_exit(self) -> None:
        """Verify clean exits (exit_code==0) do not trigger recovery."""
        terminated = SimpleNamespace(exit_code=0, reason="Completed")
        controller = SimpleNamespace(
            name="control-plane", state=SimpleNamespace(terminated=terminated)
        )
        sidecar = SimpleNamespace(name="results-sidecar", state=None)
        pod = SimpleNamespace(
            status=SimpleNamespace(container_statuses=[controller, sidecar])
        )
        assert _get_terminated_controller_info(pod) is None

    def test_returns_exit_info_on_nonzero_exit(self) -> None:
        """Verify returns (exit_code, reason) when the controller crashed."""
        terminated = SimpleNamespace(exit_code=137, reason="OOMKilled")
        controller = SimpleNamespace(
            name="control-plane", state=SimpleNamespace(terminated=terminated)
        )
        sidecar = SimpleNamespace(name="results-sidecar", state=None)
        pod = SimpleNamespace(
            status=SimpleNamespace(container_statuses=[controller, sidecar])
        )
        assert _get_terminated_controller_info(pod) == (137, "OOMKilled")

    def test_returns_exit_info_from_restarted_controller_last_state(self) -> None:
        terminated = SimpleNamespace(exit_code=137, reason="Error")
        controller = SimpleNamespace(
            name="control-plane",
            restart_count=1,
            state=SimpleNamespace(terminated=None),
            last_state=SimpleNamespace(terminated=terminated),
        )
        sidecar = SimpleNamespace(name="results-sidecar", state=None)
        pod = SimpleNamespace(
            status=SimpleNamespace(container_statuses=[controller, sidecar])
        )

        assert _get_terminated_controller_info(pod) == (137, "Error")


class TestUpdateWorkerCounts:
    """Tests for ``_update_worker_counts``."""

    def test_uses_crd_total_when_present(self) -> None:
        """Verify the CRD status total is preferred over JobSet-derived total."""
        sb, _patch = _make_status_builder()
        status = {"workers": {"total": 16}}
        jobset_status = {
            "replicatedJobsStatus": [
                {
                    "name": "workers",
                    "ready": 10,
                    "succeeded": 0,
                    "active": 6,
                    "failed": 0,
                    "suspended": 0,
                },
            ],
        }

        ready, succeeded, total = _update_worker_counts(
            status=status, jobset_status=jobset_status, sb=sb
        )

        assert (ready, succeeded, total) == (10, 0, 16)

    def test_derives_total_from_jobset_when_crd_missing(self) -> None:
        """Verify total is summed from JobSet fields when CRD total is 0."""
        sb, _patch = _make_status_builder()
        status = {"workers": {"total": 0}}
        jobset_status = {
            "replicatedJobsStatus": [
                {
                    "name": "workers",
                    "ready": 3,
                    "active": 2,
                    "succeeded": 1,
                    "failed": 1,
                    "suspended": 0,
                },
            ],
        }

        ready, succeeded, total = _update_worker_counts(
            status=status, jobset_status=jobset_status, sb=sb
        )

        assert (ready, succeeded, total) == (3, 1, 7)

    def test_fallback_total_of_one_when_all_zero(self) -> None:
        """Verify a defensive total==1 when every JobSet count is zero."""
        sb, _patch = _make_status_builder()
        status = {"workers": {"total": 0}}
        jobset_status = {
            "replicatedJobsStatus": [
                {
                    "name": "workers",
                    "ready": 0,
                    "active": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "suspended": 0,
                },
            ],
        }

        _ready, _succeeded, total = _update_worker_counts(
            status=status, jobset_status=jobset_status, sb=sb
        )

        assert total == 1

    def test_no_workers_replicated_job(self) -> None:
        """Verify zeros are returned when no 'workers' entry is present."""
        sb, _patch = _make_status_builder()
        status = {}
        jobset_status = {
            "replicatedJobsStatus": [
                {"name": "controller", "ready": 1, "active": 0, "succeeded": 0},
            ],
        }

        assert _update_worker_counts(
            status=status, jobset_status=jobset_status, sb=sb
        ) == (0, 0, 0)


class TestCleanupDeleteFailures:
    """Tests for cleanup paths that delete JobSets before terminal status."""

    @pytest.mark.asyncio
    async def test_timeout_delete_failure_retries_without_terminal_phase(self) -> None:
        """Timeout cleanup must not terminalize while JobSet deletion failed."""
        sb, patch = _make_status_builder()
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=500, reason="apiserver unavailable")
        )

        with pytest.raises(kopf.TemporaryError):
            await _check_job_timeout(
                custom,
                body={"kind": "AIPerfJob"},
                status={"startTime": "2020-01-01T00:00:00Z"},
                spec={"timeoutSeconds": 1},
                namespace="ns",
                jobset_name="js",
                job_id="job",
                key="ns/job",
                sb=sb,
            )

        assert patch.status.get("phase") != str(Phase.FAILED)

    @pytest.mark.asyncio
    async def test_unrecoverable_controller_delete_failure_retries_without_terminal_phase(
        self,
    ) -> None:
        """Controller-failure cleanup must surface delete errors to kopf."""
        sb, patch = _make_status_builder()
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=503, reason="apiserver unavailable")
        )

        with pytest.raises(kopf.TemporaryError):
            await _fail_unrecoverable_controller(
                body={"kind": "AIPerfJob"},
                namespace="ns",
                jobset_name="js",
                job_id="job",
                reason="OOMKilled",
                sb=sb,
                custom=custom,
            )

        assert patch.status.get("phase") != str(Phase.FAILED)

    @pytest.mark.asyncio
    async def test_partial_checkpoint_delete_failure_retries_without_terminal_phase(
        self,
    ) -> None:
        """Partial-checkpoint cleanup must not set Failed until JobSet delete wins."""
        sb, patch = _make_status_builder()
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=503, reason="apiserver unavailable")
        )
        result = SimpleNamespace(checkpoints=["checkpoints/partial.json"])

        with pytest.raises(kopf.TemporaryError):
            await _recover_from_partial_checkpoints(
                body={
                    "kind": "AIPerfJob",
                    "metadata": {"creationTimestamp": "2024-04-25T18:22:03Z"},
                },
                result=result,
                namespace="ns",
                jobset_name="js",
                job_id="job",
                sb=sb,
                custom=custom,
            )

        assert patch.status.get("phase") != str(Phase.FAILED)

    @pytest.mark.asyncio
    async def test_live_status_recovery_preserves_partial_metrics(self) -> None:
        """Controller-death salvage promotes CR live metrics to partial results."""
        sb, patch = _make_status_builder()
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock()
        status = {
            "liveMetrics": {
                "metrics": {
                    "request_throughput": {"avg": 12.5},
                    "request_count": {"total": 42},
                }
            },
            "liveSummary": {"throughputRps": 12.5, "requestCount": 42},
        }

        recovered = await _recover_from_live_status(
            body={"kind": "AIPerfJob"},
            status=status,
            namespace="ns",
            jobset_name="js",
            job_id="job",
            reason="Error",
            sb=sb,
            custom=custom,
        )

        assert recovered is True
        assert patch.status["phase"] == str(Phase.FAILED)
        assert patch.status["results"] == status["liveMetrics"]
        assert patch.status["summary"] == status["liveSummary"]
        assert "partial live metrics" in patch.status["error"]
        results_condition = next(
            condition
            for condition in patch.status["conditions"]
            if condition["type"] == "ResultsAvailable"
        )
        assert results_condition["status"] == "True"
        assert results_condition["reason"] == "PartialLiveMetricsRecovered"
        custom.delete_namespaced_custom_object.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sidecar_export_recovery_completes_without_controller_exit(
        self,
    ) -> None:
        """API blackhole recovery completes once sidecar exposes final exports."""
        sb, _patch = _make_status_builder()
        sidecar_client = AsyncMock()
        sidecar_client.__aenter__.return_value = sidecar_client
        sidecar_client.__aexit__.return_value = None
        sidecar_client.download_all_results.return_value = [
            "profile_export_aiperf.json",
            "profile_export_aiperf.csv",
        ]

        with (
            patch(
                "aiperf.operator.handlers.monitor.ProgressClient",
                return_value=sidecar_client,
            ) as progress_client_cls,
            patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                new=AsyncMock(return_value=True),
            ) as claim,
            patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new=AsyncMock(),
            ) as completion,
        ):
            recovered = await _maybe_recover_exported_results_from_sidecar(
                body={
                    "kind": "AIPerfJob",
                    "metadata": {
                        "name": "job",
                        "creationTimestamp": "2026-05-19T08:00:00Z",
                    },
                },
                namespace="ns",
                name="job",
                jobset_name="aiperf-job",
                job_id="job",
                status={"phase": "Running"},
                sb=sb,
                key="ns/job",
            )

        assert recovered is True
        progress_client_cls.assert_called_once()
        sidecar_client.download_all_results.assert_awaited_once()
        claim.assert_awaited_once_with(
            "ns",
            "job",
            {
                "kind": "AIPerfJob",
                "metadata": {
                    "name": "job",
                    "creationTimestamp": "2026-05-19T08:00:00Z",
                },
            },
        )
        completion.assert_awaited_once()
        result = completion.await_args.kwargs["result"]
        assert result.downloaded == [
            "profile_export_aiperf.json",
            "profile_export_aiperf.csv",
        ]
        assert result.error == ""
