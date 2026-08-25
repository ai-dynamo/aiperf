# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pure helpers in ``aiperf.operator.handlers.monitor``.

The end-to-end ``monitor_progress`` flow is exercised by ``test_main.py`` and
``test_cancellation.py``. This file targets the small helper functions
(``_classify_jobset_failure``, ``_handle_kueue_suspension``,
``_container_status_by_name``, ``_get_terminated_controller_info``,
``_update_worker_counts``) which have no direct unit tests.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import kopf
import orjson
import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.cr_refs import AIPERF_JOB_API_VERSION
from aiperf.kubernetes.phase import Phase
from aiperf.operator import results_layout, runs_index
from aiperf.operator.handlers.monitor import (
    _as_phase,
    _check_job_timeout,
    _classify_jobset_failure,
    _container_status_by_name,
    _fail_unrecoverable_controller,
    _get_pod_startup_issue,
    _get_terminated_controller_info,
    _handle_kueue_suspension,
    _maybe_recover_exported_results_from_sidecar,
    _pod_startup_message,
    _reconcile_pod_startup_issue,
    _recover_from_live_status,
    _recover_from_partial_checkpoints,
    _scheduling_issue_is_structural,
    _startup_issue_state,
    _update_worker_counts,
)
from aiperf.operator.status import StatusBuilder


def _make_status_builder() -> tuple[StatusBuilder, Any]:
    """Return a StatusBuilder wrapping a MagicMock-backed patch with .status={}."""
    patch = MagicMock()
    patch.status = {}
    return StatusBuilder(patch, {}), patch


_JOB_NAME = "aiperf-bench"
_JOB_UID = "7e5b0c93-1d24-4a8f-9b36-c0d7e2f4a681"
_JOBSET_UID = "2c8f6a10-4b39-4de7-85a1-9f0b3c7d5e24"


def _owned_body(name: str = _JOB_NAME) -> dict[str, Any]:
    """Build an AIPerfJob body carrying the immutable identity kopf supplies.

    Cleanup paths fail closed without ``metadata.uid``: they cannot prove they
    control the JobSet they are about to delete, so they abandon the delete and
    the caller skips its terminal status patch. A body without a UID therefore
    never reaches the behaviour these tests assert.
    """
    return {"kind": "AIPerfJob", "metadata": {"name": name, "uid": _JOB_UID}}


def _owned_jobset(jobset_name: str, parent_name: str = _JOB_NAME) -> dict[str, Any]:
    """Build a JobSet snapshot controlled by ``_owned_body``'s identity."""
    return {
        "metadata": {
            "name": jobset_name,
            "uid": _JOBSET_UID,
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_JOB_API_VERSION,
                    "kind": "AIPerfJob",
                    "name": parent_name,
                    "uid": _JOB_UID,
                    "controller": True,
                }
            ],
        }
    }


@contextlib.contextmanager
def _fenced_jobset_api(
    custom: MagicMock,
    jobset_name: str,
    parent_name: str = _JOB_NAME,
) -> Iterator[None]:
    """Point the shared identity helper's cluster reads at one fake API object.

    Cleanup helpers still take a ``CustomObjectsApi`` argument, but they no
    longer use it: the delete is delegated to
    ``_job_identity.delete_owned_aiperfjob_jobset``, which opens its own
    short-lived client, proves exact controller ownership, and issues a
    UID-preconditioned delete. Patching ``k8s_client`` here is what keeps these
    tests off a real apiserver -- unpatched, a live kubeconfig answers the
    ownership read for real and the outcome depends on cluster state.
    """
    custom.get_namespaced_custom_object = AsyncMock(
        return_value=_owned_jobset(jobset_name, parent_name)
    )

    @asynccontextmanager
    async def fake_client() -> AsyncIterator[MagicMock]:
        yield MagicMock(name="ApiClient")

    with (
        patch(
            "aiperf.operator.handlers._job_identity.k8s_client",
            side_effect=fake_client,
        ),
        patch(
            "aiperf.operator.handlers._job_identity.client.CustomObjectsApi",
            return_value=custom,
        ),
    ):
        yield


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


class TestPodStartupIssue:
    """Tests for container and scheduler startup blocker detection."""

    def _pod(
        self,
        *,
        pod_name: str = "aiperf-bench-controller-0",
        container_name: str = "control-plane",
        reason: str = "ImagePullBackOff",
        message: str = "Back-off pulling image 'missing:latest'",
        restarts: int = 0,
        init: bool = False,
    ) -> SimpleNamespace:
        waiting = SimpleNamespace(reason=reason, message=message)
        state = SimpleNamespace(waiting=waiting)
        container_status = SimpleNamespace(
            name=container_name,
            state=state,
            restart_count=restarts,
        )
        return SimpleNamespace(
            metadata=SimpleNamespace(name=pod_name),
            status=SimpleNamespace(
                container_statuses=[] if init else [container_status],
                init_container_statuses=[container_status] if init else [],
                conditions=[],
            ),
        )

    def _unschedulable_pod(self, message: str) -> SimpleNamespace:
        return SimpleNamespace(
            metadata=SimpleNamespace(name="aiperf-bench-worker-0"),
            status=SimpleNamespace(
                container_statuses=[],
                init_container_statuses=[],
                conditions=[
                    SimpleNamespace(
                        type="PodScheduled",
                        status="False",
                        reason="Unschedulable",
                        message=message,
                    )
                ],
            ),
        )

    @pytest.mark.parametrize(
        "reason",
        [
            param("ErrImagePull", id="err_image_pull"),
            param("ImagePullBackOff", id="image_pull_backoff"),
            param("CreateContainerConfigError", id="create_container_config_error"),
        ],
    )  # fmt: skip
    def test_detects_terminal_waiting_reasons(self, reason: str) -> None:
        """Terminal startup reasons surface the offending container."""
        issue = _get_pod_startup_issue([self._pod(reason=reason)])

        assert issue is not None
        assert issue.pod_name == "aiperf-bench-controller-0"
        assert issue.container_name == "control-plane"
        assert issue.reason == reason
        assert issue.message == "Back-off pulling image 'missing:latest'"
        assert issue.terminal_after_threshold is True

    def test_detects_init_container_config_failure(self) -> None:
        """Init-container failures cannot hide from the startup reconciler."""
        issue = _get_pod_startup_issue(
            [self._pod(reason="CreateContainerConfigError", init=True)]
        )

        assert issue is not None
        assert issue.category == "ContainerConfig"

    def test_crash_loop_requires_configured_restart_threshold(self) -> None:
        """A single startup restart remains recoverable; a stable loop is fatal."""
        below = _get_pod_startup_issue(
            [self._pod(reason="CrashLoopBackOff", restarts=1)]
        )
        at_threshold = _get_pod_startup_issue(
            [self._pod(reason="CrashLoopBackOff", restarts=2)]
        )

        assert below is not None and below.terminal_after_threshold is False
        assert (
            at_threshold is not None and at_threshold.terminal_after_threshold is True
        )

    def test_ignores_container_creating(self) -> None:
        """Normal ContainerCreating startup is not fatal."""
        assert (
            _get_pod_startup_issue(
                [self._pod(reason="ContainerCreating", message="creating container")]
            )
            is None
        )

    def test_message_names_jobset_reason_and_image_detail(self) -> None:
        """Formatted operator error includes all user-actionable context."""
        issue = _get_pod_startup_issue([self._pod()])
        assert issue is not None

        message = _pod_startup_message("aiperf-bench", "aiperf-bench-js", issue)

        assert "aiperf-bench" in message
        assert "aiperf-bench-js" in message
        assert "ImagePullBackOff" in message
        assert "missing:latest" in message
        assert "Back-off pulling image" in message

    @pytest.mark.parametrize(
        "message",
        [
            param("0/4 nodes are available: 4 Insufficient nvidia.com/gpu.", id="gpu_capacity"),
            param("0/4 nodes are available: pod has unbound immediate PersistentVolumeClaims.", id="pvc_pending"),
            param("0/4 nodes are available: 4 preemption is not helpful for scheduling.", id="preemption"),
            param("0/4 nodes are available: unknown future scheduler reason.", id="unknown"),
        ],
    )  # fmt: skip
    def test_unschedulable_capacity_and_unknown_reasons_remain_recoverable(
        self, message: str
    ) -> None:
        """Capacity and unknown scheduler states never auto-fail a job."""
        issue = _get_pod_startup_issue([self._unschedulable_pod(message)])

        assert issue is not None
        assert issue.category == "SchedulingDelay"
        assert issue.terminal_after_threshold is False

    @pytest.mark.parametrize(
        "message",
        [
            param("0/4 nodes didn't match Pod's node affinity/selector.", id="selector"),
            param("0/4 nodes had untolerated taint {gpu: reserved}.", id="taint"),
            param("0/4 nodes had volume node affinity conflict.", id="volume_affinity"),
        ],
    )  # fmt: skip
    def test_structural_unschedulable_reasons_are_terminal_after_threshold(
        self, message: str
    ) -> None:
        """Structural placement mistakes may be terminalized after a grace period."""
        issue = _get_pod_startup_issue([self._unschedulable_pod(message)])

        assert issue is not None
        assert issue.category == "SchedulingConstraint"
        assert issue.terminal_after_threshold is True

    def test_capacity_signal_wins_over_structural_signal_in_aggregate(self) -> None:
        """Mixed scheduler summaries stay recoverable when capacity may free up."""
        message = (
            "0/4 nodes are available: 2 Insufficient nvidia.com/gpu, "
            "2 didn't match Pod's node affinity/selector."
        )
        assert _scheduling_issue_is_structural(message) is False

    def test_terminal_issue_wins_over_capacity_issue(self) -> None:
        """A fatal image problem is not hidden by a lower-sorted capacity pod."""
        capacity = self._unschedulable_pod("0/4 nodes: 4 Insufficient cpu")
        image = self._pod(pod_name="z-image-pod")

        issue = _get_pod_startup_issue([capacity, image])

        assert issue is not None
        assert issue.category == "ImagePull"

    def test_persisted_fingerprint_keeps_first_observed_time(self) -> None:
        """Operator restarts do not reset a stable issue's critical timer."""
        issue = _get_pod_startup_issue([self._pod()])
        assert issue is not None
        first = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
        state, elapsed = _startup_issue_state(
            {
                "startupIssue": {
                    "fingerprint": issue.fingerprint,
                    "firstObservedTime": first,
                    "warningEmitted": True,
                }
            },
            issue,
        )

        assert elapsed >= 119
        assert state["firstObservedTime"] == first
        assert state["warningEmitted"] is True


class TestReconcilePodStartupIssue:
    """Tests for startup warnings, recovery, and terminal failure."""

    @staticmethod
    def _image_pull_pod() -> SimpleNamespace:
        return SimpleNamespace(
            metadata=SimpleNamespace(name="aiperf-bench-worker-0"),
            status=SimpleNamespace(
                init_container_statuses=[],
                conditions=[],
                container_statuses=[
                    SimpleNamespace(
                        name="worker",
                        restart_count=0,
                        state=SimpleNamespace(
                            waiting=SimpleNamespace(
                                reason="ImagePullBackOff",
                                message="Back-off pulling image 'missing:latest'",
                            )
                        ),
                    )
                ],
            ),
        )

    @staticmethod
    def _old_issue_status() -> dict[str, Any]:
        return {
            "startupIssue": {
                "fingerprint": "ImagePull:aiperf-bench-worker-0:worker",
                "firstObservedTime": (
                    datetime.now(UTC) - timedelta(seconds=120)
                ).isoformat(),
                "warningEmitted": True,
            }
        }

    async def _reconcile(
        self,
        pod: SimpleNamespace,
        *,
        status: dict[str, Any] | None = None,
    ) -> tuple[bool, Any, MagicMock]:
        sb, status_patch = _make_status_builder()
        core = MagicMock()
        core.list_namespaced_pod = AsyncMock(return_value=SimpleNamespace(items=[pod]))
        custom = MagicMock()
        custom.delete_namespaced_custom_object = AsyncMock()
        with (
            patch(
                "aiperf.operator.handlers.monitor.client.CoreV1Api", return_value=core
            ),
            patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=custom,
            ),
            _fenced_jobset_api(custom, "aiperf-bench-js"),
        ):
            handled = await _reconcile_pod_startup_issue(
                MagicMock(),
                body=_owned_body(),
                status=status or {},
                patch=status_patch,
                namespace="ns",
                name=_JOB_NAME,
                jobset_name="aiperf-bench-js",
                job_id=_JOB_NAME,
                key=f"ns/{_JOB_NAME}@{_JOB_UID}",
                sb=sb,
            )
            if not handled:
                sb.finalize()
        return handled, status_patch, custom

    @pytest.mark.asyncio
    async def test_fresh_image_pull_is_warned_not_failed(self) -> None:
        """A transient registry outage gets its full recovery grace period."""
        handled, status_patch, custom = await self._reconcile(self._image_pull_pod())

        assert handled is False
        assert "phase" not in status_patch.status
        assert status_patch.status["startupIssue"]["category"] == "ImagePull"
        custom.delete_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_critical_issue_defers_cleanup_to_claiming_deadline(self) -> None:
        """The broad recovery path leaves destructive cleanup to the claim timer."""
        handled, status_patch, custom = await self._reconcile(
            self._image_pull_pod(), status=self._old_issue_status()
        )

        assert handled is True
        assert "phase" not in status_patch.status
        assert "error" not in status_patch.status
        assert status_patch.status["startupIssue"]["reason"] == "ImagePullBackOff"
        workers_ready = next(
            condition
            for condition in status_patch.status["conditions"]
            if condition["type"] == "WorkersReady"
        )
        assert workers_ready["reason"] == "PodStartupBlocked"
        custom.delete_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_capacity_unschedulable_never_fails_after_critical_threshold(
        self,
    ) -> None:
        """Resource pressure remains retryable even after the critical timer."""
        pod = SimpleNamespace(
            metadata=SimpleNamespace(name="aiperf-bench-worker-0"),
            status=SimpleNamespace(
                init_container_statuses=[],
                container_statuses=[],
                conditions=[
                    SimpleNamespace(
                        type="PodScheduled",
                        status="False",
                        reason="Unschedulable",
                        message="0/4 nodes are available: 4 Insufficient gpu.",
                    )
                ],
            ),
        )
        status = {
            "startupIssue": {
                "fingerprint": "SchedulingDelay:aiperf-bench-worker-0:pod",
                "firstObservedTime": (
                    datetime.now(UTC) - timedelta(minutes=30)
                ).isoformat(),
                "warningEmitted": True,
            }
        }

        handled, status_patch, custom = await self._reconcile(pod, status=status)

        assert handled is False
        condition = next(
            c for c in status_patch.status["conditions"] if c["type"] == "WorkersReady"
        )
        assert condition["reason"] == "SchedulingDelayed"
        custom.delete_namespaced_custom_object.assert_not_awaited()

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
            handled = await _reconcile_pod_startup_issue(
                MagicMock(),
                body={"kind": "AIPerfJob", "metadata": {"name": "aiperf-bench"}},
                status={},
                patch=patch_status,
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
        """Verify returns None when no non-sidecar container has terminated."""
        sidecar = SimpleNamespace(name="results-sidecar", state=None)
        pod = SimpleNamespace(status=SimpleNamespace(container_statuses=[sidecar]))
        assert _get_terminated_controller_info(pod) is None

    def test_sibling_container_crash_triggers_salvage(self) -> None:
        """Every control-plane service is its own container in this pod.

        Only `control-plane` was inspected, so a records-manager that
        OOMKilled left salvage un-triggered and the job hung until its
        timeout reporting a generic message instead of the real cause.
        """
        controller = SimpleNamespace(
            name="control-plane", state=SimpleNamespace(terminated=None)
        )
        records = SimpleNamespace(
            name="records-manager",
            state=SimpleNamespace(
                terminated=SimpleNamespace(exit_code=137, reason="OOMKilled")
            ),
        )
        sidecar = SimpleNamespace(name="results-sidecar", state=None)
        pod = SimpleNamespace(
            status=SimpleNamespace(container_statuses=[controller, records, sidecar])
        )
        exit_code, reason = _get_terminated_controller_info(pod)
        assert exit_code == 137
        assert "records-manager" in reason
        assert "OOMKilled" in reason

    def test_control_plane_wins_when_several_containers_died(self) -> None:
        """The primary failure should name the reason users act on."""
        controller = SimpleNamespace(
            name="control-plane",
            state=SimpleNamespace(
                terminated=SimpleNamespace(exit_code=1, reason="Error")
            ),
        )
        records = SimpleNamespace(
            name="records-manager",
            state=SimpleNamespace(
                terminated=SimpleNamespace(exit_code=137, reason="OOMKilled")
            ),
        )
        sidecar = SimpleNamespace(name="results-sidecar", state=None)
        pod = SimpleNamespace(
            status=SimpleNamespace(container_statuses=[records, controller, sidecar])
        )
        assert _get_terminated_controller_info(pod) == (1, "Error")

    def test_sidecar_own_death_is_not_a_trigger(self) -> None:
        """Salvage reads through the sidecar; its own exit is not the signal."""
        controller = SimpleNamespace(
            name="control-plane", state=SimpleNamespace(terminated=None)
        )
        sidecar = SimpleNamespace(
            name="results-sidecar",
            state=SimpleNamespace(
                terminated=SimpleNamespace(exit_code=2, reason="Error")
            ),
        )
        pod = SimpleNamespace(
            status=SimpleNamespace(container_statuses=[controller, sidecar])
        )
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
            status=status, jobset_status=jobset_status, spec={}, sb=sb
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
            status=status, jobset_status=jobset_status, spec={}, sb=sb
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
            status=status, jobset_status=jobset_status, spec={}, sb=sb
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
            status=status, jobset_status=jobset_status, spec={}, sb=sb
        ) == (0, 0, 0)

    def test_workers_per_pod_scales_ready_and_succeeded(self) -> None:
        """Pod counts from the JobSet are multiplied by workersPerPod.

        With workers: 4 and workersPerPod: 2 the cluster has 2 pods.  The
        JobSet reports ready=2 (pods), but ``aiperf kube list`` must show 4/4
        (processes), not 2/4.
        """
        sb, _patch = _make_status_builder()
        # status.workers.total was set at job-creation time to the process count.
        status = {"workers": {"total": 4}}
        jobset_status = {
            "replicatedJobsStatus": [
                {
                    "name": "workers",
                    "ready": 2,
                    "succeeded": 0,
                    "active": 0,
                    "failed": 0,
                    "suspended": 0,
                },
            ],
        }
        spec = {"benchmark": {"runtime": {"workersPerPod": 2}}}

        ready, succeeded, total = _update_worker_counts(
            status=status, jobset_status=jobset_status, spec=spec, sb=sb
        )

        assert (ready, succeeded, total) == (4, 0, 4)

    def test_workers_per_pod_scales_succeeded(self) -> None:
        """Succeeded pod count is also scaled to process units."""
        sb, _patch = _make_status_builder()
        status = {"workers": {"total": 6}}
        jobset_status = {
            "replicatedJobsStatus": [
                {
                    "name": "workers",
                    "ready": 0,
                    "succeeded": 3,
                    "active": 0,
                    "failed": 0,
                    "suspended": 0,
                },
            ],
        }
        spec = {"benchmark": {"runtime": {"workersPerPod": 2}}}

        ready, succeeded, total = _update_worker_counts(
            status=status, jobset_status=jobset_status, spec=spec, sb=sb
        )

        assert (ready, succeeded, total) == (0, 6, 6)

    def test_workers_per_pod_fallback_total_also_scaled(self) -> None:
        """Fallback total (CRD status missing) is scaled by workersPerPod."""
        sb, _patch = _make_status_builder()
        # Simulate the case where status.workers.total was not yet written.
        status = {"workers": {"total": 0}}
        jobset_status = {
            "replicatedJobsStatus": [
                {
                    "name": "workers",
                    "ready": 1,
                    "active": 1,
                    "succeeded": 0,
                    "failed": 0,
                    "suspended": 0,
                },
            ],
        }
        spec = {"benchmark": {"runtime": {"workersPerPod": 2}}}

        ready, succeeded, total = _update_worker_counts(
            status=status, jobset_status=jobset_status, spec=spec, sb=sb
        )

        # 1 ready pod * 2 = 2 ready processes; (1+1) pods * 2 = 4 total processes
        assert (ready, succeeded, total) == (2, 0, 4)

    def test_workers_per_pod_missing_from_spec_defaults_to_one(self) -> None:
        """Absent workersPerPod in spec is treated as 1 (no scaling)."""
        sb, _patch = _make_status_builder()
        status = {"workers": {"total": 2}}
        jobset_status = {
            "replicatedJobsStatus": [
                {
                    "name": "workers",
                    "ready": 2,
                    "succeeded": 0,
                    "active": 0,
                    "failed": 0,
                    "suspended": 0,
                },
            ],
        }

        ready, succeeded, total = _update_worker_counts(
            status=status, jobset_status=jobset_status, spec={}, sb=sb
        )

        assert (ready, succeeded, total) == (2, 0, 2)


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

        with (
            _fenced_jobset_api(custom, "js"),
            pytest.raises(kopf.TemporaryError),
        ):
            await _check_job_timeout(
                custom,
                body=_owned_body(),
                status={"startTime": "2020-01-01T00:00:00Z"},
                spec={"timeoutSeconds": 1},
                namespace="ns",
                jobset_name="js",
                job_id="job",
                key=f"ns/job@{_JOB_UID}",
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

        with (
            _fenced_jobset_api(custom, "js"),
            pytest.raises(kopf.TemporaryError),
        ):
            await _fail_unrecoverable_controller(
                body=_owned_body(),
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

        body = _owned_body()
        body["metadata"]["creationTimestamp"] = "2024-04-25T18:22:03Z"

        with (
            _fenced_jobset_api(custom, "js"),
            pytest.raises(kopf.TemporaryError),
        ):
            await _recover_from_partial_checkpoints(
                body=body,
                result=result,
                namespace="ns",
                jobset_name="js",
                job_id="job",
                sb=sb,
                custom=custom,
            )

        assert patch.status.get("phase") != str(Phase.FAILED)

    @pytest.mark.asyncio
    async def test_partial_checkpoint_recovery_synchronizes_index_before_ungating(
        self,
        tmp_path: Path,
    ) -> None:
        """Ready partial artifacts must not expose stale create-time run metadata."""
        sb, _ = _make_status_builder()
        body = _owned_body()
        body["metadata"]["creationTimestamp"] = "2024-04-25T18:22:03Z"
        epoch = results_layout.epoch_key_from_body(body)
        base = tmp_path / "results"
        checkpoint = (
            results_layout.run_dir(base, "ns", "job", epoch)
            / "checkpoints"
            / "partial.json"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(
            orjson.dumps({"request_throughput": {"avg": 12.5, "unit": "req/s"}})
        )
        (checkpoint.parent.parent / "job_spec.json").write_bytes(b"{}")
        await runs_index.close()
        await runs_index.open(base / ".aiperf_index.sqlite")
        await runs_index.upsert_run_created("ns", "job", epoch, spec={})
        runs_index.mark_catalog_complete(base)

        try:
            with (
                patch(
                    "aiperf.operator.handlers.monitor.OperatorEnvironment.RESULTS",
                    SimpleNamespace(DIR=base),
                ),
                patch(
                    "aiperf.operator.handlers.monitor._delete_jobset_or_retry",
                    new=AsyncMock(return_value=True),
                ),
                patch("aiperf.operator.handlers.monitor.events.results_stored"),
                patch("aiperf.operator.handlers.monitor.events.failed"),
            ):
                await _recover_from_partial_checkpoints(
                    body=body,
                    result=SimpleNamespace(checkpoints=["checkpoints/partial.json"]),
                    namespace="ns",
                    jobset_name="js",
                    job_id="job",
                    sb=sb,
                    custom=MagicMock(),
                )

            disk_runs = results_layout.list_runs(base, "ns", "job")
            indexed_runs = await results_layout.list_runs_async(base, "ns", "job")
            indexed_row = await runs_index.get_run("ns", "job", epoch)

            assert indexed_runs == disk_runs
            assert indexed_row is not None
            assert indexed_row.phase == "Failed"
            assert indexed_row.error is not None
            assert runs_index.catalog_is_complete(base) is True
        finally:
            await runs_index.close()

    @pytest.mark.asyncio
    async def test_partial_checkpoint_index_failure_keeps_catalog_gated(
        self,
        tmp_path: Path,
    ) -> None:
        """A failed metadata sync must leave readers on disk fallback."""
        sb, _ = _make_status_builder()
        body = _owned_body()
        body["metadata"]["creationTimestamp"] = "2024-04-25T18:22:03Z"
        runs_index.mark_catalog_complete(tmp_path)

        with (
            patch(
                "aiperf.operator.handlers.monitor.OperatorEnvironment.RESULTS",
                SimpleNamespace(DIR=tmp_path),
            ),
            patch(
                "aiperf.operator.handlers.monitor._delete_jobset_or_retry",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "aiperf.operator.handlers.monitor._parse_metrics_from_files",
                return_value=None,
            ),
            patch("aiperf.operator.handlers.monitor.write_ready_marker"),
            patch(
                "aiperf.operator.runs_index.upsert_run_completed",
                new=AsyncMock(side_effect=RuntimeError("sqlite unavailable")),
            ),
            patch("aiperf.operator.handlers.monitor.events.results_stored"),
            patch("aiperf.operator.handlers.monitor.events.failed"),
        ):
            await _recover_from_partial_checkpoints(
                body=body,
                result=SimpleNamespace(checkpoints=["checkpoints/partial.json"]),
                namespace="ns",
                jobset_name="js",
                job_id="job",
                sb=sb,
                custom=MagicMock(),
            )

        assert runs_index.catalog_is_complete(tmp_path) is False

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

        with _fenced_jobset_api(custom, "js"):
            recovered = await _recover_from_live_status(
                body=_owned_body(),
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


class TestSidecarRecoveryPhaseGate:
    """Which phases may still reach the sidecar completion path."""

    def test_initializing_is_recoverable(self) -> None:
        """Regression: a run that finishes between two monitor ticks never
        leaves Initializing, because the controller API it would be promoted
        to Running by dies with the run. Gating recovery on Running alone made
        completion unreachable for the fastest benchmarks."""
        from aiperf.kubernetes.phase import Phase
        from aiperf.operator.handlers.monitor import (
            _NON_TERMINAL_RECOVERABLE_PHASES,
        )

        assert Phase.INITIALIZING in _NON_TERMINAL_RECOVERABLE_PHASES
        assert Phase.RUNNING in _NON_TERMINAL_RECOVERABLE_PHASES
        assert Phase.COMPLETED not in _NON_TERMINAL_RECOVERABLE_PHASES
        assert Phase.FAILED not in _NON_TERMINAL_RECOVERABLE_PHASES


class TestPhaseCoercionGate:
    """Regression tests for the sidecar-recovery phase gate.

    ``status.phase`` arrives from the apiserver as a plain ``str``. Because
    ``Enum.__hash__`` hashes by member name, ``"Running" in
    frozenset({Phase.RUNNING})`` is ``False``, which silently closed the
    sidecar-recovery gate: a benchmark that finished between two monitor ticks
    (the controller API dies with it) could never be completed, so the
    AIPerfJob stayed ``Running`` forever with results sitting on the sidecar.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            param("Running", Phase.RUNNING, id="running"),
            param("running", Phase.RUNNING, id="lowercase"),
            param(Phase.INITIALIZING, Phase.INITIALIZING, id="already-member"),
            param("NotAPhase", Phase.PENDING, id="unknown-falls-back"),
            param(None, Phase.PENDING, id="none-falls-back"),
        ],
    )  # fmt: skip
    def test_as_phase_coerces_to_member(self, raw: Any, expected: Phase) -> None:
        assert _as_phase(raw) is expected
        assert _as_phase(raw) in frozenset({expected})
