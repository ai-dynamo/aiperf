# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pure helpers in ``aiperf.operator.handlers.monitor``.

The end-to-end ``monitor_progress`` flow is exercised by ``test_main.py`` and
``test_cancellation.py``. This file targets the small helper functions
(``_classify_jobset_failure``, ``_should_poll_progress``,
``_handle_kueue_suspension``, ``_container_status_by_name``,
``_get_terminated_controller_info``, ``_update_worker_counts``,
``_apply_controller_progress_status``) which have no direct unit tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.operator.handlers.monitor import (
    _apply_controller_progress_status,
    _classify_jobset_failure,
    _container_status_by_name,
    _get_terminated_controller_info,
    _handle_kueue_suspension,
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


class TestApplyControllerProgressStatus:
    """Tests for ``_apply_controller_progress_status``."""

    def _progress(
        self, *, current_phase: str | None, workers: dict[str, int] | None = None
    ) -> MagicMock:
        p = MagicMock()
        p.current_phase = current_phase
        p.workers = MagicMock()
        p.workers.model_dump = MagicMock(
            return_value=workers if workers is not None else {"ready": 1, "total": 1}
        )
        return p

    def test_sets_running_when_profiling(self) -> None:
        """Verify 'profiling' controller phase promotes the CR to RUNNING."""
        sb, patch = _make_status_builder()
        progress = self._progress(current_phase="profiling")

        _apply_controller_progress_status(patch, sb, progress, Phase.INITIALIZING)

        assert patch.status["currentPhase"] == "profiling"
        assert patch.status["phase"] == str(Phase.RUNNING)

    def test_sets_initializing_when_pre_profiling_in_pending(self) -> None:
        """Verify pre-profiling phases advance PENDING to INITIALIZING."""
        sb, patch = _make_status_builder()
        progress = self._progress(current_phase="warmup")

        _apply_controller_progress_status(patch, sb, progress, Phase.PENDING)

        assert patch.status["currentPhase"] == "warmup"
        assert patch.status["phase"] == str(Phase.INITIALIZING)

    def test_noop_when_controller_has_no_phase(self) -> None:
        """Verify no phase fields are written when the controller has no phase yet."""
        sb, patch = _make_status_builder()
        progress = self._progress(current_phase=None)

        _apply_controller_progress_status(patch, sb, progress, Phase.RUNNING)

        assert "phase" not in patch.status
        assert "currentPhase" not in patch.status

    def test_does_not_demote_running_from_non_profiling_phase(self) -> None:
        """Verify a RUNNING CR is not demoted back to INITIALIZING mid-run."""
        sb, patch = _make_status_builder()
        progress = self._progress(current_phase="warmup")

        _apply_controller_progress_status(patch, sb, progress, Phase.RUNNING)

        # Controller currentPhase still recorded, but CR phase unchanged.
        assert patch.status["currentPhase"] == "warmup"
        assert "phase" not in patch.status
