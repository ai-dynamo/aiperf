# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `aiperf.cli_commands.kube._debug_extract`.

The public ``_extract_pod_info`` orchestration is exercised by
``test_debug.py`` (where the function is also re-exported). This file targets
the per-helper functions that the orchestrator delegates to:
- ``_get_serializer``: canonical open-client requirement
- ``_pod_to_raw``: legacy ``.raw`` mock vs real V1Pod paths
- ``_waiting_problem`` / ``_oom_problem`` / ``_container_problems``: the
  per-state detector branches
- ``_unschedulable_problem``: PodScheduled condition decoding
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.cli_commands.kube._debug_extract import (
    _PROBLEM_STATES,
    _container_problems,
    _extract_pod_info,
    _get_serializer,
    _oom_problem,
    _pod_to_raw,
    _unschedulable_problem,
    _waiting_problem,
)

# ============================================================
# Helpers
# ============================================================


def _raw_pod(
    *,
    name: str = "pod",
    namespace: str = "ns",
    phase: str = "Running",
    node: str = "node-1",
    container_statuses: list[dict[str, Any]] | None = None,
    init_container_statuses: list[dict[str, Any]] | None = None,
    conditions: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a legacy ``.raw``-shape mock pod (matches existing test_debug pattern)."""
    pod = MagicMock()
    pod.name = name
    pod.raw = {
        "metadata": {"name": name, "namespace": namespace},
        "spec": {"nodeName": node},
        "status": {
            "phase": phase,
            "containerStatuses": container_statuses or [],
            "initContainerStatuses": init_container_statuses or [],
            "conditions": conditions or [],
        },
    }
    return pod


# ============================================================
# _get_serializer
# ============================================================


class TestGetSerializer:
    """Verify serialization uses the caller's managed API client."""

    def test_passthrough_when_api_provided(self) -> None:
        api = SimpleNamespace(sanitize_for_serialization=lambda x: x)
        assert _get_serializer(api) is api

    def test_missing_api_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="api is required"):
            _get_serializer(None)


# ============================================================
# _pod_to_raw
# ============================================================


class TestPodToRaw:
    """Verify both the legacy ``.raw`` path and the V1Pod sanitize path."""

    def test_raw_mock_pod_uses_pod_dot_name(self) -> None:
        pod = MagicMock()
        pod.name = "explicit-name"
        pod.raw = {"metadata": {"name": "ignored-by-name-attr"}}
        name, raw = _pod_to_raw(pod)
        assert name == "explicit-name"
        assert raw is pod.raw

    def test_raw_mock_pod_falls_back_to_metadata_name(self) -> None:
        """When .name is empty/None, metadata.name is used."""
        pod = MagicMock()
        pod.name = None
        pod.raw = {"metadata": {"name": "from-metadata"}}
        name, _ = _pod_to_raw(pod)
        assert name == "from-metadata"

    def test_real_v1pod_uses_sanitize_for_serialization(self) -> None:
        sentinel_dict = {"metadata": {"name": "real-pod"}, "status": {}}
        api = MagicMock()
        api.sanitize_for_serialization = MagicMock(return_value=sentinel_dict)

        # Real V1Pod-like: no .raw attribute. We must bypass MagicMock's
        # auto-attr behaviour which would invent a non-None .raw.
        class FakePod:
            pass

        pod = FakePod()
        name, raw = _pod_to_raw(pod, api=api)
        assert name == "real-pod"
        assert raw is sentinel_dict
        api.sanitize_for_serialization.assert_called_once_with(pod)

    def test_real_v1pod_with_none_sanitize_result_returns_empty(self) -> None:
        api = MagicMock()
        api.sanitize_for_serialization = MagicMock(return_value=None)

        class FakePod:
            pass

        name, raw = _pod_to_raw(FakePod(), api=api)
        assert name == ""
        assert raw == {}


# ============================================================
# _waiting_problem
# ============================================================


class TestWaitingProblem:
    """Verify waiting-state classification."""

    @pytest.mark.parametrize(
        "reason",
        [
            param(r, id=r)
            for r in _PROBLEM_STATES
            # _waiting_problem only handles waiting reasons; OOMKilled is
            # terminated, so skip it here.
            if r != "OOMKilled"
        ],
    )  # fmt: skip
    def test_known_problem_reasons_yield_full_problem_dict(self, reason: str) -> None:
        problem = _waiting_problem(
            "main", {"reason": reason, "message": "msg"}, phase="Pending"
        )
        assert problem is not None
        expected_severity, expected_suggestion = _PROBLEM_STATES[reason]
        assert problem["container"] == "main"
        assert problem["state"] == reason
        assert problem["severity"] == expected_severity
        assert problem["suggestion"] == expected_suggestion
        assert problem["message"] == "msg"

    def test_unknown_reason_on_pending_is_warning(self) -> None:
        problem = _waiting_problem(
            "main", {"reason": "ContainerCreating", "message": ""}, phase="Pending"
        )
        assert problem is not None
        assert problem["severity"] == "WARNING"
        assert problem["state"] == "ContainerCreating"

    def test_unknown_reason_on_running_returns_none(self) -> None:
        problem = _waiting_problem(
            "main", {"reason": "WeirdState", "message": ""}, phase="Running"
        )
        assert problem is None

    def test_empty_reason_returns_none(self) -> None:
        problem = _waiting_problem(
            "main", {"reason": "", "message": ""}, phase="Pending"
        )
        assert problem is None

    def test_missing_reason_key_returns_none(self) -> None:
        problem = _waiting_problem("main", {}, phase="Pending")
        assert problem is None


# ============================================================
# _oom_problem
# ============================================================


class TestOomProblem:
    """Verify OOM detection from terminated state."""

    def test_oomkilled_current_state(self) -> None:
        problem = _oom_problem(
            "main", {"reason": "OOMKilled", "message": "killed"}, previous=False
        )
        assert problem is not None
        assert problem["state"] == "OOMKilled"
        assert problem["severity"] == "CRITICAL"
        assert problem["message"] == "killed"

    def test_oomkilled_previous_state_labelled(self) -> None:
        problem = _oom_problem(
            "main", {"reason": "OOMKilled", "message": ""}, previous=True
        )
        assert problem is not None
        assert problem["state"] == "OOMKilled (previous)"

    @pytest.mark.parametrize(
        "reason",
        [
            param("Completed", id="completed"),
            param("Error", id="error"),
            param("", id="empty"),
        ],
    )  # fmt: skip
    def test_non_oom_reason_returns_none(self, reason: str) -> None:
        problem = _oom_problem("main", {"reason": reason}, previous=False)
        assert problem is None


# ============================================================
# _container_problems
# ============================================================


class TestContainerProblems:
    """Verify per-container aggregation across waiting + terminated + lastState."""

    def test_no_state_returns_empty(self) -> None:
        cs = {"name": "main", "state": {}}
        assert _container_problems(cs, phase="Running") == []

    def test_waiting_terminated_and_last_state_oom_all_combined(self) -> None:
        cs = {
            "name": "main",
            "state": {
                "waiting": {"reason": "CrashLoopBackOff", "message": "back-off"},
                "terminated": {"reason": "OOMKilled", "message": "current"},
            },
            "lastState": {"terminated": {"reason": "OOMKilled", "message": "previous"}},
        }
        problems = _container_problems(cs, phase="Running")

        states = [p["state"] for p in problems]
        assert states == ["CrashLoopBackOff", "OOMKilled", "OOMKilled (previous)"]

    def test_unnamed_container_uses_unknown(self) -> None:
        cs = {"state": {"waiting": {"reason": "ImagePullBackOff", "message": ""}}}
        problems = _container_problems(cs, phase="Pending")
        assert problems[0]["container"] == "unknown"


# ============================================================
# _unschedulable_problem
# ============================================================


class TestUnschedulableProblem:
    """Verify decoding of the PodScheduled=False/Unschedulable condition."""

    def test_unschedulable_condition_yields_problem(self) -> None:
        conds = [
            {
                "type": "PodScheduled",
                "status": "False",
                "reason": "Unschedulable",
                "message": "0/3 nodes available",
            }
        ]
        problem = _unschedulable_problem(conds)
        assert problem is not None
        assert problem["state"] == "Unschedulable"
        assert problem["container"] == "-"
        assert problem["severity"] == "CRITICAL"
        assert "0/3 nodes available" in problem["message"]

    def test_scheduled_condition_returns_none(self) -> None:
        conds = [{"type": "PodScheduled", "status": "True"}]
        assert _unschedulable_problem(conds) is None

    def test_unrelated_conditions_return_none(self) -> None:
        conds = [
            {"type": "Ready", "status": "False"},
            {"type": "Initialized", "status": "True"},
        ]
        assert _unschedulable_problem(conds) is None

    def test_empty_conditions_return_none(self) -> None:
        assert _unschedulable_problem([]) is None

    def test_condition_with_different_reason_returns_none(self) -> None:
        """PodScheduled=False but reason is not Unschedulable (e.g. SchedulingDisabled)."""
        conds = [
            {
                "type": "PodScheduled",
                "status": "False",
                "reason": "SchedulingGated",
                "message": "gated",
            }
        ]
        assert _unschedulable_problem(conds) is None


# ============================================================
# _extract_pod_info — light edge-case coverage not in test_debug.py
# ============================================================


class TestExtractPodInfoEdges:
    """Edge cases beyond what test_debug.py covers."""

    def test_unicode_pod_and_message_preserved(self) -> None:
        pod = _raw_pod(
            name="ポッド",
            container_statuses=[
                {
                    "name": "メイン",
                    "restartCount": 0,
                    "state": {
                        "waiting": {"reason": "CrashLoopBackOff", "message": "失敗"}
                    },
                }
            ],
        )
        info = _extract_pod_info(pod)
        assert info["name"] == "ポッド"
        assert info["problems"][0]["container"] == "メイン"
        assert info["problems"][0]["message"] == "失敗"

    def test_completely_empty_raw_dict_does_not_raise(self) -> None:
        pod = MagicMock()
        pod.name = "bare"
        pod.raw = {}
        info = _extract_pod_info(pod)
        # Defaults stand in for every missing key.
        assert info["name"] == "bare"
        assert info["phase"] == "Unknown"
        assert info["restarts"] == 0
        assert info["problems"] == []
        assert info["container_statuses"] == []
        assert info["node"] == ""
        assert info["namespace"] == ""

    def test_init_containers_processed_before_main(self) -> None:
        """Init container statuses must precede main containers in container_statuses."""
        pod = _raw_pod(
            init_container_statuses=[
                {"name": "init-1", "restartCount": 0, "state": {}}
            ],
            container_statuses=[{"name": "main", "restartCount": 0, "state": {}}],
        )
        info = _extract_pod_info(pod)
        names = [cs["name"] for cs in info["container_statuses"]]
        assert names == ["init-1", "main"]
