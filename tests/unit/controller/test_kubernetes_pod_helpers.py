# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kubernetes_pod_helpers.

Focuses on:
- PodInfo dataclass defaults and is_terminal predicate
- container_statuses_as_dicts conversion (waiting / terminated / lastState branches)
- conditions_as_dicts None-tolerance
- extract_container_issues deduplication and reason extraction
- format_pod_failure_reason branches: terminated, waiting, conditions
- extract_pod_snapshot label-gating: replicated_job filter, missing pod_index, missing status
- aggregate_pods_by_index dedup with non-terminal preference
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pytest import param

from aiperf.controller.kubernetes_pod_helpers import (
    PodInfo,
    aggregate_pods_by_index,
    conditions_as_dicts,
    container_statuses_as_dicts,
    extract_container_issues,
    extract_pod_snapshot,
    format_pod_failure_reason,
)
from aiperf.kubernetes.constants import JobSetLabels
from aiperf.kubernetes.enums import PodPhase

# ============================================================
# Helpers — minimal SimpleNamespace stand-ins for kubernetes_asyncio
# ============================================================


def make_container_status(
    *,
    name: str = "control-plane",
    restart_count: int = 0,
    waiting_reason: str | None = None,
    waiting_message: str | None = None,
    terminated_reason: str | None = None,
    terminated_message: str | None = None,
    terminated_exit_code: int | None = None,
    last_terminated_reason: str | None = None,
    state: Any = ...,  # sentinel: build from waiting/terminated unless explicit
    last_state: Any = ...,
) -> SimpleNamespace:
    """Build a V1ContainerStatus-like namespace mirroring kubernetes_asyncio shape."""
    if state is ...:
        waiting = (
            SimpleNamespace(reason=waiting_reason, message=waiting_message)
            if waiting_reason is not None or waiting_message is not None
            else None
        )
        terminated = (
            SimpleNamespace(
                reason=terminated_reason,
                message=terminated_message,
                exit_code=terminated_exit_code,
            )
            if (
                terminated_reason is not None
                or terminated_message is not None
                or terminated_exit_code is not None
            )
            else None
        )
        state = SimpleNamespace(waiting=waiting, terminated=terminated)
    if last_state is ...:
        last_terminated = (
            SimpleNamespace(reason=last_terminated_reason)
            if last_terminated_reason is not None
            else None
        )
        last_state = SimpleNamespace(terminated=last_terminated)
    return SimpleNamespace(
        name=name,
        restart_count=restart_count,
        state=state,
        last_state=last_state,
    )


def make_pod(
    *,
    name: str | None = "pod-x",
    labels: dict[str, str] | None = None,
    phase: str | None = "Running",
    container_statuses: list[Any] | None = None,
    conditions: list[Any] | None = None,
    metadata: Any = ...,
    status: Any = ...,
) -> SimpleNamespace:
    """Build a V1Pod-like namespace."""
    if metadata is ...:
        metadata = SimpleNamespace(name=name, labels=labels)
    if status is ...:
        status = SimpleNamespace(
            phase=phase,
            container_statuses=container_statuses,
            conditions=conditions,
        )
    return SimpleNamespace(metadata=metadata, status=status)


# ============================================================
# PodInfo
# ============================================================


class TestPodInfoDefaults:
    """Default-constructed PodInfo carries the safe Pending baseline."""

    def test_required_fields_only_apply_defaults(self) -> None:
        pod = PodInfo(pod_index="0", pod_name="aiperf-worker-0-0-abc")
        assert pod.pod_index == "0"
        assert pod.pod_name == "aiperf-worker-0-0-abc"
        assert pod.phase == PodPhase.PENDING
        assert pod.restart_count == 0
        assert pod.container_issues == []
        assert pod.last_checked_ns == 0
        assert pod.failed is False

    def test_default_factory_returns_independent_lists(self) -> None:
        a = PodInfo(pod_index="0", pod_name="a")
        b = PodInfo(pod_index="1", pod_name="b")
        a.container_issues.append("OOMKilled")
        assert b.container_issues == []


class TestPodInfoIsTerminal:
    """is_terminal flips for FAILED and UNKNOWN, false otherwise."""

    @pytest.mark.parametrize(
        "phase,expected",
        [
            (PodPhase.PENDING, False),
            (PodPhase.RUNNING, False),
            (PodPhase.SUCCEEDED, False),
            (PodPhase.FAILED, True),
            (PodPhase.UNKNOWN, True),
        ],
    )  # fmt: skip
    def test_is_terminal_matches_phase(self, phase: PodPhase, expected: bool) -> None:
        pod = PodInfo(pod_index="0", pod_name="p", phase=phase)
        assert pod.is_terminal is expected


# ============================================================
# container_statuses_as_dicts
# ============================================================


class TestContainerStatusesAsDicts:
    """Conversion preserves legacy dict shape used downstream."""

    def test_empty_input_returns_empty_list(self) -> None:
        assert container_statuses_as_dicts([]) == []

    def test_waiting_state_converted(self) -> None:
        cs = make_container_status(
            name="control-plane",
            waiting_reason="ImagePullBackOff",
            waiting_message="failed to pull image",
        )
        result = container_statuses_as_dicts([cs])
        assert len(result) == 1
        assert result[0]["name"] == "control-plane"
        assert result[0]["state"]["waiting"] == {
            "reason": "ImagePullBackOff",
            "message": "failed to pull image",
        }
        assert "terminated" not in result[0]["state"]

    def test_terminated_state_converted(self) -> None:
        cs = make_container_status(
            terminated_reason="OOMKilled",
            terminated_message="oom",
            terminated_exit_code=137,
        )
        result = container_statuses_as_dicts([cs])
        assert result[0]["state"]["terminated"] == {
            "reason": "OOMKilled",
            "message": "oom",
            "exitCode": 137,
        }

    def test_last_state_terminated_carries_reason_only(self) -> None:
        cs = make_container_status(last_terminated_reason="Error")
        result = container_statuses_as_dicts([cs])
        assert result[0]["lastState"] == {"terminated": {"reason": "Error"}}

    def test_none_state_yields_empty_state_dict(self) -> None:
        cs = make_container_status(state=None, last_state=None)
        result = container_statuses_as_dicts([cs])
        assert result[0]["state"] == {}
        assert result[0]["lastState"] == {}

    def test_none_reasons_become_empty_strings(self) -> None:
        # Both reason and message are explicit None on the API object → ""
        waiting = SimpleNamespace(reason=None, message=None)
        state = SimpleNamespace(waiting=waiting, terminated=None)
        cs = SimpleNamespace(
            name="c",
            restart_count=2,
            state=state,
            last_state=SimpleNamespace(terminated=None),
        )
        result = container_statuses_as_dicts([cs])
        assert result[0]["state"]["waiting"] == {"reason": "", "message": ""}

    def test_missing_name_defaults_to_unknown(self) -> None:
        cs = make_container_status(name="")
        # name="" is falsy → default to "unknown"
        result = container_statuses_as_dicts([cs])
        assert result[0]["name"] == "unknown"

    def test_none_name_defaults_to_unknown(self) -> None:
        cs = SimpleNamespace(
            name=None,
            restart_count=None,
            state=None,
            last_state=None,
        )
        result = container_statuses_as_dicts([cs])
        assert result[0]["name"] == "unknown"
        assert result[0]["restartCount"] == 0

    def test_restart_count_passed_through(self) -> None:
        cs = make_container_status(restart_count=7)
        result = container_statuses_as_dicts([cs])
        assert result[0]["restartCount"] == 7

    def test_multiple_container_statuses_preserved(self) -> None:
        cs1 = make_container_status(
            name="c1", terminated_reason="Completed", terminated_exit_code=0
        )
        cs2 = make_container_status(name="c2", waiting_reason="ContainerCreating")
        result = container_statuses_as_dicts([cs1, cs2])
        assert [r["name"] for r in result] == ["c1", "c2"]


# ============================================================
# conditions_as_dicts
# ============================================================


class TestConditionsAsDicts:
    """Condition list conversion is straightforward but tolerates Nones."""

    def test_empty_input(self) -> None:
        assert conditions_as_dicts([]) == []

    def test_full_condition_passes_through(self) -> None:
        cond = SimpleNamespace(type="Ready", status="True", message="hello")
        result = conditions_as_dicts([cond])
        assert result == [{"type": "Ready", "status": "True", "message": "hello"}]

    def test_none_fields_become_empty_strings(self) -> None:
        cond = SimpleNamespace(type=None, status=None, message=None)
        result = conditions_as_dicts([cond])
        assert result == [{"type": "", "status": "", "message": ""}]


# ============================================================
# extract_container_issues
# ============================================================


class TestExtractContainerIssues:
    """Issues come from waiting, terminated, and lastState.terminated; deduped in order."""

    def test_empty_input_yields_no_issues(self) -> None:
        assert extract_container_issues([]) == []

    def test_collects_waiting_reason(self) -> None:
        cs = [
            {
                "name": "c",
                "restartCount": 0,
                "state": {"waiting": {"reason": "ImagePullBackOff", "message": ""}},
                "lastState": {},
            }
        ]
        assert extract_container_issues(cs) == ["ImagePullBackOff"]

    def test_collects_terminated_reason(self) -> None:
        cs = [
            {
                "name": "c",
                "restartCount": 1,
                "state": {
                    "terminated": {
                        "reason": "OOMKilled",
                        "message": "",
                        "exitCode": 137,
                    }
                },
                "lastState": {},
            }
        ]
        assert extract_container_issues(cs) == ["OOMKilled"]

    def test_collects_last_terminated_reason(self) -> None:
        cs = [
            {
                "name": "c",
                "restartCount": 5,
                "state": {},
                "lastState": {"terminated": {"reason": "CrashLoopBackOff"}},
            }
        ]
        assert extract_container_issues(cs) == ["CrashLoopBackOff"]

    def test_dedupes_repeated_reasons_across_containers(self) -> None:
        cs = [
            {
                "name": "c1",
                "restartCount": 0,
                "state": {"waiting": {"reason": "ImagePullBackOff", "message": ""}},
                "lastState": {},
            },
            {
                "name": "c2",
                "restartCount": 0,
                "state": {"waiting": {"reason": "ImagePullBackOff", "message": ""}},
                "lastState": {},
            },
        ]
        assert extract_container_issues(cs) == ["ImagePullBackOff"]

    def test_collects_multiple_distinct_reasons_in_order(self) -> None:
        cs = [
            {
                "name": "c1",
                "restartCount": 1,
                "state": {
                    "terminated": {"reason": "Error", "message": "", "exitCode": 1}
                },
                "lastState": {"terminated": {"reason": "OOMKilled"}},
            },
            {
                "name": "c2",
                "restartCount": 0,
                "state": {"waiting": {"reason": "CrashLoopBackOff", "message": ""}},
                "lastState": {},
            },
        ]
        result = extract_container_issues(cs)
        assert result == ["Error", "OOMKilled", "CrashLoopBackOff"]

    def test_empty_reason_strings_are_skipped(self) -> None:
        cs = [
            {
                "name": "c",
                "restartCount": 0,
                "state": {
                    "waiting": {"reason": "", "message": ""},
                    "terminated": {
                        "reason": "OOMKilled",
                        "message": "",
                        "exitCode": 137,
                    },
                },
                "lastState": {"terminated": {"reason": ""}},
            }
        ]
        assert extract_container_issues(cs) == ["OOMKilled"]

    def test_missing_state_keys_handled(self) -> None:
        cs = [{"name": "c", "restartCount": 0, "state": {}, "lastState": {}}]
        assert extract_container_issues(cs) == []


# ============================================================
# format_pod_failure_reason
# ============================================================


class TestFormatPodFailureReason:
    """Builds a human-readable string aggregating phase, container states, conditions."""

    def test_phase_only_when_no_containers_or_conditions(self) -> None:
        reason = format_pod_failure_reason("worker-0", PodPhase.FAILED, [], {})
        assert reason == "K8s pod 'worker-0' is Failed"

    def test_terminated_branch_includes_reason_exit_code_message(self) -> None:
        cs = [
            {
                "name": "control-plane",
                "state": {
                    "terminated": {
                        "reason": "OOMKilled",
                        "exitCode": 137,
                        "message": "memory limit exceeded",
                    }
                },
            }
        ]
        reason = format_pod_failure_reason("worker-0", PodPhase.FAILED, cs, {})
        assert "K8s pod 'worker-0' is Failed" in reason
        assert "container 'control-plane': terminated" in reason
        assert "(OOMKilled)" in reason
        assert "exit_code=137" in reason
        assert "memory limit exceeded" in reason

    def test_terminated_truncates_long_message_to_200_chars(self) -> None:
        long_msg = "x" * 500
        cs = [
            {
                "name": "c",
                "state": {
                    "terminated": {
                        "reason": "Error",
                        "exitCode": 1,
                        "message": long_msg,
                    }
                },
            }
        ]
        reason = format_pod_failure_reason("p", PodPhase.FAILED, cs, {})
        # Truncated to 200 chars; verify by checking the truncated chunk shows up
        # but the full 500-char string does not.
        assert ("x" * 200) in reason
        assert ("x" * 201) not in reason

    def test_terminated_without_reason_or_exit_code(self) -> None:
        cs = [{"name": "c", "state": {"terminated": {"reason": "", "message": ""}}}]
        reason = format_pod_failure_reason("p", PodPhase.FAILED, cs, {})
        assert "container 'c': terminated" in reason
        assert "(" not in reason.split("|")[1]  # no parenthetical reason

    def test_waiting_branch_includes_reason_and_message(self) -> None:
        cs = [
            {
                "name": "ctl",
                "state": {
                    "waiting": {
                        "reason": "ImagePullBackOff",
                        "message": "rpc error: code = Unknown",
                    }
                },
            }
        ]
        reason = format_pod_failure_reason("p", PodPhase.PENDING, cs, {})
        assert "container 'ctl': waiting (ImagePullBackOff)" in reason
        assert "rpc error: code = Unknown" in reason

    def test_waiting_with_empty_reason_skipped(self) -> None:
        cs = [{"name": "c", "state": {"waiting": {"reason": "", "message": "ignored"}}}]
        reason = format_pod_failure_reason("p", PodPhase.PENDING, cs, {})
        # An empty waiting reason produces no extra container detail.
        assert reason == "K8s pod 'p' is Pending"

    def test_conditions_with_status_false_appended(self) -> None:
        status = {
            "conditions": [
                {"type": "PodScheduled", "status": "False", "message": "no nodes"},
                {"type": "Ready", "status": "True", "message": "ok"},
            ]
        }
        reason = format_pod_failure_reason("p", PodPhase.PENDING, [], status)
        assert "condition PodScheduled: no nodes" in reason
        # status=True conditions are not included
        assert "Ready" not in reason

    def test_condition_without_message_skipped(self) -> None:
        status = {"conditions": [{"type": "Ready", "status": "False", "message": ""}]}
        reason = format_pod_failure_reason("p", PodPhase.FAILED, [], status)
        assert "condition Ready" not in reason

    def test_condition_message_truncated_to_200(self) -> None:
        long_msg = "y" * 500
        status = {
            "conditions": [
                {"type": "PodScheduled", "status": "False", "message": long_msg}
            ]
        }
        reason = format_pod_failure_reason("p", PodPhase.PENDING, [], status)
        assert ("y" * 200) in reason
        assert ("y" * 201) not in reason

    def test_combined_pipe_separated_segments(self) -> None:
        cs = [
            {
                "name": "c",
                "state": {
                    "terminated": {"reason": "Error", "exitCode": 1, "message": "boom"}
                },
            }
        ]
        status = {
            "conditions": [{"type": "Ready", "status": "False", "message": "down"}]
        }
        reason = format_pod_failure_reason("worker-0", PodPhase.FAILED, cs, status)
        parts = reason.split(" | ")
        assert parts[0] == "K8s pod 'worker-0' is Failed"
        assert "container 'c': terminated" in parts[1]
        assert parts[-1] == "condition Ready: down"


# ============================================================
# extract_pod_snapshot
# ============================================================


class TestExtractPodSnapshot:
    """Returns (pod_index, snapshot) when label-gated; None otherwise."""

    def test_happy_path_workers_with_pod_index(self) -> None:
        pod = make_pod(
            name="aiperf-worker-0-0-abc",
            labels={
                JobSetLabels.REPLICATED_JOB_NAME: "workers",
                JobSetLabels.POD_INDEX: "3",
            },
            phase="Running",
        )
        result = extract_pod_snapshot(pod)
        assert result is not None
        pod_index, (pod_name, phase, cs_dicts, status_dict) = result
        assert pod_index == "3"
        assert pod_name == "aiperf-worker-0-0-abc"
        assert phase == PodPhase.RUNNING
        assert cs_dicts == []
        assert status_dict == {"conditions": [], "containerStatuses": []}

    def test_no_replicated_job_label_still_accepted_with_pod_index(self) -> None:
        pod = make_pod(
            labels={JobSetLabels.POD_INDEX: "0"},
            phase="Running",
        )
        result = extract_pod_snapshot(pod)
        assert result is not None
        assert result[0] == "0"

    def test_non_workers_replicated_job_filtered_out(self) -> None:
        pod = make_pod(
            labels={
                JobSetLabels.REPLICATED_JOB_NAME: "controller",
                JobSetLabels.POD_INDEX: "0",
            },
            phase="Running",
        )
        assert extract_pod_snapshot(pod) is None

    def test_missing_pod_index_label_returns_none(self) -> None:
        pod = make_pod(
            labels={JobSetLabels.REPLICATED_JOB_NAME: "workers"},
            phase="Running",
        )
        assert extract_pod_snapshot(pod) is None

    def test_missing_metadata_returns_none(self) -> None:
        # No metadata → labels default to {} → no pod_index → None
        pod = make_pod(metadata=None, phase="Running")
        assert extract_pod_snapshot(pod) is None

    def test_missing_status_yields_unknown_phase_and_empty_lists(self) -> None:
        pod = make_pod(
            labels={JobSetLabels.POD_INDEX: "1"},
            status=None,
        )
        result = extract_pod_snapshot(pod)
        assert result is not None
        _, (_, phase, cs_dicts, status_dict) = result
        assert phase == PodPhase.UNKNOWN
        assert cs_dicts == []
        assert status_dict == {"conditions": [], "containerStatuses": []}

    def test_none_phase_falls_back_to_unknown(self) -> None:
        pod = make_pod(
            labels={JobSetLabels.POD_INDEX: "0"},
            phase=None,
        )
        result = extract_pod_snapshot(pod)
        assert result is not None
        assert result[1][1] == PodPhase.UNKNOWN

    def test_unknown_phase_string_raises(self) -> None:
        # PodPhase is a CaseInsensitiveStrEnum — unrecognized value → ValueError
        pod = make_pod(
            labels={JobSetLabels.POD_INDEX: "0"},
            phase="DoesNotExist",
        )
        with pytest.raises(ValueError):
            extract_pod_snapshot(pod)

    @pytest.mark.parametrize(
        "phase_str,expected",
        [
            ("Pending", PodPhase.PENDING),
            ("Running", PodPhase.RUNNING),
            ("Succeeded", PodPhase.SUCCEEDED),
            ("Failed", PodPhase.FAILED),
            ("Unknown", PodPhase.UNKNOWN),
            param("running", PodPhase.RUNNING, id="case-insensitive-lowercase"),
            param("FAILED", PodPhase.FAILED, id="case-insensitive-uppercase"),
        ],
    )  # fmt: skip
    def test_phase_mapping_covers_all_values(
        self, phase_str: str, expected: PodPhase
    ) -> None:
        pod = make_pod(
            labels={JobSetLabels.POD_INDEX: "0"},
            phase=phase_str,
        )
        result = extract_pod_snapshot(pod)
        assert result is not None
        assert result[1][1] == expected

    def test_pod_with_container_statuses_passes_through(self) -> None:
        cs = make_container_status(
            name="control-plane",
            terminated_reason="OOMKilled",
            terminated_exit_code=137,
        )
        pod = make_pod(
            labels={JobSetLabels.POD_INDEX: "0"},
            phase="Failed",
            container_statuses=[cs],
        )
        result = extract_pod_snapshot(pod)
        assert result is not None
        _, (_, _, cs_dicts, status_dict) = result
        assert len(cs_dicts) == 1
        assert cs_dicts[0]["state"]["terminated"]["reason"] == "OOMKilled"
        assert status_dict["containerStatuses"] == cs_dicts

    def test_pod_with_no_metadata_name_defaults_to_unknown(self) -> None:
        pod = make_pod(
            name=None,
            labels={JobSetLabels.POD_INDEX: "0"},
            phase="Running",
        )
        result = extract_pod_snapshot(pod)
        assert result is not None
        assert result[1][0] == "unknown"


# ============================================================
# aggregate_pods_by_index
# ============================================================


class TestAggregatePodsByIndex:
    """Snapshot per pod_index; non-terminal beats terminal when both exist."""

    def test_empty_input_yields_empty_dict(self) -> None:
        assert aggregate_pods_by_index([]) == {}

    def test_skips_pods_filtered_by_extract(self) -> None:
        # No pod_index → filtered out
        ignored = make_pod(labels={JobSetLabels.REPLICATED_JOB_NAME: "workers"})
        # Non-workers replicated_job → filtered out
        controller = make_pod(
            labels={
                JobSetLabels.REPLICATED_JOB_NAME: "controller",
                JobSetLabels.POD_INDEX: "0",
            }
        )
        result = aggregate_pods_by_index([ignored, controller])
        assert result == {}

    def test_single_pod_kept(self) -> None:
        pod = make_pod(
            name="aiperf-worker-0-0-abc",
            labels={JobSetLabels.POD_INDEX: "0"},
            phase="Running",
        )
        result = aggregate_pods_by_index([pod])
        assert "0" in result
        pod_name, phase, _, _ = result["0"]
        assert pod_name == "aiperf-worker-0-0-abc"
        assert phase == PodPhase.RUNNING

    def test_running_replacement_replaces_failed_for_same_index(self) -> None:
        failed = make_pod(
            name="aiperf-worker-0-0-old",
            labels={JobSetLabels.POD_INDEX: "0"},
            phase="Failed",
        )
        running = make_pod(
            name="aiperf-worker-0-0-new",
            labels={JobSetLabels.POD_INDEX: "0"},
            phase="Running",
        )
        result = aggregate_pods_by_index([failed, running])
        assert result["0"][0] == "aiperf-worker-0-0-new"
        assert result["0"][1] == PodPhase.RUNNING

    def test_failed_does_not_overwrite_running_first(self) -> None:
        running = make_pod(
            name="new",
            labels={JobSetLabels.POD_INDEX: "0"},
            phase="Running",
        )
        failed = make_pod(
            name="old",
            labels={JobSetLabels.POD_INDEX: "0"},
            phase="Failed",
        )
        result = aggregate_pods_by_index([running, failed])
        assert result["0"][0] == "new"

    def test_unknown_overwritten_by_running(self) -> None:
        unknown = make_pod(
            name="ghost",
            labels={JobSetLabels.POD_INDEX: "0"},
            phase="Unknown",
        )
        running = make_pod(
            name="alive",
            labels={JobSetLabels.POD_INDEX: "0"},
            phase="Running",
        )
        result = aggregate_pods_by_index([unknown, running])
        assert result["0"][0] == "alive"

    def test_running_does_not_replace_running(self) -> None:
        # When both incumbent and incoming are non-terminal, the first wins.
        first = make_pod(
            name="first",
            labels={JobSetLabels.POD_INDEX: "0"},
            phase="Running",
        )
        second = make_pod(
            name="second",
            labels={JobSetLabels.POD_INDEX: "0"},
            phase="Running",
        )
        result = aggregate_pods_by_index([first, second])
        assert result["0"][0] == "first"

    def test_multiple_indices_kept_independently(self) -> None:
        a = make_pod(name="a", labels={JobSetLabels.POD_INDEX: "0"}, phase="Running")
        b = make_pod(name="b", labels={JobSetLabels.POD_INDEX: "1"}, phase="Failed")
        c = make_pod(name="c", labels={JobSetLabels.POD_INDEX: "2"}, phase="Pending")
        result = aggregate_pods_by_index([a, b, c])
        assert set(result.keys()) == {"0", "1", "2"}
        assert result["0"][0] == "a"
        assert result["1"][0] == "b"
        assert result["2"][0] == "c"
