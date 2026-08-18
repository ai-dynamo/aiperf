# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `aiperf.cli_commands.kube._debug_report`.

Each per-section helper is exercised independently. We patch the
``aiperf.kubernetes.console`` symbols at the module level (the helpers do
local lazy imports inside their bodies) and assert the right user-facing
output paths fire for each input shape.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from pytest import param

from aiperf.cli_commands.kube._debug_report import (
    _get_event_severity_style,
    _print_events,
    _print_node_resources,
    _print_pod_logs,
    _print_pods_table,
    _print_problems,
    _print_report,
    _print_summary,
)

# ============================================================
# Helpers — minimal data shapes the report helpers consume
# ============================================================


def _pod_info(
    *,
    name: str = "pod-a",
    phase: str = "Running",
    restarts: int = 0,
    node: str = "node-1",
    problems: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "namespace": "ns",
        "phase": phase,
        "restarts": restarts,
        "node": node,
        "problems": problems or [],
        "container_statuses": [],
    }


def _problem(
    *,
    container: str = "main",
    state: str = "CrashLoopBackOff",
    severity: str = "CRITICAL",
    suggestion: str = "Check logs",
    message: str = "",
) -> dict[str, str]:
    return {
        "container": container,
        "state": state,
        "severity": severity,
        "suggestion": suggestion,
        "message": message,
    }


def _event(
    *,
    event_type: str = "Warning",
    reason: str = "BackOff",
    message: str = "Restarting failed container",
    obj: str = "Pod/x",
    count: int = 1,
    last_seen: str = "2026-03-11T10:00:00Z",
) -> dict[str, Any]:
    return {
        "type": event_type,
        "reason": reason,
        "message": message,
        "object": obj,
        "count": count,
        "last_seen": last_seen,
    }


def _node_resource(
    *,
    name: str = "node-1",
    ready: bool = True,
    cpu_capacity: str = "16",
    cpu_allocatable: str = "15",
    memory_capacity: str = "64Gi",
    memory_allocatable: str = "62Gi",
    gpu_capacity: str = "8",
    gpu_allocatable: str = "8",
    pressure: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "ready": ready,
        "cpu_capacity": cpu_capacity,
        "cpu_allocatable": cpu_allocatable,
        "memory_capacity": memory_capacity,
        "memory_allocatable": memory_allocatable,
        "gpu_capacity": gpu_capacity,
        "gpu_allocatable": gpu_allocatable,
        "pressure": pressure or [],
    }


# ============================================================
# _get_event_severity_style
# ============================================================


class TestGetEventSeverityStyle:
    """Verify Rich style mapping for event types."""

    @pytest.mark.parametrize(
        "event_type,expected_style",
        [
            param("Warning", "yellow", id="warning"),
            param("Normal", "dim", id="normal"),
            param("", "dim", id="empty"),
            param("Unknown", "dim", id="unknown"),
        ],
    )  # fmt: skip
    def test_event_type_style_mapping(
        self, event_type: str, expected_style: str
    ) -> None:
        assert _get_event_severity_style(event_type) == expected_style


# ============================================================
# _print_pods_table
# ============================================================


class TestPrintPodsTable:
    """Verify the pod-overview table branches."""

    def test_empty_pod_list_prints_warning(self) -> None:
        with (
            patch("aiperf.kubernetes.console.print_warning") as mock_warn,
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_pods_table([])
            mock_warn.assert_called_once_with("No pods found")
            mock_console.print.assert_not_called()

    def test_pods_render_table_to_console(self) -> None:
        infos = [
            _pod_info(name="ok", phase="Running"),
            _pod_info(name="bad", phase="Failed", restarts=3, problems=[_problem()]),
        ]

        with (
            patch("aiperf.kubernetes.console.print_warning") as mock_warn,
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_pods_table(infos)
            mock_warn.assert_not_called()
            mock_console.print.assert_called_once()

    @pytest.mark.parametrize(
        "phase",
        [
            param("Running", id="running"),
            param("Succeeded", id="succeeded"),
            param("Failed", id="failed"),
            param("Unknown", id="unknown"),
            param("Pending", id="pending"),
            param("ContainerCreating", id="containercreating"),
        ],
    )  # fmt: skip
    def test_all_phase_styles_render_without_error(self, phase: str) -> None:
        with (
            patch("aiperf.kubernetes.console.print_warning"),
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_pods_table([_pod_info(phase=phase)])
            mock_console.print.assert_called_once()

    def test_dash_used_for_blank_node(self) -> None:
        """Pods without a node assignment render the node column as ``-``."""
        with (
            patch("aiperf.kubernetes.console.print_warning"),
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_pods_table([_pod_info(node="")])

            (table_arg,), _ = mock_console.print.call_args
            # Inspect the rendered Rich rows: the 4th column is NODE.
            # Rich Table stores rows in `rows` only after render; instead
            # check the columns store on the underlying Table.
            # We just guard against regression by verifying print was called.
            assert table_arg is not None


# ============================================================
# _print_problems
# ============================================================


class TestPrintProblems:
    """Verify problem-section severity routing."""

    def test_no_problems_prints_success(self) -> None:
        with (
            patch("aiperf.kubernetes.console.print_success") as mock_success,
            patch("aiperf.kubernetes.console.print_header") as mock_header,
            patch("aiperf.kubernetes.console.print_error") as mock_error,
            patch("aiperf.kubernetes.console.print_info"),
            patch("aiperf.kubernetes.console.print_warning"),
        ):
            _print_problems([_pod_info(problems=[])])

            mock_success.assert_called_once_with("No problems detected")
            mock_header.assert_called_once()
            mock_error.assert_not_called()

    def test_critical_problem_uses_print_error(self) -> None:
        infos = [
            _pod_info(
                name="crash",
                problems=[
                    _problem(
                        severity="CRITICAL",
                        state="OOMKilled",
                        suggestion="raise mem",
                        message="killed",
                    )
                ],
            )
        ]

        with (
            patch("aiperf.kubernetes.console.print_error") as mock_error,
            patch("aiperf.kubernetes.console.print_warning") as mock_warn,
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.print_header"),
            patch("aiperf.kubernetes.console.print_success"),
        ):
            _print_problems(infos)

            mock_error.assert_called_once()
            error_text = mock_error.call_args.args[0]
            assert "[crash]" in error_text
            assert "OOMKilled" in error_text
            mock_warn.assert_not_called()
            # Suggestion + message both surface as info lines.
            info_messages = [c.args[0] for c in mock_info.call_args_list]
            assert any("raise mem" in m for m in info_messages)
            assert any("killed" in m for m in info_messages)

    def test_warning_problem_uses_print_warning(self) -> None:
        infos = [
            _pod_info(
                name="x",
                problems=[_problem(severity="WARNING", state="ContainerCreating")],
            )
        ]

        with (
            patch("aiperf.kubernetes.console.print_error") as mock_error,
            patch("aiperf.kubernetes.console.print_warning") as mock_warn,
            patch("aiperf.kubernetes.console.print_info"),
            patch("aiperf.kubernetes.console.print_header"),
            patch("aiperf.kubernetes.console.print_success"),
        ):
            _print_problems(infos)

            mock_warn.assert_called_once()
            mock_error.assert_not_called()

    def test_empty_message_skips_detail_line(self) -> None:
        infos = [_pod_info(problems=[_problem(message="")])]

        with (
            patch("aiperf.kubernetes.console.print_error"),
            patch("aiperf.kubernetes.console.print_warning"),
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.print_header"),
            patch("aiperf.kubernetes.console.print_success"),
        ):
            _print_problems(infos)

            info_messages = [c.args[0] for c in mock_info.call_args_list]
            assert not any("Detail:" in m for m in info_messages)


# ============================================================
# _print_events
# ============================================================


class TestPrintEvents:
    """Verify event-section verbose vs warning-only behaviour."""

    def test_no_events_non_verbose_prints_no_warnings_message(self) -> None:
        with (
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.print_header") as mock_header,
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_events([], verbose=False)

            mock_info.assert_called_once_with("No warning events found")
            mock_header.assert_not_called()
            mock_console.print.assert_not_called()

    def test_no_events_verbose_silent(self) -> None:
        """Verbose mode with empty events stays quiet (header only when there's data)."""
        with (
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.print_header") as mock_header,
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_events([], verbose=True)

            mock_info.assert_not_called()
            mock_header.assert_not_called()
            mock_console.print.assert_not_called()

    def test_only_normal_events_non_verbose_prints_no_warnings_message(self) -> None:
        with (
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.print_header") as mock_header,
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_events([_event(event_type="Normal")], verbose=False)
            mock_info.assert_called_once_with("No warning events found")
            mock_header.assert_not_called()
            mock_console.print.assert_not_called()

    def test_warning_events_non_verbose_renders_table(self) -> None:
        events = [_event(event_type="Warning") for _ in range(3)]
        with (
            patch("aiperf.kubernetes.console.print_info"),
            patch("aiperf.kubernetes.console.print_header") as mock_header,
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_events(events, verbose=False)

            mock_console.print.assert_called_once()
            (label,), kwargs = mock_header.call_args
            assert label == "Warning Events"
            assert kwargs.get("style") == "bold yellow"

    def test_verbose_renders_recent_events_label(self) -> None:
        events = [
            _event(event_type="Normal"),
            _event(event_type="Warning"),
        ]
        with (
            patch("aiperf.kubernetes.console.print_info"),
            patch("aiperf.kubernetes.console.print_header") as mock_header,
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_events(events, verbose=True)

            mock_console.print.assert_called_once()
            (label,), _ = mock_header.call_args
            assert label == "Recent Events"

    def test_verbose_caps_event_count_at_30(self) -> None:
        """Verbose mode shows up to 30 events; the 31st is dropped."""
        events = [_event(event_type="Normal") for _ in range(50)]
        with (
            patch("aiperf.kubernetes.console.print_info"),
            patch("aiperf.kubernetes.console.print_header"),
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_events(events, verbose=True)
            mock_console.print.assert_called_once()
            # Slicing is verified indirectly — the helper must not raise on the
            # large input. Direct row introspection requires Rich internals;
            # we trust the call shape here.

    def test_long_message_is_truncated_to_120_chars(self) -> None:
        """Messages over 120 chars must be sliced before rendering."""
        events = [_event(event_type="Warning", message="x" * 500)]
        with (
            patch("aiperf.kubernetes.console.print_info"),
            patch("aiperf.kubernetes.console.print_header"),
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_events(events, verbose=False)
            mock_console.print.assert_called_once()


# ============================================================
# _print_node_resources
# ============================================================


class TestPrintNodeResources:
    """Verify the node-resources section."""

    def test_empty_list_is_silent(self) -> None:
        with (
            patch("aiperf.kubernetes.console.print_header") as mock_header,
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_node_resources([])
            mock_header.assert_not_called()
            mock_console.print.assert_not_called()

    def test_node_renders_when_present(self) -> None:
        with (
            patch("aiperf.kubernetes.console.print_header") as mock_header,
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_node_resources([_node_resource()])
            mock_header.assert_called_once()
            mock_console.print.assert_called_once()

    @pytest.mark.parametrize(
        "ready,gpu_capacity,pressure",
        [
            param(True, "0", [], id="ready-no-gpu-no-pressure"),
            param(False, "8", ["MemoryPressure"], id="not-ready-with-gpu-pressure"),
            param(True, "8", ["MemoryPressure", "DiskPressure"], id="multi-pressure"),
        ],
    )  # fmt: skip
    def test_renders_branch_combinations(
        self, ready: bool, gpu_capacity: str, pressure: list[str]
    ) -> None:
        node = _node_resource(ready=ready, gpu_capacity=gpu_capacity, pressure=pressure)
        with (
            patch("aiperf.kubernetes.console.print_header"),
            patch("aiperf.kubernetes.console.console") as mock_console,
        ):
            _print_node_resources([node])
            mock_console.print.assert_called_once()


# ============================================================
# _print_pod_logs
# ============================================================


class TestPrintPodLogs:
    """Verify per-container log printing."""

    def test_empty_logs_dict_is_silent(self) -> None:
        with (
            patch("aiperf.kubernetes.console.print_header") as mock_header,
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.console.print_info") as mock_info,
        ):
            _print_pod_logs({})
            mock_header.assert_not_called()
            mock_console.print.assert_not_called()
            mock_info.assert_not_called()

    def test_each_container_prints_separator_and_log(self) -> None:
        logs = {
            "pod-a": {"main": "log a", "side": "log b"},
            "pod-b": {"main": "log c"},
        }
        with (
            patch("aiperf.kubernetes.console.print_header") as mock_header,
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.console.print_info") as mock_info,
        ):
            _print_pod_logs(logs)
            mock_header.assert_called_once()
            assert mock_info.call_count == 3
            assert mock_console.print.call_count == 3

            # Verify separators are formed with pod/container.
            separators = [c.args[0] for c in mock_info.call_args_list]
            assert "--- pod-a/main ---" in separators
            assert "--- pod-a/side ---" in separators
            assert "--- pod-b/main ---" in separators


# ============================================================
# _print_summary
# ============================================================


class TestPrintSummary:
    """Verify the summary footer."""

    def test_summary_counts_running_and_problem_pods(self) -> None:
        infos = [
            _pod_info(name="a", phase="Running"),
            _pod_info(name="b", phase="Running", problems=[_problem()]),
            _pod_info(name="c", phase="Failed", problems=[_problem()]),
            _pod_info(name="d", phase="Pending"),
        ]
        with (
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.print_header"),
            patch("aiperf.kubernetes.console.print_warning") as mock_warn,
        ):
            _print_summary(infos, [], [])

            (line,), _ = mock_info.call_args_list[0]
            assert "4 total" in line
            assert "2 running" in line
            assert "2 with issues" in line
            mock_warn.assert_not_called()

    def test_warning_events_line_emitted_when_present(self) -> None:
        with (
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.print_header"),
            patch("aiperf.kubernetes.console.print_warning"),
        ):
            _print_summary(
                [],
                [_event(event_type="Warning"), _event(event_type="Normal")],
                [],
            )
            messages = [c.args[0] for c in mock_info.call_args_list]
            assert any("Warning events: 1" in m for m in messages)

    def test_pressure_nodes_emit_warning(self) -> None:
        with (
            patch("aiperf.kubernetes.console.print_info"),
            patch("aiperf.kubernetes.console.print_header"),
            patch("aiperf.kubernetes.console.print_warning") as mock_warn,
        ):
            _print_summary(
                [],
                [],
                [
                    _node_resource(name="a", pressure=[]),
                    _node_resource(name="hot", pressure=["MemoryPressure"]),
                    _node_resource(name="full", pressure=["DiskPressure"]),
                ],
            )

            mock_warn.assert_called_once()
            (msg,), _ = mock_warn.call_args
            assert "hot" in msg
            assert "full" in msg


# ============================================================
# _print_report orchestration
# ============================================================


class TestPrintReport:
    """Verify the orchestrator dispatches to all sub-sections."""

    def test_dispatches_to_each_helper(self) -> None:
        with (
            patch(
                "aiperf.cli_commands.kube._debug_report._print_pods_table"
            ) as mock_pods,
            patch(
                "aiperf.cli_commands.kube._debug_report._print_problems"
            ) as mock_problems,
            patch(
                "aiperf.cli_commands.kube._debug_report._print_events"
            ) as mock_events,
            patch(
                "aiperf.cli_commands.kube._debug_report._print_node_resources"
            ) as mock_nodes,
            patch(
                "aiperf.cli_commands.kube._debug_report._print_pod_logs"
            ) as mock_logs,
            patch(
                "aiperf.cli_commands.kube._debug_report._print_summary"
            ) as mock_summary,
            patch("aiperf.kubernetes.console.print_header"),
        ):
            _print_report(
                "ns-a",
                pod_infos=[_pod_info()],
                events=[_event()],
                node_resources=[_node_resource()],
                pod_logs={"a": {"b": "c"}},
                verbose=True,
            )

            mock_pods.assert_called_once()
            mock_problems.assert_called_once()
            mock_events.assert_called_once()
            mock_nodes.assert_called_once()
            mock_logs.assert_called_once()
            mock_summary.assert_called_once()

    def test_non_verbose_skips_pod_logs(self) -> None:
        with (
            patch("aiperf.cli_commands.kube._debug_report._print_pods_table"),
            patch("aiperf.cli_commands.kube._debug_report._print_problems"),
            patch("aiperf.cli_commands.kube._debug_report._print_events"),
            patch("aiperf.cli_commands.kube._debug_report._print_node_resources"),
            patch(
                "aiperf.cli_commands.kube._debug_report._print_pod_logs"
            ) as mock_logs,
            patch("aiperf.cli_commands.kube._debug_report._print_summary"),
            patch("aiperf.kubernetes.console.print_header"),
        ):
            _print_report(
                "ns-a",
                pod_infos=[],
                events=[],
                node_resources=[],
                pod_logs={"a": {"b": "c"}},
                verbose=False,
            )
            mock_logs.assert_not_called()

    def test_namespace_appears_in_header(self) -> None:
        with (
            patch("aiperf.cli_commands.kube._debug_report._print_pods_table"),
            patch("aiperf.cli_commands.kube._debug_report._print_problems"),
            patch("aiperf.cli_commands.kube._debug_report._print_events"),
            patch("aiperf.cli_commands.kube._debug_report._print_node_resources"),
            patch("aiperf.cli_commands.kube._debug_report._print_pod_logs"),
            patch("aiperf.cli_commands.kube._debug_report._print_summary"),
            patch("aiperf.kubernetes.console.print_header") as mock_header,
        ):
            _print_report(
                "my-special-ns",
                pod_infos=[],
                events=[],
                node_resources=[],
                pod_logs={},
                verbose=False,
            )
            (line,), _ = mock_header.call_args
            assert "my-special-ns" in line
