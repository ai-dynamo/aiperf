# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the tool-time JSON artifact written by AgentGraphReplayStrategy."""

import json

from aiperf.timing.strategies.agent_graph_replay import _write_tool_time_artifact


def test_write_tool_time_artifact_creates_valid_json(tmp_path):
    path = tmp_path / "profile_export_graph_tool_time.json"
    durations = [0.038, 0.036, 0.040, 0.034, 0.033]
    _write_tool_time_artifact(path, durations=durations, traces=1, backend="local")
    assert path.exists()
    data = json.loads(path.read_bytes())
    assert data["command_count"] == 5
    assert data["trace_count"] == 1
    assert data["backend"] == "local"
    assert abs(data["total_s"] - sum(durations)) < 1e-9
    assert abs(data["mean_s"] - sum(durations) / 5) < 1e-9
    assert "median_s" in data
    assert "max_s" in data
    assert "durations_s" in data
    assert data["durations_s"] == durations


def test_write_tool_time_artifact_docker_backend(tmp_path):
    path = tmp_path / "profile_export_graph_tool_time.json"
    _write_tool_time_artifact(
        path,
        durations=[0.1, 0.2],
        traces=2,
        backend="docker:agent-trace-pinchbench:latest",
    )
    data = json.loads(path.read_bytes())
    assert data["backend"] == "docker:agent-trace-pinchbench:latest"
    assert data["trace_count"] == 2
    assert data["command_count"] == 2


def test_write_tool_time_artifact_single_command(tmp_path):
    path = tmp_path / "profile_export_graph_tool_time.json"
    _write_tool_time_artifact(path, durations=[0.050], traces=1, backend="local")
    data = json.loads(path.read_bytes())
    assert abs(data["median_s"] - 0.050) < 1e-9
    assert abs(data["max_s"] - 0.050) < 1e-9
    assert data["command_count"] == 1


def test_write_tool_time_artifact_many_commands_sorted_order(tmp_path):
    """Durations_s in the artifact preserve insertion order, not sorted order."""
    path = tmp_path / "profile_export_graph_tool_time.json"
    durations = [0.3, 0.1, 0.5, 0.2]
    _write_tool_time_artifact(path, durations=durations, traces=2, backend="local")
    data = json.loads(path.read_bytes())
    assert data["durations_s"] == durations
    assert abs(data["max_s"] - 0.5) < 1e-9
    # median of [0.1, 0.2, 0.3, 0.5] is 0.25
    assert abs(data["median_s"] - 0.25) < 1e-9
