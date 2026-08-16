# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Agent Trace Replay-specific ScenarioSpec validators and the swe-mini-agent spec."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from aiperf.common.scenario.base import ScenarioSpec
from aiperf.common.scenario.validator import (
    _apply_forbid_open_loop_replay,
    _apply_require_execute_tools,
    _apply_require_graph_format,
    _apply_require_server_token_count,
)
from aiperf.plugin.enums import TimingMode


def _make_spec(
    *,
    forbid_open_loop_replay: bool = False,
    require_server_token_count: bool = False,
    require_graph_format: str | None = None,
    require_execute_tools: bool = False,
) -> ScenarioSpec:
    return ScenarioSpec(
        name="test-agent_trace",
        timing_mode=TimingMode.AGENT_GRAPH,
        require_ignore_eos=False,
        forbid_input_truncation=False,
        require_loader=None,
        min_benchmark_duration_seconds=1,
        forbid_open_loop_replay=forbid_open_loop_replay,
        require_server_token_count=require_server_token_count,
        require_graph_format=require_graph_format,
        require_execute_tools=require_execute_tools,
    )


def _make_run(
    *,
    open_loop_replay: bool | None = None,
    use_server_token_count: bool | None = None,
    graph_format: str | None = None,
    graph_execute_tools: bool | None = None,
) -> Any:
    """Build a minimal mock BenchmarkRun with configurable graph/endpoint state."""
    dataset = MagicMock()
    dataset.model_fields_set = set()

    if open_loop_replay is not None:
        dataset.open_loop_replay = open_loop_replay
        dataset.model_fields_set.add("open_loop_replay")
    else:
        dataset.open_loop_replay = True  # default

    if graph_format is not None:
        dataset.graph_format = graph_format
    else:
        dataset.graph_format = None

    if graph_execute_tools is not None:
        dataset.graph_execute_tools = graph_execute_tools
        dataset.model_fields_set.add("graph_execute_tools")
    else:
        dataset.graph_execute_tools = False

    endpoint = MagicMock()
    endpoint.model_fields_set = set()
    if use_server_token_count is not None:
        endpoint.use_server_token_count = use_server_token_count
        endpoint.model_fields_set.add("use_server_token_count")
    else:
        endpoint.use_server_token_count = False  # default

    run = MagicMock()
    run.cfg.get_default_dataset.return_value = dataset
    run.cfg.endpoint = endpoint
    return run


# ---------------------------------------------------------------------------
# _apply_forbid_open_loop_replay
# ---------------------------------------------------------------------------


def test_forbid_open_loop_replay_noop_when_spec_false() -> None:
    spec = _make_spec(forbid_open_loop_replay=False)
    run = _make_run(open_loop_replay=True)
    violations: list = []
    applied: list = []
    _apply_forbid_open_loop_replay(run, spec, violations, applied)
    assert violations == []
    assert applied == []


def test_forbid_open_loop_replay_accepts_false_value() -> None:
    spec = _make_spec(forbid_open_loop_replay=True)
    run = _make_run(open_loop_replay=False)
    violations: list = []
    applied: list = []
    _apply_forbid_open_loop_replay(run, spec, violations, applied)
    assert violations == []
    assert "forbid_open_loop_replay" in applied


def test_forbid_open_loop_replay_rejects_true_value() -> None:
    spec = _make_spec(forbid_open_loop_replay=True)
    run = _make_run(open_loop_replay=True)
    violations: list = []
    applied: list = []
    _apply_forbid_open_loop_replay(run, spec, violations, applied)
    assert len(violations) == 1
    assert violations[0].flag == "--open-loop-replay"
    assert applied == []


def test_forbid_open_loop_replay_skips_dataset_without_attribute() -> None:
    """Synthetic datasets have no open_loop_replay field; validator must skip."""
    spec = _make_spec(forbid_open_loop_replay=True)
    run = _make_run()
    # Remove the attribute entirely to simulate a dataset type that lacks it
    del run.cfg.get_default_dataset.return_value.open_loop_replay
    violations: list = []
    applied: list = []
    _apply_forbid_open_loop_replay(run, spec, violations, applied)
    assert violations == []


# ---------------------------------------------------------------------------
# _apply_require_server_token_count
# ---------------------------------------------------------------------------


def test_require_server_token_count_noop_when_spec_false() -> None:
    spec = _make_spec(require_server_token_count=False)
    run = _make_run(use_server_token_count=False)
    violations: list = []
    applied: list = []
    _apply_require_server_token_count(run, spec, violations, applied)
    assert violations == []
    assert applied == []


def test_require_server_token_count_already_true() -> None:
    spec = _make_spec(require_server_token_count=True)
    run = _make_run(use_server_token_count=True)
    violations: list = []
    applied: list = []
    _apply_require_server_token_count(run, spec, violations, applied)
    assert violations == []
    assert "server_token_count" in applied


def test_require_server_token_count_auto_fills_when_unset() -> None:
    """When the flag is absent (not in model_fields_set), auto-fill to True."""
    spec = _make_spec(require_server_token_count=True)
    run = _make_run()  # use_server_token_count not explicitly set
    violations: list = []
    applied: list = []
    _apply_require_server_token_count(run, spec, violations, applied)
    assert violations == []
    assert "server_token_count" in applied
    assert run.cfg.endpoint.use_server_token_count is True


def test_require_server_token_count_violation_when_explicit_false() -> None:
    """Explicit --no-use-server-token-count with a spec that requires it must produce a violation."""
    spec = _make_spec(require_server_token_count=True)
    run = _make_run(use_server_token_count=False)
    # Mark it as explicitly set
    run.cfg.endpoint.model_fields_set.add("use_server_token_count")
    violations: list = []
    applied: list = []
    _apply_require_server_token_count(run, spec, violations, applied)
    assert len(violations) == 1
    assert violations[0].flag == "--use-server-token-count"
    assert applied == []


# ---------------------------------------------------------------------------
# _apply_require_execute_tools
# ---------------------------------------------------------------------------


def test_require_execute_tools_noop_when_spec_false() -> None:
    spec = _make_spec(require_execute_tools=False)
    run = _make_run(graph_execute_tools=False)
    violations: list = []
    applied: list = []

    _apply_require_execute_tools(run, spec, violations, applied)

    assert violations == []
    assert applied == []


def test_require_execute_tools_auto_fills_when_unset() -> None:
    spec = _make_spec(require_execute_tools=True)
    run = _make_run()
    violations: list = []
    applied: list = []

    _apply_require_execute_tools(run, spec, violations, applied)

    assert violations == []
    assert run.cfg.get_default_dataset.return_value.graph_execute_tools is True
    assert applied == ["execute_tools"]


def test_require_execute_tools_rejects_explicit_false() -> None:
    spec = _make_spec(require_execute_tools=True)
    run = _make_run(graph_execute_tools=False)
    violations: list = []
    applied: list = []

    _apply_require_execute_tools(run, spec, violations, applied)

    assert len(violations) == 1
    assert violations[0].flag == "--graph-execute-tools"
    assert applied == []


# ---------------------------------------------------------------------------
# _apply_require_graph_format
# ---------------------------------------------------------------------------


def test_require_graph_format_noop_when_spec_none() -> None:
    spec = _make_spec(require_graph_format=None)
    run = _make_run(graph_format=None)
    violations: list = []
    applied: list = []
    _apply_require_graph_format(run, spec, violations, applied)
    assert violations == []
    assert applied == []


def test_require_graph_format_accepts_matching_string() -> None:
    spec = _make_spec(require_graph_format="mini_swe_agent_trace")
    run = _make_run(graph_format="mini_swe_agent_trace")
    violations: list = []
    applied: list = []
    _apply_require_graph_format(run, spec, violations, applied)
    assert violations == []
    assert "graph_format" in applied


def test_require_graph_format_violation_when_none() -> None:
    """No --graph-format when spec requires one must produce a violation."""
    spec = _make_spec(require_graph_format="mini_swe_agent_trace")
    run = _make_run(graph_format=None)
    violations: list = []
    applied: list = []
    _apply_require_graph_format(run, spec, violations, applied)
    assert len(violations) == 1
    assert violations[0].flag == "--graph-format"
    assert applied == []


def test_require_graph_format_violation_when_wrong_format() -> None:
    spec = _make_spec(require_graph_format="mini_swe_agent_trace")
    run = _make_run(graph_format="dynamo_trace")
    violations: list = []
    applied: list = []
    _apply_require_graph_format(run, spec, violations, applied)
    assert len(violations) == 1
    assert violations[0].flag == "--graph-format"
    assert violations[0].required_value == "mini_swe_agent_trace"
    assert applied == []


def test_require_graph_format_accepts_enum_value() -> None:
    """When dataset.graph_format is an enum, str() comparison must still work."""
    spec = _make_spec(require_graph_format="mini_swe_agent_trace")
    run = _make_run()

    class _FakeEnum:
        def __str__(self) -> str:
            return "mini_swe_agent_trace"

    run.cfg.get_default_dataset.return_value.graph_format = _FakeEnum()
    violations: list = []
    applied: list = []
    _apply_require_graph_format(run, spec, violations, applied)
    assert violations == []
    assert "graph_format" in applied


# ---------------------------------------------------------------------------
# swe-mini-agent ScenarioSpec registration
# ---------------------------------------------------------------------------


def test_swe_mini_agent_registered() -> None:
    from aiperf.common.scenario.registry import get_scenario

    spec = get_scenario("swe-mini-agent")
    assert spec.name == "swe-mini-agent"
    assert spec.timing_mode == TimingMode.AGENT_GRAPH
    assert spec.require_streaming is True
    assert spec.forbid_input_truncation is True
    assert spec.forbid_open_loop_replay is True
    assert spec.require_server_token_count is True
    assert spec.require_execute_tools is True
    assert spec.require_graph_format == "mini_swe_agent_trace"
    assert spec.require_ignore_eos is False
    assert spec.require_loader is None
