# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the scenario invariant-lock registry and the inferencex-agentx-mvp spec."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario import (
    INFERENCEX_AGENTX_MVP,
    SCENARIOS,
    ScenarioSpec,
    UnknownScenarioError,
    get_scenario,
)
from aiperf.plugin.enums import TimingMode


def test_get_scenario_returns_inferencex_agentx_mvp_spec() -> None:
    spec = get_scenario("inferencex-agentx-mvp")
    assert spec is INFERENCEX_AGENTX_MVP
    assert isinstance(spec, ScenarioSpec)
    assert spec.name == "inferencex-agentx-mvp"


def test_inferencex_agentx_mvp_field_values() -> None:
    spec = get_scenario("inferencex-agentx-mvp")
    assert spec.timing_mode == TimingMode.GRAPH_IR
    assert spec.require_ignore_eos is True
    assert spec.require_streaming is True
    assert spec.forbid_input_truncation is True
    assert spec.require_loader == (
        "semianalysis_cc_traces_weka_with_subagents",
        "semianalysis_cc_traces_weka_with_subagents_256k",
        "semianalysis_cc_traces_weka_with_subagents_060226",
        "semianalysis_cc_traces_weka_with_subagents_060226_256k",
        "semianalysis_cc_traces_weka_with_subagents_060526",
        "semianalysis_cc_traces_weka_with_subagents_060526_256k",
        "semianalysis_cc_traces_weka_with_subagents_060826",
        "semianalysis_cc_traces_weka_with_subagents_060826_256k",
        "semianalysis_cc_traces_weka_061326",
        "semianalysis_cc_traces_weka_061326_256k",
        "semianalysis_cc_traces_weka_061526",
        "semianalysis_cc_traces_weka_061526_256k",
        "semianalysis_cc_traces_weka_062126",
        "semianalysis_cc_traces_weka_062126_256k",
        "weka_trace",
        "weka_hf",
    )
    assert spec.min_benchmark_duration_seconds == 900
    assert spec.default_benchmark_duration_seconds == 1800
    assert spec.default_trajectory_start_min_ratio == 0.0
    assert spec.default_trajectory_start_max_ratio == 1.0
    assert spec.trace_idle_gap_cap_seconds == 10.0
    assert spec.require_cache_bust == CacheBustTarget.FIRST_TURN_PREFIX


def test_get_scenario_unknown_name_raises_listing_valid_names() -> None:
    with pytest.raises(UnknownScenarioError) as exc_info:
        get_scenario("nope")
    message = str(exc_info.value)
    assert "nope" in message
    assert "inferencex-agentx-mvp" in message


def test_registry_contains_inferencex_agentx_mvp() -> None:
    assert SCENARIOS["inferencex-agentx-mvp"] is INFERENCEX_AGENTX_MVP


def test_scenario_spec_is_frozen() -> None:
    spec = get_scenario("inferencex-agentx-mvp")
    with pytest.raises(ValidationError):
        spec.name = "mutated"
