# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.scenario import ScenarioSpec, UnknownScenarioError
from aiperf.common.scenario.registry import SCENARIOS, get_scenario
from aiperf.plugin.enums import TimingMode


def test_inferencex_agentx_mvp_registered():
    spec = SCENARIOS["inferencex-agentx-mvp"]
    assert isinstance(spec, ScenarioSpec)
    assert spec.timing_mode == TimingMode.AGENTIC_REPLAY
    assert spec.require_ignore_eos is True
    assert spec.require_use_think_time_only is False
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
    assert spec.inter_turn_delay_cap_seconds is None
    assert spec.trace_idle_gap_cap_seconds is None
    assert spec.system_idle_gap_cap_seconds == 10.0
    assert spec.minimum_profile_metric_coverage_ratio == 0.95
    assert spec.forbid_trace_idle_gap_cap is False
    assert spec.forbid_inter_turn_delay_cap is True


def test_get_scenario_returns_spec():
    spec = get_scenario("inferencex-agentx-mvp")
    assert spec.name == "inferencex-agentx-mvp"


def test_get_scenario_unknown_raises():
    with pytest.raises(UnknownScenarioError) as exc_info:
        get_scenario("nonsense-scenario-v9")
    assert "inferencex-agentx-mvp" in str(exc_info.value)
