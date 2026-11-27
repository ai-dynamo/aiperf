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
        "mooncake_trace",
    )
    assert spec.min_benchmark_duration_seconds == 900
    assert spec.inter_turn_delay_cap_seconds is None
    assert spec.trace_idle_gap_cap_seconds is None
    assert spec.system_idle_gap_cap_seconds == 10.0
    assert spec.minimum_profile_metric_coverage_ratio == 0.95
    assert spec.forbid_inter_turn_delay_cap is True


_AGENTX_PRESET_FIELDS = {
    "name",
    "cli_defaults",
    "environment_defaults",
    "default_benchmark_duration_seconds",
    "default_trajectory_start_min_ratio",
    "default_trajectory_start_max_ratio",
}


def test_agentx_inherits_every_mvp_locked_invariant():
    """A lock edited on AGENTX alone would let a run pass agentx but fail the MVP."""
    agentx = SCENARIOS["agentx"]
    mvp = SCENARIOS["inferencex-agentx-mvp"]
    assert agentx.model_dump(exclude=_AGENTX_PRESET_FIELDS) == mvp.model_dump(
        exclude=_AGENTX_PRESET_FIELDS
    )


def test_agentx_environment_defaults_are_pinned():
    """Runtime timeouts are part of the comparable preset; changing one must be deliberate."""
    assert SCENARIOS["agentx"].environment_defaults == {
        "DATASET": {"CONFIGURATION_TIMEOUT": 1800},
        "SERVICE": {"PROFILE_CONFIGURE_TIMEOUT": 1800},
        "UI": {"REALTIME_METRICS_ENABLED": True},
        "HTTP": {"TCP_USER_TIMEOUT": 900000},
    }


def test_get_scenario_returns_spec():
    spec = get_scenario("inferencex-agentx-mvp")
    assert spec.name == "inferencex-agentx-mvp"


def test_get_scenario_unknown_raises():
    with pytest.raises(UnknownScenarioError) as exc_info:
        get_scenario("nonsense-scenario-v9")
    assert "inferencex-agentx-mvp" in str(exc_info.value)
