# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario.base import ScenarioSpec
from aiperf.plugin.enums import TimingMode

INFERENCEX_AGENTX_MVP = ScenarioSpec(
    name="inferencex-agentx-mvp",
    timing_mode=TimingMode.AGENTIC_REPLAY,
    require_ignore_eos=True,
    require_streaming=True,
    forbid_ignore_trace_delays=True,
    forbid_input_truncation=True,
    require_loader=(
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
    ),
    min_benchmark_duration_seconds=900,
    default_benchmark_duration_seconds=1800,
    default_trajectory_start_min_ratio=0.0,
    default_trajectory_start_max_ratio=1.0,
    system_idle_gap_cap_seconds=10.0,
    forbid_inter_turn_delay_cap=True,
    require_cache_bust=CacheBustTarget.FIRST_TURN_PREFIX,
)
