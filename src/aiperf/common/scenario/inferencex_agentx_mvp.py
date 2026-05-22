# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario.base import ScenarioSpec
from aiperf.plugin.enums import TimingMode

INFERENCEX_AGENTX_MVP = ScenarioSpec(
    name="inferencex-agentx-mvp",
    timing_mode=TimingMode.AGENTIC_REPLAY,
    require_ignore_eos=True,
    forbid_input_truncation=True,
    require_loader=(
        "semianalysis_cc_traces_weka_with_subagents",
        "weka_trace",
        "weka_hf",
    ),
    min_benchmark_duration_seconds=900,
    trace_idle_gap_cap_seconds=60.0,
    require_cache_bust=CacheBustTarget.FIRST_TURN_PREFIX,
)
