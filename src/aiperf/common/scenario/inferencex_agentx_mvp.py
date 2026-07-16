# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The inferencex-agentx-mvp scenario spec.

Ported from ``ajc/aiperf-graph-ir:src/aiperf/common/scenario/inferencex_agentx_mvp.py``.

Adaptations for ajc/rust:

* ``timing_mode`` is the plain string marker ``"graph_ir"`` (no
  ``TimingMode.GRAPH_IR`` enum on this tree).
* ``require_cache_bust`` is the plain string marker ``"first_turn_prefix"`` (no
  ``CacheBustTarget`` enum / ``endpoint.cache_bust`` knob on this tree). The
  corresponding lock is a documented skip.
"""

from aiperf.common.scenario.base import ScenarioSpec

INFERENCEX_AGENTX_MVP = ScenarioSpec(
    name="inferencex-agentx-mvp",
    timing_mode="graph_ir",
    require_ignore_eos=True,
    require_streaming=True,
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
    trace_idle_gap_cap_seconds=10.0,
    require_cache_bust="first_turn_prefix",
)
