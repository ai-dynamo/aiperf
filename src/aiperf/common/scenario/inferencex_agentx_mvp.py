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
        "mooncake_trace",
    ),
    min_benchmark_duration_seconds=900,
    default_benchmark_duration_seconds=1800,
    default_trajectory_start_min_ratio=0.0,
    default_trajectory_start_max_ratio=1.0,
    system_idle_gap_cap_seconds=10.0,
    forbid_inter_turn_delay_cap=True,
    require_cache_bust=CacheBustTarget.FIRST_TURN_PREFIX,
    minimum_profile_metric_coverage_ratio=0.95,
)


AGENTX = INFERENCEX_AGENTX_MVP.model_copy(
    update={
        "name": "agentx",
        "default_benchmark_duration_seconds": 3600,
        "default_trajectory_start_min_ratio": 0.25,
        "default_trajectory_start_max_ratio": 0.75,
        "cli_defaults": {
            "endpoint_type": "chat",
            "streaming": True,
            "stats_interval": 30,
            "random_seed": 42,
            "failed_request_threshold": 0.10,
            "warmup_requests_per_lane": 10,
            "warmup_grace_period": 1800,
            "trace_idle_gap_cap_seconds": 300,
            "use_server_token_count": True,
            "no_gpu_telemetry": True,
            "conversation_num_dataset_entries": 393,
            "slice_duration": 1.0,
        },
        "environment_defaults": {
            "DATASET": {"CONFIGURATION_TIMEOUT": 1800},
            "SERVICE": {"PROFILE_CONFIGURE_TIMEOUT": 1800},
            "UI": {"REALTIME_METRICS_ENABLED": True},
            "HTTP": {"TCP_USER_TIMEOUT": 900000},
        },
    }
)
