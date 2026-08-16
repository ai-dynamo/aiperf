# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario.base import ScenarioSpec
from aiperf.plugin.enums import TimingMode

AGENT_TRACE_DEFAULT = ScenarioSpec(
    name="swe-mini-agent",
    timing_mode=TimingMode.AGENT_GRAPH,
    require_ignore_eos=False,
    require_streaming=True,
    forbid_input_truncation=True,
    require_loader=None,
    require_graph_format="mini_swe_agent_trace",
    forbid_open_loop_replay=True,
    require_server_token_count=True,
    # Tool execution: replay must run recorded bash commands, not substitute a timing delay.
    require_execute_tools=True,
    # Warmup: mirrors Agent Trace Replay's per-recording "Reply with exactly: ok" call
    # that primes the server KV cache before the real trajectory begins.
    require_emit_warmup=True,
    # Cache isolation: each trace instance receives a stable system-prefix
    # marker, preventing another trace from warming its KV cache.
    require_cache_bust=CacheBustTarget.SYSTEM_PREFIX,
)
