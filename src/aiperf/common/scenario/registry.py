# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.scenario.agent_trace_default import AGENT_TRACE_DEFAULT
from aiperf.common.scenario.base import ScenarioSpec, UnknownScenarioError
from aiperf.common.scenario.inferencex_agentx_mvp import INFERENCEX_AGENTX_MVP

SCENARIOS: dict[str, ScenarioSpec] = {
    INFERENCEX_AGENTX_MVP.name: INFERENCEX_AGENTX_MVP,
    AGENT_TRACE_DEFAULT.name: AGENT_TRACE_DEFAULT,
}


def get_scenario(name: str) -> ScenarioSpec:
    if name not in SCENARIOS:
        valid = ", ".join(sorted(SCENARIOS.keys()))
        raise UnknownScenarioError(
            f"Unknown scenario {name!r}. Valid scenarios: {valid}"
        )
    return SCENARIOS[name]
