# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario config-lock package (adapted from ajc/aiperf-graph-ir).

DEFERRED (not ported): ``context_overflow.py`` and ``submission_outcome.py``
depend on ``Environment.AGENTX`` and the record parser, which do not exist on
ajc/rust yet. Track them as a follow-up.
"""

from aiperf.common.scenario.base import (
    ScenarioLockError,
    ScenarioOutcome,
    ScenarioSpec,
    ScenarioViolation,
    TrajectoryWarmupFailedError,
    UnknownScenarioError,
)
from aiperf.common.scenario.inferencex_agentx_mvp import INFERENCEX_AGENTX_MVP
from aiperf.common.scenario.registry import SCENARIOS, get_scenario
from aiperf.common.scenario.validator import apply_scenario

__all__ = [
    "INFERENCEX_AGENTX_MVP",
    "SCENARIOS",
    "ScenarioLockError",
    "ScenarioOutcome",
    "ScenarioSpec",
    "ScenarioViolation",
    "TrajectoryWarmupFailedError",
    "UnknownScenarioError",
    "apply_scenario",
    "get_scenario",
]
