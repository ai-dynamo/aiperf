# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin registration test for the agentic_replay timing strategy."""

from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, TimingMode


def test_agentic_replay_enum_value_exists():
    assert TimingMode.AGENTIC_REPLAY == "agentic_replay"


def test_agentic_replay_strategy_class_registered():
    cls = plugins.get_class(PluginType.TIMING_STRATEGY, "agentic_replay")
    assert cls is not None
    assert cls.__name__ == "AgenticReplayStrategy"
