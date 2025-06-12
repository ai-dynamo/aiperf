#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from aiperf.common.config.config_defaults import (
    SessionsDefaults,
    SessionTurnDelayDefaults,
    SessionTurnsDefaults,
)
from aiperf.common.config.input.sessions.sessions_config import (
    SessionsConfig,
    SessionTurnDelayConfig,
    SessionTurnsConfig,
)


def test_sessions_config_defaults():
    """
    Test the default values of the SessionsConfig class.

    This test verifies that the SessionsConfig object is initialized with the correct
    default values as defined in the SessionsDefaults class.
    """
    config = SessionsConfig()
    assert config.num == SessionsDefaults.NUM
    assert config.turns.mean == SessionTurnsDefaults.MEAN
    assert config.turns.stddev == SessionTurnsDefaults.STDDEV
    assert config.turn_delay.mean == SessionTurnDelayDefaults.MEAN
    assert config.turn_delay.stddev == SessionTurnDelayDefaults.STDDEV
    assert config.turn_delay.ratio == SessionTurnDelayDefaults.RATIO


def test_sessions_config_custom_values():
    """
    Test the SessionsConfig class with custom values.

    This test verifies that the SessionsConfig class correctly initializes its attributes
    when provided with a dictionary of custom values.
    """
    custom_values = {
        "num": 100,
        "turns": SessionTurnsConfig(mean=5.0, stddev=1.0),
        "turn_delay": SessionTurnDelayConfig(mean=10.0, stddev=2.0, ratio=1.5),
    }
    config = SessionsConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value
