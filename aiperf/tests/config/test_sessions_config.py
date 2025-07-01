#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import (
    ConversationConfig,
    ConversationDefaults,
    TurnConfig,
    TurnDefaults,
    TurnDelayConfig,
    TurnDelayDefaults,
)


def test_sessions_config_defaults():
    """
    Test the default values of the SessionsConfig class.

    This test verifies that the SessionsConfig object is initialized with the correct
    default values as defined in the SessionsDefaults class.
    """
    config = ConversationConfig()
    assert config.num == ConversationDefaults.NUM
    assert config.turn.mean == TurnDefaults.MEAN
    assert config.turn.stddev == TurnDefaults.STDDEV
    assert config.turn.delay.mean == TurnDelayDefaults.MEAN
    assert config.turn.delay.stddev == TurnDelayDefaults.STDDEV
    assert config.turn.delay.ratio == TurnDelayDefaults.RATIO


def test_sessions_config_custom_values():
    """
    Test the SessionsConfig class with custom values.

    This test verifies that the SessionsConfig class correctly initializes its attributes
    when provided with a dictionary of custom values.
    """
    custom_values = {
        "num": 100,
        "turn": TurnConfig(
            mean=5.0,
            stddev=1.0,
            delay=TurnDelayConfig(mean=10.0, stddev=2.0, ratio=1.5),
        ),
    }
    config = ConversationConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value
