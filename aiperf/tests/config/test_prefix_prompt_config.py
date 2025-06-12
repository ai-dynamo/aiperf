#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from aiperf.common.config.config_defaults import PrefixPromptDefaults
from aiperf.common.config.input.prefix_prompt.prefix_prompt_config import (
    PrefixPromptConfig,
)


def test_prefix_prompt_config_defaults():
    """
    Test the default values of the PrefixPromptConfig class.

    This test verifies that the PrefixPromptConfig object is initialized with the correct
    default values as defined in the PrefixPromptDefaults class.
    """
    config = PrefixPromptConfig()
    assert config.num == PrefixPromptDefaults.NUM
    assert config.length == PrefixPromptDefaults.LENGTH


def test_prefix_prompt_config_custom_values():
    """
    Test the PrefixPromptConfig class with custom values.

    This test verifies that the PrefixPromptConfig class correctly initializes its attributes
    when provided with a dictionary of custom values.
    """
    custom_values = {
        "num": 100,
        "length": 10,
    }
    config = PrefixPromptConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value
