#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from aiperf.common.config.input.output_tokens.output_tokens_config import (
    OutputTokensConfig,
)
from aiperf.common.config.config_defaults import OutputTokensDefaults


def test_output_tokens_config_defaults():
    """
    Test the default values of the OutputTokensConfig class.

    This test verifies that the OutputTokensConfig object is initialized with the correct
    default values as defined in the ImageDefaults class.
    """
    config = OutputTokensConfig()
    assert config.mean == OutputTokensDefaults.MEAN
    assert config.deterministic == OutputTokensDefaults.DETERMINISTIC
    assert config.stddev == OutputTokensDefaults.STDDEV


def test_output_tokens_config_custom_values():
    """
    Test the OutputTokensConfig class with custom values.

    This test verifies that the OutputTokensConfig class correctly initializes its attributes
    when provided with a dictionary of custom values.
    """
    custom_values = {
        "mean": 100.0,
        "deterministic": True,
        "stddev": 10.0,
    }
    config = OutputTokensConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value
