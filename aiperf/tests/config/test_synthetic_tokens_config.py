#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from aiperf.common.config.config_defaults import SyntheticTokensDefaults
from aiperf.common.config.input.synthetic_tokens.synthetic_tokens_config import (
    SyntheticTokensConfig,
)


def test_synthetic_tokens_config_defaults():
    """
    Test the default values of the SyntheticTokensConfig class.

    This test verifies that the SyntheticTokensConfig object is initialized with the correct
    default values as defined in the SyntheticTokensDefaults class.
    """
    config = SyntheticTokensConfig()
    assert config.mean == SyntheticTokensDefaults.MEAN
    assert config.stddev == SyntheticTokensDefaults.STDDEV


def test_synthetic_tokens_config_custom_values():
    """
    Test the SyntheticTokensConfig class with custom values.

    This test verifies that the SyntheticTokensConfig class correctly initializes its attributes
    when provided with a dictionary of custom values.
    """
    custom_values = {
        "mean": 100,
        "stddev": 10.0,
    }
    config = SyntheticTokensConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value
