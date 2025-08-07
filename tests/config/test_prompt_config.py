# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import (
    InputTokensConfig,
    InputTokensDefaults,
    OutputTokensConfig,
    PrefixPromptConfig,
    PrefixPromptDefaults,
    PromptConfig,
    PromptDefaults,
)


def test_prompt_config_defaults():
    """
    Test the default values of the PromptConfig class.
    """
    config = PromptConfig()
    assert config.batch_size == PromptDefaults.BATCH_SIZE


def test_input_tokens_config_defaults():
    """
    Test the default values of the InputTokensConfig class.

    This test verifies that the InputTokensConfig object is initialized with the correct
    default values as defined in the SyntheticTokensDefaults class.
    """
    config = InputTokensConfig()
    assert config.mean == InputTokensDefaults.MEAN
    assert config.stddev == InputTokensDefaults.STDDEV
    assert config.block_size == InputTokensDefaults.BLOCK_SIZE


def test_input_tokens_config_custom_values():
    """
    Test the InputTokensConfig class with custom values.

    This test verifies that the InputTokensConfig class correctly initializes its attributes
    when provided with a dictionary of custom values.
    """
    custom_values = {
        "mean": 100,
        "stddev": 10.0,
    }
    config = InputTokensConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value


def test_output_tokens_config_defaults():
    """
    Test the default values of the OutputTokensConfig class.

    This test verifies that the OutputTokensConfig object is initialized with the correct
    default values as defined in the OutputTokensDefaults class.
    """
    config = OutputTokensConfig()
    assert config.mean is None
    assert config.stddev is None


def test_output_tokens_config_custom_values():
    """
    Test the OutputTokensConfig class with custom values.

    This test verifies that the OutputTokensConfig class correctly initializes its attributes
    when provided with a dictionary of custom values.
    """
    custom_values = {
        "mean": 100,
        "stddev": 10.0,
    }
    config = OutputTokensConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value


def test_prefix_prompt_config_defaults():
    """
    Test the default values of the PrefixPromptConfig class.

    This test verifies that the PrefixPromptConfig object is initialized with the correct
    default values as defined in the PrefixPromptDefaults class.
    """
    config = PrefixPromptConfig()
    assert config.pool_size == PrefixPromptDefaults.POOL_SIZE
    assert config.length == PrefixPromptDefaults.LENGTH


def test_prefix_prompt_config_custom_values():
    """
    Test the PrefixPromptConfig class with custom values.

    This test verifies that the PrefixPromptConfig class correctly initializes its attributes
    when provided with a dictionary of custom values.
    """
    custom_values = {
        "pool_size": 100,
        "length": 10,
    }
    config = PrefixPromptConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value
