# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.config import (
    InputTokensConfig,
    InputTokensDefaults,
    OutputTokensConfig,
    OutputTokensDefaults,
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
    assert config.stddev is OutputTokensDefaults.STDDEV


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


def test_prompt_config_sequence_distribution_defaults():
    """Test that sequence_distribution defaults to None."""
    config = PromptConfig()
    assert config.sequence_distribution is None
    assert config.get_sequence_distribution() is None


def test_prompt_config_sequence_distribution_valid():
    """Test setting a valid sequence distribution."""
    config = PromptConfig()
    config.sequence_distribution = "256,128:60;512,256:40"

    # Should not raise an exception during validation
    assert config.sequence_distribution == "256,128:60;512,256:40"

    # Should return a proper distribution object
    dist = config.get_sequence_distribution()
    assert dist is not None
    assert len(dist.pairs) == 2


def test_prompt_config_sequence_distribution_invalid_format():
    """Test that invalid sequence distribution formats are rejected."""
    with pytest.raises(ValueError, match="Invalid sequence distribution format"):
        PromptConfig(sequence_distribution="invalid_format")


def test_prompt_config_sequence_distribution_invalid_probabilities():
    """Test that invalid probability sums are rejected."""
    with pytest.raises(ValueError, match="Invalid sequence distribution format"):
        PromptConfig(sequence_distribution="256,128:30;512,256:40")  # Sum = 70


def test_prompt_config_get_sequence_distribution_with_stddev():
    """Test getting sequence distribution with standard deviations."""
    config = PromptConfig()
    config.sequence_distribution = "256|10,128|5:60;512|20,256|15:40"

    dist = config.get_sequence_distribution()
    assert dist is not None
    assert len(dist.pairs) == 2
    assert dist.pairs[0].input_seq_len_stddev == 10.0
    assert dist.pairs[0].output_seq_len_stddev == 5.0


def test_prompt_config_sequence_distribution_none_handling():
    """Test that None sequence_distribution is handled correctly."""
    config = PromptConfig(sequence_distribution=None)
    assert config.sequence_distribution is None
    assert config.get_sequence_distribution() is None
