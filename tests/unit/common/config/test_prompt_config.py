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
    assert config.prefix_reuse_rate == PrefixPromptDefaults.PREFIX_REUSE_RATE


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


def test_prefix_prompt_config_prefix_reuse_rate_custom_value():
    """
    Test the PrefixPromptConfig class with a custom prefix_reuse_rate value.
    """
    config = PrefixPromptConfig(prefix_reuse_rate=0.5)
    assert config.prefix_reuse_rate == 0.5
    assert config.pool_size == 0  # pool_size should remain 0


def test_prefix_prompt_config_prefix_reuse_rate_conflicts_with_pool_size():
    """
    Test that using prefix_reuse_rate with pool_size raises a validation error.
    """
    with pytest.raises(
        ValueError,
        match="Cannot use --prefix-reuse-rate with --prefix-prompt-pool-size",
    ):
        PrefixPromptConfig(prefix_reuse_rate=0.5, pool_size=5)


def test_prefix_prompt_config_prefix_reuse_rate_zero_with_pool_size():
    """
    Test that prefix_reuse_rate=0 is allowed with pool_size > 0 (no conflict).
    """
    config = PrefixPromptConfig(prefix_reuse_rate=0.0, pool_size=5, length=100)
    assert config.prefix_reuse_rate == 0.0
    assert config.pool_size == 5


def test_prefix_prompt_config_pool_size_zero_with_prefix_reuse_rate():
    """
    Test that pool_size=0 is allowed with prefix_reuse_rate > 0 (no conflict).
    """
    config = PrefixPromptConfig(prefix_reuse_rate=0.5, pool_size=0)
    assert config.prefix_reuse_rate == 0.5
    assert config.pool_size == 0


def test_prefix_prompt_config_prefix_reuse_rate_bounds():
    """
    Test that prefix_reuse_rate is bounded between 0.0 and 1.0.
    """
    # Valid values
    config = PrefixPromptConfig(prefix_reuse_rate=0.0)
    assert config.prefix_reuse_rate == 0.0

    config = PrefixPromptConfig(prefix_reuse_rate=1.0)
    assert config.prefix_reuse_rate == 1.0

    # Invalid values
    with pytest.raises(ValueError):
        PrefixPromptConfig(prefix_reuse_rate=-0.1)

    with pytest.raises(ValueError):
        PrefixPromptConfig(prefix_reuse_rate=1.1)


def test_prompt_config_prefix_reuse_rate_requires_isl():
    """
    Test that prefix_reuse_rate requires ISL (input_tokens.mean) to be set.
    """
    # Should fail when prefix_reuse_rate > 0 but ISL mean = 0
    config = PromptConfig()
    config.input_tokens.mean = 0
    config.prefix_prompt.prefix_reuse_rate = 0.5

    with pytest.raises(
        ValueError, match="When using --prefix-reuse-rate, you must also specify --isl"
    ):
        config.model_validate(config.model_dump())

    # Should succeed when both are set
    config = PromptConfig()
    config.input_tokens.mean = 1000
    config.prefix_prompt.prefix_reuse_rate = 0.5
    validated = config.model_validate(config.model_dump())
    assert validated.prefix_prompt.prefix_reuse_rate == 0.5
    assert validated.input_tokens.mean == 1000

    # Should succeed when prefix_reuse_rate = 0 (feature disabled)
    config = PromptConfig()
    config.input_tokens.mean = 0
    config.prefix_prompt.prefix_reuse_rate = 0.0
    validated = config.model_validate(config.model_dump())
    assert validated.prefix_prompt.prefix_reuse_rate == 0.0
    assert validated.input_tokens.mean == 0


def test_prefix_prompt_config_prefix_reuse_rate_conflicts_with_length():
    """
    Test that using prefix_reuse_rate with prefix_prompt_length raises a validation error.
    """
    with pytest.raises(
        ValueError, match="Cannot use --prefix-reuse-rate with --prefix-prompt-length"
    ):
        PrefixPromptConfig(prefix_reuse_rate=0.5, length=100)


def test_prefix_prompt_config_prefix_reuse_rate_zero_with_length():
    """
    Test that prefix_reuse_rate=0 is allowed with length > 0 (no conflict).
    """
    config = PrefixPromptConfig(prefix_reuse_rate=0.0, length=100)
    assert config.prefix_reuse_rate == 0.0
    assert config.length == 100
