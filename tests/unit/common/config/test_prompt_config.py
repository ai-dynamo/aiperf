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
    assert config.block_size is None


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


def test_prompt_config_random_range_ratio_defaults():
    """Test that random_range_ratio defaults to None."""
    config = PromptConfig()
    assert config.random_range_ratio is None


def test_prompt_config_random_range_ratio_float_builds_distribution():
    """A plain float string produces a RangeRatioDistribution applied to both dims."""
    import math

    from aiperf.common.models.sequence_distribution import RangeRatioDistribution

    config = PromptConfig(
        random_range_ratio="0.3",
        input_tokens=InputTokensConfig(mean=1024),
        output_tokens=OutputTokensConfig(mean=128),
    )
    dist = config.get_sequence_distribution()
    assert isinstance(dist, RangeRatioDistribution)
    assert dist.input_bounds == (math.floor(1024 * 0.7), math.ceil(1024 * 1.3))
    assert dist.output_bounds == (math.floor(128 * 0.7), math.ceil(128 * 1.3))


def test_prompt_config_random_range_ratio_json_dict_builds_distribution():
    """A JSON dict sets input and output ratios independently."""
    from aiperf.common.models.sequence_distribution import RangeRatioDistribution

    config = PromptConfig(
        random_range_ratio='{"input": 0.2, "output": 0.5}',
        input_tokens=InputTokensConfig(mean=1000),
        output_tokens=OutputTokensConfig(mean=100),
    )
    dist = config.get_sequence_distribution()
    assert isinstance(dist, RangeRatioDistribution)
    assert dist.input_bounds == (800, 1200)
    assert dist.output_bounds == (50, 150)


def test_prompt_config_random_range_ratio_defaults_osl_to_128():
    """When --osl is not set, OSL mean defaults to 128 (vllm parity)."""
    from aiperf.common.models.sequence_distribution import RangeRatioDistribution

    config = PromptConfig(
        random_range_ratio="0.0",
        input_tokens=InputTokensConfig(mean=1024),
    )
    dist = config.get_sequence_distribution()
    assert isinstance(dist, RangeRatioDistribution)
    assert dist.output_bounds == (128, 128)


def test_prompt_config_random_range_ratio_invalid_value_rejected():
    """Bad ratio value is rejected at validation time, not on first use."""
    with pytest.raises(ValueError, match="Invalid --random-range-ratio value"):
        PromptConfig(random_range_ratio="1.5")


def test_prompt_config_random_range_ratio_conflicts_with_seq_dist():
    """Setting both --random-range-ratio and --seq-dist is rejected."""
    with pytest.raises(ValueError, match="cannot be combined with --seq-dist"):
        PromptConfig(
            random_range_ratio="0.3",
            sequence_distribution="256,128:100",
        )


def test_prompt_config_random_range_ratio_conflicts_with_isl_stddev():
    """Setting both --random-range-ratio and --isl-stddev > 0 is rejected."""
    with pytest.raises(ValueError, match="cannot be combined with --isl-stddev"):
        PromptConfig(
            random_range_ratio="0.3",
            input_tokens=InputTokensConfig(mean=1024, stddev=10.0),
        )


def test_prompt_config_random_range_ratio_conflicts_with_osl_stddev():
    """Setting both --random-range-ratio and --osl-stddev > 0 is rejected."""
    with pytest.raises(ValueError, match="cannot be combined with --osl-stddev"):
        PromptConfig(
            random_range_ratio="0.3",
            output_tokens=OutputTokensConfig(mean=128, stddev=5.0),
        )


def test_prompt_config_random_range_ratio_zero_stddev_is_allowed():
    """stddev=0 is the default; it must not trip the mutual-exclusion check."""
    config = PromptConfig(
        random_range_ratio="0.3",
        input_tokens=InputTokensConfig(mean=1024, stddev=0.0),
        output_tokens=OutputTokensConfig(mean=128, stddev=0.0),
    )
    assert config.random_range_ratio == "0.3"


def test_prompt_config_random_range_ratio_mode_defaults_to_vllm():
    from aiperf.common.enums import RangeRatioMode

    config = PromptConfig()
    assert config.random_range_ratio_mode == RangeRatioMode.VLLM


def test_prompt_config_random_range_ratio_sglang_mode_builds_sglang_distribution():
    from aiperf.common.enums import RangeRatioMode
    from aiperf.common.models.sequence_distribution import RangeRatioDistribution

    config = PromptConfig(
        random_range_ratio="0.5",
        random_range_ratio_mode=RangeRatioMode.SGLANG,
        input_tokens=InputTokensConfig(mean=1024),
        output_tokens=OutputTokensConfig(mean=128),
    )
    dist = config.get_sequence_distribution()
    assert isinstance(dist, RangeRatioDistribution)
    assert dist.mode == RangeRatioMode.SGLANG
    assert dist.input_bounds == (512, 1024)
    assert dist.output_bounds == (64, 128)


def test_prompt_config_random_range_ratio_sglang_mode_allows_ratio_one():
    """sglang mode accepts r=1.0 (fixed at mean); vllm mode rejects it."""
    from aiperf.common.enums import RangeRatioMode

    config = PromptConfig(
        random_range_ratio="1.0",
        random_range_ratio_mode=RangeRatioMode.SGLANG,
        input_tokens=InputTokensConfig(mean=1024),
        output_tokens=OutputTokensConfig(mean=128),
    )
    dist = config.get_sequence_distribution()
    assert dist.input_bounds == (1024, 1024)
    assert dist.output_bounds == (128, 128)


def test_prompt_config_random_range_ratio_vllm_mode_rejects_ratio_one():
    with pytest.raises(ValueError, match=r"\[0, 1\) for vllm mode"):
        PromptConfig(
            random_range_ratio="1.0",
            input_tokens=InputTokensConfig(mean=1024),
            output_tokens=OutputTokensConfig(mean=128),
        )
