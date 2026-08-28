# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for random_range_ratio fields on the v2 PromptConfig."""

import math

import pytest

from aiperf.config.dataset.content import PromptConfig
from aiperf.config.types import FixedDistribution


def test_prompt_config_random_range_ratio_defaults_to_none():
    """Test that random_range_ratio defaults to None."""
    config = PromptConfig()
    assert config.random_range_ratio is None


def test_prompt_config_random_corpus_style_defaults_to_vllm():
    from aiperf.common.enums import RandomCorpusStyle

    config = PromptConfig()
    assert config.random_corpus_style == RandomCorpusStyle.VLLM


def test_prompt_config_random_range_ratio_float_builds_distribution():
    """A plain float string produces a RangeRatioDistribution applied to both dims."""
    from aiperf.common.models.sequence_distribution import RangeRatioDistribution

    config = PromptConfig(
        random_range_ratio="0.3",
        isl=FixedDistribution(value=1024),
        osl=FixedDistribution(value=128),
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
        isl=FixedDistribution(value=1000),
        osl=FixedDistribution(value=100),
    )
    dist = config.get_sequence_distribution()
    assert isinstance(dist, RangeRatioDistribution)
    assert dist.input_bounds == (800, 1200)
    assert dist.output_bounds == (50, 150)


def test_prompt_config_random_range_ratio_requires_osl():
    """--random-range-ratio without --osl raises at parse time."""
    with pytest.raises(ValueError, match="--osl"):
        PromptConfig(
            random_range_ratio="0.0",
            isl=FixedDistribution(value=1024),
        )


def test_prompt_config_random_range_ratio_requires_isl():
    """--random-range-ratio without --isl raises at parse time."""
    with pytest.raises(ValueError, match="--isl"):
        PromptConfig(
            random_range_ratio="0.0",
            osl=FixedDistribution(value=128),
        )


def test_prompt_config_random_range_ratio_rejects_isl_stddev():
    """--isl-stddev with --random-range-ratio is rejected at validation time."""
    from aiperf.config.distributions import NormalDistribution

    with pytest.raises(ValueError, match="--isl-stddev"):
        PromptConfig(
            random_range_ratio="0.3",
            isl=NormalDistribution(mean=128, stddev=20),
            osl=FixedDistribution(value=128),
        )


def test_prompt_config_random_range_ratio_rejects_osl_stddev():
    """--osl-stddev with --random-range-ratio is rejected at validation time."""
    from aiperf.config.distributions import NormalDistribution

    with pytest.raises(ValueError, match="--osl-stddev"):
        PromptConfig(
            random_range_ratio="0.3",
            isl=FixedDistribution(value=128),
            osl=NormalDistribution(mean=128, stddev=20),
        )


def test_prompt_config_random_range_ratio_invalid_value_rejected():
    """Bad ratio value is rejected at validation time, not on first use."""
    with pytest.raises(ValueError, match="Invalid random_range_ratio value"):
        PromptConfig(
            random_range_ratio="1.5",
            isl=FixedDistribution(value=128),
            osl=FixedDistribution(value=128),
        )


def test_prompt_config_random_range_ratio_conflicts_with_sequence_distribution():
    """Setting both random_range_ratio and sequence_distribution is rejected."""
    from aiperf.config.types import SequenceDistributionEntry

    with pytest.raises(
        ValueError, match="cannot be combined with sequence_distribution"
    ):
        PromptConfig(
            random_range_ratio="0.3",
            sequence_distribution=[
                SequenceDistributionEntry(
                    isl=FixedDistribution(value=256),
                    osl=FixedDistribution(value=128),
                    probability=100.0,
                )
            ],
        )


def test_prompt_config_random_range_ratio_none_returns_none():
    """When random_range_ratio is None, get_sequence_distribution returns None."""
    config = PromptConfig()
    assert config.get_sequence_distribution() is None


def test_prompt_config_random_range_ratio_zero_is_valid():
    """Ratio of 0.0 is valid (fixed at mean)."""
    config = PromptConfig(
        random_range_ratio="0.0",
        isl=FixedDistribution(value=512),
        osl=FixedDistribution(value=128),
    )
    assert config.random_range_ratio == "0.0"
    dist = config.get_sequence_distribution()
    assert dist is not None
    assert dist.input_bounds == (512, 512)
    assert dist.output_bounds == (128, 128)


def test_prompt_config_random_range_ratio_sglang_mode_builds_sglang_distribution():
    from aiperf.common.enums import RandomCorpusStyle
    from aiperf.common.models.sequence_distribution import RangeRatioDistribution

    config = PromptConfig(
        random_range_ratio="0.5",
        random_corpus_style=RandomCorpusStyle.SGLANG,
        isl=FixedDistribution(value=1024),
        osl=FixedDistribution(value=128),
    )
    dist = config.get_sequence_distribution()
    assert isinstance(dist, RangeRatioDistribution)
    assert dist.mode == RandomCorpusStyle.SGLANG
    assert dist.input_bounds == (512, 1024)
    assert dist.output_bounds == (64, 128)


def test_prompt_config_random_range_ratio_sglang_mode_allows_ratio_one():
    """sglang mode accepts r=1.0 (fixed at mean); vllm mode rejects it."""
    from aiperf.common.enums import RandomCorpusStyle

    config = PromptConfig(
        random_range_ratio="1.0",
        random_corpus_style=RandomCorpusStyle.SGLANG,
        isl=FixedDistribution(value=1024),
        osl=FixedDistribution(value=128),
    )
    dist = config.get_sequence_distribution()
    assert dist.input_bounds == (1024, 1024)
    assert dist.output_bounds == (128, 128)


def test_prompt_config_random_range_ratio_vllm_mode_rejects_ratio_one():
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\)"):
        PromptConfig(
            random_range_ratio="1.0",
            isl=FixedDistribution(value=1024),
            osl=FixedDistribution(value=128),
        )


def test_prompt_config_get_sequence_distribution_passes_num_special_tokens():
    """num_special_tokens is forwarded to RangeRatioDistribution via the style's adjust_mean."""
    from aiperf.common.models.sequence_distribution import RangeRatioDistribution

    config = PromptConfig(
        random_range_ratio="0.3",
        isl=FixedDistribution(value=512),
        osl=FixedDistribution(value=128),
    )
    dist = config.get_sequence_distribution(num_special_tokens=1)
    assert isinstance(dist, RangeRatioDistribution)
    assert dist.input_bounds == (math.floor(511 * 0.7), math.ceil(511 * 1.3))


def test_prompt_config_get_sequence_distribution_default_num_special_tokens_zero():
    """Default num_special_tokens=0 leaves bounds at the configured isl_mean."""
    from aiperf.common.models.sequence_distribution import RangeRatioDistribution

    config = PromptConfig(
        random_range_ratio="0.3",
        isl=FixedDistribution(value=512),
        osl=FixedDistribution(value=128),
    )
    dist_default = config.get_sequence_distribution()
    dist_explicit = config.get_sequence_distribution(num_special_tokens=0)
    assert isinstance(dist_default, RangeRatioDistribution)
    assert dist_default.input_bounds == dist_explicit.input_bounds


def test_prompt_config_random_range_ratio_accepts_native_float():
    """A native Python float (e.g. from YAML) is coerced to string before validation."""
    from aiperf.common.models.sequence_distribution import RangeRatioDistribution

    config = PromptConfig(
        random_range_ratio=0.3,
        isl=FixedDistribution(value=1024),
        osl=FixedDistribution(value=128),
    )
    assert config.random_range_ratio == "0.3"
    dist = config.get_sequence_distribution()
    assert isinstance(dist, RangeRatioDistribution)
    assert dist.input_bounds == (math.floor(1024 * 0.7), math.ceil(1024 * 1.3))


def test_prompt_config_random_range_ratio_accepts_native_dict():
    """A native Python dict (e.g. from YAML) is JSON-serialized before validation."""
    from aiperf.common.models.sequence_distribution import RangeRatioDistribution

    config = PromptConfig(
        random_range_ratio={"input": 0.2, "output": 0.5},
        isl=FixedDistribution(value=1000),
        osl=FixedDistribution(value=100),
    )
    dist = config.get_sequence_distribution()
    assert isinstance(dist, RangeRatioDistribution)
    assert dist.input_bounds == (800, 1200)
    assert dist.output_bounds == (50, 150)
