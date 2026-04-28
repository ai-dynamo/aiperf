# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Agentic Code gen tests."""

from __future__ import annotations

import pytest

from aiperf.dataset.agentic_code_gen.distributions import lognormal_from_mean_median
from aiperf.dataset.agentic_code_gen.models import (
    CacheLayerConfig,
    Layer15GroupConfig,
    MixtureDelayConfig,
    ResetConfig,
    SessionDistributionConfig,
    TurnCountConfig,
)


@pytest.fixture
def coding_config() -> SessionDistributionConfig:
    """Default coding config."""
    return SessionDistributionConfig()


@pytest.fixture
def small_config() -> SessionDistributionConfig:
    """Small config for fast tests - low max_prompt_tokens to force resets."""
    return SessionDistributionConfig(
        new_tokens_per_turn=lognormal_from_mean_median(mean=200, median=100),
        generation_length=lognormal_from_mean_median(mean=50, median=30),
        inter_turn_delay=MixtureDelayConfig(
            agentic_fraction=0.7,
            agentic_delay=lognormal_from_mean_median(mean=3_000, median=2_000),
            human_delay=lognormal_from_mean_median(mean=45_000, median=30_000),
        ),
        reset=ResetConfig(base_probability=0.02, context_scaling=2.0),
        max_prompt_tokens=5_000,
        block_size=64,
        cache=CacheLayerConfig(
            layer1_tokens=100,
            layer1_5_tokens=50,
            layer2=lognormal_from_mean_median(mean=200, median=150),
            layer1_5_groups=Layer15GroupConfig(num_groups=5, zipf_alpha=1.2),
        ),
    )


@pytest.fixture
def turns_config() -> SessionDistributionConfig:
    """Explicit-turn config for deterministic turn-count mode tests."""
    return SessionDistributionConfig(
        new_tokens_per_turn=lognormal_from_mean_median(mean=150, median=100),
        generation_length=lognormal_from_mean_median(mean=40, median=30),
        inter_turn_delay=MixtureDelayConfig(
            agentic_fraction=0.7,
            agentic_delay=lognormal_from_mean_median(mean=3_000, median=2_000),
            human_delay=lognormal_from_mean_median(mean=45_000, median=30_000),
        ),
        turns=TurnCountConfig(
            mean=4,
            median=4,
            min=4,
            max=4,
            max_session_attempts=20,
        ),
        max_prompt_tokens=5_000,
        block_size=64,
        cache=CacheLayerConfig(
            layer1_tokens=100,
            layer1_5_tokens=50,
            layer2=lognormal_from_mean_median(mean=200, median=150),
            layer1_5_groups=Layer15GroupConfig(num_groups=5, zipf_alpha=1.2),
        ),
    )
