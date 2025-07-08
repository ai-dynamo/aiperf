#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.config import (
    AudioConfig,
    AudioLengthConfig,
    ConversationConfig,
    ImageConfig,
    ImageHeightConfig,
    ImageWidthConfig,
    InputConfig,
    InputTokensConfig,
    PrefixPromptConfig,
    PromptConfig,
    TurnConfig,
    TurnDelayConfig,
)
from aiperf.common.enums import CustomDatasetType


@pytest.fixture
def mock_tokenizer(mock_tokenizer_cls):
    """Mock tokenizer class."""
    return mock_tokenizer_cls.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    )


# ============================================================================
# Synthetic Composer Fixtures
# ============================================================================


@pytest.fixture
def synthetic_config():
    """Basic synthetic configuration for testing."""
    config = InputConfig(
        conversation=ConversationConfig(num=5),
        prompt=PromptConfig(
            input_tokens=InputTokensConfig(mean=10, stddev=2),
        ),
    )
    return config


@pytest.fixture
def image_config():
    """Synthetic configuration with image generation enabled."""
    config = InputConfig(
        conversation=ConversationConfig(num=3),
        prompt=PromptConfig(
            input_tokens=InputTokensConfig(mean=10, stddev=2),
        ),
        image=ImageConfig(
            batch_size=1,
            width=ImageWidthConfig(mean=10),
            height=ImageHeightConfig(mean=10),
        ),
    )
    return config


@pytest.fixture
def audio_config():
    """Synthetic configuration with audio generation enabled."""
    config = InputConfig(
        conversation=ConversationConfig(num=3),
        prompt=PromptConfig(
            input_tokens=InputTokensConfig(mean=10, stddev=2),
        ),
        audio=AudioConfig(
            batch_size=1,
            length=AudioLengthConfig(mean=2),
        ),
    )
    return config


@pytest.fixture
def prefix_prompt_config():
    """Synthetic configuration with prefix prompts enabled."""
    config = InputConfig(
        conversation=ConversationConfig(num=5),
        prompt=PromptConfig(
            input_tokens=InputTokensConfig(mean=10, stddev=2),
            prefix_prompt=PrefixPromptConfig(pool_size=3, length=20),
        ),
    )
    return config


@pytest.fixture
def multimodal_config():
    """Synthetic configuration with multimodal data generation enabled."""
    config = InputConfig(
        conversation=ConversationConfig(num=2),
        prompt=PromptConfig(
            batch_size=2,
            input_tokens=InputTokensConfig(mean=10, stddev=2),
            prefix_prompt=PrefixPromptConfig(pool_size=2, length=15),
        ),
        image=ImageConfig(
            batch_size=2,
            width=ImageWidthConfig(mean=10),
            height=ImageHeightConfig(mean=10),
        ),
        audio=AudioConfig(
            batch_size=2,
            length=AudioLengthConfig(mean=2),
        ),
    )
    return config


@pytest.fixture
def multiturn_config():
    """Synthetic configuration with multiturn settings."""
    config = InputConfig(
        conversation=ConversationConfig(
            num=3,
            turn=TurnConfig(
                mean=2,
                stddev=0,
                delay=TurnDelayConfig(mean=1500, stddev=0),
            ),
        ),
        prompt=PromptConfig(
            input_tokens=InputTokensConfig(mean=10, stddev=2),
            prefix_prompt=PrefixPromptConfig(pool_size=0),
        ),
    )
    return config


# ============================================================================
# Custom Composer Fixtures
# ============================================================================


@pytest.fixture
def custom_config() -> InputConfig:
    """Basic custom configuration for testing."""
    # Use model_construct to bypass validation for testing
    return InputConfig.model_construct(
        file="test_data.jsonl",
        custom_dataset_type=CustomDatasetType.SINGLE_TURN,
    )


@pytest.fixture
def trace_config() -> InputConfig:
    """Configuration for TRACE dataset type."""
    # Use model_construct to bypass validation for testing
    return InputConfig.model_construct(
        file="trace_data.jsonl",
        custom_dataset_type=CustomDatasetType.TRACE,
    )
