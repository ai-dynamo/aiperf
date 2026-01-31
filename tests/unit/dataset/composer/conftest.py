# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from PIL import Image

from aiperf.common.config import (
    AudioConfig,
    AudioLengthConfig,
    ConversationConfig,
    EndpointConfig,
    ImageConfig,
    ImageHeightConfig,
    ImageWidthConfig,
    InputConfig,
    InputTokensConfig,
    PrefixPromptConfig,
    PromptConfig,
    TurnConfig,
    TurnDelayConfig,
    UserConfig,
)
from aiperf.plugin.enums import CustomDatasetType


@pytest.fixture(autouse=True)
def mock_image_loading():
    """Mock image loading for all composer tests to avoid filesystem dependencies."""
    with (
        patch("aiperf.dataset.generator.image.glob.glob") as mock_glob,
        patch("aiperf.dataset.generator.image.Image.open") as mock_open,
    ):
        # Return a fake image path
        mock_glob.return_value = ["/fake/path/test_image.jpg"]

        # Create a mock image with copy() method
        mock_image = Mock(spec=Image.Image)
        mock_image.copy.return_value = mock_image

        # Support context manager protocol
        mock_open.return_value.__enter__ = Mock(return_value=mock_image)
        mock_open.return_value.__exit__ = Mock(return_value=None)

        yield


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
def synthetic_config() -> UserConfig:
    """Basic synthetic configuration for testing."""
    config = UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig(
            conversation=ConversationConfig(num_dataset_entries=5),
            prompt=PromptConfig(
                input_tokens=InputTokensConfig(mean=10, stddev=2),
            ),
        ),
    )
    return config


@pytest.fixture
def image_config() -> UserConfig:
    """Synthetic configuration with image generation enabled."""
    config = UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig(
            conversation=ConversationConfig(num_dataset_entries=3),
            prompt=PromptConfig(
                input_tokens=InputTokensConfig(mean=10, stddev=2),
            ),
            image=ImageConfig(
                batch_size=1,
                width=ImageWidthConfig(mean=10),
                height=ImageHeightConfig(mean=10),
            ),
        ),
    )
    return config


@pytest.fixture
def audio_config() -> UserConfig:
    """Synthetic configuration with audio generation enabled."""
    config = UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig(
            conversation=ConversationConfig(num_dataset_entries=3),
            prompt=PromptConfig(
                input_tokens=InputTokensConfig(mean=10, stddev=2),
            ),
            audio=AudioConfig(
                batch_size=1,
                length=AudioLengthConfig(mean=2),
            ),
        ),
    )
    return config


@pytest.fixture
def prefix_prompt_config() -> UserConfig:
    """Synthetic configuration with prefix prompts enabled."""
    config = UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig(
            conversation=ConversationConfig(num_dataset_entries=5),
            prompt=PromptConfig(
                input_tokens=InputTokensConfig(mean=10, stddev=2),
                prefix_prompt=PrefixPromptConfig(pool_size=3, length=20),
            ),
        ),
    )
    return config


@pytest.fixture
def multimodal_config() -> UserConfig:
    """Synthetic configuration with multimodal data generation enabled."""
    config = UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig(
            conversation=ConversationConfig(num_dataset_entries=2),
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
        ),
    )
    return config


@pytest.fixture
def multiturn_config():
    """Synthetic configuration with multiturn settings."""
    config = UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig(
            conversation=ConversationConfig(
                num=3,
                num_dataset_entries=4,
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
        ),
    )
    return config


# ============================================================================
# Custom Composer Fixtures
# ============================================================================


@pytest.fixture
def custom_config() -> UserConfig:
    """Basic custom configuration for testing."""
    # Use model_construct to bypass validation for testing
    return UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig.model_construct(
            file="test_data.jsonl",
            custom_dataset_type=CustomDatasetType.SINGLE_TURN,
            conversation=ConversationConfig(num_dataset_entries=5),
        ),
    )


@pytest.fixture
def trace_config() -> UserConfig:
    """Configuration for TRACE dataset type."""
    # Use model_construct to bypass validation for testing
    return UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig.model_construct(
            file="trace_data.jsonl",
            custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
        ),
    )
