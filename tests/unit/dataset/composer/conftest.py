# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from PIL import Image

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli


def make_run(cli_config: CLIConfig):
    """Build a BenchmarkRun from a v1 CLIConfig fixture for composer tests."""
    return make_run_from_cli(cli_config)


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
def synthetic_config() -> CLIConfig:
    """Basic synthetic configuration for testing."""
    config = CLIConfig(
        model_names=["test-model"],
        conversation_num_dataset_entries=5,
        prompt_input_tokens_mean=10,
        prompt_input_tokens_stddev=2,
    )
    return config


@pytest.fixture
def image_config() -> CLIConfig:
    """Synthetic configuration with image generation enabled."""
    config = CLIConfig(
        model_names=["test-model"],
        conversation_num_dataset_entries=3,
        prompt_input_tokens_mean=10,
        prompt_input_tokens_stddev=2,
        image_batch_size=1,
        image_width_mean=10,
        image_height_mean=10,
    )
    return config


@pytest.fixture
def audio_config() -> CLIConfig:
    """Synthetic configuration with audio generation enabled."""
    config = CLIConfig(
        model_names=["test-model"],
        conversation_num_dataset_entries=3,
        prompt_input_tokens_mean=10,
        prompt_input_tokens_stddev=2,
        audio_batch_size=1,
        audio_length_mean=2,
    )
    return config


@pytest.fixture
def prefix_prompt_config() -> CLIConfig:
    """Synthetic configuration with prefix prompts enabled."""
    config = CLIConfig(
        model_names=["test-model"],
        conversation_num_dataset_entries=5,
        prompt_input_tokens_mean=10,
        prompt_input_tokens_stddev=2,
        prompt_prefix_pool_size=3,
        prompt_prefix_length=20,
    )
    return config


@pytest.fixture
def multimodal_config() -> CLIConfig:
    """Synthetic configuration with multimodal data generation enabled."""
    config = CLIConfig(
        model_names=["test-model"],
        conversation_num_dataset_entries=2,
        prompt_batch_size=2,
        prompt_input_tokens_mean=10,
        prompt_input_tokens_stddev=2,
        prompt_prefix_pool_size=2,
        prompt_prefix_length=15,
        image_batch_size=2,
        image_width_mean=10,
        image_height_mean=10,
        audio_batch_size=2,
        audio_length_mean=2,
    )
    return config


@pytest.fixture
def multiturn_config():
    """Synthetic configuration with multiturn settings."""
    config = CLIConfig(
        model_names=["test-model"],
        conversation_num=3,
        conversation_num_dataset_entries=4,
        conversation_turn_mean=2,
        conversation_turn_stddev=0,
        conversation_turn_delay_mean=1500,
        conversation_turn_delay_stddev=0,
        prompt_input_tokens_mean=10,
        prompt_input_tokens_stddev=2,
    )
    return config


# ============================================================================
# Custom Composer Fixtures
# ============================================================================


@pytest.fixture
def custom_config() -> CLIConfig:
    """Basic custom configuration for testing."""
    # Use model_construct to bypass validation for testing
    return CLIConfig.model_construct(
        model_names=["test-model"],
        input_file="test_data.jsonl",
        custom_dataset_type=CustomDatasetType.SINGLE_TURN,
        conversation_num_dataset_entries=5,
    )


@pytest.fixture
def trace_config() -> CLIConfig:
    """Configuration for TRACE dataset type."""
    # Use model_construct to bypass validation for testing
    return CLIConfig.model_construct(
        model_names=["test-model"],
        input_file="trace_data.jsonl",
        custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
    )
