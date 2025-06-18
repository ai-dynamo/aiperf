#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from aiperf.common.dataset_models import Conversation, Turn
from aiperf.common.enums import CustomDatasetType
from aiperf.services.dataset.composer.custom import CustomDatasetComposer
from aiperf.services.dataset.config import DatasetConfig
from aiperf.services.dataset.loader import (
    MultiTurnDatasetLoader,
    RandomPoolDatasetLoader,
    SingleTurnDatasetLoader,
    TraceDatasetLoader,
)

MOCK_TRACE_CONTENT = """{"timestamp": 0, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
{"timestamp": 10535, "input_length": 672, "output_length": 26, "hash_ids": [46, 47]}
{"timestamp": 27482, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
"""


# Shared fixtures for all test classes
@pytest.fixture
def basic_config(mock_tokenizer_cls) -> DatasetConfig:
    """Basic configuration for testing."""
    return DatasetConfig(
        filename=Path("test_data.jsonl"),
        custom_dataset_type=CustomDatasetType.SINGLE_TURN,
        tokenizer=mock_tokenizer_cls.from_pretrained("gpt2"),
    )


@pytest.fixture
def trace_config(mock_tokenizer_cls) -> DatasetConfig:
    """Configuration for TRACE dataset type."""
    return DatasetConfig(
        filename=Path("trace_data.jsonl"),
        custom_dataset_type=CustomDatasetType.TRACE,
        tokenizer=mock_tokenizer_cls.from_pretrained("gpt2"),
    )


class TestInitialization:
    """Test class for CustomDatasetComposer basic initialization."""

    def test_initialization(self, basic_config):
        """Test that CustomDatasetComposer can be instantiated with valid config."""
        composer = CustomDatasetComposer(basic_config)

        assert composer is not None
        assert isinstance(composer, CustomDatasetComposer)

    def test_config_storage(self, basic_config):
        """Test that the config is properly stored."""
        composer = CustomDatasetComposer(basic_config)

        assert composer.config is basic_config
        assert composer.config.filename == Path("test_data.jsonl")
        assert composer.config.custom_dataset_type == CustomDatasetType.SINGLE_TURN


class TestCoreFunctionality:
    """Test class for CustomDatasetComposer core functionality."""

    @pytest.mark.parametrize(
        "dataset_type,expected_instance",
        [
            (CustomDatasetType.SINGLE_TURN, SingleTurnDatasetLoader),
            (CustomDatasetType.MULTI_TURN, MultiTurnDatasetLoader),
            (CustomDatasetType.RANDOM_POOL, RandomPoolDatasetLoader),
            (CustomDatasetType.TRACE, TraceDatasetLoader),
        ],
    )
    def test_create_loader_instance_dataset_types(
        self, basic_config, dataset_type, expected_instance
    ):
        """Test _create_loader_instance with different dataset types."""
        basic_config.custom_dataset_type = dataset_type
        composer = CustomDatasetComposer(basic_config)
        composer._create_loader_instance(dataset_type)
        assert isinstance(composer.loader, expected_instance)

    @patch("aiperf.services.dataset.composer.custom.utils.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_TRACE_CONTENT))
    def test_create_dataset_trace(self, mock_check_file, trace_config):
        """Test that create_dataset returns correct type."""
        composer = CustomDatasetComposer(trace_config)
        conversations = composer.create_dataset()

        assert len(conversations) == 1
        assert len(conversations[0].turns) == 3
        assert isinstance(conversations[0], Conversation)
        assert all(isinstance(turn, Turn) for turn in conversations[0].turns)
        assert all(len(turn.text) == 1 for turn in conversations[0].turns)

    @patch("aiperf.services.dataset.composer.custom.utils.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_TRACE_CONTENT))
    def test_create_dataset_trace_multiple_sessions(
        self, mock_check_file, trace_config
    ):
        """Test that create_dataset returns correct type."""
        composer = CustomDatasetComposer(trace_config)
        conversations = composer.create_dataset()

        assert len(conversations) == 1
        assert len(conversations[0].turns) == 3
        assert isinstance(conversations[0], Conversation)
        assert all(isinstance(turn, Turn) for turn in conversations[0].turns)
        assert all(len(turn.text) == 1 for turn in conversations[0].turns)


class TestErrorHandling:
    """Test class for CustomDatasetComposer error handling scenarios."""

    def test_invalid_dataset_factory_creation(self, basic_config):
        """Test invalid dataset factory creation."""
        composer = CustomDatasetComposer(basic_config)

        with pytest.raises(ValueError):
            composer._create_loader_instance(CustomDatasetType.NONE)

    def test_create_loader_instance_invalid_dataset_type(self, basic_config):
        """Test _create_loader_instance with unsupported dataset type."""
        composer = CustomDatasetComposer(basic_config)

        with pytest.raises(ValueError):
            composer._create_loader_instance(CustomDatasetType.NONE)  # invalid type

    @patch("aiperf.services.dataset.composer.custom.utils.check_file_exists")
    @patch(
        "aiperf.services.dataset.composer.custom.CustomDatasetFactory.create_instance"
    )
    def test_create_dataset_empty_result(
        self, mock_factory, mock_check_file, basic_config
    ):
        """Test create_dataset when loader returns empty data."""
        mock_check_file.return_value = None
        mock_loader = Mock()
        mock_loader.load_dataset.return_value = {}
        mock_loader.convert_to_conversations.return_value = []
        mock_factory.return_value = mock_loader

        composer = CustomDatasetComposer(basic_config)
        result = composer.create_dataset()

        assert isinstance(result, list)
        assert len(result) == 0
