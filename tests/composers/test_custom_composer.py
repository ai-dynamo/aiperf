# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, mock_open, patch

import pytest

from aiperf.common.enums import CustomDatasetType
from aiperf.common.models import Conversation, Turn
from aiperf.dataset import (
    MooncakeTraceDatasetLoader,
    MultiTurnDatasetLoader,
    RandomPoolDatasetLoader,
    SingleTurnDatasetLoader,
)
from aiperf.dataset.composer.custom import CustomDatasetComposer


class TestInitialization:
    """Test class for CustomDatasetComposer basic initialization."""

    def test_initialization(self, custom_config, mock_tokenizer):
        """Test that CustomDatasetComposer can be instantiated with valid config."""
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        assert composer is not None
        assert isinstance(composer, CustomDatasetComposer)

    def test_config_storage(self, custom_config, mock_tokenizer):
        """Test that the config is properly stored."""
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        input_config = composer.config.input
        assert input_config is custom_config.input
        assert input_config.file == "test_data.jsonl"
        assert input_config.custom_dataset_type == CustomDatasetType.SINGLE_TURN


MOCK_TRACE_CONTENT = """{"timestamp": 0, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
{"timestamp": 10535, "input_length": 672, "output_length": 52, "hash_ids": [46, 47]}
{"timestamp": 27482, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
"""


class TestCoreFunctionality:
    """Test class for CustomDatasetComposer core functionality."""

    @pytest.mark.parametrize(
        "dataset_type,expected_instance",
        [
            (CustomDatasetType.SINGLE_TURN, SingleTurnDatasetLoader),
            (CustomDatasetType.MULTI_TURN, MultiTurnDatasetLoader),
            (CustomDatasetType.RANDOM_POOL, RandomPoolDatasetLoader),
            (CustomDatasetType.MOONCAKE_TRACE, MooncakeTraceDatasetLoader),
        ],
    )
    def test_create_loader_instance_dataset_types(
        self, custom_config, dataset_type, expected_instance, mock_tokenizer
    ):
        """Test _create_loader_instance with different dataset types."""
        custom_config.input.custom_dataset_type = dataset_type
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)
        composer._create_loader_instance(dataset_type)
        assert isinstance(composer.loader, expected_instance)

    @patch("aiperf.dataset.composer.custom.utils.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_TRACE_CONTENT))
    def test_create_dataset_trace(self, mock_check_file, trace_config, mock_tokenizer):
        """Test that create_dataset returns correct type."""
        composer = CustomDatasetComposer(trace_config, mock_tokenizer)
        conversations = composer.create_dataset()

        assert len(conversations) == 3
        assert all(isinstance(c, Conversation) for c in conversations)
        assert all(isinstance(turn, Turn) for c in conversations for turn in c.turns)
        assert all(len(turn.texts) == 1 for c in conversations for turn in c.turns)

    @patch("aiperf.dataset.composer.custom.utils.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_TRACE_CONTENT))
    def test_max_tokens_config(self, mock_check_file, trace_config, mock_tokenizer):
        trace_config.input.prompt.output_tokens.mean = 120
        trace_config.input.prompt.output_tokens.stddev = 8.0

        composer = CustomDatasetComposer(trace_config, mock_tokenizer)

        with patch(
            "aiperf.dataset.utils.sample_positive_normal_integer", return_value=20
        ):
            conversations = composer.create_dataset()

        assert len(conversations) > 0
        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.max_tokens == 20

    @patch("aiperf.dataset.composer.custom.utils.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_TRACE_CONTENT))
    @patch("pathlib.Path.iterdir", return_value=[])
    def test_max_tokens_mooncake(
        self, mock_iterdir, mock_check_file, custom_config, mock_tokenizer
    ):
        """Test that max_tokens can be set from the custom file"""
        mock_check_file.return_value = None
        custom_config.input.custom_dataset_type = CustomDatasetType.MOONCAKE_TRACE

        composer = CustomDatasetComposer(custom_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.max_tokens == 52


class TestErrorHandling:
    """Test class for CustomDatasetComposer error handling scenarios."""

    @patch("aiperf.dataset.composer.custom.utils.check_file_exists")
    @patch("aiperf.dataset.composer.custom.CustomDatasetFactory.create_instance")
    def test_create_dataset_empty_result(
        self, mock_factory, mock_check_file, custom_config, mock_tokenizer
    ):
        """Test create_dataset when loader returns empty data."""
        mock_check_file.return_value = None
        mock_loader = Mock()
        mock_loader.load_dataset.return_value = {}
        mock_loader.convert_to_conversations.return_value = []
        mock_factory.return_value = mock_loader

        composer = CustomDatasetComposer(custom_config, mock_tokenizer)
        result = composer.create_dataset()

        assert isinstance(result, list)
        assert len(result) == 0
