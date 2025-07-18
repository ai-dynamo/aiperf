# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, mock_open, patch

import pytest

from aiperf.common.enums import CustomDatasetType
from aiperf.common.models import Conversation, Turn
from aiperf.services.dataset.composer.custom import CustomDatasetComposer
from aiperf.services.dataset.loader import (
    MultiTurnDatasetLoader,
    RandomPoolDatasetLoader,
    SingleTurnDatasetLoader,
    TraceDatasetLoader,
)


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

        assert composer.config is custom_config
        assert composer.config.file == "test_data.jsonl"
        assert composer.config.custom_dataset_type == CustomDatasetType.SINGLE_TURN


MOCK_TRACE_CONTENT = """{"timestamp": 0, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
{"timestamp": 10535, "input_length": 672, "output_length": 26, "hash_ids": [46, 47]}
{"timestamp": 27482, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
"""

MOCK_SESSION_TRACE_CONTENT = """{"session_id": "123", "delay": 0, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
{"session_id": "456", "delay": 0, "input_length": 655, "output_length": 52, "hash_ids": [10, 11]}
{"session_id": "123", "delay": 1000, "input_length": 672, "output_length": 26, "hash_ids": [46, 47]}
"""


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
        self, custom_config, dataset_type, expected_instance, mock_tokenizer
    ):
        """Test _create_loader_instance with different dataset types."""
        custom_config.custom_dataset_type = dataset_type
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)
        composer._create_loader_instance(dataset_type)
        assert isinstance(composer.loader, expected_instance)

    @patch("aiperf.services.dataset.composer.custom.utils.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_TRACE_CONTENT))
    def test_create_dataset_trace(self, mock_check_file, trace_config, mock_tokenizer):
        """Test that create_dataset returns correct type."""
        composer = CustomDatasetComposer(trace_config, mock_tokenizer)
        conversations = composer.create_dataset()

        assert len(conversations) == 3
        assert all(isinstance(c, Conversation) for c in conversations)
        assert all(isinstance(turn, Turn) for c in conversations for turn in c.turns)
        assert all(len(turn.text) == 1 for c in conversations for turn in c.turns)

    @patch("aiperf.services.dataset.composer.custom.utils.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_SESSION_TRACE_CONTENT))
    def test_create_dataset_trace_multiple_sessions(
        self, mock_check_file, trace_config, mock_tokenizer
    ):
        """Test that create_dataset returns correct type."""
        composer = CustomDatasetComposer(trace_config, mock_tokenizer)
        conversations = composer.create_dataset()

        assert len(conversations) == 2
        assert conversations[0].session_id == "123"
        assert conversations[1].session_id == "456"
        assert len(conversations[0].turns) == 2
        assert len(conversations[1].turns) == 1


class TestErrorHandling:
    """Test class for CustomDatasetComposer error handling scenarios."""

    @patch("aiperf.services.dataset.composer.custom.utils.check_file_exists")
    @patch(
        "aiperf.services.dataset.composer.custom.CustomDatasetFactory.create_instance"
    )
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
