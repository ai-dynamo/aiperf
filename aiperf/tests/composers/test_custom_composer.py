#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from unittest.mock import mock_open, patch

import pytest

from aiperf.common.dataset_models import Conversation, Turn
from aiperf.common.enums import CustomDatasetType
from aiperf.services.dataset.composer.custom import CustomDatasetComposer
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

MOCK_SESSION_TRACE_CONTENT = """{"session_id": "123", "delay": 0, "input_length": 655, "output_length": 52, "hash_ids": [46, 47]}
{"session_id": "456", "delay": 0, "input_length": 655, "output_length": 52, "hash_ids": [10, 11]}
{"session_id": "123", "delay": 1000, "input_length": 672, "output_length": 26, "hash_ids": [46, 47]}
"""

MOCK_EMPTY_TRACE_CONTENT = """"""


class TestCustomDatasetComposer:
    """Test class for CustomDatasetComposer basic initialization."""

    async def test_initialization(self, custom_config, mock_tokenizer):
        """Test that CustomDatasetComposer can be instantiated with valid config."""
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)

        assert composer.config is custom_config
        assert composer.tokenizer is mock_tokenizer
        assert composer.prompt_generator is None
        assert composer.image_generator is None
        assert composer.audio_generator is None
        assert not composer.composer_initialized.is_set()

        await composer.initialize()

        assert composer.prompt_generator is not None
        assert composer.image_generator is not None
        assert composer.audio_generator is not None
        assert composer.composer_initialized.is_set()
        assert not hasattr(composer, "loader")

    @pytest.mark.parametrize(
        "dataset_type,expected_instance",
        [
            (CustomDatasetType.SINGLE_TURN, SingleTurnDatasetLoader),
            (CustomDatasetType.MULTI_TURN, MultiTurnDatasetLoader),
            (CustomDatasetType.RANDOM_POOL, RandomPoolDatasetLoader),
            (CustomDatasetType.TRACE, TraceDatasetLoader),
        ],
    )
    async def test_create_loader_instance_dataset_types(
        self, custom_config, dataset_type, expected_instance, mock_tokenizer
    ):
        """Test _create_loader_instance with different dataset types."""
        custom_config.custom_dataset_type = dataset_type
        composer = CustomDatasetComposer(custom_config, mock_tokenizer)
        await composer.initialize()
        composer._create_loader_instance(dataset_type)
        assert isinstance(composer.loader, expected_instance)

    # ============================================================================
    # Trace Dataset
    # ============================================================================

    @patch("aiperf.services.dataset.composer.custom.utils.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_TRACE_CONTENT))
    async def test_create_dataset_trace(
        self, mock_check_file, trace_config, mock_tokenizer
    ):
        """Test that create_dataset returns correct type."""
        composer = CustomDatasetComposer(trace_config, mock_tokenizer)
        await composer.initialize()
        conversations = await composer.create_dataset()

        assert len(conversations) == 3
        assert all(isinstance(c, Conversation) for c in conversations)
        assert all(isinstance(turn, Turn) for c in conversations for turn in c.turns)
        assert all(len(turn.texts) == 1 for c in conversations for turn in c.turns)

    @patch("aiperf.services.dataset.composer.custom.utils.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_SESSION_TRACE_CONTENT))
    async def test_create_dataset_trace_multiple_sessions(
        self, mock_check_file, trace_config, mock_tokenizer
    ):
        """Test that create_dataset returns correct type."""
        composer = CustomDatasetComposer(trace_config, mock_tokenizer)
        await composer.initialize()
        conversations = await composer.create_dataset()

        assert len(conversations) == 2
        assert conversations[0].id == "123"
        assert conversations[1].id == "456"
        assert len(conversations[0].turns) == 2
        assert len(conversations[1].turns) == 1

    @patch("aiperf.services.dataset.composer.custom.utils.check_file_exists")
    @patch("builtins.open", mock_open(read_data=MOCK_EMPTY_TRACE_CONTENT))
    async def test_create_dataset_empty_trace(
        self, mock_check_file, trace_config, mock_tokenizer
    ):
        """Test create_dataset when loader returns empty data."""
        composer = CustomDatasetComposer(trace_config, mock_tokenizer)
        await composer.initialize()
        conversations = await composer.create_dataset()

        assert isinstance(conversations, list)
        assert len(conversations) == 0
