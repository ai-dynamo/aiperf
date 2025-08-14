# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the dataset processor service.
"""

from unittest.mock import patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.enums import ServiceType
from aiperf.common.messages import ProcessSyntheticDatasetMessage
from aiperf.common.models import Conversation, Turn
from aiperf.dataset import DatasetProcessor
from aiperf.dataset.generator.prompt import PromptGenerator


def create_service(user_config: UserConfig | None = None, filename: str | None = None):
    """Create a dataset processor service."""
    service_config = ServiceConfig(
        service_id="test-service-id",
    )
    user_config = user_config or UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
        )
    )
    user_config.input.file = filename
    return DatasetProcessor(
        service_config=service_config,
        user_config=user_config,
    )


@pytest.mark.asyncio
class TestDatasetProcessorService:
    """
    Tests for dataset processor service functionalities and basic properties.

    This test class extends BaseTestComponentService to leverage common
    component service tests while adding dataset processor specific tests
    for service properties and request handling.
    """

    async def test_service_initialization(self):
        """Test that the dataset processor initializes properly with service configuration."""
        service = create_service()

        assert service.service_type == ServiceType.DATASET_PROCESSOR

    @patch("aiperf.dataset.processor.utils.sample_positive_normal_integer")
    async def test_create_conversations(
        self, mock_sample, synthetic_user_config, mock_tokenizer
    ):
        """Test basic dataset creation with text-only conversations."""
        # Mock the number of turns per conversation
        mock_sample.return_value = 2

        service = create_service(synthetic_user_config)
        service.tokenizer = mock_tokenizer
        service.prompt_generator = PromptGenerator(
            synthetic_user_config.input.prompt, mock_tokenizer
        )

        await service._reset_states(
            ProcessSyntheticDatasetMessage(
                service_id="test-service-id",
                num_conversations=1,
            )
        )

        conversation = await service._create_conversation()

        assert isinstance(conversation, Conversation)
        assert conversation.session_id is not None
        assert len(conversation.turns) == 2  # mocked value

        for turn in conversation.turns:
            assert isinstance(turn, Turn)
            assert len(turn.texts) == 1  # single text field per turn
            assert len(turn.texts[0].contents) == 1  # batch_size = 1
            assert len(turn.images) == 0  # no images
            assert len(turn.audios) == 0  # no audio
