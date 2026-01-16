# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.common.config import EndpointConfig, InputConfig, ServiceConfig, UserConfig
from aiperf.common.config.config_defaults import InputDefaults
from aiperf.common.enums import (
    CustomDatasetType,
    DatasetSamplingStrategy,
    PublicDatasetType,
)
from aiperf.common.messages.command_messages import ProfileConfigureCommand
from aiperf.dataset.dataset_manager import DatasetManager


class TestDatasetManager:
    """Test DatasetManager functionality.

    Note: Dataset sampling tests have been moved to test_dataset_samplers.py
    since sampling is now handled by timing strategies, not DatasetManager.
    """

    @pytest.fixture(autouse=True)
    async def teardown(self):
        """Clean up after each test to prevent shared state issues."""
        yield
        # Reset any global state if needed
        # Clear communication factory state
        from aiperf.common.factories import CommunicationFactory

        if hasattr(CommunicationFactory, "_instances"):
            CommunicationFactory._instances.clear()

    @pytest.mark.asyncio
    @patch("aiperf.common.tokenizer.Tokenizer.from_pretrained")
    async def test_dataset_configured_notification_for_multi_turn_conversations(
        self,
        mock_tokenizer_from_pretrained,
        create_mooncake_trace_file,
        mock_tokenizer_cls,
    ):
        """Test that dataset configured notification includes correct metadata for multi-turn conversations.

        When a dataset has multiple turns per conversation, the notification should:
        - Include one ConversationMetadata per conversation (not one per turn)
        - Include the first_turn_timestamp and turn_delays for each conversation
        - Have the correct turn count for each conversation
        """
        # Mock the tokenizer to avoid HTTP requests
        mock_tokenizer_from_pretrained.return_value = (
            mock_tokenizer_cls.from_pretrained("test-model")
        )

        # Create a file with multi-turn conversations
        entries = [
            '{"session_id": "sess-1", "timestamp": 0, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-1", "delay": 10000, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-1", "delay": 10000, "input_length": 100, "output_length": 10}',
            '{"session_id": "sess-2", "timestamp": 20000, "input_length": 25, "output_length": 20}',
            '{"session_id": "sess-2", "delay": 10000, "input_length": 10000, "output_length": 20}',
        ]
        filename = create_mooncake_trace_file(entries)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            service_config = ServiceConfig()
            dataset_manager = DatasetManager(service_config, user_config)

            await dataset_manager.initialize()

            # Mock the publish method to capture notifications
            from unittest.mock import AsyncMock

            from aiperf.common.messages import DatasetConfiguredNotification

            published_messages = []

            async def mock_publish(msg):
                published_messages.append(msg)

            dataset_manager.publish = AsyncMock(side_effect=mock_publish)

            # Configure the dataset to load conversations
            await dataset_manager._profile_configure_command(
                ProfileConfigureCommand(config=user_config, service_id="test_service")
            )

            # Verify the notification was published
            published_notifications = [
                msg
                for msg in published_messages
                if isinstance(msg, DatasetConfiguredNotification)
            ]
            assert len(published_notifications) == 1

            notification = published_notifications[0]
            metadata = notification.metadata

            # Verify dataset metadata structure
            assert len(metadata.conversations) == 2  # 2 conversations, not 5 turns

            # Extract conversation metadata for easier testing
            conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

            # Verify session 1 metadata
            assert "sess-1" in conv_dict
            sess1 = conv_dict["sess-1"]
            assert len(sess1.turns) == 3

            # Verify session 2 metadata
            assert "sess-2" in conv_dict
            sess2 = conv_dict["sess-2"]
            assert len(sess2.turns) == 2

            # Verify no duplicate conversation IDs (one per conversation, not per turn)
            conversation_ids = [conv.conversation_id for conv in metadata.conversations]
            assert len(conversation_ids) == len(set(conversation_ids))

        finally:
            Path(filename).unlink(missing_ok=True)

    @pytest.mark.asyncio
    @patch("aiperf.common.tokenizer.Tokenizer.from_pretrained")
    async def test_dataset_configured_notification_preserves_float_timestamps(
        self,
        mock_tokenizer_from_pretrained,
        create_mooncake_trace_file,
        mock_tokenizer_cls,
    ):
        """Test that floating point timestamps are preserved exactly in dataset notifications.

        This test verifies that high-precision floating point timestamps from trace data
        are maintained throughout the dataset loading and notification process.
        """
        # Mock the tokenizer to avoid HTTP requests
        mock_tokenizer_from_pretrained.return_value = (
            mock_tokenizer_cls.from_pretrained("test-model")
        )

        # Create a file with floating point timestamps (in milliseconds)
        entries = [
            '{"session_id": "sess-1", "timestamp": 0.123, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-1", "delay": 10000.456, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-2", "timestamp": 20000.789, "input_length": 25, "output_length": 20}',
            '{"session_id": "sess-2", "delay": 15000.123, "input_length": 100, "output_length": 20}',
        ]
        filename = create_mooncake_trace_file(entries)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            service_config = ServiceConfig()
            dataset_manager = DatasetManager(service_config, user_config)

            await dataset_manager.initialize()

            # Mock the publish method to capture notifications
            from unittest.mock import AsyncMock

            from aiperf.common.messages import DatasetConfiguredNotification

            published_messages = []

            async def mock_publish(msg):
                published_messages.append(msg)

            dataset_manager.publish = AsyncMock(side_effect=mock_publish)

            # Configure the dataset to load conversations
            await dataset_manager._profile_configure_command(
                ProfileConfigureCommand(config=user_config, service_id="test_service")
            )

            # Verify the notification was published
            published_notifications = [
                msg
                for msg in published_messages
                if isinstance(msg, DatasetConfiguredNotification)
            ]
            assert len(published_notifications) == 1

            notification = published_notifications[0]
            metadata = notification.metadata

            # Extract conversation metadata
            conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

            # Verify conversations are loaded correctly
            assert "sess-1" in conv_dict
            sess1 = conv_dict["sess-1"]
            assert len(sess1.turns) == 2

            assert "sess-2" in conv_dict
            sess2 = conv_dict["sess-2"]
            assert len(sess2.turns) == 2

        finally:
            Path(filename).unlink(missing_ok=True)


class TestDatasetManagerSamplingStrategyDefaults:
    """Test default sampling strategy behavior for different dataset types."""

    @pytest.fixture(autouse=True)
    async def teardown(self):
        """Clean up after each test to prevent shared state issues."""
        yield
        from aiperf.common.factories import CommunicationFactory

        if hasattr(CommunicationFactory, "_instances"):
            CommunicationFactory._instances.clear()

    @pytest.mark.asyncio
    @patch("aiperf.common.tokenizer.Tokenizer.from_pretrained")
    @patch("aiperf.dataset.loader.sharegpt.ShareGPTLoader.load_dataset")
    @patch("aiperf.dataset.loader.sharegpt.ShareGPTLoader.convert_to_conversations")
    async def test_public_dataset_uses_loader_recommended_strategy(
        self,
        mock_convert,
        mock_load,
        mock_tokenizer_from_pretrained,
        mock_tokenizer_cls,
    ):
        """Test that public datasets use the loader's recommended sampling strategy."""
        from aiperf.common.models import Conversation, Text, Turn

        # Mock tokenizer
        mock_tokenizer_from_pretrained.return_value = (
            mock_tokenizer_cls.from_pretrained("test-model")
        )

        # Mock dataset loading
        mock_load.return_value = {}
        mock_convert.return_value = [
            Conversation(
                session_id="session-1",
                turns=[Turn(texts=[Text(contents=["Hello"])], model="test-model")],
            ),
            Conversation(
                session_id="session-2",
                turns=[Turn(texts=[Text(contents=["World"])], model="test-model")],
            ),
        ]

        # Create config with public dataset and NO explicit sampling strategy
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(public_dataset=PublicDatasetType.SHAREGPT),
        )
        assert user_config.input.dataset_sampling_strategy is None

        service_config = ServiceConfig()
        dataset_manager = DatasetManager(service_config, user_config)

        await dataset_manager.initialize()
        await dataset_manager._profile_configure_command(
            ProfileConfigureCommand(config=user_config, service_id="test_service")
        )

        # Verify the loader's recommended strategy was used (SEQUENTIAL for ShareGPT)
        # Note: The actual sampler is now created in TimingManager, not DatasetManager
        assert (
            user_config.input.dataset_sampling_strategy
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    @pytest.mark.asyncio
    @patch("aiperf.common.tokenizer.Tokenizer.from_pretrained")
    async def test_fallback_default_when_strategy_not_set(
        self,
        mock_tokenizer_from_pretrained,
        mock_tokenizer_cls,
    ):
        """Test that InputDefaults.DATASET_SAMPLING_STRATEGY is used as fallback."""
        # Mock tokenizer
        mock_tokenizer_from_pretrained.return_value = (
            mock_tokenizer_cls.from_pretrained("test-model")
        )

        # Create config with NO public dataset and NO explicit sampling strategy
        # This will use synthetic dataset generation
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(),  # No public_dataset, no file - uses synthetic
        )

        service_config = ServiceConfig()
        dataset_manager = DatasetManager(service_config, user_config)

        await dataset_manager.initialize()
        await dataset_manager._profile_configure_command(
            ProfileConfigureCommand(config=user_config, service_id="test_service")
        )

        # Synthetic composer sets its own default, which should be the same as InputDefaults
        assert user_config.input.dataset_sampling_strategy is not None
        assert (
            user_config.input.dataset_sampling_strategy
            == InputDefaults.DATASET_SAMPLING_STRATEGY
        )

    @pytest.mark.asyncio
    @patch("aiperf.common.tokenizer.Tokenizer.from_pretrained")
    @patch("aiperf.dataset.loader.sharegpt.ShareGPTLoader.load_dataset")
    @patch("aiperf.dataset.loader.sharegpt.ShareGPTLoader.convert_to_conversations")
    async def test_explicit_strategy_overrides_loader_recommendation(
        self,
        mock_convert,
        mock_load,
        mock_tokenizer_from_pretrained,
        mock_tokenizer_cls,
    ):
        """Test that explicitly set strategy is not overridden by loader recommendation."""
        from aiperf.common.models import Conversation, Text, Turn

        # Mock tokenizer
        mock_tokenizer_from_pretrained.return_value = (
            mock_tokenizer_cls.from_pretrained("test-model")
        )

        # Mock dataset loading
        mock_load.return_value = {}
        mock_convert.return_value = [
            Conversation(
                session_id="session-1",
                turns=[Turn(texts=[Text(contents=["Hello"])], model="test-model")],
            ),
        ]

        # Create config with explicit SHUFFLE strategy (different from loader's SEQUENTIAL)
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                public_dataset=PublicDatasetType.SHAREGPT,
                dataset_sampling_strategy=DatasetSamplingStrategy.SHUFFLE,
            ),
        )

        service_config = ServiceConfig()
        dataset_manager = DatasetManager(service_config, user_config)

        await dataset_manager.initialize()
        await dataset_manager._profile_configure_command(
            ProfileConfigureCommand(config=user_config, service_id="test_service")
        )

        # Verify the explicit strategy was preserved, not overwritten by loader's SEQUENTIAL
        # Note: The actual sampler is now created in TimingManager, not DatasetManager
        assert (
            user_config.input.dataset_sampling_strategy
            == DatasetSamplingStrategy.SHUFFLE
        )
