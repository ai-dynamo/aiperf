# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aiperf.common.config import EndpointConfig, InputConfig, ServiceConfig, UserConfig
from aiperf.common.enums import CustomDatasetType
from aiperf.common.messages.command_messages import ProfileConfigureCommand
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.dataset_samplers import SequentialSampler


class TestDatasetManagerSequentialIteration:
    """Test sequential iteration behavior for custom datasets."""

    @pytest.fixture
    def mock_prompt_generator(self):
        """Mock prompt generator."""
        generator = Mock()
        generator.generate.return_value = "Generated prompt"
        return generator

    @pytest.fixture(autouse=True)
    async def teardown(self):
        """Clean up after each test to prevent shared state issues."""
        yield
        # Reset any global state if needed
        # Clear communication factory state
        from aiperf.common.factories import CommunicationFactory

        if hasattr(CommunicationFactory, "_instances"):
            CommunicationFactory._instances.clear()

    @patch("aiperf.common.tokenizer.Tokenizer.from_pretrained")
    async def test_sequential_iteration_order(
        self,
        mock_tokenizer_from_pretrained,
        create_mooncake_trace_file,
        mock_tokenizer_cls,
    ):
        """Test that custom datasets iterate sequentially, not randomly."""
        # Mock the tokenizer to avoid HTTP requests
        mock_tokenizer_from_pretrained.return_value = (
            mock_tokenizer_cls.from_pretrained("test-model")
        )

        # Create a file with distinct input_lengths for easy verification
        entries = [
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}',
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}',
            '{"input_length": 300, "hash_ids": [3], "timestamp": 3000}',
            '{"input_length": 400, "hash_ids": [4], "timestamp": 4000}',
            '{"input_length": 500, "hash_ids": [5], "timestamp": 5000}',
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

            # Configure the dataset to load conversations
            await dataset_manager._profile_configure_command(
                ProfileConfigureCommand(config=user_config, service_id="test_service")
            )

            # Get conversations multiple times and verify order
            conversations = []
            for _ in range(5):
                conv = dataset_manager._return_any_conversation("test_session")
                conversations.append(conv)

            # Verify we got 5 conversations
            assert len(conversations) == 5

            # The key test: sequential iteration should mean we get the same order
            # when we reset and iterate again
            dataset_manager._sequential_iterator_index = 0  # Reset iterator
            conversations_repeat = []
            for _ in range(5):
                conv = dataset_manager._return_any_conversation("test_session")
                conversations_repeat.append(conv)

            # Verify that the order is identical (sequential), not different (random)
            for i in range(5):
                assert (
                    conversations[i].conversation.session_id
                    == conversations_repeat[i].conversation.session_id
                )

        finally:
            Path(filename).unlink(missing_ok=True)

    @patch("aiperf.common.tokenizer.Tokenizer.from_pretrained")
    async def test_sequential_vs_random_behavior(
        self,
        mock_tokenizer_from_pretrained,
        create_mooncake_trace_file,
        mock_prompt_generator,
        mock_tokenizer_cls,
    ):
        """Test that custom datasets use sequential iteration while synthetic use random."""
        # Mock the tokenizer to avoid HTTP requests
        mock_tokenizer_from_pretrained.return_value = (
            mock_tokenizer_cls.from_pretrained("test-model")
        )

        entries = [
            '{"input_length": 111, "hash_ids": [1], "timestamp": 1000}',
            '{"input_length": 222, "hash_ids": [2], "timestamp": 2000}',
            '{"input_length": 333, "hash_ids": [3], "timestamp": 3000}',
        ]
        filename = create_mooncake_trace_file(entries)

        try:
            # Test 1: Custom dataset (should be sequential)
            custom_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename,
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                ),
            )

            service_config = ServiceConfig()
            custom_manager = DatasetManager(service_config, custom_config)

            await custom_manager.initialize()

            # Configure the dataset
            await custom_manager._profile_configure_command(
                ProfileConfigureCommand(config=custom_config, service_id="test_service")
            )

            assert custom_manager._dataset_sampler is not None
            assert isinstance(custom_manager._dataset_sampler, SequentialSampler)

            # Get sessions in order for custom dataset
            custom_sessions = []
            for _ in range(6):  # More than dataset size to test wraparound
                conv = custom_manager._return_any_conversation("test_session")
                custom_sessions.append(conv.conversation.session_id)

            # Should repeat pattern: session1, session2, session3, session1, session2, session3
            assert (
                custom_sessions[0] == custom_sessions[3]
            )  # First repeats at position 3
            assert (
                custom_sessions[1] == custom_sessions[4]
            )  # Second repeats at position 4
            assert (
                custom_sessions[2] == custom_sessions[5]
            )  # Third repeats at position 5

        finally:
            Path(filename).unlink(missing_ok=True)

    @patch("aiperf.common.tokenizer.Tokenizer.from_pretrained")
    async def test_sequential_iterator_wraparound(
        self,
        mock_tokenizer_from_pretrained,
        create_mooncake_trace_file,
        mock_prompt_generator,
        mock_tokenizer_cls,
    ):
        """Test that sequential iterator wraps around correctly."""
        # Mock the tokenizer to avoid HTTP requests
        mock_tokenizer_from_pretrained.return_value = (
            mock_tokenizer_cls.from_pretrained("test-model")
        )

        entries = [
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}',
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}',
        ]
        filename = create_mooncake_trace_file(entries)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename,
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                ),
            )

            service_config = ServiceConfig()
            dataset_manager = DatasetManager(service_config, user_config)

            await dataset_manager.initialize()

            # Configure the dataset
            await dataset_manager._profile_configure_command(
                ProfileConfigureCommand(config=user_config, service_id="test_service")
            )

            # Get more conversations than dataset size
            session_ids = []
            for _ in range(5):  # 5 requests for 2-entry dataset
                conv = dataset_manager._return_any_conversation("test_session")
                session_ids.append(conv.conversation.session_id)

            # Should follow pattern: entry1, entry2, entry1, entry2, entry1
            assert (
                session_ids[0] == session_ids[2] == session_ids[4]
            )  # 1st, 3rd, 5th same
            assert session_ids[1] == session_ids[3]  # 2nd, 4th same
            assert session_ids[0] != session_ids[1]  # Different entries

        finally:
            Path(filename).unlink(missing_ok=True)

    @pytest.mark.asyncio
    @patch("aiperf.common.tokenizer.Tokenizer.from_pretrained")
    async def test_dataset_timing_request_for_multi_turn_conversations(
        self,
        mock_tokenizer_from_pretrained,
        create_mooncake_trace_file,
        mock_tokenizer_cls,
    ):
        """Test that dataset timing request returns first turn timestamp for each conversation.

        When a dataset has multiple turns per conversation, the timing dataset should:
        - Return one entry per conversation (not one per turn)
        - Use the first turn's timestamp for scheduling
        - All turns within a conversation are sent sequentially after the conversation is scheduled
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

            # Configure the dataset to load conversations
            await dataset_manager._profile_configure_command(
                ProfileConfigureCommand(config=user_config, service_id="test_service")
            )

            # Request timing data
            from aiperf.common.messages import DatasetTimingRequest

            timing_response = await dataset_manager._handle_dataset_timing_request(
                DatasetTimingRequest(service_id="test_service")
            )

            # Verify timing dataset structure
            assert len(timing_response.timing_data) == 2  # 2 conversations, not 5 turns

            # Extract timing data for easier testing
            timing_dict = {
                conv_id: timestamp for timestamp, conv_id in timing_response.timing_data
            }

            # Verify session 1 is scheduled at its first turn's timestamp (0)
            assert "sess-1" in timing_dict
            assert timing_dict["sess-1"] == 0

            # Verify session 2 is scheduled at its first turn's timestamp (20000)
            assert "sess-2" in timing_dict
            assert timing_dict["sess-2"] == 20000

            # Verify no duplicate session IDs (one per conversation, not per turn)
            session_ids = [conv_id for _, conv_id in timing_response.timing_data]
            assert len(session_ids) == len(set(session_ids))

            # Test with conversations containing empty turns (should be skipped)
            # Manually add a conversation with no turns to dataset
            from aiperf.common.models import Conversation

            empty_conversation = Conversation(
                session_id="empty-session",
                turns=[],  # Empty turns list
            )
            dataset_manager.dataset["empty-session"] = empty_conversation

            # Request timing data again
            timing_response_with_empty = (
                await dataset_manager._handle_dataset_timing_request(
                    DatasetTimingRequest(service_id="test_service")
                )
            )

            # Verify empty conversation is skipped - should still have 2 entries, not 3
            assert len(timing_response_with_empty.timing_data) == 2
            timing_dict_with_empty = {
                conv_id: timestamp
                for timestamp, conv_id in timing_response_with_empty.timing_data
            }
            # Empty session should not be in timing data
            assert "empty-session" not in timing_dict_with_empty
            assert "sess-1" in timing_dict_with_empty
            assert "sess-2" in timing_dict_with_empty

        finally:
            Path(filename).unlink(missing_ok=True)
