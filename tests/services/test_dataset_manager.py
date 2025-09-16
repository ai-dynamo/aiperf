# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aiperf.common.config import EndpointConfig, InputConfig, ServiceConfig, UserConfig
from aiperf.common.enums import CustomDatasetType
from aiperf.dataset.dataset_manager import DatasetManager


class TestDatasetManagerSequentialIteration:
    """Test sequential iteration behavior for custom datasets."""

    @pytest.fixture
    def create_mooncake_trace_file(self):
        """Create a temporary mooncake trace file with distinct inputs."""

        def _create_file(entries):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                for entry in entries:
                    f.write(f"{entry}\n")
                return f.name

        return _create_file

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
            await dataset_manager.initialize()  # Initialize the service

            # Configure the dataset to load conversations
            await dataset_manager._configure_dataset()

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
            await custom_manager.initialize()  # Initialize the service

            # Configure the dataset
            await custom_manager._configure_dataset()

            # Verify sequential iteration is enabled for custom datasets
            assert custom_manager._use_sequential_iteration is True

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
            await dataset_manager.initialize()  # Initialize the service

            # Configure the dataset
            await dataset_manager._configure_dataset()

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
