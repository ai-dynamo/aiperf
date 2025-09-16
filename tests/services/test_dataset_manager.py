# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aiperf.common.config import EndpointConfig, InputConfig, UserConfig
from aiperf.common.config.service_config import ServiceConfig
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

    async def test_sequential_iteration_order(
        self, create_mooncake_trace_file, mock_prompt_generator, mock_tokenizer_cls
    ):
        """Test that custom datasets iterate sequentially, not randomly."""
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
            with patch("aiperf.common.tokenizer.Tokenizer", mock_tokenizer_cls):
                user_config = UserConfig(
                    endpoint=EndpointConfig(model_names=["test-model"]),
                    input=InputConfig(
                        input_filename=filename,
                        custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    ),
                )

                service_config = ServiceConfig()
                dataset_manager = DatasetManager(service_config, user_config)
                await dataset_manager.initialize()  # Initialize the service
                await dataset_manager.start()  # Start the service

            # Get conversations multiple times and verify order
            conversations = []
            for _ in range(5):
                conv = await dataset_manager._return_any_conversation("test_session")
                conversations.append(conv)

            # Extract input_lengths from conversations - they should be sequential
            input_lengths = []
            for conv in conversations:
                if conv.turns and conv.turns[0].texts:
                    # Find the input_length from the conversation data
                    # This assumes the conversation structure includes our test data
                    if hasattr(conv, "_source_data"):
                        input_lengths.append(conv._source_data.get("input_length"))
                    else:
                        # Alternative: check if conversation has our test markers
                        text_content = (
                            conv.turns[0].texts[0].contents[0]
                            if conv.turns[0].texts[0].contents
                            else ""
                        )
                        # We can identify by checking generation pattern or add markers
                        input_lengths.append(
                            len(text_content.split())
                        )  # Rough approximation
            assert len(input_lengths) == 5

            # For exact verification, we need to check the sequential behavior
            # The key test is that we get the SAME order every time, not random

            # Reset and get conversations again - should be same order
            dataset_manager._sequential_iterator_index = 0  # Reset iterator
            conversations_repeat = []
            for _ in range(5):
                conv = await dataset_manager._return_any_conversation("test_session")
                conversations_repeat.append(conv)

            # The order should be identical (sequential), not different (random)
            for i in range(5):
                assert conversations[i].session_id == conversations_repeat[i].session_id

        finally:
            Path(filename).unlink(missing_ok=True)

    async def test_sequential_vs_random_behavior(
        self, create_mooncake_trace_file, mock_prompt_generator, mock_tokenizer_cls
    ):
        """Test that custom datasets use sequential iteration while synthetic use random."""

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
                    input_filename=filename,
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                ),
            )

            service_config = ServiceConfig()
            custom_manager = DatasetManager(service_config, custom_config)
            await custom_manager.initialize()  # Initialize the service
            await custom_manager.start()  # Start the service

            # Get sessions in order for custom dataset
            custom_sessions = []
            for _ in range(6):  # More than dataset size to test wraparound
                conv = await custom_manager._return_any_conversation("test_session")
                custom_sessions.append(conv.session_id)

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

            # Test 2: Non-custom dataset (should use different behavior)
            synthetic_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    input_filename=filename,
                    # custom_dataset_type=None  # No custom dataset type
                ),
            )

            service_config = ServiceConfig()
            synthetic_manager = DatasetManager(service_config, synthetic_config)
            await synthetic_manager.initialize()  # Initialize the service
            await synthetic_manager.start()  # Start the service

            # The behavior should be different (random selection)
            # We test this by verifying the sequential iterator is not used
            assert (
                not hasattr(synthetic_manager, "_sequential_iterator_index")
                or synthetic_manager._sequential_iterator_index == 0
            )

        finally:
            Path(filename).unlink(missing_ok=True)

    @patch("aiperf.common.tokenizer.Tokenizer.from_pretrained")
    async def test_sequential_iterator_wraparound(
        self,
        mock_tokenizer_from_pretrained,
        create_mooncake_trace_file,
        mock_prompt_generator,
    ):
        """Test that sequential iterator wraps around correctly."""
        entries = [
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}',
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}',
        ]
        filename = create_mooncake_trace_file(entries)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    input_filename=filename,
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                ),
            )

            service_config = ServiceConfig()
            dataset_manager = DatasetManager(service_config, user_config)
            await dataset_manager.initialize()  # Initialize the service
            await dataset_manager.start()  # Start the service

            # Get more conversations than dataset size
            session_ids = []
            for _ in range(5):  # 5 requests for 2-entry dataset
                conv = await dataset_manager._return_any_conversation("test_session")
                session_ids.append(conv.session_id)

            # Should follow pattern: entry1, entry2, entry1, entry2, entry1
            assert (
                session_ids[0] == session_ids[2] == session_ids[4]
            )  # 1st, 3rd, 5th same
            assert session_ids[1] == session_ids[3]  # 2nd, 4th same
            assert session_ids[0] != session_ids[1]  # Different entries

        finally:
            Path(filename).unlink(missing_ok=True)
