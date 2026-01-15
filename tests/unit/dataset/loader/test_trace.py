# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    UserConfig,
)
from aiperf.common.enums import CustomDatasetType
from aiperf.dataset import MooncakeTrace, MooncakeTraceDatasetLoader


class TestMooncakeTrace:
    """Basic functionality tests for MooncakeTrace model."""

    def test_create_with_input_length(self):
        """Test creating MooncakeTrace with input_length."""
        data = MooncakeTrace(input_length=100, hash_ids=[123, 456, 789], timestamp=1000)

        assert data.input_length == 100
        assert data.output_length is None  # Optional field
        assert data.text_input is None
        assert data.hash_ids == [123, 456, 789]
        assert data.timestamp == 1000
        assert data.type == CustomDatasetType.MOONCAKE_TRACE

    def test_create_with_text_input(self):
        """Test creating MooncakeTrace with text_input."""
        data = MooncakeTrace(text_input="This is test input text", timestamp=1000)

        assert data.text_input == "This is test input text"
        assert data.input_length is None
        assert data.output_length is None  # Optional field
        assert data.hash_ids is None  # Not allowed with text_input
        assert data.timestamp == 1000

    def test_create_with_both_input_fields_and_hash_ids(self):
        """Test that input_length and text_input cannot be provided together."""
        with pytest.raises(
            ValidationError,
            match="'input_length' and 'text_input' cannot be provided together",
        ):
            MooncakeTrace(
                input_length=100,
                text_input="This is test input text",
                hash_ids=[123],
                timestamp=1000,
            )

    def test_create_with_optional_output_length(self):
        """Test creating MooncakeTrace with optional output_length."""
        data = MooncakeTrace(
            input_length=100, output_length=50, hash_ids=[123], timestamp=1000
        )

        assert data.output_length == 50

    def test_validation_missing_input_fields_errors(self):
        """Test validation errors when neither input_length nor text_input provided."""
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError,
            match="Either 'input_length' or 'text_input' must be provided",
        ):
            MooncakeTrace(hash_ids=[123], timestamp=1000)

    def test_validation_missing_required_fields_errors(self):
        """Test validation errors for MooncakeTrace missing other required fields."""
        from pydantic import ValidationError

        # When hash_ids is provided but no input is provided, should fail with general input validation
        with pytest.raises(
            ValidationError,
            match="Either 'input_length' or 'text_input' must be provided",
        ):
            MooncakeTrace(hash_ids=[123], timestamp=1000)

        # text_input does not require hash_ids, so this should work
        data = MooncakeTrace(text_input="test input")
        assert data.text_input == "test input"
        assert data.hash_ids is None

    def test_validation_hash_ids_requires_input_length(self):
        """Test that hash_ids is only allowed with input_length, not text_input."""
        from pydantic import ValidationError

        # Validation prevents text_input + hash_ids combination
        with pytest.raises(
            ValidationError,
            match="'hash_ids' is only allowed when 'input_length' is provided, not when 'text_input' is provided",
        ):
            MooncakeTrace(text_input="test input", hash_ids=[123], timestamp=1000)


class TestMooncakeTraceDatasetLoader:
    """Basic functionality tests for MooncakeTraceDatasetLoader."""

    @pytest.fixture
    def mock_prompt_generator(self):
        """Create a mock prompt generator for testing."""
        generator = Mock()
        generator.generate.return_value = "Generated prompt text"
        # Required for convert_to_conversations() to check string cache
        generator._decoded_cache = {}
        # Mock _build_token_sequence to return a simple token list
        generator._build_token_sequence.return_value = [1, 2, 3, 4, 5]
        return generator

    @pytest.fixture
    def default_user_config(self):
        """Create a default UserConfig for testing."""
        return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))

    def make_user_config(
        self, start_offset: int | None = None, end_offset: int | None = None
    ):
        """Create a UserConfig for testing."""
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                fixed_schedule_start_offset=start_offset,
                fixed_schedule_end_offset=end_offset,
            ),
        )

    def test_load_dataset_basic_functionality(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Test basic JSONL file loading."""
        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123, 456], "timestamp": 1000}',
            '{"input_length": 200, "output_length": 75, "hash_ids": [789], "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        assert isinstance(dataset, dict)
        assert len(dataset) == 2  # Two different sessions (auto-generated UUIDs)

        # Check that each session has one trace
        for _, traces in dataset.items():
            assert len(traces) == 1
            assert isinstance(traces[0], MooncakeTrace)

        traces = list(dataset.values())
        assert traces[0][0].input_length == 100
        assert traces[0][0].output_length == 50
        assert traces[0][0].hash_ids == [123, 456]
        assert traces[0][0].timestamp == 1000

        assert traces[1][0].input_length == 200
        assert traces[1][0].output_length == 75
        assert traces[1][0].hash_ids == [789]
        assert traces[1][0].timestamp == 2000

    def test_load_dataset_with_text_input(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Test loading JSONL file with text_input fields."""
        content = [
            '{"text_input": "This is the first test input", "timestamp": 1000}',
            '{"text_input": "This is the second test input", "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 2
        traces = list(dataset.values())

        assert traces[0][0].text_input == "This is the first test input"
        assert traces[0][0].input_length is None
        assert traces[1][0].text_input == "This is the second test input"
        assert traces[1][0].input_length is None

    def test_load_dataset_mixed_input_types(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Test loading JSONL file with mixed input_length and text_input entries (but not both in same entry)."""
        content = [
            '{"input_length": 100, "hash_ids": [123], "timestamp": 1000}',
            '{"text_input": "Mixed input test", "timestamp": 2000}',
            '{"input_length": 200, "output_length": 50, "timestamp": 3000}',
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 3
        traces = list(dataset.values())

        # First entry: input_length with hash_ids
        assert traces[0][0].input_length == 100
        assert traces[0][0].text_input is None
        assert traces[0][0].hash_ids == [123]

        # Second entry: text_input only
        assert traces[1][0].input_length is None
        assert traces[1][0].text_input == "Mixed input test"

        # Third entry: input_length with output_length
        assert traces[2][0].input_length == 200
        assert traces[2][0].output_length == 50
        assert traces[2][0].text_input is None

    def test_load_dataset_skips_empty_lines(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Test that empty lines are skipped."""
        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123], "timestamp": 1000}',
            "",  # Empty line
            '{"input_length": 200, "output_length": 75, "hash_ids": [456], "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        result = loader.load_dataset()

        assert len(result) == 2  # Should skip empty line

    def test_load_dataset_with_timestamps(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Test loading dataset with timestamp fields."""
        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123], "timestamp": 1000}',
            '{"input_length": 200, "output_length": 75, "hash_ids": [456], "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        traces = list(dataset.values())
        assert traces[0][0].timestamp == 1000
        assert traces[1][0].timestamp == 2000

    @pytest.mark.parametrize(
        "start_offset,end_offset,expected_count,description",
        [
            (None, None, 4, "no filtering"),
            (1500, None, 3, "start offset only - keeps timestamps >= 1500"),
            (None, 2500, 3, "end offset only - keeps timestamps <= 2500"),
            (1500, 2500, 2, "both offsets - keeps timestamps in range [1500, 2500]"),
        ],
    )  # fmt: skip
    def test_load_dataset_with_offset_filtering(
        self,
        create_jsonl_file,
        mock_prompt_generator,
        start_offset,
        end_offset,
        expected_count,
        description,
    ):
        """Test dataset loading with start and end offset filtering."""
        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123], "timestamp": 1000}',  # Before start
            '{"input_length": 150, "output_length": 60, "hash_ids": [456], "timestamp": 2000}',  # In range
            '{"input_length": 200, "output_length": 70, "hash_ids": [789], "timestamp": 2500}',  # At end boundary
            '{"input_length": 250, "output_length": 80, "hash_ids": [111], "timestamp": 3000}',  # After end
        ]  # fmt: skip
        filename = create_jsonl_file(content)

        user_config = self.make_user_config(start_offset, end_offset)
        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        assert len(dataset) == expected_count, f"Failed for {description}"

    @pytest.mark.parametrize(
        "start_offset,end_offset,expected_skipped",
        [
            (2500, None, 2),  # Skip timestamps < 2500 (1000, 2000)
            (None, 1500, 2),  # Skip timestamps > 1500 (2000, 3000)
        ],
    )  # fmt: skip
    def test_load_dataset_logs_skipped_traces(
        self,
        create_jsonl_file,
        mock_prompt_generator,
        caplog,
        start_offset,
        end_offset,
        expected_skipped,
    ):
        """Test that skipped traces are properly logged."""
        caplog.set_level(logging.INFO)

        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123], "timestamp": 1000}',
            '{"input_length": 150, "output_length": 60, "hash_ids": [456], "timestamp": 2000}',
            '{"input_length": 200, "output_length": 70, "hash_ids": [789], "timestamp": 3000}',
        ]
        filename = create_jsonl_file(content)

        user_config = self.make_user_config(start_offset, end_offset)
        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        loader.load_dataset()

        # Check that the skipped traces message is logged
        assert f"Skipped {expected_skipped:,} traces" in caplog.text

    @patch("aiperf.dataset.loader.mooncake_trace.parallel_decode")
    def test_convert_to_conversations(
        self, mock_parallel_decode, mock_prompt_generator, default_user_config
    ):
        """Test conversion of trace data to conversations."""
        # Mock parallel_decode to return decoded prompts
        mock_parallel_decode.return_value = [
            "decoded prompt 1",
            "decoded prompt 2",
            "decoded prompt 3",
        ]

        # Setup trace data
        trace_data = {
            "session-1": [
                MooncakeTrace(
                    input_length=100,
                    output_length=50,
                    hash_ids=[123, 456],
                    timestamp=1000,
                ),
            ],
            "session-2": [
                MooncakeTrace(
                    input_length=200,
                    output_length=100,
                    hash_ids=[111, 222, 333],
                    timestamp=2000,
                )
            ],
            "session-3": [
                MooncakeTrace(
                    input_length=150,
                    output_length=75,
                    hash_ids=[789],
                    timestamp=3000,
                )
            ],
        }

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        conversations = loader.convert_to_conversations(trace_data)

        assert len(conversations) == 3

        # Check first conversation
        conv1 = conversations[0]
        assert conv1.session_id == "session-1"
        assert len(conv1.turns) == 1
        assert conv1.turns[0].timestamp == 1000

        # Check second conversation
        conv2 = conversations[1]
        assert conv2.session_id == "session-2"
        assert len(conv2.turns) == 1
        assert conv2.turns[0].timestamp == 2000

        # Check third conversation
        conv3 = conversations[2]
        assert conv3.session_id == "session-3"
        assert len(conv3.turns) == 1
        assert conv3.turns[0].timestamp == 3000

    def test_convert_to_conversations_empty_data(
        self, mock_prompt_generator, default_user_config
    ):
        """Test conversion with empty trace data."""
        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        conversations = loader.convert_to_conversations({})

        assert len(conversations) == 0

    def test_convert_to_conversations_with_text_input(
        self, mock_prompt_generator, default_user_config
    ):
        """Test conversion uses text_input when provided - covers 'if trace.text_input is not None' line."""
        # Create traces with text_input to cover the uncovered line
        trace_data = {
            "session1": [
                MooncakeTrace(text_input="Hello, how are you?", timestamp=1000),
                MooncakeTrace(text_input="What is the weather like?", timestamp=2000),
            ]
        }

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        conversations = loader.convert_to_conversations(trace_data)

        assert len(conversations) == 1  # One conversation with multiple turns
        conversation = conversations[0]

        assert len(conversation.turns) == 2
        assert conversation.turns[0].texts[0].contents[0] == "Hello, how are you?"
        assert conversation.turns[1].texts[0].contents[0] == "What is the weather like?"

    def test_load_dataset_with_session_ids(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Test loading JSONL file with session_id fields."""
        content = [
            '{"session_id": "session-1", "input_length": 100, "output_length": 50, "hash_ids": [123], "timestamp": 1000}',
            '{"session_id": "session-1", "input_length": 150, "output_length": 60, "hash_ids": [456], "timestamp": 2000}',
            '{"session_id": "session-2", "text_input": "This is session 2 input", "timestamp": 3000}',
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 2

        assert len(dataset["session-1"]) == 2
        assert dataset["session-1"][0].input_length == 100
        assert dataset["session-1"][1].input_length == 150

        assert len(dataset["session-2"]) == 1
        assert dataset["session-2"][0].text_input == "This is session 2 input"

    def test_load_dataset_with_delay_field(
        self, create_jsonl_file, mock_prompt_generator, default_user_config
    ):
        """Test loading JSONL file with delay fields."""
        content = [
            '{"session_id": "abc", "input_length": 100, "output_length": 50, "delay": 500}',
            '{"session_id": "def", "text_input": "This is test input", "delay": 1000}',
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 2
        traces = list(dataset.values())

        assert traces[0][0].delay == 500
        assert traces[1][0].delay == 1000


class TestMooncakeTraceReproducibility:
    """Tests for reproducibility of Mooncake trace prompt generation.

    These tests verify that the two-phase Mooncake flow with parallel_decode
    yields identical prompts across runs when the RNG is seeded consistently.
    """

    @pytest.fixture
    def mock_prompt_generator(self):
        """Create a mock prompt generator for testing."""
        generator = Mock()
        generator.generate.return_value = "Generated prompt text"
        generator._decoded_cache = {}
        generator._build_token_sequence.return_value = [1, 2, 3, 4, 5]
        return generator

    @pytest.fixture
    def user_config_for_reproducibility(self):
        """Create a UserConfig suitable for reproducibility testing."""
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                prompt=PromptConfig(
                    input_tokens=InputTokensConfig(mean=100, stddev=0, block_size=64),
                ),
            ),
        )

    @patch("aiperf.dataset.loader.mooncake_trace.parallel_decode")
    def test_mooncake_flow_reproducibility_with_same_seed(
        self, mock_parallel_decode, mock_tokenizer_cls, user_config_for_reproducibility
    ):
        """Verify Mooncake flow produces identical prompts across runs with same seed.

        This guards the reproducibility contract: seeding RNG, running conversion twice,
        and asserting identical prompts.
        """
        from aiperf.common import random_generator as rng
        from aiperf.dataset.generator import PromptGenerator

        # Mock parallel_decode to return deterministic results based on input
        def deterministic_decode(token_sequences, tokenizer_name):
            return [
                f"decoded_prompt_{i}_{len(seq)}"
                for i, seq in enumerate(token_sequences)
            ]

        mock_parallel_decode.side_effect = deterministic_decode

        # Create trace data with hash_ids to exercise the two-phase flow
        trace_data = {
            "session-1": [
                MooncakeTrace(
                    input_length=128, output_length=50, hash_ids=[1, 2], timestamp=1000
                ),
                MooncakeTrace(
                    input_length=192,
                    output_length=75,
                    hash_ids=[3, 4, 5],
                    timestamp=2000,
                ),
            ],
            "session-2": [
                MooncakeTrace(
                    input_length=256,
                    output_length=100,
                    hash_ids=[6, 7, 8, 9],
                    timestamp=3000,
                ),
            ],
        }

        # First run: seed, create generator, convert
        rng.reset()
        rng.init(42)

        tokenizer1 = mock_tokenizer_cls.from_pretrained("test-model")
        prompt_config1 = user_config_for_reproducibility.input.prompt
        generator1 = PromptGenerator(prompt_config1, tokenizer1)

        loader1 = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config_for_reproducibility,
            prompt_generator=generator1,
        )
        conversations1 = loader1.convert_to_conversations(trace_data)
        prompts1 = [
            turn.texts[0].contents[0] for conv in conversations1 for turn in conv.turns
        ]

        # Second run: re-seed with same value, create fresh generator, convert
        rng.reset()
        rng.init(42)

        tokenizer2 = mock_tokenizer_cls.from_pretrained("test-model")
        prompt_config2 = user_config_for_reproducibility.input.prompt
        generator2 = PromptGenerator(prompt_config2, tokenizer2)

        loader2 = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config_for_reproducibility,
            prompt_generator=generator2,
        )
        conversations2 = loader2.convert_to_conversations(trace_data)
        prompts2 = [
            turn.texts[0].contents[0] for conv in conversations2 for turn in conv.turns
        ]

        # Assert identical prompts
        assert len(prompts1) == len(prompts2), "Prompt count mismatch"
        assert prompts1 == prompts2, (
            "Prompts should be identical across runs with the same seed. "
            f"First run: {prompts1}, Second run: {prompts2}"
        )

    @patch("aiperf.dataset.loader.mooncake_trace.parallel_decode")
    def test_parallel_decode_length_mismatch_raises(
        self, mock_parallel_decode, mock_prompt_generator, default_user_config
    ):
        """Verify that length mismatch between pending_decodes and decoded_prompts raises.

        This tests the strict=True behavior in zip() that guards against silent data loss.
        """
        # Mock parallel_decode to return FEWER results than expected
        mock_parallel_decode.return_value = ["decoded prompt 1"]  # Only 1, expecting 3

        trace_data = {
            "session-1": [
                MooncakeTrace(input_length=100, hash_ids=[1, 2], timestamp=1000),
            ],
            "session-2": [
                MooncakeTrace(input_length=200, hash_ids=[3, 4, 5], timestamp=2000),
            ],
            "session-3": [
                MooncakeTrace(input_length=150, hash_ids=[6], timestamp=3000),
            ],
        }

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )

        # Should raise ValueError due to strict=True in zip
        with pytest.raises(ValueError, match="zip"):
            loader.convert_to_conversations(trace_data)
