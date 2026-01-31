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
    SynthesisConfig,
    UserConfig,
)
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from aiperf.plugin.enums import CustomDatasetType


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
        self,
        start_offset: int | None = None,
        end_offset: int | None = None,
        file: str | None = None,
    ):
        """Create a UserConfig for testing."""
        # Only set fixed_schedule=True when offsets are provided (requires a file)
        has_offsets = start_offset is not None or end_offset is not None
        input_config = (
            InputConfig(
                file=file,
                fixed_schedule=True,
                fixed_schedule_start_offset=start_offset,
                fixed_schedule_end_offset=end_offset,
            )
            if has_offsets
            else InputConfig()
        )
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=input_config,
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

        user_config = self.make_user_config(start_offset, end_offset, file=filename)
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

        user_config = self.make_user_config(start_offset, end_offset, file=filename)
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

    @pytest.mark.parametrize(
        "max_isl,expected_count,description",
        [
            (None, 4, "no filtering when max_isl is None"),
            (500, 4, "all traces pass when max_isl is high enough"),
            (250, 4, "input_length=250 passes when max_isl=250"),
            (249, 3, "filters traces with input_length > 249"),
            (150, 2, "filters traces with input_length > 150"),
            (50, 0, "filters all traces when max_isl is very low"),
        ],
    )  # fmt: skip
    def test_load_dataset_with_max_isl_filtering(
        self,
        create_jsonl_file,
        mock_prompt_generator,
        max_isl,
        expected_count,
        description,
    ):
        """Test dataset loading with max_isl filtering."""
        content = [
            '{"input_length": 100, "output_length": 50, "timestamp": 1000}',
            '{"input_length": 150, "output_length": 60, "timestamp": 2000}',
            '{"input_length": 200, "output_length": 70, "timestamp": 3000}',
            '{"input_length": 250, "output_length": 80, "timestamp": 4000}',
        ]
        filename = create_jsonl_file(content)

        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(synthesis=SynthesisConfig(max_isl=max_isl)),
        )
        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        assert len(dataset) == expected_count, f"Failed for {description}"

    def test_load_dataset_max_isl_does_not_filter_text_input(
        self, create_jsonl_file, mock_prompt_generator
    ):
        """Test that max_isl does not filter traces with text_input (no input_length)."""
        content = [
            '{"text_input": "Hello world", "timestamp": 1000}',
            '{"text_input": "This is a longer text input", "timestamp": 2000}',
            '{"input_length": 500, "output_length": 50, "timestamp": 3000}',
        ]
        filename = create_jsonl_file(content)

        # max_isl=100 should only filter the input_length=500 trace
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(synthesis=SynthesisConfig(max_isl=100)),
        )
        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        # Should have 2 traces (text_input ones pass, input_length=500 filtered)
        assert len(dataset) == 2
        traces = list(dataset.values())
        assert traces[0][0].text_input == "Hello world"
        assert traces[1][0].text_input == "This is a longer text input"

    def test_load_dataset_max_isl_logs_skipped_traces(
        self, create_jsonl_file, mock_prompt_generator, caplog
    ):
        """Test that skipped traces due to max_isl are properly logged."""
        caplog.set_level(logging.INFO)

        content = [
            '{"input_length": 100, "output_length": 50, "timestamp": 1000}',
            '{"input_length": 200, "output_length": 60, "timestamp": 2000}',
            '{"input_length": 300, "output_length": 70, "timestamp": 3000}',
        ]
        filename = create_jsonl_file(content)

        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(synthesis=SynthesisConfig(max_isl=150)),
        )
        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        loader.load_dataset()

        # Should skip 2 traces (input_length=200 and 300 exceed max_isl=150)
        assert (
            "Skipped 2 traces because input_length exceeded max_isl of 150"
            in caplog.text
        )

    def test_load_dataset_max_isl_combined_with_offset_filtering(
        self, create_jsonl_file, mock_prompt_generator
    ):
        """Test that max_isl and offset filtering work together."""
        content = [
            '{"input_length": 100, "output_length": 50, "timestamp": 500}',   # Before start offset
            '{"input_length": 100, "output_length": 50, "timestamp": 1500}',  # In range, passes max_isl
            '{"input_length": 300, "output_length": 60, "timestamp": 2000}',  # In range, exceeds max_isl
            '{"input_length": 100, "output_length": 70, "timestamp": 3500}',  # After end offset
        ]  # fmt: skip
        filename = create_jsonl_file(content)

        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=filename,
                fixed_schedule=True,
                fixed_schedule_start_offset=1000,
                fixed_schedule_end_offset=3000,
                synthesis=SynthesisConfig(max_isl=200),
            ),
        )
        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        # Only one trace should pass: timestamp=1500, input_length=100
        assert len(dataset) == 1
        traces = list(dataset.values())
        assert traces[0][0].input_length == 100
        assert traces[0][0].timestamp == 1500

    @pytest.mark.parametrize(
        "max_osl,expected_output_lengths,description",
        [
            (None, [50, 100, 150, 200], "no capping when max_osl is None"),
            (500, [50, 100, 150, 200], "no capping when max_osl is high enough"),
            (200, [50, 100, 150, 200], "output_length=200 not capped when max_osl=200"),
            (150, [50, 100, 150, 150], "caps output_length > 150 to 150"),
            (75, [50, 75, 75, 75], "caps output_length > 75 to 75"),
            (25, [25, 25, 25, 25], "caps all output_lengths to 25"),
        ],
    )  # fmt: skip
    def test_load_dataset_with_max_osl_capping(
        self,
        create_jsonl_file,
        mock_prompt_generator,
        max_osl,
        expected_output_lengths,
        description,
    ):
        """Test dataset loading with max_osl capping (not filtering)."""
        content = [
            '{"input_length": 100, "output_length": 50, "timestamp": 1000}',
            '{"input_length": 100, "output_length": 100, "timestamp": 2000}',
            '{"input_length": 100, "output_length": 150, "timestamp": 3000}',
            '{"input_length": 100, "output_length": 200, "timestamp": 4000}',
        ]
        filename = create_jsonl_file(content)

        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(synthesis=SynthesisConfig(max_osl=max_osl)),
        )
        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        # All traces should be kept (capping, not filtering)
        assert len(dataset) == 4, f"Failed for {description}"

        # Check output_lengths are capped correctly
        traces = list(dataset.values())
        actual_output_lengths = [t[0].output_length for t in traces]
        assert actual_output_lengths == expected_output_lengths, (
            f"Failed for {description}"
        )

    def test_load_dataset_max_osl_does_not_cap_none_output_length(
        self, create_jsonl_file, mock_prompt_generator
    ):
        """Test that max_osl does not affect traces without output_length."""
        content = [
            '{"input_length": 100, "timestamp": 1000}',
            '{"input_length": 100, "output_length": 200, "timestamp": 2000}',
            '{"text_input": "Hello world", "timestamp": 3000}',
        ]
        filename = create_jsonl_file(content)

        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(synthesis=SynthesisConfig(max_osl=50)),
        )
        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        # All 3 traces should be kept
        assert len(dataset) == 3
        traces = list(dataset.values())

        # First trace: no output_length, should remain None
        assert traces[0][0].output_length is None

        # Second trace: output_length=200 should be capped to 50
        assert traces[1][0].output_length == 50

        # Third trace: text_input, no output_length, should remain None
        assert traces[2][0].output_length is None

    def test_load_dataset_max_osl_logs_capped_traces(
        self, create_jsonl_file, mock_prompt_generator, caplog
    ):
        """Test that capped traces due to max_osl are properly logged."""
        caplog.set_level(logging.INFO)

        content = [
            '{"input_length": 100, "output_length": 50, "timestamp": 1000}',
            '{"input_length": 100, "output_length": 100, "timestamp": 2000}',
            '{"input_length": 100, "output_length": 150, "timestamp": 3000}',
        ]
        filename = create_jsonl_file(content)

        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(synthesis=SynthesisConfig(max_osl=75)),
        )
        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        loader.load_dataset()

        # Should cap 2 traces (output_length=100 and 150 exceed max_osl=75)
        assert "2 traces exceeded max_osl of 75 and were capped to 75" in caplog.text

    def test_load_dataset_max_isl_and_max_osl_combined(
        self, create_jsonl_file, mock_prompt_generator
    ):
        """Test that max_isl filtering and max_osl capping work together."""
        content = [
            '{"input_length": 100, "output_length": 200, "timestamp": 1000}',  # Passes max_isl, capped by max_osl
            '{"input_length": 300, "output_length": 50, "timestamp": 2000}',   # Filtered by max_isl
            '{"input_length": 150, "output_length": 150, "timestamp": 3000}',  # Passes max_isl, capped by max_osl
            '{"input_length": 50, "output_length": 50, "timestamp": 4000}',    # Passes both, no capping
        ]  # fmt: skip
        filename = create_jsonl_file(content)

        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(synthesis=SynthesisConfig(max_isl=200, max_osl=100)),
        )
        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        # 3 traces should remain (one filtered by max_isl)
        assert len(dataset) == 3

        traces = list(dataset.values())
        # First: input_length=100 passes, output_length=200 capped to 100
        assert traces[0][0].input_length == 100
        assert traces[0][0].output_length == 100

        # Second: input_length=150 passes, output_length=150 capped to 100
        assert traces[1][0].input_length == 150
        assert traces[1][0].output_length == 100

        # Third: input_length=50 passes, output_length=50 not capped
        assert traces[2][0].input_length == 50
        assert traces[2][0].output_length == 50


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


# ============================================================================
# Synthesis Integration Tests
# ============================================================================


def make_synthesis_config(
    speedup_ratio: float = 1.0,
    prefix_len_multiplier: float = 1.0,
    prefix_root_multiplier: int = 1,
    prompt_len_multiplier: float = 1.0,
    max_isl: int | None = None,
    block_size: int = 512,
) -> UserConfig:
    """Helper to create UserConfig with synthesis settings."""
    return UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        input=InputConfig(
            synthesis=SynthesisConfig(
                speedup_ratio=speedup_ratio,
                prefix_len_multiplier=prefix_len_multiplier,
                prefix_root_multiplier=prefix_root_multiplier,
                prompt_len_multiplier=prompt_len_multiplier,
                max_isl=max_isl,
            ),
            prompt=PromptConfig(
                input_tokens=InputTokensConfig(block_size=block_size),
            ),
        ),
    )


class TestMooncakeTraceSynthesisIntegration:
    """Tests for _apply_synthesis integration in MooncakeTraceDatasetLoader."""

    @pytest.fixture
    def mock_prompt_generator(self):
        """Create a mock prompt generator for testing."""
        generator = Mock()
        generator.generate.return_value = "Generated prompt text"
        generator._decoded_cache = {}
        generator._build_token_sequence.return_value = [1, 2, 3, 4, 5]
        return generator

    @pytest.fixture
    def sample_trace_data(self) -> dict[str, list[MooncakeTrace]]:
        """Sample trace data grouped by session."""
        return {
            "session-1": [
                MooncakeTrace(input_length=512, output_length=64, hash_ids=[1, 2]),
                MooncakeTrace(input_length=768, output_length=128, hash_ids=[1, 2, 3]),
            ],
            "session-2": [
                MooncakeTrace(input_length=1024, output_length=256, hash_ids=[4, 5]),
            ],
        }

    # ============================================================================
    # Basic Functionality
    # ============================================================================

    def test_synthesis_not_applied_when_disabled(
        self, mock_prompt_generator, sample_trace_data
    ):
        """Test that synthesis is skipped when should_synthesize() returns False."""
        user_config = make_synthesis_config()  # All defaults, should not synthesize

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )

        # Directly call _apply_synthesis should still work
        result = loader._apply_synthesis(sample_trace_data)

        # Should have same structure
        assert set(result.keys()) == {"session-1", "session-2"}
        assert len(result["session-1"]) == 2
        assert len(result["session-2"]) == 1

    def test_synthesis_preserves_session_grouping(
        self, mock_prompt_generator, sample_trace_data
    ):
        """Test that session grouping is preserved through synthesis."""
        user_config = make_synthesis_config(prefix_len_multiplier=2.0)

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )

        result = loader._apply_synthesis(sample_trace_data)

        assert set(result.keys()) == {"session-1", "session-2"}
        assert len(result["session-1"]) == 2
        assert len(result["session-2"]) == 1

    def test_synthesis_returns_mooncake_trace_objects(
        self, mock_prompt_generator, sample_trace_data
    ):
        """Test that synthesis returns MooncakeTrace objects, not dicts."""
        user_config = make_synthesis_config(speedup_ratio=2.0)

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )

        result = loader._apply_synthesis(sample_trace_data)

        for traces in result.values():
            for trace in traces:
                assert isinstance(trace, MooncakeTrace)

    # ============================================================================
    # Synthesis Parameters
    # ============================================================================

    def test_speedup_ratio_applied(self, mock_prompt_generator):
        """Test that speedup_ratio scales timestamps."""
        data = {
            "session-1": [
                MooncakeTrace(input_length=512, output_length=64, timestamp=1000),
                MooncakeTrace(input_length=512, output_length=64, timestamp=2000),
            ],
        }
        user_config = make_synthesis_config(speedup_ratio=2.0)

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )

        result = loader._apply_synthesis(data)

        assert result["session-1"][0].timestamp == 500
        assert result["session-1"][1].timestamp == 1000

    def test_prefix_len_multiplier_extends_hash_ids(self, mock_prompt_generator):
        """Test that prefix_len_multiplier extends hash_ids."""
        data = {
            "session-1": [
                MooncakeTrace(input_length=512, output_length=64, hash_ids=[1, 2]),
            ],
        }
        user_config = make_synthesis_config(prefix_len_multiplier=2.0)

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )

        result = loader._apply_synthesis(data)

        # Hash IDs should be extended
        assert len(result["session-1"][0].hash_ids) > 2

    def test_max_isl_caps_input_length(self, mock_prompt_generator):
        """Test that max_isl caps synthesized input_length."""
        data = {
            "session-1": [
                MooncakeTrace(input_length=5000, output_length=64),
            ],
        }
        user_config = make_synthesis_config(max_isl=4096)

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )

        result = loader._apply_synthesis(data)

        assert result["session-1"][0].input_length <= 4096

    @pytest.mark.parametrize(
        "speedup,input_ts,expected_ts",
        [
            (1.0, 1000, 1000),
            (2.0, 1000, 500),
            (4.0, 1000, 250),
            (0.5, 1000, 2000),
        ],
    )  # fmt: skip
    def test_speedup_ratio_variations(
        self, mock_prompt_generator, speedup, input_ts, expected_ts
    ):
        """Parametrized test for various speedup ratios."""
        data = {
            "session-1": [
                MooncakeTrace(input_length=512, output_length=64, timestamp=input_ts),
            ],
        }
        user_config = make_synthesis_config(speedup_ratio=speedup)

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )

        result = loader._apply_synthesis(data)

        assert result["session-1"][0].timestamp == expected_ts

    # ============================================================================
    # Field Preservation
    # ============================================================================

    def test_delay_field_preserved(self, mock_prompt_generator):
        """Test that delay field is preserved through synthesis."""
        data = {
            "session-1": [
                MooncakeTrace(input_length=512, output_length=64, delay=500),
                MooncakeTrace(input_length=512, output_length=64, delay=1000),
            ],
        }
        user_config = make_synthesis_config(speedup_ratio=2.0)

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )

        result = loader._apply_synthesis(data)

        assert result["session-1"][0].delay == 500
        assert result["session-1"][1].delay == 1000

    # ============================================================================
    # Edge Cases
    # ============================================================================

    def test_empty_input(self, mock_prompt_generator):
        """Test synthesis with empty input data."""
        user_config = make_synthesis_config(prefix_len_multiplier=2.0)

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )

        result = loader._apply_synthesis({})

        assert result == {}

    def test_empty_session_preserved(self, mock_prompt_generator):
        """Test that empty sessions are preserved, not dropped."""
        data = {
            "empty-session": [],
            "non-empty": [
                MooncakeTrace(input_length=512, output_length=64),
            ],
        }
        user_config = make_synthesis_config(speedup_ratio=2.0)

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )

        result = loader._apply_synthesis(data)

        assert set(result.keys()) == {"empty-session", "non-empty"}
        assert result["empty-session"] == []
        assert len(result["non-empty"]) == 1

    def test_block_size_passed_to_synthesis(self, mock_prompt_generator):
        """Test that user-configured block_size is passed to synthesis."""
        data = {
            "session-1": [
                MooncakeTrace(input_length=512, output_length=64, hash_ids=[1, 2]),
            ],
        }
        # Use non-default block_size (256 instead of default 512)
        user_config = make_synthesis_config(prefix_len_multiplier=2.0, block_size=256)

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )

        # Verify block_size is set correctly on loader
        assert loader._block_size == 256

        # Apply synthesis - block_size affects hash_id/input_length calculations
        result = loader._apply_synthesis(data)

        # With block_size=256 and prefix_len_multiplier=2.0:
        # Original: 2 hash_ids, input_length=512 (2 full blocks of 256)
        # Extended: should add more hash_ids based on 256-token blocks
        assert len(result["session-1"]) == 1
        trace = result["session-1"][0]
        assert trace.input_length > 512  # Should be extended

    def test_traces_without_hash_ids(self, mock_prompt_generator):
        """Test synthesis with traces that have no hash_ids."""
        data = {
            "session-1": [
                MooncakeTrace(input_length=512, output_length=64),
                MooncakeTrace(input_length=768, output_length=128),
            ],
        }
        user_config = make_synthesis_config(speedup_ratio=2.0)

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )

        result = loader._apply_synthesis(data)

        assert len(result["session-1"]) == 2
        for trace in result["session-1"]:
            assert trace.input_length is not None
            assert trace.output_length is not None

    def test_single_trace_single_session(self, mock_prompt_generator):
        """Test minimal case: one session, one trace."""
        data = {
            "only-session": [
                MooncakeTrace(input_length=512, output_length=64, hash_ids=[1]),
            ],
        }
        user_config = make_synthesis_config(prefix_len_multiplier=1.5)

        loader = MooncakeTraceDatasetLoader(
            filename="dummy.jsonl",
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )

        result = loader._apply_synthesis(data)

        assert "only-session" in result
        assert len(result["only-session"]) == 1
        assert isinstance(result["only-session"][0], MooncakeTrace)

    # ============================================================================
    # End-to-End: load_dataset with synthesis
    # ============================================================================

    def test_load_dataset_applies_synthesis(
        self, create_jsonl_file, mock_prompt_generator
    ):
        """Test that load_dataset applies synthesis when configured."""
        content = [
            '{"input_length": 512, "output_length": 64, "hash_ids": [1, 2], "timestamp": 1000}',
            '{"input_length": 768, "output_length": 128, "hash_ids": [1, 2, 3], "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)

        user_config = make_synthesis_config(speedup_ratio=2.0)

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        # Timestamps should be scaled by speedup_ratio
        traces = list(dataset.values())
        assert traces[0][0].timestamp == 500
        assert traces[1][0].timestamp == 1000

    def test_load_dataset_skips_synthesis_when_disabled(
        self, create_jsonl_file, mock_prompt_generator
    ):
        """Test that load_dataset skips synthesis when not configured."""
        content = [
            '{"input_length": 512, "output_length": 64, "timestamp": 1000}',
        ]
        filename = create_jsonl_file(content)

        # Default config - synthesis disabled
        user_config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))

        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            user_config=user_config,
            prompt_generator=mock_prompt_generator,
        )
        dataset = loader.load_dataset()

        # Timestamp should be unchanged
        traces = list(dataset.values())
        assert traces[0][0].timestamp == 1000
