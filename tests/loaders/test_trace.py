# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from aiperf.common.config import EndpointConfig, InputConfig, UserConfig
from aiperf.common.enums import CustomDatasetType
from aiperf.dataset import MooncakeTrace, MooncakeTraceDatasetLoader


class TestMooncakeTrace:
    """Basic functionality tests for MooncakeTrace model."""

    def test_create_with_required_fields_only(self):
        """Test creating MooncakeTrace with only required fields."""
        data = MooncakeTrace(
            input_length=100, output_length=50, hash_ids=[123, 456, 789], timestamp=1000
        )

        assert data.input_length == 100
        assert data.output_length == 50
        assert data.hash_ids == [123, 456, 789]
        assert data.timestamp == 1000
        assert data.type == CustomDatasetType.MOONCAKE_TRACE

    def test_validation_missing_fields_errors(self):
        """Test validation errors for MooncakeTrace."""
        # Missing required fields
        with pytest.raises(ValueError):
            MooncakeTrace()  # type: ignore


class TestMooncakeTraceDatasetLoader:
    """Basic functionality tests for MooncakeTraceDatasetLoader."""

    @pytest.fixture
    def mock_prompt_generator(self):
        """Create a mock prompt generator for testing."""
        generator = Mock()
        generator.generate.return_value = "Generated prompt text"
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
            filename, mock_prompt_generator, default_user_config
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
            filename, mock_prompt_generator, default_user_config
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
            filename, mock_prompt_generator, default_user_config
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
            filename, mock_prompt_generator, user_config
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
        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123], "timestamp": 1000}',
            '{"input_length": 150, "output_length": 60, "hash_ids": [456], "timestamp": 2000}',
            '{"input_length": 200, "output_length": 70, "hash_ids": [789], "timestamp": 3000}',
        ]
        filename = create_jsonl_file(content)

        user_config = self.make_user_config(start_offset, end_offset)
        loader = MooncakeTraceDatasetLoader(
            filename, mock_prompt_generator, user_config
        )
        loader.load_dataset()

        # Check that the skipped traces message is logged
        assert f"Skipped {expected_skipped:,} traces" in caplog.text

    def test_convert_to_conversations(self, mock_prompt_generator, default_user_config):
        """Test conversion of trace data to conversations."""
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
            "dummy.jsonl", mock_prompt_generator, default_user_config
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
            "dummy.jsonl", mock_prompt_generator, default_user_config
        )
        conversations = loader.convert_to_conversations({})

        assert len(conversations) == 0
