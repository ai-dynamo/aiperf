# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from aiperf.common.enums import CustomDatasetType
from aiperf.services.dataset.loader import MooncakeTraceDatasetLoader
from aiperf.services.dataset.loader.models import MooncakeTrace


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
            MooncakeTrace()


class TestMooncakeTraceDatasetLoader:
    """Basic functionality tests for MooncakeTraceDatasetLoader."""

    @pytest.fixture
    def mock_prompt_generator(self):
        """Create a mock prompt generator for testing."""
        generator = Mock()
        generator.generate.return_value = "Generated prompt text"
        return generator

    def test_load_dataset_basic_functionality(
        self, create_jsonl_file, mock_prompt_generator
    ):
        """Test basic JSONL file loading."""
        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123, 456], "timestamp": 1000}',
            '{"input_length": 200, "output_length": 75, "hash_ids": [789], "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(filename, mock_prompt_generator)
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
        self, create_jsonl_file, mock_prompt_generator
    ):
        """Test that empty lines are skipped."""
        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123], "timestamp": 1000}',
            "",  # Empty line
            '{"input_length": 200, "output_length": 75, "hash_ids": [456], "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(filename, mock_prompt_generator)
        result = loader.load_dataset()

        assert len(result) == 2  # Should skip empty line

    def test_load_dataset_with_timestamps(
        self, create_jsonl_file, mock_prompt_generator
    ):
        """Test loading dataset with timestamp fields."""
        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123], "timestamp": 1000}',
            '{"input_length": 200, "output_length": 75, "hash_ids": [456], "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)

        loader = MooncakeTraceDatasetLoader(filename, mock_prompt_generator)
        dataset = loader.load_dataset()

        traces = list(dataset.values())
        assert traces[0][0].timestamp == 1000
        assert traces[1][0].timestamp == 2000

    def test_convert_to_conversations(self, mock_prompt_generator):
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

        loader = MooncakeTraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
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

    def test_convert_to_conversations_empty_data(self, mock_prompt_generator):
        """Test conversion with empty trace data."""
        loader = MooncakeTraceDatasetLoader("dummy.jsonl", mock_prompt_generator)
        conversations = loader.convert_to_conversations({})

        assert len(conversations) == 0
