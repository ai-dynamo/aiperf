# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for mooncake_trace functionality in UserConfig.

This module tests:
1. get_effective_request_count() - request count logic for mooncake_trace vs other datasets
2. _should_use_fixed_schedule_for_mooncake_trace() - timestamp detection for scheduling
3. Integration with existing UserConfig functionality

All tests use proper mocking to avoid file I/O dependencies.
"""

from unittest.mock import mock_open, patch

import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    UserConfig,
)
from aiperf.common.enums import CustomDatasetType


class TestMooncakeTraceRequestCount:
    """Test get_effective_request_count() for mooncake_trace datasets."""

    def test_no_custom_dataset_uses_configured_count(self):
        """Test that configured request count is used when no custom dataset."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=100),
        )

        result = config.get_effective_request_count()
        assert result == 100

    def test_no_custom_dataset_uses_default_count(self):
        """Test that default request count is used when no explicit count."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            # Uses default loadgen with request_count=10
        )

        result = config.get_effective_request_count()
        assert result == 10

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_mooncake_trace_uses_dataset_size_always(self, mock_is_file, mock_exists):
        """Test that mooncake_trace always uses dataset size, never request_count."""
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}\n'
            '{"input_length": 300, "hash_ids": [3], "timestamp": 3000}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=999),  # Should be ignored
            input=InputConfig(
                file="/fake/path/test.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = config.get_effective_request_count()
            assert result == 3  # Dataset size, not configured count

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_mooncake_trace_skips_empty_lines(self, mock_is_file, mock_exists):
        """Test that empty lines are not counted in mooncake_trace files."""
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            "\n"  # Empty line
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}\n'
            "   \n"  # Whitespace line
            '{"input_length": 300, "hash_ids": [3], "timestamp": 3000}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=100),
            input=InputConfig(
                file="/fake/path/test.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = config.get_effective_request_count()
            assert result == 3  # Only non-empty lines counted

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_mooncake_trace_empty_file_raises_error(self, mock_is_file, mock_exists):
        """Test that empty mooncake_trace file raises an error."""
        mock_file_content = ""

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=50),
            input=InputConfig(
                file="/fake/path/empty.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with (
            patch("builtins.open", mock_open(read_data=mock_file_content)),
            pytest.raises(ValueError, match="Empty mooncake_trace dataset file"),
        ):
            config.get_effective_request_count()

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_mooncake_trace_file_error_raises_exception(
        self, mock_is_file, mock_exists
    ):
        """Test that mooncake_trace file read errors raise exceptions."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=42),
            input=InputConfig(
                file="/fake/path/test.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with (
            patch("builtins.open", side_effect=OSError("File read error")),
            pytest.raises(
                ValueError, match="Could not read mooncake_trace dataset file"
            ),
        ):
            config.get_effective_request_count()

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_other_custom_dataset_uses_request_count(self, mock_is_file, mock_exists):
        """Test that non-mooncake_trace custom datasets always use request count."""
        mock_file_content = '{"some": "data"}\n{"other": "data"}\n'

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=75),
            input=InputConfig(
                file="/fake/path/other.jsonl",
                custom_dataset_type=CustomDatasetType.SINGLE_TURN,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = config.get_effective_request_count()
            assert result == 75  # Always uses configured count for non-mooncake_trace

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_other_custom_dataset_uses_default_count(self, mock_is_file, mock_exists):
        """Test that non-mooncake_trace custom datasets use default when no explicit count."""
        mock_file_content = '{"some": "data"}\n{"other": "data"}\n'

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            # No explicit loadgen - uses default request_count=10
            input=InputConfig(
                file="/fake/path/other.jsonl",
                custom_dataset_type=CustomDatasetType.SINGLE_TURN,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = config.get_effective_request_count()
            assert result == 10  # Uses default count, not dataset size


class TestMooncakeTraceTimingDetection:
    """Test _should_use_fixed_schedule_for_mooncake_trace() for automatic timing detection."""

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_mooncake_trace_with_timestamps_enables_fixed_schedule(
        self, mock_is_file, mock_exists
    ):
        """Test that timestamps in mooncake_trace trigger fixed schedule."""
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/with_timestamps.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = config._should_use_fixed_schedule_for_mooncake_trace()
            assert result is True

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_mooncake_trace_without_timestamps_no_fixed_schedule(
        self, mock_is_file, mock_exists
    ):
        """Test that missing timestamps don't trigger fixed schedule."""
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1]}\n'
            '{"input_length": 200, "hash_ids": [2]}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/without_timestamps.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = config._should_use_fixed_schedule_for_mooncake_trace()
            assert result is False

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_non_mooncake_trace_dataset_no_auto_detection(
        self, mock_is_file, mock_exists
    ):
        """Test that non-mooncake_trace datasets don't trigger auto-detection."""
        mock_file_content = '{"timestamp": 1000, "data": "test"}\n'

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/other_dataset.jsonl",
                custom_dataset_type=CustomDatasetType.SINGLE_TURN,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = config._should_use_fixed_schedule_for_mooncake_trace()
            assert result is False

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_empty_mooncake_trace_file_no_fixed_schedule(
        self, mock_is_file, mock_exists
    ):
        """Test behavior with empty mooncake_trace file."""
        mock_file_content = ""

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/empty.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = config._should_use_fixed_schedule_for_mooncake_trace()
            assert result is False

    def test_no_file_no_fixed_schedule(self):
        """Test behavior when no file is specified."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE),
        )

        result = config._should_use_fixed_schedule_for_mooncake_trace()
        assert result is False

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_mixed_entries_some_with_timestamps(self, mock_is_file, mock_exists):
        """Test file with mix of entries with/without timestamps."""
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            '{"input_length": 200, "hash_ids": [2]}\n'  # No timestamp
            '{"input_length": 300, "hash_ids": [3], "timestamp": 3000}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/mixed.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = config._should_use_fixed_schedule_for_mooncake_trace()
            # Should return True if ANY valid entry has timestamps
            assert result is True

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_invalid_json_lines_handled_gracefully(self, mock_is_file, mock_exists):
        """Test that invalid JSON lines don't crash detection."""
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            "invalid json line\n"  # This should be skipped
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/invalid.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = config._should_use_fixed_schedule_for_mooncake_trace()
            # Should still detect timestamps from valid lines
            assert result is True

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_file_read_error_handled_gracefully(self, mock_is_file, mock_exists):
        """Test that file read errors are handled gracefully."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/test.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        # Mock the file reading to raise an exception
        with patch("builtins.open", side_effect=OSError("File read error")):
            result = config._should_use_fixed_schedule_for_mooncake_trace()
            # Should return False on read errors, not crash
            assert result is False
