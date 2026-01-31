# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for mooncake_trace functionality in UserConfig.

This module tests:
1. get_effective_request_count() - request count logic for mooncake_trace vs other datasets
2. Integration with existing UserConfig functionality
"""

from unittest.mock import mock_open, patch

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    UserConfig,
)
from aiperf.plugin.enums import CustomDatasetType


class TestMooncakeTraceRequestCount:
    """Test get_effective_request_count() for mooncake_trace datasets."""

    def test_no_custom_dataset_uses_configured_count(self):
        """Test that configured request count is used when no custom dataset."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=100),
        )

        assert config.loadgen.request_count == 100

    def test_no_custom_dataset_uses_default_count(self):
        """Test that default request count is used when no explicit count."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
        )

        assert config.loadgen.request_count == 10

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_mooncake_trace_uses_dataset_size(self, mock_is_file, mock_exists):
        """Test that mooncake_trace uses dataset size."""
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
            result = config._count_dataset_entries()
            assert result == 3

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
            result = config._count_dataset_entries()
            assert result == 3  # Only non-empty lines counted

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_mooncake_trace_empty_file_returns_zero(self, mock_is_file, mock_exists):
        """Test that empty mooncake_trace file returns 0."""
        mock_file_content = ""

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=50),
            input=InputConfig(
                file="/fake/path/empty.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = config._count_dataset_entries()
            assert result == 0

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_mooncake_trace_file_error_returns_zero(self, mock_is_file, mock_exists):
        """Test that mooncake_trace file read errors return 0 and log error."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=42),
            input=InputConfig(
                file="/fake/path/test.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", side_effect=OSError("File read error")):
            result = config._count_dataset_entries()
            assert result == 0

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_other_custom_dataset_counts_file_entries(self, mock_is_file, mock_exists):
        """Test that non-mooncake_trace custom datasets count file entries."""
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
            result = config._count_dataset_entries()
            # Returns actual file line count, not request_count
            assert result == 2

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_other_custom_dataset_uses_default_count(self, mock_is_file, mock_exists):
        """Test that non-mooncake_trace custom datasets use default when no explicit count."""
        mock_file_content = '{"some": "data"}\n{"other": "data"}\n'

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/other.jsonl",
                custom_dataset_type=CustomDatasetType.SINGLE_TURN,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            assert config.loadgen.request_count == 10

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_count_dataset_entries_with_edge_cases(self, mock_is_file, mock_exists):
        """Test _count_dataset_entries handles empty lines and malformed JSON."""
        mock_file_content = (
            '{"input_length": 50, "timestamp": 1000}\n'
            "\n"  # Empty line
            "   \n"  # Whitespace-only line
            '{"input_length": 100}\n'  # Valid JSON
            "invalid json line\n"  # Malformed JSON
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/test.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            # Should count all non-empty lines (including malformed JSON)
            count = config._count_dataset_entries()
            assert count == 3  # 3 non-empty/non-whitespace lines


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
    def test_file_parsing_with_empty_lines_and_malformed_json(
        self, mock_is_file, mock_exists
    ):
        """Test file parsing handles empty lines and malformed JSON gracefully."""
        # Content with empty lines, whitespace, and malformed JSON
        mock_file_content = (
            '{"input_length": 50, "timestamp": 1000}\n'
            "\n"  # Empty line
            "   \n"  # Whitespace-only line
            '{"input_length": 100}\n'  # Valid JSON, no timestamp
            "\n"  # Another empty line
            "invalid json line\n"  # Malformed JSON
            '{"missing": "required_fields"}\n'  # JSON missing required fields
            "   \n"  # More whitespace
            '{"input_length": 150, "timestamp": 3000}\n'  # Valid with timestamp
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/test.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            # Should handle malformed JSON gracefully and still detect timestamps
            has_timestamps = config._should_use_fixed_schedule_for_mooncake_trace()
            assert has_timestamps is True  # Should find valid timestamps despite errors

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_empty_file_timing_detection(self, mock_is_file, mock_exists):
        """Test timing detection with completely empty files."""
        mock_file_content = ""  # Completely empty file

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/empty.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            # Should handle empty file gracefully
            assert config._should_use_fixed_schedule_for_mooncake_trace() is False

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_only_malformed_json_timing_detection(self, mock_is_file, mock_exists):
        """Test timing detection with only malformed JSON entries."""
        mock_file_content = (
            "not json at all\n"
            '{"incomplete": \n'  # Incomplete JSON
            "random text\n"
            '{"missing_required": "fields"}\n'  # Valid JSON but missing required fields
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/malformed.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            # Should find no timestamps in malformed JSON
            assert config._should_use_fixed_schedule_for_mooncake_trace() is False
