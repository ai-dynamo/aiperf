# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for UserConfig._should_use_fixed_schedule_for_mooncake_trace() method.

This module tests the automatic detection of whether mooncake_trace datasets
should use fixed schedule timing based on the presence of timestamp fields.
"""

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    UserConfig,
)
from aiperf.common.enums import CustomDatasetType


class TestMooncakeTraceTimingDetection:
    """Test automatic fixed schedule detection for mooncake_trace datasets."""

    def test_mooncake_trace_with_timestamps_enables_fixed_schedule(self, tmp_path):
        """Test that timestamps in mooncake_trace trigger fixed schedule."""
        test_file = tmp_path / "with_timestamps.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        result = config._should_use_fixed_schedule_for_mooncake_trace()
        assert result is True

    def test_mooncake_trace_without_timestamps_no_fixed_schedule(self, tmp_path):
        """Test that missing timestamps don't trigger fixed schedule."""
        test_file = tmp_path / "without_timestamps.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1]}\n'
            '{"input_length": 200, "hash_ids": [2]}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        result = config._should_use_fixed_schedule_for_mooncake_trace()
        assert result is False

    def test_non_mooncake_trace_dataset_no_auto_detection(self, tmp_path):
        """Test that non-mooncake_trace datasets don't trigger auto-detection."""
        test_file = tmp_path / "other_dataset.jsonl"
        test_file.write_text('{"timestamp": 1000, "data": "test"}\n')

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file), custom_dataset_type=CustomDatasetType.SINGLE_TURN
            ),
        )

        result = config._should_use_fixed_schedule_for_mooncake_trace()
        assert result is False

    def test_empty_mooncake_trace_file_no_fixed_schedule(self, tmp_path):
        """Test behavior with empty mooncake_trace file."""
        test_file = tmp_path / "empty.jsonl"
        test_file.write_text("")

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

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


class TestTimingDetectionEdgeCases:
    """Test edge cases for timing detection."""

    def test_mixed_entries_some_with_timestamps(self, tmp_path):
        """Test file with mix of entries with/without timestamps."""
        test_file = tmp_path / "mixed.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            '{"input_length": 200, "hash_ids": [2]}\n'  # No timestamp
            '{"input_length": 300, "hash_ids": [3], "timestamp": 3000}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        result = config._should_use_fixed_schedule_for_mooncake_trace()
        # Should return True if ANY valid entry has timestamps
        assert result is True

    def test_invalid_json_lines_handled_gracefully(self, tmp_path):
        """Test that invalid JSON lines don't crash detection."""
        test_file = tmp_path / "invalid.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            "invalid json line\n"  # This should be skipped
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        result = config._should_use_fixed_schedule_for_mooncake_trace()
        # Should still detect timestamps from valid lines
        assert result is True

    def test_file_read_error_handled_gracefully(self, tmp_path):
        """Test that file read errors are handled gracefully."""
        nonexistent_file = tmp_path / "nonexistent.jsonl"
        # Don't create the file

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(nonexistent_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        result = config._should_use_fixed_schedule_for_mooncake_trace()
        # Should return False on read errors, not crash
        assert result is False
