# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for UserConfig._should_use_fixed_schedule_for_mooncake_trace() method.

This module comprehensively tests the automatic detection of whether mooncake_trace
datasets should use fixed schedule timing based on the presence of timestamp fields.

The logic being tested:
- mooncake_trace with timestamps → fixed schedule enabled
- mooncake_trace without timestamps → fixed schedule disabled
- Non-mooncake_trace datasets → no auto-detection (always disabled)
- Error cases → graceful fallback to disabled
"""

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    UserConfig,
)
from aiperf.common.enums import CustomDatasetType


class TestTimingDetectionCore:
    """Test core timing detection logic."""

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

        assert config._should_use_fixed_schedule_for_mooncake_trace() is True

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

        assert config._should_use_fixed_schedule_for_mooncake_trace() is False

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

        assert config._should_use_fixed_schedule_for_mooncake_trace() is False

    def test_no_file_specified_no_fixed_schedule(self):
        """Test behavior when no file is specified."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE),
        )

        assert config._should_use_fixed_schedule_for_mooncake_trace() is False


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

        # Should return True if ANY valid entry has timestamps
        assert config._should_use_fixed_schedule_for_mooncake_trace() is True

    def test_null_timestamp_values_ignored(self, tmp_path):
        """Test that null timestamp values don't trigger fixed schedule."""
        test_file = tmp_path / "null_timestamps.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1], "timestamp": null}\n'
            '{"input_length": 200, "hash_ids": [2], "timestamp": null}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        assert config._should_use_fixed_schedule_for_mooncake_trace() is False

    def test_zero_timestamp_values_trigger_fixed_schedule(self, tmp_path):
        """Test that zero timestamp values still trigger fixed schedule."""
        test_file = tmp_path / "zero_timestamps.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1], "timestamp": 0}\n'
            '{"input_length": 200, "hash_ids": [2], "timestamp": 0}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        assert config._should_use_fixed_schedule_for_mooncake_trace() is True

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

        assert config._should_use_fixed_schedule_for_mooncake_trace() is False


class TestTimingDetectionErrorHandling:
    """Test error handling for timing detection."""

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

        # Should still detect timestamps from valid lines
        assert config._should_use_fixed_schedule_for_mooncake_trace() is True

    def test_file_read_error_handled_gracefully(self, tmp_path):
        """Test that file read errors are handled gracefully."""
        # Create file first to pass InputConfig validation, then delete it
        nonexistent_file = tmp_path / "nonexistent.jsonl"
        nonexistent_file.write_text('{"input_length": 100, "hash_ids": [1]}\n')

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(nonexistent_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        # Now delete the file to trigger the error
        nonexistent_file.unlink()

        # Should return False on read errors, not crash
        assert config._should_use_fixed_schedule_for_mooncake_trace() is False

    def test_permission_denied_handled_gracefully(self, tmp_path):
        """Test that permission errors are handled gracefully."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
        )

        # Make file unreadable (this may not work on all systems)
        import os

        try:
            os.chmod(test_file, 0o000)

            config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=str(test_file),
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                ),
            )

            # Should return False on permission errors, not crash
            assert config._should_use_fixed_schedule_for_mooncake_trace() is False

        finally:
            # Restore permissions for cleanup
            os.chmod(test_file, 0o644)


class TestTimingDetectionSpecialCases:
    """Test special cases for timing detection."""

    def test_only_empty_lines_no_fixed_schedule(self, tmp_path):
        """Test file with only empty/whitespace lines."""
        test_file = tmp_path / "only_empty.jsonl"
        test_file.write_text("\n   \n\t\n")

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        assert config._should_use_fixed_schedule_for_mooncake_trace() is False

    def test_timestamp_field_wrong_type_accepted_by_implementation(self, tmp_path):
        """Test that current implementation accepts various timestamp types."""
        test_file = tmp_path / "string_timestamps.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1], "timestamp": "1000"}\n'  # String timestamp
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}\n'  # Numeric timestamp
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        # Current implementation detects any timestamp field presence, regardless of type
        assert config._should_use_fixed_schedule_for_mooncake_trace() is True

    def test_large_file_stops_at_first_timestamp(self, tmp_path):
        """Test that detection stops at first valid timestamp (performance)."""
        test_file = tmp_path / "large.jsonl"
        lines = ['{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n']
        lines.extend(
            ['{"input_length": 100, "hash_ids": [1]}\n'] * 1000
        )  # Many lines without timestamps
        test_file.write_text("".join(lines))

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        # Should detect immediately from first line
        assert config._should_use_fixed_schedule_for_mooncake_trace() is True
