# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for mooncake_trace functionality in UserConfig.

This module tests:
1. get_effective_request_count() - request count logic for mooncake_trace vs other datasets
2. _should_use_fixed_schedule_for_mooncake_trace() - timestamp detection for scheduling
3. Integration with existing UserConfig functionality
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
        )

        result = config.get_effective_request_count()
        assert result == 10

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
            result = config.get_effective_request_count()
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
            assert result == 75

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
            result = config.get_effective_request_count()
            assert result == 10


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


class TestMooncakeTraceFixedScheduleValidation:
    """Test validation that all entries have timestamps when fixed schedule is enabled."""

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_fixed_schedule_explicit_all_entries_valid(self, mock_is_file, mock_exists):
        """Test that explicit fixed schedule succeeds when all entries have timestamps."""
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}\n'
            '{"input_length": 300, "hash_ids": [3], "timestamp": 3000}\n'
        )

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file="/fake/path/all_timestamps.jsonl",
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    fixed_schedule=True,  # Explicitly enable fixed schedule
                ),
            )
            assert config._timing_mode.name == "FIXED_SCHEDULE"

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_fixed_schedule_explicit_missing_timestamps_raises_error(
        self, mock_is_file, mock_exists
    ):
        """Test that explicit fixed schedule fails when some entries lack timestamps."""
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            '{"input_length": 200, "hash_ids": [2]}\n'  # No timestamp - line 2
            '{"input_length": 300, "hash_ids": [3], "timestamp": 3000}\n'
            '{"input_length": 400, "hash_ids": [4]}\n'  # No timestamp - line 4
        )

        with (
            patch("builtins.open", mock_open(read_data=mock_file_content)),
            pytest.raises(
                ValueError,
                match=r"Fixed schedule mode requires all entries to have timestamps.*Found 2 entries without timestamps.*lines: 2, 4",
            ),
        ):
            UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file="/fake/path/mixed_timestamps.jsonl",
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    fixed_schedule=True,
                ),
            )

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_fixed_schedule_auto_missing_timestamps_raises_error(
        self, mock_is_file, mock_exists
    ):
        """Test that auto-enabled fixed schedule fails when some entries lack timestamps."""
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            '{"input_length": 200, "hash_ids": [2]}\n'  # No timestamp
            '{"input_length": 300, "hash_ids": [3], "timestamp": 3000}\n'
        )

        with (
            patch("builtins.open", mock_open(read_data=mock_file_content)),
            pytest.raises(
                ValueError,
                match=r"Fixed schedule mode requires all entries to have timestamps",
            ),
        ):
            UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file="/fake/path/mixed_timestamps.jsonl",
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    # fixed_schedule not explicitly set - should auto-enable and then fail validation
                ),
            )

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_fixed_schedule_invalid_json_raises_error(self, mock_is_file, mock_exists):
        """Test that invalid JSON entries are treated as missing timestamps."""
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            "invalid json line\n"  # Invalid JSON - line 2
            '{"input_length": 300, "hash_ids": [3], "timestamp": 3000}\n'
        )

        with (
            patch("builtins.open", mock_open(read_data=mock_file_content)),
            pytest.raises(
                ValueError,
                match=r"Fixed schedule mode requires all entries to have timestamps.*Found 1 entries without timestamps.*lines: 2",
            ),
        ):
            UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file="/fake/path/invalid_json.jsonl",
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    fixed_schedule=True,
                ),
            )

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_fixed_schedule_many_missing_timestamps_limits_error_display(
        self, mock_is_file, mock_exists
    ):
        """Test that error message limits the number of line numbers displayed."""
        # Create 10 lines, all without timestamps
        mock_file_content = (
            "\n".join(
                [
                    f'{{"input_length": {100 + i * 10}, "hash_ids": [{i}]}}'
                    for i in range(10)
                ]
            )
            + "\n"
        )

        with (
            patch("builtins.open", mock_open(read_data=mock_file_content)),
            pytest.raises(ValueError) as exc_info,
        ):
            UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file="/fake/path/many_missing.jsonl",
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    fixed_schedule=True,
                ),
            )

            error_message = str(exc_info.value)
            assert "Found 10 entries without timestamps" in error_message
            assert "lines: 1, 2, 3, 4, 5 (and 5 more)" in error_message

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_fixed_schedule_non_mooncake_trace_no_validation(
        self, mock_is_file, mock_exists
    ):
        """Test that non-mooncake_trace datasets don't trigger timestamp validation."""
        # Should not raise an error even though we're using fixed schedule
        # because it's not a mooncake_trace dataset
        with patch("builtins.open", mock_open(read_data="")):
            config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file="/fake/path/non_mooncake.jsonl",  # Need file for fixed_schedule
                    custom_dataset_type=CustomDatasetType.SINGLE_TURN,
                    fixed_schedule=True,
                ),
            )
            assert config._timing_mode.name == "FIXED_SCHEDULE"

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_no_fixed_schedule_no_validation(self, mock_is_file, mock_exists):
        """Test that non-fixed-schedule modes don't trigger validation."""
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1]}\n'  # No timestamps
            '{"input_length": 200, "hash_ids": [2]}\n'
        )

        # Should not perform validation when not using fixed schedule
        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file="/fake/path/mooncake_no_schedule.jsonl",
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    # fixed_schedule=False (default)
                ),
                loadgen=LoadGeneratorConfig(request_rate=10.0),  # Use request rate mode
            )
            assert config._timing_mode.name == "REQUEST_RATE"
