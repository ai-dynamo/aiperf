# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for UserConfig.get_effective_request_count() method.

This module tests the logic that determines how many requests should be sent
based on the dataset type and configuration. The core behavior is:

- mooncake_trace datasets: Always use dataset size (exact replay)
- All other scenarios: Use configured request_count
"""

import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    UserConfig,
)
from aiperf.common.enums import CustomDatasetType


class TestRequestCountLogic:
    """Test the get_effective_request_count() method across all scenarios."""

    def test_no_dataset_uses_configured_count(self):
        """Test that configured request count is used when no custom dataset."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=100),
        )

        result = config.get_effective_request_count()
        assert result == 100

    def test_no_dataset_uses_default_count(self):
        """Test that default request count is used when no explicit count."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            # No explicit loadgen - uses defaults
        )

        result = config.get_effective_request_count()
        assert result == 10  # LoadGeneratorConfig default

    def test_non_mooncake_custom_dataset_uses_configured_count(self, tmp_path):
        """Test that non-mooncake_trace custom datasets always use request count."""
        test_file = tmp_path / "other.jsonl"
        test_file.write_text('{"some": "data"}\n{"other": "data"}\n')

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=75),
            input=InputConfig(
                file=str(test_file), custom_dataset_type=CustomDatasetType.SINGLE_TURN
            ),
        )

        result = config.get_effective_request_count()
        assert result == 75  # Uses configured count, ignores file size (2)

    def test_non_mooncake_custom_dataset_uses_default_count(self, tmp_path):
        """Test that non-mooncake_trace custom datasets use default when no explicit count."""
        test_file = tmp_path / "other.jsonl"
        test_file.write_text('{"some": "data"}\n{"other": "data"}\n')

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            # No explicit loadgen - uses default request_count=10
            input=InputConfig(
                file=str(test_file), custom_dataset_type=CustomDatasetType.SINGLE_TURN
            ),
        )

        result = config.get_effective_request_count()
        assert result == 10  # Uses default count, ignores file size (2)


class TestMooncakeTraceRequestCount:
    """Test mooncake_trace specific request count behavior."""

    def test_mooncake_trace_always_uses_dataset_size(self, tmp_path):
        """Test that mooncake_trace dataset size always overrides request count."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}\n'
            '{"input_length": 300, "hash_ids": [3], "timestamp": 3000}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=999),  # Should be overridden
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        result = config.get_effective_request_count()
        assert result == 3  # Dataset size overrides configured count

    def test_mooncake_trace_uses_dataset_size_with_default_config(self, tmp_path):
        """Test that mooncake_trace uses dataset size even with default config."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1]}\n'
            '{"input_length": 200, "hash_ids": [2]}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            # No explicit loadgen - default would be 10
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        result = config.get_effective_request_count()
        assert result == 2  # Dataset size, not default (10)

    def test_mooncake_trace_skips_empty_lines(self, tmp_path):
        """Test that empty lines are not counted in mooncake_trace files."""
        test_file = tmp_path / "test_with_empty.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            "\n"  # Empty line
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}\n'
            "   \n"  # Whitespace line
            '{"input_length": 300, "hash_ids": [3], "timestamp": 3000}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=100),  # Should be overridden
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        result = config.get_effective_request_count()
        assert result == 3  # Only non-empty lines counted


class TestMooncakeTraceErrorCases:
    """Test error handling for mooncake_trace datasets."""

    def test_mooncake_trace_empty_file_raises_error(self, tmp_path):
        """Test that empty mooncake_trace file raises an error."""
        test_file = tmp_path / "empty.jsonl"
        test_file.write_text("")

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=50),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with pytest.raises(ValueError, match="Empty mooncake_trace dataset file"):
            config.get_effective_request_count()

    def test_mooncake_trace_file_read_error_raises_exception(self, tmp_path):
        """Test that mooncake_trace file read errors raise exceptions."""
        # Use a directory path to trigger read error
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=42),
            input=InputConfig(
                file=str(tmp_path),  # Directory, not file - will cause error
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with pytest.raises(
            ValueError, match="Could not read mooncake_trace dataset file"
        ):
            config.get_effective_request_count()
