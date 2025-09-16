# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for UserConfig.get_effective_request_count() method.

This module comprehensively tests the request count determination logic:
- Normal scenarios: No custom dataset, explicit vs default counts
- mooncake_trace behavior: Always uses dataset size for exact replay
- Other custom datasets: Always use configured request_count
- Error handling: Empty files, read errors, malformed data
- Edge cases: Empty lines, whitespace, mixed content

The core rule being tested:
- mooncake_trace datasets → dataset size (exact replay)
- Everything else → configured request_count
"""

import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    UserConfig,
)
from aiperf.common.enums import CustomDatasetType


class TestRequestCountBasicLogic:
    """Test request count logic for non-dataset scenarios."""

    def test_no_dataset_uses_explicit_count(self):
        """Test explicit request count is used when no custom dataset."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=100),
        )

        assert config.get_effective_request_count() == 100

    def test_no_dataset_uses_default_count(self):
        """Test default request count when no explicit count specified."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            # No explicit loadgen - uses defaults
        )

        assert config.get_effective_request_count() == 10  # LoadGeneratorConfig default


class TestMooncakeTraceRequestCount:
    """Test mooncake_trace specific request count behavior (always dataset size)."""

    def test_mooncake_trace_overrides_explicit_count(self, tmp_path):
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

        assert config.get_effective_request_count() == 3  # Dataset size wins

    def test_mooncake_trace_overrides_default_count(self, tmp_path):
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

        assert config.get_effective_request_count() == 2  # Dataset size, not default

    def test_mooncake_trace_skips_empty_lines(self, tmp_path):
        """Test that empty lines are not counted in mooncake_trace files."""
        test_file = tmp_path / "test_with_empty.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1]}\n'
            "\n"  # Empty line
            '{"input_length": 200, "hash_ids": [2]}\n'
            "   \n"  # Whitespace line
            '{"input_length": 300, "hash_ids": [3]}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=100),  # Should be overridden
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        assert config.get_effective_request_count() == 3  # Only non-empty lines


class TestOtherCustomDatasetRequestCount:
    """Test non-mooncake_trace custom datasets (always use request_count)."""

    def test_other_dataset_uses_explicit_count(self, tmp_path):
        """Test that non-mooncake_trace datasets always use explicit request count."""
        test_file = tmp_path / "other.jsonl"
        test_file.write_text('{"some": "data"}\n{"other": "data"}\n')

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=75),
            input=InputConfig(
                file=str(test_file), custom_dataset_type=CustomDatasetType.SINGLE_TURN
            ),
        )

        assert (
            config.get_effective_request_count() == 75
        )  # Uses config, ignores file size (2)

    def test_other_dataset_uses_default_count(self, tmp_path):
        """Test that non-mooncake_trace datasets use default when no explicit count."""
        test_file = tmp_path / "other.jsonl"
        test_file.write_text('{"some": "data"}\n{"other": "data"}\n')

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            # No explicit loadgen - uses default request_count=10
            input=InputConfig(
                file=str(test_file), custom_dataset_type=CustomDatasetType.SINGLE_TURN
            ),
        )

        assert (
            config.get_effective_request_count() == 10
        )  # Uses default, ignores file size


class TestRequestCountErrorHandling:
    """Test error handling for request count determination."""

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

    def test_mooncake_trace_nonexistent_file_raises_exception(self, tmp_path):
        """Test that nonexistent mooncake_trace file raises exception during get_effective_request_count."""
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

        with pytest.raises(
            ValueError, match="Could not read mooncake_trace dataset file"
        ):
            config.get_effective_request_count()


class TestRequestCountEdgeCases:
    """Test edge cases for request count determination."""

    def test_mooncake_trace_file_with_only_empty_lines(self, tmp_path):
        """Test mooncake_trace file with only empty/whitespace lines."""
        test_file = tmp_path / "only_empty.jsonl"
        test_file.write_text("\n   \n\t\n")

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        with pytest.raises(ValueError, match="Empty mooncake_trace dataset file"):
            config.get_effective_request_count()

    def test_mooncake_trace_with_various_formats(self, tmp_path):
        """Test mooncake_trace with different valid entry formats."""
        test_file = tmp_path / "mixed_formats.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1]}\n'  # No timestamp
            '{"text_input": "hello", "hash_ids": [2], "timestamp": 1000}\n'  # With timestamp
            '{"input_length": 200, "hash_ids": [3], "output_length": 50}\n'  # With output_length
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        assert config.get_effective_request_count() == 3  # All valid entries counted
