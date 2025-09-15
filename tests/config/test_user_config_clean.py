# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    UserConfig,
)
from aiperf.common.enums import CustomDatasetType


class TestGetEffectiveRequestCount:
    """Test the get_effective_request_count() method."""

    def test_default_behavior_without_custom_dataset(self):
        """Test that default request count is used when no custom dataset."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=100),
        )

        result = config.get_effective_request_count()
        assert result == 100

    def test_mooncake_trace_explicit_request_count_raises_exception(self, tmp_path):
        """Test that explicit request_count raises exception for mooncake_trace."""
        # Create a test file with 3 lines
        test_file = tmp_path / "test.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1]}\n'
            '{"input_length": 200, "hash_ids": [2]}\n'
            '{"input_length": 300, "hash_ids": [3]}\n'
        )

        with pytest.raises(
            ValueError,
            match="request_count cannot be explicitly set for mooncake_trace datasets",
        ):
            UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                loadgen=LoadGeneratorConfig(
                    request_count=100
                ),  # This should cause exception
                input=InputConfig(
                    file=str(test_file),
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                ),
            ).get_effective_request_count()

    def test_mooncake_trace_uses_dataset_size_when_no_explicit_count(self, tmp_path):
        """Test that mooncake_trace uses dataset size when no explicit request_count."""
        # Create a test file with 3 lines
        test_file = tmp_path / "test.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1]}\n'
            '{"input_length": 200, "hash_ids": [2]}\n'
            '{"input_length": 300, "hash_ids": [3]}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            # No explicit loadgen - will use defaults without explicit request_count
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        result = config.get_effective_request_count()
        assert result == 3  # Should use dataset size

    def test_mooncake_trace_empty_file_returns_zero(self, tmp_path):
        """Test that empty mooncake_trace file returns 0."""
        test_file = tmp_path / "empty.jsonl"
        test_file.write_text("")

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        result = config.get_effective_request_count()
        assert result == 0

    def test_mooncake_trace_skips_empty_lines(self, tmp_path):
        """Test that empty lines are not counted in mooncake_trace."""
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
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        result = config.get_effective_request_count()
        assert result == 3  # Only non-empty lines counted

    def test_non_mooncake_trace_custom_dataset_uses_explicit_count(self, tmp_path):
        """Test that non-mooncake_trace custom datasets can use explicit request_count."""
        test_file = tmp_path / "other.jsonl"
        test_file.write_text('{"text": "data1"}\n{"text": "data2"}\n')

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(
                request_count=75
            ),  # Should be used, not file size
            input=InputConfig(
                file=str(test_file), custom_dataset_type=CustomDatasetType.SINGLE_TURN
            ),
        )

        result = config.get_effective_request_count()
        assert result == 75  # Uses explicit count, not file size (2)

    def test_no_custom_dataset_uses_explicit_count(self):
        """Test that non-custom datasets use explicit request_count."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=42),
        )

        result = config.get_effective_request_count()
        assert result == 42


class TestFixedScheduleAutoDetection:
    """Test the _should_use_fixed_schedule_for_mooncake_trace() method."""

    def test_mooncake_trace_with_timestamps(self, tmp_path):
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

    def test_mooncake_trace_without_timestamps(self, tmp_path):
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

    def test_non_mooncake_trace_dataset(self, tmp_path):
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

    def test_empty_file(self, tmp_path):
        """Test behavior with empty file."""
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
