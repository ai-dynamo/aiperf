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
    """Test the get_effective_request_count() method for mooncake_trace validation."""

    def test_default_behavior_without_custom_dataset(self):
        """Test that default request count is used when no custom dataset."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            load_generator=LoadGeneratorConfig(request_count=100),
        )

        result = config.get_effective_request_count()
        assert result == 100

    def test_mooncake_trace_rejects_explicit_request_count(self, tmp_path):
        """Test that explicit request_count raises exception for mooncake_trace datasets."""
        # Create a test file with 3 lines
        test_file = tmp_path / "test.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}\n'
            '{"input_length": 300, "hash_ids": [3], "timestamp": 3000}\n'
        )

        with pytest.raises(
            ValueError,
            match="request_count cannot be explicitly set for mooncake_trace datasets",
        ):
            config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                load_generator=LoadGeneratorConfig(
                    request_count=100
                ),  # This should cause exception
                input=InputConfig(
                    file=str(test_file),
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                ),
            )
            config.get_effective_request_count()

    def test_mooncake_trace_uses_dataset_size(self, tmp_path):
        """Test that mooncake_trace uses dataset size when no explicit request_count."""
        # Create a test file with 3 lines
        test_file = tmp_path / "test.jsonl"
        test_file.write_text(
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
            '{"input_length": 200, "hash_ids": [2], "timestamp": 2000}\n'
            '{"input_length": 300, "hash_ids": [3], "timestamp": 3000}\n'
        )

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            # No explicit request_count set - should use default
            input=InputConfig(
                file=str(test_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )

        result = config.get_effective_request_count()
        assert result == 3  # Dataset size should be used

    def test_mooncake_trace_with_empty_file(self, tmp_path):
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

        result = config.get_effective_request_count()
        assert result == 0

    def test_other_custom_dataset_respects_explicit_request_count(self, tmp_path):
        """Test that non-mooncake_trace custom datasets respect explicit request_count."""
        test_file = tmp_path / "other.jsonl"
        test_file.write_text('{"some": "data"}\n{"other": "data"}\n')

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            load_generator=LoadGeneratorConfig(request_count=75),  # Should be respected
            input=InputConfig(
                file=str(test_file), custom_dataset_type=CustomDatasetType.SINGLE_TURN
            ),
        )

        result = config.get_effective_request_count()
        assert result == 75  # Explicit request_count should be used

    def test_other_custom_dataset_uses_dataset_size_when_not_explicit(self, tmp_path):
        """Test that non-mooncake_trace custom datasets use dataset size when no explicit request_count."""
        test_file = tmp_path / "other.jsonl"
        test_file.write_text('{"some": "data"}\n{"other": "data"}\n')

        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            # No explicit request_count - should use dataset size
            input=InputConfig(
                file=str(test_file), custom_dataset_type=CustomDatasetType.SINGLE_TURN
            ),
        )

        result = config.get_effective_request_count()
        assert result == 2  # Dataset size should be used


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
