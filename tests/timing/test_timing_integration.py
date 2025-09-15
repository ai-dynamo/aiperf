# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    UserConfig,
)
from aiperf.common.enums import CustomDatasetType, TimingMode
from aiperf.timing.config import TimingManagerConfig


class TestTimingConfigurationIntegration:
    """Test timing configuration integration with effective request count."""

    @pytest.fixture
    def create_mooncake_trace_file(self):
        """Create a temporary mooncake trace file."""

        def _create_file(entry_count, include_timestamps=False):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                for i in range(entry_count):
                    if include_timestamps:
                        entry = f'{{"input_length": {100 + i * 50}, "hash_ids": [{i}], "timestamp": {1000 + i * 1000}}}'
                    else:
                        entry = f'{{"input_length": {100 + i * 50}, "hash_ids": [{i}]}}'
                    f.write(f"{entry}\n")
                return f.name

        return _create_file

    def test_effective_request_count_in_timing_config(self, create_mooncake_trace_file):
        """Test that TimingManagerConfig uses effective request count from dataset."""
        filename = create_mooncake_trace_file(3)  # 3 entries

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                # Don't set request_count explicitly - should use dataset size
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            timing_config = TimingManagerConfig.from_user_config(user_config)

            # Should use dataset size (3)
            assert timing_config.request_count == 3

        finally:
            Path(filename).unlink(missing_ok=True)

    def test_timing_mode_selection_with_timestamps(self, create_mooncake_trace_file):
        """Test that timing mode switches to fixed schedule when timestamps present."""
        filename = create_mooncake_trace_file(3, include_timestamps=True)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            timing_config = TimingManagerConfig.from_user_config(user_config)

            # Should auto-detect fixed schedule due to timestamps
            assert timing_config.timing_mode == TimingMode.FIXED_SCHEDULE
            assert timing_config.request_count == 3

        finally:
            Path(filename).unlink(missing_ok=True)

    def test_timing_mode_selection_without_timestamps(self, create_mooncake_trace_file):
        """Test that timing mode uses request rate when no timestamps."""
        filename = create_mooncake_trace_file(3, include_timestamps=False)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            timing_config = TimingManagerConfig.from_user_config(user_config)

            # Should use request rate (default) since no timestamps
            assert timing_config.timing_mode == TimingMode.REQUEST_RATE
            assert timing_config.request_count == 3

        finally:
            Path(filename).unlink(missing_ok=True)

    def test_non_custom_dataset_uses_original_count(self):
        """Test that non-custom datasets use original request count."""
        user_config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            load_generator=LoadGeneratorConfig(request_count=42),
            # No custom dataset configuration
        )

        timing_config = TimingManagerConfig.from_user_config(user_config)

        # Should use original request_count since no custom dataset
        assert timing_config.request_count == 42

    def test_empty_dataset_file_behavior(self, create_mooncake_trace_file):
        """Test behavior with empty dataset file."""
        # Create empty file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filename = f.name

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            timing_config = TimingManagerConfig.from_user_config(user_config)

            # Should use default since empty file
            assert timing_config.request_count == 10

        finally:
            Path(filename).unlink(missing_ok=True)

    def test_mixed_custom_dataset_timing_integration(self, create_mooncake_trace_file):
        """Test end-to-end timing integration with mixed content."""
        filename = create_mooncake_trace_file(2, include_timestamps=True)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                load_generator=LoadGeneratorConfig(
                    request_rate=10  # Should be ignored due to fixed schedule
                ),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            # Test that all the pieces work together
            effective_count = user_config.get_effective_request_count()
            should_use_fixed = (
                user_config._should_use_fixed_schedule_for_mooncake_trace()
            )
            timing_config = TimingManagerConfig.from_user_config(user_config)

            assert effective_count == 2
            assert should_use_fixed is True
            assert timing_config.request_count == 2
            assert timing_config.timing_mode == TimingMode.FIXED_SCHEDULE

        finally:
            Path(filename).unlink(missing_ok=True)
