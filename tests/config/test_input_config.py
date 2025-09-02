# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from aiperf.common.config import (
    AudioConfig,
    InputConfig,
    InputDefaults,
)
from aiperf.common.enums import CustomDatasetType, PublicDatasetType


def test_input_config_defaults():
    """Test default values and nested config instances."""
    config = InputConfig()
    assert config.extra == InputDefaults.EXTRA
    assert config.file == InputDefaults.FILE
    assert config.custom_dataset_type == InputDefaults.CUSTOM_DATASET_TYPE
    assert isinstance(config.audio, AudioConfig)


def test_input_config_custom_values():
    """Test custom field values with type conversion."""
    config = InputConfig(
        extra={"key": "value"},
        random_seed=42,
        custom_dataset_type=CustomDatasetType.MULTI_TURN,
    )
    assert config.extra == [("key", "value")]  # Dict converted to tuple list
    assert config.random_seed == 42
    assert config.custom_dataset_type == CustomDatasetType.MULTI_TURN


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


class TestFileAndDatasetValidators:
    """Test file and dataset type validation logic."""

    def test_file_sets_default_dataset_type(self, temp_file):
        """File input sets MOONCAKE_TRACE by default."""
        config = InputConfig(file=temp_file)
        assert config.custom_dataset_type == CustomDatasetType.MOONCAKE_TRACE

    def test_file_preserves_explicit_dataset_type(self, temp_file):
        """File input preserves explicit dataset type."""
        config = InputConfig(
            file=temp_file, custom_dataset_type=CustomDatasetType.SINGLE_TURN
        )
        assert config.custom_dataset_type == CustomDatasetType.SINGLE_TURN

    def test_both_dataset_types_raises_error(self):
        """Cannot set both public and custom dataset types."""
        with pytest.raises(ValidationError):
            InputConfig(
                public_dataset=PublicDatasetType.SHAREGPT,
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            )


class TestScheduleValidators:
    """Test fixed schedule and offset validation logic."""

    def test_fixed_schedule_requires_dataset(self):
        """Fixed schedule requires a dataset type."""
        with pytest.raises(ValidationError):
            InputConfig(fixed_schedule=True)

    @pytest.mark.parametrize(
        "custom_dataset_type,public_dataset,expected_schedule",
        [
            (
                CustomDatasetType.MOONCAKE_TRACE,
                None,
                True,
            ),  # Custom dataset type triggers fixed schedule
            (
                None,
                PublicDatasetType.SHAREGPT,
                True,
            ),  # Public dataset triggers fixed schedule
            (
                None,
                None,
                False,
            ),  # No custom dataset type or public dataset, no fixed schedule
        ],
    )
    def test_fixed_schedule_auto_inference(
        self, custom_dataset_type, public_dataset, expected_schedule
    ):
        """Fixed schedule is auto-inferred from custom_dataset_type or public_dataset presence."""
        config = InputConfig(
            custom_dataset_type=custom_dataset_type, public_dataset=public_dataset
        )
        assert config.fixed_schedule == expected_schedule

    @pytest.mark.parametrize(
        "start_offset,end_offset",
        [
            (1000, None),
            (None, 2000),
            (1000, 2000),  # Any offset without fixed_schedule
        ],
    )
    def test_offsets_require_fixed_schedule(self, start_offset, end_offset):
        """Offsets require fixed_schedule to be enabled."""
        with pytest.raises(ValidationError):
            InputConfig(
                fixed_schedule_start_offset=start_offset,
                fixed_schedule_end_offset=end_offset,
            )

    def test_start_offset_conflicts_with_auto_offset(self):
        """Start offset conflicts with auto offset."""
        with pytest.raises(ValidationError):
            InputConfig(
                fixed_schedule=True,
                public_dataset=PublicDatasetType.SHAREGPT,
                fixed_schedule_start_offset=1000,
                fixed_schedule_auto_offset=True,
            )

    def test_start_offset_must_be_lte_end_offset(self):
        """Start offset must be <= end offset."""
        with pytest.raises(ValidationError):
            InputConfig(
                fixed_schedule=True,
                public_dataset=PublicDatasetType.SHAREGPT,
                fixed_schedule_start_offset=2000,
                fixed_schedule_end_offset=1000,
            )


class TestValidatorIntegration:
    """Integration tests showing cascading validator behavior."""

    def test_file_triggers_cascading_validations(self, temp_file):
        """File input triggers: dataset type → fixed schedule → offsets allowed."""
        config = InputConfig(
            file=temp_file,
            fixed_schedule_start_offset=500,
            fixed_schedule_end_offset=1500,
        )
        # File sets dataset type, which enables fixed_schedule, which allows offsets
        assert config.custom_dataset_type == CustomDatasetType.MOONCAKE_TRACE
        assert config.fixed_schedule is True
        assert config.fixed_schedule_start_offset == 500
