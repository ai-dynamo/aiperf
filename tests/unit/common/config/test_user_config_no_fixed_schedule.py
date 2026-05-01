# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for --no-fixed-schedule, --ignore-trace-delays, --use-think-time-only user config flags."""

from unittest.mock import mock_open, patch

import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    UserConfig,
)
from aiperf.plugin.enums import CustomDatasetType, TimingMode


class TestDisableAutoFixedSchedule:
    """`--no-fixed-schedule` opts trace datasets out of the auto-trigger."""

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_disable_auto_skips_auto_detection_for_trace_with_timestamps(
        self, mock_is_file, mock_exists
    ):
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
        )
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/with_timestamps.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                disable_auto_fixed_schedule=True,
            ),
        )
        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            assert config._should_use_fixed_schedule_for_trace_dataset() is False

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_default_keeps_auto_detection(self, mock_is_file, mock_exists):
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
        )
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/with_timestamps.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )
        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            assert config._should_use_fixed_schedule_for_trace_dataset() is True

    def test_explicit_fixed_schedule_with_disable_auto_raises(self, tmp_path):
        f = tmp_path / "x.jsonl"
        f.write_text('{"input_length": 100, "timestamp": 1000}\n')
        with pytest.raises(ValueError, match="cannot be used together"):
            InputConfig(
                file=str(f),
                fixed_schedule=True,
                disable_auto_fixed_schedule=True,
            )

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_disable_auto_resolves_to_non_fixed_timing_mode(
        self, mock_is_file, mock_exists
    ):
        mock_file_content = (
            '{"input_length": 100, "hash_ids": [1], "timestamp": 1000}\n'
        )
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                file="/fake/path/with_timestamps.jsonl",
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                disable_auto_fixed_schedule=True,
            ),
        )
        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            assert config.timing_mode != TimingMode.FIXED_SCHEDULE


class TestIgnoreTraceDelaysField:
    """`--ignore-trace-delays` is settable on InputConfig and defaults False."""

    def test_default_false(self):
        cfg = InputConfig()
        assert cfg.ignore_trace_delays is False

    def test_can_be_enabled(self):
        cfg = InputConfig(ignore_trace_delays=True)
        assert cfg.ignore_trace_delays is True


class TestUseThinkTimeOnlyField:
    """`--use-think-time-only` is settable on InputConfig and defaults False."""

    def test_default_false(self):
        cfg = InputConfig()
        assert cfg.use_think_time_only is False

    def test_can_be_enabled(self):
        cfg = InputConfig(use_think_time_only=True)
        assert cfg.use_think_time_only is True

    def test_mutex_with_ignore_trace_delays(self):
        with pytest.raises(ValueError, match="cannot be used together"):
            InputConfig(ignore_trace_delays=True, use_think_time_only=True)
