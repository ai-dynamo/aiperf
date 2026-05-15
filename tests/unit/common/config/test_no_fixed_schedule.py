# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for --no-fixed-schedule (disable_auto_fixed_schedule)."""

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    UserConfig,
)
from aiperf.plugin.enums import CustomDatasetType, TimingMode


class TestNoFixedSchedule:
    def test_default_false(self):
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["m"]),
        )
        assert config.input.disable_auto_fixed_schedule is False

    def test_flag_disables_auto_promotion(self, tmp_path):
        # Trace dataset that would otherwise auto-promote to FIXED_SCHEDULE.
        trace_file = tmp_path / "trace.jsonl"
        trace_file.write_text(
            '{"timestamp": 0, "input_length": 10, "output_length": 5, "hash_ids": []}\n'
        )
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["m"]),
            input=InputConfig(
                file=str(trace_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                disable_auto_fixed_schedule=True,
            ),
            loadgen=LoadGeneratorConfig(concurrency=4),
        )
        assert config._timing_mode != TimingMode.FIXED_SCHEDULE
        assert config.input.fixed_schedule is False

    def test_flag_unset_still_auto_promotes(self, tmp_path):
        trace_file = tmp_path / "trace.jsonl"
        trace_file.write_text(
            '{"timestamp": 0, "input_length": 10, "output_length": 5, "hash_ids": []}\n'
        )
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["m"]),
            input=InputConfig(
                file=str(trace_file),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            ),
        )
        assert config._timing_mode == TimingMode.FIXED_SCHEDULE
