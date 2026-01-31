# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    UserConfig,
)
from aiperf.plugin.enums import CustomDatasetType, TimingMode
from aiperf.timing.config import TimingConfig


class TestTimingConfigurationIntegration:
    def test_explicit_request_count_honored(self, create_mooncake_trace_file):
        fname = create_mooncake_trace_file(3)
        try:
            ucfg = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                loadgen=LoadGeneratorConfig(request_count=100),
                input=InputConfig(
                    file=fname, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )
            tcfg = TimingConfig.from_user_config(ucfg)
            assert tcfg.phase_configs[0].total_expected_requests == 100
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_timestamps_triggers_fixed_schedule(self, create_mooncake_trace_file):
        fname = create_mooncake_trace_file(3, include_timestamps=True)
        try:
            ucfg = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=fname, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )
            tcfg = TimingConfig.from_user_config(ucfg)
            assert tcfg.phase_configs[0].timing_mode == TimingMode.FIXED_SCHEDULE
            assert tcfg.phase_configs[0].total_expected_requests == 3
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_no_timestamps_uses_request_rate(self, create_mooncake_trace_file):
        fname = create_mooncake_trace_file(3, include_timestamps=False)
        try:
            ucfg = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=fname, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )
            tcfg = TimingConfig.from_user_config(ucfg)
            assert tcfg.phase_configs[0].timing_mode == TimingMode.REQUEST_RATE
            assert tcfg.phase_configs[0].total_expected_requests == 10
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_non_custom_dataset_uses_original_count(self):
        ucfg = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            loadgen=LoadGeneratorConfig(request_count=42),
        )
        tcfg = TimingConfig.from_user_config(ucfg)
        assert tcfg.phase_configs[0].total_expected_requests == 42

    def test_empty_dataset_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            fname = f.name
        try:
            ucfg = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=fname, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )
            tcfg = TimingConfig.from_user_config(ucfg)
            assert tcfg.phase_configs[0].total_expected_requests == 10
            assert tcfg.phase_configs[0].timing_mode == TimingMode.REQUEST_RATE
        finally:
            Path(fname).unlink(missing_ok=True)
