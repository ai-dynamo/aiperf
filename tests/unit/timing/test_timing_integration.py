# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import CustomDatasetType, TimingMode
from aiperf.timing.config import TimingConfig
from tests.unit.conftest import make_run_from_cli


class TestTimingConfigurationIntegration:
    def test_explicit_request_count_honored(self, create_mooncake_trace_file):
        fname = create_mooncake_trace_file(3)
        try:
            ucfg = CLIConfig(
                model_names=["test-model"],
                **CLIConfig(request_count=100).model_dump(exclude_unset=True),
                input_file=fname,
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            )
            tcfg = TimingConfig.from_run(make_run_from_cli(ucfg))
            assert tcfg.phase_configs[0].total_expected_requests == 100
        finally:
            Path(fname).unlink(missing_ok=True)

    # ``test_timestamps_triggers_fixed_schedule`` was removed in v2: trace-timestamp
    # auto-detection of FIXED_SCHEDULE happened inside the legacy
    # ``TimingConfig.from_cfg`` path. The v1 -> v2 resolver builds the
    # ``BenchmarkConfig`` (and its phases) before any dataset file is read, so it
    # cannot flip a phase to FIXED_SCHEDULE based on file content. Users now opt
    # in explicitly via ``--fixed-schedule`` (or the ``trace_replay`` YAML
    # template).

    def test_no_timestamps_uses_request_rate(self, create_mooncake_trace_file):
        fname = create_mooncake_trace_file(3, include_timestamps=False)
        try:
            ucfg = CLIConfig(
                model_names=["test-model"],
                input_file=fname,
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            )
            tcfg = TimingConfig.from_run(make_run_from_cli(ucfg))
            assert tcfg.phase_configs[0].timing_mode == TimingMode.REQUEST_RATE
            assert tcfg.phase_configs[0].total_expected_requests == 10
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_non_custom_dataset_uses_original_count(self):
        ucfg = CLIConfig(
            model_names=["test-model"],
            **CLIConfig(request_count=42).model_dump(exclude_unset=True),
        )
        tcfg = TimingConfig.from_run(make_run_from_cli(ucfg))
        assert tcfg.phase_configs[0].total_expected_requests == 42

    def test_empty_dataset_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            fname = f.name
        try:
            ucfg = CLIConfig(
                model_names=["test-model"],
                input_file=fname,
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            )
            tcfg = TimingConfig.from_run(make_run_from_cli(ucfg))
            assert tcfg.phase_configs[0].total_expected_requests == 10
            assert tcfg.phase_configs[0].timing_mode == TimingMode.REQUEST_RATE
        finally:
            Path(fname).unlink(missing_ok=True)
