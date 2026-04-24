# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.config import AIPerfConfig
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import TimingConfig

_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    datasets={
        "default": {
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
)


class TestTimingConfigurationIntegration:
    def test_explicit_request_count_honored(self) -> None:
        config = AIPerfConfig(
            **_BASE,
            phases={"profiling": {"type": "poisson", "rate": 10.0, "requests": 100}},
        )
        tcfg = TimingConfig.from_config(config)
        assert tcfg.phase_configs[0].total_expected_requests == 100

    def test_fixed_schedule_phase_uses_fixed_schedule_mode(self) -> None:
        config = AIPerfConfig(
            **_BASE,
            phases={"profiling": {"type": "fixed_schedule", "requests": 100}},
        )
        tcfg = TimingConfig.from_config(config)
        assert tcfg.phase_configs[0].timing_mode == TimingMode.FIXED_SCHEDULE

    def test_poisson_phase_uses_request_rate_mode(self) -> None:
        config = AIPerfConfig(
            **_BASE,
            phases={"profiling": {"type": "poisson", "rate": 10.0, "requests": 10}},
        )
        tcfg = TimingConfig.from_config(config)
        assert tcfg.phase_configs[0].timing_mode == TimingMode.REQUEST_RATE
        assert tcfg.phase_configs[0].total_expected_requests == 10

    def test_request_count_from_load_phase(self) -> None:
        config = AIPerfConfig(
            **_BASE,
            phases={
                "profiling": {"type": "concurrency", "concurrency": 4, "requests": 42}
            },
        )
        tcfg = TimingConfig.from_config(config)
        assert tcfg.phase_configs[0].total_expected_requests == 42

    def test_duration_based_phase_has_no_request_count(self) -> None:
        config = AIPerfConfig(
            **_BASE,
            phases={"profiling": {"type": "poisson", "rate": 10.0, "duration": 60}},
        )
        tcfg = TimingConfig.from_config(config)
        assert tcfg.phase_configs[0].total_expected_requests is None
        assert tcfg.phase_configs[0].timing_mode == TimingMode.REQUEST_RATE
