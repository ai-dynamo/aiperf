# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario presets respect explicit choices during CLI conversion."""

import pytest

from aiperf.common.scenario import SCENARIOS, get_scenario
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config


@pytest.fixture
def preset(monkeypatch: pytest.MonkeyPatch) -> str:
    scenario = get_scenario("inferencex-agentx-mvp").model_copy(
        update={
            "name": "test-preset",
            "cli_defaults": {
                "use_server_token_count": True,
                "no_gpu_telemetry": True,
            },
        }
    )
    monkeypatch.setitem(SCENARIOS, scenario.name, scenario)
    return scenario.name


def test_explicit_cli_values_override_defaults(preset: str) -> None:
    config = resolve_config(
        CLIConfig(
            scenario=preset,
            model_names=["test-model"],
            benchmark_duration=1200,
            use_server_token_count=False,
            gpu_telemetry=["http://localhost:9400/metrics"],
        )
    )
    assert config.benchmark.get_profiling_phases()[0].duration == 1200
    assert config.benchmark.endpoint.use_server_token_count is False
    assert config.benchmark.gpu_telemetry.enabled is True


def test_scenario_duration_prevents_fallback_request_limit(preset: str) -> None:
    config = resolve_config(
        CLIConfig(
            scenario=preset,
            model_names=["test-model"],
        )
    )
    phase = config.benchmark.get_profiling_phases()[0]
    assert phase.duration == get_scenario(preset).default_benchmark_duration_seconds
    assert phase.requests is None
