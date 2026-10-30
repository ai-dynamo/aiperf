# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit choices take precedence over the AgentX CLI preset."""

import os

import pytest

from aiperf.common.environment import _Environment
from aiperf.common.scenario import get_scenario
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.plugin.enums import EndpointType


def test_agentx_preset_resolves_pinned_values() -> None:
    """The preset exists to make runs comparable, so every value it supplies is pinned."""
    config = resolve_config(
        CLIConfig(
            scenario="agentx",
            model_names=["test-model"],
            public_dataset="semianalysis_cc_traces_weka_062126",
        )
    )
    benchmark = config.benchmark
    phase = benchmark.get_profiling_phases()[0]
    dataset = benchmark.get_default_dataset()

    assert benchmark.endpoint.type == EndpointType.CHAT
    assert benchmark.endpoint.streaming is True
    assert benchmark.endpoint.use_server_token_count is True
    assert benchmark.runtime.stats_interval == 30
    assert benchmark.gpu_telemetry.enabled is False
    assert benchmark.artifacts.slice_duration == 1.0
    assert dataset.random_seed == 42
    assert dataset.entries == 393
    assert dataset.trace_idle_gap_cap_seconds == 300
    assert phase.duration == 3600
    assert phase.trajectory_start_min_ratio == 0.25
    assert phase.trajectory_start_max_ratio == 0.75
    assert phase.failed_request_threshold == 0.10
    assert phase.warmup_requests_per_lane == 10
    assert phase.agentic_warmup_grace_period == 1800


def test_agentx_environment_defaults_apply_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in tuple(os.environ):
        if key.startswith("AIPERF_"):
            monkeypatch.delenv(key)
    settings = _Environment()

    with settings.defaults(get_scenario("agentx").environment_defaults):
        assert settings.DATASET.CONFIGURATION_TIMEOUT == 1800
        assert settings.SERVICE.PROFILE_CONFIGURE_TIMEOUT == 1800
        assert settings.UI.REALTIME_METRICS_ENABLED is True
        assert settings.HTTP.TCP_USER_TIMEOUT == 900000
        assert os.environ["AIPERF_HTTP_TCP_USER_TIMEOUT"] == "900000"


def test_explicit_cli_values_override_soft_defaults() -> None:
    config = resolve_config(
        CLIConfig(
            scenario="agentx",
            model_names=["test-model"],
            public_dataset="semianalysis_cc_traces_weka_062126",
            benchmark_duration=1200,
            use_server_token_count=False,
            gpu_telemetry=["http://localhost:9400/metrics"],
        )
    )
    assert config.benchmark.get_profiling_phases()[0].duration == 1200
    assert config.benchmark.endpoint.use_server_token_count is False
    assert config.benchmark.gpu_telemetry.enabled is True


def test_duration_warmup_does_not_inject_request_warmup() -> None:
    config = resolve_config(
        CLIConfig(
            scenario="agentx",
            model_names=["test-model"],
            public_dataset="semianalysis_cc_traces_weka_062126",
            agentic_cache_warmup_duration=60,
        )
    )
    phase = config.benchmark.get_profiling_phases()[0]
    assert phase.agentic_cache_warmup_duration == 60
    assert phase.warmup_requests_per_lane is None
