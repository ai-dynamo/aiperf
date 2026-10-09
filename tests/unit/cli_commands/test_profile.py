# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for profile command wiring."""

import os
from unittest.mock import Mock

import pytest

from aiperf.cli_commands.profile import app
from aiperf.common.environment import Environment
from aiperf.common.scenario import SCENARIOS, get_scenario
from aiperf.config import BenchmarkPlan


def test_profile_passes_scenario_environment_defaults_to_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = get_scenario("inferencex-agentx-mvp").model_copy(
        update={
            "name": "test-preset",
            "environment_defaults": {"HTTP": {"TCP_USER_TIMEOUT": 900000}},
        }
    )
    monkeypatch.setitem(SCENARIOS, scenario.name, scenario)
    monkeypatch.delenv("AIPERF_HTTP_TCP_USER_TIMEOUT", raising=False)
    monkeypatch.setattr(Environment, "HTTP", type(Environment.HTTP)())
    original_http = Environment.HTTP

    def observe_environment(plan: BenchmarkPlan) -> None:
        assert plan.configs[0].scenario == scenario.name
        assert Environment.HTTP.TCP_USER_TIMEOUT == 900000
        assert os.environ["AIPERF_HTTP_TCP_USER_TIMEOUT"] == "900000"

    run_benchmark = Mock(side_effect=observe_environment)
    monkeypatch.setattr("aiperf.cli_runner.run_benchmark", run_benchmark)

    app(
        ["--model", "test-model", "--scenario", scenario.name],
        result_action="return_value",
    )

    run_benchmark.assert_called_once()
    assert Environment.HTTP is original_http
    assert "AIPERF_HTTP_TCP_USER_TIMEOUT" not in os.environ
