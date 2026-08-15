# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--scenario`` / ``--unsafe-override`` parse onto the CLI DTO, thread to BenchmarkConfig, and keep ``model_fields_set`` faithful to user-explicitness."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pytest import param

from aiperf.config.config import BenchmarkConfig
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf

_SCENARIO = "inferencex-agentx-mvp"


def test_cyclopts_parses_scenario_flag(
    parse_cli_args: Callable[[list[str]], CLIConfig], endpoint_cli_args: list[str]
) -> None:
    """``--scenario NAME`` lands on the flat CLI DTO as an explicit field."""
    uc = parse_cli_args([*endpoint_cli_args, "--scenario", _SCENARIO])
    assert uc.scenario == _SCENARIO
    assert "scenario" in uc.model_fields_set


def test_cyclopts_parses_unsafe_override_flag(
    parse_cli_args: Callable[[list[str]], CLIConfig], endpoint_cli_args: list[str]
) -> None:
    """``--unsafe-override`` lands as an explicit True on the CLI DTO."""
    uc = parse_cli_args([*endpoint_cli_args, "--unsafe-override"])
    assert uc.unsafe_override is True
    assert "unsafe_override" in uc.model_fields_set


def test_cyclopts_scenario_defaults_unset(
    parse_cli_args: Callable[[list[str]], CLIConfig], endpoint_cli_args: list[str]
) -> None:
    """With neither flag passed both stay at their defaults and out of ``model_fields_set``."""
    uc = parse_cli_args(endpoint_cli_args)
    assert uc.scenario is None
    assert uc.unsafe_override is False
    assert "scenario" not in uc.model_fields_set
    assert "unsafe_override" not in uc.model_fields_set


def test_converter_threads_scenario_fields_to_benchmark(
    parse_cli_args: Callable[[list[str]], CLIConfig], endpoint_cli_args: list[str]
) -> None:
    """The converter threads both scenario-lock fields onto ``BenchmarkConfig``."""
    uc = parse_cli_args(
        [*endpoint_cli_args, "--scenario", _SCENARIO, "--unsafe-override"]
    )
    benchmark = convert_cli_to_aiperf(uc).benchmark
    assert benchmark.scenario == _SCENARIO
    assert benchmark.unsafe_override is True


def test_converter_scenario_fields_default_when_omitted(
    parse_cli_args: Callable[[list[str]], CLIConfig], endpoint_cli_args: list[str]
) -> None:
    """Omitted scenario flags reach ``BenchmarkConfig`` as unset defaults."""
    benchmark = convert_cli_to_aiperf(parse_cli_args(endpoint_cli_args)).benchmark
    assert benchmark.scenario is None
    assert benchmark.unsafe_override is False
    assert "scenario" not in benchmark.model_fields_set
    assert "unsafe_override" not in benchmark.model_fields_set


def _benchmark_for_cli(**kwargs: object) -> BenchmarkConfig:
    """Resolved BenchmarkConfig for a minimal CLIConfig plus the given explicit fields."""
    cli = CLIConfig(
        model_names=["test-model"],
        urls=["http://localhost:8000/test"],
        **kwargs,
    )
    return convert_cli_to_aiperf(cli).benchmark


@pytest.mark.parametrize(
    ("kwargs", "field", "present"),
    [
        param({}, "streaming", False, id="streaming-unset-absent"),
        param({"streaming": True}, "streaming", True, id="streaming-set-present"),
        param(
            {"streaming": False},
            "streaming",
            True,
            id="streaming-explicit-false-present",
        ),
    ],
)  # fmt: skip
def test_endpoint_model_fields_set_reflects_explicitness(
    kwargs: dict, field: str, present: bool
) -> None:
    """``EndpointConfig.model_fields_set`` carries ``streaming`` only when the user set it, so the scenario validator can test membership directly."""
    endpoint = _benchmark_for_cli(**kwargs).endpoint
    assert (field in endpoint.model_fields_set) is present
