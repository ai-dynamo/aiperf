# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T3 + Q6 coverage for the scenario-lock CLI flags.

T3: ``--scenario`` / ``--unsafe-override`` parse through cyclopts onto the
flat ``CLIConfig`` and thread to ``BenchmarkConfig`` via the converter.

Q6: the converter writes endpoint fields only when the user explicitly set
them (``cli.model_fields_set & ENDPOINT_FIELDS``), so the resolved
``EndpointConfig.model_fields_set`` faithfully reflects user-explicitness for
``streaming`` / ``cache_bust`` -- the scenario validator can use membership
directly (no explicit-set sentinel needed on this branch's converter path).
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf


def _parse_cli_args(argv: list[str]) -> CLIConfig:
    """Parse ``argv`` through cyclopts into a ``CLIConfig`` (no execution)."""
    from cyclopts import App

    captured: dict[str, CLIConfig] = {}
    app = App(name="test_profile")

    @app.default
    def _runner(*, cli_config: CLIConfig) -> None:  # pragma: no cover - capture only
        captured["uc"] = cli_config

    try:
        app(argv, exit_on_error=False)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    return captured["uc"]


def _required_endpoint_args() -> list[str]:
    return ["--url", "http://localhost:8000/test", "--model", "test-model"]


# ---------------------------------------------------------------------------
# T3: parse + converter pass-through
# ---------------------------------------------------------------------------


def test_cyclopts_parses_scenario_flag() -> None:
    uc = _parse_cli_args(
        [*_required_endpoint_args(), "--scenario", "inferencex-agentx-mvp"]
    )
    assert uc.scenario == "inferencex-agentx-mvp"
    assert "scenario" in uc.model_fields_set


def test_cyclopts_parses_unsafe_override_flag() -> None:
    uc = _parse_cli_args([*_required_endpoint_args(), "--unsafe-override"])
    assert uc.unsafe_override is True
    assert "unsafe_override" in uc.model_fields_set


def test_cyclopts_scenario_defaults_unset() -> None:
    uc = _parse_cli_args(_required_endpoint_args())
    assert uc.scenario is None
    assert uc.unsafe_override is False
    assert "scenario" not in uc.model_fields_set
    assert "unsafe_override" not in uc.model_fields_set


def test_converter_threads_scenario_fields_to_benchmark() -> None:
    uc = _parse_cli_args(
        [
            *_required_endpoint_args(),
            "--scenario",
            "inferencex-agentx-mvp",
            "--unsafe-override",
        ]
    )
    benchmark = convert_cli_to_aiperf(uc).benchmark
    assert benchmark.scenario == "inferencex-agentx-mvp"
    assert benchmark.unsafe_override is True


def test_converter_scenario_fields_default_when_omitted() -> None:
    uc = _parse_cli_args(_required_endpoint_args())
    benchmark = convert_cli_to_aiperf(uc).benchmark
    assert benchmark.scenario is None
    assert benchmark.unsafe_override is False
    assert "scenario" not in benchmark.model_fields_set
    assert "unsafe_override" not in benchmark.model_fields_set


# ---------------------------------------------------------------------------
# Q6: model_fields_set faithfully reflects user-explicitness
# ---------------------------------------------------------------------------


def _benchmark_for_cli(**kwargs):
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
        param({}, "cache_bust", False, id="cache_bust-unset-absent"),
        param(
            {"cache_bust": "first_turn_prefix"},
            "cache_bust",
            True,
            id="cache_bust-set-present",
        ),
    ],
)  # fmt: skip
def test_endpoint_model_fields_set_reflects_explicitness(
    kwargs: dict, field: str, present: bool
) -> None:
    """Q6: ``EndpointConfig.model_fields_set`` carries ``streaming`` /
    ``cache_bust`` only when the user explicitly set the field.

    An explicit ``streaming=False`` is still treated as set, distinguishing it
    from the default -- exactly the auto-fill-vs-validate signal the scenario
    validator needs. The converter writes endpoint keys only for fields in
    ``cli.model_fields_set & ENDPOINT_FIELDS``, so membership is faithful and
    no explicit-set sentinel is required on this converter path.
    """
    endpoint = _benchmark_for_cli(**kwargs).endpoint
    assert (field in endpoint.model_fields_set) is present
