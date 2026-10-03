# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario presets respect explicit choices during CLI conversion."""

from pathlib import Path

import pytest
from pytest import param

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


@pytest.mark.parametrize(
    ("defaults", "explicit", "section", "enabled"),
    [
        param(
            {"gpu_telemetry": ["http://localhost:9400/metrics"]},
            {"no_gpu_telemetry": True}, "gpu_telemetry", False, id="disable-gpu",
        ),
        param(
            {"server_metrics": ["http://localhost:8000/metrics"]},
            {"no_server_metrics": True}, "server_metrics", False, id="disable-server",
        ),
        param(
            {"no_server_metrics": True},
            {"server_metrics": ["http://localhost:8000/metrics"]},
            "server_metrics", True, id="enable-server",
        ),
    ],
)  # fmt: skip
def test_explicit_telemetry_flags_override_opposite_defaults(
    preset: str,
    defaults: dict[str, object],
    explicit: dict[str, object],
    section: str,
    enabled: bool,
) -> None:
    SCENARIOS[preset] = SCENARIOS[preset].model_copy(update={"cli_defaults": defaults})

    config = resolve_config(
        CLIConfig(scenario=preset, model_names=["test-model"], **explicit)
    )

    assert getattr(config.benchmark, section).enabled is enabled


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


def test_scenario_defaults_parse_model_names(preset: str) -> None:
    SCENARIOS[preset] = SCENARIOS[preset].model_copy(
        update={"cli_defaults": {"model_names": "foo"}}
    )

    config = resolve_config(CLIConfig(scenario=preset))

    assert [model.name for model in config.benchmark.models.items] == ["foo"]


@pytest.mark.parametrize(
    ("defaults", "error", "match"),
    [
        param({"random_sead": 19}, TypeError, "random_sead", id="unknown-field"),
        param({"random_seed": "invalid"}, ValueError, "random_seed", id="invalid-value"),
    ],
)  # fmt: skip
def test_invalid_scenario_defaults_are_rejected(
    preset: str, defaults: dict[str, object], error: type[Exception], match: str
) -> None:
    SCENARIOS[preset] = SCENARIOS[preset].model_copy(update={"cli_defaults": defaults})

    with pytest.raises(error, match=match):
        resolve_config(CLIConfig(scenario=preset, model_names=["test-model"]))


def test_config_file_values_are_not_replaced_by_scenario_defaults(
    preset: str, tmp_path: Path
) -> None:
    config_file = tmp_path / "benchmark.yaml"
    config_file.write_text(
        """\
benchmark:
  models:
    items: [{name: test-model}]
  endpoint:
    urls: [http://localhost:8000]
    useServerTokenCount: false
  gpuTelemetry:
    enabled: true
  datasets:
    - name: workload
      type: synthetic
      prompts:
        isl: {mean: 128}
        osl: {mean: 32}
  phases:
    - name: profiling
      kind: profiling
      type: concurrency
      concurrency: 1
      duration: 1200
""",
        encoding="utf-8",
    )

    config = resolve_config(CLIConfig(scenario=preset, config_file=config_file))

    assert config.benchmark.get_profiling_phases()[0].duration == 1200
    assert config.benchmark.endpoint.use_server_token_count is False
    assert config.benchmark.gpu_telemetry.enabled is True
