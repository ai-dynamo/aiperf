# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep-family CLI flags combined with ``--config``.

The CLI-only path (``convert_cli_to_aiperf``) is the reference behavior; each
test pins one way the ``--config`` path used to diverge from it, or one
companion a flag cannot mean anything without.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from aiperf.config import AIPerfConfig
from aiperf.config.flags import CLIConfig
from aiperf.config.flags._config_flag_routing import (
    RECIPE_INPUT_FIELDS,
    flag_names_for,
)
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader.errors import ConfigurationError

_PLAIN_YAML = textwrap.dedent("""\
    schemaVersion: "2.0"
    benchmark:
      model: test-model
      endpoint:
        url: http://localhost:8000
        streaming: true
      datasets:
        - name: main
          type: synthetic
          entries: 16
      phases:
        type: concurrency
        concurrency: 1
        requests: 5
""")

_SWEEP_BLOCK = textwrap.dedent("""\
    sweep:
      type: grid
      parameters:
        phases.profiling.concurrency: [1, 2]
""")


def _resolve(
    tmp_path: Path, *, yaml_text: str = _PLAIN_YAML, **flags: Any
) -> AIPerfConfig:
    path = tmp_path / "base.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return resolve_config(CLIConfig(**flags), path)


def _sweep(config: AIPerfConfig) -> dict[str, Any]:
    return config.model_dump(mode="json", exclude_none=True)["sweep"]


# --- recipe output ----------------------------------------------------------


def test_grid_recipe_keeps_recipe_name(tmp_path: Path) -> None:
    sweep = _sweep(_resolve(tmp_path, search_recipe="concurrency-ramp"))
    assert sweep["type"] == "grid"
    assert sweep["recipe_name"] == "concurrency-ramp"


def test_scenario_recipe_emits_its_scenarios(tmp_path: Path) -> None:
    config = _resolve(
        tmp_path,
        search_recipe="pareto-sweep",
        isl_osl_pairs="128/128,256/256",
        concurrency=[1, 4],
    )
    sweep = _sweep(config)
    assert sweep["type"] == "scenarios"
    assert sweep["recipe_name"] == "pareto-sweep"
    assert "parameters" not in sweep
    assert [run["name"] for run in sweep["runs"]] == [
        "shape_128_128_c1",
        "shape_128_128_c4",
        "shape_256_256_c1",
        "shape_256_256_c4",
    ]
    assert config._raw_envelope["sweep"]["type"] == "scenarios"


def test_grid_recipe_plus_magic_list_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="mutually exclusive with magic-list flags"):
        _resolve(tmp_path, search_recipe="concurrency-ramp", concurrency=[1, 2])


# --- companions -------------------------------------------------------------

# One valid value per recipe input. The completeness test below makes a new
# recipe input fail here until it is given one.
_RECIPE_INPUT_VALUES: dict[str, Any] = {
    "concurrency_max": 64,
    "concurrency_min": 2,
    "concurrency_steps": 4,
    "degradation_metric_tag": "time_to_first_token",
    "degradation_stat": "p90",
    "degradation_threshold": 0.3,
    "e2e_sla_ms": 1000.0,
    "error_rate_sla": 0.05,
    "isl_max": 4096,
    "isl_min": 64,
    "isl_osl_pairs": "128/128",
    "isl_steps": 3,
    "itl_sla_ms": 5.0,
    "osl_max": 512,
    "osl_min": 16,
    "osl_steps": 3,
    "search_style": "monotonic",
    "slo_attainment_fraction": 0.9,
    "tpot_sla_ms": 5.0,
    "ttft_sla_ms": 100.0,
}

_CONVERGENCE_BLOCK = textwrap.dedent("""\
    multiRun:
      numRuns: 5
      convergence:
        metric: time_to_first_token
        stat: avg
""")


def test_recipe_input_value_table_is_complete() -> None:
    assert set(_RECIPE_INPUT_VALUES) == RECIPE_INPUT_FIELDS


@pytest.mark.parametrize("field", sorted(_RECIPE_INPUT_VALUES))
def test_recipe_input_without_recipe_is_rejected(tmp_path: Path, field: str) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        _resolve(tmp_path, **{field: _RECIPE_INPUT_VALUES[field]})
    message = str(excinfo.value)
    assert flag_names_for(field)[0] in message
    assert "--search-recipe" in message


def test_recipe_input_with_recipe_takes_effect(tmp_path: Path) -> None:
    sweep = _sweep(
        _resolve(
            tmp_path,
            search_recipe="concurrency-ramp",
            concurrency_min=2,
            concurrency_max=64,
            concurrency_steps=4,
        )
    )
    assert sweep["parameters"]["phases.profiling.concurrency"] == [2, 6, 20, 64]


def test_every_missing_companion_is_reported_at_once(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        _resolve(tmp_path, concurrency_min=2, convergence_stat="p90")
    message = str(excinfo.value)
    assert "--concurrency-min" in message
    assert "--convergence-stat" in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("convergence_mode", "ci_width"),
        ("convergence_stat", "p90"),
        ("convergence_threshold", 0.05),
    ],
)
def test_convergence_detail_without_metric_is_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        _resolve(tmp_path, **{field: value})
    message = str(excinfo.value)
    assert flag_names_for(field)[0] in message
    assert "--convergence-metric" in message


def test_convergence_detail_with_metric_takes_effect(tmp_path: Path) -> None:
    config = _resolve(
        tmp_path,
        convergence_metric="time_to_first_token",
        num_profile_runs=5,
        convergence_stat="p90",
    )
    assert config.model_dump(mode="json")["multi_run"]["convergence"]["stat"] == "p90"


def test_convergence_details_override_yaml_block(tmp_path: Path) -> None:
    config = _resolve(
        tmp_path,
        yaml_text=_PLAIN_YAML + _CONVERGENCE_BLOCK,
        convergence_stat="p90",
        convergence_threshold=0.05,
    )
    convergence = config.model_dump(mode="json")["multi_run"]["convergence"]
    assert convergence["metric"] == "time_to_first_token"
    assert convergence["stat"] == "p90"
    assert convergence["threshold"] == 0.05
    raw = config._raw_envelope
    raw_multi_run = raw.get("multi_run") or raw["multiRun"]
    assert raw_multi_run["convergence"]["stat"] == "p90"


@pytest.mark.parametrize("sweep_type", ["zip", "grid"])
def test_sweep_type_without_lists_is_rejected(tmp_path: Path, sweep_type: str) -> None:
    """``grid`` is the default; passing it explicitly still counts as set."""
    with pytest.raises(ConfigurationError, match="--sweep-type"):
        _resolve(tmp_path, sweep_type=sweep_type)


def test_sweep_type_against_yaml_sweep_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="sweep.type"):
        _resolve(
            tmp_path,
            yaml_text=_PLAIN_YAML + _SWEEP_BLOCK,
            sweep_type="zip",
            concurrency=[1, 2],
        )


def test_sweep_type_zips_cli_lists(tmp_path: Path) -> None:
    sweep = _sweep(
        _resolve(
            tmp_path,
            concurrency=[1, 2],
            prompt_input_tokens_mean=[64, 128],
            sweep_type="zip",
        )
    )
    assert sweep["type"] == "zip"


def test_recipe_against_yaml_sweep_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="--search-recipe"):
        _resolve(
            tmp_path,
            yaml_text=_PLAIN_YAML + _SWEEP_BLOCK,
            search_recipe="concurrency-ramp",
        )
