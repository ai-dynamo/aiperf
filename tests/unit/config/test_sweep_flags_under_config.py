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
from pytest import param

from aiperf.config import AIPerfConfig
from aiperf.config.flags import CLIConfig
from aiperf.config.flags._config_flag_routing import (
    RECIPE_INPUT_FIELDS,
    flag_names_for,
)
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader import build_benchmark_plan
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


# --- parameter-sweep knobs --------------------------------------------------

_CAMEL_SWEEP_BLOCK = textwrap.dedent("""\
    sweep:
      type: grid
      iterationOrder: repeated
      cooldownSeconds: 1.0
      parameters:
        phases.profiling.concurrency: [1, 2]
""")

_SEARCH_SPACE_FLAGS: dict[str, Any] = {
    "search_space": ["phases.profiling.concurrency:1,1000:int"],
    "search_metric": "output_token_throughput",
    "search_direction": "maximize",
    "search_max_iterations": 10,
}


def test_parameter_sweep_flags_apply_to_promoted_lists(tmp_path: Path) -> None:
    sweep = _sweep(
        _resolve(
            tmp_path,
            concurrency=[1, 2],
            parameter_sweep_mode="independent",
            parameter_sweep_same_seed=True,
            parameter_sweep_cooldown_seconds=3.0,
        )
    )
    assert sweep["iteration_order"] == "independent"
    assert sweep["same_seed"] is True
    assert sweep["cooldown_seconds"] == 3.0


def test_parameter_sweep_flags_override_yaml_sweep(tmp_path: Path) -> None:
    config = _resolve(
        tmp_path,
        yaml_text=_PLAIN_YAML + _CAMEL_SWEEP_BLOCK,
        parameter_sweep_mode="independent",
        parameter_sweep_cooldown_seconds=3.0,
    )
    sweep = _sweep(config)
    assert sweep["iteration_order"] == "independent"
    assert sweep["cooldown_seconds"] == 3.0
    raw_sweep = config._raw_envelope["sweep"]
    assert raw_sweep["iteration_order"] == "independent"
    assert "iterationOrder" not in raw_sweep
    assert "cooldownSeconds" not in raw_sweep


@pytest.mark.parametrize(
    "flags",
    [
        param({"parameter_sweep_mode": "independent"}, id="mode"),
        param({"parameter_sweep_mode": "repeated"}, id="mode-explicit-default"),
        param({"parameter_sweep_cooldown_seconds": 3.0}, id="cooldown"),
    ],
)
def test_parameter_sweep_flag_without_a_sweep_is_rejected(
    tmp_path: Path, flags: dict[str, Any]
) -> None:
    with pytest.raises(ConfigurationError, match="declares one"):
        _resolve(tmp_path, **flags)


def test_ordering_flag_on_adaptive_recipe_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        _resolve(
            tmp_path,
            search_recipe="max-throughput-ttft-sla",
            ttft_sla_ms=100.0,
            parameter_sweep_mode="independent",
        )
    message = str(excinfo.value)
    assert "--parameter-sweep-mode" in message
    assert "adaptive_search" in message


def test_ordering_flag_on_search_space_sweep_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="--parameter-sweep-same-seed"):
        _resolve(tmp_path, **_SEARCH_SPACE_FLAGS, parameter_sweep_same_seed=True)


def test_cooldown_applies_to_adaptive_sweep(tmp_path: Path) -> None:
    sweep = _sweep(
        _resolve(
            tmp_path,
            search_recipe="max-throughput-ttft-sla",
            ttft_sla_ms=100.0,
            parameter_sweep_cooldown_seconds=3.0,
        )
    )
    assert sweep["type"] == "adaptive_search"
    assert sweep["cooldown_seconds"] == 3.0


def test_cli_only_ordering_flags_apply_to_zip_sweep() -> None:
    """The CLI-only helper used to stamp ordering only on grid/scenarios."""
    config = convert_cli_to_aiperf(
        CLIConfig(
            model_names=["test-model"],
            urls=["http://localhost:8000"],
            request_count=5,
            concurrency=[1, 2],
            prompt_input_tokens_mean=[64, 128],
            sweep_type="zip",
            parameter_sweep_mode="independent",
        )
    )
    assert config.model_dump(mode="json")["sweep"]["iteration_order"] == "independent"


# --- variants ---------------------------------------------------------------

_VARIANTS = ["low: concurrency=2", "high: concurrency=8"]


def test_variants_build_a_scenarios_sweep_over_the_yaml(tmp_path: Path) -> None:
    config = _resolve(tmp_path, sweep_variants=_VARIANTS)
    sweep = _sweep(config)
    assert sweep["type"] == "scenarios"
    assert [run["name"] for run in sweep["runs"]] == ["low", "high"]

    plan = build_benchmark_plan(config)
    concurrencies = [
        next(p for p in bench.phases if p.name == "profiling").concurrency
        for bench in plan.configs
    ]
    assert concurrencies == [2, 8]
    assert all(bench.datasets[0].entries == 16 for bench in plan.configs)

    raw_runs = config._raw_envelope["sweep"]["runs"]
    assert [run["name"] for run in raw_runs] == ["low", "high"]


def test_parameter_sweep_flags_apply_to_variant_runs(tmp_path: Path) -> None:
    sweep = _sweep(
        _resolve(tmp_path, sweep_variants=_VARIANTS, parameter_sweep_mode="independent")
    )
    assert sweep["iteration_order"] == "independent"


def test_variants_against_yaml_sweep_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="--variant"):
        _resolve(
            tmp_path, yaml_text=_PLAIN_YAML + _SWEEP_BLOCK, sweep_variants=_VARIANTS
        )


def test_variants_against_search_space_sweep_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="--variant"):
        _resolve(tmp_path, **_SEARCH_SPACE_FLAGS, sweep_variants=_VARIANTS)


def test_single_variant_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="single occurrence is rejected"):
        _resolve(tmp_path, sweep_variants=["concurrency=2"])


def test_variant_with_recipe_input_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        _resolve(tmp_path, sweep_variants=["a: concurrency-min=4", "b: concurrency=2"])
    message = str(excinfo.value)
    assert "--concurrency-min" in message
    assert "--search-recipe" in message


def test_variant_with_unrouted_flag_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="--no-fixed-schedule"):
        _resolve(
            tmp_path,
            sweep_variants=["a: no-fixed-schedule=true", "b: concurrency=2"],
        )


def test_variant_changing_a_run_level_setting_is_rejected(tmp_path: Path) -> None:
    """Runs carry only benchmark overlays; a multi_run change would vanish."""
    with pytest.raises(ConfigurationError, match="multi_run"):
        _resolve(tmp_path, sweep_variants=["a: num_profile_runs=3", "b: concurrency=2"])


def test_variant_setting_parameter_sweep_flag_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="--parameter-sweep-mode"):
        _resolve(
            tmp_path,
            sweep_variants=["a: parameter-sweep-mode=independent", "b: concurrency=2"],
        )


def test_variant_setting_convergence_detail_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="--convergence-stat"):
        _resolve(
            tmp_path,
            yaml_text=_PLAIN_YAML + _CONVERGENCE_BLOCK,
            sweep_variants=["a: convergence-stat=p90", "b: concurrency=2"],
        )


def test_variant_repeating_outer_parameter_sweep_value_is_allowed(
    tmp_path: Path,
) -> None:
    sweep = _sweep(
        _resolve(
            tmp_path,
            sweep_variants=_VARIANTS,
            parameter_sweep_cooldown_seconds=3.0,
        )
    )
    assert sweep["cooldown_seconds"] == 3.0


# --- parity with the CLI-only path -----------------------------------------

_CLI_BASE: dict[str, Any] = {
    "model_names": ["test-model"],
    "urls": ["http://localhost:8000"],
    "streaming": True,
    "request_count": 5,
}


def _sweep_sections(config: AIPerfConfig) -> dict[str, Any]:
    dump = config.model_dump(mode="json", exclude_none=True)
    return {
        "sweep": dump.get("sweep"),
        "multi_run": dump.get("multi_run"),
        "slos": dump["benchmark"].get("slos"),
    }


@pytest.mark.parametrize(
    "flags",
    [
        param(
            {
                "search_recipe": "concurrency-ramp",
                "concurrency_min": 2,
                "concurrency_max": 64,
                "concurrency_steps": 4,
                "degradation_metric_tag": "time_to_first_token",
                "degradation_stat": "p90",
            },
            id="concurrency-ramp",
        ),
        param(
            {"search_recipe": "prefill-ttft-curve", "isl_min": 64, "isl_max": 1024, "isl_steps": 3},
            id="prefill-ttft-curve",
        ),
        param(
            {"search_recipe": "decode-itl-curve", "osl_min": 16, "osl_max": 256, "osl_steps": 3},
            id="decode-itl-curve",
        ),
        param({"search_recipe": "max-throughput-ttft-sla", "ttft_sla_ms": 123.0}, id="ttft-sla"),
        param({"search_recipe": "max-throughput-itl-sla", "itl_sla_ms": 7.0}, id="itl-sla"),
        param(
            {
                "search_recipe": "max-concurrency-under-sla",
                "ttft_sla_ms": 100.0,
                "e2e_sla_ms": 999.0,
                "error_rate_sla": 0.05,
                "search_style": "monotonic",
            },
            id="max-concurrency-under-sla",
        ),
        param(
            {
                "search_recipe": "max-goodput-under-slo",
                "ttft_sla_ms": 100.0,
                "tpot_sla_ms": 10.0,
                "e2e_sla_ms": 1000.0,
                "slo_attainment_fraction": 0.9,
            },
            id="max-goodput-under-slo",
        ),
        param(
            {"search_recipe": "pareto-sweep", "isl_osl_pairs": "128/128,256/256", "concurrency": [1, 4]},
            id="pareto-sweep",
        ),
        param(
            {
                "convergence_metric": "time_to_first_token",
                "num_profile_runs": 5,
                "convergence_mode": "ci_width",
                "convergence_stat": "p90",
                "convergence_threshold": 0.05,
            },
            id="convergence",
        ),
        param(
            {
                "concurrency": [1, 2],
                "prompt_input_tokens_mean": [64, 128],
                "sweep_type": "zip",
                "parameter_sweep_mode": "independent",
                "parameter_sweep_same_seed": True,
                "parameter_sweep_cooldown_seconds": 3.0,
            },
            id="zip-with-parameter-sweep-knobs",
        ),
        param(
            {"sweep_variants": ["low: concurrency=2", "high: concurrency=8"], "parameter_sweep_cooldown_seconds": 3.0},
            id="variants",
        ),
    ],
)  # fmt: skip
def test_config_path_matches_cli_only_path(
    tmp_path: Path, flags: dict[str, Any]
) -> None:
    cli_only = convert_cli_to_aiperf(CLIConfig(**_CLI_BASE, **flags))
    with_config = _resolve(tmp_path, **_CLI_BASE, **flags)
    assert _sweep_sections(with_config) == _sweep_sections(cli_only)
