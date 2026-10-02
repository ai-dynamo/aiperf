# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Path-based sweeps must work against the loader's unnormalized envelope."""

import copy
from typing import Any

import pytest
from pytest import param

from aiperf.config.loader import (
    build_benchmark_plan,
    load_config_from_mapping,
    load_config_from_string,
)
from aiperf.config.templates import load_template_content


def _sweep(sweep_type: str) -> dict[str, Any]:
    if sweep_type in {"grid", "zip"}:
        return {
            "type": sweep_type,
            "parameters": {
                "datasets.default.prompts.isl": [128, 512],
                "concurrency": [2, 4],
            },
        }
    return {
        "type": sweep_type,
        "samples": 4,
        "seed": 42,
        "dimensions": [
            {"path": "datasets.default.prompts.isl", "choices": [128, 512]},
            {"path": "concurrency", "choices": [2, 4]},
        ],
    }


@pytest.mark.parametrize("sweep_type", ["grid", "zip", "sobol", "latin_hypercube"])
@pytest.mark.parametrize(
    "phase_fields",
    [
        param({"phases": {"type": "concurrency", "requests": 10, "concurrency": 1}}, id="flat-phase"),
        param({"profiling": {"type": "concurrency", "requests": 10, "concurrency": 1}}, id="profiling-shorthand"),
    ],
)  # fmt: skip
def test_path_sweep_preserves_shorthand_values(
    sweep_type: str, phase_fields: dict[str, Any]
) -> None:
    source = {
        "benchmark": {
            "model": "test-model",
            "endpoint": {"url": "http://localhost:8000"},
            "dataset": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 64, "osl": "{{ dataset.prompts.isl }}"},
            },
            **phase_fields,
        },
        "sweep": _sweep(sweep_type),
    }
    original = copy.deepcopy(source)
    config = load_config_from_mapping(source)
    raw_before = copy.deepcopy(config._raw_envelope)
    plan = build_benchmark_plan(config)

    assert len(plan.configs) == (2 if sweep_type == "zip" else 4)
    for benchmark, variation in zip(plan.configs, plan.variations, strict=True):
        dataset = benchmark.datasets[0]
        assert dataset.name == "default"
        assert dataset.entries == 100
        assert (
            dataset.prompts.isl.expected_value
            == variation.values["datasets.default.prompts.isl"]
        )
        assert (
            dataset.prompts.osl.expected_value
            == variation.values["datasets.default.prompts.isl"]
        )
        assert (
            benchmark.phases[0].concurrency
            == variation.values["phases.profiling.concurrency"]
        )
        assert benchmark.phases[0].requests == 10
    assert source == original
    assert config._raw_envelope == raw_before


@pytest.mark.parametrize("sweep_type", ["grid", "zip", "sobol", "latin_hypercube"])
def test_shorthand_sweep_renders_each_variation(sweep_type: str) -> None:
    sweep = _sweep(sweep_type)
    if sweep_type in {"grid", "zip"}:
        sweep["parameters"]["variables.output_tokens"] = [8, 16]
    else:
        sweep["dimensions"].append(
            {"path": "variables.output_tokens", "choices": [8, 16]}
        )
    config = load_config_from_mapping(
        {
            "variables": {"output_tokens": 4},
            "benchmark": {
                "model": "test-model",
                "endpoint": {"url": "http://localhost:8000"},
                "dataset": {
                    "type": "synthetic",
                    "prompts": {"isl": 64, "osl": "{{ output_tokens }}"},
                },
                "phases": {"type": "concurrency", "requests": 10, "concurrency": 1},
            },
            "sweep": sweep,
        }
    )
    plan = build_benchmark_plan(config)
    for benchmark, variation in zip(plan.configs, plan.variations, strict=True):
        assert (
            benchmark.datasets[0].prompts.osl.expected_value
            == variation.values["variables.output_tokens"]
        )


def test_shorthand_sweep_rejects_wrong_dataset_name() -> None:
    config = load_config_from_mapping(
        {
            "benchmark": {
                "model": "test-model",
                "endpoint": {"url": "http://localhost:8000"},
                "dataset": {
                    "name": "workload",
                    "type": "synthetic",
                    "prompts": {"isl": 64},
                },
                "phases": {"type": "concurrency", "requests": 10},
            },
            "sweep": {
                "type": "grid",
                "parameters": {"datasets.typo.prompts.isl": [128, 512]},
            },
        }
    )
    with pytest.raises(ValueError, match="no entry named 'typo'"):
        build_benchmark_plan(config)


def test_bundled_distribution_sweep_expands_nine_variations() -> None:
    config = load_config_from_string(load_template_content("sweep_distributions"))
    plan = build_benchmark_plan(config)
    assert len(plan.configs) == 9
    assert {
        (benchmark.datasets[0].prompts.isl.expected_value, benchmark.phases[1].rate)
        for benchmark in plan.configs
    } == {(isl, rate) for isl in (128, 512, 2048) for rate in (10.0, 30.0, 50.0)}


@pytest.mark.parametrize(
    "phase_path", ["phases.measure.concurrency", "phases.0.concurrency", "concurrency"]
)
def test_shorthand_sweep_resolves_explicit_names_and_phase_indices(
    phase_path: str,
) -> None:
    config = load_config_from_mapping(
        {
            "benchmark": {
                "model": "test-model",
                "endpoint": {"url": "http://localhost:8000"},
                "dataset": {
                    "name": "workload",
                    "type": "synthetic",
                    "prompts": {"isl": 64},
                },
                "phases": {"name": "measure", "type": "concurrency", "requests": 10},
            },
            "sweep": {
                "type": "zip",
                "parameters": {
                    "datasets.workload.prompts.isl": [128, 512],
                    phase_path: [2, 4],
                },
            },
        }
    )
    plan = build_benchmark_plan(config)
    assert [
        benchmark.datasets[0].prompts.isl.expected_value for benchmark in plan.configs
    ] == [128, 512]
    assert [benchmark.phases[0].concurrency for benchmark in plan.configs] == [2, 4]


@pytest.mark.parametrize(
    "phase_path", ["phases.warmup.concurrency", "phases.0.concurrency"]
)
def test_shorthand_sweep_can_target_warmup(phase_path: str) -> None:
    config = load_config_from_mapping(
        {
            "benchmark": {
                "model": "test-model",
                "endpoint": {"url": "http://localhost:8000"},
                "warmup": {"type": "concurrency", "requests": 5},
                "dataset": {"type": "synthetic", "prompts": {"isl": 64}},
                "profiling": {"type": "concurrency", "requests": 10, "concurrency": 8},
            },
            "sweep": {"type": "grid", "parameters": {phase_path: [1, 2]}},
        }
    )
    plan = build_benchmark_plan(config)
    assert [benchmark.phases[0].concurrency for benchmark in plan.configs] == [1, 2]
    assert [benchmark.phases[1].concurrency for benchmark in plan.configs] == [8, 8]
