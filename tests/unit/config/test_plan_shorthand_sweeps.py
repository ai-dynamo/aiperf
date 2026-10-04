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


@pytest.mark.parametrize("sweep_type", ["grid", "zip", "sobol"])
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


@pytest.mark.parametrize(
    "prompt_path,dataset_fields,jinja_ref",
    [
        param("isl", {"isl": 64}, "dataset.isl", id="isl-scalar"),
        param("osl", {"osl": 64}, "dataset.osl", id="osl-scalar"),
        param("isl.mean", {"isl": {"mean": 64, "stddev": 8}}, "dataset.isl.mean", id="isl-distribution"),
        param("osl.mean", {"osl": {"mean": 64, "stddev": 8}}, "dataset.osl.mean", id="osl-distribution"),
        param("isl.stddev", {"isl": {"mean": 64, "stddev": 8}, "prompts": {"isl": {"mean": 96}}}, "dataset.isl.stddev", id="inherited-distribution-field"),
        param("isl", {"isl": 64, "prompts": {"isl": 96}}, "dataset.prompts.isl", id="explicit-prompt-precedence"),
        param("isl.mean", {"isl": {"mean": 64, "stddev": 8}, "prompts": {"isl": {"mean": 96}}}, "dataset.prompts.isl.mean", id="explicit-distribution-precedence"),
    ],
)  # fmt: skip
def test_swept_prompt_shorthand_rerenders_its_source_reference(
    prompt_path: str, dataset_fields: dict[str, Any], jinja_ref: str
) -> None:
    """Sweep the effective field while keeping its raw Jinja name and precedence."""
    source = {
        "benchmark": {
            "model": "test-model",
            "endpoint": {"url": "http://localhost:8000"},
            "dataset": {
                "type": "synthetic",
                "entries": "{{ " + jinja_ref + " }}",
                **dataset_fields,
            },
            "phases": {"type": "concurrency", "requests": 10},
        },
        "sweep": {
            "type": "grid",
            "parameters": {"datasets.default.prompts." + prompt_path: [128, 512]},
        },
    }
    original = copy.deepcopy(source)
    config = load_config_from_mapping(source)
    raw_before = copy.deepcopy(config._raw_envelope)
    plan = build_benchmark_plan(config)

    assert [benchmark.datasets[0].entries for benchmark in plan.configs] == [128, 512]
    prompt, _, distribution_field = prompt_path.partition(".")
    distributions = [
        getattr(benchmark.datasets[0].prompts, prompt) for benchmark in plan.configs
    ]
    assert [
        getattr(distribution, distribution_field or "expected_value")
        for distribution in distributions
    ] == [128, 512]
    if distribution_field == "mean":
        assert [distribution.stddev for distribution in distributions] == [8, 8]
    elif distribution_field == "stddev":
        assert [distribution.mean for distribution in distributions] == [96, 96]
    assert source == original
    assert config._raw_envelope == raw_before


def test_named_dataset_prompt_shorthand_rerenders_its_source_reference() -> None:
    config = load_config_from_mapping(
        {
            "benchmark": {
                "model": "test-model",
                "endpoint": {"url": "http://localhost:8000"},
                "datasets": [
                    {
                        "name": "workload",
                        "type": "synthetic",
                        "isl": 64,
                        "entries": "{{ datasets.workload.isl }}",
                    }
                ],
                "phases": {"type": "concurrency", "requests": 10},
            },
            "sweep": {
                "type": "grid",
                "parameters": {"datasets.workload.prompts.isl": [128, 512]},
            },
        }
    )
    plan = build_benchmark_plan(config)
    assert [benchmark.datasets[0].entries for benchmark in plan.configs] == [128, 512]
    assert [
        benchmark.datasets[0].prompts.isl.expected_value for benchmark in plan.configs
    ] == [128, 512]


@pytest.mark.parametrize("prompt", ["isl", "osl"])
@pytest.mark.parametrize("named", [False, True], ids=["singular", "named"])
@pytest.mark.parametrize(
    "distribution_fields",
    [
        param(("mean",), id="grid-mean"),
        param(("mean", "stddev"), id="qmc-mean-first"),
        param(("stddev", "mean"), id="qmc-stddev-first"),
        param(("max", "mean"), id="qmc-max-first"),
    ],
)  # fmt: skip
def test_scalar_prompt_shorthand_can_be_overridden_by_distribution_path(
    prompt: str, named: bool, distribution_fields: tuple[str, ...]
) -> None:
    reference = f"{'datasets.default' if named else 'dataset'}.{prompt}"
    dataset = {
        prompt: 64,
        "entries": f"{{{{ {reference}.mean | default({reference}) }}}}",
    }
    values = {"mean": [128, 512], "stddev": [8, 16], "max": [1024, 2048]}
    parameters = {
        f"datasets.default.prompts.{prompt}.{field}": values[field]
        for field in distribution_fields
    }
    source = {
        "benchmark": {
            "model": "test-model",
            "endpoint": {"url": "http://localhost:8000"},
            **(
                {"datasets": [{"name": "default", **dataset}]}
                if named
                else {"dataset": dataset}
            ),
            "phases": {"type": "concurrency", "requests": 10},
        },
        "sweep": (
            {"type": "grid", "parameters": parameters}
            if len(distribution_fields) == 1
            else {
                "type": "sobol",
                "samples": 4,
                "seed": 42,
                "dimensions": [
                    {"path": path, "choices": choices}
                    for path, choices in parameters.items()
                ],
            }
        ),
    }
    original = copy.deepcopy(source)
    config = load_config_from_mapping(source)
    raw_before = copy.deepcopy(config._raw_envelope)
    plan = build_benchmark_plan(config)
    expected_means = [
        variation.values[f"datasets.default.prompts.{prompt}.mean"]
        for variation in plan.variations
    ]
    assert [
        benchmark.datasets[0].entries for benchmark in plan.configs
    ] == expected_means
    for field in distribution_fields:
        assert [
            getattr(getattr(benchmark.datasets[0].prompts, prompt), field)
            for benchmark in plan.configs
        ] == [
            variation.values[f"datasets.default.prompts.{prompt}.{field}"]
            for variation in plan.variations
        ]
    assert source == original
    assert config._raw_envelope == raw_before


def test_scalar_prompt_mean_sweep_retains_explicit_prompt_precedence() -> None:
    config = load_config_from_mapping(
        {
            "benchmark": {
                "model": "test-model",
                "endpoint": {"url": "http://localhost:8000"},
                "dataset": {
                    "isl": 64,
                    "prompts": {"isl": {"mean": 96}},
                    "entries": "{{ dataset.isl }}",
                },
                "phases": {"type": "concurrency", "requests": 10},
            },
            "sweep": {
                "type": "grid",
                "parameters": {"datasets.default.prompts.isl.mean": [128, 512]},
            },
        }
    )
    plan = build_benchmark_plan(config)
    assert [benchmark.datasets[0].entries for benchmark in plan.configs] == [64, 64]
    assert [benchmark.datasets[0].prompts.isl.mean for benchmark in plan.configs] == [
        128,
        512,
    ]


def test_scalar_prompt_stddev_sweep_does_not_inherit_a_mean() -> None:
    config = load_config_from_mapping(
        {
            "benchmark": {
                "model": "test-model",
                "endpoint": {"url": "http://localhost:8000"},
                "dataset": {"isl": 64},
                "phases": {"type": "concurrency", "requests": 10},
            },
            "sweep": {
                "type": "grid",
                "parameters": {"datasets.default.prompts.isl.stddev": [8, 16]},
            },
        }
    )
    with pytest.raises(ValueError, match=r"prompts.isl.normal.mean\s+Field required"):
        build_benchmark_plan(config)


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
