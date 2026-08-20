# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Envelope-level CRD keys must survive the spec -> AIPerfConfig conversion.

``random_seed`` and ``variables`` are AIPerfConfig envelope fields --
BenchmarkConfig forbids them -- so the CRD declares them on ``spec`` itself,
not under ``spec.benchmark``. The converter used to read them only from the
benchmark body, so ``spec.randomSeed`` was accepted at admission and then
silently dropped: the run neither used nor recorded a seed. The
operator-vs-bare-pod audit surfaced it as a missing ``run_info.random_seed``
on the operator side only.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.kubernetes.spec_converter import AIPerfJobSpecConverter

_BENCHMARK = {
    "models": {"items": [{"name": "mock"}]},
    "endpoint": {"urls": ["http://mock:8000"]},
    "datasets": [{"name": "main", "type": "synthetic"}],
    "phases": [
        {"name": "profiling", "type": "concurrency", "concurrency": 1, "requests": 2}
    ],
    "tokenizer": {"name": "builtin"},
}


def _convert(spec: dict):
    return AIPerfJobSpecConverter(
        spec=spec, name="job", namespace="ns"
    ).to_aiperf_config()


class TestEnvelopeKeyExtraction:
    @pytest.mark.parametrize(
        "spec_extra, benchmark_extra, expected",
        [
            param({"randomSeed": 1234}, {}, 1234, id="spec-camelCase"),
            param({"random_seed": 99}, {}, 99, id="spec-snake_case"),
            param({}, {"randomSeed": 777}, 777, id="benchmark-camelCase"),
            param({}, {"random_seed": 555}, 555, id="benchmark-snake_case"),
            param({}, {}, None, id="unset"),
        ],
    )  # fmt: skip
    def test_random_seed_read_from_either_level(
        self, spec_extra: dict, benchmark_extra: dict, expected: int | None
    ) -> None:
        spec = {**spec_extra, "benchmark": {**_BENCHMARK, **benchmark_extra}}
        assert _convert(spec).random_seed == expected

    def test_benchmark_level_wins_over_spec_level(self) -> None:
        """The body is the more specific location, so it takes precedence."""
        spec = {"randomSeed": 1, "benchmark": {**_BENCHMARK, "randomSeed": 2}}
        assert _convert(spec).random_seed == 2

    def test_spec_level_variables_are_lifted(self) -> None:
        spec = {"variables": {"a": "b"}, "benchmark": dict(_BENCHMARK)}
        assert _convert(spec).variables == {"a": "b"}

    def test_spec_variables_render_benchmark_fields(self) -> None:
        benchmark = dict(_BENCHMARK)
        benchmark["phases"] = [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": "{{ conc }}",
                "requests": 2,
            }
        ]
        spec = {"variables": {"conc": 256}, "benchmark": benchmark}

        config = _convert(spec)

        assert config.benchmark.phases[0].concurrency == 256

    def test_top_level_multi_run_survives_conversion(self) -> None:
        spec = {
            "multiRun": {"numRuns": 3},
            "benchmark": dict(_BENCHMARK),
        }

        assert _convert(spec).multi_run.num_runs == 3

    def test_calculate_workers_renders_with_spec_variables(self) -> None:
        benchmark = dict(_BENCHMARK)
        benchmark["phases"] = [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": "{{ conc }}",
                "requests": 2,
            }
        ]
        converter = AIPerfJobSpecConverter(
            spec={"variables": {"conc": 250}, "benchmark": benchmark},
            name="job",
            namespace="ns",
        )

        assert converter.calculate_workers() == 3

    def test_conversion_does_not_mutate_the_caller_spec(self) -> None:
        """The spec dict belongs to kopf; popping from it corrupts the CR body."""
        spec = {"randomSeed": 42, "benchmark": dict(_BENCHMARK)}
        _convert(spec)
        assert spec["randomSeed"] == 42, "spec-level key must survive conversion"
