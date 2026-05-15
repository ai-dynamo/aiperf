# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-2 adversarial regressions for the sweep + config system.

Each class is named by hypothesis ID (H9-H14, continuing from
``test_sweep_qmc_adversarial.py``). Each test pins a behavior that
silently misbehaved or crashed unhelpfully before the round-2 fixes:

- H9: QMC sweeps now body-root paths (no more phantom top-level keys
  that ``build_benchmark_plan`` silently discards).
- H10: Path validators reject non-sweepable top-level prefixes
  (multi_run, random_seed, benchmark) and the redundant ``benchmark.``
  prefix.
- H11: SamplingDimension rejects unhashable choices entries.
- H12: SearchSpaceDimension applies the same path + finite-bounds
  validation as SamplingDimension.
- H13: ScenarioSweep rejects duplicate ``name`` across runs (which
  would otherwise collide cell directories).
- H14: GridSweep rejects empty parameter value lists at expansion.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from aiperf.config.sweep import (
    GridSweep,
    SamplingDimension,
    ScenarioSweep,
    expand_sweep,
)
from aiperf.config.sweep.adaptive import SearchSpaceDimension

# -- H9: QMC body-rooting ---------------------------------------------------


class TestH9QmcBodyRooting:
    """Sobol/LHS used to write paths against the envelope dict, producing
    a phantom top-level ``phases:`` that ``build_benchmark_plan`` silently
    discarded -- so every variant ran with the base BenchmarkConfig.
    """

    @pytest.fixture
    def base(self) -> dict:
        return {
            "endpoint": {"model_names": ["m"], "url": "http://x"},
            "benchmark": {
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 8,
                    }
                ]
            },
        }

    def test_sobol_writes_into_benchmark_body(self, base: dict) -> None:
        base["sweep"] = {
            "type": "sobol",
            "samples": 4,
            "seed": 42,
            "dimensions": [
                {
                    "path": "phases.profiling.concurrency",
                    "lo": 1,
                    "hi": 64,
                    "kind": "int",
                }
            ],
        }
        variants = expand_sweep(base)
        assert len(variants) == 4
        for variant, var in variants:
            sampled = var.values["phases.profiling.concurrency"]
            actual = variant["benchmark"]["phases"][0]["concurrency"]
            assert actual == sampled, (
                f"variant {var.label}: variant.benchmark.phases[0].concurrency "
                f"= {actual!r}, expected {sampled!r}"
            )
            # And the envelope must NOT carry a phantom top-level `phases`
            assert "phases" not in variant, (
                f"variant {var.label} has a phantom top-level phases dict"
            )

    def test_lhs_writes_into_benchmark_body(self, base: dict) -> None:
        base["sweep"] = {
            "type": "latin_hypercube",
            "samples": 4,
            "seed": 7,
            "dimensions": [
                {
                    "path": "phases.profiling.concurrency",
                    "lo": 1,
                    "hi": 64,
                    "kind": "int",
                }
            ],
        }
        variants = expand_sweep(base)
        assert len(variants) == 4
        for variant, var in variants:
            sampled = var.values["phases.profiling.concurrency"]
            actual = variant["benchmark"]["phases"][0]["concurrency"]
            assert actual == sampled
            assert "phases" not in variant

    def test_sobol_envelope_variables_paired_sweep(self, base: dict) -> None:
        # variables.* is the documented envelope-level escape (paired
        # sweep of a Jinja var that templates into multiple body fields).
        # It must still write at the envelope, not the benchmark body.
        base["variables"] = {"foo": 1}
        base["sweep"] = {
            "type": "sobol",
            "samples": 4,
            "seed": 42,
            "dimensions": [
                {"path": "variables.foo", "lo": 1, "hi": 100, "kind": "int"}
            ],
        }
        variants = expand_sweep(base)
        assert len(variants) == 4
        for variant, var in variants:
            sampled = var.values["variables.foo"]
            assert variant["variables"]["foo"] == sampled

    def test_e2e_build_benchmark_plan_applies_sampled_concurrency(self) -> None:
        # End-to-end: confirm the bug repro from the round-2 report passes.
        # Before the fix, all 8 variants ended up with concurrency=8.
        from aiperf.config.config import AIPerfConfig
        from aiperf.config.loader.plan import build_benchmark_plan

        cfg = AIPerfConfig.model_validate(
            {
                "random_seed": 42,
                "sweep": {
                    "type": "sobol",
                    "samples": 8,
                    "seed": 42,
                    "iteration_order": "independent",
                    "dimensions": [
                        {
                            "path": "phases.profiling.concurrency",
                            "lo": 1,
                            "hi": 32,
                            "scale": "log",
                            "kind": "int",
                        }
                    ],
                },
                "benchmark": {
                    "models": ["test-model"],
                    "endpoint": {
                        "urls": ["http://localhost:8000"],
                        "type": "chat",
                    },
                    "datasets": [
                        {
                            "name": "profiling",
                            "type": "synthetic",
                            "entries": 20,
                            "prompts": {"isl": 128, "osl": 32},
                        }
                    ],
                    "phases": [
                        {
                            "name": "profiling",
                            "type": "concurrency",
                            "concurrency": 8,
                            "requests": 10,
                        }
                    ],
                },
                "multi_run": {"num_runs": 1},
            }
        )
        plan = build_benchmark_plan(cfg)
        assert len(plan.configs) == 8
        for c, v in zip(plan.configs, plan.variations, strict=True):
            sampled = v.values["phases.profiling.concurrency"]
            actual = next(p for p in c.phases if p.name == "profiling").concurrency
            assert actual == sampled, (
                f"variation {v.label}: BenchmarkConfig.phases[0].concurrency "
                f"= {actual!r}, expected {sampled!r}"
            )


# -- H10: extended path validation -----------------------------------------


class TestH10NonSweepableTopLevels:
    """SamplingDimension and SearchSpaceDimension reject paths that target
    envelope-level fields (multi_run, random_seed) or use the redundant
    ``benchmark.`` prefix.
    """

    @pytest.mark.parametrize(
        "bad_path",
        [
            "multi_run.num_runs",
            "random_seed",
            "benchmark.phases.profiling.concurrency",
        ],
    )
    def test_sampling_dimension_rejects(self, bad_path: str) -> None:
        with pytest.raises(ValidationError):
            SamplingDimension(path=bad_path, lo=1, hi=10)

    @pytest.mark.parametrize(
        "bad_path",
        [
            "multi_run.num_runs",
            "random_seed",
            "benchmark.phases.profiling.concurrency",
            "sweep.samples",
            "",
            ".rate",
            "rate.",
            "phases..rate",
        ],
    )
    def test_search_space_dimension_rejects(self, bad_path: str) -> None:
        with pytest.raises(ValidationError):
            SearchSpaceDimension(path=bad_path, lo=1, hi=10)

    def test_benchmark_substring_in_segment_allowed(self) -> None:
        # First segment must be exactly "benchmark" to be rejected;
        # "benchmarks" is fine.
        dim = SamplingDimension(path="benchmarks.x", lo=1, hi=10)
        assert dim.path == "benchmarks.x"


# -- H11: hashable choices --------------------------------------------------


class TestH11ChoicesHashable:
    """SamplingDimension(choices=[unhashable]) used to validate fine and then
    poison ``SweepVariation.values`` with un-hashable entries.
    """

    @pytest.mark.parametrize(
        "bad_choices",
        [
            [[1, 2]],
            [{"a": 1}],
            [{"k"}],
            [1, [2, 3]],
        ],
    )
    def test_unhashable_rejected(self, bad_choices: list) -> None:
        with pytest.raises(ValidationError, match="hashable"):
            SamplingDimension(path="x", choices=bad_choices)

    def test_mixed_hashable_scalars_allowed(self) -> None:
        # Mixed types are fine as long as each entry is hashable. Tuples
        # are hashable iff their contents are; we don't recurse, so a
        # tuple of (int, str, None) goes through.
        dim = SamplingDimension(path="x", choices=[1, "two", None, 3.14, (1, "x")])
        assert dim.choices == [1, "two", None, 3.14, (1, "x")]


# -- H12: SearchSpaceDimension validator parity -----------------------------


class TestH12SearchSpaceDimensionFiniteBounds:
    """SearchSpaceDimension used to accept NaN/+inf for lo/hi -- the BO
    planner would then sample non-finite suggestions and either crash or
    produce nonsense.
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_lo_rejected(self, bad: float) -> None:
        with pytest.raises(ValidationError, match="finite"):
            SearchSpaceDimension(path="phases.profiling.rate", lo=bad, hi=10)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_hi_rejected(self, bad: float) -> None:
        with pytest.raises(ValidationError, match="finite"):
            SearchSpaceDimension(path="phases.profiling.rate", lo=1, hi=bad)

    def test_finite_bounds_allowed(self) -> None:
        dim = SearchSpaceDimension(path="phases.profiling.rate", lo=0.0, hi=math.pi)
        assert dim.lo == 0.0
        assert dim.hi == math.pi


# -- H13: scenario duplicate names -----------------------------------------


class TestH13ScenarioUniqueRunNames:
    """Duplicate scenario ``name`` would silently produce two variants whose
    cell directories collide on disk -- second clobbers first.
    """

    def test_duplicate_named_runs_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique names"):
            ScenarioSweep(
                runs=[
                    {"name": "a", "benchmark": {"phases": [{"name": "p"}]}},
                    {"name": "a", "benchmark": {"phases": [{"name": "p"}]}},
                ]
            )

    def test_unique_named_runs_allowed(self) -> None:
        sw = ScenarioSweep(
            runs=[
                {"name": "small", "benchmark": {}},
                {"name": "large", "benchmark": {}},
            ]
        )
        assert len(sw.runs) == 2

    def test_unnamed_runs_unaffected(self) -> None:
        # Anonymous runs get auto-labels (scenario_0, scenario_1) and don't
        # collide; the validator skips them.
        sw = ScenarioSweep(runs=[{"benchmark": {}}, {"benchmark": {}}])
        assert len(sw.runs) == 2


# -- H14: empty grid value list --------------------------------------------


class TestH14GridEmptyValueList:
    """``parameters: {x: []}`` used to silently produce zero combinations,
    which expand_sweep then masked by returning the base config alone.
    """

    def test_empty_value_list_rejected(self) -> None:
        data = {
            "benchmark": {},
            "sweep": {
                "type": "grid",
                "parameters": {"phases.profiling.rate": []},
            },
        }
        with pytest.raises(ValueError, match="non-empty"):
            expand_sweep(data)

    def test_singleton_value_list_allowed(self) -> None:
        # One value is a degenerate-but-valid pin, not an error.
        gs = GridSweep(parameters={"phases.profiling.rate": [42]})
        assert gs.parameters == {"phases.profiling.rate": [42]}
