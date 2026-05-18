# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for QMC sampling sweep types (Sobol, Latin Hypercube)."""

import json
import math
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from aiperf.config.sweep import (
    LatinHypercubeSweep,
    SamplingDimension,
    SobolSweep,
    expand_sweep,
)
from aiperf.config.sweep.expand_qmc import expand_qmc_sweep


class TestSamplingDimension:
    def test_continuous_real_dim_valid(self):
        dim = SamplingDimension(path="phases.profiling.rate", lo=1.0, hi=100.0)
        assert dim.scale == "linear"
        assert dim.kind == "real"
        assert dim.choices is None

    def test_continuous_int_log_dim_valid(self):
        dim = SamplingDimension(
            path="phases.profiling.concurrency",
            lo=1,
            hi=256,
            scale="log",
            kind="int",
        )
        assert dim.scale == "log"
        assert dim.kind == "int"

    def test_choices_dim_valid(self):
        dim = SamplingDimension(path="model", choices=["a", "b", "c"])
        assert dim.choices == ["a", "b", "c"]
        assert dim.lo is None
        assert dim.hi is None

    def test_neither_range_nor_choices_rejected(self):
        with pytest.raises(ValidationError, match="either"):
            SamplingDimension(path="phases.profiling.concurrency")

    def test_both_range_and_choices_rejected(self):
        with pytest.raises(ValidationError, match="either"):
            SamplingDimension(
                path="phases.profiling.concurrency",
                lo=1,
                hi=10,
                choices=[1, 2, 3],
            )

    def test_lo_ge_hi_rejected(self):
        with pytest.raises(ValidationError, match="must be > lo"):
            SamplingDimension(path="x", lo=10, hi=5)

    def test_log_scale_with_zero_lo_rejected(self):
        with pytest.raises(ValidationError, match="log-scale requires lo > 0"):
            SamplingDimension(path="x", lo=0, hi=10, scale="log")

    def test_log_scale_with_negative_lo_rejected(self):
        with pytest.raises(ValidationError, match="log-scale requires lo > 0"):
            SamplingDimension(path="x", lo=-1, hi=10, scale="log")

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            SamplingDimension(path="x", lo=1, hi=10, garbage=True)


class TestSobolSweep:
    def test_minimum_fields_valid(self):
        sweep = SobolSweep(
            samples=16,
            dimensions=[
                {
                    "path": "phases.profiling.concurrency",
                    "lo": 1,
                    "hi": 256,
                    "scale": "log",
                    "kind": "int",
                },
            ],
        )
        assert sweep.type == "sobol"
        assert sweep.samples == 16
        assert sweep.scramble is True
        assert sweep.label_format == "index"

    def test_samples_below_two_rejected(self):
        with pytest.raises(ValidationError):
            SobolSweep(
                samples=1,
                dimensions=[{"path": "x", "lo": 1, "hi": 10}],
            )

    def test_zero_dimensions_rejected(self):
        with pytest.raises(ValidationError):
            SobolSweep(samples=8, dimensions=[])

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            SobolSweep(
                samples=8,
                dimensions=[{"path": "x", "lo": 1, "hi": 10}],
                garbage=True,
            )

    def test_seed_optional(self):
        sweep = SobolSweep(
            samples=8,
            seed=42,
            dimensions=[{"path": "x", "lo": 1, "hi": 10}],
        )
        assert sweep.seed == 42

    def test_label_format_kv_accepted(self):
        sweep = SobolSweep(
            samples=8,
            label_format="kv",
            dimensions=[{"path": "x", "lo": 1, "hi": 10}],
        )
        assert sweep.label_format == "kv"


class TestLatinHypercubeSweep:
    def test_minimum_fields_valid(self):
        sweep = LatinHypercubeSweep(
            samples=12,
            dimensions=[{"path": "model", "choices": ["a", "b", "c"]}],
        )
        assert sweep.type == "latin_hypercube"
        assert sweep.optimization == "random-cd"

    def test_optimization_none_accepted(self):
        sweep = LatinHypercubeSweep(
            samples=12,
            optimization=None,
            dimensions=[{"path": "x", "lo": 1, "hi": 10}],
        )
        assert sweep.optimization is None


def _base_data():
    """Minimal AIPerfConfig dict for expansion tests."""
    return {
        "benchmark": {
            "model": "test-model",
            "endpoint": {"url": "http://localhost:8000", "type": "chat"},
            "phases": [{"name": "profiling", "concurrency": 1, "duration": 10}],
        }
    }


class TestExpandQMC:
    def test_count_equals_samples(self):
        dims = [
            SamplingDimension(
                path="phases.profiling.concurrency",
                lo=1,
                hi=128,
                scale="log",
                kind="int",
            )
        ]
        out = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=16,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )
        assert len(out) == 16

    def test_deterministic_with_seed(self):
        dims = [
            SamplingDimension(
                path="phases.profiling.concurrency",
                lo=1,
                hi=128,
                scale="log",
                kind="int",
            )
        ]
        a = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=8,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )
        b = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=8,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )
        assert [v.values for _, v in a] == [v.values for _, v in b]

    def test_different_seed_different_points(self):
        dims = [
            SamplingDimension(
                path="phases.profiling.concurrency",
                lo=1,
                hi=128,
                scale="log",
                kind="int",
            )
        ]
        a = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=8,
            seed=1,
            dimensions=dims,
            options={"scramble": True},
        )
        b = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=8,
            seed=2,
            dimensions=dims,
            options={"scramble": True},
        )
        assert [v.values for _, v in a] != [v.values for _, v in b]

    def test_int_kind_produces_integers(self):
        dims = [
            SamplingDimension(
                path="phases.profiling.concurrency",
                lo=1,
                hi=128,
                scale="log",
                kind="int",
            )
        ]
        out = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=8,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )
        for _, var in out:
            v = var.values["phases.profiling.concurrency"]
            assert isinstance(v, int)

    def test_real_kind_produces_floats(self):
        dims = [
            SamplingDimension(
                path="phases.profiling.rate",
                lo=1.0,
                hi=100.0,
                scale="linear",
                kind="real",
            )
        ]
        out = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=8,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )
        for _, var in out:
            assert isinstance(var.values["phases.profiling.rate"], float)

    def test_log_scale_dim_spans_decades(self):
        dims = [SamplingDimension(path="x", lo=1, hi=1024, scale="log", kind="int")]
        out = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=64,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )
        values = [v.values["x"] for _, v in out]
        # With 64 samples on [1, 1024] log scale, at least one value per decade.
        decades = {int(math.log10(max(v, 1))) for v in values}
        assert decades >= {0, 1, 2}, f"missing decades, got {decades}"

    def test_choices_dim_only_uses_provided_values(self):
        choices = ["a", "b", "c", "d"]
        dims = [SamplingDimension(path="model", choices=choices)]
        out = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=16,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )
        seen = {v.values["model"] for _, v in out}
        assert seen <= set(choices)

    def test_lhs_marginal_each_bin_hit(self):
        dims = [SamplingDimension(path="x", lo=0.0, hi=10.0, scale="linear")]
        out = expand_qmc_sweep(
            _base_data(),
            sweep_type="latin_hypercube",
            samples=10,
            seed=42,
            dimensions=dims,
            options={"optimization": "random-cd"},
        )
        values = sorted(v.values["x"] for _, v in out)
        # LHS guarantees one sample per [0,1), [1,2), ..., [9,10) bin.
        bins = [int(v) for v in values]
        assert sorted(bins) == list(range(10))

    def test_path_set_via_set_nested_value(self):
        """Verify the variant dict has the value at the dotted path."""
        dims = [
            SamplingDimension(
                path="phases.profiling.concurrency",
                lo=1,
                hi=128,
                scale="log",
                kind="int",
            )
        ]
        out = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=4,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )
        for variant, var in out:
            expected = var.values["phases.profiling.concurrency"]
            actual = variant["benchmark"]["phases"][0]["concurrency"]
            assert actual == expected

    def test_label_index_format_default(self):
        dims = [SamplingDimension(path="x", lo=1, hi=10)]
        out = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=4,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )
        labels = [v.label for _, v in out]
        assert labels == ["sobol_0000", "sobol_0001", "sobol_0002", "sobol_0003"]

    def test_label_kv_format(self):
        dims = [SamplingDimension(path="rate", lo=1, hi=10, kind="int")]
        out = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=2,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
            label_format="kv",
        )
        for _, var in out:
            assert "rate=" in var.label

    def test_variation_index_monotonic(self):
        dims = [SamplingDimension(path="x", lo=1, hi=10)]
        out = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=8,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )
        assert [v.index for _, v in out] == list(range(8))

    def test_variation_values_match_dimensions(self):
        dims = [
            SamplingDimension(path="a", lo=1, hi=10),
            SamplingDimension(path="b", lo=1, hi=10),
        ]
        out = expand_qmc_sweep(
            _base_data(),
            sweep_type="sobol",
            samples=4,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )
        for _, var in out:
            assert set(var.values.keys()) == {"a", "b"}

    def test_qmc_samples_non_power_of_two_warns_only(self):
        import warnings

        from aiperf.config.sweep.expand_qmc import expand_qmc_sweep

        dims = [SamplingDimension(path="x", lo=1, hi=10)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            expand_qmc_sweep(
                {"benchmark": {}},
                sweep_type="sobol",
                samples=10,
                seed=42,
                dimensions=dims,
                options={"scramble": True},
            )
            assert any("powers of 2" in str(warning.message) for warning in w)

    def test_qmc_samples_power_of_two_no_warning(self):
        import warnings

        from aiperf.config.sweep.expand_qmc import expand_qmc_sweep

        dims = [SamplingDimension(path="x", lo=1, hi=10)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            expand_qmc_sweep(
                {"benchmark": {}},
                sweep_type="sobol",
                samples=8,
                seed=42,
                dimensions=dims,
                options={"scramble": True},
            )
            assert not any("powers of 2" in str(warning.message) for warning in w)

    def test_unknown_sweep_type_raises(self):
        dims = [SamplingDimension(path="x", lo=1, hi=10)]
        with pytest.raises(ValueError, match="unknown sampling sweep type"):
            expand_qmc_sweep(
                _base_data(),
                sweep_type="halton",
                samples=4,
                seed=42,
                dimensions=dims,
                options={},
            )


class TestSamplingDesignArtifact:
    @pytest.mark.asyncio
    async def test_sampling_design_written_for_sobol(self, tmp_path):
        from aiperf.config.resolution.plan import BenchmarkPlan
        from aiperf.config.sweep import SamplingDimension, SobolSweep
        from aiperf.orchestrator.orchestrator import MultiRunOrchestrator

        sweep = SobolSweep(
            samples=4,
            seed=42,
            dimensions=[SamplingDimension(path="x", lo=1, hi=10, kind="int")],
        )
        dims = [SamplingDimension(path="x", lo=1, hi=10, kind="int")]
        variations = expand_qmc_sweep(
            {"benchmark": {}},
            sweep_type="sobol",
            samples=4,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )

        # Build a minimal BenchmarkPlan with sweep + variations.
        plan = MagicMock(spec=BenchmarkPlan)
        plan.sweep = sweep
        plan.variations = [v for _, v in variations]
        plan.configs = [c for c, _ in variations]
        plan.trials = 1
        plan.is_adaptive_search = False
        plan.iteration_order = "REPEATED"  # only used for non-QMC paths

        orch = MultiRunOrchestrator(base_dir=tmp_path)
        # Just trigger the design-writing path without running cells.
        orch._maybe_write_sampling_design(plan)

        design_path = tmp_path / "sweep_aggregate" / "sampling_design.json"
        assert design_path.exists()
        design = json.loads(design_path.read_text())
        assert design["type"] == "sobol"
        assert design["samples"] == 4
        assert design["seed"] == 42
        assert design["scramble"] is True
        assert len(design["dimensions"]) == 1
        assert len(design["samples_mapped"]) == 4
        # Audit must reflect the variations actually produced upstream,
        # not a fresh re-draw from the QMC engine.
        expected = [[v.values["x"]] for v in plan.variations]
        assert design["samples_mapped"] == expected

    @pytest.mark.asyncio
    async def test_sampling_design_not_written_for_grid(self, tmp_path):
        from aiperf.config.resolution.plan import BenchmarkPlan
        from aiperf.config.sweep import GridSweep
        from aiperf.orchestrator.orchestrator import MultiRunOrchestrator

        sweep = GridSweep(parameters={"x": [1, 2, 3]})
        plan = MagicMock(spec=BenchmarkPlan)
        plan.sweep = sweep
        plan.variations = []
        plan.configs = []
        plan.trials = 1
        plan.is_adaptive_search = False
        plan.iteration_order = "REPEATED"

        orch = MultiRunOrchestrator(base_dir=tmp_path)
        orch._maybe_write_sampling_design(plan)

        assert not (tmp_path / "sweep_aggregate" / "sampling_design.json").exists()


class TestExpandSweepDispatchQMC:
    def test_sobol_yaml_dispatches(self):
        data = {
            "sweep": {
                "type": "sobol",
                "samples": 8,
                "seed": 42,
                "dimensions": [
                    {
                        "path": "phases.profiling.concurrency",
                        "lo": 1,
                        "hi": 128,
                        "scale": "log",
                        "kind": "int",
                    },
                ],
            },
            "benchmark": {
                "model": "test-model",
                "endpoint": {"url": "http://localhost:8000", "type": "chat"},
                "phases": [{"name": "profiling", "concurrency": 1, "duration": 10}],
            },
        }
        variations = expand_sweep(data)
        assert len(variations) == 8
        assert all(v.label.startswith("sobol_") for _, v in variations)

    def test_latin_hypercube_yaml_dispatches(self):
        data = {
            "sweep": {
                "type": "latin_hypercube",
                "samples": 6,
                "seed": 7,
                "dimensions": [
                    {"path": "model", "choices": ["a", "b", "c"]},
                ],
            },
            "benchmark": {
                "model": "base-model",
                "endpoint": {"url": "http://localhost:8000", "type": "chat"},
            },
        }
        variations = expand_sweep(data)
        assert len(variations) == 6
        assert all(v.label.startswith("latin_hypercube_") for _, v in variations)
