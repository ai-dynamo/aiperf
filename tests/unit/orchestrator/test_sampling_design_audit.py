# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for `_maybe_write_sampling_design`.

Pinned behavior: the audit JSON written to
`sweep_aggregate/sampling_design.json` must reflect the variations that
were actually expanded upstream (and will be executed), not a fresh
re-draw from the QMC engine. With `seed=None` (the default), re-drawing
produces an unrelated sample set whose audit trail is fiction.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import pytest

from aiperf.config.resolution.plan import BenchmarkPlan
from aiperf.config.sweep import (
    LatinHypercubeSweep,
    SamplingDimension,
    SobolSweep,
    SweepVariation,
)
from aiperf.config.sweep.expand_qmc import expand_qmc_sweep
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator


def _build_plan(sweep, variations):
    plan = MagicMock(spec=BenchmarkPlan)
    plan.sweep = sweep
    plan.variations = [v for _, v in variations]
    plan.configs = [c for c, _ in variations]
    plan.trials = 1
    plan.is_adaptive_search = False
    plan.iteration_order = "REPEATED"
    return plan


class TestSamplingDesignMatchesVariations:
    """The audit JSON must equal the variants the orchestrator runs."""

    def test_sobol_seed_none_audit_matches_variations(self, tmp_path):
        # seed=None reproduces the original bug: a re-draw inside
        # _maybe_write_sampling_design used a fresh OS-entropy seed and
        # produced an unrelated sample set.
        dims = [
            SamplingDimension(
                path="phases.profiling.concurrency", lo=1, hi=32, kind="int"
            )
        ]
        sweep = SobolSweep(samples=4, seed=None, dimensions=dims)
        variations = expand_qmc_sweep(
            {"benchmark": {}},
            sweep_type="sobol",
            samples=4,
            seed=None,
            dimensions=dims,
            options={"scramble": True},
        )
        plan = _build_plan(sweep, variations)

        orch = MultiRunOrchestrator(base_dir=tmp_path)
        orch._maybe_write_sampling_design(plan)

        design = json.loads(
            (tmp_path / "sweep_aggregate" / "sampling_design.json").read_text()
        )
        path = "phases.profiling.concurrency"
        expected = [[v.values[path]] for v in plan.variations]
        assert design["samples_mapped"] == expected
        assert design["seed"] is None

    def test_latin_hypercube_seed_none_audit_matches_variations(self, tmp_path):
        dims = [
            SamplingDimension(
                path="phases.profiling.rate", lo=1.0, hi=100.0, scale="log"
            ),
            SamplingDimension(path="conc", lo=1, hi=64, kind="int"),
        ]
        sweep = LatinHypercubeSweep(samples=5, seed=None, dimensions=dims)
        variations = expand_qmc_sweep(
            {"benchmark": {}},
            sweep_type="latin_hypercube",
            samples=5,
            seed=None,
            dimensions=dims,
            options={},
        )
        plan = _build_plan(sweep, variations)

        orch = MultiRunOrchestrator(base_dir=tmp_path)
        orch._maybe_write_sampling_design(plan)

        design = json.loads(
            (tmp_path / "sweep_aggregate" / "sampling_design.json").read_text()
        )
        expected = [
            [v.values["phases.profiling.rate"], v.values["conc"]]
            for v in plan.variations
        ]
        assert design["samples_mapped"] == expected

    def test_seed_42_reproducible(self, tmp_path):
        # Deterministic seed: re-calling expand twice yields identical
        # variations and the audit reproduces those values byte-for-byte.
        dims = [SamplingDimension(path="x", lo=1, hi=128, kind="int")]
        sweep = SobolSweep(samples=8, seed=42, dimensions=dims)
        variations_a = expand_qmc_sweep(
            {"benchmark": {}},
            sweep_type="sobol",
            samples=8,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )
        variations_b = expand_qmc_sweep(
            {"benchmark": {}},
            sweep_type="sobol",
            samples=8,
            seed=42,
            dimensions=dims,
            options={"scramble": True},
        )
        # Sanity: deterministic seed produces identical variations.
        assert [v.values for _, v in variations_a] == [
            v.values for _, v in variations_b
        ]

        plan = _build_plan(sweep, variations_a)
        orch = MultiRunOrchestrator(base_dir=tmp_path)
        orch._maybe_write_sampling_design(plan)

        design = json.loads(
            (tmp_path / "sweep_aggregate" / "sampling_design.json").read_text()
        )
        expected = [[v.values["x"]] for _, v in variations_a]
        assert design["samples_mapped"] == expected
        assert design["seed"] == 42


class TestSamplingDesignDefensiveGuards:
    def test_non_finite_value_raises(self, tmp_path):
        # Defense-in-depth: SamplingDimension validators should already
        # reject non-finite lo/hi, but if a malformed variation slipped
        # through we must NOT silently coerce nan/inf to null in JSON.
        dims = [SamplingDimension(path="phases.profiling.rate", lo=1.0, hi=100.0)]
        sweep = SobolSweep(samples=2, seed=42, dimensions=dims)
        plan = MagicMock(spec=BenchmarkPlan)
        plan.sweep = sweep
        plan.variations = [
            SweepVariation(
                index=0, label="bad", values={"phases.profiling.rate": math.inf}
            ),
            SweepVariation(index=1, label="ok", values={"phases.profiling.rate": 50.0}),
        ]
        plan.configs = [{}, {}]

        orch = MultiRunOrchestrator(base_dir=tmp_path)
        with pytest.raises(ValueError, match="non-finite"):
            orch._maybe_write_sampling_design(plan)

    def test_grid_sweep_writes_nothing(self, tmp_path):
        from aiperf.config.sweep import GridSweep

        plan = MagicMock(spec=BenchmarkPlan)
        plan.sweep = GridSweep(parameters={"x": [1, 2]})
        plan.variations = []
        plan.configs = []

        orch = MultiRunOrchestrator(base_dir=tmp_path)
        orch._maybe_write_sampling_design(plan)
        assert not (tmp_path / "sweep_aggregate" / "sampling_design.json").exists()
