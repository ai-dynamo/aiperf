# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration smoke test: 8-sample Sobol sweep through the orchestrator.

Wires the Sobol sweep type from YAML through ``build_benchmark_plan`` and a
``MultiRunOrchestrator`` against a stub executor (mirroring the pattern in
``test_search_e2e.py`` so the smoke does not depend on a running mock server).
The point of this test is to catch regressions in:

- Each Sobol variation produces its own cell directory (named via
  ``SweepVariation.dir_name`` -- ``concurrency_N`` for a single-dim sweep,
  matching the origin/main artifact-layout convention).
- The ``sweep_aggregate/sampling_design.json`` audit artifact being written
  before any cell runs.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.config import AIPerfConfig
from aiperf.config.loader.plan import build_benchmark_plan
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.config.sweep import SobolSweep
from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator

pytestmark = pytest.mark.component_integration


class _StubExecutor(RunExecutor):
    """Stub executor that returns a deterministic RunResult per variation."""

    def derive_id(self, plan, var_idx, trial):
        return f"stub-v{var_idx}-t{trial}"

    async def execute(self, run: BenchmarkRun) -> RunResult:
        # Materialize the artifact dir so cell_dirs is observable on disk.
        run.artifact_dir.mkdir(parents=True, exist_ok=True)
        return RunResult(
            label=run.label,
            success=True,
            summary_metrics={
                "output_token_throughput": JsonMetricResult(unit="tok/s", avg=1.0),
            },
            artifacts_path=run.artifact_dir,
        )


@pytest.mark.asyncio
async def test_sobol_sweep_8_samples_smoke(tmp_path: Path) -> None:
    """An 8-sample Sobol sweep produces 8 cell dirs and a sampling_design.json."""
    cfg = AIPerfConfig.model_validate(
        {
            "random_seed": 42,
            "sweep": {
                "type": "sobol",
                "samples": 8,
                "seed": 42,
                # Use INDEPENDENT so cell dirs land directly under base_dir
                # rather than under profile_runs/trial_NNNN/.
                "iteration_order": "independent",
                "dimensions": [
                    {
                        "path": "phases.profiling.concurrency",
                        "lo": 1,
                        "hi": 32,
                        "scale": "log",
                        "kind": "int",
                    },
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
    assert isinstance(plan.sweep, SobolSweep)
    assert len(plan.configs) == 8
    assert len(plan.variations) == 8

    # Each variation's sampled concurrency must actually land on the
    # corresponding BenchmarkConfig phase. Without this assertion the
    # H9 body-rooting bug went undetected: variants reported sampled
    # values in `var.values` while every BenchmarkConfig stayed at the
    # base concurrency.
    sampled = [v.values["phases.profiling.concurrency"] for v in plan.variations]
    actual = [
        next(p for p in c.phases if p.name == "profiling").concurrency
        for c in plan.configs
    ]
    assert actual == sampled
    assert len(set(actual)) > 1, (
        "Sobol with samples=8 should produce more than one distinct "
        "concurrency value; saw all-identical -- body-rooting regression?"
    )

    orch = MultiRunOrchestrator(base_dir=tmp_path)
    results = await orch.execute(plan, _StubExecutor())

    assert len(results) == 8
    assert all(r.success for r in results)

    cell_dirs = sorted(
        p.name
        for p in tmp_path.iterdir()
        if p.is_dir() and p.name.startswith("concurrency_")
    )
    expected = sorted({v.dir_name for v in plan.variations})
    assert cell_dirs == expected

    design_path = tmp_path / "sweep_aggregate" / "sampling_design.json"
    assert design_path.exists()
    design = orjson.loads(design_path.read_bytes())
    assert design["type"] == "sobol"
    assert design["samples"] == 8
