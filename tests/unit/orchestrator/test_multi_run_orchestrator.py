# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunOrchestrator (variations x trials iteration via RunExecutor)."""

from __future__ import annotations

import pytest

from aiperf.config.benchmark import BenchmarkPlan, BenchmarkRun
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import SweepVariation
from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator


class FakeExecutor(RunExecutor):
    """Records every (var_idx, trial, label) tuple it sees."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, str]] = []

    def derive_id(self, plan, var_idx: int, trial: int) -> str:
        return f"v{var_idx}-t{trial}"

    async def execute(self, run: BenchmarkRun) -> RunResult:
        var_idx = run.variation.index if run.variation else -1
        self.calls.append((var_idx, run.trial, run.label))
        # request_count > 0 so the strategy classifies the run as successful;
        # but RunResult here just needs success=True for the orchestrator.
        return RunResult(
            label=run.label,
            success=True,
            artifacts_path=run.artifact_dir,
        )


def _make_plan(num_variations: int, trials: int) -> BenchmarkPlan:
    """Build a BenchmarkPlan with N distinct configs (representing variations)."""
    base_cfg = BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": {
                "default": {
                    "type": "synthetic",
                    "entries": 10,
                    "prompts": {"isl": 8, "osl": 8},
                },
            },
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                },
            ],
        }
    )
    configs = [base_cfg.model_copy(deep=True) for _ in range(num_variations)]
    variations = [
        SweepVariation(index=i, label=f"v{i}", values={}) for i in range(num_variations)
    ]
    return BenchmarkPlan(configs=configs, variations=variations, trials=trials)


@pytest.mark.asyncio
async def test_orchestrator_iterates_all_variations_x_trials(tmp_path):
    """Latent bug fix: the orchestrator now iterates all configs x trials,
    not just configs[0]."""
    plan = _make_plan(num_variations=3, trials=2)
    executor = FakeExecutor()
    orchestrator = MultiRunOrchestrator(base_dir=tmp_path)

    results = await orchestrator.execute(plan, executor)

    assert len(results) == 6  # 3 variations x 2 trials
    assert len(executor.calls) == 6
    # Variations iterated in order, all trials per variation before moving to next.
    assert [c[0] for c in executor.calls] == [0, 0, 1, 1, 2, 2]
    assert [c[1] for c in executor.calls] == [0, 1, 0, 1, 0, 1]


@pytest.mark.asyncio
async def test_orchestrator_single_variation_single_trial(tmp_path):
    """Trivial case: one variation, one trial."""
    plan = _make_plan(num_variations=1, trials=1)
    executor = FakeExecutor()
    orchestrator = MultiRunOrchestrator(base_dir=tmp_path)
    results = await orchestrator.execute(plan, executor)
    assert len(results) == 1
    assert executor.calls[0][0] == 0  # variation 0
    assert executor.calls[0][1] == 0  # trial 0


@pytest.mark.asyncio
async def test_orchestrator_applies_cooldown_between_trials(tmp_path, monkeypatch):
    """Cooldown is applied between trials within a variation, not after the last."""
    import aiperf.orchestrator.orchestrator as orch_mod

    plan = _make_plan(num_variations=1, trials=3)
    plan = plan.model_copy(update={"cooldown_seconds": 1.5})

    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(orch_mod.asyncio, "sleep", fake_sleep)

    executor = FakeExecutor()
    orchestrator = MultiRunOrchestrator(base_dir=tmp_path)
    await orchestrator.execute(plan, executor)

    # 3 trials -> 2 inter-trial cooldowns; orchestrator reads from strategy
    # which derives from plan.cooldown_seconds.
    assert sleeps == [1.5, 1.5]
