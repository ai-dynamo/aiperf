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
            "datasets": [
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 10,
                    "prompts": {"isl": 8, "osl": 8},
                }
            ],
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
async def test_orchestrator_stamps_variation_metadata_on_each_result(tmp_path):
    """Each RunResult carries variation_label, variation_values, and trial_index."""
    plan = _make_plan(num_variations=2, trials=2)
    # Override variations to carry meaningful values for the assertion.
    plan = plan.model_copy(
        update={
            "variations": [
                SweepVariation(
                    index=0, label="concurrency=10", values={"concurrency": 10}
                ),
                SweepVariation(
                    index=1, label="concurrency=20", values={"concurrency": 20}
                ),
            ]
        }
    )
    executor = FakeExecutor()
    orchestrator = MultiRunOrchestrator(base_dir=tmp_path)

    results = await orchestrator.execute(plan, executor)

    assert [r.variation_label for r in results] == [
        "concurrency=10",
        "concurrency=10",
        "concurrency=20",
        "concurrency=20",
    ]
    assert [r.variation_values["concurrency"] for r in results] == [10, 10, 20, 20]
    assert [r.trial_index for r in results] == [0, 1, 0, 1]


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


@pytest.mark.asyncio
async def test_orchestrator_inter_variation_cooldown_sleeps_between_variations(
    tmp_path, monkeypatch
):
    """parameter_sweep_cooldown_seconds is honored between variations only."""
    import aiperf.orchestrator.orchestrator as orch_mod

    plan = _make_plan(num_variations=2, trials=1)
    plan = plan.model_copy(update={"parameter_sweep_cooldown_seconds": 4.0})

    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(orch_mod.asyncio, "sleep", fake_sleep)

    executor = FakeExecutor()
    orchestrator = MultiRunOrchestrator(base_dir=tmp_path)
    await orchestrator.execute(plan, executor)

    # 2 variations x 1 trial: no inter-trial cooldown (single trial), one
    # inter-variation cooldown before variation 1; nothing after the last.
    assert sleeps == [4.0]


@pytest.mark.asyncio
async def test_orchestrator_inter_variation_cooldown_default_zero_no_sleep(
    tmp_path, monkeypatch
):
    """Default parameter_sweep_cooldown_seconds=0 emits no inter-variation sleep."""
    import aiperf.orchestrator.orchestrator as orch_mod

    plan = _make_plan(num_variations=3, trials=1)

    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(orch_mod.asyncio, "sleep", fake_sleep)

    executor = FakeExecutor()
    orchestrator = MultiRunOrchestrator(base_dir=tmp_path)
    await orchestrator.execute(plan, executor)

    assert sleeps == []


# ---------------------------------------------------------------------------
# Adversarial regression-locks: cancel_check semantics in the orchestrator.
#
# Locks in the just-fixed orchestrator behavior:
#   - cancel_check polled before each variation (between-variations bail).
#   - cancel_check polled inside a variation's trial loop (mid-cell bail).
#   - cancel_check=None preserves prior behavior (compatibility).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_cancel_check_between_variations_returns_partial_results(
    tmp_path,
):
    """cancel_check goes True after the first variation's trials -> stop before var 1."""
    plan = _make_plan(num_variations=3, trials=2)
    executor = FakeExecutor()
    orchestrator = MultiRunOrchestrator(base_dir=tmp_path)

    # Flip cancel after variation 0 finishes (2 calls done).
    state = {"flipped": False}

    def cancel_check() -> bool:
        # Once we've completed all trials of variation 0, signal cancel.
        if not state["flipped"] and len(executor.calls) >= 2:
            state["flipped"] = True
        return state["flipped"]

    results = await orchestrator.execute(plan, executor, cancel_check=cancel_check)

    # Only variation 0's two trials should have run.
    assert [c[0] for c in executor.calls] == [0, 0]
    assert len(results) == 2


@pytest.mark.asyncio
async def test_orchestrator_cancel_check_none_preserves_full_iteration(tmp_path):
    """cancel_check=None => behavior unchanged (compat lock)."""
    plan = _make_plan(num_variations=3, trials=2)
    executor = FakeExecutor()
    orchestrator = MultiRunOrchestrator(base_dir=tmp_path)
    results = await orchestrator.execute(plan, executor, cancel_check=None)
    assert len(results) == 6
    assert [c[0] for c in executor.calls] == [0, 0, 1, 1, 2, 2]


@pytest.mark.asyncio
async def test_orchestrator_cancel_check_inside_trial_loop_truncates_cell(tmp_path):
    """cancel_check goes True mid-cell -> orchestrator bails before next trial.

    The cancel check sits at the top of the trial loop, BEFORE issuing the
    next run. Flipping after 2 trials in variation 0 means only those 2 trials
    execute; trial 3+ are skipped and remaining variations are skipped too.
    """
    plan = _make_plan(num_variations=1, trials=5)
    executor = FakeExecutor()
    orchestrator = MultiRunOrchestrator(base_dir=tmp_path)

    state = {"count": 0}

    def cancel_check() -> bool:
        # The orchestrator polls cancel_check both before each variation and
        # at the top of each trial iteration. Returning True after 2 trials
        # have completed means the 3rd-trial check fires and the loop bails.
        return state["count"] >= 2

    # We can't mutate state from FakeExecutor.execute directly without
    # rewriting it — use a thin wrapper subclass.
    class CountingExecutor(FakeExecutor):
        async def execute(self, run):
            result = await super().execute(run)
            state["count"] += 1
            return result

    executor = CountingExecutor()
    results = await orchestrator.execute(plan, executor, cancel_check=cancel_check)

    # Exactly two trials ran; the 3rd-trial top-of-loop cancel_check fires.
    assert len(executor.calls) == 2
    assert [c[1] for c in executor.calls] == [0, 1]
    assert len(results) == 2
