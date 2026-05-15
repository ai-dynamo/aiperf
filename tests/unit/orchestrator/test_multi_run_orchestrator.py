# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunOrchestrator (variations x trials iteration via RunExecutor)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.common.enums import SweepMode
from aiperf.config.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkPlan, BenchmarkRun
from aiperf.config.sweep import GridSweep, SweepVariation
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
    return BenchmarkPlan(
        configs=configs,
        variations=variations,
        trials=trials,
        sweep=GridSweep(
            parameters={"phases.profiling.concurrency": [1]},
            iteration_order=SweepMode.INDEPENDENT,
        ),
    )


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
    """sweep.cooldown_seconds is honored between variations only."""
    import aiperf.orchestrator.orchestrator as orch_mod

    plan = _make_plan(num_variations=2, trials=1)
    plan = plan.model_copy(
        update={
            "sweep": GridSweep(
                parameters={"phases.profiling.concurrency": [1]},
                iteration_order=SweepMode.INDEPENDENT,
                cooldown_seconds=4.0,
            )
        }
    )

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


# ---------------------------------------------------------------------------
# Sweep iteration_order dispatch: independent vs repeated.
#
# Locks:
#   - INDEPENDENT default: variations outer, trials inner.
#   - REPEATED: trials outer, variations inner.
#   - REPEATED artifact tree includes the trial_NNNN/<variation>/ prefix.
# ---------------------------------------------------------------------------


def _make_two_variation_plan(mode: SweepMode, trials: int = 3) -> BenchmarkPlan:
    """Build a minimal 2-variation plan with the given iteration_order."""
    plan = _make_plan(num_variations=2, trials=trials)
    return plan.model_copy(
        update={
            "variations": [
                SweepVariation(
                    index=0,
                    label="phases.profiling.concurrency=10",
                    values={"phases.profiling.concurrency": 10},
                ),
                SweepVariation(
                    index=1,
                    label="phases.profiling.concurrency=20",
                    values={"phases.profiling.concurrency": 20},
                ),
            ],
            "sweep": GridSweep(
                parameters={"phases.profiling.concurrency": [10, 20]},
                iteration_order=mode,
            ),
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_order"),
    [
        param(
            SweepMode.INDEPENDENT,
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            id="independent-variation-outer",
        ),
        param(
            SweepMode.REPEATED,
            [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)],
            id="repeated-trial-outer",
        ),
    ],
)  # fmt: skip
async def test_iteration_order_honors_parameter_sweep_mode(
    tmp_path: Path,
    mode: SweepMode,
    expected_order: list[tuple[int, int]],
) -> None:
    """Iteration order is variation-outer for independent, trial-outer for repeated."""
    plan = _make_two_variation_plan(mode, trials=3)
    executor = FakeExecutor()
    await MultiRunOrchestrator(base_dir=tmp_path).execute(plan, executor)
    assert [(c[0], c[1]) for c in executor.calls] == expected_order


@pytest.mark.asyncio
async def test_repeated_mode_artifact_path_includes_trial_prefix(tmp_path):
    """Repeated artifact dirs are <base>/profile_runs/trial_NNNN/<dir_name>/.

    Layout for sweep + multi-run REPEATED:
    one run per (trial, variation) cell, with the per-variation
    directory using the ``{last_seg}_{value}`` form
    (``concurrency_10``) — not the dotted-path variation label.
    """
    plan = _make_two_variation_plan(SweepMode.REPEATED, trials=2)
    seen: list[Path] = []

    class PathRecorder(FakeExecutor):
        async def execute(self, run):
            seen.append(run.artifact_dir)
            return await super().execute(run)

    await MultiRunOrchestrator(base_dir=tmp_path).execute(plan, PathRecorder())
    # Order under repeated: (v0, t0), (v1, t0), (v0, t1), (v1, t1).
    # Artifact path = <base>/profile_runs/trial_NNNN/<dir_name>
    expected_v0_t0 = tmp_path / "profile_runs" / "trial_0001" / "concurrency_10"
    expected_v1_t1 = tmp_path / "profile_runs" / "trial_0002" / "concurrency_20"
    assert seen[0] == expected_v0_t0
    assert seen[3] == expected_v1_t1


@pytest.mark.asyncio
async def test_repeated_mode_passes_growing_prior_results_to_strategy(
    tmp_path, monkeypatch
):
    """Each successive trial in repeated mode passes a growing prior-results list.

    REGRESSION-LOCK: If you are tempted to replace
    ``strategy.get_next_config(cfg, per_variation_history[var_idx])``
    in ``MultiRunOrchestrator._execute_repeated`` with
    ``strategy.get_next_config(cfg, [])`` because "the list is never read",
    this test will fail and tell you why. The source-side comment in
    ``orchestrator.py`` documents the invariant; this test enforces it.

    FixedTrialsStrategy.get_next_config keys disable_warmup_after_first off
    ``len(prior) > 0``. Passing ``[]`` every trial re-enables warmup on
    every trial - silently diverging from main's _execute_trials_then_sweep
    semantic where warmup runs only in trial 1. The asserted progression
    is ``[0, 1, 2]`` per variation; with the regression it would be
    ``[0, 0, 0]``.
    """
    plan = _make_two_variation_plan(SweepMode.REPEATED, trials=3)
    captured: list[tuple[int, int]] = []  # (var_idx_seen, prior_len)
    counter = {"n": 0}

    class _SpyStrategy:
        def __init__(self, var_idx: int) -> None:
            self.var_idx = var_idx

        def validate_config(self, cfg) -> None:
            pass

        def get_next_config(self, cfg, prior):
            captured.append((self.var_idx, len(prior)))
            return cfg

        def get_run_label(self, trial):
            return f"run_{trial + 1:04d}"

        def get_run_path(self, base, trial):
            return Path(base) / "profile_runs" / f"run_{trial + 1:04d}"

        def get_cooldown_seconds(self):
            return 0.0

    def _fake_build(plan, logger):
        s = _SpyStrategy(counter["n"])
        counter["n"] += 1
        return s

    # _execute_repeated imports build_strategy from cli_runner._strategy
    # locally, so patch the source module.
    import aiperf.cli_runner._strategy as strategy_mod

    monkeypatch.setattr(strategy_mod, "build_strategy", _fake_build)

    executor = FakeExecutor()
    await MultiRunOrchestrator(base_dir=tmp_path).execute(plan, executor)

    # Group prior-len progression by var_idx.
    by_var: dict[int, list[int]] = {0: [], 1: []}
    for var_idx, prior_len in captured:
        by_var[var_idx].append(prior_len)
    # Each variation should see prior-results length grow 0 -> 1 -> 2 across
    # its three trials. If repeated mode passed [] every trial (the regression
    # this locks against), every entry would be 0.
    assert by_var[0] == [0, 1, 2]
    assert by_var[1] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Per-mode completion logging: sweeps emit "{mode} mode complete: N/M",
# non-sweeps emit the generic "All runs complete: N/M".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_mode_logs_per_mode_completion(tmp_path, caplog):
    """Sweep + REPEATED logs 'Repeated mode complete: N/M runs successful'."""
    import logging

    plan = _make_two_variation_plan(SweepMode.REPEATED, trials=2)
    executor = FakeExecutor()
    orchestrator = MultiRunOrchestrator(base_dir=tmp_path)
    with caplog.at_level(logging.INFO, logger="aiperf.orchestrator.orchestrator"):
        await orchestrator.execute(plan, executor)

    msgs = [r.message for r in caplog.records]
    assert any("Repeated mode complete: 4/4 runs successful" in m for m in msgs)
    assert not any(m.startswith("All runs complete:") for m in msgs)


@pytest.mark.asyncio
async def test_independent_mode_logs_per_mode_completion(tmp_path, caplog):
    """Sweep + INDEPENDENT logs 'Independent mode complete: N/M runs successful'."""
    import logging

    plan = _make_two_variation_plan(SweepMode.INDEPENDENT, trials=2)
    executor = FakeExecutor()
    orchestrator = MultiRunOrchestrator(base_dir=tmp_path)
    with caplog.at_level(logging.INFO, logger="aiperf.orchestrator.orchestrator"):
        await orchestrator.execute(plan, executor)

    msgs = [r.message for r in caplog.records]
    assert any("Independent mode complete: 4/4 runs successful" in m for m in msgs)
    assert not any(m.startswith("All runs complete:") for m in msgs)


@pytest.mark.asyncio
async def test_non_sweep_run_logs_generic_completion(tmp_path, caplog):
    """Non-sweep (single config, multiple trials) keeps the generic completion line."""
    import logging

    # Single variation -> plan.is_sweep is False even with multiple trials.
    plan = _make_plan(num_variations=1, trials=2)
    executor = FakeExecutor()
    orchestrator = MultiRunOrchestrator(base_dir=tmp_path)
    with caplog.at_level(logging.INFO, logger="aiperf.orchestrator.orchestrator"):
        await orchestrator.execute(plan, executor)

    msgs = [r.message for r in caplog.records]
    assert any("All runs complete: 2/2 successful" in m for m in msgs)
    assert not any("Independent mode complete" in m for m in msgs)
    assert not any("Repeated mode complete" in m for m in msgs)


# ----------------------------------------------------------------------
# Layout cases: lock all 5 folder shapes.
# ----------------------------------------------------------------------
#
# | sweep | trials | mode        | layout                                          |
# |-------|--------|-------------|-------------------------------------------------|
# | no    | 1      | -           | <base>/                                         |
# | no    | >1     | -           | <base>/profile_runs/run_NNNN/                   |
# | yes   | 1      | -           | <base>/<dir_name>/                              |
# | yes   | >1     | REPEATED    | <base>/profile_runs/trial_NNNN/<dir_name>/      |
# | yes   | >1     | INDEPENDENT | <base>/<dir_name>/profile_runs/run_NNNN/        |


def _make_sweep_plan(mode: SweepMode, trials: int) -> BenchmarkPlan:
    """Plan with 2 concurrency variations carrying real ``values`` so dir_name resolves."""
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
    configs = [base_cfg.model_copy(deep=True), base_cfg.model_copy(deep=True)]
    variations = [
        SweepVariation(
            index=0,
            label="phases.profiling.concurrency=10",
            values={"phases.profiling.concurrency": 10},
        ),
        SweepVariation(
            index=1,
            label="phases.profiling.concurrency=20",
            values={"phases.profiling.concurrency": 20},
        ),
    ]
    return BenchmarkPlan(
        configs=configs,
        variations=variations,
        trials=trials,
        sweep=GridSweep(
            parameters={"phases.profiling.concurrency": [10, 20]},
            iteration_order=mode,
        ),
    )


class _RecordingExecutor(FakeExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[Path] = []

    async def execute(self, run):
        self.paths.append(run.artifact_dir)
        return await super().execute(run)


@pytest.mark.asyncio
async def test_layout_no_sweep_single_run(tmp_path):
    """no sweep + trials=1 -> <base>/ (artifacts directly)."""
    plan = _make_plan(num_variations=1, trials=1)
    rec = _RecordingExecutor()
    await MultiRunOrchestrator(base_dir=tmp_path).execute(plan, rec)
    assert rec.paths == [tmp_path]


@pytest.mark.asyncio
async def test_layout_no_sweep_multi_run(tmp_path):
    """no sweep + trials>1 -> <base>/profile_runs/run_NNNN/."""
    plan = _make_plan(num_variations=1, trials=3)
    rec = _RecordingExecutor()
    await MultiRunOrchestrator(base_dir=tmp_path).execute(plan, rec)
    assert rec.paths == [
        tmp_path / "profile_runs" / "run_0001",
        tmp_path / "profile_runs" / "run_0002",
        tmp_path / "profile_runs" / "run_0003",
    ]


@pytest.mark.asyncio
async def test_layout_sweep_single_run(tmp_path):
    """sweep + trials=1 -> <base>/<dir_name>/ (flat at top)."""
    plan = _make_sweep_plan(SweepMode.INDEPENDENT, trials=1)
    rec = _RecordingExecutor()
    await MultiRunOrchestrator(base_dir=tmp_path).execute(plan, rec)
    assert rec.paths == [
        tmp_path / "concurrency_10",
        tmp_path / "concurrency_20",
    ]


@pytest.mark.asyncio
async def test_layout_sweep_multi_run_repeated(tmp_path):
    """sweep + trials>1 + REPEATED -> <base>/profile_runs/trial_NNNN/<dir_name>/."""
    plan = _make_sweep_plan(SweepMode.REPEATED, trials=2)
    rec = _RecordingExecutor()
    await MultiRunOrchestrator(base_dir=tmp_path).execute(plan, rec)
    # Order: (v0,t0), (v1,t0), (v0,t1), (v1,t1)
    assert rec.paths == [
        tmp_path / "profile_runs" / "trial_0001" / "concurrency_10",
        tmp_path / "profile_runs" / "trial_0001" / "concurrency_20",
        tmp_path / "profile_runs" / "trial_0002" / "concurrency_10",
        tmp_path / "profile_runs" / "trial_0002" / "concurrency_20",
    ]


@pytest.mark.asyncio
async def test_layout_sweep_multi_run_independent(tmp_path):
    """sweep + trials>1 + INDEPENDENT -> <base>/<dir_name>/profile_runs/trial_NNNN/."""
    plan = _make_sweep_plan(SweepMode.INDEPENDENT, trials=2)
    rec = _RecordingExecutor()
    await MultiRunOrchestrator(base_dir=tmp_path).execute(plan, rec)
    # Order: (v0,t0), (v0,t1), (v1,t0), (v1,t1)
    assert rec.paths == [
        tmp_path / "concurrency_10" / "profile_runs" / "trial_0001",
        tmp_path / "concurrency_10" / "profile_runs" / "trial_0002",
        tmp_path / "concurrency_20" / "profile_runs" / "trial_0001",
        tmp_path / "concurrency_20" / "profile_runs" / "trial_0002",
    ]


def test_layout_adaptive_search_uses_variation_label(tmp_path):
    """adaptive (BO) layout uses variation.label, not dir_name.

    BO iterations get ``search_iter_NNNN`` labels and we want each one
    in its own iteration-numbered subtree, not a coordinates-named
    directory that could collide if the planner re-proposes nearby
    points. ``_resolve_artifact_dir`` short-circuits on
    ``plan.is_adaptive_search`` for this reason.
    """
    from unittest.mock import MagicMock

    from aiperf.config.sweep import SweepVariation
    from aiperf.orchestrator.orchestrator import _resolve_artifact_dir

    plan = MagicMock()
    plan.is_adaptive_search = True
    variation = SweepVariation(
        index=3,
        label="search_iter_0004",
        values={"phases.profiling.concurrency": 142},
    )
    out = _resolve_artifact_dir(tmp_path, plan, variation, trial_index=0)
    assert out == tmp_path / "search_iter_0004" / "profile_runs" / "run_0001"
