# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reserved profiling pools survive skipped warmup and runtime JSON transport."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.cli_runner._strategy import build_strategy
from aiperf.common.enums import CreditPhase
from aiperf.config import BenchmarkPlan, BenchmarkRun
from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader.plan import build_benchmark_plan
from aiperf.config.sweep.multi_run import ConvergenceConfig
from aiperf.orchestrator.models import RunResult
from aiperf.timing.config import TimingConfig


@pytest.fixture
def partitioned_plan() -> BenchmarkPlan:
    """Resolve an authored two-trial plan with default warmup skipping."""
    config = resolve_config(
        CLIConfig(
            model_names=["test"],
            request_count=9,
            warmup_request_count=5,
            conversation_num_dataset_entries=6,
            warmup_dataset_entries=2,
            num_profile_runs=2,
            random_seed=42,
        )
    )
    return build_benchmark_plan(config)


@pytest.mark.parametrize("adaptive", [False, True])
def test_two_trials_preserve_partition_through_runtime_serialization(
    partitioned_plan: BenchmarkPlan,
    tmp_path: Path,
    adaptive: bool,
) -> None:
    """Retain the reserved pool through fixed and convergence trial transport."""
    if adaptive:
        partitioned_plan.multi_run.convergence = ConvergenceConfig(
            metric="time_to_first_token"
        )
    strategy = build_strategy(partitioned_plan, MagicMock())
    original = partitioned_plan.configs[0]
    results = []
    for trial in range(2):
        cfg = strategy.get_next_config(original, results)
        run = BenchmarkRun(
            benchmark_id=f"trial-{trial}",
            cfg=cfg,
            trial=trial,
            artifact_dir=tmp_path / str(trial),
            random_seed=42,
        )
        restored = BenchmarkRun.model_validate_json(
            run.model_dump_json(exclude_none=True)
        )
        timing = TimingConfig.from_run(restored)
        assert restored.cfg.get_default_dataset().warmup_entries == 2
        assert timing.warmup_dataset_entries == 2
        assert [p.phase for p in timing.phase_configs].count(CreditPhase.WARMUP) == (
            1 if trial == 0 else 0
        )
        results.append(RunResult(label=f"trial-{trial}", success=True))
    assert original.get_warmup_phases()
    assert original.get_default_dataset().warmup_entries == 2


def test_initial_plan_still_requires_declared_warmup(
    partitioned_plan: BenchmarkPlan,
) -> None:
    """Distinguish invalid initial plans from valid no-warmup trial copies."""
    strategy = build_strategy(partitioned_plan, MagicMock())
    runtime_cfg = strategy.get_next_config(
        partitioned_plan.configs[0],
        [RunResult(label="first", success=True)],
    )
    with pytest.raises(ValueError, match="requires a warmup phase"):
        BenchmarkPlan(configs=[runtime_cfg], trials=2)
