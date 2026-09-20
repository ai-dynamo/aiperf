# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate dynamically proposed exports without running an optimizer or benchmark."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.cli_runner._strategy import validate_convergence_config
from aiperf.config import AIPerfConfig, BenchmarkConfig
from aiperf.config.loader import build_benchmark_plan
from aiperf.config.sweep import SweepVariation, _set_nested_value
from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, records, raw_values, rejected_index",
    [
        param("distribution", False, [0], 0, id="summary-first"),
        param("distribution", False, [1, 0], 1, id="summary-after-valid-point"),
        param("distribution", False, [1, 1], None, id="raw-points"),
        param("distribution", True, [0, 1], None, id="mixed-records-raw"),
        param("ci_width", False, [1, 0], None, id="ci-width-allows-summary"),
        param("cv", False, [1, 0], None, id="cv-allows-summary"),
        param(None, False, [1, 0], None, id="fixed-trials-allow-summary"),
    ],
)  # fmt: skip
async def test_adaptive_search_validates_each_proposed_export(
    mode: str | None,
    records: bool,
    raw_values: list[int],
    rejected_index: int | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an invalid proposal before any of its trials can be dispatched."""
    config = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": [{"name": "default", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 10,
                    }
                ],
                "artifacts": {"records": ["jsonl"] if records else False, "raw": True},
            },
            "multi_run": {
                "num_runs": 3,
                "convergence": {"metric": "request_latency", "mode": mode}
                if mode is not None
                else None,
            },
            "sweep": {
                "type": "adaptive_search",
                "search_space": [
                    {"path": "artifacts.raw", "lo": 0, "hi": 1, "kind": "int"}
                ],
                "objectives": [
                    {
                        "metric": "output_token_throughput",
                        "stat": "avg",
                        "direction": "maximize",
                    }
                ],
                "max_iterations": 10,
            },
        }
    )
    plan = build_benchmark_plan(config)
    validate_convergence_config(plan)

    proposals = []
    for index, raw in enumerate(raw_values):
        cfg_dict = plan.configs[0].model_dump(mode="python", exclude_none=True)
        _set_nested_value(cfg_dict, "artifacts.raw", raw)
        proposals.append(
            (
                BenchmarkConfig.model_validate(cfg_dict),
                SweepVariation(
                    index=index,
                    label=f"search_iter_{index:04d}",
                    values={"artifacts.raw": raw},
                ),
            )
        )
    planner = MagicMock()
    planner.history.return_value = []
    planner.ask.side_effect = [*proposals, None]
    planner.iter_count = len(proposals)
    planner.convergence_reason.return_value = "max_iterations"

    orchestrator = MultiRunOrchestrator(base_dir=tmp_path)
    run_cell = AsyncMock(return_value=([], False))
    monkeypatch.setattr(orchestrator, "_run_independent_cell", run_cell)
    monkeypatch.setattr(
        "aiperf.exporters.search_history.write_search_history", MagicMock()
    )
    monkeypatch.setattr(
        "aiperf.orchestrator.search_planner.write_search_checkpoint", MagicMock()
    )
    executor = MagicMock(spec=RunExecutor)
    if rejected_index is not None:
        with pytest.raises(
            ValueError,
            match=f"search iteration {rejected_index} has export level 'summary'",
        ):
            await orchestrator.execute(plan, executor, search_planner=planner)
    else:
        await orchestrator.execute(plan, executor, search_planner=planner)

    executed_count = rejected_index if rejected_index is not None else len(proposals)
    assert run_cell.await_count == executed_count
    assert [
        call.kwargs["variation"].index for call in run_cell.await_args_list
    ] == list(range(executed_count))
    assert planner.tell.call_count == executed_count
