# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry points for executing a BenchmarkPlan from the CLI.

Public surface:

* :func:`run_benchmark` - dispatch single-run vs multi-run by plan shape.
* :class:`CompletedRun` / :data:`OnComplete` - post-run callback contract.

Single-run execution lives in :mod:`aiperf.cli_runner._single_run`,
multi-run in :mod:`aiperf.cli_runner._multi_run`. Per-domain helpers split
across ``_strategy.py``, ``_aggregate.py``, ``_sweep_aggregate.py``,
``_sweep_table.py``, ``_banner.py``, ``_preflight.py``, ``_process_setup.py``,
``_callbacks.py``, and ``_post_process.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from aiperf.cli_runner._aggregate import (
    print_aggregate_summary as _print_aggregate_summary,
)
from aiperf.cli_runner._callbacks import (
    CompletedRun,
    OnComplete,
    _invoke_callbacks,  # noqa: F401  re-exported for tests/other internal callers
)
from aiperf.cli_runner._multi_run import _run_multi_benchmark
from aiperf.cli_runner._preflight import (
    _preflight_artifact_dir,
    _preflight_endpoint_ready,
    _preflight_fd_limit,
)
from aiperf.cli_runner._single_run import _run_single_benchmark

if TYPE_CHECKING:
    from aiperf.config import BenchmarkConfig, BenchmarkPlan, BenchmarkRun


__all__ = [
    "CompletedRun",
    "OnComplete",
    "_print_aggregate_summary",
    "_run_multi_benchmark",
    "_run_single_benchmark",
    "run_benchmark",
]


def run_benchmark(plan: BenchmarkPlan) -> None:
    """Run benchmarks from a BenchmarkPlan.

    For single-config single-trial plans, runs directly (Dashboard works).
    For multi-config or multi-trial plans, uses the MultiRunOrchestrator.

    Args:
        plan: BenchmarkPlan to execute.
    """
    if plan.use_adaptive and plan.trials <= 1:
        raise ValueError(
            "--convergence-metric requires --num-profile-runs > 1. "
            "Set --num-profile-runs to at least 2 to enable adaptive convergence."
        )

    _preflight_artifact_dir(plan)
    _preflight_fd_limit()
    _preflight_endpoint_ready(plan)

    callbacks: list[OnComplete] = []
    if plan.configs[0].artifacts.auto_plot:
        from aiperf.plot.auto_plot import build_auto_plot_callback

        callbacks.append(
            build_auto_plot_callback(
                plot_required=plan.configs[0].artifacts.plot_required,
                plot_envelope=plan.plot,
            )
        )

    if plan.is_single_run:
        from aiperf.orchestrator.orchestrator import _resolve_run_seed

        seed = _resolve_run_seed(plan, plan.variations[0])
        run = _make_benchmark_run(
            plan.configs[0], random_seed=seed, variables=plan.variables
        )
        _run_single_benchmark(run, on_complete=callbacks)
    else:
        _run_multi_benchmark(plan, on_complete=callbacks)


def _make_benchmark_run(
    config: BenchmarkConfig,
    *,
    benchmark_id: str | None = None,
    trial: int = 0,
    artifact_dir: Path | None = None,
    random_seed: int | None = None,
    variables: dict[str, Any] | None = None,
) -> BenchmarkRun:
    """Wrap a BenchmarkConfig into a BenchmarkRun.

    Used by :func:`run_benchmark` for the single-run path and by
    ``_multi_run._run_multi_benchmark`` for the probe-run preflight log
    setup. Lives at the package root because both consumers live in
    different submodules.
    """
    from aiperf.config import BenchmarkRun

    return BenchmarkRun(
        benchmark_id=benchmark_id or uuid4().hex[:12],
        cfg=config,
        trial=trial,
        artifact_dir=artifact_dir or config.artifacts.dir,
        random_seed=random_seed,
        variables=dict(variables or {}),
    )
