# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry points for executing a BenchmarkPlan from the CLI.

Public surface:

* :func:`run_benchmark` - dispatch single-run vs multi-run by plan shape.
* :class:`CompletedRun` / :data:`OnComplete` - post-run callback contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from aiperf.cli_runner._callbacks import CompletedRun, OnComplete
from aiperf.cli_runner._multi_run import _run_multi_benchmark
from aiperf.cli_runner._preflight import (
    _preflight_accuracy_deps,
    _preflight_artifact_dir,
    _preflight_fd_limit,
)
from aiperf.cli_runner._single_run import _run_single_benchmark

if TYPE_CHECKING:
    from aiperf.config import BenchmarkConfig, BenchmarkPlan, BenchmarkRun


__all__ = [
    "CompletedRun",
    "OnComplete",
    "run_benchmark",
]

# Human-facing pointer appended to every native-only rejection message.
_USE_NATIVE_HINT = (
    "This configuration is only supported by the native `aiperf` binary. "
    "Run `aiperf profile ...` directly instead of `python -m aiperf.cli profile ...`."
)


def _reject_native_only_configs(plan: BenchmarkPlan) -> None:
    """Fail fast if the plan needs a capability the Python service mesh lacks.

    The Python frontend runs only the pure-Python service mesh (HTTP transport,
    single process). Native-only capabilities are rejected here pointing at the
    native ``aiperf`` binary rather than silently degrading:

    - non-HTTP transports (gRPC / dynosim) and explicit native workload
      selection are rejected structurally at config load -- the Config-v2 schema
      no longer models a ``transport``/``workload`` block, so those keys fail
      ``extra="forbid"`` validation before this guard runs;
    - multi-cell partitioning (``runtime.cells > 1``) and bounded-memory metric
      sketches (``--sketch-metrics``) are still valid config, so they are
      rejected here.

    This guard lives on the local ``run_benchmark`` entry, NOT in shared
    ``BenchmarkConfig`` validation, so the native cellular role (which resolves
    the same config and drives the native binary) is unaffected.
    """
    from aiperf.common.environment import Environment
    from aiperf.config.loader.errors import ConfigurationError

    for config in plan.configs:
        cells = getattr(config.runtime, "cells", 1) or 1
        if cells > 1:
            raise ConfigurationError(
                f"runtime.cells={cells} (cellular execution) is not supported by the "
                f"Python engine. {_USE_NATIVE_HINT}"
            )

    if Environment.METRICS.SKETCH:
        raise ConfigurationError(
            "--sketch-metrics (bounded-memory metric retention) is not supported by "
            f"the Python engine. {_USE_NATIVE_HINT}"
        )


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

    _reject_native_only_configs(plan)

    _preflight_artifact_dir(plan)
    _preflight_accuracy_deps(plan)
    _preflight_fd_limit()
    # Do not probe endpoint readiness here: Python cannot synthesize a valid
    # request for runner-owned dialect IDs. The selected native adapter owns
    # validation, normalization, readiness, and the first inference request.

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
        from aiperf.orchestrator.orchestrator import resolve_run_seed

        seed = resolve_run_seed(plan, plan.variations[0])
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
