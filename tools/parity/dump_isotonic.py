# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture the probe sequence the real `SmoothIsotonicSLAPlanner` produces.

Like `dump_monotonic.py` but for the default `--search-style smooth_isotonic`.
Drives the *production* planner with a synthetic deterministic per-x margin
oracle so the fit (scipy PAVA+PCHIP) is exercised end-to-end. Single-trial
probes keep per-x variance at zero, so the replicate budget is always 0 and the
non-deterministic `scipy.stats.bootstrap` branch is never reached — making the
bracket/fit/cliff probe sequence fully reproducible.

The native planner (which delegates the same scipy fit via pyo3) must reproduce
this `ask()` sequence byte-for-byte (`rust/cli/tests/isotonic_parity.rs`).

Usage: python tools/parity/dump_isotonic.py
Emits {cases: [{lo, hi, max_iterations, boundary, slope, threshold, asks: [int],
feasible_max, infeasible_min, convergence_reason, boundary_type,
binding_constraint}]}.
"""

from __future__ import annotations

import shlex
import sys

import orjson

_THRESHOLD = 200.0
_BASE_ARGS = "--model m --url http://localhost:8000 --endpoint-type chat --streaming"

# (lo, hi, max_iterations, boundary, slope) — a linear TTFT model
# ttft(x) = threshold + slope*(x - boundary), so ttft < threshold iff x <
# boundary: a smooth (non-cliff) monotone crossing the SLA at `boundary`.
_CASES: tuple[tuple[int, int, int, int, float], ...] = (
    (1, 1000, 30, 255, 0.5),
    (1, 1000, 30, 100, 1.0),
    (1, 1000, 30, 10000, 0.5),  # never breaches -> no_failure_in_range
    (1, 1000, 30, 0, 0.5),  # always breaches -> no_pass_in_range
    (1, 512, 30, 40, 2.0),
)


def _base_config():
    from aiperf.cli import app
    from aiperf.config.flags.resolver import resolve_config
    from aiperf.config.loader import build_benchmark_plan

    _command, bound, _ = app.parse_args(
        ["profile", *shlex.split(_BASE_ARGS)], exit_on_error=False
    )
    cli_config = bound.arguments["cli_config"]
    config = resolve_config(cli_config, cli_config.config_file)
    return build_benchmark_plan(config).configs[0]


def _run_case(
    lo: int, hi: int, max_iterations: int, boundary: int, slope: float
) -> dict:
    from aiperf.common.enums import OptimizationDirection
    from aiperf.common.models.export_models import JsonMetricResult
    from aiperf.config.sweep import AdaptiveSearchSweep, Objective
    from aiperf.config.sweep.adaptive import SLAFilter, SearchSpaceDimension
    from aiperf.orchestrator.models import RunResult
    from aiperf.orchestrator.search_planner.smooth_isotonic import (
        SmoothIsotonicSLAPlanner,
    )

    cfg = AdaptiveSearchSweep(
        planner="smooth_isotonic",
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=lo, hi=hi, kind="int"
            )
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        max_iterations=max_iterations,
        n_initial_points=1,
        sla_filters=[
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=_THRESHOLD,
            )
        ],
    )
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)

    asks: list[int] = []
    while not planner.is_converged():
        proposal = planner.ask()
        if proposal is None:
            break
        _cfg, variation = proposal
        value = int(variation.values["phases.profiling.concurrency"])
        asks.append(value)
        # Linear synthetic TTFT: crosses the SLA exactly at `boundary`.
        observed = _THRESHOLD + slope * (value - boundary)
        result = RunResult(
            label=variation.label,
            success=True,
            summary_metrics={
                "time_to_first_token": JsonMetricResult(unit="ms", p95=observed)
            },
        )
        planner.tell(variation, [result])

    return {
        "lo": lo,
        "hi": hi,
        "max_iterations": max_iterations,
        "boundary": boundary,
        "slope": slope,
        "threshold": _THRESHOLD,
        "asks": asks,
        "feasible_max": planner.feasible_max,
        "infeasible_min": planner.infeasible_min,
        "convergence_reason": planner.convergence_reason(),
        "boundary_type": planner.boundary_type,
        "binding_constraint": planner.binding_constraint,
    }


def main() -> int:
    cases = [_run_case(*c) for c in _CASES]
    sys.stdout.buffer.write(orjson.dumps({"cases": cases}, option=orjson.OPT_INDENT_2))
    sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
