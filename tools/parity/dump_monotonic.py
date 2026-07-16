# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture the probe sequence the real `MonotonicSLASearchPlanner` produces.

The monotonic SLA planner (`--search-recipe max-concurrency-under-sla
--search-style monotonic`) is a dynamic ask-tell loop, so — unlike a static
sweep — its cell list is metric-dependent and cannot be golden-gated against
`dump_sweep`. Instead we drive the *production* planner with a synthetic,
deterministic feasibility oracle (`feasible iff concurrency <= boundary`) and
dump the exact `ask()` sequence plus the final bracket / convergence reason.
The native `MonotonicPlanner` must reproduce this list byte-for-byte for the
same oracle (see `rust/cli/tests/monotonic_parity.rs`).

Usage: python tools/parity/dump_monotonic.py
Emits a JSON object: {cases: [{lo, hi, max_iterations, stability_trials,
boundary, threshold, asks: [int], feasible_max, infeasible_min,
convergence_reason}]}.
"""

from __future__ import annotations

import shlex
import sys

import orjson

# Minimal argv producing a valid streaming chat BenchmarkConfig. The base config
# only seeds the planner's `_mutate_base`; the probe sequence is driven entirely
# by the synthetic oracle, so the concrete endpoint/model are immaterial.
_BASE_ARGS = "--model m --url http://localhost:8000 --endpoint-type chat --streaming"


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

# One SLA filter: TTFT p95 < threshold. The synthetic oracle drives feasibility
# purely off the swept concurrency, so the threshold is a fixed sentinel and the
# observed metric is threshold-1 (feasible) or threshold+1 (infeasible).
_THRESHOLD = 200.0

# (lo, hi, max_iterations, stability_trials, boundary) tuples exercising every
# branch: an interior boundary (probe->bisect), a boundary above hi
# (no_failure_in_range), a boundary below lo (no_pass_in_range), a tight bracket
# (precision), a budget-exhausting run (max_iterations), and a >1 stability
# window (re-ask until agreement).
_CASES: tuple[tuple[int, int, int, int, int], ...] = (
    (1, 1000, 20, 2, 255),
    (1, 1000, 20, 2, 10000),  # every probe passes -> no_failure_in_range
    (1, 1000, 20, 2, 0),  # every probe fails -> no_pass_in_range
    (1, 1000, 20, 2, 500),
    (1, 1000, 3, 2, 255),  # tiny budget -> max_iterations
    (1, 64, 20, 3, 33),  # stability window 3
    (1, 1000, 20, 1, 128),  # single-trial verdicts
)


def _run_case(
    lo: int, hi: int, max_iterations: int, stability_trials: int, boundary: int
) -> dict:
    from aiperf.common.enums import OptimizationDirection
    from aiperf.common.models.export_models import JsonMetricResult
    from aiperf.config.sweep import AdaptiveSearchSweep, Objective
    from aiperf.config.sweep.adaptive import SLAFilter, SearchSpaceDimension
    from aiperf.orchestrator.models import RunResult
    from aiperf.orchestrator.search_planner.monotonic import (
        MonotonicSLASearchPlanner,
    )

    cfg = AdaptiveSearchSweep(
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
        monotonic_stability_trials=stability_trials,
    )
    base = _base_config()
    planner = MonotonicSLASearchPlanner(base, cfg)

    asks: list[int] = []
    while not planner.is_converged():
        proposal = planner.ask()
        if proposal is None:
            break
        _cfg, variation = proposal
        value = int(variation.values["phases.profiling.concurrency"])
        asks.append(value)
        # Synthetic oracle: feasible iff concurrency <= boundary. Report a TTFT
        # p95 that satisfies (threshold-1) or breaches (threshold+1) the filter.
        observed = _THRESHOLD - 1.0 if value <= boundary else _THRESHOLD + 1.0
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
        "stability_trials": stability_trials,
        "boundary": boundary,
        "threshold": _THRESHOLD,
        "asks": asks,
        "feasible_max": planner.feasible_max,
        "infeasible_min": planner.infeasible_min,
        "convergence_reason": planner.convergence_reason(),
    }


def main() -> int:
    cases = [_run_case(*c) for c in _CASES]
    sys.stdout.buffer.write(orjson.dumps({"cases": cases}, option=orjson.OPT_INDENT_2))
    sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
