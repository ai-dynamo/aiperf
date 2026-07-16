# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture the probe sequence the real `OptunaSearchPlanner` produces (TPE).

For `--search-style bo|optuna`. Drives the *production* planner with a
deterministic synthetic oracle and the seeded `tpe` sampler (optuna-core, no
torch) so the suggestion sequence is byte-reproducible: optuna's TPE is a
seeded numpy-RNG sampler, so the same seed + same (objective, constraint) tells
yield the same suggestions in any process.

The native planner (`crate::bayes::OptunaPlanner`) drives the SAME optuna study
via pyo3 and must reproduce this `ask()` sequence byte-for-byte
(`rust/cli/tests/bayes_parity.rs`).

(The default `botorch` GP path is exercised e2e but NOT golden-gated: its
torch-based acquisition is not guaranteed byte-reproducible across processes.)

Usage: python tools/parity/dump_bayes.py
Emits {cases: [{seed, lo, hi, max_iterations, n_initial_points, ttft_slope,
threshold, asks: [int], convergence_reason}]}.
"""

from __future__ import annotations

import shlex
import sys

import orjson

_THRESHOLD = 200.0
_BASE_ARGS = "--model m --url http://localhost:8000 --endpoint-type chat --streaming"

# (seed, lo, hi, max_iterations, n_initial_points, ttft_slope)
_CASES: tuple[tuple[int, int, int, int, int, float], ...] = (
    (7, 1, 1000, 12, 5, 0.5),
    (123, 1, 1000, 15, 5, 1.0),
    (42, 1, 512, 10, 4, 2.0),
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
    seed: int, lo: int, hi: int, max_iterations: int, n_initial: int, ttft_slope: float
) -> dict:
    from aiperf.common.enums import OptimizationDirection
    from aiperf.common.models.export_models import JsonMetricResult
    from aiperf.config.sweep import AdaptiveSearchSweep, Objective
    from aiperf.config.sweep.adaptive import SLAFilter, SearchSpaceDimension
    from aiperf.orchestrator.models import RunResult
    from aiperf.orchestrator.search_planner.optuna_planner import OptunaSearchPlanner

    cfg = AdaptiveSearchSweep(
        planner="optuna",
        optuna_sampler="tpe",
        random_seed=seed,
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency",
                lo=lo,
                hi=hi,
                kind="int",
                prior="log-uniform",
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
        n_initial_points=n_initial,
        sla_filters=[
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=_THRESHOLD,
            )
        ],
    )
    planner = OptunaSearchPlanner(_base_config(), cfg)

    asks: list[int] = []
    while not planner.is_converged():
        proposal = planner.ask()
        if proposal is None:
            break
        _cfg, variation = proposal
        value = int(variation.values["phases.profiling.concurrency"])
        asks.append(value)
        # Deterministic oracle: throughput grows with concurrency; TTFT is
        # linear and crosses the SLA threshold as concurrency rises.
        throughput = float(value) * 10.0
        ttft = _THRESHOLD - 50.0 + ttft_slope * float(value)
        result = RunResult(
            label=variation.label,
            success=True,
            summary_metrics={
                "output_token_throughput": JsonMetricResult(
                    unit="tokens/sec", avg=throughput
                ),
                "time_to_first_token": JsonMetricResult(unit="ms", p95=ttft),
            },
        )
        planner.tell(variation, [result])

    return {
        "seed": seed,
        "lo": lo,
        "hi": hi,
        "max_iterations": max_iterations,
        "n_initial_points": n_initial,
        "ttft_slope": ttft_slope,
        "threshold": _THRESHOLD,
        "asks": asks,
        "convergence_reason": planner.convergence_reason(),
    }


def main() -> int:
    cases = [_run_case(*c) for c in _CASES]
    sys.stdout.buffer.write(orjson.dumps({"cases": cases}, option=orjson.OPT_INDENT_2))
    sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
