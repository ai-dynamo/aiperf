# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture the per-cell protocol-v2 requests a sweeping `aiperf profile` produces.

Reuses the production `resolve_config` -> `build_benchmark_plan` -> per-variation
`BenchmarkRun` construction (`_resolve_artifact_dir` / `resolve_run_seed`) so the
dumped list is byte-identical to what the Python orchestrator would hand each
`aiperf-runner` cell. The native Rust sweep engine must reproduce this list.

Usage: python tools/parity/dump_sweep.py <fixture.args>
Emits a JSON object: {sweep_id_present, trials, cells: [{index, trial, label,
dir_name, artifact_dir, request}]}. `sweep_id` values are normalized to a fixed
sentinel (Python defaults it to a random uuid) so the golden is stable.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import orjson

_PARITY_BENCHMARK_ID = "parity-benchmark"
_PARITY_SWEEP_ID = "parity-sweep"


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: dump_sweep.py <fixture.args>\n")
        return 2
    argv = shlex.split(Path(sys.argv[1]).read_text().strip())

    from aiperf.cli import app
    from aiperf.config import BenchmarkRun
    from aiperf.config.flags.resolver import resolve_config
    from aiperf.config.loader import build_benchmark_plan
    from aiperf.orchestrator.orchestrator import (
        _bind_artifact_dir,
        _resolve_artifact_dir,
        resolve_run_seed,
    )
    from aiperf.orchestrator.rust_wire import build_authored_run_request

    _command, bound, _ = app.parse_args(["profile", *argv], exit_on_error=False)
    cli_config = bound.arguments["cli_config"]
    if getattr(cli_config, "sketch_metrics", False):
        from aiperf.common.environment import Environment

        Environment.METRICS.SKETCH = True

    config = resolve_config(cli_config, cli_config.config_file)
    plan = build_benchmark_plan(config)
    base = plan.configs[0].artifacts.dir

    cells = []
    for trial in range(plan.trials):
        for idx, (cfg, variation) in enumerate(zip(plan.configs, plan.variations)):
            artifact_dir = _resolve_artifact_dir(base, plan, variation, trial)
            cfg_bound = _bind_artifact_dir(cfg, artifact_dir)
            seed = resolve_run_seed(plan, variation, trial)
            run = BenchmarkRun(
                benchmark_id=_PARITY_BENCHMARK_ID,
                cfg=cfg_bound,
                trial=trial,
                sweep_id=_PARITY_SWEEP_ID,
                variation=variation,
                artifact_dir=artifact_dir,
                random_seed=seed,
                variables=dict(plan.variables or {}),
            )
            request = build_authored_run_request(run, operation="execute")
            cells.append(
                {
                    "index": idx,
                    "trial": trial,
                    "label": variation.label,
                    "dir_name": variation.dir_name,
                    "artifact_dir": str(artifact_dir),
                    "random_seed": seed,
                    "request": request,
                }
            )

    out = {
        "trials": plan.trials,
        "is_sweep": plan.is_sweep,
        "cells": cells,
    }
    sys.stdout.buffer.write(
        orjson.dumps(out, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2)
    )
    sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
