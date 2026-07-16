# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture the exact protocol-v2 request ``aiperf profile <args>`` sends the runner.

Single-run only. Reuses the production ``resolve_config`` -> ``build_benchmark_plan``
-> single-run ``BenchmarkRun`` construction -> ``build_authored_run_request`` path so
the captured JSON is byte-identical to what the Python orchestrator hands
``aiperf``. The runner request is the golden vector the native Rust CLI
must reproduce (runner-consumed projection; see the plan's parity mechanism).

Determinism: ``benchmark_id`` is pinned (Python defaults it to a random uuid) so
the golden is stable across regenerations. ``artifact_dir`` comes from the
resolved config (fixtures pin it via ``--artifact-dir``).

Usage:
    python tools/parity/dump_golden.py <fixture.args>

A ``.args`` fixture is one shell-word-split argv line (no ``aiperf``/``profile``
prefix).
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import orjson

# Pinned so the golden vector is reproducible; the runner does not interpret it.
_PARITY_BENCHMARK_ID = "parity-benchmark"


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: dump_golden.py <fixture.args>\n")
        return 2
    argv = shlex.split(Path(sys.argv[1]).read_text().strip())

    from aiperf.cli import app
    from aiperf.config import BenchmarkRun
    from aiperf.config.flags.resolver import resolve_config
    from aiperf.config.loader import build_benchmark_plan
    from aiperf.orchestrator.orchestrator import resolve_run_seed
    from aiperf.orchestrator.rust_wire import build_authored_run_request

    # Drive cyclopts' real parser over ["profile", *argv] to obtain the exact
    # CLIConfig `aiperf profile <argv>` would bind — without executing the run.
    _command, bound, _ = app.parse_args(["profile", *argv], exit_on_error=False)
    cli_config = bound.arguments["cli_config"]

    # Mirror profile.py: --sketch-metrics toggles the env-backed runtime setting
    # that rust_wire reads when projecting metrics/artifacts.
    if getattr(cli_config, "sketch_metrics", False):
        from aiperf.common.environment import Environment
        Environment.METRICS.SKETCH = True

    config = resolve_config(cli_config, cli_config.config_file)
    plan = build_benchmark_plan(config)
    if not plan.is_single_run:
        sys.stderr.write(
            f"fixture resolves to {len(plan.configs)} configs x {plan.trials} trials; "
            "the oracle is single-run only\n"
        )
        return 3

    run = BenchmarkRun(
        benchmark_id=_PARITY_BENCHMARK_ID,
        cfg=plan.configs[0],
        trial=0,
        artifact_dir=plan.configs[0].artifacts.dir,
        random_seed=resolve_run_seed(plan, plan.variations[0]),
        variables=dict(plan.variables or {}),
    )
    request = build_authored_run_request(run, operation="execute")

    sys.stdout.buffer.write(
        orjson.dumps(request, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2)
    )
    sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
