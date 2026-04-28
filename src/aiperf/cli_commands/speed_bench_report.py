# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for assembling SPEED-Bench matrix reports."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

app = App(name="speed-bench-report")


@app.default
def speed_bench_report(
    paths: list[Path],
    output: Path = Path("speed_bench_report.csv"),
    output_format: Annotated[
        Literal["csv", "table", "both"],
        Parameter(name=["--format"]),
    ] = "both",
    metric: Literal["accept_length", "accept_rate", "throughput"] = "accept_length",
) -> None:
    """Assemble per-category SPEED-Bench aiperf results into a matrix report.

    Run ``aiperf profile`` once per SPEED-Bench category, then point this command
    at the output directories to produce a matrix matching the SPEED-Bench paper format.

    Examples:
        # Scan a parent directory for per-category run subdirectories
        aiperf speed-bench-report ./artifacts/

        # List run directories explicitly
        aiperf speed-bench-report ./artifacts/run_coding/ ./artifacts/run_math/

        # Acceptance rate matrix (accepted / draft tokens)
        aiperf speed-bench-report ./artifacts/ --metric accept_rate

        # Throughput matrix (output tokens/sec per category)
        aiperf speed-bench-report ./artifacts/ --metric throughput

    Args:
        paths: Run directories or parent directories containing run subdirectories.
        output: Output CSV file path. Defaults to ./speed_bench_report.csv.
        output_format: Output format - 'csv', 'table', or 'both'. Defaults to 'both'.
        metric: Which metric to report - 'accept_length', 'accept_rate', or 'throughput'.
            Defaults to 'accept_length'.
    """
    from aiperf.analysis.speed_bench_report import (
        SpeedBenchReportError,
        generate_report,
    )

    try:
        generate_report(
            paths, output=output, output_format=output_format, metric=metric
        )
    except SpeedBenchReportError as e:
        print(f"Error: {e}.", file=sys.stderr)
        sys.exit(1)
