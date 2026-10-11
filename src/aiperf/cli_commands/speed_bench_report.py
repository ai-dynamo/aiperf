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
    *,
    output: Path = Path("speed_bench_report.csv"),
    output_format: Annotated[
        Literal["csv", "table", "both"],
        Parameter(name=["--format"]),
    ] = "both",
    metric: Literal["accept_length", "accept_rate", "throughput"] = "accept_length",
    source: Literal["auto", "records", "summary", "server"] = "auto",
) -> None:
    """Assemble SPEED-Bench aiperf results into a matrix report.

    Point this command at one or more ``aiperf profile`` output directories. A
    single run over an aggregate SPEED-Bench split (``speed_bench_qualitative``)
    already carries every category, and each of its per-request records is
    stamped with the category it came from, so one run produces the whole
    matrix. Runs over a single category still work and contribute one column
    each.

    Examples:
        # One run over all categories: matrix comes from its per-request records
        aiperf speed-bench-report ./artifacts/speed_bench_qualitative/

        # Scan a parent directory for per-category run subdirectories
        aiperf speed-bench-report ./artifacts/

        # List run directories explicitly
        aiperf speed-bench-report ./artifacts/run_coding/ ./artifacts/run_math/

        # Acceptance rate matrix (accepted / draft tokens)
        aiperf speed-bench-report ./artifacts/ --metric accept_rate

        # Throughput matrix (output tokens/sec per run)
        aiperf speed-bench-report ./artifacts/ --metric throughput

        # Force the Prometheus scrape, ignoring any per-request records
        aiperf speed-bench-report ./artifacts/ --source server

    Args:
        paths: Run directories or parent directories containing run subdirectories.
        output: Output CSV file path. Defaults to ./speed_bench_report.csv.
        output_format: Output format - 'csv', 'table', or 'both'. Defaults to 'both'.
        metric: Which metric to report - 'accept_length', 'accept_rate', or 'throughput'.
            Defaults to 'accept_length'.
        source: Where acceptance numbers come from - 'records' (per-request
            ``profile_export.jsonl``), 'summary' (the same per-request data reduced
            to run-level scalars in ``profile_export_aiperf.json``, the only source
            left at ``--export-level summary``), 'server' (Prometheus scrape in
            ``server_metrics_export.json``), or 'auto' to try them in that order.
            Only 'records' can split one run into per-category columns; only
            'server' reads server-side data. Defaults to 'auto'.
    """
    from aiperf.analysis.speed_bench_report import (
        SpeedBenchReportError,
        generate_report,
    )

    try:
        generate_report(
            paths,
            output=output,
            output_format=output_format,
            metric=metric,
            source=source,
        )
    except SpeedBenchReportError as e:
        print(f"Error: {e}.", file=sys.stderr)
        sys.exit(1)
