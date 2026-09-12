# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for scraping Prometheus metrics into Tachometer artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

app = App(name="watch")


@app.default
def watch(
    *,
    config: Annotated[
        Path | None,
        Parameter(
            name="--config",
            help="Tachometer TOML config file. Mutually exclusive with --endpoint.",
        ),
    ] = None,
    endpoints: Annotated[
        list[str] | None,
        Parameter(
            name="--endpoint",
            help="Prometheus endpoint in NAME=URL form. May be repeated.",
        ),
    ] = None,
    frequency: Annotated[
        float,
        Parameter(name="--freq", help="Polling frequency in Hz for --endpoint mode."),
    ] = 0.2,
    storage: Annotated[
        str | None,
        Parameter(
            name="--storage",
            help="Output storage path for --endpoint mode.",
        ),
    ] = None,
    rows_per_parquet: Annotated[
        int,
        Parameter(
            name="--rows-per-parquet",
            help="Number of rows per intermediate parquet file.",
        ),
    ] = 1_000_000,
    save_interval_secs: Annotated[
        int,
        Parameter(
            name="--save-interval",
            help="Interval in seconds between Arrow checkpoint saves.",
        ),
    ] = 5,
    filters: Annotated[
        list[str] | None,
        Parameter(name="--filter", help="Metric filter name. May be repeated."),
    ] = None,
    local_dir: Annotated[
        Path | None,
        Parameter(
            name="--local-dir",
            help="Local directory for intermediate Tachometer files.",
        ),
    ] = None,
    sync_interval_secs: Annotated[
        int,
        Parameter(
            name="--sync-interval",
            help="Interval in seconds between remote sync attempts.",
        ),
    ] = 0,
) -> None:
    """Watch Prometheus endpoints and write Tachometer parquet artifacts.

    This command delegates the scraper runtime to the vendored Rust
    implementation. A config file is the preferred interface for production
    runs; the direct endpoint flags remain available for debugging.
    """
    from aiperf._tachometer import run_tachometer_cli
    from aiperf.cli_utils import exit_on_error

    args: list[str] = []
    if config is not None:
        args.extend(["--config", str(config)])
    for endpoint in endpoints or []:
        args.extend(["--endpoint", endpoint])
    args.extend(["--freq", str(frequency)])
    if storage is not None:
        args.extend(["--storage", storage])
    args.extend(["--rows-per-parquet", str(rows_per_parquet)])
    args.extend(["--save-interval", str(save_interval_secs)])
    for filter_name in filters or []:
        args.extend(["--filter", filter_name])
    if local_dir is not None:
        args.extend(["--local-dir", str(local_dir)])
    args.extend(["--sync-interval", str(sync_interval_secs)])

    with exit_on_error(title="Error Running AIPerf Watch", show_traceback=False):
        run_tachometer_cli(args)
