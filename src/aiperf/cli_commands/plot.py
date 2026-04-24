# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for generating visualizations from AIPerf profiling data."""

from typing import Annotated, Literal

from cyclopts import App, Parameter

app = App(name="plot")


@app.default
def plot(
    *,
    paths: Annotated[
        list[str] | None,
        Parameter(
            help="Paths to profiling run directories. Defaults to ./artifacts if not specified."
        ),
    ] = None,
    output: Annotated[
        str | None,
        Parameter(
            help="Directory to save generated plots. Defaults to <first_path>/plots if not specified."
        ),
    ] = None,
    theme: Annotated[
        Literal["light", "dark"],
        Parameter(
            help="Plot theme: 'light' (white background) or 'dark' (dark background)."
        ),
    ] = "light",
    config: Annotated[
        str | None,
        Parameter(
            help="Path to custom plot configuration YAML file. If not specified, auto-creates and uses ~/.aiperf/plot_config.yaml."
        ),
    ] = None,
    verbose: Annotated[
        bool,
        Parameter(
            help="Show detailed error tracebacks in console (errors are always logged to ~/.aiperf/plot.log)."
        ),
    ] = False,
    dashboard: Annotated[
        bool,
        Parameter(
            help="Launch interactive dashboard server instead of generating static PNGs."
        ),
    ] = False,
    host: Annotated[
        str, Parameter(help="Host for dashboard server (only used with --dashboard).")
    ] = "127.0.0.1",
    port: Annotated[
        int, Parameter(help="Port for dashboard server (only used with --dashboard).")
    ] = 8050,
) -> None:
    """Generate visualizations from AIPerf profiling data.

    On first run, automatically creates ~/.aiperf/plot_config.yaml which you can edit to
    customize plots, including experiment classification (baseline vs treatment runs).
    Use --config to specify a different config file.

    _**Note:** PNG export requires Chrome or Chromium to be installed on your system, as it is used by kaleido to render Plotly figures to static images._

    _**Note:** The plot command expects default export filenames (e.g., `profile_export.jsonl`). Runs created with `--profile-export-file` or custom `--profile-export-prefix` use different filenames and will not be detected by the plot command._

    Examples:
        # Generate plots (auto-creates ~/.aiperf/plot_config.yaml on first run)
        aiperf plot

        # Use custom config
        aiperf plot --config my_plots.yaml

        # Show detailed error tracebacks
        aiperf plot --verbose
    """
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title="Error Running Plot Command", show_traceback=verbose):
        from aiperf.plot.cli_runner import run_plot_controller

        run_plot_controller(
            paths,
            output,
            theme=theme,
            config=config,
            verbose=verbose,
            dashboard=dashboard,
            host=host,
            port=port,
        )
