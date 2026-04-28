# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for generating visualizations from AIPerf profiling data."""

from typing import Literal

from cyclopts import App

app = App(name="plot")


@app.default
def plot(
    paths: list[str] | None = None,
    *,
    output: str | None = None,
    theme: Literal["light", "dark"] = "light",
    config: str | None = None,
    verbose: bool = False,
    dashboard: bool = False,
    host: str = "127.0.0.1",
    port: int = 8050,
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

    Args:
        paths: Paths to profiling run directories. Defaults to ./artifacts if not specified.
        output: Directory to save generated plots. Defaults to <first_path>/plots if not specified.
        theme: Plot theme to use: 'light' (white background) or 'dark' (dark background). Defaults to 'light'.
        config: Path to custom plot configuration YAML file. If not specified, auto-creates and uses ~/.aiperf/plot_config.yaml.
        verbose: Show detailed error tracebacks in console (errors are always logged to ~/.aiperf/plot.log).
        dashboard: Launch interactive dashboard server instead of generating static PNGs.
        host: Host for dashboard server (only used with --dashboard). Defaults to 127.0.0.1.
        port: Port for dashboard server (only used with --dashboard). Defaults to 8050.
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
