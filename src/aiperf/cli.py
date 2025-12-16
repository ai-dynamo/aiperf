# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main CLI entry point for the AIPerf system."""

################################################################################
# NOTE: Keep the imports here to a minimum. This file is read every time
# the CLI is run, including to generate the help text. Any imports here
# will cause a performance penalty during this process.
################################################################################

from cyclopts import App

from aiperf.cli_utils import exit_on_error
from aiperf.common.config import ServiceConfig, UserConfig

app = App(name="aiperf", help="NVIDIA AIPerf")


@app.command(name="profile")
def profile(
    user_config: UserConfig,
    service_config: ServiceConfig | None = None,
) -> None:
    """Run the Profile subcommand.

    Args:
        user_config: User configuration for the benchmark
        service_config: Service configuration options
    """
    with exit_on_error(title="Error Running AIPerf System"):
        from aiperf.cli_runner import run_system_controller
        from aiperf.common.config import load_service_config

        service_config = service_config or load_service_config()
        run_system_controller(user_config, service_config)


@app.command(name="plot")
def plot(
    paths: list[str] | None = None,
    output: str | None = None,
    theme: str = "light",
    config: str | None = None,
    verbose: bool = False,
    dashboard: bool = False,
    port: int = 8050,
) -> None:
    """Generate visualizations from AIPerf profiling data.

    On first run, automatically creates ~/.aiperf/plot_config.yaml which you can edit to
    customize plots, including experiment classification (baseline vs treatment runs).
    Use --config to specify a different config file.

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
        port: Port for dashboard server (only used with --dashboard). Defaults to 8050.
    """
    with exit_on_error(title="Error Running Plot Command", show_traceback=verbose):
        from aiperf.plot.cli_runner import run_plot_controller

        run_plot_controller(
            paths,
            output,
            theme=theme,
            config=config,
            verbose=verbose,
            dashboard=dashboard,
            port=port,
        )
