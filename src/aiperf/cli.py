# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main CLI entry point for the AIPerf system."""

################################################################################
# NOTE: Keep the imports here to a minimum. This file is read every time
# the CLI is run, including to generate the help text. Any imports here
# will cause a performance penalty during this process.
################################################################################

from typing import Literal

from cyclopts import App

from aiperf.cli_commands.plugins_cli import plugins_app
from aiperf.cli_utils import exit_on_error
from aiperf.common.config import ServiceConfig, UserConfig


def _get_help_text() -> str:
    """Generate help text with installed plugin information."""
    # Get aiperf version for the title
    from aiperf import __version__ as aiperf_version
    from aiperf.plugin import plugins

    packages = plugins.list_packages()
    plugin_list = []
    for pkg in packages:
        meta = plugins.get_package_metadata(pkg)
        plugin_list.append(f"{pkg} (v{meta.version})")

    plugins_str = ", ".join(plugin_list) if plugin_list else "none"
    return f"NVIDIA AIPerf v{aiperf_version} - AI Performance Benchmarking Tool\n\nInstalled Plugin Packages: {plugins_str}"


app = App(name="aiperf", help=_get_help_text(), help_on_error=True)

# Register --install-completion flag to install completion for the current shell
app.register_install_completion_command()

# Add plugins subcommand
app.command(plugins_app)


def _register_service_command() -> None:
    """Register the service command."""
    from aiperf.cli_commands.service import service_app

    app.command(service_app)


def _register_trace_commands() -> None:
    """Register trace analysis commands."""
    from aiperf.cli_commands.analyze_trace import analyze_app

    app.command(analyze_app)


# Register subcommands
_register_trace_commands()
_register_service_command()


@app.command(name="profile")
def profile(
    user_config: UserConfig,
    service_config: ServiceConfig | None = None,
) -> None:
    """Run the Profile subcommand.

    Benchmark generative AI models and measure performance metrics including throughput,
    latency, token statistics, and resource utilization.

    Examples:
        # Basic profiling with streaming
        aiperf profile --model Qwen/Qwen3-0.6B --url localhost:8000 --endpoint-type chat --streaming

        # Concurrency-based benchmarking
        aiperf profile --model your_model --url localhost:8000 --concurrency 10 --request-count 100

        # Request rate benchmarking (Poisson distribution)
        aiperf profile --model your_model --url localhost:8000 --request-rate 5.0 --benchmark-duration 60

        # Time-based benchmarking with grace period
        aiperf profile --model your_model --url localhost:8000 --benchmark-duration 300 --benchmark-grace-period 30

        # Custom dataset with fixed schedule replay
        aiperf profile --model your_model --url localhost:8000 --input-file trace.jsonl --fixed-schedule

        # Multi-turn conversations with ShareGPT dataset
        aiperf profile --model your_model --url localhost:8000 --public-dataset sharegpt --num-sessions 50

        # Goodput measurement with SLOs
        aiperf profile --model your_model --url localhost:8000 --goodput "request_latency:250 inter_token_latency:10"

    Args:
        user_config: User configuration for the benchmark
        service_config: Service configuration options
    """
    with exit_on_error(title="Error Running AIPerf System"):
        from aiperf.cli_runner import run_system_controller
        from aiperf.common.config.loader import load_service_config

        service_config = service_config or load_service_config()
        run_system_controller(user_config, service_config)


@app.command(name="plot")
def plot(
    paths: list[str] | None = None,
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
