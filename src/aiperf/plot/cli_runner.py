# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI runner for plot command."""

from pathlib import Path

from aiperf.plot.constants import PLOT_LOG_FILE, PlotMode, PlotTheme
from aiperf.plot.plot_controller import PlotController


def run_plot_controller(
    paths: list[str] | None = None,
    output: str | None = None,
    mode: PlotMode | str = PlotMode.PNG,
    theme: PlotTheme | str = PlotTheme.LIGHT,
    config: str | None = None,
    verbose: bool = False,
    dashboard: bool = False,
    host: str = "127.0.0.1",
    port: int = 8050,
) -> None:
    """Generate plots from AIPerf profiling data.

    Note: PNG export requires Chrome or Chromium to be installed on your system,
    as it is used by kaleido to render Plotly figures to static images.

    Args:
        paths: Paths to profiling run directories. Defaults to ./artifacts if not specified.
        output: Directory to save generated plots. Defaults to <first_path>/plots if not specified.
        mode: Output mode for plots. Defaults to PNG.
        theme: Plot theme to use (LIGHT or DARK). Defaults to LIGHT.
        config: Path to custom plot configuration YAML file. If not specified, uses default config.
        verbose: Show detailed error tracebacks in console.
        dashboard: Launch interactive dashboard server instead of generating static PNGs.
        host: Host for dashboard server (only used with --dashboard). Defaults to 127.0.0.1.
        port: Port for dashboard server (only used with dashboard=True). Defaults to 8050.
    """
    input_paths = paths or ["./artifacts"]
    input_paths = [Path(p) for p in input_paths]

    output_dir = Path(output) if output else input_paths[0] / "plots"

    # Override mode if dashboard flag is set
    if dashboard:
        mode = PlotMode.DASHBOARD

    if isinstance(mode, str):
        mode = PlotMode(mode.lower())
    if isinstance(theme, str):
        theme = PlotTheme(theme.lower())

    config_path = Path(config) if config else None

    controller = PlotController(
        paths=input_paths,
        output_dir=output_dir,
        mode=mode,
        theme=theme,
        config_path=config_path,
        verbose=verbose,
        host=host,
        port=port,
    )

    result = controller.run()

    # Only print file count for non-dashboard modes
    if mode != PlotMode.DASHBOARD:
        plot_word = "plot" if len(result) == 1 else "plots"
        print(f"\nSaved {len(result)} {plot_word} to: {output_dir}")
    print(f"Logs: {output_dir / PLOT_LOG_FILE}")
