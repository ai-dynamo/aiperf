# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dashboard server for AIPerf interactive visualization.

This module provides the main Dash application server that hosts
the interactive dashboard with all features.
"""

import os
import signal
import webbrowser
from pathlib import Path

import dash
import dash_bootstrap_components as dbc

from aiperf.plot.config import PlotConfig
from aiperf.plot.constants import PlotTheme
from aiperf.plot.core.data_loader import DataLoader, RunData
from aiperf.plot.core.mode_detector import VisualizationMode
from aiperf.plot.dashboard.builder import DashboardBuilder
from aiperf.plot.dashboard.callbacks import register_all_callbacks
from aiperf.plot.dashboard.styling import get_all_themes_css


class DashboardServer:
    """
    Main dashboard server using Plotly Dash.

    This class initializes and runs the interactive dashboard,
    handling layout generation, callback registration, and server lifecycle.

    Args:
        runs: List of RunData objects to visualize
        mode: Visualization mode (single-run or multi-run)
        theme: Plot theme (light or dark)
        plot_config: Plot configuration object
        port: Port number for the server

    Example:
        >>> server = DashboardServer(runs=runs, mode=VisualizationMode.MULTI_RUN,
        ...                          theme=PlotTheme.DARK, plot_config=config, port=8050)
        >>> server.run()
    """

    def __init__(
        self,
        runs: list[RunData],
        run_dirs: list[Path],
        mode: VisualizationMode,
        theme: PlotTheme,
        plot_config: PlotConfig,
        loader: DataLoader,
        port: int = 8050,
    ):
        """Initialize the dashboard server."""
        self.runs = runs
        self.run_dirs = run_dirs
        self.mode = mode
        self.theme = theme
        self.plot_config = plot_config
        self.loader = loader
        self.port = port

        # Initialize Dash app with Bootstrap theme
        external_stylesheets = [
            dbc.themes.BOOTSTRAP if theme == PlotTheme.LIGHT else dbc.themes.DARKLY
        ]
        self.app = dash.Dash(
            __name__,
            external_stylesheets=external_stylesheets,
            suppress_callback_exceptions=True,
            title="AIPerf Dashboard",
        )

        # Inject custom CSS
        self.app.index_string = self._get_index_html()

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _get_index_html(self) -> str:
        """Generate HTML template with custom CSS."""
        custom_css = get_all_themes_css()

        return f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            {custom_css}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C."""
        print("\n\nShutting down dashboard...")
        exit(0)

    def build_layout(self):
        """Build dashboard layout using DashboardBuilder."""
        builder = DashboardBuilder(
            runs=self.runs,
            mode=self.mode,
            theme=self.theme,
            plot_config=self.plot_config,
        )

        return builder.build()

    def register_callbacks(self):
        """Register all Dash callbacks."""
        register_all_callbacks(
            self.app,
            self.runs,
            self.run_dirs,
            self.mode,
            self.theme,
            self.plot_config,
            self.loader,
        )

    def run(self):
        """
        Start the Dash server.

        This method will block until the server is stopped (Ctrl+C).
        Opens browser automatically on startup.
        """
        # Build layout
        self.app.layout = self.build_layout()

        # Register callbacks
        self.register_callbacks()

        # Print startup message
        url = f"http://127.0.0.1:{self.port}"
        mode_name = self.mode.value.replace("_", "-")
        run_word = "run" if len(self.runs) == 1 else "runs"

        print(f"\nDashboard ready: {url}")
        print(
            f"Mode: {mode_name} ({len(self.runs)} {run_word}), Theme: {self.theme.value}"
        )
        print("Press Ctrl+C to stop\n")

        # Auto-open browser (suppress stderr to avoid WSL2 gio warnings)
        try:
            # Redirect stderr to suppress gio warnings in WSL
            with open(os.devnull, "w") as devnull:
                old_stderr = os.dup(2)
                os.dup2(devnull.fileno(), 2)

                try:
                    webbrowser.open(url)
                finally:
                    # Restore stderr
                    os.dup2(old_stderr, 2)
                    os.close(old_stderr)
        except Exception:
            print(f"Please open {url} manually in your browser\n")

        # Start server (this blocks until Ctrl+C)
        self.app.run(debug=False, host="127.0.0.1", port=self.port, use_reloader=False)
