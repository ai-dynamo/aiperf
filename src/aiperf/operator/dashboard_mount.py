# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dashboard mount for embedding Plotly Dash inside the results server.

Provides:
- DashboardProxy: Mutable WSGI wrapper for hot-swapping the Dash app
- build_dashboard: Factory that scans PVC results and builds a Dash app
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import dash
    from _typeshed.wsgi import StartResponse, WSGIApplication, WSGIEnvironment

logger = logging.getLogger(__name__)


class DashboardProxy:
    """Mutable WSGI proxy for hot-swapping the inner Dash app.

    Mount this once via WSGIMiddleware. On refresh, swap ``self.app``
    to point at the new Dash server -- the outer route stays stable.

    Args:
        app: Initial WSGI callable (e.g. ``dash_app.server``).
    """

    def __init__(self, app: WSGIApplication) -> None:
        self.app = app

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> Iterable[bytes]:
        return self.app(environ, start_response)


def build_dashboard(results_dir: Path) -> tuple[dash.Dash | None, int]:
    """Build a Dash app from PVC results.

    Scans ``results_dir`` recursively for run directories (supporting
    .zst compressed marker files), loads run data, and constructs a
    Dash application mounted at ``/dashboard/``.

    Args:
        results_dir: Root of the results PVC (e.g. ``/data``).

    Returns:
        Tuple of (dash_app, run_count). dash_app is None if no runs found.
    """
    import dash
    import dash_bootstrap_components as dbc

    from aiperf.plot.config import PlotConfig
    from aiperf.plot.constants import PlotTheme
    from aiperf.plot.core.data_loader import DataLoader
    from aiperf.plot.core.mode_detector import ModeDetector, VisualizationMode
    from aiperf.plot.dashboard.builder import DashboardBuilder
    from aiperf.plot.dashboard.callbacks import register_all_callbacks
    from aiperf.plot.dashboard.styling import get_all_themes_css
    from aiperf.plot.exceptions import ModeDetectionError

    detector = ModeDetector()
    try:
        run_dirs = detector.find_run_directories([results_dir])
    except ModeDetectionError:
        logger.info("No benchmark runs found on PVC, dashboard not mounted")
        return None, 0

    if len(run_dirs) == 1:
        viz_mode = VisualizationMode.SINGLE_RUN
    else:
        viz_mode = VisualizationMode.MULTI_RUN

    loader = DataLoader()
    runs = []
    for run_dir in run_dirs:
        try:
            load_detail = viz_mode == VisualizationMode.SINGLE_RUN
            run_data = loader.load_run(run_dir, load_per_request_data=load_detail)
            runs.append(run_data)
        except Exception as e:  # noqa: BLE001 - best-effort: one bad run must not abort dashboard build
            logger.warning(f"Failed to load run from {run_dir}: {e}")

    if not runs:
        logger.warning("All runs failed to load, dashboard not mounted")
        return None, 0

    logger.info(f"Building dashboard with {len(runs)} runs in {viz_mode.value} mode")

    theme = PlotTheme.DARK
    plot_config = PlotConfig(verbose=False)

    dash_app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
        title="AIPerf Dashboard",
        requests_pathname_prefix="/dashboard/",
    )

    custom_css = get_all_themes_css()
    dash_app.index_string = f"""
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

    builder = DashboardBuilder(
        runs=runs,
        mode=viz_mode,
        theme=theme,
        plot_config=plot_config,
    )
    dash_app.layout = builder.build()

    register_all_callbacks(
        dash_app, runs, run_dirs, viz_mode, theme, plot_config, loader
    )

    return dash_app, len(runs)
