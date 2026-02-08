# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot controller for generating visualizations from profiling data."""

import logging
from pathlib import Path

from aiperf.plot.config import PlotConfig
from aiperf.plot.constants import PlotMode, PlotTheme
from aiperf.plot.core.data_loader import DataLoader
from aiperf.plot.core.mode_detector import ModeDetector, VisualizationMode
from aiperf.plot.dashboard.server import DashboardServer
from aiperf.plot.exporters.png.multi_run import MultiRunPNGExporter
from aiperf.plot.exporters.png.single_run import SingleRunPNGExporter
from aiperf.plot.logging import setup_console_only_logging, setup_plot_logging

logger = logging.getLogger(__name__)

__all__ = ["PlotController"]


class PlotController:
    """Controller for generating plots from AIPerf profiling data.

    Orchestrates the plot generation pipeline: mode detection, data loading,
    and export. Designed to support multiple output modes (PNG)
    in the future.

    Args:
        paths: List of paths to profiling run directories
        output_dir: Directory to save generated plots
        mode: Output mode (currently only PNG supported)
        theme: Plot theme (LIGHT or DARK). Defaults to LIGHT.
        config_path: Optional path to custom plot configuration YAML file
        verbose: Show detailed error tracebacks in console
    """

    def __init__(
        self,
        paths: list[Path],
        output_dir: Path,
        mode: PlotMode = PlotMode.PNG,
        theme: PlotTheme = PlotTheme.LIGHT,
        config_path: Path | None = None,
        verbose: bool = False,
        host: str = "127.0.0.1",
        port: int = 8050,
    ):
        self.paths = paths
        self.output_dir = output_dir
        self.mode = mode
        self.theme = theme
        self.verbose = verbose
        self.host = host
        self.port = port

        log_level = "DEBUG" if verbose else "INFO"
        try:
            setup_plot_logging(output_dir, log_level=log_level)
        except (OSError, PermissionError) as e:
            setup_console_only_logging(log_level=log_level)
            logger.warning(
                f"Could not set up file logging to {output_dir}: {e}. Using console only.",
                exc_info=self.verbose,
            )

        self.mode_detector = ModeDetector()
        self.plot_config = PlotConfig(config_path, verbose=verbose)

        classification_config = self.plot_config.get_experiment_classification_config()
        if classification_config:
            logger.info(
                "Experiment classification enabled: grouping runs by baseline/treatment patterns"
            )

        downsampling_config = self.plot_config.get_downsampling_config()
        if not downsampling_config["enabled"]:
            logger.info("Server metrics downsampling disabled by configuration")
        else:
            logger.info(
                f"Server metrics downsampling: {downsampling_config['window_size_seconds']}s windows, "
                f"{downsampling_config['aggregation_method']} aggregation"
            )

        self.loader = DataLoader(
            classification_config=classification_config,
            downsampling_config=downsampling_config,
        )

    def run(self) -> list[Path] | None:
        """Execute plot generation pipeline.

        Returns:
            List of paths to generated plot files (PNG mode) or None (dashboard mode)
        """
        if self.mode == PlotMode.PNG:
            return self._generate_png_plots()
        elif self.mode == PlotMode.DASHBOARD:
            self._launch_dashboard_server()
            return None
        else:
            raise ValueError(
                f"Unsupported mode: {self.mode}. Currently only '{PlotMode.PNG}' and '{PlotMode.DASHBOARD}' are supported."
            )

    def _validate_paths(self) -> None:
        """Validate that all input paths exist and are directories."""
        for path in self.paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Path does not exist: {path}. Please check the path and try again."
                )
            if not path.is_dir():
                raise NotADirectoryError(
                    f"Path is not a directory: {path}. Please provide a directory containing profiling runs."
                )

    def _detect_visualization_mode(self) -> tuple[VisualizationMode, list[Path]]:
        """Detect whether to generate single-run or multi-run plots.

        Returns:
            Tuple of (visualization mode, list of run directories)
        """
        mode, run_dirs = self.mode_detector.detect_mode(self.paths)

        if not run_dirs:
            raise ValueError(
                f"No valid profiling runs found in: {self.paths}. "
                "Please ensure the directory contains AIPerf profiling output."
            )

        return mode, run_dirs

    def _generate_png_plots(self) -> list[Path]:
        """Generate static PNG plot images.

        Returns:
            List of paths to generated PNG files
        """
        self._validate_paths()
        viz_mode, run_dirs = self._detect_visualization_mode()

        mode_name = viz_mode.value.replace("_", "-")
        run_count = len(run_dirs)
        run_word = "run" if run_count == 1 else "runs"
        logger.info(f"Generating {mode_name} plots ({run_count} {run_word})")

        if viz_mode == VisualizationMode.MULTI_RUN:
            return self._export_multi_run_plots(run_dirs)
        else:
            return self._export_single_run_plots(run_dirs[0])

    def _export_multi_run_plots(self, run_dirs: list[Path]) -> list[Path]:
        """Export multi-run comparison plots.

        Args:
            run_dirs: List of run directories to compare

        Returns:
            List of paths to generated plot files
        """
        runs = []
        for run_dir in run_dirs:
            try:
                run_data = self.loader.load_run(run_dir, load_per_request_data=False)
                runs.append(run_data)
            except Exception as e:
                logger.warning(f"Failed to load run from {run_dir}: {e}")

        if not runs:
            raise ValueError("Failed to load any valid profiling runs")

        available = self.loader.get_available_metrics(runs[0])
        plot_specs = self.plot_config.get_multi_run_plot_specs()
        classification_config = self.plot_config.get_experiment_classification_config()
        exporter = MultiRunPNGExporter(self.output_dir, theme=self.theme)
        return exporter.export(
            runs,
            available,
            plot_specs=plot_specs,
            classification_config=classification_config,
        )

    def _export_single_run_plots(self, run_dir: Path) -> list[Path]:
        """Export single-run time series plots.

        Args:
            run_dir: Run directory to generate plots from

        Returns:
            List of paths to generated plot files
        """
        run_data = self.loader.load_run(run_dir, load_per_request_data=True)
        available = self.loader.get_available_metrics(run_data)
        plot_specs = self.plot_config.get_single_run_plot_specs()
        exporter = SingleRunPNGExporter(self.output_dir, theme=self.theme)
        return exporter.export(run_data, available, plot_specs=plot_specs)

    def _launch_dashboard_server(self) -> None:
        """Launch interactive Dash dashboard server.

        This method will not return until the server is stopped (Ctrl+C).
        """
        self._validate_paths()
        viz_mode, run_dirs = self._detect_visualization_mode()

        run_count = len(run_dirs)
        run_word = "run" if run_count == 1 else "runs"
        logger.info(f"Loading {run_count} {run_word}...")

        # Load run data based on visualization mode
        if viz_mode == VisualizationMode.MULTI_RUN:
            runs = []
            for run_dir in run_dirs:
                try:
                    run_data = self.loader.load_run(
                        run_dir, load_per_request_data=False
                    )
                    runs.append(run_data)
                except Exception as e:
                    logger.warning(f"Failed to load run from {run_dir}: {e}")

            if not runs:
                raise ValueError("Failed to load any valid profiling runs")
        else:
            # Single-run mode: load with per-request data
            run_data = self.loader.load_run(run_dirs[0], load_per_request_data=True)
            runs = [run_data]

        server = DashboardServer(
            runs=runs,
            run_dirs=run_dirs,
            mode=viz_mode,
            theme=self.theme,
            plot_config=self.plot_config,
            loader=self.loader,
            host=self.host,
            port=self.port,
        )

        server.run()
