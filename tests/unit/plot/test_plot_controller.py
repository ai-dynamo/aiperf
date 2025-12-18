# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PlotController."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiperf.plot.constants import PlotMode, PlotTheme
from aiperf.plot.core.mode_detector import VisualizationMode
from aiperf.plot.exceptions import ModeDetectionError
from aiperf.plot.plot_controller import PlotController


class TestPlotControllerInit:
    """Tests for PlotController initialization."""

    def test_init_with_defaults(self, tmp_path: Path) -> None:
        """Test PlotController initialization with default parameters."""
        controller = PlotController(
            paths=[tmp_path],
            output_dir=tmp_path / "output",
        )

        assert controller.paths == [tmp_path]
        assert controller.output_dir == tmp_path / "output"
        assert controller.mode == PlotMode.PNG
        assert controller.theme == PlotTheme.LIGHT
        assert controller.loader is not None
        assert controller.mode_detector is not None

    def test_init_with_png_mode(self, tmp_path: Path) -> None:
        """Test PlotController initialization with PNG mode."""
        controller = PlotController(
            paths=[tmp_path],
            output_dir=tmp_path / "output",
            mode=PlotMode.PNG,
        )

        assert controller.mode == PlotMode.PNG

    def test_init_with_dark_theme(self, tmp_path: Path) -> None:
        """Test PlotController initialization with dark theme."""
        controller = PlotController(
            paths=[tmp_path],
            output_dir=tmp_path / "output",
            theme=PlotTheme.DARK,
        )

        assert controller.theme == PlotTheme.DARK

    def test_init_with_light_theme(self, tmp_path: Path) -> None:
        """Test PlotController initialization with light theme."""
        controller = PlotController(
            paths=[tmp_path],
            output_dir=tmp_path / "output",
            theme=PlotTheme.LIGHT,
        )

        assert controller.theme == PlotTheme.LIGHT

    def test_init_with_multiple_paths(self, tmp_path: Path) -> None:
        """Test PlotController initialization with multiple paths."""
        paths = [tmp_path / "run1", tmp_path / "run2", tmp_path / "run3"]
        controller = PlotController(
            paths=paths,
            output_dir=tmp_path / "output",
        )

        assert controller.paths == paths


class TestPlotControllerValidatePaths:
    """Tests for PlotController._validate_paths method."""

    def test_validate_paths_with_existing_path(
        self, single_run_dir: Path, tmp_path: Path
    ) -> None:
        """Test path validation with existing path."""
        controller = PlotController(
            paths=[single_run_dir],
            output_dir=tmp_path / "output",
        )

        # Should not raise
        controller._validate_paths()

    def test_validate_paths_with_nonexistent_path(self, tmp_path: Path) -> None:
        """Test path validation with nonexistent path raises error."""
        nonexistent = tmp_path / "nonexistent"
        controller = PlotController(
            paths=[nonexistent],
            output_dir=tmp_path / "output",
        )

        with pytest.raises(
            FileNotFoundError, match=f"Path does not exist: {nonexistent}"
        ):
            controller._validate_paths()

    def test_validate_paths_with_multiple_invalid_paths(self, tmp_path: Path) -> None:
        """Test path validation with multiple nonexistent paths."""
        nonexistent1 = tmp_path / "nonexistent1"
        nonexistent2 = tmp_path / "nonexistent2"
        controller = PlotController(
            paths=[nonexistent1, nonexistent2],
            output_dir=tmp_path / "output",
        )

        # Should raise on first invalid path
        with pytest.raises(FileNotFoundError):
            controller._validate_paths()

    def test_validate_paths_with_mixed_valid_invalid(
        self, single_run_dir: Path, tmp_path: Path
    ) -> None:
        """Test path validation with mix of valid and invalid paths."""
        nonexistent = tmp_path / "nonexistent"
        controller = PlotController(
            paths=[single_run_dir, nonexistent],
            output_dir=tmp_path / "output",
        )

        with pytest.raises(FileNotFoundError):
            controller._validate_paths()

    def test_validate_paths_with_multiple_valid_paths(
        self, multiple_run_dirs: list[Path], tmp_path: Path
    ) -> None:
        """Test path validation with multiple valid paths."""
        controller = PlotController(
            paths=multiple_run_dirs,
            output_dir=tmp_path / "output",
        )

        # Should not raise
        controller._validate_paths()


class TestPlotControllerDetectVisualizationMode:
    """Tests for PlotController._detect_visualization_mode method."""

    def test_detect_single_run_mode(self, single_run_dir: Path, tmp_path: Path) -> None:
        """Test detection of single run mode."""
        controller = PlotController(
            paths=[single_run_dir],
            output_dir=tmp_path / "output",
        )

        mode, run_dirs = controller._detect_visualization_mode()

        assert mode == VisualizationMode.SINGLE_RUN
        assert len(run_dirs) == 1
        assert run_dirs[0] == single_run_dir

    def test_detect_multi_run_mode(
        self, multiple_run_dirs: list[Path], tmp_path: Path
    ) -> None:
        """Test detection of multi-run mode."""
        controller = PlotController(
            paths=multiple_run_dirs,
            output_dir=tmp_path / "output",
        )

        mode, run_dirs = controller._detect_visualization_mode()

        assert mode == VisualizationMode.MULTI_RUN
        assert len(run_dirs) == 3
        assert set(run_dirs) >= set(multiple_run_dirs)

    def test_detect_mode_with_no_valid_runs_raises_error(self, tmp_path: Path) -> None:
        """Test that detection with no valid runs raises ModeDetectionError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        controller = PlotController(
            paths=[empty_dir],
            output_dir=tmp_path / "output",
        )

        with pytest.raises(ModeDetectionError, match="does not contain any valid"):
            controller._detect_visualization_mode()

    def test_detect_mode_from_parent_directory(
        self, parent_dir_with_runs: Path, tmp_path: Path
    ) -> None:
        """Test mode detection from parent directory."""
        controller = PlotController(
            paths=[parent_dir_with_runs],
            output_dir=tmp_path / "output",
        )

        mode, run_dirs = controller._detect_visualization_mode()

        assert mode == VisualizationMode.MULTI_RUN
        assert len(run_dirs) == 3


class TestPlotControllerExportMultiRun:
    """Tests for PlotController._export_multi_run_plots method."""

    @patch("aiperf.plot.plot_controller.MultiRunPNGExporter")
    def test_export_multi_run_success(
        self,
        mock_exporter_class: MagicMock,
        multiple_run_dirs: list[Path],
        tmp_path: Path,
    ) -> None:
        """Test successful multi-run export."""
        # Setup mock
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [tmp_path / "plot1.png"]
        mock_exporter_class.return_value = mock_exporter

        controller = PlotController(
            paths=multiple_run_dirs,
            output_dir=tmp_path / "output",
            theme=PlotTheme.DARK,
        )

        # Mock loader to return valid data
        controller.loader.load_run = MagicMock(return_value={"test": "data"})
        controller.loader.get_available_metrics = MagicMock(
            return_value={"metric1": {"unit": "ms"}}
        )

        result = controller._export_multi_run_plots(multiple_run_dirs)

        # Verify exporter was created with correct params
        mock_exporter_class.assert_called_once_with(
            tmp_path / "output", theme=PlotTheme.DARK
        )

        # Verify loader was called for each run (2 paths passed directly)
        assert controller.loader.load_run.call_count == 2

        # Verify export was called
        mock_exporter.export.assert_called_once()
        assert result == [tmp_path / "plot1.png"]

    @patch("aiperf.plot.plot_controller.MultiRunPNGExporter")
    def test_export_multi_run_with_load_failures(
        self,
        mock_exporter_class: MagicMock,
        multiple_run_dirs: list[Path],
        tmp_path: Path,
        capsys,
    ) -> None:
        """Test multi-run export when some runs fail to load."""
        # Setup mock
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [tmp_path / "plot1.png"]
        mock_exporter_class.return_value = mock_exporter

        controller = PlotController(
            paths=multiple_run_dirs,
            output_dir=tmp_path / "output",
        )

        # Mock loader to fail on first run, succeed on others
        def mock_load_run(run_dir, **_kwargs):
            if run_dir == multiple_run_dirs[0]:
                raise ValueError("Failed to load run")
            return {"test": "data"}

        controller.loader.load_run = MagicMock(side_effect=mock_load_run)
        controller.loader.get_available_metrics = MagicMock(
            return_value={"metric1": {"unit": "ms"}}
        )

        result = controller._export_multi_run_plots(multiple_run_dirs)

        # Verify warning was logged
        captured = capsys.readouterr()
        assert "Failed to load run" in captured.out

        # Verify export was still called with successful runs
        assert result == [tmp_path / "plot1.png"]

    @patch("aiperf.plot.plot_controller.MultiRunPNGExporter")
    def test_export_multi_run_all_failures_raises_error(
        self,
        _mock_exporter_class: MagicMock,
        multiple_run_dirs: list[Path],
        tmp_path: Path,
    ) -> None:
        """Test multi-run export when all runs fail to load."""
        controller = PlotController(
            paths=multiple_run_dirs,
            output_dir=tmp_path / "output",
        )

        # Mock loader to always fail
        controller.loader.load_run = MagicMock(
            side_effect=ValueError("Failed to load run")
        )

        with pytest.raises(ValueError, match="Failed to load any valid profiling runs"):
            controller._export_multi_run_plots(multiple_run_dirs)


class TestPlotControllerExportSingleRun:
    """Tests for PlotController._export_single_run_plots method."""

    @patch("aiperf.plot.plot_controller.SingleRunPNGExporter")
    def test_export_single_run_success(
        self,
        mock_exporter_class: MagicMock,
        single_run_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test successful single-run export."""
        # Setup mock
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [tmp_path / "plot1.png"]
        mock_exporter_class.return_value = mock_exporter

        controller = PlotController(
            paths=[single_run_dir],
            output_dir=tmp_path / "output",
            theme=PlotTheme.LIGHT,
        )

        # Mock loader
        controller.loader.load_run = MagicMock(return_value={"test": "data"})
        controller.loader.get_available_metrics = MagicMock(
            return_value={"metric1": {"unit": "ms"}}
        )

        result = controller._export_single_run_plots(single_run_dir)

        # Verify exporter was created with correct params
        mock_exporter_class.assert_called_once_with(
            tmp_path / "output", theme=PlotTheme.LIGHT
        )

        # Verify loader was called with per_request_data=True
        controller.loader.load_run.assert_called_once_with(
            single_run_dir, load_per_request_data=True
        )

        # Verify export was called
        mock_exporter.export.assert_called_once()
        assert result == [tmp_path / "plot1.png"]

    @patch("aiperf.plot.plot_controller.SingleRunPNGExporter")
    def test_export_single_run_load_failure_propagates(
        self,
        _mock_exporter_class: MagicMock,
        single_run_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test that single-run export propagates load failures."""
        controller = PlotController(
            paths=[single_run_dir],
            output_dir=tmp_path / "output",
        )

        # Mock loader to fail
        controller.loader.load_run = MagicMock(
            side_effect=ValueError("Failed to load run")
        )

        with pytest.raises(ValueError, match="Failed to load run"):
            controller._export_single_run_plots(single_run_dir)


class TestPlotControllerGeneratePNGPlots:
    """Tests for PlotController._generate_png_plots integration."""

    @patch("aiperf.plot.plot_controller.setup_plot_logging")
    @patch("aiperf.plot.plot_controller.SingleRunPNGExporter")
    def test_generate_png_plots_single_run_integration(
        self,
        mock_exporter_class: MagicMock,
        mock_setup_logging: MagicMock,
        single_run_dir: Path,
        tmp_path: Path,
        caplog,
    ) -> None:
        """Test full PNG generation flow for single run."""
        # Setup mock
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [tmp_path / "plot1.png"]
        mock_exporter_class.return_value = mock_exporter

        with caplog.at_level(logging.INFO, logger="aiperf.plot.plot_controller"):
            controller = PlotController(
                paths=[single_run_dir],
                output_dir=tmp_path / "output",
            )

            # Mock loader
            controller.loader.load_run = MagicMock(return_value={"test": "data"})
            controller.loader.get_available_metrics = MagicMock(
                return_value={"metric1": {"unit": "ms"}}
            )

            result = controller._generate_png_plots()

        # Verify log message
        assert "single-run" in caplog.text
        assert "(1 run)" in caplog.text

        assert result == [tmp_path / "plot1.png"]

    @patch("aiperf.plot.plot_controller.setup_plot_logging")
    @patch("aiperf.plot.plot_controller.MultiRunPNGExporter")
    def test_generate_png_plots_multi_run_integration(
        self,
        mock_exporter_class: MagicMock,
        mock_setup_logging: MagicMock,
        multiple_run_dirs: list[Path],
        tmp_path: Path,
        caplog,
    ) -> None:
        """Test full PNG generation flow for multi-run."""
        # Setup mock
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = [tmp_path / "plot1.png"]
        mock_exporter_class.return_value = mock_exporter

        with caplog.at_level(logging.INFO, logger="aiperf.plot.plot_controller"):
            controller = PlotController(
                paths=multiple_run_dirs,
                output_dir=tmp_path / "output",
            )

            # Mock loader
            controller.loader.load_run = MagicMock(return_value={"test": "data"})
            controller.loader.get_available_metrics = MagicMock(
                return_value={"metric1": {"unit": "ms"}}
            )

            result = controller._generate_png_plots()

        # Verify log message
        assert "multi-run" in caplog.text
        assert "(3 runs)" in caplog.text

        assert result == [tmp_path / "plot1.png"]

    def test_generate_png_plots_invalid_path_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid path raises error during generation."""
        nonexistent = tmp_path / "nonexistent"
        controller = PlotController(
            paths=[nonexistent],
            output_dir=tmp_path / "output",
        )

        with pytest.raises(FileNotFoundError):
            controller._generate_png_plots()
