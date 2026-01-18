# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI runner."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiperf.plot.cli_runner import run_plot_controller
from aiperf.plot.constants import PlotMode, PlotTheme


class TestRunPlotController:
    """Tests for run_plot_controller function."""

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_default_paths(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that paths defaults to ['./artifacts'] when None."""
        mock_controller = MagicMock()
        mock_controller.run.return_value = [tmp_path / "plot1.png"]
        mock_controller_class.return_value = mock_controller

        run_plot_controller(paths=None, output=str(tmp_path / "output"))

        mock_controller_class.assert_called_once()
        call_args = mock_controller_class.call_args
        assert call_args.kwargs["paths"] == [Path("./artifacts")]

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_default_output(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that output defaults to first_path/plots when None."""
        mock_controller = MagicMock()
        mock_controller.run.return_value = [tmp_path / "plot1.png"]
        mock_controller_class.return_value = mock_controller

        input_paths = [str(tmp_path / "run1"), str(tmp_path / "run2")]
        run_plot_controller(paths=input_paths, output=None)

        mock_controller_class.assert_called_once()
        call_args = mock_controller_class.call_args
        expected_output = Path(input_paths[0]) / "plots"
        assert call_args.kwargs["output_dir"] == expected_output

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_string_to_plot_mode_enum(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that string mode is converted to PlotMode enum."""
        mock_controller = MagicMock()
        mock_controller.run.return_value = []
        mock_controller_class.return_value = mock_controller

        run_plot_controller(
            paths=[str(tmp_path)], output=str(tmp_path / "output"), mode="png"
        )

        mock_controller_class.assert_called_once()
        call_args = mock_controller_class.call_args
        assert call_args.kwargs["mode"] == PlotMode.PNG
        assert isinstance(call_args.kwargs["mode"], PlotMode)

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_string_to_plot_theme_enum(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that string theme is converted to PlotTheme enum."""
        mock_controller = MagicMock()
        mock_controller.run.return_value = []
        mock_controller_class.return_value = mock_controller

        run_plot_controller(
            paths=[str(tmp_path)], output=str(tmp_path / "output"), theme="dark"
        )

        mock_controller_class.assert_called_once()
        call_args = mock_controller_class.call_args
        assert call_args.kwargs["theme"] == PlotTheme.DARK
        assert isinstance(call_args.kwargs["theme"], PlotTheme)

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_string_theme_light(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that 'light' string is converted to PlotTheme.LIGHT."""
        mock_controller = MagicMock()
        mock_controller.run.return_value = []
        mock_controller_class.return_value = mock_controller

        run_plot_controller(
            paths=[str(tmp_path)], output=str(tmp_path / "output"), theme="light"
        )

        mock_controller_class.assert_called_once()
        call_args = mock_controller_class.call_args
        assert call_args.kwargs["theme"] == PlotTheme.LIGHT

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_enum_mode_passed_directly(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that PlotMode enum is passed through without conversion."""
        mock_controller = MagicMock()
        mock_controller.run.return_value = []
        mock_controller_class.return_value = mock_controller

        run_plot_controller(
            paths=[str(tmp_path)],
            output=str(tmp_path / "output"),
            mode=PlotMode.PNG,
        )

        mock_controller_class.assert_called_once()
        call_args = mock_controller_class.call_args
        assert call_args.kwargs["mode"] == PlotMode.PNG

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_enum_theme_passed_directly(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that PlotTheme enum is passed through without conversion."""
        mock_controller = MagicMock()
        mock_controller.run.return_value = []
        mock_controller_class.return_value = mock_controller

        run_plot_controller(
            paths=[str(tmp_path)],
            output=str(tmp_path / "output"),
            theme=PlotTheme.DARK,
        )

        mock_controller_class.assert_called_once()
        call_args = mock_controller_class.call_args
        assert call_args.kwargs["theme"] == PlotTheme.DARK

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_single_custom_path(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test with single custom path."""
        mock_controller = MagicMock()
        mock_controller.run.return_value = []
        mock_controller_class.return_value = mock_controller

        custom_path = str(tmp_path / "custom_run")
        run_plot_controller(paths=[custom_path], output=str(tmp_path / "output"))

        mock_controller_class.assert_called_once()
        call_args = mock_controller_class.call_args
        assert call_args.kwargs["paths"] == [Path(custom_path)]

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_multiple_custom_paths(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test with multiple custom paths."""
        mock_controller = MagicMock()
        mock_controller.run.return_value = []
        mock_controller_class.return_value = mock_controller

        custom_paths = [
            str(tmp_path / "run1"),
            str(tmp_path / "run2"),
            str(tmp_path / "run3"),
        ]
        run_plot_controller(paths=custom_paths, output=str(tmp_path / "output"))

        mock_controller_class.assert_called_once()
        call_args = mock_controller_class.call_args
        assert call_args.kwargs["paths"] == [Path(p) for p in custom_paths]

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_custom_output_directory(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test with custom output directory."""
        mock_controller = MagicMock()
        mock_controller.run.return_value = []
        mock_controller_class.return_value = mock_controller

        custom_output = str(tmp_path / "custom_output")
        run_plot_controller(paths=[str(tmp_path / "run")], output=custom_output)

        mock_controller_class.assert_called_once()
        call_args = mock_controller_class.call_args
        assert call_args.kwargs["output_dir"] == Path(custom_output)

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_controller_run_is_called(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that PlotController.run() is called."""
        mock_controller = MagicMock()
        mock_controller.run.return_value = [tmp_path / "plot1.png"]
        mock_controller_class.return_value = mock_controller

        run_plot_controller(paths=[str(tmp_path)], output=str(tmp_path / "output"))

        mock_controller.run.assert_called_once()

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_output_message_with_plots(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test output message displays correct number of plots."""
        output_dir = tmp_path / "output"
        mock_controller = MagicMock()
        mock_controller.run.return_value = [
            output_dir / "plot1.png",
            output_dir / "plot2.png",
            output_dir / "plot3.png",
        ]
        mock_controller_class.return_value = mock_controller

        run_plot_controller(paths=[str(tmp_path)], output=str(output_dir))

        captured = capsys.readouterr()
        assert "Saved 3 plots" in captured.out
        assert f"to: {output_dir}" in captured.out

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_output_message_with_no_plots(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test output message when no plots generated."""
        output_dir = tmp_path / "output"
        mock_controller = MagicMock()
        mock_controller.run.return_value = []
        mock_controller_class.return_value = mock_controller

        run_plot_controller(paths=[str(tmp_path)], output=str(output_dir))

        captured = capsys.readouterr()
        assert "Saved 0 plots" in captured.out
        assert f"to: {output_dir}" in captured.out

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_all_parameters_passed_to_controller(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that all parameters are correctly passed to PlotController."""
        mock_controller = MagicMock()
        mock_controller.run.return_value = []
        mock_controller_class.return_value = mock_controller

        paths = [str(tmp_path / "run1"), str(tmp_path / "run2")]
        output = str(tmp_path / "output")

        run_plot_controller(
            paths=paths,
            output=output,
            mode=PlotMode.PNG,
            theme=PlotTheme.DARK,
            verbose=True,
        )

        mock_controller_class.assert_called_once_with(
            paths=[Path(p) for p in paths],
            output_dir=Path(output),
            mode=PlotMode.PNG,
            theme=PlotTheme.DARK,
            config_path=None,
            verbose=True,
            port=8050,
        )

    def test_invalid_mode_string_raises_value_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that invalid mode string raises ValueError."""
        with pytest.raises(ValueError, match="'invalid_mode' is not a valid PlotMode"):
            run_plot_controller(
                paths=[str(tmp_path)],
                output=str(tmp_path / "output"),
                mode="invalid_mode",
            )

    def test_invalid_theme_string_raises_value_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that invalid theme string raises ValueError."""
        with pytest.raises(
            ValueError, match="'invalid_theme' is not a valid PlotTheme"
        ):
            run_plot_controller(
                paths=[str(tmp_path)],
                output=str(tmp_path / "output"),
                theme="invalid_theme",
            )
