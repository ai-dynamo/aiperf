# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI runner."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.config import MLflowDefaults
from aiperf.plot.cli_runner import (
    _resolve_mlflow_upload_target,
    _upload_generated_plots_to_mlflow,
    run_plot_controller,
)
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
            host="127.0.0.1",
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

    @patch("aiperf.plot.cli_runner._upload_generated_plots_to_mlflow")
    @patch("aiperf.plot.cli_runner.PlotController")
    def test_mlflow_upload_invoked_only_when_enabled(
        self,
        mock_controller_class: MagicMock,
        mock_upload: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test generated plots are uploaded to MLflow only with --mlflow-upload."""
        output_dir = tmp_path / "output"
        run_dir = tmp_path / "run1"
        mock_controller = MagicMock()
        mock_controller.run.return_value = [output_dir / "ttft_over_time.png"]
        mock_controller_class.return_value = mock_controller

        run_plot_controller(
            paths=[str(run_dir)],
            output=str(output_dir),
            mlflow_upload=True,
            mlflow_tracking_uri="http://mlflow:5000",
            mlflow_run_id="run-123",
        )

        mock_upload.assert_called_once_with(
            generated_files=[output_dir / "ttft_over_time.png"],
            input_paths=[run_dir],
            output_dir=output_dir,
            tracking_uri="http://mlflow:5000",
            run_id="run-123",
        )

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_mlflow_upload_rejected_in_dashboard_mode(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test --mlflow-upload cannot be combined with --dashboard."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            run_plot_controller(
                paths=[str(tmp_path / "run1")],
                output=str(tmp_path / "output"),
                dashboard=True,
                mlflow_upload=True,
            )
        mock_controller_class.assert_not_called()

    @patch("aiperf.plot.cli_runner.PlotController")
    def test_plot_dashboard_and_mlflow_upload_rejected_before_startup(
        self,
        mock_controller_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test --dashboard and --mlflow-upload raises ValueError before PlotController is constructed."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            run_plot_controller(
                paths=[str(tmp_path / "run1")],
                output=str(tmp_path / "output"),
                dashboard=True,
                mlflow_upload=True,
            )
        mock_controller_class.assert_not_called()


class TestResolveMlflowUploadTarget:
    def test_resolves_tracking_and_run_from_metadata(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        run_dir.mkdir(parents=True)
        metadata_file = run_dir / MLflowDefaults.EXPORT_METADATA_FILE
        metadata_file.write_text(
            '{"tracking_uri":"http://mlflow:5000","run_id":"run-abc"}',
            encoding="utf-8",
        )

        tracking_uri, run_id = _resolve_mlflow_upload_target(
            input_paths=[run_dir],
            tracking_uri=None,
            run_id=None,
        )

        assert tracking_uri == "http://mlflow:5000"
        assert run_id == "run-abc"

    def test_requires_single_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="exactly one run path"):
            _resolve_mlflow_upload_target(
                input_paths=[tmp_path / "run1", tmp_path / "run2"],
                tracking_uri="http://mlflow:5000",
                run_id="run-123",
            )

    def test_rejects_non_object_metadata(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        run_dir.mkdir(parents=True)
        metadata_file = run_dir / MLflowDefaults.EXPORT_METADATA_FILE
        metadata_file.write_text('["not-an-object"]', encoding="utf-8")

        with pytest.raises(
            ValueError, match="expected JSON object for MLflow metadata"
        ):
            _resolve_mlflow_upload_target(
                input_paths=[run_dir],
                tracking_uri=None,
                run_id=None,
            )

    def test_rejects_malformed_metadata_json(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        run_dir.mkdir(parents=True)
        metadata_file = run_dir / MLflowDefaults.EXPORT_METADATA_FILE
        metadata_file.write_text("{invalid-json", encoding="utf-8")

        with pytest.raises(ValueError, match="failed to decode"):
            _resolve_mlflow_upload_target(
                input_paths=[run_dir],
                tracking_uri=None,
                run_id=None,
            )

    def test_rejects_redacted_fallback_tracking_uri(self, tmp_path: Path) -> None:
        """When the on-disk tracking URI has userinfo redacted (e.g. postgresql://
        <redacted>@db/mlflow), the plot command cannot authenticate with it, so
        surface a clear error asking the user to pass --mlflow-tracking-uri."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir(parents=True)
        metadata_file = run_dir / MLflowDefaults.EXPORT_METADATA_FILE
        metadata_file.write_text(
            '{"tracking_uri":"postgresql://<redacted>@db:5432/mlflow","run_id":"run-xyz"}',
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="redacted credentials"):
            _resolve_mlflow_upload_target(
                input_paths=[run_dir],
                tracking_uri=None,
                run_id=None,
            )

    def test_accepts_explicit_tracking_uri_when_on_disk_redacted(
        self, tmp_path: Path
    ) -> None:
        """User-provided --mlflow-tracking-uri overrides the on-disk redacted value."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir(parents=True)
        metadata_file = run_dir / MLflowDefaults.EXPORT_METADATA_FILE
        metadata_file.write_text(
            '{"tracking_uri":"postgresql://<redacted>@db:5432/mlflow","run_id":"run-xyz"}',
            encoding="utf-8",
        )

        tracking_uri, run_id = _resolve_mlflow_upload_target(
            input_paths=[run_dir],
            tracking_uri="postgresql://u:p@db:5432/mlflow",
            run_id=None,
        )
        assert tracking_uri == "postgresql://u:p@db:5432/mlflow"
        assert run_id == "run-xyz"


class TestUploadGeneratedPlotsToMlflow:
    @patch("aiperf.plot.cli_runner.MLflowDataExporter.upload_artifacts_to_run")
    def test_skips_upload_when_no_generated_files(
        self,
        mock_upload: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        _upload_generated_plots_to_mlflow(
            generated_files=[],
            input_paths=[tmp_path / "run1"],
            output_dir=tmp_path / "plots",
            tracking_uri="http://mlflow:5000",
            run_id="run-123",
        )

        mock_upload.assert_not_called()
        captured = capsys.readouterr()
        assert "No plots were generated; skipping MLflow plot upload." in captured.out

    @patch("aiperf.plot.cli_runner.MLflowDataExporter.upload_artifacts_to_run")
    @patch("aiperf.plot.cli_runner._resolve_mlflow_upload_target")
    def test_uploads_and_prints_summary(
        self,
        mock_resolve_target: MagicMock,
        mock_upload: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        output_dir = tmp_path / "plots"
        generated_files = [
            output_dir / "ttft_over_time.png",
            output_dir / "ttft_p99.png",
        ]
        mock_resolve_target.return_value = ("http://mlflow:5000", "run-123")
        mock_upload.return_value = ["ttft_over_time.png", "ttft_p99.png"]

        _upload_generated_plots_to_mlflow(
            generated_files=generated_files,
            input_paths=[tmp_path / "run1"],
            output_dir=output_dir,
            tracking_uri=None,
            run_id=None,
        )

        mock_resolve_target.assert_called_once_with(
            input_paths=[tmp_path / "run1"],
            tracking_uri=None,
            run_id=None,
        )
        mock_upload.assert_called_once_with(
            tracking_uri="http://mlflow:5000",
            run_id="run-123",
            artifact_directory=output_dir,
            artifact_files=generated_files,
        )
        captured = capsys.readouterr()
        assert "Uploaded 2 plot artifacts to MLflow run run-123." in captured.out
