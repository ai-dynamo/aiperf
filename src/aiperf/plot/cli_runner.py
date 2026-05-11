# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI runner for plot command."""

from pathlib import Path
from typing import cast

import orjson

from aiperf.common.config import MLflowDefaults
from aiperf.common.redact import REDACTED_VALUE
from aiperf.exporters.mlflow_data_exporter import MLflowDataExporter
from aiperf.exporters.mlflow_metadata import MLflowExportMetadata
from aiperf.plot.constants import PLOT_LOG_FILE, PlotMode, PlotTheme
from aiperf.plot.plot_controller import PlotController


def _load_mlflow_metadata(metadata_file: Path) -> MLflowExportMetadata:
    if not metadata_file.exists():
        return {}
    try:
        metadata = orjson.loads(metadata_file.read_bytes())
    except orjson.JSONDecodeError as exc:
        raise ValueError(
            f"expected JSON object for MLflow metadata in {metadata_file} but failed to decode"
        ) from exc

    # Runtime guard: mlflow_export.json is on disk and may be hand-edited
    # or corrupted. The TypedDict only asserts shape to the type checker.
    if not isinstance(metadata, dict):
        raise ValueError(
            f"expected JSON object for MLflow metadata in {metadata_file} "
            f"but got {type(metadata).__name__}"
        )
    return cast(MLflowExportMetadata, metadata)


def _resolve_mlflow_upload_target(
    *,
    input_paths: list[Path],
    tracking_uri: str | None,
    run_id: str | None,
) -> tuple[str, str]:
    if len(input_paths) != 1:
        raise ValueError(
            "--mlflow-upload requires exactly one run path in --paths so the target run can be resolved."
        )

    metadata_file = input_paths[0] / MLflowDefaults.EXPORT_METADATA_FILE
    metadata = _load_mlflow_metadata(metadata_file)
    resolved_tracking_uri = tracking_uri or metadata.get("tracking_uri")
    resolved_run_id = run_id or metadata.get("run_id")

    if not resolved_tracking_uri:
        raise ValueError(
            "MLflow tracking URI is required for --mlflow-upload. "
            "Provide --mlflow-tracking-uri or ensure "
            f"{metadata_file} contains 'tracking_uri'."
        )
    # The on-disk tracking URI is persisted with userinfo redacted (see
    # MLflowDataExporter._write_export_metadata), so a credentialed backend
    # like postgresql://user:secret@db/mlflow round-trips as
    # postgresql://<redacted>@db/mlflow — not a usable connection string.
    # Only fall back to the on-disk value when the user did not pass
    # --mlflow-tracking-uri explicitly, and raise early if the fallback is
    # unusable so the user isn't blocked on a surprising MLflow client error.
    if tracking_uri is None and REDACTED_VALUE in str(resolved_tracking_uri):
        raise ValueError(
            "MLflow tracking URI in "
            f"{metadata_file} has redacted credentials "
            f"({resolved_tracking_uri!r}) and cannot be used directly. "
            "Pass --mlflow-tracking-uri with the original credentialed URI."
        )
    if not resolved_run_id:
        raise ValueError(
            "MLflow run id is required for --mlflow-upload. "
            "Provide --mlflow-run-id or ensure "
            f"{metadata_file} contains 'run_id'."
        )
    return str(resolved_tracking_uri), str(resolved_run_id)


def _upload_generated_plots_to_mlflow(
    *,
    generated_files: list[Path],
    input_paths: list[Path],
    output_dir: Path,
    tracking_uri: str | None,
    run_id: str | None,
) -> None:
    if not generated_files:
        print("No plots were generated; skipping MLflow plot upload.")
        return

    resolved_tracking_uri, resolved_run_id = _resolve_mlflow_upload_target(
        input_paths=input_paths,
        tracking_uri=tracking_uri,
        run_id=run_id,
    )
    uploaded = MLflowDataExporter.upload_artifacts_to_run(
        tracking_uri=resolved_tracking_uri,
        run_id=resolved_run_id,
        artifact_directory=output_dir,
        artifact_files=generated_files,
    )
    artifact_word = "artifact" if len(uploaded) == 1 else "artifacts"
    print(
        f"Uploaded {len(uploaded)} plot {artifact_word} to MLflow run {resolved_run_id}."
    )


def run_plot_controller(
    paths: list[str] | None = None,
    output: str | None = None,
    *,
    mode: PlotMode | str = PlotMode.PNG,
    theme: PlotTheme | str = PlotTheme.LIGHT,
    config: str | None = None,
    verbose: bool = False,
    dashboard: bool = False,
    host: str = "127.0.0.1",
    port: int = 8050,
    mlflow_upload: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_run_id: str | None = None,
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
        mlflow_upload: Upload generated plot artifacts to an existing MLflow run.
        mlflow_tracking_uri: MLflow tracking URI override used with --mlflow-upload.
        mlflow_run_id: MLflow run id override used with --mlflow-upload.
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

    if dashboard and mlflow_upload:
        raise ValueError("--dashboard and --mlflow-upload are mutually exclusive")

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
        result = result or []
        plot_word = "plot" if len(result) == 1 else "plots"
        print(f"\nSaved {len(result)} {plot_word} to: {output_dir}")
        if mlflow_upload:
            _upload_generated_plots_to_mlflow(
                generated_files=result,
                input_paths=input_paths,
                output_dir=output_dir,
                tracking_uri=mlflow_tracking_uri,
                run_id=mlflow_run_id,
            )
    print(f"Logs: {output_dir / PLOT_LOG_FILE}")
