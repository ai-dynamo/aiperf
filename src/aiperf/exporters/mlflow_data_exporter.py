# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.config import MLflowDefaults
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.optional_dependencies import mlflow_dependency_message
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


class MLflowDataExporter(AIPerfLoggerMixin):
    """Uploads benchmark summary metrics and artifacts to MLflow Tracking."""

    _PLOT_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".html"}

    def __init__(self, exporter_config: ExporterConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._user_config = exporter_config.user_config

        if not self._user_config.mlflow_enabled:
            raise DataExporterDisabled(
                "MLflow export is disabled (set --mlflow-tracking-uri to enable)."
            )
        if self._results is None:
            raise DataExporterDisabled(
                "MLflow export is disabled (no profile results available)."
            )

        self._tracking_uri = self._user_config.mlflow_tracking_uri
        self._experiment_name = self._user_config.mlflow_experiment
        self._run_name = self._user_config.mlflow_run_name
        self._artifact_directory = self._user_config.output.artifact_directory
        self._artifact_globs = self._user_config.mlflow_resolved_artifact_globs
        self._metadata_file = (
            self._artifact_directory / MLflowDefaults.EXPORT_METADATA_FILE
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="MLflow Tracking Export Metadata",
            file_path=self._metadata_file,
        )

    async def export(self) -> None:
        """Run blocking MLflow client operations in a worker thread."""
        await asyncio.to_thread(self._export_sync)

    @classmethod
    def _import_mlflow_module(cls) -> Any:
        """Import mlflow with a consistent dependency error message."""
        cls._configure_mlflow_http_request_defaults()
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                mlflow_dependency_message("MLflow export is enabled")
            ) from exc
        return mlflow

    @staticmethod
    def _configure_mlflow_http_request_defaults() -> None:
        """Set bounded MLflow HTTP defaults without overriding user-provided values."""
        os.environ.setdefault(
            "MLFLOW_HTTP_REQUEST_TIMEOUT",
            str(MLflowDefaults.REQUEST_TIMEOUT_SECONDS),
        )
        os.environ.setdefault(
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES",
            str(MLflowDefaults.REQUEST_MAX_RETRIES),
        )

    @classmethod
    def resolve_artifact_path(
        cls,
        *,
        artifact_directory: Path,
        artifact_file: Path,
    ) -> str:
        """Classify artifact destination under MLflow artifact tree."""
        try:
            relative_parent = artifact_file.relative_to(artifact_directory).parent
        except ValueError:
            relative_parent = Path(".")

        base = (
            "plots" if artifact_file.suffix.lower() in cls._PLOT_SUFFIXES else "exports"
        )
        parts = list(relative_parent.parts)
        if parts and parts[0] == base:
            parts = parts[1:]
            relative_parent = Path(*parts) if parts else Path(".")
        if relative_parent.as_posix() == ".":
            return base
        return f"{base}/{relative_parent.as_posix()}"

    @staticmethod
    def _relative_artifact_name(
        *,
        artifact_directory: Path,
        artifact_file: Path,
    ) -> str:
        try:
            return artifact_file.relative_to(artifact_directory).as_posix()
        except ValueError:
            return artifact_file.name

    @classmethod
    def log_artifacts(
        cls,
        *,
        artifact_directory: Path,
        artifact_files: list[Path],
        log_artifact: Callable[[str, str | None], None],
    ) -> list[str]:
        """Log artifacts using provided callback and return uploaded names."""
        uploaded_artifacts = cls.uploaded_artifact_names(
            artifact_directory=artifact_directory,
            artifact_files=artifact_files,
        )
        for artifact_file in artifact_files:
            artifact_path = cls.resolve_artifact_path(
                artifact_directory=artifact_directory,
                artifact_file=artifact_file,
            )
            log_artifact(str(artifact_file), artifact_path)
        return uploaded_artifacts

    @classmethod
    def uploaded_artifact_names(
        cls,
        *,
        artifact_directory: Path,
        artifact_files: list[Path],
    ) -> list[str]:
        """Return the relative artifact names that will be recorded in metadata."""
        return [
            cls._relative_artifact_name(
                artifact_directory=artifact_directory,
                artifact_file=artifact_file,
            )
            for artifact_file in artifact_files
        ]

    @classmethod
    def upload_artifacts_to_run(
        cls,
        *,
        tracking_uri: str,
        run_id: str,
        artifact_directory: Path,
        artifact_files: list[Path],
    ) -> list[str]:
        """Upload artifacts to an existing MLflow run."""
        mlflow = cls._import_mlflow_module()
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_id=run_id):
            return cls.log_artifacts(
                artifact_directory=artifact_directory,
                artifact_files=artifact_files,
                log_artifact=mlflow.log_artifact,
            )

    def _export_sync(self) -> None:
        mlflow = self._import_mlflow_module()
        try:
            from mlflow.entities import Metric, Param, RunTag
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise RuntimeError(
                mlflow_dependency_message("MLflow export is enabled")
            ) from exc

        if self._tracking_uri is None:
            raise RuntimeError("MLflow tracking URI is unexpectedly None")

        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        client = MlflowClient()

        existing_metadata = self._load_existing_metadata()
        existing_live_run_id = self._resolve_live_streaming_run_id(existing_metadata)
        existing_live_run_name = existing_metadata.get("run_name")
        run_name = self._run_name
        if run_name is None and isinstance(existing_live_run_name, str):
            normalized_run_name = existing_live_run_name.strip()
            if normalized_run_name:
                run_name = normalized_run_name
        run_name = run_name or self._derive_default_run_name()
        timestamp_ms = int(time.time() * 1000)
        metric_payload = self._build_metric_payload()
        param_payload = self._build_param_payload()
        tag_payload = self._build_tag_payload()
        uploaded_artifacts: list[str] = []

        if existing_live_run_id is not None:
            run_context = mlflow.start_run(run_id=existing_live_run_id)
        else:
            run_context = mlflow.start_run(run_name=run_name)

        with run_context as run:
            run_id = run.info.run_id

            metrics = [
                Metric(key, value, timestamp_ms, 0)
                for key, value in metric_payload.items()
            ]
            params = [Param(key, value) for key, value in param_payload.items()]
            tags = [RunTag(key, value) for key, value in tag_payload.items()]

            if metrics or params or tags:
                client.log_batch(
                    run_id=run_id,
                    metrics=metrics,
                    params=params,
                    tags=tags,
                )

            artifact_files = self._iter_artifact_files()
            artifact_files = [
                artifact_file
                for artifact_file in artifact_files
                if artifact_file.resolve() != self._metadata_file.resolve()
            ]
            uploaded_artifacts, failed_artifacts = self._log_artifacts_for_export(
                artifact_files=artifact_files,
                log_artifact=mlflow.log_artifact,
            )
            self._write_export_metadata(
                run_id=run_id,
                run_name=run_name,
                metric_keys=sorted(metric_payload),
                param_keys=sorted(param_payload),
                tag_keys=sorted(tag_payload),
                uploaded_artifacts=uploaded_artifacts,
                failed_artifacts=failed_artifacts,
                reused_live_run=existing_live_run_id is not None,
                live_streaming=bool(existing_metadata.get("live_streaming")),
            )
            try:
                self.log_artifacts(
                    artifact_directory=self._artifact_directory,
                    artifact_files=[self._metadata_file],
                    log_artifact=mlflow.log_artifact,
                )
            except Exception as exc:
                self.warning(f"Failed to upload MLflow export metadata: {exc!r}")
        self.info(
            f"Uploaded MLflow run '{run_name}' ({run_id}) with "
            f"{len(metric_payload)} metrics and {len(uploaded_artifacts)} artifacts."
        )

    def _derive_default_run_name(self) -> str:
        benchmark_id = self._user_config.benchmark_id
        if benchmark_id:
            return f"aiperf-{benchmark_id[:8]}"
        return f"aiperf-{int(time.time())}"

    def _build_metric_payload(self) -> dict[str, float]:
        payload: dict[str, float] = {}
        for metric in self._results.records or []:
            if metric.avg is None:
                continue
            try:
                payload[metric.tag] = float(metric.avg)
            except (TypeError, ValueError):
                self.debug(
                    f"Skipping non-numeric metric for MLflow export: {metric.tag}"
                )
        payload["aiperf.completed_requests"] = float(self._results.completed)
        if self._results.total_expected is not None:
            payload["aiperf.total_expected_requests"] = float(
                self._results.total_expected
            )
        return payload

    def _build_param_payload(self) -> dict[str, str]:
        params: dict[str, str] = {
            "endpoint.type": str(self._user_config.endpoint.type),
            "endpoint.models": ",".join(self._user_config.endpoint.model_names),
            "endpoint.urls": ",".join(self._user_config.endpoint.urls),
            "timing.mode": str(self._user_config.timing_mode),
            "output.artifact_directory": str(
                self._user_config.output.artifact_directory
            ),
        }

        if self._user_config.loadgen.concurrency is not None:
            params["loadgen.concurrency"] = str(self._user_config.loadgen.concurrency)
        if self._user_config.loadgen.request_rate is not None:
            params["loadgen.request_rate"] = str(self._user_config.loadgen.request_rate)
        if self._user_config.loadgen.request_count is not None:
            params["loadgen.request_count"] = str(
                self._user_config.loadgen.request_count
            )
        if self._user_config.loadgen.benchmark_duration is not None:
            params["loadgen.benchmark_duration"] = str(
                self._user_config.loadgen.benchmark_duration
            )
        if self._user_config.cli_command:
            params["aiperf.cli_command"] = self._user_config.cli_command

        return params

    def _build_tag_payload(self) -> dict[str, str]:
        from aiperf import __version__ as aiperf_version

        tags = {
            "aiperf.version": aiperf_version,
            "aiperf.was_cancelled": str(self._results.was_cancelled).lower(),
        }
        if self._user_config.benchmark_id:
            tags["benchmark_id"] = self._user_config.benchmark_id
        tags.update(self._user_config.mlflow_tags_dict)
        return tags

    def _iter_artifact_files(self) -> list[Path]:
        files: list[Path] = []
        seen: set[str] = set()
        for pattern in self._artifact_globs:
            for candidate in sorted(self._artifact_directory.glob(pattern)):
                if not candidate.is_file():
                    continue
                resolved = str(candidate.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                files.append(candidate)
        return files

    def _log_artifacts_for_export(
        self,
        *,
        artifact_files: list[Path],
        log_artifact: Callable[[str, str | None], None],
    ) -> tuple[list[str], list[str]]:
        uploaded_artifacts: list[str] = []
        failed_artifacts: list[str] = []
        for artifact_file in artifact_files:
            artifact_name = self._relative_artifact_name(
                artifact_directory=self._artifact_directory,
                artifact_file=artifact_file,
            )
            artifact_path = self.resolve_artifact_path(
                artifact_directory=self._artifact_directory,
                artifact_file=artifact_file,
            )
            try:
                log_artifact(str(artifact_file), artifact_path)
            except Exception as exc:
                self.warning(
                    f"Failed to upload MLflow artifact {artifact_name}: {exc!r}"
                )
                failed_artifacts.append(artifact_name)
                continue
            uploaded_artifacts.append(artifact_name)
        return uploaded_artifacts, failed_artifacts

    def _load_existing_metadata(self) -> dict[str, Any]:
        if not self._metadata_file.exists():
            return {}
        try:
            metadata = orjson.loads(self._metadata_file.read_bytes())
        except orjson.JSONDecodeError:
            self.warning(
                f"Ignoring malformed MLflow metadata file: {self._metadata_file}"
            )
            return {}
        if not isinstance(metadata, dict):
            self.warning(
                "Ignoring unexpected MLflow metadata payload type in "
                f"{self._metadata_file}: {type(metadata).__name__}"
            )
            return {}
        return metadata

    def _resolve_live_streaming_run_id(self, metadata: dict[str, Any]) -> str | None:
        if metadata.get("live_streaming") is not True:
            return None

        metadata_run_id = metadata.get("run_id")
        metadata_tracking_uri = metadata.get("tracking_uri")
        metadata_benchmark_id = metadata.get("benchmark_id")
        if (
            not isinstance(metadata_run_id, str)
            or not metadata_run_id
            or metadata_tracking_uri != self._tracking_uri
        ):
            return None

        current_benchmark_id = self._user_config.benchmark_id
        if (
            not isinstance(metadata_benchmark_id, str)
            or metadata_benchmark_id != current_benchmark_id
        ):
            return None
        return metadata_run_id

    def _write_export_metadata(
        self,
        *,
        run_id: str,
        run_name: str,
        metric_keys: list[str],
        param_keys: list[str],
        tag_keys: list[str],
        uploaded_artifacts: list[str],
        failed_artifacts: list[str],
        reused_live_run: bool,
        live_streaming: bool,
    ) -> None:
        self._artifact_directory.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, Any] = {
            "tracking_uri": self._tracking_uri,
            "experiment": self._experiment_name,
            "run_id": run_id,
            "run_name": run_name,
            "benchmark_id": self._user_config.benchmark_id,
            "live_streaming": live_streaming,
            "reused_live_run": reused_live_run,
            "metric_keys": metric_keys,
            "param_keys": param_keys,
            "tag_keys": tag_keys,
            "uploaded_artifacts": uploaded_artifacts,
            "failed_artifacts": failed_artifacts,
            "exported_at_ns": time.time_ns(),
        }
        self._metadata_file.write_bytes(
            orjson.dumps(metadata, option=orjson.OPT_INDENT_2)
        )
