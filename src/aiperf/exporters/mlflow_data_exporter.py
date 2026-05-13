# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import orjson

from aiperf.common.config import MLflowDefaults
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.optional_dependencies import mlflow_dependency_message
from aiperf.common.redact import redact_cli_command, redact_url
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.mlflow_metadata import (
    MLflowExportMetadata,
    normalize_mlflow_uri,
)


class MLflowDataExporter(AIPerfLoggerMixin):
    """Uploads benchmark summary metrics and artifacts to MLflow Tracking."""

    _PLOT_SUFFIXES: frozenset[str] = frozenset(
        {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".html"}
    )
    is_deferred = True  # runs after all local exporters write their files

    def __init__(self, exporter_config: ExporterConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Keep the whole config so we can ship it to the export subprocess
        # without re-plumbing through kwargs.
        self._exporter_config = exporter_config
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
        """Run blocking MLflow client operations in a terminable subprocess."""
        from aiperf.common.environment import Environment
        from aiperf.exporters.mlflow_export_subprocess import export_with_timeout

        await export_with_timeout(
            exporter_config=self._exporter_config,
            export_timeout=Environment.MLFLOW.EXPORT_TIMEOUT_SECONDS,
            warn=self.warning,
        )

    @classmethod
    def _import_mlflow_module(cls) -> Any:
        """Import mlflow with a consistent dependency error message."""
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                mlflow_dependency_message("MLflow export is enabled")
            ) from exc
        return mlflow

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

        is_plot = artifact_file.suffix.lower() in cls._PLOT_SUFFIXES
        base = "plots" if is_plot else "exports"
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

        assert self._tracking_uri is not None  # invariant: __init__ guards None
        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        client = MlflowClient()

        existing_metadata = self._load_existing_metadata()
        existing_live_run_id = self._resolve_live_streaming_run_id(existing_metadata)
        timestamp_ms = int(time.time() * 1000)
        metric_payload = self._build_metric_payload()
        param_payload = self._build_param_payload()
        tag_payload = self._build_tag_payload()
        uploaded_artifacts: list[str] = []

        reused_live_run = existing_live_run_id is not None
        if reused_live_run:
            # On reuse, carry forward the parent_run_id from the live metadata.
            resolved_parent_run_id: str | None = existing_metadata.get("parent_run_id")
            cli_parent = self._user_config.mlflow_parent_run_id
            if cli_parent and cli_parent != resolved_parent_run_id:
                self.info("parent_run_id ignored on live-run reuse")
            run_context = mlflow.start_run(run_id=existing_live_run_id)
        else:
            resolved_parent_run_id = self._user_config.mlflow_parent_run_id
            # Fresh run: compute the name up front so _start_new_run can pass it
            # to mlflow.start_run. Reused runs keep the name MLflow already stored.
            new_run_name = self._run_name or self._derive_default_run_name()
            run_context, resolved_parent_run_id = self._start_new_run(
                mlflow, new_run_name, resolved_parent_run_id
            )

        with run_context as run:
            run_id = run.info.run_id
            # Authoritative run name = the name MLflow actually has on disk.
            # On reuse, the fanout may have let MLflow auto-generate one when
            # the user did not pass --mlflow-run-name, so re-reading from the
            # run info is the only way to avoid a metadata / MLflow desync.
            run_name = run.info.run_name or self._derive_default_run_name()

            # Log batched metrics/params/tags atomically (single round-trip).
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

            # Enumerate artifact files, excluding mlflow_export.json (written below).
            artifact_files = self._collect_artifact_files()

            # Compute uploaded_artifact_names including mlflow_export.json itself.
            uploaded_artifacts = self.uploaded_artifact_names(
                artifact_directory=self._artifact_directory,
                artifact_files=artifact_files,
            ) + [MLflowDefaults.EXPORT_METADATA_FILE.name]

            # Write final mlflow_export.json to disk before upload to guarantee
            # byte-equality between what we hash locally and what MLflow stores.
            self._write_export_metadata(
                run_id=run_id,
                run_name=run_name,
                metric_keys=sorted(metric_payload),
                param_keys=sorted(param_payload),
                tag_keys=sorted(tag_payload),
                uploaded_artifacts=uploaded_artifacts,
                reused_live_run=reused_live_run,
                live_streaming=bool(existing_metadata.get("live_streaming")),
                parent_run_id=resolved_parent_run_id,
            )

            # Upload all artifacts plus mlflow_export.json in one pass so a partial
            # upload doesn't leave the run in an inconsistent state.
            self.log_artifacts(
                artifact_directory=self._artifact_directory,
                artifact_files=artifact_files,
                log_artifact=mlflow.log_artifact,
            )
            mlflow.log_artifact(
                str(self._metadata_file),
                self.resolve_artifact_path(
                    artifact_directory=self._artifact_directory,
                    artifact_file=self._metadata_file,
                ),
            )
        self.info(
            f"Uploaded MLflow run '{run_name}' ({run_id}) with "
            f"{len(metric_payload)} metrics and {len(uploaded_artifacts)} artifacts."
        )

    def _start_new_run(
        self, mlflow: Any, run_name: str, parent_run_id: str | None
    ) -> tuple[Any, str | None]:
        """Start a new MLflow run, falling back to a root run if parent is missing."""
        start_kwargs: dict[str, Any] = {"run_name": run_name}
        if parent_run_id:
            start_kwargs["parent_run_id"] = parent_run_id
        try:
            return mlflow.start_run(**start_kwargs), parent_run_id
        except Exception as exc:
            if not parent_run_id or not self._is_parent_missing(exc):
                raise
            self.warning(
                f"parent_run_id {parent_run_id} not found; "
                f"creating root MLflow run instead. Original error: {exc!r}"
            )
            return mlflow.start_run(run_name=run_name), None

    @staticmethod
    def _is_parent_missing(exc: BaseException) -> bool:
        """Check whether the exception indicates the parent run doesn't exist."""
        try:
            from mlflow.exceptions import MlflowException

            if (
                isinstance(exc, MlflowException)
                and getattr(exc, "error_code", None) == "RESOURCE_DOES_NOT_EXIST"
            ):
                return True
        except ImportError:
            pass
        # Fallback for older mlflow versions without error_code.
        return "RESOURCE_DOES_NOT_EXIST" in repr(exc)

    def _derive_default_run_name(self) -> str:
        benchmark_id = self._user_config.benchmark_id
        if benchmark_id:
            return f"aiperf-{benchmark_id[:8]}"
        return f"aiperf-{int(time.time())}"

    # Statistic fields on JsonMetricResult / MetricResult that are pushed to
    # MLflow. The exporter skips fields that are None, so listing a superset is
    # safe — metrics that don't produce a given percentile simply omit it. Keep
    # in sync with JsonMetricResult in common/models/export_models.py.
    _STAT_FIELDS = ("avg", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "min", "max", "std")  # fmt: skip

    def _build_metric_payload(self) -> dict[str, float]:
        payload: dict[str, float] = {}
        for metric in self._results.records or []:
            for field in self._STAT_FIELDS:
                value = getattr(metric, field, None)
                if value is None:
                    continue
                try:
                    key = metric.tag if field == "avg" else f"{metric.tag}.{field}"
                    payload[key] = float(value)
                except (TypeError, ValueError):
                    self.debug(
                        f"Skipping non-numeric metric for MLflow export: "
                        f"{metric.tag}.{field}"
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
            "endpoint.urls": ",".join(
                redact_url(url) for url in self._user_config.endpoint.urls
            ),
            "timing.mode": str(self._user_config.timing_mode),
            "output.artifact_directory": str(
                self._user_config.output.artifact_directory
            ),
        }

        if self._user_config.loadgen.concurrency is not None:
            params["loadgen.concurrency"] = str(self._user_config.loadgen.concurrency)
        if self._user_config.loadgen.request_rate is not None:
            params["loadgen.request_rate"] = str(self._user_config.loadgen.request_rate)
        loadgen = self._user_config.loadgen
        if loadgen.request_count is not None:
            params["loadgen.request_count"] = str(loadgen.request_count)
        if loadgen.benchmark_duration is not None:
            params["loadgen.benchmark_duration"] = str(loadgen.benchmark_duration)
        if self._user_config.cli_command:
            params["aiperf.cli_command"] = redact_cli_command(
                self._user_config.cli_command
            )

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

    def _collect_artifact_files(self) -> list[Path]:
        """Enumerate artifact files matching configured globs, excluding mlflow_export.json."""
        files: list[Path] = []
        seen: set[str] = set()
        metadata_resolved = str(self._metadata_file.resolve())
        for pattern in self._artifact_globs:
            for candidate in sorted(self._artifact_directory.glob(pattern)):
                if not candidate.is_file():
                    continue
                resolved = str(candidate.resolve())
                if resolved in seen:
                    continue
                # Exclude the metadata file; it is written after enumeration
                # and uploaded separately to guarantee byte-equality.
                if resolved == metadata_resolved:
                    continue
                seen.add(resolved)
                files.append(candidate)
        return files

    def _load_existing_metadata(self) -> MLflowExportMetadata:
        if not self._metadata_file.exists():
            return {}
        try:
            metadata = orjson.loads(self._metadata_file.read_bytes())
        except orjson.JSONDecodeError:
            self.warning(f"Ignoring malformed MLflow metadata: {self._metadata_file}")
            return {}
        # Runtime guard: mlflow_export.json may be hand-edited or corrupted.
        if not isinstance(metadata, dict):
            self.warning(
                f"Ignoring unexpected MLflow metadata in {self._metadata_file}: "
                f"{type(metadata).__name__}"
            )
            return {}
        return cast(MLflowExportMetadata, metadata)

    def _resolve_live_streaming_run_id(
        self, metadata: MLflowExportMetadata
    ) -> str | None:
        if metadata.get("live_streaming") is not True:
            return None

        metadata_run_id = metadata.get("run_id")
        metadata_tracking_uri = metadata.get("tracking_uri")
        metadata_benchmark_id = metadata.get("benchmark_id")
        # Redact both sides so same-backend reuse still matches after userinfo
        # has been stripped from the on-disk URI (see _write_export_metadata).
        disk_uri = (
            redact_url(metadata_tracking_uri)
            if isinstance(metadata_tracking_uri, str)
            else None
        )
        memory_uri = redact_url(self._tracking_uri) if self._tracking_uri else None
        if (
            not isinstance(metadata_run_id, str)
            or not metadata_run_id
            or normalize_mlflow_uri(disk_uri) != normalize_mlflow_uri(memory_uri)
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
        reused_live_run: bool,
        live_streaming: bool,
        parent_run_id: str | None = None,
    ) -> None:
        self._artifact_directory.mkdir(parents=True, exist_ok=True)
        # mlflow_export.json is uploaded as a run artifact; redact userinfo so
        # credentials never round-trip. Reuse still works because
        # _resolve_live_streaming_run_id redacts both sides before comparing.
        assert self._tracking_uri is not None  # invariant: _export_sync guards None
        metadata: MLflowExportMetadata = {
            "tracking_uri": redact_url(self._tracking_uri),
            "experiment": self._experiment_name,
            "run_id": run_id,
            "run_name": run_name,
            "benchmark_id": self._user_config.benchmark_id,
            "parent_run_id": parent_run_id,
            "live_streaming": live_streaming,
            "reused_live_run": reused_live_run,
            "metric_keys": metric_keys,
            "param_keys": param_keys,
            "tag_keys": tag_keys,
            "uploaded_artifacts": uploaded_artifacts,
            "exported_at_ns": time.time_ns(),
        }
        payload = orjson.dumps(metadata, option=orjson.OPT_INDENT_2)
        # Atomic write: write to temp file then rename to avoid corruption
        # on crash or power loss mid-write.
        tmp_file = self._metadata_file.with_suffix(".json.tmp")
        tmp_file.write_bytes(payload)
        tmp_file.replace(self._metadata_file)
