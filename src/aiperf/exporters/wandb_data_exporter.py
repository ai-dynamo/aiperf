# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, ClassVar

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.finite import is_finite_value
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.optional_dependencies import wandb_dependency_message
from aiperf.common.redact import redact_cli_command
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.metrics.metric_registry import MetricRegistry


class WandbDataExporter(AIPerfLoggerMixin):
    """Uploads the final benchmark results table to Weights & Biases."""

    is_deferred = True  # runs after all local exporters write their files

    _ARTIFACT_GLOBS: ClassVar[tuple[str, ...]] = (
        "*.json",
        "*.csv",
        "*.jsonl",
        "*.parquet",
        "*_timeslices.*",
        "**/*.png",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.svg",
        "**/*.html",
    )

    def __init__(self, exporter_config: ExporterConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Keep the whole config so we can ship it to the export subprocess
        self._exporter_config = exporter_config
        self._results = exporter_config.results
        self._cfg = exporter_config.cfg
        run = exporter_config.run
        self._benchmark_id = run.benchmark_id if run is not None else None
        self._cli_command = run.cli_command if run is not None else None

        if not self._cfg.wandb.enabled:
            raise DataExporterDisabled(
                "Weights & Biases export is disabled (set --wandb-project to enable)."
            )
        if self._results is None:
            raise DataExporterDisabled(
                "Weights & Biases export is disabled (no profile results available)."
            )

        self._artifact_directory = self._cfg.artifacts.artifact_directory

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Weights & Biases Run Upload",
            file_path=self._artifact_directory,
        )

    async def export(self) -> None:
        """Run blocking wandb client operations in a terminable subprocess."""
        from aiperf.common.environment import Environment
        from aiperf.exporters.wandb_export_subprocess import export_with_timeout

        await export_with_timeout(
            exporter_config=self._exporter_config,
            export_timeout=Environment.WANDB.EXPORT_TIMEOUT_SECONDS,
            warn=self.warning,
        )

    def _export_sync(self) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                wandb_dependency_message("Weights & Biases export is enabled")
            ) from exc

        run_name = self._cfg.wandb.run_name or self._derive_default_run_name()
        run = wandb.init(
            entity=self._cfg.wandb.entity,
            project=self._cfg.wandb.project,
            name=run_name,
            dir=str(self._artifact_directory),
            config=self._build_config_payload(),
            tags=self._build_tags(),
        )
        artifact_files: list[Path] = []
        try:
            run.log(
                {
                    "summary_metrics": wandb.Table(
                        columns=["Metric", *ConsoleMetricsExporter.STAT_COLUMN_KEYS],
                        data=self._build_metric_table_rows(),
                    )
                }
            )
            artifact_files = self._collect_artifact_files()
            if artifact_files:
                artifact = wandb.Artifact(name=f"aiperf-{run.id}", type="aiperf-run")
                for artifact_file in artifact_files:
                    artifact.add_file(
                        str(artifact_file),
                        name=artifact_file.relative_to(
                            self._artifact_directory
                        ).as_posix(),
                    )
                run.log_artifact(artifact)
        finally:
            run.finish()
        self.info(
            f"Uploaded Weights & Biases run '{run_name}' ({run.url}) with "
            f"{len(artifact_files)} artifact files."
        )

    def _derive_default_run_name(self) -> str:
        if self._benchmark_id:
            return f"aiperf-{self._benchmark_id[:8]}"
        return f"aiperf-{int(time.time())}"

    def _build_metric_table_rows(self) -> list[list[Any]]:
        """One row per metric, mirroring the console Real-Time Metrics table:
        same visibility rules, display order, short labels, and stat columns.
        """

        def label(record: Any, cls: Any) -> str:
            name = cls.short_header or record.header
            if cls.short_header and cls.short_header_hide_unit:
                return name
            return f"{name} ({record.unit})"

        visible = [
            (record, cls)
            for record in self._results.records or []
            if (cls := MetricRegistry.get_class_or_none(record.tag)) is not None
            and cls.missing_flags(ConsoleMetricsExporter.exclude_flags)
            and cls.console_group in (ConsoleMetricsExporter.console_groups or ())
        ]
        visible.sort(key=lambda rc: rc[1].display_order or sys.maxsize)

        rows: list[list[Any]] = []
        for record, cls in visible:
            row: list[Any] = [label(record, cls)]
            for stat in ConsoleMetricsExporter.STAT_COLUMN_KEYS:
                value = getattr(record, stat, None)
                row.append(round(float(value), 2) if is_finite_value(value) else None)
            rows.append(row)
        return rows

    def _build_config_payload(self) -> dict[str, Any]:
        # The full resolved benchmark config; JSON-mode serializers redact
        # secrets (api_key, auth headers, URL userinfo) at the model layer.
        config: dict[str, Any] = self._cfg.model_dump(mode="json", exclude_none=True)
        if self._cli_command:
            config["aiperf.cli_command"] = redact_cli_command(self._cli_command)
        return config

    def _build_tags(self) -> list[str]:
        from aiperf import __version__ as aiperf_version

        tags = [f"aiperf-{aiperf_version}"]
        if self._benchmark_id:
            tags.append(f"benchmark-{self._benchmark_id[:8]}")
        tags.extend(self._cfg.wandb.tags or [])
        return tags

    def _collect_artifact_files(self) -> list[Path]:
        files: list[Path] = []
        seen: set[str] = set()
        # wandb.init(dir=...) writes its own state under <artifact_dir>/wandb/
        # before collection runs; the recursive globs must not re-upload it.
        wandb_state_dir = (self._artifact_directory / "wandb").resolve()
        for pattern in self._ARTIFACT_GLOBS:
            for candidate in sorted(self._artifact_directory.glob(pattern)):
                if not candidate.is_file():
                    continue
                resolved_path = candidate.resolve()
                if resolved_path.is_relative_to(wandb_state_dir):
                    continue
                resolved = str(resolved_path)
                if resolved in seen:
                    continue
                seen.add(resolved)
                files.append(candidate)
        return files
