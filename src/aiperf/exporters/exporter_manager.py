# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
from rich.console import Console

from aiperf.common.environment import Environment
from aiperf.common.exceptions import (
    ConsoleExporterDisabled,
    DataExporterDisabled,
)
from aiperf.common.finite import scrub_non_finite
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ProfileResults
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_csv_exporter import MetricsCsvExporter
from aiperf.exporters.metrics_json_exporter import MetricsJsonExporter
from aiperf.exporters.protocols import ConsoleExporterProtocol, DataExporterProtocol
from aiperf.plugin import plugins
from aiperf.plugin.enums import DataExporterType, PluginType

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class ExporterManager(AIPerfLoggerMixin):
    """
    ExporterManager is responsible for exporting records using all
    registered data exporters.
    """

    def __init__(
        self,
        *,
        results: ProfileResults,
        run: "BenchmarkRun",
        telemetry_results: TelemetryExportData | None,
        server_metrics_results: ServerMetricsResults | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._results = results
        self._run = run
        self._tasks: set[asyncio.Task] = set()
        self._exporter_config = ExporterConfig(
            results=self._results,
            cfg=run.cfg,
            telemetry_results=telemetry_results,
            server_metrics_results=server_metrics_results,
            run=run,
        )

    def _task_done_callback(self, task: asyncio.Task) -> None:
        self.debug(lambda: f"Task done: {task}")
        if task.exception():
            self.error(f"Error exporting records: {task.exception()}")
        else:
            self.debug(f"Exported records: {task.result()}")
        self._tasks.discard(task)

    async def export_data(self) -> None:
        self.info("Exporting all records")
        deferred_exporters: list[DataExporterProtocol] = []

        for exporter_entry, ExporterClass in plugins.iter_all(PluginType.DATA_EXPORTER):
            if exporter_entry.name == DataExporterType.SERVER_METRICS_PARQUET:
                # TODO: Until the exporters move to the records manager, we need to skip the
                # parquet exporter here, as it requires the server metrics accumulator to be available.
                continue

            try:
                exporter: DataExporterProtocol = ExporterClass(
                    exporter_config=self._exporter_config
                )
            except DataExporterDisabled:
                self.debug(
                    f"Data exporter {exporter_entry.name} is disabled and will not be used"
                )
                continue
            except Exception as e:
                self.error(f"Error creating data exporter: {e!r}")
                continue

            # Deferred exporters run after all local exporters finish
            # so their artifacts (JSON, CSV, etc.) are available for upload.
            if getattr(exporter, "is_deferred", False):
                deferred_exporters.append(exporter)
                continue

            self.debug(f"Creating task for exporter: {exporter_entry.name}")
            task = asyncio.create_task(exporter.export())
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        try:
            await self._export_phase_metric_artifacts()
        except (OSError, ValueError) as exc:
            self.warning(f"Failed to export phase metric artifacts: {exc}")

        for exporter in deferred_exporters:
            self.debug(f"Running deferred exporter: {exporter.__class__.__name__}")
            task = asyncio.create_task(exporter.export())
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.debug("Exporting all records completed")

    async def _export_phase_metric_artifacts(self) -> None:
        phase_records = getattr(self._results, "phase_records", None) or []
        if not phase_records:
            return
        manifest_entries: list[dict[str, Any]] = []
        for phase_result in phase_records:
            phase_dir = self._run.cfg.artifacts.dir / "phases" / phase_result.phase_name
            await asyncio.to_thread(phase_dir.mkdir, parents=True, exist_ok=True)
            completed = (
                phase_result.successful_request_count + phase_result.error_request_count
            )
            phase_profile = ProfileResults(
                records=phase_result.records,
                completed=completed,
                start_ns=phase_result.start_ns or self._results.start_ns,
                end_ns=phase_result.end_ns or self._results.end_ns,
                was_cancelled=phase_result.was_cancelled,
                successful_request_count=phase_result.successful_request_count,
                error_request_count=phase_result.error_request_count,
                error_summary=phase_result.error_summary,
            )
            entry: dict[str, Any] = {
                "phase_index": phase_result.phase_index,
                "profiling_index": phase_result.profiling_index,
                "phase_name": phase_result.phase_name,
                "phase_kind": phase_result.phase_kind,
                "start_ns": phase_result.start_ns,
                "end_ns": phase_result.end_ns,
                "was_cancelled": phase_result.was_cancelled,
                "successful_request_count": phase_result.successful_request_count,
                "error_request_count": phase_result.error_request_count,
                "total_request_count": completed,
                "error_summary": [
                    item.model_dump(mode="json") for item in phase_result.error_summary
                ],
            }
            await self._write_phase_export(
                exporter_cls=MetricsJsonExporter,
                phase_profile=phase_profile,
                file_path=phase_dir
                / self._run.cfg.artifacts.profile_export_json_file.name,
                manifest_entry=entry,
                manifest_key="metrics_json",
            )
            await self._write_phase_export(
                exporter_cls=MetricsCsvExporter,
                phase_profile=phase_profile,
                file_path=phase_dir
                / self._run.cfg.artifacts.profile_export_csv_file.name,
                manifest_entry=entry,
                manifest_key="metrics_csv",
            )
            await self._write_phase_observability_export(
                phase_result=phase_result,
                phase_dir=phase_dir,
                manifest_entry=entry,
                attr="telemetry_results",
                warnings_attr="telemetry_warnings",
                file_name="gpu_telemetry.json",
                manifest_key="gpu_telemetry_json",
            )
            await self._write_phase_observability_export(
                phase_result=phase_result,
                phase_dir=phase_dir,
                manifest_entry=entry,
                attr="server_metrics_results",
                warnings_attr="server_metrics_warnings",
                file_name="server_metrics.json",
                manifest_key="server_metrics_json",
            )
            manifest_entries.append(entry)
        try:
            await asyncio.to_thread(self._write_phase_manifest, manifest_entries)
        except (OSError, ValueError) as exc:
            self.warning(f"Failed to write phase artifact manifest: {exc}")

    async def _write_phase_observability_export(
        self,
        *,
        phase_result,
        phase_dir: Path,
        manifest_entry: dict[str, Any],
        attr: str,
        warnings_attr: str,
        file_name: str,
        manifest_key: str,
    ) -> None:
        data = getattr(phase_result, attr, None)
        warnings = list(getattr(phase_result, warnings_attr, []) or [])
        if data is None and not warnings:
            return
        file_path = phase_dir / file_name
        payload = {
            "schema_version": 1,
            "phase": {
                "phase_index": phase_result.phase_index,
                "profiling_index": phase_result.profiling_index,
                "phase_name": phase_result.phase_name,
                "phase_kind": phase_result.phase_kind,
                "start_ns": phase_result.start_ns,
                "end_ns": phase_result.end_ns,
                "was_cancelled": phase_result.was_cancelled,
            },
            "data": data.model_dump(mode="json", exclude_none=True) if data else None,
            "warnings": warnings,
        }
        try:
            content = orjson.dumps(
                scrub_non_finite(payload),
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            ).decode("utf-8")
            await asyncio.to_thread(
                file_path.write_text, content + "\n", encoding="utf-8"
            )
        except Exception as exc:
            self.error(
                f"Failed to write phase observability export {file_path}: {exc!r}"
            )
            return
        manifest_entry[manifest_key] = file_path.relative_to(
            self._run.cfg.artifacts.dir
        ).as_posix()

    async def _write_phase_export(
        self,
        *,
        exporter_cls: type[MetricsJsonExporter] | type[MetricsCsvExporter],
        phase_profile: ProfileResults,
        file_path: Path,
        manifest_entry: dict[str, Any],
        manifest_key: str,
    ) -> None:
        config = ExporterConfig(
            results=phase_profile,
            cfg=self._run.cfg,
            telemetry_results=None,
            server_metrics_results=None,
            run=self._run,
        )
        try:
            exporter = exporter_cls(exporter_config=config)
        except DataExporterDisabled:
            return
        except Exception as exc:
            self.error(
                f"Error creating phase exporter {exporter_cls.__name__} "
                f"for {manifest_entry.get('phase_name')}: {exc!r}"
            )
            return
        try:
            content = exporter._generate_content()
            await asyncio.to_thread(file_path.write_text, content, encoding="utf-8")
        except Exception as exc:
            self.error(f"Failed to write phase export {file_path}: {exc!r}")
            return
        manifest_entry[manifest_key] = file_path.relative_to(
            self._run.cfg.artifacts.dir
        ).as_posix()

    def _write_phase_manifest(self, entries: list[dict[str, Any]]) -> None:
        manifest_path = self._run.cfg.artifacts.dir / "phase_manifest.json"
        payload = {
            "schema_version": 1,
            "phases": sorted(
                entries,
                key=lambda item: (
                    item.get("phase_index")
                    if item.get("phase_index") is not None
                    else 10**9
                ),
            ),
        }
        manifest_path.write_bytes(
            orjson.dumps(
                scrub_non_finite(payload),
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            )
            + b"\n"
        )

    def get_exported_file_infos(self) -> list[FileExportInfo]:
        """Get the file infos for all exported files."""
        file_infos = []
        for exporter_entry, ExporterClass in plugins.iter_all(PluginType.DATA_EXPORTER):
            if exporter_entry.name == DataExporterType.SERVER_METRICS_PARQUET:
                # TODO: Until the exporters move to the records manager, we need to skip the
                # parquet exporter here, as it requires the server metrics accumulator to be available.
                continue

            try:
                exporter: DataExporterProtocol = ExporterClass(
                    exporter_config=self._exporter_config
                )
            except DataExporterDisabled:
                self.debug(
                    f"Data exporter {exporter_entry.name} is disabled and will not be used"
                )
                continue
            except Exception as e:
                self.error(f"Error creating data exporter: {e!r}")
                continue

            file_infos.append(exporter.get_export_info())
        return file_infos

    async def export_console(self, console: Console) -> None:
        self.info("Exporting console data")

        width = Environment.UI.CONSOLE_EXPORT_WIDTH

        # The recording console stays pinned to the configured width so the saved
        # profile_export_console.txt artifact (and non-tty CI logs) match a fixed
        # layout regardless of the live terminal size.
        recording_console = self._fixed_width_console(
            width=width,
            record=True,
            file=io.StringIO(),
            force_terminal=True,
        )
        await self._run_console_exporters(recording_console)
        await self._write_console_txt(recording_console)

        styles_enabled = (
            console.is_terminal
            and not console.no_color
            and console.color_system is not None
        )
        if console.is_terminal and styles_enabled:
            # Render a fresh copy at the live terminal's own width so interactive
            # tables aren't hard-wrapped to the fixed export width.
            live_width = getattr(console, "_width", None) or console.width
            live_console = self._fixed_width_console(
                width=live_width,
                file=console.file,
                force_terminal=True,
            )
            await self._run_console_exporters(live_console)
        else:
            if console.is_terminal:
                # Plain-text TTY (NO_COLOR / no color_system): still re-render at
                # the live terminal width so interactive tables match the pane,
                # then replay without ANSI. The .txt artifact stays on the
                # fixed-width recording pass above.
                live_width = getattr(console, "_width", None) or console.width
                replay_console = self._fixed_width_console(
                    width=live_width,
                    record=True,
                    file=io.StringIO(),
                    force_terminal=True,
                )
                await self._run_console_exporters(replay_console)
            else:
                replay_console = recording_console

            styled = replay_console.export_text(styles=styles_enabled, clear=False)
            if styled.strip():
                console.file.write(styled)
                console.file.flush()

        self.debug("Exporting console data completed")

    @staticmethod
    def _fixed_width_console(
        *,
        width: int,
        file: Any,
        record: bool = False,
        force_terminal: bool | None = None,
    ) -> Console:
        return Console(
            record=record,
            file=file,
            force_terminal=force_terminal,
            width=width,
            _environ={"TERM": "xterm", "COLUMNS": str(width)},
        )

    async def _run_console_exporters(self, console: Console) -> None:
        """Run every registered console exporter, rendering into `console`."""
        for exporter_entry, ExporterClass in plugins.iter_all(
            PluginType.CONSOLE_EXPORTER
        ):
            try:
                exporter: ConsoleExporterProtocol = ExporterClass(
                    exporter_config=self._exporter_config
                )
            except ConsoleExporterDisabled:
                self.debug(
                    f"Console exporter {exporter_entry.name} is disabled and will not be used"
                )
                continue
            except Exception as e:
                self.error(f"Error creating console exporter: {e!r}")
                continue

            self.debug(f"Creating task for exporter: {exporter_entry.name}")
            task = asyncio.create_task(exporter.export(console=console))
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _write_console_txt(self, recording_console: Console) -> None:
        """Write the recorded console output to a plain-text file."""
        try:
            txt_path = self._run.cfg.artifacts.profile_export_console_txt_file
            plain_text = recording_console.export_text(styles=False, clear=False)
            await asyncio.to_thread(txt_path.write_text, plain_text, encoding="utf-8")
            self.debug(f"Console export written to {txt_path}")
        except (OSError, ValueError) as e:
            self.warning(f"Failed to write console export file: {e}")
