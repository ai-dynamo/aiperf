# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the steady-state exporters (JSON + CSV + console).

These exporters are the sole terminal sinks for
``ExporterConfig.steady_state_results`` (produced by the registered
``SteadyStateAnalyzer``). They must be registered in ``plugins.yaml`` so the
exporter manager instantiates them, and they must self-disable cleanly when no
steady-state results are available.
"""

from __future__ import annotations

from io import StringIO

import orjson
import pytest
from rich.console import Console

from aiperf.common.exceptions import ConsoleExporterDisabled, DataExporterDisabled
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.exporters.console_steady_state_exporter import ConsoleSteadyStateExporter
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.exporter_manager import ExporterManager
from aiperf.exporters.steady_state_csv_exporter import SteadyStateCsvExporter
from aiperf.exporters.steady_state_json_exporter import SteadyStateJsonExporter
from aiperf.plugin import plugins
from aiperf.plugin.enums import ConsoleExporterType, DataExporterType, PluginType
from aiperf.post_processors.steady_state_models import (
    SteadyStateSummary,
    SteadyStateWindowMetadata,
)


def _mk_metric(tag: str, unit: str = "tokens/sec") -> MetricResult:
    return MetricResult(
        tag=tag,
        header=tag.replace("_", " ").title(),
        unit=unit,
        avg=100.0,
        p50=95.0,
        p90=120.0,
        p95=125.0,
        p99=130.0,
        min=80.0,
        max=140.0,
        std=5.0,
    )


def _window_metadata() -> SteadyStateWindowMetadata:
    return SteadyStateWindowMetadata(
        ramp_up_end_ns=1_000.0,
        ramp_down_start_ns=9_000.0,
        steady_state_duration_ns=8_000.0,
        total_requests=100,
        steady_state_requests=80,
        detection_method="cusum",
        fraction_retained=0.8,
        variance_inflation_factor=1.25,
        effective_p99_sample_size=8,
        sample_size_warning=True,
    )


def _steady_summary() -> SteadyStateSummary:
    return SteadyStateSummary(
        results={"time_to_first_token": _mk_metric("time_to_first_token", "ms")},
        effective_concurrency=_mk_metric("effective_concurrency", "requests"),
        effective_throughput=_mk_metric("effective_throughput"),
        effective_prefill_throughput=_mk_metric("effective_prefill_throughput"),
        effective_generation_concurrency=_mk_metric(
            "effective_generation_concurrency", "requests"
        ),
        effective_prefill_concurrency=_mk_metric(
            "effective_prefill_concurrency", "requests"
        ),
        effective_total_throughput=_mk_metric("effective_total_throughput"),
        effective_throughput_per_user=_mk_metric("effective_throughput_per_user"),
        effective_prefill_throughput_per_user=_mk_metric(
            "effective_prefill_throughput_per_user"
        ),
        tokens_in_flight=_mk_metric("tokens_in_flight", "tokens"),
        window_metadata=_window_metadata(),
    )


def _exporter_config(config, steady=None, tmp_path=None) -> ExporterConfig:
    if tmp_path is not None:
        config.benchmark.artifacts.dir = tmp_path
    return ExporterConfig(
        results=ProfileResults(
            records=[], start_ns=0, end_ns=0, completed=0, error_summary=[]
        ),
        config=config.benchmark,
        steady_state_results=steady,
    )


class TestSteadyStateJsonExporter:
    async def test_writes_artifact_when_results_present(self, config, tmp_path):
        exporter = SteadyStateJsonExporter(
            _exporter_config(config, steady=_steady_summary(), tmp_path=tmp_path)
        )
        await exporter.export()

        out = config.benchmark.artifacts.profile_export_steady_state_json_file
        assert out.exists()
        data = orjson.loads(out.read_text())
        assert data["window_metadata"]["detection_method"] == "cusum"
        assert "effective_throughput" in data["metrics"]

    def test_self_disables_when_results_absent(self, config):
        with pytest.raises(DataExporterDisabled):
            SteadyStateJsonExporter(_exporter_config(config, steady=None))


class TestSteadyStateCsvExporter:
    async def test_writes_artifact_when_results_present(self, config, tmp_path):
        exporter = SteadyStateCsvExporter(
            _exporter_config(config, steady=_steady_summary(), tmp_path=tmp_path)
        )
        await exporter.export()

        out = config.benchmark.artifacts.profile_export_steady_state_csv_file
        assert out.exists()
        content = out.read_text()
        assert "Steady-State Window Metadata" in content
        assert "detection_method" in content

    def test_self_disables_when_results_absent(self, config):
        with pytest.raises(DataExporterDisabled):
            SteadyStateCsvExporter(_exporter_config(config, steady=None))


class TestConsoleSteadyStateExporter:
    async def _render(self, exporter: ConsoleSteadyStateExporter) -> str:
        output = StringIO()
        await exporter.export(Console(file=output, width=120, legacy_windows=False))
        return output.getvalue()

    async def test_renders_table_when_results_present(self, config):
        exporter = ConsoleSteadyStateExporter(
            _exporter_config(config, steady=_steady_summary())
        )
        out = await self._render(exporter)
        assert "Steady-State Metrics" in out
        assert "Throughput" in out

    def test_self_disables_when_results_absent(self, config):
        with pytest.raises(ConsoleExporterDisabled):
            ConsoleSteadyStateExporter(_exporter_config(config, steady=None))


class TestSteadyStateExportersRegistered:
    """The registration regression these tests guard against: the modules
    exist but were absent from plugins.yaml, so the manager never loaded them."""

    def test_registered_in_plugin_registry(self):
        data = {e.name: cls for e, cls in plugins.iter_all(PluginType.DATA_EXPORTER)}
        console = {
            e.name: cls for e, cls in plugins.iter_all(PluginType.CONSOLE_EXPORTER)
        }
        assert data[DataExporterType.STEADY_STATE_JSON] is SteadyStateJsonExporter
        assert data[DataExporterType.STEADY_STATE_CSV] is SteadyStateCsvExporter
        assert console[ConsoleExporterType.STEADY_STATE] is ConsoleSteadyStateExporter

    async def test_manager_writes_steady_state_artifacts(self, config, tmp_path):
        config.benchmark.artifacts.dir = tmp_path
        manager = ExporterManager(
            results=ProfileResults(
                records=[], start_ns=0, end_ns=0, completed=0, error_summary=[]
            ),
            config=config.benchmark,
            telemetry_results=None,
            steady_state_results=_steady_summary(),
        )
        await manager.export_data()
        assert config.benchmark.artifacts.profile_export_steady_state_json_file.exists()
        assert config.benchmark.artifacts.profile_export_steady_state_csv_file.exists()

    async def test_manager_skips_steady_state_when_absent(self, config, tmp_path):
        config.benchmark.artifacts.dir = tmp_path
        manager = ExporterManager(
            results=ProfileResults(
                records=[], start_ns=0, end_ns=0, completed=0, error_summary=[]
            ),
            config=config.benchmark,
            telemetry_results=None,
            steady_state_results=None,
        )
        await manager.export_data()
        assert not config.benchmark.artifacts.profile_export_steady_state_json_file.exists()
        assert (
            not config.benchmark.artifacts.profile_export_steady_state_csv_file.exists()
        )
