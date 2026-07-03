# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the energy-efficiency exporters (data + console).

These exporters are the sole terminal sinks for
``ExporterConfig.energy_efficiency_results`` (populated controller-side by
``compute_energy_efficiency_from_summaries``). They must be registered in
``plugins.yaml`` so the exporter manager instantiates them, and they must
self-disable cleanly when no energy results are available.
"""

from __future__ import annotations

from io import StringIO

import orjson
import pytest
from rich.console import Console

from aiperf.analysis.energy_analyzer import EnergyEfficiencySummary, EnergySource
from aiperf.common.exceptions import ConsoleExporterDisabled, DataExporterDisabled
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.exporters.console_energy_exporter import ConsoleEnergyExporter
from aiperf.exporters.energy_json_exporter import EnergyJsonExporter
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.exporter_manager import ExporterManager
from aiperf.plugin import plugins
from aiperf.plugin.enums import ConsoleExporterType, DataExporterType, PluginType


def _energy_summary() -> EnergyEfficiencySummary:
    return EnergyEfficiencySummary(
        total_gpu_energy_j=1234.5,
        average_gpu_power_w=250.0,
        gpu_count=2,
        energy_source=EnergySource.DCGM_COUNTER,
        energy_per_output_token_mj=5.0,
        energy_per_request_j=12.3,
        energy_per_total_token_mj=3.0,
        performance_per_watt=0.04,
        output_tps_per_watt=1.5,
        goodput_per_watt=0.03,
        metric_results={
            "total_gpu_energy": MetricResult(
                tag="total_gpu_energy",
                header="Total GPU Energy",
                unit="J",
                avg=1234.5,
            )
        },
    )


def _exporter_config(config, energy=None, tmp_path=None) -> ExporterConfig:
    if tmp_path is not None:
        config.benchmark.artifacts.dir = tmp_path
    return ExporterConfig(
        results=ProfileResults(
            records=[], start_ns=0, end_ns=0, completed=0, error_summary=[]
        ),
        config=config.benchmark,
        energy_efficiency_results=energy,
    )


class TestEnergyJsonExporter:
    async def test_writes_artifact_when_results_present(self, config, tmp_path):
        exporter = EnergyJsonExporter(
            _exporter_config(config, energy=_energy_summary(), tmp_path=tmp_path)
        )
        await exporter.export()

        out = config.benchmark.artifacts.profile_export_energy_efficiency_json_file
        assert out.exists()
        data = orjson.loads(out.read_text())
        assert data["source"]["gpu_count"] == 2
        assert data["metrics"]["performance_per_watt"] == 0.04

    def test_self_disables_when_results_absent(self, config):
        with pytest.raises(DataExporterDisabled):
            EnergyJsonExporter(_exporter_config(config, energy=None))


class TestConsoleEnergyExporter:
    async def _render(self, exporter: ConsoleEnergyExporter) -> str:
        output = StringIO()
        await exporter.export(Console(file=output, width=120, legacy_windows=False))
        return output.getvalue()

    async def test_renders_table_when_results_present(self, config):
        exporter = ConsoleEnergyExporter(
            _exporter_config(config, energy=_energy_summary())
        )
        out = await self._render(exporter)
        assert "Energy Efficiency Metrics" in out
        assert "Performance Per Watt" in out

    def test_self_disables_when_results_absent(self, config):
        with pytest.raises(ConsoleExporterDisabled):
            ConsoleEnergyExporter(_exporter_config(config, energy=None))


class TestEnergyExportersRegistered:
    """The registration regression these tests guard against: the modules
    exist but were absent from plugins.yaml, so the manager never loaded them."""

    def test_registered_in_plugin_registry(self):
        data = {e.name: cls for e, cls in plugins.iter_all(PluginType.DATA_EXPORTER)}
        console = {
            e.name: cls for e, cls in plugins.iter_all(PluginType.CONSOLE_EXPORTER)
        }
        assert data[DataExporterType.ENERGY_EFFICIENCY_JSON] is EnergyJsonExporter
        assert console[ConsoleExporterType.ENERGY_EFFICIENCY] is ConsoleEnergyExporter

    async def test_manager_writes_energy_artifact(self, config, tmp_path):
        config.benchmark.artifacts.dir = tmp_path
        manager = ExporterManager(
            results=ProfileResults(
                records=[], start_ns=0, end_ns=0, completed=0, error_summary=[]
            ),
            config=config.benchmark,
            telemetry_results=None,
            energy_efficiency_results=_energy_summary(),
        )
        await manager.export_data()
        assert config.benchmark.artifacts.profile_export_energy_efficiency_json_file.exists()

    async def test_manager_skips_energy_artifact_when_absent(self, config, tmp_path):
        config.benchmark.artifacts.dir = tmp_path
        manager = ExporterManager(
            results=ProfileResults(
                records=[], start_ns=0, end_ns=0, completed=0, error_summary=[]
            ),
            config=config.benchmark,
            telemetry_results=None,
            energy_efficiency_results=None,
        )
        await manager.export_data()
        assert not config.benchmark.artifacts.profile_export_energy_efficiency_json_file.exists()
