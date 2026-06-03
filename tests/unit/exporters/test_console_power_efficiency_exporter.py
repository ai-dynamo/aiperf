# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from rich.console import Console

from aiperf.common.enums import MetricConsoleGroup
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter
from aiperf.exporters.console_power_efficiency_exporter import (
    ConsolePowerEfficiencyExporter,
)
from aiperf.metrics.types.power_efficiency_metrics import (
    EnergyPerUserMetric,
    OutputTokensPerJouleMetric,
    TotalGpuEnergyMetric,
    TotalGpuPowerMetric,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.exporters.conftest import make_exporter_config

EFFICIENCY_RECORDS = [
    MetricResult(
        tag="total_gpu_power", header="Total GPU Power (4 GPUs)", unit="W", avg=1558.27
    ),  # fmt: skip
    MetricResult(
        tag="total_gpu_energy",
        header="Total GPU Energy (4 GPUs)",
        unit="J",
        avg=1307261.98,
    ),  # fmt: skip
    MetricResult(
        tag="output_tokens_per_joule",
        header="Output Tokens per Joule (4 GPUs)",
        unit="tokens/J",
        avg=0.32,
    ),  # fmt: skip
    MetricResult(
        tag="energy_per_user",
        header="Energy per User (4 GPUs)",
        unit="joules/user",
        avg=163407.75,
    ),  # fmt: skip
]

NON_EFFICIENCY_RECORDS = [
    MetricResult(tag="request_latency", header="Request Latency", unit="ms", avg=15.3),
    MetricResult(
        tag="request_throughput",
        header="Request Throughput",
        unit="requests/sec",
        avg=95.0,
    ),  # fmt: skip
]


def _config(records: list[MetricResult]):
    cli_config = CLIConfig(
        endpoint_type=EndpointType.CHAT, streaming=True, model_names=["test-model"]
    )
    return make_exporter_config(
        results=ProfileResults(records=records, start_ns=0, end_ns=0, completed=0),
        cli_config=cli_config,
    )


class TestConsolePowerEfficiencyExporter:
    @pytest.mark.asyncio
    async def test_renders_efficiency_section_with_title_and_rows(self, capsys) -> None:
        """With efficiency metrics present, a 'GPU Power Efficiency (NVIDIA)' table prints."""
        exporter = ConsolePowerEfficiencyExporter(_config(EFFICIENCY_RECORDS))
        await exporter.export(Console(width=120))
        output = capsys.readouterr().out

        assert "GPU Power Efficiency (NVIDIA)" in output
        assert "Total GPU Power" in output
        assert "Total GPU Energy" in output
        assert "Output Tokens per Joule" in output
        assert "Energy per User" in output
        # These are single aggregate values, so only the average column is shown.
        for percentile_header in ("p99", "p90", "p50", "min", "max", "std"):
            assert percentile_header not in output

    def test_renders_only_average_column(self) -> None:
        """The efficiency section drops the non-statistical percentile columns."""
        assert ConsolePowerEfficiencyExporter.STAT_COLUMN_KEYS == ["avg"]

    @pytest.mark.asyncio
    async def test_omits_section_when_no_efficiency_metrics(self, capsys) -> None:
        """Without efficiency metrics (e.g. --gpu-telemetry off), nothing prints."""
        exporter = ConsolePowerEfficiencyExporter(_config(NON_EFFICIENCY_RECORDS))
        await exporter.export(Console(width=120))
        assert capsys.readouterr().out.strip() == ""

    def test_efficiency_metrics_use_gpu_power_efficiency_group(self) -> None:
        """The four efficiency metrics are tagged into the GPU_POWER_EFFICIENCY group."""
        for metric_cls in (
            TotalGpuPowerMetric,
            TotalGpuEnergyMetric,
            OutputTokensPerJouleMetric,
            EnergyPerUserMetric,
        ):
            assert metric_cls.console_group == MetricConsoleGroup.GPU_POWER_EFFICIENCY

    @pytest.mark.asyncio
    async def test_efficiency_metrics_absent_from_main_metrics_table(
        self, capsys
    ) -> None:
        """The main metrics exporter must no longer render the efficiency totals."""
        exporter = ConsoleMetricsExporter(_config(EFFICIENCY_RECORDS))
        await exporter.export(Console(width=120))
        output = capsys.readouterr().out
        assert "Total GPU Power" not in output
        assert "Total GPU Energy" not in output
