# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from rich.console import Console

from aiperf.common.enums import MetricConsoleGroup
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter
from aiperf.exporters.console_power_efficiency_exporter import (
    ConsoleAmdPowerEfficiencyExporter,
    ConsoleNvidiaPowerEfficiencyExporter,
)
from aiperf.metrics.types.power_efficiency_metrics import (
    AmdEnergyPerUserMetric,
    AmdOutputTokensPerJouleMetric,
    AmdTotalGpuEnergyMetric,
    AmdTotalGpuPowerMetric,
    NvidiaEnergyPerUserMetric,
    NvidiaOutputTokensPerJouleMetric,
    NvidiaTotalGpuEnergyMetric,
    NvidiaTotalGpuPowerMetric,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.exporters.conftest import make_exporter_config


def _records(prefix: str) -> list[MetricResult]:
    return [
        MetricResult(
            tag=f"{prefix}_total_gpu_power",
            header="Total GPU Power (4 GPUs)",
            unit="W",
            avg=1558.27,
        ),  # fmt: skip
        MetricResult(
            tag=f"{prefix}_total_gpu_energy",
            header="Total GPU Energy (4 GPUs)",
            unit="J",
            avg=1307261.98,
        ),  # fmt: skip
        MetricResult(
            tag=f"{prefix}_output_tokens_per_joule",
            header="Output Tokens per Joule (4 GPUs)",
            unit="tokens/J",
            avg=0.32,
        ),  # fmt: skip
        MetricResult(
            tag=f"{prefix}_energy_per_user",
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


VENDOR_CASES = [
    pytest.param(
        ConsoleNvidiaPowerEfficiencyExporter,
        "nvidia",
        "GPU Power Efficiency (NVIDIA)",
        id="nvidia",
    ),
    pytest.param(
        ConsoleAmdPowerEfficiencyExporter,
        "amd",
        "GPU Power Efficiency (AMD)",
        id="amd",
    ),
]  # fmt: skip


class TestConsolePowerEfficiencyExporters:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("exporter_cls, prefix, title", VENDOR_CASES)
    async def test_renders_vendor_section_with_title_and_rows(
        self, exporter_cls, prefix, title, capsys
    ) -> None:
        """Each vendor exporter prints its own titled, average-only table."""
        exporter = exporter_cls(_config(_records(prefix)))
        await exporter.export(Console(width=120))
        output = capsys.readouterr().out

        assert title in output
        assert "Total GPU Power" in output
        assert "Total GPU Energy" in output
        assert "Output Tokens per Joule" in output
        assert "Energy per User" in output
        for percentile_header in ("p99", "p90", "p50", "min", "max", "std"):
            assert percentile_header not in output

    @pytest.mark.parametrize("exporter_cls, prefix, title", VENDOR_CASES)
    def test_renders_only_average_column(self, exporter_cls, prefix, title) -> None:
        assert exporter_cls.STAT_COLUMN_KEYS == ["avg"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exporter_cls, prefix, title", VENDOR_CASES)
    async def test_omits_section_when_no_efficiency_metrics(
        self, exporter_cls, prefix, title, capsys
    ) -> None:
        """Without that vendor's metrics, nothing prints."""
        exporter = exporter_cls(_config(NON_EFFICIENCY_RECORDS))
        await exporter.export(Console(width=120))
        assert capsys.readouterr().out.strip() == ""

    @pytest.mark.asyncio
    async def test_nvidia_exporter_ignores_amd_metrics(self, capsys) -> None:
        """The NVIDIA exporter renders nothing for an AMD-only result set."""
        exporter = ConsoleNvidiaPowerEfficiencyExporter(_config(_records("amd")))
        await exporter.export(Console(width=120))
        assert "GPU Power Efficiency (NVIDIA)" not in capsys.readouterr().out

    def test_metrics_use_their_vendor_console_group(self) -> None:
        nvidia = (
            NvidiaTotalGpuPowerMetric,
            NvidiaTotalGpuEnergyMetric,
            NvidiaOutputTokensPerJouleMetric,
            NvidiaEnergyPerUserMetric,
        )
        amd = (
            AmdTotalGpuPowerMetric,
            AmdTotalGpuEnergyMetric,
            AmdOutputTokensPerJouleMetric,
            AmdEnergyPerUserMetric,
        )
        for cls in nvidia:
            assert cls.console_group == MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA
        for cls in amd:
            assert cls.console_group == MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD

    @pytest.mark.asyncio
    async def test_efficiency_metrics_absent_from_main_metrics_table(
        self, capsys
    ) -> None:
        """The main metrics exporter must not render either vendor's efficiency totals."""
        exporter = ConsoleMetricsExporter(_config(_records("nvidia") + _records("amd")))
        await exporter.export(Console(width=120))
        output = capsys.readouterr().out
        assert "Total GPU Power" not in output
        assert "Total GPU Energy" not in output
