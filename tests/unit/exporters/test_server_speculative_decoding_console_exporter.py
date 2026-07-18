# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from rich.console import Console

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.common.models import (
    GaugeMetricData,
    GaugeSeries,
    GaugeStats,
    ProfileResults,
    ServerMetricsEndpointInfo,
    ServerMetricsEndpointSummary,
    ServerMetricsResults,
)
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.server_speculative_decoding_console_exporter import (
    ServerSpeculativeDecodingConsoleExporter,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import create_exporter_config


def _endpoint_info() -> ServerMetricsEndpointInfo:
    return ServerMetricsEndpointInfo(
        total_fetches=3,
        first_fetch_ns=1_000_000_000,
        last_fetch_ns=3_000_000_000,
        avg_fetch_latency_ms=1.0,
        unique_updates=3,
        first_update_ns=1_000_000_000,
        last_update_ns=3_000_000_000,
        duration_seconds=2.0,
        avg_update_interval_ms=1_000.0,
    )


def _metric(
    avg: float, min_value: float, max_value: float, p50: float, p90: float
) -> GaugeMetricData:
    return GaugeMetricData(
        type=PrometheusMetricType.GAUGE,
        description="speculative decoding metric",
        series=[
            GaugeSeries(
                stats=GaugeStats(
                    avg=avg,
                    min=min_value,
                    max=max_value,
                    p50=p50,
                    p90=p90,
                )
            )
        ],
    )


def _config(server_metrics_results: ServerMetricsResults) -> ExporterConfig:
    return create_exporter_config(
        profile_results=ProfileResults(records=[], start_ns=0, end_ns=0, completed=0),
        cli_config=CLIConfig(
            endpoint_type=EndpointType.CHAT,
            model_names=["test-model"],
        ),
        server_metrics_results=server_metrics_results,
    )


def _results(metrics: dict[str, GaugeMetricData]) -> ServerMetricsResults:
    endpoint = ServerMetricsEndpointSummary(
        endpoint_url="http://localhost:8081/metrics",
        info=_endpoint_info(),
        metrics=metrics,
    )
    return ServerMetricsResults(
        endpoint_summaries={"localhost:8081": endpoint},
        start_ns=1_000_000_000,
        end_ns=3_000_000_000,
        endpoints_configured=["http://localhost:8081/metrics"],
        endpoints_successful=["http://localhost:8081/metrics"],
    )


@pytest.mark.asyncio
async def test_export_prints_sglang_speculative_decoding_table(capsys) -> None:
    server_metrics_results = _results(
        {
            "sglang:spec_accept_rate": _metric(0.695, 0.5, 0.9, 0.7, 0.86),
            "sglang:spec_accept_length": _metric(2.78125, 1.5, 4.0, 2.75, 3.8),
        }
    )

    exporter = ServerSpeculativeDecodingConsoleExporter(_config(server_metrics_results))
    await exporter.export(Console(width=115))

    output = capsys.readouterr().out
    assert "NVIDIA AIPerf | Server Metrics: Speculative Decoding" in output
    assert "SGLang Spec Accept Rate (%)" in output
    assert "69.5" in output
    assert "50.0" in output
    assert "90.0" in output
    assert "70.0" in output
    assert "86.0" in output
    assert "SGLang Spec Accept Length" in output
    assert "2.78" in output
    assert "1.50" in output
    assert "4.00" in output
    assert "2.75" in output
    assert "3.80" in output


def test_init_disables_when_speculative_decoding_metrics_are_missing() -> None:
    server_metrics_results = _results({})

    with pytest.raises(ConsoleExporterDisabled):
        ServerSpeculativeDecodingConsoleExporter(_config(server_metrics_results))
