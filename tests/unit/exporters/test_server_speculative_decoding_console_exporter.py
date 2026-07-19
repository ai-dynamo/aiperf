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
                labels={"model_name": "Test-Model", "pp_rank": "0", "tp_rank": "0"},
                stats=GaugeStats(
                    avg=avg,
                    min=min_value,
                    max=max_value,
                    p50=p50,
                    p90=p90,
                ),
            )
        ],
    )


def _metric_with_other_model_series(
    avg: float, min_value: float, max_value: float, p50: float, p90: float
) -> GaugeMetricData:
    metric = _metric(avg, min_value, max_value, p50, p90)
    metric.series.append(
        GaugeSeries(
            labels={"model_name": "other-model", "pp_rank": "0", "tp_rank": "0"},
            stats=GaugeStats(avg=0.01, min=0.01, max=0.01, p50=0.01, p90=0.01),
        )
    )
    metric.series.append(
        GaugeSeries(
            labels={"model_name": "test-model", "pp_rank": "1", "tp_rank": "0"},
            stats=GaugeStats(avg=0.999, min=0.999, max=0.999, p50=0.999, p90=0.999),
        )
    )
    metric.series.append(
        GaugeSeries(
            labels={"model_name": "test-model", "pp_rank": "0", "tp_rank": "1"},
            stats=GaugeStats(avg=0.888, min=0.888, max=0.888, p50=0.888, p90=0.888),
        )
    )
    return metric


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


def _results_for_endpoints(
    metrics_by_endpoint: dict[str, dict[str, GaugeMetricData]],
) -> ServerMetricsResults:
    endpoint_summaries = {
        endpoint_url: ServerMetricsEndpointSummary(
            endpoint_url=endpoint_url,
            info=_endpoint_info(),
            metrics=metrics,
        )
        for endpoint_url, metrics in metrics_by_endpoint.items()
    }
    endpoints = list(metrics_by_endpoint)
    return ServerMetricsResults(
        endpoint_summaries=endpoint_summaries,
        start_ns=1_000_000_000,
        end_ns=3_000_000_000,
        endpoints_configured=endpoints,
        endpoints_successful=endpoints,
    )


@pytest.mark.asyncio
async def test_export_prints_sglang_speculative_decoding_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    server_metrics_results = _results(
        {
            "sglang:spec_accept_rate": _metric_with_other_model_series(
                0.695, 0.5, 0.9, 0.7, 0.86
            ),
            "sglang:spec_accept_length": _metric_with_other_model_series(
                2.78125, 1.5, 4.0, 2.75, 3.8
            ),
        }
    )

    exporter = ServerSpeculativeDecodingConsoleExporter(_config(server_metrics_results))
    await exporter.export(Console(width=115))

    output = capsys.readouterr().out
    assert "NVIDIA AIPerf | Server Metrics: Speculative Decoding" in output
    assert "Accept Rate (%)" in output
    assert "69.5" in output
    assert "50.0" in output
    assert "90.0" in output
    assert "70.0" in output
    assert "86.0" in output
    assert "99.9" not in output
    assert "88.8" not in output
    assert "Accept Length" in output
    assert "2.78" in output
    assert "1.50" in output
    assert "4.00" in output
    assert "2.75" in output
    assert "3.80" in output


def test_init_disables_when_speculative_decoding_metrics_are_missing() -> None:
    server_metrics_results = _results({})

    with pytest.raises(ConsoleExporterDisabled):
        ServerSpeculativeDecodingConsoleExporter(_config(server_metrics_results))


@pytest.mark.asyncio
async def test_export_renders_multiple_matching_series_without_averaging(
    capsys: pytest.CaptureFixture[str],
) -> None:
    server_metrics_results = _results(
        {
            "sglang:spec_accept_rate": GaugeMetricData(
                type=PrometheusMetricType.GAUGE,
                description="speculative decoding metric",
                series=[
                    GaugeSeries(
                        labels={
                            "model_name": "test-model",
                            "pp_rank": "0",
                            "tp_rank": "0",
                            "dp_rank": "0",
                        },
                        stats=GaugeStats(
                            avg=0.6,
                            min=0.5,
                            max=0.62,
                            p50=0.61,
                            p90=0.63,
                        ),
                    ),
                    GaugeSeries(
                        labels={
                            "model_name": "test-model",
                            "pp_rank": "0",
                            "tp_rank": "0",
                            "dp_rank": "1",
                        },
                        stats=GaugeStats(
                            avg=0.8,
                            min=0.73,
                            max=0.82,
                            p50=0.81,
                            p90=0.83,
                        ),
                    ),
                ],
            )
        }
    )

    exporter = ServerSpeculativeDecodingConsoleExporter(_config(server_metrics_results))
    await exporter.export(Console(width=115))

    output = capsys.readouterr().out
    assert "dp_rank=0" in output
    assert "dp_rank=1" in output
    assert "60.0" in output
    assert "80.0" in output
    assert "70.0" not in output


@pytest.mark.asyncio
async def test_export_distinguishes_matching_series_from_different_endpoints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    server_metrics_results = _results_for_endpoints(
        {
            "http://host-a:8081/metrics": {
                "sglang:spec_accept_rate": _metric(0.6, 0.5, 0.62, 0.61, 0.63)
            },
            "http://host-b:8081/metrics": {
                "sglang:spec_accept_rate": _metric(0.8, 0.73, 0.82, 0.81, 0.83)
            },
        }
    )

    exporter = ServerSpeculativeDecodingConsoleExporter(_config(server_metrics_results))
    await exporter.export(Console(width=115))

    output = capsys.readouterr().out
    assert "endpoint=host-a:8081" in output
    assert "endpoint=host-b:8081" in output


@pytest.mark.asyncio
async def test_export_escapes_server_metric_labels(
    capsys: pytest.CaptureFixture[str],
) -> None:
    server_metrics_results = _results(
        {
            "sglang:spec_accept_rate": GaugeMetricData(
                type=PrometheusMetricType.GAUGE,
                description="speculative decoding metric",
                series=[
                    GaugeSeries(
                        labels={
                            "model_name": "test-model",
                            "pp_rank": "0",
                            "tp_rank": "0",
                            "engine_type": "[red]unified[/red]",
                        },
                        stats=GaugeStats(
                            avg=0.6,
                            min=0.5,
                            max=0.62,
                            p50=0.61,
                            p90=0.63,
                        ),
                    ),
                    GaugeSeries(
                        labels={
                            "model_name": "test-model",
                            "pp_rank": "0",
                            "tp_rank": "0",
                            "engine_type": "other",
                        },
                        stats=GaugeStats(
                            avg=0.8,
                            min=0.73,
                            max=0.82,
                            p50=0.81,
                            p90=0.83,
                        ),
                    ),
                ],
            )
        }
    )

    exporter = ServerSpeculativeDecodingConsoleExporter(_config(server_metrics_results))
    await exporter.export(Console(width=115))

    output = capsys.readouterr().out
    assert "engine_type=[red]unified[/red]" in output


def test_init_disables_when_speculative_decoding_model_label_does_not_match() -> None:
    server_metrics_results = _results(
        {
            "sglang:spec_accept_rate": GaugeMetricData(
                type=PrometheusMetricType.GAUGE,
                description="speculative decoding metric",
                series=[
                    GaugeSeries(
                        labels={"model_name": "other-model"},
                        stats=GaugeStats(
                            avg=0.695,
                            min=0.5,
                            max=0.9,
                            p50=0.7,
                            p90=0.86,
                        ),
                    )
                ],
            )
        }
    )

    with pytest.raises(ConsoleExporterDisabled):
        ServerSpeculativeDecodingConsoleExporter(_config(server_metrics_results))


def test_init_disables_when_speculative_decoding_stats_are_non_finite(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server_metrics_results = _results(
        {
            "sglang:spec_accept_rate": GaugeMetricData(
                type=PrometheusMetricType.GAUGE,
                description="speculative decoding metric",
                series=[
                    GaugeSeries(
                        labels={
                            "model_name": "test-model",
                            "pp_rank": "0",
                            "tp_rank": "0",
                        },
                        stats=GaugeStats(
                            avg=float("nan"),
                            min=0.5,
                            max=0.9,
                            p50=0.7,
                            p90=0.86,
                        ),
                    )
                ],
            )
        }
    )

    caplog.set_level("WARNING")

    with pytest.raises(ConsoleExporterDisabled):
        ServerSpeculativeDecodingConsoleExporter(_config(server_metrics_results))

    assert "Skipping SGLang speculative decoding console row" in caplog.text
    assert "non-finite gauge summary values" in caplog.text


def test_scaled_stat_rejects_non_finite_scaled_value() -> None:
    assert ServerSpeculativeDecodingConsoleExporter._scaled_stat(1e308, 100.0) is None
