# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson

from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    MetricResult,
    ServerMetricsResults,
)
from aiperf.common.models.record_models import ProfileResults
from aiperf.config import BenchmarkRun
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.metrics_json_exporter import MetricsJsonExporter


def test_telemetry_failure_is_separate_from_complete_inference_results(
    benchmark_run: BenchmarkRun,
) -> None:
    """Missing optional telemetry cannot masquerade as failed inference requests."""
    results = ProfileResults(
        records=[
            MetricResult(tag="request_latency", header="Latency", unit="ms", avg=2.0)
        ],
        completed=916,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        is_complete=True,
    )
    error = ErrorDetails(
        type="ServerMetricsRestarted",
        message="Server metrics container restarted; telemetry is incomplete",
    )
    server_results = ServerMetricsResults(
        start_ns=0,
        end_ns=0,
        error_summary=[ErrorDetailsCount(error_details=error, count=1)],
    )
    exporter = MetricsJsonExporter(
        exporter_config=ExporterConfig(
            telemetry_results=None,
            run=benchmark_run,
            cfg=benchmark_run.cfg,
            results=results,
            server_metrics_results=server_results,
        )
    )
    content = orjson.loads(exporter._generate_content())
    assert content["is_complete"] is True
    assert not content.get("error_summary")
    assert (
        content["server_metrics_error_summary"][0]["error_details"]["type"]
        == "ServerMetricsRestarted"
    )
    assert results.completed == 916
    assert results.is_complete is True
