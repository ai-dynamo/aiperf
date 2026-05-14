# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: decode real OTLP payloads and assert their contents.

`test_otel_live_export.py` proves POSTs arrive; it does not decode them. This
file spins up an OTLP/HTTP sink, parses the protobuf body as an
``ExportMetricsServiceRequest``, and asserts the shape a downstream OTel
Collector (and therefore Prometheus / Grafana) would actually see:

    * Resource attributes carry ``service.name=aiperf``, ``aiperf.model.name``,
      ``aiperf.benchmark.id`` and any ``--otel-resource-attributes`` extras.
    * Metric names emitted by AIPerf use the ``aiperf.`` prefix for its own
      metrics and stay free of empty strings / None units.
    * At least one histogram lands with numeric data points (not a
      zero-point shell that passes mid-run flush but has no bucketed data).

These assertions would catch the GenAI semconv / resource-attribute
regressions that the in-process fake sink in ``test_otel_live_export.py``
cannot detect.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


class _DecodingOTLPSinkHandler(BaseHTTPRequestHandler):
    """OTLP HTTP sink that parses POST bodies as protobuf ExportMetricsServiceRequest."""

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/metrics":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            request = ExportMetricsServiceRequest()
            # Tolerate partial/empty payloads (PeriodicExportingMetricReader
            # flushes even empty batches at shutdown on some versions). Decode
            # errors are surfaced to the test via the exports list.
            try:
                request.ParseFromString(body)
                self.server.exports.append(request)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - surface parse errors to the test
                self.server.decode_errors.append(repr(exc))  # type: ignore[attr-defined]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, fmt: str, *args: object) -> None:
        pass


class _DecodingOTLPSinkServer(HTTPServer):
    """HTTPServer that retains decoded OTLP export requests."""

    def __init__(self, port: int) -> None:
        self.exports: list[ExportMetricsServiceRequest] = []
        self.decode_errors: list[str] = []
        super().__init__(("127.0.0.1", port), _DecodingOTLPSinkHandler)


@pytest.fixture
def decoding_otlp_sink() -> tuple[_DecodingOTLPSinkServer, int]:
    server = _DecodingOTLPSinkServer(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, port
    server.shutdown()
    thread.join(timeout=5)


def _all_resource_attributes(
    exports: list[ExportMetricsServiceRequest],
) -> dict[str, str]:
    """Flatten resource attributes across every export batch.

    Later batches win on key collisions — the exporter repeats the same
    resource on every export, so this is a stable merge.
    """
    attrs: dict[str, str] = {}
    for request in exports:
        for resource_metrics in request.resource_metrics:
            for kv in resource_metrics.resource.attributes:
                # OTLP AnyValue: string_value / int_value / double_value / bool_value
                field = kv.value.WhichOneof("value")
                if field is not None:
                    attrs[kv.key] = str(getattr(kv.value, field))
    return attrs


def _all_metric_names(exports: list[ExportMetricsServiceRequest]) -> set[str]:
    names: set[str] = set()
    for request in exports:
        for resource_metrics in request.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    names.add(metric.name)
    return names


def _histograms_with_data_points(
    exports: list[ExportMetricsServiceRequest],
) -> dict[str, int]:
    """Count data points per histogram metric across every export batch."""
    counts: dict[str, int] = {}
    for request in exports:
        for resource_metrics in request.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.HasField("histogram"):
                        counts[metric.name] = counts.get(metric.name, 0) + len(
                            metric.histogram.data_points
                        )
    return counts


@pytest.mark.component_integration
@pytest.mark.asyncio
class TestOTelPayloadContent:
    """Decode real OTLP payloads and assert the shape a Collector would see."""

    async def test_resource_attributes_and_aiperf_metrics_present(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        decoding_otlp_sink: tuple[_DecodingOTLPSinkServer, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-to-end OTLP payload validation against a decoding sink.

        A small flush interval plus a modest request count guarantees at
        least one mid-run batch; the shutdown flush adds a final batch.
        All batches share the same Resource, so the set of keys is
        deterministic.
        """
        server, port = decoding_otlp_sink
        otel_url = f"http://127.0.0.1:{port}"

        # Sub-second flush so we collect multiple batches in a short run.
        monkeypatch.setenv("AIPERF_OTEL_FLUSH_INTERVAL_SECONDS", "0.5")

        cli_cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --concurrency 2 \
                --request-count 20 \
                --streaming \
                --otel-url {otel_url} \
                --stream default \
                --otel-resource-attributes team=inference,env=ci
        """

        result = await cli.run(cli_cmd, timeout=120.0)
        assert result.exit_code == 0, (
            f"aiperf profile failed with exit code {result.exit_code}"
        )

        # Sanity — decoding sink is the one under test here; a protobuf
        # decode failure is a schema regression we want to surface loudly.
        assert not server.decode_errors, (
            f"OTLP sink failed to decode protobuf payloads: {server.decode_errors}"
        )
        assert server.exports, "OTLP sink received zero export batches"

        # --- Resource attributes ---------------------------------------------
        resource_attrs = _all_resource_attributes(server.exports)
        assert resource_attrs.get("service.name") == "aiperf", (
            f"service.name missing or wrong: {resource_attrs!r}"
        )
        # aiperf.model.name is set from endpoint.model_names[0]; matches the
        # value AIPerf auto-resolves from IntegrationTestDefaults.model.
        assert resource_attrs.get("aiperf.model.name"), (
            f"aiperf.model.name missing from resource: {resource_attrs!r}"
        )
        # --otel-resource-attributes pass-through.
        assert resource_attrs.get("team") == "inference", (
            f"team=inference not propagated into OTLP Resource: {resource_attrs!r}"
        )
        assert resource_attrs.get("env") == "ci", (
            f"env=ci not propagated into OTLP Resource: {resource_attrs!r}"
        )
        # benchmark_id is auto-generated; assert the key exists and is non-empty.
        assert resource_attrs.get("aiperf.benchmark.id"), (
            f"aiperf.benchmark.id missing or empty: {resource_attrs!r}"
        )

        # --- Metric names ----------------------------------------------------
        metric_names = _all_metric_names(server.exports)
        # AIPerf's own streaming metrics must carry the aiperf. prefix —
        # anything without a prefix is suspicious.
        aiperf_prefixed = {n for n in metric_names if n.startswith("aiperf.")}
        assert aiperf_prefixed, (
            f"No metrics with 'aiperf.' prefix found. All names: {metric_names!r}"
        )
        # Empty / whitespace metric names would break Collector ingestion.
        assert all(n.strip() for n in metric_names), (
            f"Found empty metric name(s): {metric_names!r}"
        )

        # --- Histogram data points -------------------------------------------
        # AIPerf exports request_latency_ns (and token-related) metrics as
        # histograms. At least one histogram must carry at least one data
        # point — an empty histogram would mean the instrument was created
        # but never recorded, which bypasses the whole point of live export.
        histogram_points = _histograms_with_data_points(server.exports)
        assert histogram_points, (
            f"No histogram metrics with data points. Metric names: {metric_names!r}"
        )
        assert any(count > 0 for count in histogram_points.values()), (
            f"All histograms have zero data points: {histogram_points!r}"
        )
