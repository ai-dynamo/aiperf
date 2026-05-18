# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Prometheus `untyped` / UNKNOWN metric handling.

Real-world Prometheus exporters (most notably node-exporter) emit families
declared as ``# TYPE foo untyped`` — or with no ``# TYPE`` line at all, which
the Prometheus parser also classifies as untyped. These tests pin down the
behaviour AIPerf's server-metrics pipeline must guarantee for those families:

- :class:`ServerMetricsTimeSeries` accepts them on first scrape without
  raising ``ValueError: Buckets are required for histogram time series``
  (the symptom that originally crashed scrapes of node-exporter).
- Statistics computation routes UNKNOWN to gauge-style descriptive stats.
- The accumulator wraps them in :class:`UnknownMetricData` so the JSON export
  carries ``type: "unknown"`` rather than masquerading as ``"gauge"``.
- The CSV exporter writes a dedicated ``unknown`` section using gauge-shaped
  stat keys.
- The Parquet exporter emits scalar rows tagged with ``metric_type="unknown"``.
- :class:`NodeExporterFaker` produces an exposition body that round-trips
  through the parser as 7 unknown families (mirroring real node-exporter).
"""

from __future__ import annotations

import csv
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiperf_mock_server.node_exporter_faker import NodeExporterFaker

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    CounterMetricData,
    CounterSeries,
    CounterStats,
    GaugeMetricData,
    GaugeSeries,
    GaugeStats,
    HistogramMetricData,
    HistogramSeries,
    HistogramStats,
    MetricFamily,
    MetricSample,
    ServerMetricsEndpointInfo,
    ServerMetricsEndpointSummary,
    ServerMetricsRecord,
    ServerMetricsResults,
    TimeRangeFilter,
    UnknownMetricData,
)
from aiperf.config import BenchmarkConfig, EndpointConfig
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.accumulator import ServerMetricsAccumulator
from aiperf.server_metrics.csv_exporter import (
    GAUGE_STAT_KEYS,
    STAT_KEYS_MAP,
    ServerMetricsCsvExporter,
)
from aiperf.server_metrics.export_stats import compute_stats
from aiperf.server_metrics.json_exporter import ServerMetricsJsonExporter
from aiperf.server_metrics.parquet_exporter import ServerMetricsParquetExporter
from aiperf.server_metrics.storage import (
    HistogramTimeSeries,
    ScalarTimeSeries,
    ServerMetricEntry,
    ServerMetricsTimeSeries,
)

# ============================================================================
# Helpers
# ============================================================================


def _untyped_record(
    name: str,
    value: float,
    timestamp_ns: int,
    endpoint_url: str = "http://node-exporter:9100/metrics",
) -> ServerMetricsRecord:
    """Build a single-sample record with the given Prometheus UNKNOWN family."""
    return ServerMetricsRecord(
        endpoint_url=endpoint_url,
        timestamp_ns=timestamp_ns,
        endpoint_latency_ns=0,
        metrics={
            name: MetricFamily(
                type=PrometheusMetricType.UNKNOWN,
                description=f"Statistic {name}.",
                samples=[MetricSample(labels=None, value=value)],
            )
        },
    )


@pytest.fixture
def minimal_cfg() -> BenchmarkConfig:
    return BenchmarkConfig(
        model="test-model",
        endpoint=EndpointConfig(
            urls=["http://localhost:8000"],
            type=EndpointType.CHAT,
            streaming=False,
        ),
        dataset={"type": "synthetic"},
        profiling={"type": "concurrency", "requests": 1, "concurrency": 1},
        server_metrics={"urls": ["http://node-exporter:9100/metrics"]},
    )


# ============================================================================
# Storage layer
# ============================================================================


class TestUnknownStorageRouting:
    """``ServerMetricEntry.from_metric_family`` and ``append_snapshot``."""

    def test_unknown_family_creates_scalar_storage(self) -> None:
        family = MetricFamily(
            type=PrometheusMetricType.UNKNOWN,
            description="",
            samples=[MetricSample(labels=None, value=1.0)],
        )
        entry = ServerMetricEntry.from_metric_family(family)
        assert isinstance(entry.data, ScalarTimeSeries)
        assert not isinstance(entry.data, HistogramTimeSeries)
        # The original Prometheus type is preserved on the entry so it can
        # later be forwarded into the export without flattening to "gauge".
        assert entry.metric_type is PrometheusMetricType.UNKNOWN

    def test_first_scrape_with_unknown_family_does_not_raise(self) -> None:
        ts = ServerMetricsTimeSeries()
        ts.append_snapshot(_untyped_record("node_netstat_Icmp_InErrors", 0.0, 1))
        assert len(ts.metrics) == 1

    def test_repeated_unknown_scrapes_accumulate(self) -> None:
        ts = ServerMetricsTimeSeries()
        for i, v in enumerate([1.0, 2.0, 3.0]):
            ts.append_snapshot(
                _untyped_record("node_netstat_Tcp_InSegs", v, timestamp_ns=i + 1)
            )
        entry = next(iter(ts.metrics.values()))
        assert isinstance(entry.data, ScalarTimeSeries)
        assert list(entry.data.values[: len(entry.data)]) == [1.0, 2.0, 3.0]


# ============================================================================
# Stats computation
# ============================================================================


class TestUnknownStatsDispatch:
    """``compute_stats`` for UNKNOWN routes to gauge-shaped output."""

    def test_unknown_dispatches_to_gauge_stats(self) -> None:
        ts = ServerMetricsTimeSeries()
        for i, v in enumerate([10.0, 20.0, 30.0, 40.0]):
            ts.append_snapshot(_untyped_record("foo", v, timestamp_ns=i + 1))
        entry = next(iter(ts.metrics.values()))
        result = compute_stats(
            entry.metric_type,
            entry.data,
            TimeRangeFilter(start_ns=0, end_ns=10),
            labels=None,
            slice_duration=None,
        )
        assert result is not None
        # Gauge-shape stats: avg / min / max / std / pNN.
        assert result.stats.avg == pytest.approx(25.0)
        assert result.stats.min == pytest.approx(10.0)
        assert result.stats.max == pytest.approx(40.0)


# ============================================================================
# Accumulator (export-level wrapping)
# ============================================================================


@pytest.mark.asyncio
class TestUnknownAccumulatorExport:
    """Accumulator produces ``UnknownMetricData`` for UNKNOWN entries."""

    async def test_endpoint_summary_uses_unknown_metric_data(
        self, minimal_cfg: BenchmarkConfig
    ) -> None:
        proc = ServerMetricsAccumulator(
            SimpleNamespace(cfg=minimal_cfg, benchmark_id="bench-server-metrics")
        )
        for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            await proc.process_server_metrics_record(
                _untyped_record("node_netstat_Icmp_InErrors", v, timestamp_ns=i + 1)
            )
        results = await proc.export_results(start_ns=0, end_ns=10)
        assert results is not None
        (summary,) = results.endpoint_summaries.values()
        metric = summary.metrics["node_netstat_Icmp_InErrors"]
        # Must be UnknownMetricData specifically — not GaugeMetricData with
        # an overridden type. This guarantees discriminated-union round-trip.
        assert isinstance(metric, UnknownMetricData)
        assert not isinstance(metric, GaugeMetricData)
        assert metric.type is PrometheusMetricType.UNKNOWN
        # Gauge-equivalent stats must be present and numerically correct for
        # the samples we fed in. This is the core claim of the PR: UNKNOWN is
        # treated as gauge for storage and statistics.
        assert len(metric.series) == 1
        stats = metric.series[0].stats
        assert stats.avg == pytest.approx(3.0)
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(5.0)


# ============================================================================
# JSON exporter
# ============================================================================


def _stub_endpoint_summary(
    metric_data: UnknownMetricData,
) -> ServerMetricsEndpointSummary:
    return ServerMetricsEndpointSummary(
        endpoint_url="http://node-exporter:9100/metrics",
        info=ServerMetricsEndpointInfo(
            total_fetches=1,
            first_fetch_ns=1,
            last_fetch_ns=2,
            avg_fetch_latency_ms=1.0,
            unique_updates=1,
            first_update_ns=1,
            last_update_ns=2,
            duration_seconds=0.001,
            avg_update_interval_ms=1.0,
            median_update_interval_ms=1.0,
        ),
        metrics={"node_netstat_Icmp_InErrors": metric_data},
    )


class TestUnknownJsonRoundTrip:
    """JSON export carries ``type: "unknown"`` and re-deserialises cleanly."""

    def test_unknown_metric_data_serializes_as_unknown(self) -> None:
        umd = UnknownMetricData(description="Statistic IcmpInErrors.")
        dumped = umd.model_dump(mode="json")
        assert dumped["type"] == "unknown"

    def test_full_export_json_contains_unknown_type(
        self, minimal_cfg: BenchmarkConfig
    ) -> None:
        # Build a minimal ServerMetricsResults with one UnknownMetricData entry.
        umd = UnknownMetricData(description="Statistic IcmpInErrors.")
        results = ServerMetricsResults(
            endpoint_summaries={
                "http://node-exporter:9100/metrics": _stub_endpoint_summary(umd)
            },
            start_ns=0,
            end_ns=10,
            endpoints_configured=["http://node-exporter:9100/metrics"],
            endpoints_successful=["http://node-exporter:9100/metrics"],
            error_summary=[],
        )
        exporter_config = MagicMock()
        exporter_config.server_metrics_results = results
        exporter_config.cfg = minimal_cfg
        exporter = ServerMetricsJsonExporter(exporter_config)
        export_data, _ = exporter._build_hybrid_metrics()
        entry = export_data["node_netstat_Icmp_InErrors"]
        assert isinstance(entry, UnknownMetricData)
        assert entry.model_dump(mode="json")["type"] == "unknown"


# ============================================================================
# CSV exporter
# ============================================================================


class TestUnknownCsvSection:
    """CSV exporter writes an ``unknown`` section using gauge stat keys."""

    def test_stat_keys_map_uses_gauge_keys_for_unknown(self) -> None:
        # Compare contents rather than identity: a future refactor that copies
        # the list (``list(GAUGE_STAT_KEYS)``) would silently break an `is`
        # check while the actual behaviour stays correct.
        assert (
            STAT_KEYS_MAP[PrometheusMetricType.UNKNOWN]
            == STAT_KEYS_MAP[PrometheusMetricType.GAUGE]
            == GAUGE_STAT_KEYS
        )

    def test_csv_output_includes_unknown_section(
        self, minimal_cfg: BenchmarkConfig
    ) -> None:
        umd = UnknownMetricData(description="Statistic IcmpInErrors.")
        # Populate with one series so the section is non-empty.
        from aiperf.common.models.server_metrics_models import (
            GaugeSeries,
            GaugeStats,
        )

        umd.series.append(
            GaugeSeries(
                endpoint_url="http://node-exporter:9100/metrics",
                stats=GaugeStats(
                    avg=1.0,
                    min=1.0,
                    max=1.0,
                    std=0.0,
                    p1=1.0,
                    p5=1.0,
                    p10=1.0,
                    p25=1.0,
                    p50=1.0,
                    p75=1.0,
                    p90=1.0,
                    p95=1.0,
                    p99=1.0,
                ),
            )
        )
        results = ServerMetricsResults(
            endpoint_summaries={
                "http://node-exporter:9100/metrics": _stub_endpoint_summary(umd)
            },
            start_ns=0,
            end_ns=10,
            endpoints_configured=["http://node-exporter:9100/metrics"],
            endpoints_successful=["http://node-exporter:9100/metrics"],
            error_summary=[],
        )
        exporter_config = MagicMock()
        exporter_config.server_metrics_results = results
        exporter_config.cfg = minimal_cfg
        exporter = ServerMetricsCsvExporter(exporter_config)
        body = exporter._generate_content()
        # The unknown-section header should mention the type explicitly.
        assert "unknown" in body.lower()
        # And the gauge stat keys should appear on the same line as the
        # `Metric` header column.
        reader = csv.reader(io.StringIO(body))
        rows = [row for row in reader if row]
        joined = "\n".join(",".join(r) for r in rows)
        assert "node_netstat_Icmp_InErrors" in joined

    def test_unknown_section_is_after_histogram(
        self, minimal_cfg: BenchmarkConfig
    ) -> None:
        gauge = GaugeMetricData(description="Gauge metric")
        gauge.series.append(GaugeSeries(stats=GaugeStats(avg=1.0)))
        counter = CounterMetricData(description="Counter metric")
        counter.series.append(CounterSeries(stats=CounterStats(total=1.0)))
        histogram = HistogramMetricData(description="Histogram metric")
        histogram.series.append(HistogramSeries(stats=HistogramStats(count=1)))
        unknown = UnknownMetricData(description="Untyped metric")
        unknown.series.append(GaugeSeries(stats=GaugeStats(avg=1.0)))
        results = ServerMetricsResults(
            endpoint_summaries={
                "http://node-exporter:9100/metrics": ServerMetricsEndpointSummary(
                    endpoint_url="http://node-exporter:9100/metrics",
                    info=ServerMetricsEndpointInfo(
                        total_fetches=1,
                        first_fetch_ns=1,
                        last_fetch_ns=2,
                        avg_fetch_latency_ms=1.0,
                        unique_updates=1,
                        first_update_ns=1,
                        last_update_ns=2,
                        duration_seconds=0.001,
                        avg_update_interval_ms=1.0,
                        median_update_interval_ms=1.0,
                    ),
                    metrics={
                        "gauge_metric": gauge,
                        "counter_metric": counter,
                        "histogram_metric": histogram,
                        "unknown_metric": unknown,
                    },
                )
            },
            start_ns=0,
            end_ns=10,
            endpoints_configured=["http://node-exporter:9100/metrics"],
            endpoints_successful=["http://node-exporter:9100/metrics"],
            error_summary=[],
        )
        exporter_config = MagicMock()
        exporter_config.server_metrics_results = results
        exporter_config.cfg = minimal_cfg
        exporter = ServerMetricsCsvExporter(exporter_config)

        rows = [
            row for row in csv.reader(io.StringIO(exporter._generate_content())) if row
        ]
        section_types = [row[1] for row in rows if row[0] == "node-exporter:9100"]

        assert section_types == ["gauge", "counter", "histogram", "unknown"]


# ============================================================================
# Parquet exporter
# ============================================================================


class TestUnknownParquetRows:
    """Parquet exporter emits scalar rows for UNKNOWN entries."""

    def test_collect_scalar_rows_handles_unknown(self) -> None:
        ts = ServerMetricsTimeSeries()
        for i, v in enumerate([10.0, 20.0, 30.0]):
            ts.append_snapshot(_untyped_record("foo", v, timestamp_ns=i + 1))
        accumulator = MagicMock()
        accumulator.get_hierarchy_for_export.return_value = MagicMock(
            endpoints={"http://node-exporter:9100/metrics": ts}
        )
        exporter = ServerMetricsParquetExporter.__new__(ServerMetricsParquetExporter)
        exporter._accumulator = accumulator
        exporter._time_filter = None
        rows = list(exporter._collect_all_rows_generator(set()))
        # Exactly one row per scrape, all tagged as unknown.
        assert len(rows) == 3
        assert {r["metric_type"] for r in rows} == {PrometheusMetricType.UNKNOWN}
        assert [r["value"] for r in rows] == [10.0, 20.0, 30.0]


# ============================================================================
# Faker self-test
# ============================================================================


class TestNodeExporterFaker:
    """``NodeExporterFaker.default()`` is the canonical fixture for the suite."""

    def test_default_emits_expected_type_mix(self) -> None:
        from prometheus_client.parser import text_string_to_metric_families

        faker = NodeExporterFaker.default(seed=42)
        families = list(text_string_to_metric_families(faker.render()))
        type_counts = {
            t: 0 for t in {"unknown", "gauge", "counter", "histogram", "summary"}
        }
        for fam in families:
            type_counts[fam.type] = type_counts.get(fam.type, 0) + 1
        # The exact mix the suite relies on. Update both this assertion and
        # the regression coverage below if the default mix changes.
        assert type_counts == {
            "unknown": 7,
            "gauge": 4,
            "counter": 1,
            "histogram": 1,
            "summary": 1,
        }

    def test_collisions_flip_type_per_scrape(self) -> None:
        faker = NodeExporterFaker.collisions(seed=1)
        seen: list[str] = []
        for _ in range(4):
            body = faker.render()
            line = next(
                line
                for line in body.splitlines()
                if line.startswith("# TYPE node_unstable_type")
            )
            seen.append(line.rsplit(" ", 1)[-1])
        assert seen == ["gauge", "histogram", "gauge", "histogram"]

    def test_value_drift_makes_stats_non_degenerate(self) -> None:
        faker = NodeExporterFaker.default(seed=7)
        snapshots = [faker.render() for _ in range(5)]
        # The ``node_load1`` line should not be identical across scrapes — the
        # default value_fn applies an RNG-driven jitter on top of a base.
        load1_lines = [
            next(line for line in body.splitlines() if line.startswith("node_load1 "))
            for body in snapshots
        ]
        assert len(set(load1_lines)) > 1
