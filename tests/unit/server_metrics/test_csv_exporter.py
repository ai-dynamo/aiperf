# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ServerMetricsCsvExporter."""

import csv
import io
from typing import TypeAlias

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import GenericMetricUnit
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import (
    CounterMetricData,
    CounterSeries,
    CounterStats,
    GaugeMetricData,
    GaugeSeries,
    GaugeStats,
    HistogramMetricData,
    HistogramSeries,
    HistogramStats,
    ProfileResults,
    ServerMetricsEndpointInfo,
    ServerMetricsEndpointSummary,
    ServerMetricsResults,
)
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.csv_exporter import CsvMetricInfo, ServerMetricsCsvExporter
from tests.unit.conftest import create_exporter_config

MetricDataType: TypeAlias = GaugeMetricData | CounterMetricData | HistogramMetricData

# =============================================================================
# Helper Functions
# =============================================================================


def _parse_csv_content(content: str) -> list[list[str]]:
    """Parse CSV content into list of rows."""
    return list(csv.reader(io.StringIO(content)))


def _find_header_row(
    rows: list[list[str]], required_cols: list[str]
) -> list[str] | None:
    """Find a header row containing all required columns."""
    for row in rows:
        if row and row[0] == "Endpoint" and all(col in row for col in required_cols):
            return row
    return None


def _create_endpoint_summary(
    metrics: dict[str, MetricDataType],
    endpoint_url: str = "http://localhost:8081/metrics",
    duration_seconds: float = 100.0,
    scrape_count: int = 20,
) -> ServerMetricsEndpointSummary:
    """Create a ServerMetricsEndpointSummary with sensible defaults."""
    return ServerMetricsEndpointSummary(
        endpoint_url=endpoint_url,
        info=ServerMetricsEndpointInfo(
            total_fetches=scrape_count * 2,
            first_fetch_ns=1_000_000_000_000,
            last_fetch_ns=1_100_000_000_000,
            avg_fetch_latency_ms=8.0,
            unique_updates=scrape_count,
            first_update_ns=1_000_000_000_000,
            last_update_ns=1_100_000_000_000,
            duration_seconds=duration_seconds,
            avg_update_interval_ms=5263.2,
        ),
        metrics=metrics,
    )


def _create_server_metrics_results(
    endpoint_summaries: dict[str, ServerMetricsEndpointSummary],
) -> ServerMetricsResults:
    """Create ServerMetricsResults from endpoint summaries with default timestamps."""
    endpoints = [s.endpoint_url for s in endpoint_summaries.values()]
    return ServerMetricsResults(
        benchmark_id="test-benchmark-id",
        endpoint_summaries=endpoint_summaries,
        start_ns=1_000_000_000_000,
        end_ns=1_100_000_000_000,
        endpoints_configured=endpoints,
        endpoints_successful=endpoints,
        error_summary=[],
    )


def _generate_csv_content(
    mock_user_config,
    mock_profile_results,
    server_metrics_results: ServerMetricsResults,
) -> str:
    """Create exporter and generate CSV content."""
    config = create_exporter_config(
        profile_results=mock_profile_results,
        user_config=mock_user_config,
        server_metrics_results=server_metrics_results,
    )
    return ServerMetricsCsvExporter(config)._generate_content()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_user_config(tmp_path):
    """Create a UserConfig with a temp output directory."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="/v1/chat/completions",
        ),
        output={"artifact_dir": str(tmp_path)},
    )


@pytest.fixture
def mock_profile_results():
    """Create a minimal ProfileResults for exporter config."""
    return ProfileResults(
        records=[],
        completed=100,
        start_ns=1_000_000_000_000,
        end_ns=1_300_000_000_000,
    )


@pytest.fixture
def server_metrics_results_with_all_types():
    """Create ServerMetricsResults with gauge, counter, and histogram metrics."""
    endpoint1_summary = _create_endpoint_summary(
        endpoint_url="http://localhost:8081/metrics",
        duration_seconds=300.0,
        scrape_count=60,
        metrics={
            "vllm:kv_cache_usage_perc": GaugeMetricData(
                description="KV cache usage percentage",
                series=[
                    GaugeSeries(
                        labels=None,
                        stats=GaugeStats(
                            min=0.4,
                            avg=0.55,
                            p50=0.54,
                            p90=0.68,
                            p95=0.72,
                            p99=0.78,
                            max=0.8,
                            std=0.1,
                        ),
                    )
                ],
            ),
            "vllm:request_success_total": CounterMetricData(
                description="Total successful requests",
                series=[
                    CounterSeries(
                        labels=None,
                        stats=CounterStats(
                            total=1000.0,
                            rate=3.33,
                            rate_avg=3.2,
                            rate_min=2.5,
                            rate_max=4.0,
                            rate_std=0.5,
                        ),
                    )
                ],
            ),
            "vllm:time_to_first_token_seconds": HistogramMetricData(
                description="Time to first token histogram",
                series=[
                    HistogramSeries(
                        labels=None,
                        stats=HistogramStats(
                            count=1000,
                            sum=125.5,
                            avg=0.1255,
                            count_rate=3.33,
                        ),
                        buckets={"0.01": 50, "0.1": 450, "1.0": 980, "+Inf": 1000},
                    )
                ],
            ),
        },
    )

    endpoint2_summary = _create_endpoint_summary(
        endpoint_url="http://localhost:8082/metrics",
        duration_seconds=300.0,
        scrape_count=58,
        metrics={
            "vllm:kv_cache_usage_perc": GaugeMetricData(
                description="KV cache usage percentage",
                series=[
                    GaugeSeries(
                        labels=None,
                        stats=GaugeStats(
                            min=0.5,
                            avg=0.62,
                            p50=0.61,
                            p90=0.75,
                            p95=0.78,
                            p99=0.82,
                            max=0.85,
                            std=0.08,
                        ),
                    )
                ],
            ),
            "vllm:request_success_total": CounterMetricData(
                description="Total successful requests",
                series=[
                    CounterSeries(
                        labels=None,
                        stats=CounterStats(
                            total=800.0,
                            rate=2.67,
                            rate_avg=2.5,
                            rate_min=2.0,
                            rate_max=3.5,
                            rate_std=0.4,
                        ),
                    )
                ],
            ),
        },
    )

    return _create_server_metrics_results(
        {
            "localhost:8081": endpoint1_summary,
            "localhost:8082": endpoint2_summary,
        }
    )


@pytest.fixture
def server_metrics_results_with_info_metrics():
    """Create ServerMetricsResults with info metrics to test transposed section."""
    endpoint_summary = _create_endpoint_summary(
        duration_seconds=300.0,
        scrape_count=60,
        metrics={
            "vllm:kv_cache_usage_perc": GaugeMetricData(
                description="KV cache usage percentage",
                series=[
                    GaugeSeries(
                        labels={"engine": "0"},
                        stats=GaugeStats(
                            min=0.4,
                            avg=0.55,
                            p50=0.54,
                            p90=0.68,
                            p95=0.72,
                            p99=0.78,
                            max=0.8,
                            std=0.1,
                        ),
                    )
                ],
            ),
            "vllm:cache_config_info": GaugeMetricData(
                description="Information of the LLMEngine CacheConfig",
                series=[
                    GaugeSeries(
                        labels={
                            "block_size": "16",
                            "cache_dtype": "auto",
                            "enable_prefix_caching": "True",
                        },
                        stats=GaugeStats(avg=1.0),
                    )
                ],
            ),
        },
    )
    return _create_server_metrics_results({"localhost:8081": endpoint_summary})


@pytest.fixture
def server_metrics_results_with_labeled_metrics():
    """Create ServerMetricsResults with labeled metrics to test label handling."""
    endpoint_summary = _create_endpoint_summary(
        metrics={
            "http_requests_total": CounterMetricData(
                description="Total HTTP requests",
                series=[
                    CounterSeries(
                        labels={"method": "GET", "status": "200"},
                        stats=CounterStats(
                            total=500.0,
                            rate=5.0,
                            rate_avg=4.8,
                            rate_min=3.0,
                            rate_max=6.0,
                            rate_std=0.8,
                        ),
                    ),
                    CounterSeries(
                        labels={"method": "POST", "status": "200"},
                        stats=CounterStats(
                            total=300.0,
                            rate=3.0,
                            rate_avg=2.9,
                            rate_min=2.0,
                            rate_max=4.0,
                            rate_std=0.5,
                        ),
                    ),
                ],
            ),
        },
    )
    return _create_server_metrics_results({"localhost:8081": endpoint_summary})


class TestServerMetricsCsvExporterInitialization:
    """Test exporter initialization."""

    def test_initialization_with_valid_config(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that exporter initializes correctly with valid config."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        assert ServerMetricsCsvExporter(config) is not None

    def test_initialization_disabled_without_results(
        self, mock_user_config, mock_profile_results
    ):
        """Test that exporter raises DataExporterDisabled when no results."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=None,
        )
        with pytest.raises(DataExporterDisabled):
            ServerMetricsCsvExporter(config)


class TestServerMetricsCsvExporterGetExportInfo:
    """Test get_export_info method."""

    def test_get_export_info_returns_correct_type(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that export info contains correct type and path."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        info = ServerMetricsCsvExporter(config).get_export_info()
        assert info.export_type == "Server Metrics CSV Export"
        assert "server_metrics" in str(info.file_path)
        assert str(info.file_path).endswith(".csv")


class TestServerMetricsCsvExporterGenerateContent:
    """Test CSV content generation."""

    def test_generate_content_creates_valid_csv(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that generated content is valid CSV."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_all_types,
        )
        assert len(_parse_csv_content(content)) > 0

    def test_generate_content_has_sections_by_metric_type(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that CSV has separate sections for each metric type."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_all_types,
        )
        # Check for column headers for each metric type
        assert "Endpoint,Type,Metric,Unit,avg,min,max" in content  # gauge
        assert "p99,Description" in content  # gauge ends with Description
        assert "Endpoint,Type,Metric,Unit,total,rate" in content  # counter
        assert "Endpoint,Type,Metric,Unit,count,count_rate" in content  # histogram
        # Check that metric type values appear in the data
        assert ",gauge," in content
        assert ",counter," in content
        assert ",histogram," in content

    def test_generate_content_gauge_section_has_correct_columns(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that gauge section has appropriate stat columns."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_all_types,
        )
        rows = _parse_csv_content(content)
        gauge_header = _find_header_row(rows, ["p50", "p99"])

        assert gauge_header is not None
        for col in [
            "Endpoint",
            "Metric",
            "Description",
            "avg",
            "min",
            "max",
            "p50",
            "p90",
        ]:
            assert col in gauge_header

    def test_generate_content_counter_section_has_correct_columns(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that counter section has appropriate stat columns."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_all_types,
        )
        rows = _parse_csv_content(content)
        counter_header = _find_header_row(rows, ["total", "rate"])

        assert counter_header is not None
        for col in ["total", "rate", "rate_avg"]:
            assert col in counter_header

    def test_generate_content_histogram_section_has_buckets_column(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that histogram section has a buckets column."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_all_types,
        )
        rows = _parse_csv_content(content)
        hist_header = _find_header_row(rows, ["count", "count_rate", "buckets"])

        assert hist_header is not None
        assert "buckets" in hist_header

    def test_generate_content_has_normalized_endpoints(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that endpoints are normalized (without http:// and /metrics)."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_all_types,
        )
        assert "localhost:8081" in content
        assert "localhost:8082" in content
        assert "http://localhost:8081/metrics" not in content

    def test_generate_content_handles_labeled_metrics(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_labeled_metrics,
    ):
        """Test that labeled metrics have individual label columns."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_labeled_metrics,
        )
        # Labels should be individual columns
        for expected in [",method,", ",status,", ",GET,", ",POST,", ",200,"]:
            assert expected in content

    def test_generate_content_merges_metrics_from_all_endpoints(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that metrics from multiple endpoints appear in the same section."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_all_types,
        )
        rows = _parse_csv_content(content)
        kv_cache_rows = [r for r in rows if r and "vllm:kv_cache_usage_perc" in r]
        assert len(kv_cache_rows) == 2  # one per endpoint

    def test_generate_content_histogram_bucket_values_in_column(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that histogram bucket values are in key=value format."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_all_types,
        )
        for expected in ["0.01=50", "0.1=450", "1.0=980", "+Inf=1000"]:
            assert expected in content

    def test_generate_content_histograms_with_different_buckets_in_same_section(
        self, mock_user_config, mock_profile_results
    ):
        """Test that histograms with different bucket boundaries are in the same section."""
        results = _create_server_metrics_results(
            {
                "localhost:8081": _create_endpoint_summary(
                    metrics={
                        "request_duration_seconds": HistogramMetricData(
                            description="Request duration",
                            series=[
                                HistogramSeries(
                                    labels=None,
                                    stats=HistogramStats(
                                        count=100,
                                        sum=50.0,
                                        avg=0.5,
                                        count_rate=1.0,
                                    ),
                                    buckets={
                                        "0.1": 10,
                                        "0.5": 50,
                                        "1.0": 90,
                                        "+Inf": 100,
                                    },
                                )
                            ],
                        ),
                        "queue_time_seconds": HistogramMetricData(
                            description="Queue time",
                            series=[
                                HistogramSeries(
                                    labels=None,
                                    stats=HistogramStats(
                                        count=200,
                                        sum=10.0,
                                        avg=0.05,
                                        count_rate=2.0,
                                    ),
                                    buckets={
                                        "0.01": 50,
                                        "0.05": 150,
                                        "0.1": 190,
                                        "+Inf": 200,
                                    },
                                )
                            ],
                        ),
                    },
                )
            }
        )

        content = _generate_csv_content(mock_user_config, mock_profile_results, results)
        rows = _parse_csv_content(content)

        # Count histogram headers - should be exactly one
        hist_headers = [
            r
            for r in rows
            if r
            and r[0] == "Endpoint"
            and all(c in r for c in ["count", "count_rate", "buckets"])
        ]
        assert len(hist_headers) == 1

        # Both metrics and their buckets should be present
        assert "request_duration_seconds" in content
        assert "queue_time_seconds" in content
        assert "0.1=10" in content  # request_duration
        assert "0.01=50" in content  # queue_time

    def test_generate_content_has_unit_column(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that Unit column exists with correct values derived from metric names."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_all_types,
        )
        rows = _parse_csv_content(content)
        gauge_header = _find_header_row(rows, ["avg"])

        assert gauge_header is not None
        assert "Unit" in gauge_header
        # Check unit values in data
        assert ",percent," in content  # vllm:kv_cache_usage_perc
        assert ",count," in content  # vllm:request_success_total
        assert ",seconds," in content  # vllm:time_to_first_token_seconds

    def test_generate_content_info_metrics_in_transposed_section(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_info_metrics,
    ):
        """Test that info metrics appear in transposed key-value format."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_info_metrics,
        )
        assert "Endpoint,Metric,Key,Value,Description" in content
        # Info metric keys/values
        for expected in [",block_size,", ",cache_dtype,", ",enable_prefix_caching,"]:
            assert expected in content
        for expected in [",16,", ",auto,", ",True,"]:
            assert expected in content

    def test_generate_content_info_metrics_separated_from_gauges(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_info_metrics,
    ):
        """Test that info metrics don't appear in gauge section."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_info_metrics,
        )
        rows = _parse_csv_content(content)

        # Extract gauge section rows
        gauge_section_rows = []
        in_gauge_section = False
        for row in rows:
            if row and row[0] == "Endpoint" and "avg" in row:
                in_gauge_section = True
                continue
            if in_gauge_section:
                if not row or row[0] == "":
                    break
                gauge_section_rows.append(row)

        gauge_metrics = [r[2] for r in gauge_section_rows if len(r) > 2]
        assert "vllm:kv_cache_usage_perc" in gauge_metrics
        assert "vllm:cache_config_info" not in gauge_metrics

    def test_generate_content_info_metrics_have_description(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_info_metrics,
    ):
        """Test that info metric rows include description."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_info_metrics,
        )
        assert "Information of the LLMEngine CacheConfig" in content

    def test_generate_content_label_columns_exclude_info_metric_labels(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_info_metrics,
    ):
        """Test that info metric labels don't become columns in gauge section."""
        content = _generate_csv_content(
            mock_user_config,
            mock_profile_results,
            server_metrics_results_with_info_metrics,
        )
        rows = _parse_csv_content(content)
        gauge_header = _find_header_row(rows, ["avg"])

        assert gauge_header is not None
        # Info metric labels should NOT be columns
        for label in ["block_size", "cache_dtype", "enable_prefix_caching"]:
            assert label not in gauge_header
        # Regular gauge labels should be columns
        assert "engine" in gauge_header

    def test_generate_content_labels_grouped_by_cooccurrence(
        self, mock_user_config, mock_profile_results
    ):
        """Test that labels appearing together stay adjacent in columns."""
        results = _create_server_metrics_results(
            {
                "localhost:8081": _create_endpoint_summary(
                    metrics={
                        "aaa_metric": CounterMetricData(
                            description="First metric",
                            series=[
                                CounterSeries(
                                    labels={"engine": "0", "model_name": "Qwen"},
                                    stats=CounterStats(total=100.0, rate=1.0),
                                )
                            ],
                        ),
                        "zzz_metric": CounterMetricData(
                            description="Last metric",
                            series=[
                                CounterSeries(
                                    labels={"method": "GET", "status": "200"},
                                    stats=CounterStats(total=200.0, rate=2.0),
                                )
                            ],
                        ),
                    },
                )
            }
        )

        content = _generate_csv_content(mock_user_config, mock_profile_results, results)
        rows = _parse_csv_content(content)
        counter_header = _find_header_row(rows, ["total"])
        assert counter_header is not None

        engine_idx = counter_header.index("engine")
        model_name_idx = counter_header.index("model_name")
        method_idx = counter_header.index("method")
        status_idx = counter_header.index("status")

        # Labels from same metric should be adjacent
        assert abs(engine_idx - model_name_idx) == 1
        assert abs(method_idx - status_idx) == 1
        # aaa_metric's labels come first
        assert engine_idx < method_idx
        assert model_name_idx < status_idx

    def test_generate_content_overlapping_labels_merged(
        self, mock_user_config, mock_profile_results
    ):
        """Test that overlapping label sets are merged into one group."""
        results = _create_server_metrics_results(
            {
                "localhost:8081": _create_endpoint_summary(
                    metrics={
                        "aaa_metric": CounterMetricData(
                            description="First metric",
                            series=[
                                CounterSeries(
                                    labels={"engine": "0", "model_name": "Qwen"},
                                    stats=CounterStats(total=100.0, rate=1.0),
                                )
                            ],
                        ),
                        "bbb_metric": CounterMetricData(
                            description="Second metric",
                            series=[
                                CounterSeries(
                                    labels={"model_name": "Qwen", "version": "v1"},
                                    stats=CounterStats(total=200.0, rate=2.0),
                                )
                            ],
                        ),
                        "ccc_metric": CounterMetricData(
                            description="Third metric",
                            series=[
                                CounterSeries(
                                    labels={"method": "GET", "status": "200"},
                                    stats=CounterStats(total=300.0, rate=3.0),
                                )
                            ],
                        ),
                    },
                )
            }
        )

        content = _generate_csv_content(mock_user_config, mock_profile_results, results)
        rows = _parse_csv_content(content)
        counter_header = _find_header_row(rows, ["total"])
        assert counter_header is not None

        engine_idx = counter_header.index("engine")
        model_name_idx = counter_header.index("model_name")
        version_idx = counter_header.index("version")
        method_idx = counter_header.index("method")
        status_idx = counter_header.index("status")

        # Merged group (engine, model_name, version) should be adjacent
        merged_indices = sorted([engine_idx, model_name_idx, version_idx])
        assert merged_indices == list(range(merged_indices[0], merged_indices[0] + 3))
        # method and status should be adjacent
        assert abs(method_idx - status_idx) == 1
        # Merged group comes before separate group
        assert max(merged_indices) < min(method_idx, status_idx)

    def test_generate_content_metrics_without_labels(
        self, mock_user_config, mock_profile_results
    ):
        """Test that metrics without labels work correctly (no label columns)."""
        results = _create_server_metrics_results(
            {
                "localhost:8081": _create_endpoint_summary(
                    metrics={
                        "simple_gauge": GaugeMetricData(
                            description="A simple gauge without labels",
                            series=[
                                GaugeSeries(
                                    labels=None,
                                    stats=GaugeStats(avg=42.0, min=10.0, max=100.0),
                                )
                            ],
                        ),
                    },
                )
            }
        )

        content = _generate_csv_content(mock_user_config, mock_profile_results, results)
        rows = _parse_csv_content(content)
        gauge_header = _find_header_row(rows, ["avg"])
        assert gauge_header is not None

        # Description immediately after last stat (p99)
        p99_idx = gauge_header.index("p99")
        desc_idx = gauge_header.index("Description")
        assert desc_idx == p99_idx + 1

    def test_generate_content_only_info_metrics(
        self, mock_user_config, mock_profile_results
    ):
        """Test that only info metrics results in no gauge section, only info section."""
        results = _create_server_metrics_results(
            {
                "localhost:8081": _create_endpoint_summary(
                    metrics={
                        "vllm:model_info": GaugeMetricData(
                            description="Model information",
                            series=[
                                GaugeSeries(
                                    labels={"model": "Qwen", "version": "0.6B"},
                                    stats=GaugeStats(avg=1.0),
                                )
                            ],
                        ),
                    },
                )
            }
        )

        content = _generate_csv_content(mock_user_config, mock_profile_results, results)
        assert "Endpoint,Metric,Key,Value,Description" in content
        assert "vllm:model_info" in content
        assert ",model," in content
        assert ",Qwen," in content
        assert ",gauge," not in content  # No gauge section

    def test_generate_content_rows_clustered_by_fill_pattern(
        self, mock_user_config, mock_profile_results
    ):
        """Test that rows with identical fill patterns are grouped together."""
        results = _create_server_metrics_results(
            {
                "localhost:8081": _create_endpoint_summary(
                    metrics={
                        "aaa_metric": GaugeMetricData(
                            description="First metric",
                            series=[
                                GaugeSeries(
                                    labels={"x": "1", "y": "2"},
                                    stats=GaugeStats(avg=1.0),
                                )
                            ],
                        ),
                        "bbb_metric": GaugeMetricData(
                            description="Second metric",
                            series=[
                                GaugeSeries(
                                    labels={"x": "1"}, stats=GaugeStats(avg=2.0)
                                )
                            ],
                        ),
                        "ccc_metric": GaugeMetricData(
                            description="Third metric",
                            series=[
                                GaugeSeries(
                                    labels={"x": "3", "y": "4"},
                                    stats=GaugeStats(avg=3.0),
                                )
                            ],
                        ),
                    },
                )
            }
        )

        content = _generate_csv_content(mock_user_config, mock_profile_results, results)
        rows = _parse_csv_content(content)
        data_rows = [r for r in rows if r and r[0] == "localhost:8081"]
        metric_names = [r[2] for r in data_rows]

        aaa_idx = metric_names.index("aaa_metric")
        ccc_idx = metric_names.index("ccc_metric")
        bbb_idx = metric_names.index("bbb_metric")

        # aaa and ccc (same pattern {x, y}) should be adjacent
        assert abs(aaa_idx - ccc_idx) == 1, (
            f"Same pattern metrics should be adjacent: {metric_names}"
        )
        # bbb should not be between aaa and ccc
        assert not (min(aaa_idx, ccc_idx) < bbb_idx < max(aaa_idx, ccc_idx))


class TestCsvMetricInfoProperty:
    """Test the CsvMetricInfo.is_info_metric property."""

    def test_is_info_metric_by_name_suffix(self):
        """Test that metric name ending with _info is detected as info metric."""
        metric = CsvMetricInfo(
            endpoint="localhost:8081",
            metric_name="vllm:cache_config_info",
            description="Cache config",
            unit=None,
            stats=GaugeSeries(),
        )
        assert metric.is_info_metric is True

    def test_is_not_info_metric(self):
        """Test that regular metrics are not detected as info metrics."""
        metric = CsvMetricInfo(
            endpoint="localhost:8081",
            metric_name="vllm:kv_cache_usage_perc",
            description="KV cache usage",
            unit=GenericMetricUnit.PERCENT,
            stats=GaugeSeries(),
        )
        assert metric.is_info_metric is False


class TestServerMetricsCsvExporterIntegration:
    """Integration tests for full export flow."""

    @pytest.mark.asyncio
    async def test_export_creates_valid_csv_file(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
        tmp_path,
    ):
        """Test that export creates a valid CSV file."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        await exporter.export()

        # Read and parse the exported file
        output_file = mock_user_config.output.server_metrics_export_csv_file
        assert output_file.exists()

        with open(output_file) as f:
            content = f.read()

        # Verify column headers for different metric types exist
        assert "avg,min,max,std,p1,p5,p10,p25,p50,p75,p90,p95,p99" in content  # gauge
        assert "total,rate,rate_avg" in content  # counter
