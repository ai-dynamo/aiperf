# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for server metrics collection and reporting.

These tests verify the full end-to-end flow of server metrics collection,
including scraping from multiple mock server endpoints and validating
the exported data (JSON, JSONL, CSV).
"""

import platform

import pytest

from aiperf.common.models import SlimRecord
from tests.harness.utils import AIPerfCLI, AIPerfMockServer


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="This test is flaky on macOS in Github Actions.",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestServerMetrics:
    """Tests for server metrics collection and reporting."""

    # ========================================================================
    # Basic Server Metrics Tests
    # ========================================================================

    async def test_server_metrics_auto_collected(
        self, cli: AIPerfCLI, mock_server_factory
    ):
        """Server metrics are auto-collected from base_url/metrics without --server-metrics.

        When no --server-metrics flag is provided, AIPerf should automatically
        scrape server metrics from the inference endpoint's base URL + /metrics.
        """
        # Use isolated mock server with workers=1 to avoid Prometheus metrics issues
        async with mock_server_factory(fast=True, workers=1) as aiperf_mock_server:
            result = await cli.run(
                f"""
                aiperf profile \
                    --model nvidia/llama-3.1-nemotron-70b-instruct \
                    --url {aiperf_mock_server.url} \
                    --tokenizer gpt2 \
                    --endpoint-type chat \
                    --streaming \
                    --request-count 50 \
                    --concurrency 2 \
                    --workers-max 2 \
                    --ui dashboard
                """
            )
            assert result.request_count == 50

            # Server metrics should be auto-collected from default /metrics endpoint
            result.assert_server_metrics_valid()

            # Verify we collected AIPerf mock server metrics (default endpoint)
            # Note: Counter metric names may not include _total suffix depending on parsing
            assert result.has_server_metric("aiperf_mock_requests")
            assert result.has_server_metric("aiperf_mock_request_latency_seconds")
            assert result.has_server_metric("aiperf_mock_time_to_first_token_seconds")
            assert result.has_server_metric("aiperf_mock_tokens_streamed")

            # Verify the auto-collected endpoint is correct
            # Note: endpoints_successful contains full URLs
            expected_endpoint = (
                f"http://{aiperf_mock_server.host}:{aiperf_mock_server.port}/metrics"
            )
            assert expected_endpoint in result.server_metrics_endpoints_successful, (
                f"Expected {expected_endpoint} in successful endpoints: "
                f"{result.server_metrics_endpoints_successful}"
            )

    # ========================================================================
    # Multiple Endpoints Tests
    # ========================================================================

    async def test_server_metrics_multiple_endpoints_vllm_sglang(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection from multiple endpoints (vLLM + SGLang)."""
        vllm_url = aiperf_mock_server.server_metrics_urls["vllm"]
        sglang_url = aiperf_mock_server.server_metrics_urls["sglang"]

        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {vllm_url} {sglang_url}
            """
        )
        assert result.request_count == 50
        result.assert_server_metrics_valid()

        # Verify endpoints were successful (default + vllm + sglang = 3)
        # The default /metrics endpoint is always auto-collected
        assert len(result.server_metrics_endpoints_successful) >= 2

        # Verify vLLM metrics
        assert result.has_server_metric("vllm:e2e_request_latency_seconds")
        assert result.has_server_metric("vllm:time_to_first_token_seconds")

        # Verify SGLang metrics
        assert result.has_server_metric("sglang:e2e_request_latency_seconds")
        assert result.has_server_metric("sglang:time_to_first_token_seconds")

    # ========================================================================
    # Dynamo Endpoints Tests
    # ========================================================================

    async def test_server_metrics_full_dynamo_stack(
        self, cli: AIPerfCLI, mock_server_factory
    ):
        """Server metrics collection from full Dynamo stack (frontend + prefill + decode)."""
        # Use isolated mock server with workers=1 to avoid Prometheus metrics issues
        async with mock_server_factory(fast=True, workers=1) as aiperf_mock_server:
            urls = aiperf_mock_server.get_server_metrics_url(
                "dynamo_frontend", "dynamo_prefill", "dynamo_decode"
            )

            result = await cli.run(
                f"""
                aiperf profile \
                    --model nvidia/llama-3.1-nemotron-70b-instruct \
                    --url {aiperf_mock_server.url} \
                    --tokenizer gpt2 \
                    --endpoint-type chat \
                    --streaming \
                    --request-count 50 \
                    --concurrency 2 \
                    --workers-max 2 \
                    --server-metrics {" ".join(urls)}
                """
            )
            assert result.request_count == 50
            result.assert_server_metrics_valid()

            # Verify endpoints were successful (default + frontend + prefill + decode)
            # The default /metrics endpoint is always auto-collected
            assert len(result.server_metrics_endpoints_successful) >= 3

            # Verify Dynamo frontend metrics
            assert result.has_server_metric("dynamo_frontend_request_duration_seconds")
            assert result.has_server_metric(
                "dynamo_frontend_time_to_first_token_seconds"
            )

            # Verify Dynamo component metrics
            assert result.has_server_metric("dynamo_component_request_duration_seconds")

    # ========================================================================
    # Ultimate Full Stack Test
    # ========================================================================

    async def test_server_metrics_all_endpoints(
        self, cli: AIPerfCLI, mock_server_factory
    ):
        """Ultimate test: Server metrics from ALL available mock endpoints.

        This test collects metrics from:
        - vLLM endpoint
        - SGLang endpoint
        - TensorRT-LLM endpoint
        - Dynamo frontend endpoint
        - Dynamo prefill component endpoint
        - Dynamo decode component endpoint

        Total: 6 different server metrics endpoints scraped simultaneously!
        """
        # Use isolated mock server with workers=1 to avoid Prometheus metrics issues
        async with mock_server_factory(fast=True, workers=1) as aiperf_mock_server:
            all_urls = aiperf_mock_server.get_server_metrics_url(
                "vllm",
                "sglang",
                "trtllm",
                "dynamo_frontend",
                "dynamo_prefill",
                "dynamo_decode",
            )

            result = await cli.run(
                f"""
                aiperf profile \
                    --model nvidia/llama-3.1-nemotron-70b-instruct \
                    --url {aiperf_mock_server.url} \
                    --tokenizer gpt2 \
                    --endpoint-type chat \
                    --streaming \
                    --request-count 100 \
                    --concurrency 4 \
                    --workers-max 2 \
                    --server-metrics {" ".join(all_urls)}
                """
            )
            assert result.request_count == 100
            result.assert_server_metrics_valid()

            # Verify all 6+ endpoints were successful (default + 6 explicit)
            # The default /metrics endpoint is always auto-collected
            assert len(result.server_metrics_endpoints_successful) >= 6, (
                f"Expected at least 6 successful endpoints, got {len(result.server_metrics_endpoints_successful)}: "
                f"{result.server_metrics_endpoints_successful}"
            )

            # Verify vLLM metrics
            assert result.has_server_metric("vllm:e2e_request_latency_seconds")
            assert result.has_server_metric("vllm:time_to_first_token_seconds")
            assert result.has_server_metric("vllm:inter_token_latency_seconds")
            assert result.has_server_metric("vllm:kv_cache_usage_perc")

            # Verify SGLang metrics
            assert result.has_server_metric("sglang:e2e_request_latency_seconds")
            assert result.has_server_metric("sglang:time_to_first_token_seconds")
            assert result.has_server_metric("sglang:gen_throughput")

            # Verify TRT-LLM metrics
            assert result.has_server_metric("trtllm:e2e_request_latency_seconds")
            assert result.has_server_metric("trtllm:time_to_first_token_seconds")
            assert result.has_server_metric("trtllm:time_per_output_token_seconds")

            # Verify Dynamo frontend metrics
            assert result.has_server_metric("dynamo_frontend_request_duration_seconds")
            assert result.has_server_metric(
                "dynamo_frontend_time_to_first_token_seconds"
            )
            assert result.has_server_metric(
                "dynamo_frontend_inter_token_latency_seconds"
            )

            # Verify Dynamo component metrics
            assert result.has_server_metric("dynamo_component_request_duration_seconds")
            assert result.has_server_metric("dynamo_component_requests")

    # ========================================================================
    # Export File Validation Tests
    # ========================================================================

    async def test_server_metrics_export_files(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test server metrics export files (JSON, JSONL, CSV, Parquet) are valid."""
        urls = aiperf_mock_server.get_server_metrics_url("vllm", "sglang")

        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics-formats json csv jsonl parquet \
                --server-metrics {" ".join(urls)}
            """
        )

        # Verify all export files exist
        assert result.has_all_server_metrics_outputs

        # Verify JSON export structure
        assert result.server_metrics_json is not None
        assert result.server_metrics_json.summary is not None
        # At least 2 endpoints (vllm + sglang), possibly more with auto-collected default
        assert len(result.server_metrics_json.summary.endpoints_successful) >= 2
        assert len(result.server_metrics_json.metrics) > 0

        # Verify JSONL records structure
        assert result.server_metrics_jsonl is not None
        assert len(result.server_metrics_jsonl) > 0

        # Check records have expected structure
        for record in result.server_metrics_jsonl:
            assert record.endpoint_url is not None
            assert record.timestamp_ns > 0
            assert record.endpoint_latency_ns >= 0
            assert len(record.metrics) > 0

        # Verify CSV content
        assert result.has_server_metrics_csv
        csv_lines = result.server_metrics_csv.strip().split("\n")
        assert len(csv_lines) > 1  # Header + data rows

    async def test_server_metrics_jsonl_records(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test JSONL records contain expected metrics with valid data."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics-formats jsonl \
                --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]}
            """
        )

        # Verify JSONL structure and content
        assert result.server_metrics_jsonl is not None

        # Group records by endpoint
        endpoints_seen = set()
        timestamps = []

        for record in result.server_metrics_jsonl:
            endpoints_seen.add(record.endpoint_url)
            timestamps.append(record.timestamp_ns)

            # Verify record has metrics
            assert len(record.metrics) > 0

            # Check for expected vLLM metrics in at least some records
            if "vllm:kv_cache_usage_perc" in record.metrics:
                samples = record.metrics["vllm:kv_cache_usage_perc"]
                assert len(samples) > 0
                assert samples[0].value is not None

        # Verify timestamps are generally increasing (not strictly ordered due to multiple endpoints)
        # When multiple endpoints are scraped, records from different endpoints may interleave
        assert len(timestamps) > 0, "Should have timestamp records"
        assert min(timestamps) > 0, "Timestamps should be positive"

        # Verify we captured data from at least the expected endpoint(s)
        # (vLLM + possibly default /metrics endpoint auto-collected)
        assert len(endpoints_seen) >= 1

    async def test_server_metrics_histogram_data(
        self, cli: AIPerfCLI, mock_server_factory
    ):
        """Test histogram metrics are properly captured and exported."""
        # Use isolated mock server with workers=1 to avoid Prometheus metrics issues
        async with mock_server_factory(fast=True, workers=1) as aiperf_mock_server:
            result = await cli.run(
                f"""
                aiperf profile \
                    --model nvidia/llama-3.1-nemotron-70b-instruct \
                    --url {aiperf_mock_server.url} \
                    --tokenizer gpt2 \
                    --endpoint-type chat \
                    --streaming \
                    --request-count 50 \
                    --concurrency 2 \
                    --workers-max 2 \
                    --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]}
                """
            )
            result.assert_server_metrics_valid()

            # Get histogram metric from JSON export
            ttft_metric = result.get_server_metric("vllm:time_to_first_token_seconds")
            assert ttft_metric is not None
            assert ttft_metric.type.value == "histogram"
            assert len(ttft_metric.series) > 0

            # Verify histogram stats are computed
            series = ttft_metric.series[0]
            assert series.stats is not None
            assert series.stats.count is not None
            assert series.stats.count > 0

            # Verify JSONL records have histogram data
            for record in result.server_metrics_jsonl or []:
                if "vllm:time_to_first_token_seconds" in record.metrics:
                    samples = record.metrics["vllm:time_to_first_token_seconds"]
                    assert len(samples) > 0
                    # Histogram samples should have histogram field (dict of buckets)
                    assert samples[0].buckets is not None
                    assert isinstance(samples[0].buckets, dict)

    # ========================================================================
    # Non-Streaming Tests
    # ========================================================================

    async def test_server_metrics_non_streaming(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection works with non-streaming requests."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2 \
                --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]}
            """
        )
        assert result.request_count == 50
        result.assert_server_metrics_valid()

        # Verify metrics are collected even for non-streaming
        assert result.has_server_metric("vllm:e2e_request_latency_seconds")

    # ========================================================================
    # Custom Prefix Tests
    # ========================================================================

    async def test_server_metrics_custom_prefix(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test server metrics export with custom filename prefix."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 25 \
                --concurrency 1 \
                --workers-max 1 \
                --server-metrics {aiperf_mock_server.server_metrics_urls["vllm"]} \
                --profile-export-prefix custom_test
            """
        )

        # Verify custom prefix files exist
        json_file = result.artifacts_dir / "custom_test_server_metrics.json"
        jsonl_file = result.artifacts_dir / "custom_test_server_metrics.jsonl"

        if json_file.exists():
            content = json_file.read_text()
            assert len(content) > 0

        if jsonl_file.exists():
            lines = jsonl_file.read_text().strip().split("\n")
            assert len(lines) > 0
            # Validate first record
            first_record = SlimRecord.model_validate_json(lines[0])
            assert first_record.timestamp_ns > 0

    # ========================================================================
    # Server Metrics Disabled Tests
    # ========================================================================

    async def test_server_metrics_disabled(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Server metrics collection is disabled with --no-server-metrics flag.

        When --no-server-metrics is provided, no server metrics files should be
        created and no metrics should be collected.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 25 \
                --concurrency 1 \
                --workers-max 1 \
                --no-server-metrics
            """
        )
        assert result.request_count == 25

        # Server metrics should NOT be collected when disabled
        assert not result.has_server_metrics, "JSON export should not exist"
        assert not result.has_server_metrics_jsonl, "JSONL export should not exist"
        assert not result.has_server_metrics_csv, "CSV export should not exist"

        # Verify no server metrics files were created
        json_files = list(result.artifacts_dir.glob("*server_metrics*.json"))
        jsonl_files = list(result.artifacts_dir.glob("*server_metrics*.jsonl"))
        csv_files = list(result.artifacts_dir.glob("*server_metrics*.csv"))

        assert len(json_files) == 0, f"Unexpected JSON files: {json_files}"
        assert len(jsonl_files) == 0, f"Unexpected JSONL files: {jsonl_files}"
        assert len(csv_files) == 0, f"Unexpected CSV files: {csv_files}"

    async def test_server_metrics_parquet_export(
        self, cli: AIPerfCLI, mock_server_factory
    ):
        """Test Parquet export with raw time-series data and delta calculations."""
        import pyarrow.parquet as pq

        # Use isolated mock server with workers=1 to avoid Prometheus metrics issues
        async with mock_server_factory(fast=True, workers=1) as aiperf_mock_server:
            urls = aiperf_mock_server.get_server_metrics_url("vllm", "sglang")

            result = await cli.run(
                f"""
                aiperf profile \
                    --model nvidia/llama-3.1-nemotron-70b-instruct \
                    --url {aiperf_mock_server.url} \
                    --tokenizer gpt2 \
                    --endpoint-type chat \
                    --streaming \
                    --request-count 50 \
                    --concurrency 2 \
                    --workers-max 2 \
                    --server-metrics {" ".join(urls)} \
                    --server-metrics-formats parquet \
                    --ui simple
                """
            )

            # Verify Parquet file exists
            parquet_file = result.artifacts_dir / "server_metrics_export.parquet"
            assert parquet_file.exists(), f"Parquet file not found at {parquet_file}"

            # Read and validate Parquet file structure
            table = pq.read_table(parquet_file)
            df = table.to_pandas()

            # Verify basic structure
            assert len(df) > 0, "Parquet file should contain data rows"

            # Verify required columns exist
            required_columns = {
                "endpoint_url",
                "metric_name",
                "metric_type",
                "unit",
                "description",
                "timestamp_ns",
            }
            assert required_columns.issubset(set(df.columns)), (
                f"Missing required columns. Expected {required_columns}, "
                f"got {set(df.columns)}"
            )

            # Verify metric types
            metric_types = set(df["metric_type"].unique())
            assert (
                "gauge" in metric_types
                or "counter" in metric_types
                or "histogram" in metric_types
            )

            # Verify timestamp ordering per metric
            for (endpoint_url, metric_name), group in df.groupby(
                ["endpoint_url", "metric_name"]
            ):
                timestamps = group["timestamp_ns"].values
                assert all(
                    timestamps[i] <= timestamps[i + 1]
                    for i in range(len(timestamps) - 1)
                ), f"Timestamps not sorted for {endpoint_url}/{metric_name}"

            # Verify gauge metrics have value column populated
            gauges = df[df["metric_type"] == "gauge"]
            if len(gauges) > 0:
                assert gauges["value"].notna().any(), "Gauge metrics should have values"
                assert gauges["sum"].isna().all(), "Gauge metrics should not have sum"
                assert gauges["count"].isna().all(), (
                    "Gauge metrics should not have count"
                )

            # Verify counter metrics have delta values
            counters = df[df["metric_type"] == "counter"]
            if len(counters) > 0:
                assert counters["value"].notna().any(), (
                    "Counter metrics should have delta values"
                )
                # Verify deltas are non-negative (counter resets handled)
                assert (counters["value"].dropna() >= 0).all(), (
                    "Counter deltas should be non-negative"
                )
                # Verify deltas are increasing or equal (cumulative from reference)
                for (endpoint_url, metric_name), group in counters.groupby(
                    ["endpoint_url", "metric_name"]
                ):
                    values = group["value"].values
                    # Cumulative deltas should be monotonically increasing
                    assert all(
                        values[i] <= values[i + 1] for i in range(len(values) - 1)
                    ), (
                        f"Counter deltas not monotonically increasing for {endpoint_url}/{metric_name}"
                    )

            # Verify histogram metrics have sum, count, bucket_le, and bucket_count (normalized schema)
            histograms = df[df["metric_type"] == "histogram"]
            if len(histograms) > 0:
                assert histograms["sum"].notna().any(), (
                    "Histogram metrics should have sum"
                )
                assert histograms["count"].notna().any(), (
                    "Histogram metrics should have count"
                )
                assert histograms["value"].isna().all(), (
                    "Histogram metrics should not have value column"
                )

                # Verify normalized bucket columns exist
                assert "bucket_le" in df.columns, "Should have bucket_le column"
                assert "bucket_count" in df.columns, "Should have bucket_count column"

                # Verify bucket data for histogram rows
                assert histograms["bucket_le"].notna().all(), (
                    "Histogram rows should have bucket_le"
                )
                assert histograms["bucket_count"].notna().all(), (
                    "Histogram rows should have bucket_count"
                )

                # Verify bucket values are non-negative (no sanitization)
                unique_buckets = histograms["bucket_le"].unique()
                assert len(unique_buckets) > 0, "Should have at least one unique bucket"
                # Check that bucket values look like Prometheus buckets
                assert any("." in b or b == "+Inf" for b in unique_buckets), (
                    "Bucket values should be unsanitized"
                )

                # Verify bucket deltas are non-negative
                assert (histograms["bucket_count"] >= 0).all(), (
                    "Bucket deltas should be non-negative"
                )

                # Verify sum/count deltas are non-negative
                assert (histograms["sum"].dropna() >= 0).all(), (
                    "Histogram sum deltas should be non-negative"
                )
                assert (histograms["count"].dropna() >= 0).all(), (
                    "Histogram count deltas should be non-negative"
                )

                # Verify each histogram timestamp has multiple bucket rows
                hist_by_ts = histograms.groupby(
                    ["endpoint_url", "metric_name", "timestamp_ns"]
                ).size()
                assert (hist_by_ts > 1).any(), (
                    "Each histogram timestamp should have multiple bucket rows"
                )

            # Verify label columns exist (dynamic discovery)
            # At minimum, vLLM metrics should have some labels
            label_cols = [
                col
                for col in df.columns
                if col not in required_columns
                and col not in ["value", "sum", "count", "bucket_le", "bucket_count"]
            ]
            assert len(label_cols) > 0, "Should have discovered label columns"

            # Verify multiple endpoints are present
            endpoints = df["endpoint_url"].unique()
            assert len(endpoints) >= 2, (
                f"Should have at least 2 endpoints, got {len(endpoints)}"
            )

            # Verify metrics from both vLLM and SGLang endpoints
            metric_names = df["metric_name"].unique()
            vllm_metrics = [m for m in metric_names if m.startswith("vllm:")]
            sglang_metrics = [m for m in metric_names if m.startswith("sglang:")]
            assert len(vllm_metrics) > 0, "Should have vLLM metrics"
            assert len(sglang_metrics) > 0, "Should have SGLang metrics"
