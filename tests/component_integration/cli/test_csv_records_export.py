# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for CSV per-record export."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestCSVRecordsExport:
    """Tests for CSV per-record metrics export."""

    def test_csv_records_file_created(self, cli: AIPerfCLI):
        """Test that profile_export_records.csv is created alongside JSONL."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.csv_records is not None, (
            "CSV records file should be created at default export level"
        )
        assert len(result.csv_records) == defaults.request_count

    def test_csv_records_has_metadata_columns(self, cli: AIPerfCLI):
        """Test that CSV records contain all expected metadata columns."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.csv_records is not None
        assert len(result.csv_records) > 0

        row = result.csv_records[0]
        expected_metadata = [
            "request_num",
            "session_num",
            "x_request_id",
            "x_correlation_id",
            "conversation_id",
            "turn_index",
            "benchmark_phase",
            "worker_id",
            "record_processor_id",
            "request_start_ns",
            "request_end_ns",
            "was_cancelled",
            "error_code",
            "error_message",
        ]
        for col in expected_metadata:
            assert col in row, f"Missing metadata column: {col}"

    def test_csv_records_has_metric_columns(self, cli: AIPerfCLI):
        """Test that CSV records contain metric columns with units in header."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.csv_records is not None
        assert len(result.csv_records) > 0

        row = result.csv_records[0]
        headers = list(row.keys())

        # Metric columns should have format "metric_tag (unit)"
        metric_headers = [h for h in headers if "(" in h and ")" in h]
        assert len(metric_headers) > 0, f"No metric columns found. Headers: {headers}"

        # Streaming chat should have request_latency and TTFT metrics
        metric_names = [h.split(" (")[0] for h in metric_headers]
        assert "request_latency" in metric_names, (
            f"request_latency not found in: {metric_names}"
        )

    def test_csv_records_metadata_values_valid(self, cli: AIPerfCLI):
        """Test that CSV record metadata values are populated and valid."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.csv_records is not None

        for row in result.csv_records:
            assert row["benchmark_phase"] == "profiling"
            assert row["worker_id"] != ""
            assert row["record_processor_id"] != ""
            assert row["x_request_id"] != ""
            assert row["request_start_ns"] != ""
            assert row["request_end_ns"] != ""
            assert int(row["request_start_ns"]) > 0
            assert int(row["request_end_ns"]) > 0
            assert int(row["request_end_ns"]) >= int(row["request_start_ns"])
            # No errors expected for successful requests
            assert row["error_code"] == ""
            assert row["error_message"] == ""

    def test_per_chunk_data_excluded_by_default(self, cli: AIPerfCLI):
        """Test that per-chunk list arrays are excluded from both CSV and JSONL by default."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # CSV: all metric values should be scalar (no "[...] arrays)
        assert result.csv_records is not None
        for row in result.csv_records:
            for header in (h for h in row if "(" in h and ")" in h):
                value = row[header]
                if value:
                    assert not value.startswith("["), (
                        f"CSV metric '{header}' should not contain list data by default"
                    )
                    float(value)

        # JSONL: no metric should have a list value
        assert result.jsonl is not None
        for record in result.jsonl:
            for tag, metric_value in record.metrics.items():
                assert not isinstance(metric_value.value, list), (
                    f"JSONL metric '{tag}' should not contain list data by default"
                )

    def test_per_chunk_data_included_with_flag(self, cli: AIPerfCLI):
        """Test that --export-per-chunk-data includes list arrays in both CSV and JSONL."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --export-per-chunk-data \
                --ui {defaults.ui}
            """
        )

        # CSV: should have at least one list-valued column
        assert result.csv_records is not None
        csv_has_list = any(
            v.startswith("[") for row in result.csv_records for v in row.values() if v
        )
        assert csv_has_list, (
            "CSV should contain list-valued metrics with --export-per-chunk-data"
        )

        # JSONL: should have at least one list-valued metric
        assert result.jsonl is not None
        jsonl_has_list = any(
            isinstance(mv.value, list)
            for record in result.jsonl
            for mv in record.metrics.values()
        )
        assert jsonl_has_list, (
            "JSONL should contain list-valued metrics with --export-per-chunk-data"
        )

    def test_csv_records_count_matches_jsonl(self, cli: AIPerfCLI):
        """Test that CSV and JSONL have the same number of records."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.csv_records is not None
        assert result.jsonl is not None
        assert len(result.csv_records) == len(result.jsonl), (
            f"CSV has {len(result.csv_records)} records but JSONL has {len(result.jsonl)}"
        )

    def test_csv_records_not_created_at_summary_level(self, cli: AIPerfCLI):
        """Test that CSV records are NOT created when export level is summary."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --export-level summary \
                --ui {defaults.ui}
            """
        )

        assert result.csv_records is None, (
            "CSV records should NOT be created at summary export level"
        )

    def test_csv_records_created_at_raw_level(self, cli: AIPerfCLI):
        """Test that CSV records are created when export level is raw."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --export-level raw \
                --ui {defaults.ui}
            """
        )

        assert result.csv_records is not None, (
            "CSV records should be created at raw export level"
        )
        assert len(result.csv_records) == defaults.request_count

    def test_csv_records_non_streaming(self, cli: AIPerfCLI):
        """Test CSV export works for non-streaming endpoint."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.csv_records is not None
        assert len(result.csv_records) == defaults.request_count

        row = result.csv_records[0]
        metric_headers = [h for h in row if "(" in h and ")" in h]
        metric_names = [h.split(" (")[0] for h in metric_headers]
        assert "request_latency" in metric_names
