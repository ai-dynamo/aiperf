# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for analyze_trace CLI command."""

import tempfile
from pathlib import Path

import orjson
import pytest

from aiperf.cli_commands.analyze_trace import _build_stats_table, analyze_trace
from aiperf.dataset.synthesis import MetricStats


class TestBuildStatsTable:
    """Tests for _build_stats_table function."""

    def test_build_stats_table_with_stats(self) -> None:
        """Test building table with valid MetricStats."""
        stats = MetricStats(
            mean=100.0,
            std_dev=10.0,
            min=50.0,
            max=150.0,
            p25=80.0,
            median=100.0,
            p75=120.0,
        )
        metrics = {"Test Metric": stats}

        table = _build_stats_table(metrics)

        assert table.title == "Trace Statistics"
        assert len(table.columns) == 8  # Metric + 7 stat columns
        assert table.row_count == 1

    def test_build_stats_table_with_none(self) -> None:
        """Test building table with None stats shows N/A."""
        metrics = {"Missing Metric": None}

        table = _build_stats_table(metrics)

        assert table.row_count == 1

    def test_build_stats_table_multiple_metrics(self) -> None:
        """Test building table with multiple metrics."""
        stats1 = MetricStats(
            mean=100.0,
            std_dev=10.0,
            min=50.0,
            max=150.0,
            p25=80.0,
            median=100.0,
            p75=120.0,
        )
        stats2 = MetricStats(
            mean=200.0,
            std_dev=20.0,
            min=100.0,
            max=300.0,
            p25=160.0,
            median=200.0,
            p75=240.0,
        )
        metrics = {
            "Metric 1": stats1,
            "Metric 2": stats2,
            "Metric 3": None,
        }

        table = _build_stats_table(metrics)

        assert table.row_count == 3


class TestAnalyzeTrace:
    """Tests for analyze_trace CLI function."""

    @pytest.fixture
    def sample_trace_file(self) -> Path:
        """Create a temporary trace file for testing."""
        traces = [
            {"input_length": 100, "output_length": 20, "hash_ids": [1, 2]},
            {"input_length": 150, "output_length": 30, "hash_ids": [1, 2, 3]},
            {"input_length": 120, "output_length": 25, "hash_ids": [1, 2]},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for trace in traces:
                f.write(orjson.dumps(trace).decode() + "\n")
            filepath = f.name

        yield Path(filepath)

        # Cleanup
        Path(filepath).unlink(missing_ok=True)

    def test_analyze_trace_basic(self, sample_trace_file: Path) -> None:
        """Test analyze_trace with valid trace file."""
        # Should not raise any exceptions
        analyze_trace(input_file=sample_trace_file, block_size=512)

    def test_analyze_trace_with_output_file(self, sample_trace_file: Path) -> None:
        """Test analyze_trace writes output to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "report.json"

            analyze_trace(
                input_file=sample_trace_file,
                block_size=512,
                output_file=output_file,
            )

            assert output_file.exists()
            # Verify it's valid JSON
            data = orjson.loads(output_file.read_bytes())
            assert "total_requests" in data

    def test_analyze_trace_nonexistent_file(self, capsys) -> None:
        """Test analyze_trace with nonexistent file prints error."""
        analyze_trace(input_file=Path("/nonexistent/file.jsonl"))

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "not found" in captured.out
