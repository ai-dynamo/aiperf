# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aggregate exporters."""

import json

from aiperf.exporters.aggregate import (
    AggregateConfidenceCsvExporter,
    AggregateConfidenceJsonExporter,
    AggregateExporterConfig,
)
from aiperf.orchestrator.aggregation.base import AggregateResult
from aiperf.orchestrator.aggregation.confidence import ConfidenceMetric


class TestAggregateExporters:
    """Tests for aggregate exporters."""

    async def test_write_aggregate_json(self, tmp_path):
        """Test writing aggregate result to JSON."""
        # Create a simple aggregate result
        aggregate = AggregateResult(
            aggregation_type="confidence",
            num_runs=3,
            num_successful_runs=3,
            failed_runs=[],
            metrics={
                "ttft_avg": ConfidenceMetric(
                    mean=105.0,
                    std=5.0,
                    min=100.0,
                    max=110.0,
                    cv=4.76,
                    se=2.89,
                    ci_low=98.5,
                    ci_high=111.5,
                    t_critical=2.262,
                    unit="ms",
                )
            },
            metadata={"confidence_level": 0.95},
        )

        # Write JSON using exporter
        output_dir = tmp_path / "aggregate"
        config = AggregateExporterConfig(result=aggregate, output_dir=output_dir)
        exporter = AggregateConfidenceJsonExporter(config)
        json_path = await exporter.export()

        # Verify file exists
        assert json_path.exists()
        assert json_path.name == "profile_export_aiperf_aggregate.json"
        assert json_path.parent == output_dir

        # Verify content
        with open(json_path) as f:
            data = json.load(f)

        # Check schema and version info (from existing exporters)
        assert "schema_version" in data
        assert "aiperf_version" in data

        # Check aggregate metadata
        assert "metadata" in data
        assert data["metadata"]["aggregation_type"] == "confidence"
        assert data["metadata"]["num_profile_runs"] == 3
        assert data["metadata"]["num_successful_runs"] == 3
        assert data["metadata"]["confidence_level"] == 0.95

        # Check metrics
        assert "metrics" in data
        assert "ttft_avg" in data["metrics"]
        assert data["metrics"]["ttft_avg"]["mean"] == 105.0
        assert data["metrics"]["ttft_avg"]["std"] == 5.0
        assert data["metrics"]["ttft_avg"]["min"] == 100.0
        assert data["metrics"]["ttft_avg"]["max"] == 110.0
        assert data["metrics"]["ttft_avg"]["unit"] == "ms"

        # Check confidence-specific fields
        assert data["metrics"]["ttft_avg"]["cv"] == 4.76
        assert data["metrics"]["ttft_avg"]["se"] == 2.89
        assert data["metrics"]["ttft_avg"]["ci_low"] == 98.5
        assert data["metrics"]["ttft_avg"]["ci_high"] == 111.5
        assert data["metrics"]["ttft_avg"]["t_critical"] == 2.262

    async def test_write_aggregate_csv(self, tmp_path):
        """Test writing aggregate result to CSV."""
        # Create aggregate result with multiple metrics
        aggregate = AggregateResult(
            aggregation_type="confidence",
            num_runs=3,
            num_successful_runs=3,
            failed_runs=[],
            metrics={
                "ttft_avg": ConfidenceMetric(
                    mean=105.0,
                    std=5.0,
                    min=100.0,
                    max=110.0,
                    cv=4.76,
                    se=2.89,
                    ci_low=98.5,
                    ci_high=111.5,
                    t_critical=2.262,
                    unit="ms",
                ),
                "tpot_avg": ConfidenceMetric(
                    mean=11.0,
                    std=1.0,
                    min=10.0,
                    max=12.0,
                    cv=9.09,
                    se=0.58,
                    ci_low=9.7,
                    ci_high=12.3,
                    t_critical=2.262,
                    unit="ms",
                ),
            },
            metadata={"confidence_level": 0.95},
        )

        # Write CSV using exporter
        output_dir = tmp_path / "aggregate"
        config = AggregateExporterConfig(result=aggregate, output_dir=output_dir)
        exporter = AggregateConfidenceCsvExporter(config)
        csv_path = await exporter.export()

        # Verify file exists
        assert csv_path.exists()
        assert csv_path.name == "profile_export_aiperf_aggregate.csv"
        assert csv_path.parent == output_dir

        # Verify content - read as text to check structure
        with open(csv_path) as f:
            content = f.read()

        # Check that metadata section exists (without "Aggregate Metadata" header)
        assert "confidence" in content
        assert "Confidence Level" in content or "confidence_level" in content

        # Check that metrics section exists
        assert "ttft_avg" in content
        assert "tpot_avg" in content
        assert "105.00" in content  # ttft mean
        assert "11.00" in content  # tpot mean

    async def test_write_creates_directory(self, tmp_path):
        """Test that write methods create output directory if it doesn't exist."""
        aggregate = AggregateResult(
            aggregation_type="confidence",
            num_runs=2,
            num_successful_runs=2,
            failed_runs=[],
            metrics={
                "metric1": ConfidenceMetric(
                    mean=100.0,
                    std=5.0,
                    min=95.0,
                    max=105.0,
                    cv=5.0,
                    se=3.54,
                    ci_low=90.0,
                    ci_high=110.0,
                    t_critical=2.0,
                    unit="ms",
                )
            },
            metadata={"key": "value"},
        )

        # Use non-existent directory
        output_dir = tmp_path / "nested" / "path" / "aggregate"
        assert not output_dir.exists()

        # Write should create directory
        config = AggregateExporterConfig(result=aggregate, output_dir=output_dir)
        exporter = AggregateConfidenceJsonExporter(config)
        json_path = await exporter.export()

        assert output_dir.exists()
        assert output_dir.is_dir()
        assert json_path.exists()

    def test_confidence_metric_to_json_result(self):
        """Test ConfidenceMetric.to_json_result() conversion."""
        metric = ConfidenceMetric(
            mean=100.0,
            std=5.0,
            min=95.0,
            max=105.0,
            cv=5.0,
            se=3.54,
            ci_low=90.0,
            ci_high=110.0,
            t_critical=2.0,
            unit="ms",
        )

        json_result = metric.to_json_result()

        # Check that it's a JsonMetricResult
        from aiperf.common.models.export_models import JsonMetricResult

        assert isinstance(json_result, JsonMetricResult)

        # Check field mapping
        assert json_result.avg == 100.0  # mean → avg
        assert json_result.std == 5.0
        assert json_result.min == 95.0
        assert json_result.max == 105.0
        assert json_result.unit == "ms"
