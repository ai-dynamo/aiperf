# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ArchetypeMetricsCsvExporter."""

import csv
import tempfile
from pathlib import Path

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import MetricResult
from aiperf.exporters.archetype_metrics_csv_exporter import ArchetypeMetricsCsvExporter
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.plugin.enums import EndpointType


@pytest.fixture
def mock_user_config():
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="custom_endpoint",
        )
    )


@pytest.fixture
def sample_archetype_results():
    return {
        "image-only": [
            MetricResult(
                tag="request_latency",
                header="Request Latency",
                unit="ms",
                avg=30.0,
                min=12.0,
                max=120.0,
                p50=28.0,
                p95=95.0,
                p99=110.0,
                std=10.0,
            ),
        ],
        "video-only": [
            MetricResult(
                tag="request_latency",
                header="Request Latency",
                unit="ms",
                avg=890.0,
                min=210.0,
                max=3400.0,
                p50=850.0,
                p95=2100.0,
                p99=3200.0,
                std=420.0,
            ),
        ],
    }


@pytest.fixture
def mock_results_with_archetypes(sample_archetype_results):
    class MockResultsWithArchetypes:
        def __init__(self):
            self.archetype_metric_results = sample_archetype_results
            self.records = []
            self.start_ns = None
            self.end_ns = None
            self.has_results = True
            self.was_cancelled = False
            self.error_summary = []

    return MockResultsWithArchetypes()


@pytest.fixture
def mock_results_without_archetypes():
    class MockResultsNoArchetypes:
        def __init__(self):
            self.archetype_metric_results = None
            self.records = []
            self.start_ns = None
            self.end_ns = None
            self.has_results = False
            self.was_cancelled = False
            self.error_summary = []

    return MockResultsNoArchetypes()


class TestArchetypeMetricsCsvExporter:
    """Initialization, file path, and tidy-format generation for the new exporter."""

    def test_disabled_without_archetype_data(
        self, mock_results_without_archetypes, mock_user_config
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)
            config = ExporterConfig(
                results=mock_results_without_archetypes,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )
            with pytest.raises(DataExporterDisabled, match="no archetype metric"):
                ArchetypeMetricsCsvExporter(config)

    def test_file_path_uses_archetypes_suffix(
        self, mock_results_with_archetypes, mock_user_config
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)
            config = ExporterConfig(
                results=mock_results_with_archetypes,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )
            exporter = ArchetypeMetricsCsvExporter(config)
            assert exporter._file_path.name.endswith("_archetypes.csv")
            assert exporter._file_path.parent == Path(temp_dir)

    @pytest.mark.asyncio
    async def test_tidy_long_format_content(
        self, mock_results_with_archetypes, mock_user_config
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)
            config = ExporterConfig(
                results=mock_results_with_archetypes,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )
            exporter = ArchetypeMetricsCsvExporter(config)
            await exporter.export()

            with open(exporter._file_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

        # Header matches tidy format
        assert rows[0] == ["Archetype", "Metric", "Unit", "Stat", "Value"]

        # Each row has 5 cols and one row per (archetype, metric, stat).
        # Both archetypes appear; sorted alphabetically.
        archetypes_in_rows = {row[0] for row in rows[1:]}
        assert archetypes_in_rows == {"image-only", "video-only"}

        # Spot-check one specific cell
        image_avg_row = next(
            row
            for row in rows[1:]
            if row[0] == "image-only"
            and row[1] == "Request Latency"
            and row[3] == "avg"
        )
        assert image_avg_row[2] == "ms"
        assert float(image_avg_row[4]) == 30.0
