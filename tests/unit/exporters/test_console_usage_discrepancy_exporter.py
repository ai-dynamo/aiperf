# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import GenericMetricUnit, MetricTimeUnit
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.exporters.console_usage_discrepancy_exporter import (
    ConsoleUsageDiscrepancyExporter,
)
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.usage_diff_metrics import UsageDiscrepancyCountMetric
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import create_exporter_config


class TestConsoleUsageDiscrepancyExporter:
    """Tests for ConsoleUsageDiscrepancyExporter."""

    @pytest.fixture
    def mock_user_config(self):
        """Create a mock user config."""
        return UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="custom_endpoint",
            )
        )

    def _create_profile_results(
        self, count: int, total_records: int = 100, include_discrepancy: bool = True
    ) -> ProfileResults:
        """Helper to create a ProfileResults with optional discrepancy count metric."""
        records = []
        if include_discrepancy:
            records.append(
                MetricResult(
                    tag=UsageDiscrepancyCountMetric.tag,
                    header="Usage Discrepancy Count",
                    unit=GenericMetricUnit.REQUESTS,
                    avg=float(count),
                    count=total_records,
                    min=float(count),
                    max=float(count),
                )
            )
        records.extend(
            [
                MetricResult(
                    tag=RequestCountMetric.tag,
                    header="Request Count",
                    unit=GenericMetricUnit.REQUESTS,
                    avg=float(total_records),
                    count=total_records,
                    min=float(total_records),
                    max=float(total_records),
                ),
                MetricResult(
                    tag="time_to_first_token",
                    header="Time to First Token",
                    unit=MetricTimeUnit.MILLISECONDS,
                    avg=100.0,
                    count=total_records,
                    min=50.0,
                    max=150.0,
                ),
            ]
        )
        return ProfileResults(
            records=records,
            completed=total_records,
            start_ns=1000000000,
            end_ns=2000000000,
        )

    async def _get_export_output(
        self, exporter: ConsoleUsageDiscrepancyExporter
    ) -> str:
        """Helper to export to console and return output string."""
        output = StringIO()
        console = Console(file=output, width=120, legacy_windows=False)
        await exporter.export(console)
        return output.getvalue()

    async def test_no_discrepancies_no_output(self, mock_user_config):
        """Test that no warning is displayed when there are no discrepancies."""
        with patch(
            "aiperf.exporters.console_usage_discrepancy_exporter.Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD",
            10.0,
        ):
            exporter = ConsoleUsageDiscrepancyExporter(
                create_exporter_config(
                    self._create_profile_results(count=0, total_records=100),
                    mock_user_config,
                )
            )
            output = await self._get_export_output(exporter)
            assert "Token Count Discrepancy Warning" not in output
            assert "requests" not in output

    async def test_discrepancies_display_warning(self, mock_user_config):
        """Test that warning is displayed when discrepancies exist."""
        with patch(
            "aiperf.exporters.console_usage_discrepancy_exporter.Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD",
            10.0,
        ):
            exporter = ConsoleUsageDiscrepancyExporter(
                create_exporter_config(
                    self._create_profile_results(count=15, total_records=100),
                    mock_user_config,
                )
            )
            output = await self._get_export_output(exporter)
            assert "Token Count Discrepancy Warning" in output
            assert "15 of 100 requests" in output
            assert "(15.0%)" in output
            assert "10%" in output  # threshold

    async def test_warning_includes_investigation_steps(self, mock_user_config):
        """Test that warning includes investigation steps and causes."""
        with patch(
            "aiperf.exporters.console_usage_discrepancy_exporter.Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD",
            10.0,
        ):
            exporter = ConsoleUsageDiscrepancyExporter(
                create_exporter_config(
                    self._create_profile_results(count=20, total_records=100),
                    mock_user_config,
                )
            )
            output = await self._get_export_output(exporter)
            # Check for causes
            assert "Possible Causes:" in output
            assert "Different tokenization methods" in output
            assert "API special tokens" in output
            # Check for investigation steps
            assert "Investigation Steps:" in output
            assert "profile_export.jsonl" in output
            assert "usage_*_diff_pct" in output
            assert "AIPERF_METRICS_USAGE_PCT_DIFF_THRESHOLD" in output

    async def test_custom_threshold_displayed(self, mock_user_config):
        """Test that custom threshold value is displayed in warning."""
        with patch(
            "aiperf.exporters.console_usage_discrepancy_exporter.Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD",
            5.0,
        ):
            exporter = ConsoleUsageDiscrepancyExporter(
                create_exporter_config(
                    self._create_profile_results(count=10, total_records=100),
                    mock_user_config,
                )
            )
            output = await self._get_export_output(exporter)
            assert "5%" in output  # custom threshold
            assert "AIPERF_METRICS_USAGE_PCT_DIFF_THRESHOLD=5" in output

    async def test_high_discrepancy_percentage(self, mock_user_config):
        """Test warning with high percentage of discrepancies."""
        with patch(
            "aiperf.exporters.console_usage_discrepancy_exporter.Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD",
            10.0,
        ):
            exporter = ConsoleUsageDiscrepancyExporter(
                create_exporter_config(
                    self._create_profile_results(count=75, total_records=100),
                    mock_user_config,
                )
            )
            output = await self._get_export_output(exporter)
            assert "75 of 100 requests" in output
            assert "(75.0%)" in output

    async def test_no_discrepancy_metric_no_output(self, mock_user_config):
        """Test that no warning is displayed when discrepancy metric is absent."""
        exporter = ConsoleUsageDiscrepancyExporter(
            create_exporter_config(
                self._create_profile_results(
                    count=0, total_records=100, include_discrepancy=False
                ),
                mock_user_config,
            )
        )
        output = await self._get_export_output(exporter)
        assert "Token Count Discrepancy Warning" not in output

    async def test_formatting_with_large_numbers(self, mock_user_config):
        """Test that large numbers are formatted with commas."""
        with patch(
            "aiperf.exporters.console_usage_discrepancy_exporter.Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD",
            10.0,
        ):
            exporter = ConsoleUsageDiscrepancyExporter(
                create_exporter_config(
                    self._create_profile_results(count=1250, total_records=10000),
                    mock_user_config,
                )
            )
            output = await self._get_export_output(exporter)
            assert "1,250 of 10,000 requests" in output
            assert "(12.5%)" in output
