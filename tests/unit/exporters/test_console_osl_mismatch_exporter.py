# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from aiperf.common.enums import GenericMetricUnit, MetricTimeUnit
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.exporters.console_osl_mismatch_exporter import (
    ConsoleOSLMismatchExporter,
)
from aiperf.metrics.types.osl_mismatch_metrics import OSLMismatchCountMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import create_exporter_config


@pytest.mark.asyncio
class TestConsoleOSLMismatchExporter:
    """Tests for ConsoleOSLMismatchExporter."""

    @pytest.fixture
    def mock_cfg(self):
        """Create a mock user config."""
        return CLIConfig(
            model_names=["test-model"],
            endpoint_type=EndpointType.CHAT,
            custom_endpoint="/custom_endpoint",
        )

    def _create_profile_results(
        self, count: int, total_records: int = 100, include_mismatch: bool = True
    ) -> ProfileResults:
        """Helper to create a ProfileResults with optional OSL mismatch count metric."""
        records = []
        if include_mismatch:
            records.append(
                MetricResult(
                    tag=OSLMismatchCountMetric.tag,
                    header="OSL Mismatch Count",
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

    async def _get_export_output(self, exporter: ConsoleOSLMismatchExporter) -> str:
        """Helper to export to console and return output string."""
        output = StringIO()
        console = Console(file=output, width=120, legacy_windows=False)
        await exporter.export(console)
        return output.getvalue()

    async def test_no_mismatches_no_output(self, mock_cfg):
        """Test that no warning is displayed when there are no OSL mismatches."""
        with patch(
            "aiperf.exporters.console_osl_mismatch_exporter.Environment.METRICS.OSL_MISMATCH_PCT_THRESHOLD",
            20.0,
        ):
            exporter = ConsoleOSLMismatchExporter(
                create_exporter_config(
                    self._create_profile_results(count=0, total_records=100),
                    mock_cfg,
                )
            )
            output = await self._get_export_output(exporter)
            assert "Output Sequence Length Mismatch Warning" not in output
            assert "requests" not in output

    async def test_mismatches_display_warning(self, mock_cfg):
        """Test that warning is displayed when OSL mismatches exist."""
        with patch(
            "aiperf.exporters.console_osl_mismatch_exporter.Environment.METRICS.OSL_MISMATCH_PCT_THRESHOLD",
            20.0,
        ):
            exporter = ConsoleOSLMismatchExporter(
                create_exporter_config(
                    self._create_profile_results(count=25, total_records=100),
                    mock_cfg,
                )
            )
            output = await self._get_export_output(exporter)
            assert "Output Sequence Length Mismatch Warning" in output
            assert "25 of 100 requests" in output
            assert "(25.0%)" in output
            assert "20%" in output  # threshold

    async def test_warning_includes_recommended_actions(self, mock_cfg):
        """Test that warning includes recommended actions."""
        with patch(
            "aiperf.exporters.console_osl_mismatch_exporter.Environment.METRICS.OSL_MISMATCH_PCT_THRESHOLD",
            20.0,
        ):
            exporter = ConsoleOSLMismatchExporter(
                create_exporter_config(
                    self._create_profile_results(count=30, total_records=100),
                    mock_cfg,
                )
            )
            output = await self._get_export_output(exporter)
            # Check for explanation
            assert "Why:" in output
            assert "EOS token" in output
            # Check for fix options
            assert "Fix Options:" in output
            assert "ignore_eos:true" in output
            assert "min_tokens" in output
            assert "--use-server-token-count" in output
            # Check for diagnostics
            assert "Diagnostics:" in output
            assert "profile_export.jsonl" in output
            assert "osl_mismatch_diff_pct" in output
            assert "AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD" in output

    async def test_custom_threshold_displayed(self, mock_cfg):
        """Test that custom threshold value is displayed in warning."""
        with patch(
            "aiperf.exporters.console_osl_mismatch_exporter.Environment.METRICS.OSL_MISMATCH_PCT_THRESHOLD",
            15.0,
        ):
            exporter = ConsoleOSLMismatchExporter(
                create_exporter_config(
                    self._create_profile_results(count=10, total_records=100),
                    mock_cfg,
                )
            )
            output = await self._get_export_output(exporter)
            assert "15%" in output  # custom threshold
            assert "AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD=15" in output

    async def test_high_mismatch_percentage(self, mock_cfg):
        """Test warning with high percentage of OSL mismatches."""
        with patch(
            "aiperf.exporters.console_osl_mismatch_exporter.Environment.METRICS.OSL_MISMATCH_PCT_THRESHOLD",
            20.0,
        ):
            exporter = ConsoleOSLMismatchExporter(
                create_exporter_config(
                    self._create_profile_results(count=80, total_records=100),
                    mock_cfg,
                )
            )
            output = await self._get_export_output(exporter)
            assert "80 of 100 requests" in output
            assert "(80.0%)" in output

    async def test_no_mismatch_metric_no_output(self, mock_cfg):
        """Test that no warning is displayed when mismatch metric is absent."""
        exporter = ConsoleOSLMismatchExporter(
            create_exporter_config(
                self._create_profile_results(
                    count=0, total_records=100, include_mismatch=False
                ),
                mock_cfg,
            )
        )
        output = await self._get_export_output(exporter)
        assert "Output Sequence Length Mismatch Warning" not in output

    async def test_formatting_with_large_numbers(self, mock_cfg):
        """Test that large numbers are formatted with commas."""
        with patch(
            "aiperf.exporters.console_osl_mismatch_exporter.Environment.METRICS.OSL_MISMATCH_PCT_THRESHOLD",
            20.0,
        ):
            exporter = ConsoleOSLMismatchExporter(
                create_exporter_config(
                    self._create_profile_results(count=2500, total_records=10000),
                    mock_cfg,
                )
            )
            output = await self._get_export_output(exporter)
            assert "2,500 of 10,000 requests" in output
            assert "(25.0%)" in output

    async def test_warning_content_is_cp1252_encodable(self, mock_cfg):
        """Regression: warning content must encode in Windows cp1252.

        When aiperf is launched as a subprocess with PIPE'd stdout on Windows,
        sys.stdout's encoding is cp1252 (not utf-8). A non-cp1252 char in this
        panel previously raised UnicodeEncodeError, which aborted
        SystemController._stop_system_controller and hung the parent until
        pytest's 450s timeout (U+2192 -> at line 107).

        Checks the warning text *content* only. Rich's panel border falls
        back to ASCII when rendering to a non-terminal stream (which is the
        production case for aiperf parent stdout under subprocess.PIPE),
        so the border is not what we control here.
        """
        with patch(
            "aiperf.exporters.console_osl_mismatch_exporter.Environment.METRICS.OSL_MISMATCH_PCT_THRESHOLD",
            20.0,
        ):
            exporter = ConsoleOSLMismatchExporter(
                create_exporter_config(
                    self._create_profile_results(count=25, total_records=100),
                    mock_cfg,
                )
            )
            warning_text = exporter._create_warning_text(
                mismatch_count=25,
                total_records=100,
                percentage=25.0,
                avg_diff=66.7,
            )
            # `strict` matches Windows' default behavior; `replace` would mask
            # the very failure mode this test exists to catch.
            warning_text.encode("cp1252", errors="strict")
