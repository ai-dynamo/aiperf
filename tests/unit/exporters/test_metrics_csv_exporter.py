# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.metrics_csv_exporter import MetricsCsvExporter
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


class _MockResults:
    def __init__(self, records_list):
        self._records_list = records_list
        self.start_ns = None
        self.end_ns = None

    @property
    def records(self):
        # MetricsCsvExporter expects a dict[str, MetricResult] *after conversion*
        # but we monkeypatch the converter to return a dict.
        # Before conversion, we return a list.
        return self._records_list

    @property
    def has_results(self):
        return bool(self._records_list)

    @property
    def was_cancelled(self):
        return False

    @property
    def error_summary(self):
        return []


@pytest.fixture
def mk_metric():
    def _mk(
        tag,
        header,
        unit,
        *,
        avg=None,
        min=None,
        max=None,
        p50=None,
        p90=None,
        p95=None,
        p99=None,
        std=None,
    ):
        return MetricResult(
            tag="time_to_first_token",
            header=header,
            unit=unit,
            avg=avg,
            min=min,
            max=max,
            p50=p50,
            p90=p90,
            p95=p95,
            p99=p99,
            std=std,
        )

    return _mk


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_metrics_csv_exporter_writes_two_sections_and_values(
    monkeypatch, mock_user_config, mk_metric
):
    """
    Verifies:
      - display-unit conversion is honored (we simulate the converter),
      - request-metrics section with STAT_KEYS appears first,
      - blank separator line exists iff both sections exist,
      - system-metrics section prints single values,
      - units included in 'Metric' column.
    """
    # - ttft: request-level metric with percentiles, already converted to ms
    # - input_tokens: system metric (count)
    converted = {
        "time_to_first_token": mk_metric(
            "time_to_first_token",
            "Time to First Token",
            "ms",
            avg=12.3456,
            min=10.0,
            max=15.0,
            p50=12.34,
            p90=14.9,
            p95=None,
            p99=15.0,
            std=1.2,
        ),
        "time_to_first_token_system": mk_metric(
            "time_to_first_token",
            'Input, Tokens "Total"',
            "ms",
            avg=1024.0,
        ),
    }

    # Before conversion the exporter sees a list (consistent with your other exporters)
    results = _MockResults(list(converted.values()))

    # Monkeypatch converter to return our dict above
    import aiperf.exporters.metrics_base_exporter as mbe

    monkeypatch.setattr(
        mbe, "convert_all_metrics_to_display_units", lambda records, reg: converted
    )

    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        mock_user_config.output.artifact_directory = outdir
        cfg = ExporterConfig(
            results=results,
            user_config=mock_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )

        exporter = MetricsCsvExporter(cfg)
        await exporter.export()

        expected = outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
        assert expected.exists()

        text = _read(expected)

        # Request section header contains common stat columns
        assert "Metric" in text
        for col in ("avg", "min", "max", "p50", "p90", "p99", "std"):
            assert col in text

        # Request row includes unit on header
        assert "Time to First Token (ms)" in text

        # Blank line separator before system section
        assert "\n\nMetric,Value" in text

        # Expected -> "Input, Tokens ""Total"" (ms)"
        assert re.search(r'"Input, Tokens ""Total"" \(ms\)",\s*1024(\.0+)?\b', text)


@pytest.mark.asyncio
async def test_metrics_csv_exporter_empty_records_creates_empty_file(
    monkeypatch, mock_user_config
):
    """
    With no records, exporter still creates the file but content is empty (no sections).
    """
    # No records pre-conversion
    results = _MockResults([])

    # Converter returns empty dict to the generator
    import aiperf.exporters.metrics_base_exporter as mbe

    monkeypatch.setattr(
        mbe, "convert_all_metrics_to_display_units", lambda records, reg: {}
    )

    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        mock_user_config.output.artifact_directory = outdir
        cfg = ExporterConfig(
            results=results,
            user_config=mock_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )

        exporter = MetricsCsvExporter(cfg)
        await exporter.export()

        expected = outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
        assert expected.exists()
        content = _read(expected)
        assert content.strip() == ""


@pytest.mark.asyncio
async def test_metrics_csv_exporter_deterministic_sort_order(
    monkeypatch, mock_user_config, mk_metric
):
    """
    Ensures metrics are sorted by tag deterministically within each section.
    """
    converted = {
        "zzz_latency": mk_metric("zzz_latency", "Z Latency", "ms", avg=3.0, p50=3.0),
        "aaa_latency": mk_metric("aaa_latency", "A Latency", "ms", avg=1.0, p50=1.0),
        "mmm_gpu_util": mk_metric("mmm_gpu_util", "GPU Util", "percent", avg=80.0),
    }
    results = _MockResults(list(converted.values()))

    import aiperf.exporters.metrics_base_exporter as mbe

    monkeypatch.setattr(
        mbe, "convert_all_metrics_to_display_units", lambda records, reg: converted
    )

    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        mock_user_config.output.artifact_directory = outdir
        cfg = ExporterConfig(
            results=results,
            user_config=mock_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )

        exporter = MetricsCsvExporter(cfg)
        await exporter.export()

        text = _read(outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE)

        # Request section should list aaa_latency then zzz_latency in order
        # Pull only the request rows region (before the blank line separator).
        request_part = text.split("\n\n")[0]
        # The first data row should be A Latency, then Z Latency
        rows = [
            r for r in request_part.splitlines() if r and not r.startswith("Metric")
        ]
        assert any("A Latency" in r for r in rows[:1])
        assert any("Z Latency" in r for r in rows[1:2])

        # System section present and contains GPU Util
        assert "Metric,Value" in text
        assert "GPU Util (percent),80.00" in text


@pytest.mark.asyncio
async def test_metrics_csv_exporter_unit_aware_number_formatting(
    monkeypatch, mock_user_config, mk_metric
):
    """
    Validates unit-aware formatting policy:
      - counts show as integers (no decimals),
      - ms show with reasonable decimals (not coerced to integers),
      - presence of percentiles does not affect formatting policy.
    """
    converted = {
        "input_seq_len": mk_metric(
            "input_seq_len", "Input Sequence Length", "tokens", avg=4096
        ),
        "req_latency": mk_metric(
            "req_latency", "Request Latency", "ms", avg=1.2345, p50=1.234, p90=1.9
        ),
    }
    results = _MockResults(list(converted.values()))

    import aiperf.exporters.metrics_base_exporter as mbe

    monkeypatch.setattr(
        mbe, "convert_all_metrics_to_display_units", lambda records, reg: converted
    )

    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        mock_user_config.output.artifact_directory = outdir
        cfg = ExporterConfig(
            results=results,
            user_config=mock_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )

        exporter = MetricsCsvExporter(cfg)
        await exporter.export()

        text = _read(outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE)

        # counts: integer
        assert re.search(r"Input Sequence Length \(tokens\),\s*4096\b", text)

        # ms floats preserve precision (2 decimal places)
        assert re.search(r"Request Latency \(ms\).*(1\.23)", text)


@pytest.mark.asyncio
async def test_metrics_csv_exporter_logs_and_raises_on_write_failure(
    monkeypatch, mock_user_config, mk_metric
):
    """
    On write failure, exporter.error should be called and the exception should propagate.
    """
    converted = {
        "req_latency": mk_metric(
            "req_latency", "Request Latency", "ms", avg=1.0, p50=1.0
        ),
    }
    results = _MockResults(list(converted.values()))

    import aiperf.exporters.metrics_base_exporter as mbe

    monkeypatch.setattr(
        mbe, "convert_all_metrics_to_display_units", lambda records, reg: converted
    )

    # Force aiofiles.open to throw
    import aiofiles

    class _Boom:
        async def __aenter__(self):
            raise OSError("disk full")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(aiofiles, "open", lambda *a, **k: _Boom())

    # Capture error log calls
    called = {"err": None}

    def _err(msg):
        called["err"] = msg

    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        mock_user_config.output.artifact_directory = outdir
        cfg = ExporterConfig(
            results=results,
            user_config=mock_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )

        exporter = MetricsCsvExporter(cfg)
        monkeypatch.setattr(exporter, "error", _err)

        with pytest.raises(OSError, match="disk full"):
            await exporter.export()

        assert called["err"] is not None
        assert "Failed to export" in called["err"]


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ""),
        (142357, "142357"),
        (0, "0"),
        (-7, "-7"),
        (123456.14159, "123456.14"),  # Preserves precision with 2 decimals
        (2.0, "2.00"),
        (-1.234, "-1.23"),
        ("string", "string"),
        (True, "True"),
        (False, "False"),
        (
            1234567.89,
            "1234567.89",
        ),  # No scientific notation, formatted with 2 decimal places
    ],
)
@pytest.mark.asyncio
async def test_format_number_various_types(mock_user_config, value, expected):
    """
    Test the `_format_number` method with various input types.

    This parameterized test verifies that the method correctly formats:
    - None as an empty string
    - Integers as strings without decimals
    - Floats with 2 decimal places (preserving precision)
    - Strings as themselves
    - Boolean values as their string representation
    """
    cfg = ExporterConfig(
        results=None,
        user_config=mock_user_config,
        service_config=ServiceConfig(),
        telemetry_results=None,
    )
    exporter = MetricsCsvExporter(cfg)
    assert exporter._format_number(value) == expected


class TestMetricsCsvExporterTelemetry:
    """Test CSV export with telemetry data."""

    @pytest.mark.asyncio
    async def test_csv_export_with_telemetry_data(
        self, mock_user_config, sample_telemetry_results
    ):
        """Test that CSV export includes telemetry data section."""
        from aiperf.common.models import ProfileResults

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            mock_user_config.output.artifact_directory = outdir

            results = ProfileResults(
                records=[
                    MetricResult(
                        tag="time_to_first_token",
                        header="Time to First Token",
                        unit="ms",
                        avg=120.5,
                    )
                ],
                start_ns=0,
                end_ns=0,
                completed=0,
            )

            cfg = ExporterConfig(
                results=results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=sample_telemetry_results,
            )

            exporter = MetricsCsvExporter(cfg)
            await exporter.export()

            csv_file = outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
            assert csv_file.exists()

            content = csv_file.read_text()
            # Check for telemetry section with structured table format
            assert "Endpoint" in content
            assert "GPU_Index" in content
            assert "GPU Power Usage (W)" in content or "GPU Power Usage" in content
            assert "GPU Utilization (%)" in content or "GPU Utilization" in content

    @pytest.mark.asyncio
    async def test_csv_export_without_telemetry_data(self, mock_user_config):
        """Test that CSV export works when telemetry_results is None."""
        from aiperf.common.models import ProfileResults

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            mock_user_config.output.artifact_directory = outdir

            results = ProfileResults(
                records=[
                    MetricResult(
                        tag="time_to_first_token",
                        header="Time to First Token",
                        unit="ms",
                        avg=120.5,
                    )
                ],
                start_ns=0,
                end_ns=0,
                completed=0,
            )

            cfg = ExporterConfig(
                results=results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = MetricsCsvExporter(cfg)
            await exporter.export()

            csv_file = outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
            assert csv_file.exists()

            content = csv_file.read_text()
            # Should not have telemetry section (check for telemetry-specific columns)
            assert "GPU_Index" not in content
            assert "GPU_UUID" not in content

    @pytest.mark.asyncio
    async def test_csv_export_telemetry_multi_gpu(
        self, mock_user_config, sample_telemetry_results
    ):
        """Test that CSV export includes data for multiple GPUs."""
        from aiperf.common.models import ProfileResults

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            mock_user_config.output.artifact_directory = outdir

            results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)

            cfg = ExporterConfig(
                results=results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=sample_telemetry_results,
            )

            exporter = MetricsCsvExporter(cfg)
            await exporter.export()

            csv_file = outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
            content = csv_file.read_text()

            # Check for both GPU models in the test data
            assert "H100" in content or "A100" in content
            # Check that GPU index column appears
            assert "GPU_Index" in content

    @pytest.mark.asyncio
    async def test_csv_export_telemetry_metric_row_exceptions(self, mock_user_config):
        """Test that metric row write handles missing metrics gracefully."""
        from datetime import datetime

        from aiperf.common.models import ProfileResults
        from aiperf.common.models.export_models import (
            EndpointData,
            GpuSummary,
            TelemetryExportData,
            TelemetrySummary,
        )

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            mock_user_config.output.artifact_directory = outdir

            # Create TelemetryExportData with GPU that has no metrics
            telemetry_results = TelemetryExportData(
                summary=TelemetrySummary(
                    endpoints_configured=["http://localhost:9400/metrics"],
                    endpoints_successful=["http://localhost:9400/metrics"],
                    start_time=datetime.fromtimestamp(0),
                    end_time=datetime.fromtimestamp(0),
                ),
                endpoints={
                    "localhost:9400": EndpointData(
                        gpus={
                            "gpu_0": GpuSummary(
                                gpu_index=0,
                                gpu_name="Test GPU",
                                gpu_uuid="GPU-123",
                                hostname="test-node",
                                metrics={},  # No metrics - should skip gracefully
                            ),
                        }
                    ),
                },
            )

            results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)

            cfg = ExporterConfig(
                results=results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = MetricsCsvExporter(cfg)
            # Should not raise exception despite missing metrics
            await exporter.export()

            csv_file = outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
            assert csv_file.exists()

    @pytest.mark.asyncio
    async def test_csv_gpu_summary_metrics_check(self, mock_user_config):
        """Test that GPU metrics are checked correctly in the new structure."""
        from aiperf.common.models.export_models import (
            GpuSummary,
            JsonMetricResult,
        )

        # GpuSummary with metrics
        gpu_summary_with_metric = GpuSummary(
            gpu_index=0,
            gpu_name="Test GPU",
            gpu_uuid="GPU-123",
            hostname="test-node",
            metrics={
                "gpu_power_usage": JsonMetricResult(
                    unit="W", avg=100.0, min=90.0, max=110.0
                ),
            },
        )

        # Metric check is now a simple dict lookup
        assert "gpu_power_usage" in gpu_summary_with_metric.metrics
        assert "invalid_metric" not in gpu_summary_with_metric.metrics

        # GpuSummary without metrics
        gpu_summary_without_metric = GpuSummary(
            gpu_index=1,
            gpu_name="Test GPU 2",
            gpu_uuid="GPU-456",
            hostname="test-node",
            metrics={},
        )

        assert "gpu_power_usage" not in gpu_summary_without_metric.metrics

    @pytest.mark.asyncio
    async def test_csv_export_telemetry_multi_endpoint(self, mock_user_config):
        """Test CSV export with multiple DCGM endpoints."""
        from datetime import datetime

        from aiperf.common.models import ProfileResults
        from aiperf.common.models.export_models import (
            EndpointData,
            GpuSummary,
            JsonMetricResult,
            TelemetryExportData,
            TelemetrySummary,
        )

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            mock_user_config.output.artifact_directory = outdir

            # Create TelemetryExportData for two endpoints
            telemetry_results = TelemetryExportData(
                summary=TelemetrySummary(
                    endpoints_configured=[
                        "http://node1:9400/metrics",
                        "http://node2:9400/metrics",
                    ],
                    endpoints_successful=[
                        "http://node1:9400/metrics",
                        "http://node2:9400/metrics",
                    ],
                    start_time=datetime.fromtimestamp(0),
                    end_time=datetime.fromtimestamp(0),
                ),
                endpoints={
                    "node1:9400": EndpointData(
                        gpus={
                            "gpu_0": GpuSummary(
                                gpu_index=0,
                                gpu_name="GPU Model 1",
                                gpu_uuid="GPU-111",
                                hostname="node1",
                                metrics={
                                    "gpu_power_usage": JsonMetricResult(
                                        unit="W",
                                        avg=105.0,
                                        min=100.0,
                                        max=110.0,
                                        std=5.0,
                                    ),
                                },
                            ),
                        }
                    ),
                    "node2:9400": EndpointData(
                        gpus={
                            "gpu_0": GpuSummary(
                                gpu_index=0,
                                gpu_name="GPU Model 2",
                                gpu_uuid="GPU-222",
                                hostname="node2",
                                metrics={
                                    "gpu_power_usage": JsonMetricResult(
                                        unit="W",
                                        avg=205.0,
                                        min=200.0,
                                        max=210.0,
                                        std=5.0,
                                    ),
                                },
                            ),
                        }
                    ),
                },
            )

            results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)

            cfg = ExporterConfig(
                results=results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = MetricsCsvExporter(cfg)
            await exporter.export()

            csv_file = outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
            content = csv_file.read_text()

            # Check for both endpoints
            assert "node1:9400" in content
            assert "node2:9400" in content
            # Check for both GPU models
            assert "GPU Model 1" in content
            assert "GPU Model 2" in content

    @pytest.mark.asyncio
    async def test_csv_export_telemetry_empty_metrics(self, mock_user_config):
        """Test CSV export when GPU has no metric data."""
        from datetime import datetime

        from aiperf.common.models import ProfileResults
        from aiperf.common.models.export_models import (
            EndpointData,
            GpuSummary,
            TelemetryExportData,
            TelemetrySummary,
        )

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            mock_user_config.output.artifact_directory = outdir

            # Create TelemetryExportData with GPU that has no metrics
            telemetry_results = TelemetryExportData(
                summary=TelemetrySummary(
                    endpoints_configured=["http://localhost:9400/metrics"],
                    endpoints_successful=["http://localhost:9400/metrics"],
                    start_time=datetime.fromtimestamp(0),
                    end_time=datetime.fromtimestamp(0),
                ),
                endpoints={
                    "localhost:9400": EndpointData(
                        gpus={
                            "gpu_0": GpuSummary(
                                gpu_index=0,
                                gpu_name="Empty GPU",
                                gpu_uuid="GPU-EMPTY",
                                hostname="test-node",
                                metrics={},  # No metrics
                            ),
                        }
                    ),
                },
            )

            results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)

            cfg = ExporterConfig(
                results=results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = MetricsCsvExporter(cfg)
            await exporter.export()

            csv_file = outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
            content = csv_file.read_text()

            # Should still have telemetry table header columns
            assert "Endpoint" in content
            assert "GPU_Index" in content
            # But no metric data rows (GPU name should not appear since no metrics)
            assert "Empty GPU" not in content

    @pytest.mark.asyncio
    async def test_csv_format_number_small_values(self, mock_user_config):
        """Test _format_number with very small values."""
        from aiperf.common.models import ProfileResults

        results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)
        cfg = ExporterConfig(
            results=results,
            user_config=mock_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )

        exporter = MetricsCsvExporter(cfg)

        # Test very small value
        result = exporter._format_number(0.00123)
        assert result == "0.00"

        # Test zero
        result = exporter._format_number(0.0)
        assert result == "0.00"

    @pytest.mark.asyncio
    async def test_csv_format_number_decimal_type(self, mock_user_config):
        """Test _format_number with Decimal type."""
        from decimal import Decimal

        from aiperf.common.models import ProfileResults

        results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)
        cfg = ExporterConfig(
            results=results,
            user_config=mock_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )

        exporter = MetricsCsvExporter(cfg)

        # Test Decimal type
        result = exporter._format_number(Decimal("123.456"))
        assert result == "123.46"


class TestOptionalTelemetryHeaders:
    """Tests for optional hostname, namespace, and pod_name columns in telemetry CSV export."""

    @staticmethod
    def _make_telemetry(
        gpus: list[tuple[str | None, str | None, str | None]],
    ):
        """Create TelemetryExportData with specified optional fields per GPU.

        Args:
            gpus: List of (hostname, namespace, pod_name) tuples, one per GPU
        """
        from datetime import datetime

        from aiperf.common.models.export_models import (
            EndpointData,
            GpuSummary,
            JsonMetricResult,
            TelemetryExportData,
            TelemetrySummary,
        )

        gpu_dict = {
            f"gpu_{i}": GpuSummary(
                gpu_index=i,
                gpu_name="NVIDIA H100",
                gpu_uuid=f"GPU-{i:05d}",
                hostname=hostname,
                namespace=namespace,
                pod_name=pod_name,
                metrics={
                    "gpu_power_usage": JsonMetricResult(
                        unit="W", avg=300.0, min=280.0, max=320.0
                    )
                },
            )
            for i, (hostname, namespace, pod_name) in enumerate(gpus)
        }

        return TelemetryExportData(
            summary=TelemetrySummary(
                endpoints_configured=["http://node1:9400/metrics"],
                endpoints_successful=["http://node1:9400/metrics"],
                start_time=datetime.fromtimestamp(0),
                end_time=datetime.fromtimestamp(0),
            ),
            endpoints={"node1:9400": EndpointData(gpus=gpu_dict)},
        )

    def _make_exporter(self, mock_user_config, telemetry):
        """Create exporter with given telemetry data."""
        from aiperf.common.models import ProfileResults

        results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)
        cfg = ExporterConfig(
            results=results,
            user_config=mock_user_config,
            service_config=ServiceConfig(),
            telemetry_results=telemetry,
        )
        return MetricsCsvExporter(cfg)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "gpus,expected_headers,expected_values,unexpected_headers",
        [
            pytest.param(
                [("host1", "ns1", "pod1")],
                ["Hostname", "Namespace", "Pod Name", "host1", "ns1", "pod1"],
                [],
                [],
                id="all_fields_present",
            ),
            pytest.param(
                [("standalone", None, None)],
                ["Hostname", "standalone"],
                [],
                ["Namespace", "Pod Name"],
                id="hostname_only",
            ),
            pytest.param(
                [(None, None, None)],
                ["Endpoint", "GPU_Index"],
                [],
                ["Hostname", "Namespace", "Pod Name"],
                id="no_optional_fields",
            ),
            pytest.param(
                [("node1", None, None), ("node1", "default", "worker-pod")],
                ["Hostname", "Namespace", "Pod Name", "node1", "default", "worker-pod"],
                [],
                [],
                id="mixed_fields_across_gpus",
            ),
        ],
    )
    async def test_csv_optional_headers(
        self,
        mock_user_config,
        gpus,
        expected_headers,
        expected_values,
        unexpected_headers,
    ):
        """Test CSV includes/excludes optional headers based on field presence."""
        telemetry = self._make_telemetry(gpus)

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            mock_user_config.output.artifact_directory = outdir

            exporter = self._make_exporter(mock_user_config, telemetry)
            await exporter.export()

            content = (
                outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
            ).read_text()

            for expected in expected_headers:
                assert expected in content, f"Expected '{expected}' in CSV"
            for unexpected in unexpected_headers:
                assert unexpected not in content, f"'{unexpected}' should not be in CSV"

    @pytest.mark.parametrize(
        "gpus,input_headers,expected_headers,expected_fields",
        [
            pytest.param(
                [("h", "n", "p")],
                ("Hostname", "Namespace", "Pod Name"),
                ["Hostname", "Namespace", "Pod Name"],
                ["hostname", "namespace", "pod_name"],
                id="all_present",
            ),
            pytest.param(
                [("h", None, None)],
                ("Hostname", "Namespace", "Pod Name"),
                ["Hostname"],
                ["hostname"],
                id="hostname_only",
            ),
            pytest.param(
                [(None, None, None)],
                ("Hostname", "Namespace", "Pod Name"),
                [],
                [],
                id="all_none",
            ),
            pytest.param(
                [("h", "n", "p")],
                ("Pod Name", "Hostname", "Namespace"),
                ["Pod Name", "Hostname", "Namespace"],
                ["pod_name", "hostname", "namespace"],
                id="preserves_input_order",
            ),
        ],
    )
    def test_get_optional_headers_and_fields(
        self, mock_user_config, gpus, input_headers, expected_headers, expected_fields
    ):
        """Test _get_optional_headers_and_fields returns correct headers and field mappings."""
        telemetry = self._make_telemetry(gpus)
        exporter = self._make_exporter(mock_user_config, telemetry)

        headers, fields = exporter._get_optional_headers_and_fields(*input_headers)

        assert headers == expected_headers
        assert fields == expected_fields


def test_metrics_csv_exporter_inherits_from_base(mock_user_config):
    """Verify MetricsCsvExporter inherits from MetricsBaseExporter."""
    from aiperf.common.models import ProfileResults

    results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)
    cfg = ExporterConfig(
        results=results,
        user_config=mock_user_config,
        service_config=ServiceConfig(),
        telemetry_results=None,
    )

    exporter = MetricsCsvExporter(cfg)

    from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter

    assert isinstance(exporter, MetricsBaseExporter)


@pytest.mark.asyncio
async def test_metrics_csv_exporter_uses_base_export(mock_user_config):
    """Verify uses base class export() method."""
    from unittest.mock import AsyncMock

    from aiperf.common.models import ProfileResults

    results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)
    cfg = ExporterConfig(
        results=results,
        user_config=mock_user_config,
        service_config=ServiceConfig(),
        telemetry_results=None,
    )

    exporter = MetricsCsvExporter(cfg)

    # Mock the base class export method
    from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter

    mock_export = AsyncMock()

    with patch.object(MetricsBaseExporter, "export", mock_export):
        await exporter.export()

        # Verify base export was called
        mock_export.assert_called_once()


def test_metrics_csv_exporter_generate_content_uses_instance_data_members(
    mock_user_config,
):
    """Verify _generate_content() uses instance data members."""
    from aiperf.common.models import ProfileResults

    # Create mock records
    mock_records = [
        MetricResult(
            tag="time_to_first_token",
            header="Time to First Token",
            unit="ms",
            avg=45.2,
        )
    ]

    results = ProfileResults(records=mock_records, start_ns=0, end_ns=0, completed=0)
    cfg = ExporterConfig(
        results=results,
        user_config=mock_user_config,
        service_config=ServiceConfig(),
        telemetry_results=None,
    )

    exporter = MetricsCsvExporter(cfg)

    # Mock conversion
    import aiperf.exporters.metrics_base_exporter as mbe

    def mock_convert(metrics, reg):
        return {m.tag: m for m in metrics}

    with (
        patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
        patch.object(exporter, "_should_export", return_value=True),
    ):
        content = exporter._generate_content()

    # Should contain data from instance members
    assert "Time to First Token" in content


def test_metrics_csv_exporter_generate_content_uses_telemetry_results_from_instance(
    mock_user_config, sample_telemetry_results
):
    """Verify _generate_content() uses self._telemetry_results."""
    from aiperf.common.models import ProfileResults

    results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)
    cfg = ExporterConfig(
        results=results,
        user_config=mock_user_config,
        service_config=ServiceConfig(),
        telemetry_results=sample_telemetry_results,
    )

    exporter = MetricsCsvExporter(cfg)

    import aiperf.exporters.metrics_base_exporter as mbe

    def mock_convert(metrics, reg):
        return {}

    with patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert):
        content = exporter._generate_content()

    # Should contain telemetry data
    assert "GPU_Index" in content or "Endpoint" in content


@pytest.mark.asyncio
async def test_metrics_csv_exporter_export_calls_generate_content_internally(
    mock_user_config,
):
    """Verify export() calls _generate_content() internally."""
    from aiperf.common.models import ProfileResults

    results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)
    cfg = ExporterConfig(
        results=results,
        user_config=mock_user_config,
        service_config=ServiceConfig(),
        telemetry_results=None,
    )

    exporter = MetricsCsvExporter(cfg)

    test_csv_content = "Metric,Value\nTest,42"

    with patch.object(
        exporter, "_generate_content", return_value=test_csv_content
    ) as mock_generate:
        await exporter.export()

        # Verify _generate_content was called
        mock_generate.assert_called_once()

        # Verify file contains the returned content
        with open(exporter._file_path) as f:
            actual_content = f.read()

        assert actual_content == test_csv_content
