# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import tempfile
from pathlib import Path

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.enums import EndpointType
from aiperf.common.models import MetricResult
from aiperf.exporters.csv_exporter import CsvExporter
from aiperf.exporters.exporter_config import ExporterConfig


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
        # CsvExporter expects a dict[str, MetricResult] *after conversion*
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
            tag="ttft",
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
async def test_csv_exporter_writes_two_sections_and_values(
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
        "ttft": mk_metric(
            "ttft",
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
        "ttft_system": mk_metric(
            "ttft",
            'Input, Tokens "Total"',
            "ms",
            avg=1024.0,
        ),
    }

    # Before conversion the exporter sees a list (consistent with your other exporters)
    results = _MockResults(list(converted.values()))

    # Monkeypatch converter to return our dict above
    import aiperf.exporters.csv_exporter as ce

    monkeypatch.setattr(
        ce, "convert_all_metrics_to_display_units", lambda records, reg: converted
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

        exporter = CsvExporter(cfg)
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
async def test_csv_exporter_empty_records_creates_empty_file(
    monkeypatch, mock_user_config
):
    """
    With no records, exporter still creates the file but content is empty (no sections).
    """
    # No records pre-conversion
    results = _MockResults([])

    # Converter returns empty dict to the generator
    import aiperf.exporters.csv_exporter as ce

    monkeypatch.setattr(
        ce, "convert_all_metrics_to_display_units", lambda records, reg: {}
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

        exporter = CsvExporter(cfg)
        await exporter.export()

        expected = outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
        assert expected.exists()
        content = _read(expected)
        assert content.strip() == ""


@pytest.mark.asyncio
async def test_csv_exporter_deterministic_sort_order(
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

    import aiperf.exporters.csv_exporter as ce

    monkeypatch.setattr(
        ce, "convert_all_metrics_to_display_units", lambda records, reg: converted
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

        exporter = CsvExporter(cfg)
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
        assert "GPU Util (percent),80" in text or "GPU Util (percent),80.0" in text


@pytest.mark.asyncio
async def test_csv_exporter_unit_aware_number_formatting(
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

    import aiperf.exporters.csv_exporter as ce

    monkeypatch.setattr(
        ce, "convert_all_metrics_to_display_units", lambda records, reg: converted
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

        exporter = CsvExporter(cfg)
        await exporter.export()

        text = _read(outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE)

        # counts: integer
        assert re.search(r"Input Sequence Length \(tokens\),\s*4096\b", text)

        # ms floats should not collapse to integer; allow 2 fixed integers
        assert re.search(r"Request Latency \(ms\).*(1\.23)", text)


@pytest.mark.asyncio
async def test_csv_exporter_logs_and_raises_on_write_failure(
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

    import aiperf.exporters.csv_exporter as ce

    monkeypatch.setattr(
        ce, "convert_all_metrics_to_display_units", lambda records, reg: converted
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

        exporter = CsvExporter(cfg)
        monkeypatch.setattr(exporter, "error", _err)

        with pytest.raises(OSError, match="disk full"):
            await exporter.export()

        assert called["err"] is not None
        assert "Failed to export CSV" in called["err"]


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ""),
        (142357, "142357"),
        (0, "0"),
        (-7, "-7"),
        (123456.14159, "123456.14"),  # Numbers < 1M use standard format
        (2.0, "2.00"),
        (-1.234, "-1.23"),
        ("string", "string"),
        (True, "True"),
        (False, "False"),
        (1234567.89, "1.23e+06"),  # Numbers >= 1M use scientific notation
    ],
)
@pytest.mark.asyncio
async def test_format_number_various_types(
    monkeypatch, mock_user_config, value, expected
):
    """
    Test the `_format_number` method of `DummyExporter` with various input types.

    This parameterized test verifies that the method correctly formats:
    - None as an empty string
    - Integers and floats as strings, with floats rounded to two decimal places
    - Strings as themselves
    - Boolean values as their string representation
    """
    cfg = ExporterConfig(
        results=None,
        user_config=mock_user_config,
        service_config=ServiceConfig(),
        telemetry_results=None,
    )
    exporter = CsvExporter(cfg)
    assert exporter._format_number(value) == expected


class TestCsvExporterTelemetry:
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
                        tag="ttft",
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

            exporter = CsvExporter(cfg)
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
                        tag="ttft",
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

            exporter = CsvExporter(cfg)
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

            exporter = CsvExporter(cfg)
            await exporter.export()

            csv_file = outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
            content = csv_file.read_text()

            # Check for both GPU models in the test data
            assert "H100" in content or "A100" in content
            # Check that GPU index column appears
            assert "GPU_Index" in content

    @pytest.mark.asyncio
    async def test_csv_export_telemetry_metric_row_exceptions(self, mock_user_config):
        """Test that metric row write handles exceptions gracefully."""
        from unittest.mock import Mock

        from aiperf.common.models import (
            GpuMetadata,
            GpuTelemetryData,
            ProfileResults,
            TelemetryHierarchy,
            TelemetryResults,
        )

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            mock_user_config.output.artifact_directory = outdir

            # Create GPU data that will fail on metric retrieval
            gpu_data = Mock(spec=GpuTelemetryData)
            gpu_data.metadata = GpuMetadata(
                gpu_index=0,
                model_name="Test GPU",
                gpu_uuid="GPU-123",
            )
            gpu_data.get_metric_result = Mock(
                side_effect=Exception("Metric not available")
            )

            hierarchy = TelemetryHierarchy()
            hierarchy.dcgm_endpoints = {
                "http://localhost:9400/metrics": {"GPU-123": gpu_data}
            }

            telemetry_results = TelemetryResults(
                telemetry_data=hierarchy,
                start_ns=0,
                end_ns=0,
                endpoints_tested=["http://localhost:9400/metrics"],
                endpoints_successful=["http://localhost:9400/metrics"],
            )

            results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)

            cfg = ExporterConfig(
                results=results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = CsvExporter(cfg)
            # Should not raise exception despite metric retrieval failures
            await exporter.export()

            csv_file = outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
            assert csv_file.exists()

    @pytest.mark.asyncio
    async def test_csv_gpu_has_metric(self, mock_user_config):
        """Test _gpu_has_metric detection method."""
        from unittest.mock import Mock

        from aiperf.common.models import (
            GpuMetadata,
            GpuTelemetryData,
            MetricResult,
            ProfileResults,
        )

        results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)

        cfg = ExporterConfig(
            results=results,
            user_config=mock_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )

        exporter = CsvExporter(cfg)

        # GPU data with working metric
        gpu_data_with_metric = Mock(spec=GpuTelemetryData)
        gpu_data_with_metric.metadata = GpuMetadata(
            gpu_index=0, model_name="Test GPU", gpu_uuid="GPU-123"
        )
        gpu_data_with_metric.get_metric_result = Mock(
            return_value=MetricResult(
                tag="gpu_power_usage", header="Power", unit="W", avg=100.0
            )
        )

        assert exporter._gpu_has_metric(gpu_data_with_metric, "gpu_power_usage") is True

        # GPU data without metric
        gpu_data_without_metric = Mock(spec=GpuTelemetryData)
        gpu_data_without_metric.metadata = GpuMetadata(
            gpu_index=1, model_name="Test GPU 2", gpu_uuid="GPU-456"
        )
        gpu_data_without_metric.get_metric_result = Mock(
            side_effect=Exception("Not available")
        )

        assert (
            exporter._gpu_has_metric(gpu_data_without_metric, "invalid_metric") is False
        )

    @pytest.mark.asyncio
    async def test_csv_export_telemetry_multi_endpoint(self, mock_user_config):
        """Test CSV export with multiple DCGM endpoints."""
        from aiperf.common.models import (
            GpuMetadata,
            GpuTelemetryData,
            ProfileResults,
            TelemetryHierarchy,
            TelemetryResults,
        )

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            mock_user_config.output.artifact_directory = outdir

            # Create telemetry data for two endpoints
            hierarchy = TelemetryHierarchy()

            # Endpoint 1
            gpu1_data = GpuTelemetryData(
                metadata=GpuMetadata(
                    gpu_index=0,
                    model_name="GPU Model 1",
                    gpu_uuid="GPU-111",
                )
            )
            gpu1_data.time_series.append_snapshot(
                {"gpu_power_usage": 100.0}, timestamp_ns=1000000000
            )
            gpu1_data.time_series.append_snapshot(
                {"gpu_power_usage": 110.0}, timestamp_ns=2000000000
            )
            gpu1_data.time_series.append_snapshot(
                {"gpu_power_usage": 105.0}, timestamp_ns=3000000000
            )

            # Endpoint 2
            gpu2_data = GpuTelemetryData(
                metadata=GpuMetadata(
                    gpu_index=0,
                    model_name="GPU Model 2",
                    gpu_uuid="GPU-222",
                )
            )
            gpu2_data.time_series.append_snapshot(
                {"gpu_power_usage": 200.0}, timestamp_ns=1000000000
            )
            gpu2_data.time_series.append_snapshot(
                {"gpu_power_usage": 210.0}, timestamp_ns=2000000000
            )
            gpu2_data.time_series.append_snapshot(
                {"gpu_power_usage": 205.0}, timestamp_ns=3000000000
            )

            hierarchy.dcgm_endpoints = {
                "http://node1:9400/metrics": {"GPU-111": gpu1_data},
                "http://node2:9400/metrics": {"GPU-222": gpu2_data},
            }

            telemetry_results = TelemetryResults(
                telemetry_data=hierarchy,
                start_ns=0,
                end_ns=0,
                endpoints_tested=[
                    "http://node1:9400/metrics",
                    "http://node2:9400/metrics",
                ],
                endpoints_successful=[
                    "http://node1:9400/metrics",
                    "http://node2:9400/metrics",
                ],
            )

            results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)

            cfg = ExporterConfig(
                results=results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = CsvExporter(cfg)
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
        from aiperf.common.models import (
            GpuMetadata,
            GpuTelemetryData,
            ProfileResults,
            TelemetryHierarchy,
            TelemetryResults,
        )

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            mock_user_config.output.artifact_directory = outdir

            # Create GPU data with no metrics
            hierarchy = TelemetryHierarchy()
            gpu_data = GpuTelemetryData(
                metadata=GpuMetadata(
                    gpu_index=0,
                    model_name="Empty GPU",
                    gpu_uuid="GPU-EMPTY",
                )
            )
            # Don't add any metrics

            hierarchy.dcgm_endpoints = {
                "http://localhost:9400/metrics": {"GPU-EMPTY": gpu_data}
            }

            telemetry_results = TelemetryResults(
                telemetry_data=hierarchy,
                start_ns=0,
                end_ns=0,
                endpoints_tested=["http://localhost:9400/metrics"],
                endpoints_successful=["http://localhost:9400/metrics"],
            )

            results = ProfileResults(records=[], start_ns=0, end_ns=0, completed=0)

            cfg = ExporterConfig(
                results=results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = CsvExporter(cfg)
            await exporter.export()

            csv_file = outdir / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
            content = csv_file.read_text()

            # Should still have telemetry table header columns
            assert "Endpoint" in content
            assert "GPU_Index" in content
            # But no metric data rows (GPU name should not appear since no metrics)
            assert "Empty GPU" not in content
