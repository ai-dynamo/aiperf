# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock, patch

import pytest

from aiperf.common.enums import EnergyMetricUnit, GenericMetricUnit, PowerMetricUnit
from aiperf.common.environment import Environment
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import MetricResult
from aiperf.common.models.server_metrics_models import TimeRangeFilter
from aiperf.common.models.telemetry_models import (
    GpuMetadata,
    GpuTelemetryData,
    TelemetryHierarchy,
    TelemetryRecord,
)
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.gpu_telemetry.accumulator import (
    GPUTelemetryAccumulator,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.post_processors.conftest import make_telemetry_record


@pytest.fixture
def mock_cfg() -> CLIConfig:
    """Provide minimal CLIConfig for testing."""
    return CLIConfig(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        streaming=False,
    )


@pytest.fixture
def mock_service_config() -> CLIConfig:
    """Provide minimal CLIConfig for testing."""
    return CLIConfig()


@pytest.fixture
def mock_run(mock_cfg, mock_service_config):
    """Provide v2 BenchmarkRun built from mock_cfg + mock_service_config."""
    from tests.unit.conftest import make_run_from_cli

    return make_run_from_cli(mock_cfg)


@pytest.fixture
def mock_pub_client():
    """Provide mock pub client for testing."""
    mock = Mock()
    mock.publish = AsyncMock()
    return mock


@pytest.fixture
def sample_telemetry_record() -> TelemetryRecord:
    """Create a sample TelemetryRecord with typical values."""
    return make_telemetry_record(
        timestamp_ns=1000000000,
        dcgm_url="http://node1:9401/metrics",
        gpu_index=0,
        gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
        gpu_model_name="NVIDIA RTX 6000 Ada Generation",
        pci_bus_id="00000000:02:00.0",
        device="nvidia0",
        hostname="node1",
        gpu_power_usage=75.5,
        energy_consumption=1000.0,
        gpu_utilization=85.0,
        gpu_memory_used=15.26,
        gpu_temperature=70.0,
        xid_errors=0.0,
        power_violation=120.0,
    )


class TestGPUTelemetryAccumulator:
    """Test cases for GPUTelemetryAccumulator."""

    def test_initialization(
        self,
        mock_run,
        mock_pub_client,
    ) -> None:
        """Test processor initialization sets up hierarchy and metric units."""
        processor = GPUTelemetryAccumulator(
            run=mock_run,
            pub_client=mock_pub_client,
        )

        assert isinstance(processor._hierarchy, TelemetryHierarchy)

    @pytest.mark.asyncio
    async def test_process_telemetry_record(
        self,
        mock_run,
        mock_pub_client,
        sample_telemetry_record: TelemetryRecord,
    ) -> None:
        """Test processing a telemetry record adds it to the hierarchy."""
        processor = GPUTelemetryAccumulator(
            run=mock_run,
            pub_client=mock_pub_client,
        )

        await processor.process_telemetry_record(sample_telemetry_record)

        dcgm_url = sample_telemetry_record.dcgm_url
        gpu_uuid = sample_telemetry_record.gpu_uuid

        assert dcgm_url in processor._hierarchy.dcgm_endpoints
        assert gpu_uuid in processor._hierarchy.dcgm_endpoints[dcgm_url]

    @pytest.mark.asyncio
    async def test_get_hierarchy(
        self,
        mock_run,
        mock_pub_client,
        sample_telemetry_record: TelemetryRecord,
    ) -> None:
        """Test get_hierarchy returns accumulated data."""
        processor = GPUTelemetryAccumulator(
            run=mock_run,
            pub_client=mock_pub_client,
        )

        # Add some records
        await processor.process_telemetry_record(sample_telemetry_record)

        # Get hierarchy
        hierarchy = processor._hierarchy

        assert isinstance(hierarchy, TelemetryHierarchy)
        assert sample_telemetry_record.dcgm_url in hierarchy.dcgm_endpoints
        assert (
            sample_telemetry_record.gpu_uuid
            in hierarchy.dcgm_endpoints[sample_telemetry_record.dcgm_url]
        )

    @pytest.mark.asyncio
    async def test_summarize_with_valid_data(
        self,
        mock_run,
        mock_pub_client,
        sample_telemetry_record: TelemetryRecord,
    ) -> None:
        """Test summarize generates MetricResults for all metrics with data."""
        processor = GPUTelemetryAccumulator(
            run=mock_run,
            pub_client=mock_pub_client,
        )

        for i in range(5):
            record = make_telemetry_record(
                timestamp_ns=1000000000 + i * 1000000,
                dcgm_url=sample_telemetry_record.dcgm_url,
                gpu_index=sample_telemetry_record.gpu_index,
                gpu_uuid=sample_telemetry_record.gpu_uuid,
                gpu_model_name=sample_telemetry_record.gpu_model_name,
                gpu_power_usage=75.0 + i,
                energy_consumption=1000.0 + i * 10,
                gpu_utilization=80.0 + i,
                gpu_memory_used=15.0 + i * 0.1,
            )
            await processor.process_telemetry_record(record)

        results = await processor.summarize()

        # Should have results for all metrics that had data
        assert len(results) > 0
        assert all(isinstance(r, MetricResult) for r in results)

        # Check that metrics are properly tagged
        result_tags = [r.tag for r in results]
        assert any("gpu_power_usage" in tag for tag in result_tags)
        assert any("energy_consumption" in tag for tag in result_tags)

    @pytest.mark.asyncio
    async def test_summarize_handles_no_metric_value(
        self,
        mock_run,
        mock_pub_client,
    ) -> None:
        """Test summarize logs debug message when metric has no data and continues."""
        processor = GPUTelemetryAccumulator(
            run=mock_run,
            pub_client=mock_pub_client,
        )

        mock_metadata = GpuMetadata(
            gpu_index=0,
            gpu_uuid="GPU-12345678",
            gpu_model_name="Test GPU",
        )
        mock_telemetry_data = GpuTelemetryData(metadata=mock_metadata)
        processor._hierarchy.dcgm_endpoints = {
            "http://test:9401/metrics": {
                "GPU-12345678": mock_telemetry_data,
            }
        }

        with patch.object(processor, "debug") as mock_debug:
            results = await processor.summarize()

            # Should have logged debug messages for missing metrics
            assert mock_debug.call_count > 0
            debug_messages = [call[0][0] for call in mock_debug.call_args_list]
            assert any("No data available" in msg for msg in debug_messages)

            # Should return empty list when no data available
            assert results == []

    @pytest.mark.asyncio
    async def test_summarize_handles_unexpected_exception(
        self,
        mock_run,
        mock_pub_client,
    ) -> None:
        """Test summarize logs exception with stack trace on unexpected errors."""
        processor = GPUTelemetryAccumulator(
            run=mock_run,
            pub_client=mock_pub_client,
        )

        mock_metadata = GpuMetadata(
            gpu_index=0,
            gpu_uuid="GPU-87654321",
            gpu_model_name="Test GPU",
        )
        mock_telemetry_data = Mock(spec=GpuTelemetryData)
        mock_telemetry_data.metadata = mock_metadata
        mock_telemetry_data.get_metric_result.side_effect = RuntimeError(
            "Unexpected error"
        )

        processor._hierarchy.dcgm_endpoints = {
            "http://test:9401/metrics": {
                "GPU-87654321": mock_telemetry_data,
            }
        }

        with patch.object(processor, "exception") as mock_exception:
            results = await processor.summarize()

            # Should have logged exception with context
            assert mock_exception.call_count > 0
            exception_messages = [call[0][0] for call in mock_exception.call_args_list]
            assert any(
                "Unexpected error generating metric result" in msg
                for msg in exception_messages
            )
            assert any(
                "GPU-87654321" in msg for msg in exception_messages
            )  # First 12 chars

            # Should return empty list when all metrics fail
            assert results == []

    @pytest.mark.asyncio
    async def test_summarize_continues_after_errors(
        self,
        mock_run,
        mock_pub_client,
    ) -> None:
        """Test summarize continues processing other metrics after encountering errors."""
        processor = GPUTelemetryAccumulator(
            run=mock_run,
            pub_client=mock_pub_client,
        )

        mock_metadata = GpuMetadata(
            gpu_index=0,
            gpu_uuid="GPU-mixed-results",
            gpu_model_name="Test GPU",
        )

        mock_telemetry_data = Mock(spec=GpuTelemetryData)
        mock_telemetry_data.metadata = mock_metadata

        # First metric raises NoMetricValue, second succeeds, third raises unexpected error
        call_count = 0

        def side_effect_func(_metric_name, tag, header, unit):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise NoMetricValue("No data for first metric")
            elif call_count == 2:
                return MetricResult(
                    tag=tag, header=header, unit=unit, avg=50.0, count=10
                )
            else:
                raise ValueError("Unexpected error")

        mock_telemetry_data.get_metric_result.side_effect = side_effect_func

        processor._hierarchy.dcgm_endpoints = {
            "http://test:9401/metrics": {
                "GPU-mixed-results": mock_telemetry_data,
            }
        }

        with (
            patch.object(processor, "debug") as mock_debug,
            patch.object(processor, "exception") as mock_exception,
        ):
            results = await processor.summarize()

            # Should have logged both types of errors
            assert mock_debug.call_count > 0
            assert mock_exception.call_count > 0

            # Should have one successful result despite errors
            assert len(results) == 1
            assert results[0].avg == 50.0

    @pytest.mark.asyncio
    async def test_summarize_generates_correct_tags(
        self,
        mock_run,
        mock_pub_client,
        sample_telemetry_record: TelemetryRecord,
    ) -> None:
        """Test summarize generates properly formatted tags with DCGM URL and GPU info."""
        processor = GPUTelemetryAccumulator(
            run=mock_run,
            pub_client=mock_pub_client,
        )

        for i in range(3):
            record = make_telemetry_record(
                timestamp_ns=1000000000 + i * 1000000,
                gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
                gpu_model_name="NVIDIA RTX 6000",
                gpu_power_usage=75.0 + i,
            )
            await processor.process_telemetry_record(record)

        results = await processor.summarize()

        # Check tag format: metric_name_dcgm_TAG_gpuINDEX_UUID
        power_results = [r for r in results if "gpu_power_usage" in r.tag]
        assert len(power_results) > 0

        tag = power_results[0].tag
        assert "gpu_power_usage" in tag
        assert "dcgm_http" in tag  # URL gets sanitized
        assert "node1" in tag
        assert "gpu0" in tag
        assert "GPU-ef6ef310" in tag  # First 12 chars of UUID

    @pytest.mark.asyncio
    async def test_summarize_multiple_gpus(
        self,
        mock_run,
        mock_pub_client,
    ) -> None:
        """Test summarize handles multiple GPUs correctly."""
        processor = GPUTelemetryAccumulator(
            run=mock_run,
            pub_client=mock_pub_client,
        )

        for gpu_index in range(2):
            for i in range(3):
                record = make_telemetry_record(
                    timestamp_ns=1000000000 + i * 1000000,
                    gpu_index=gpu_index,
                    gpu_uuid=f"GPU-0000000{gpu_index}-0000-0000-0000-000000000000",
                    gpu_model_name="NVIDIA RTX 6000",
                    gpu_power_usage=75.0 + gpu_index * 10 + i,
                )
                await processor.process_telemetry_record(record)

        results = await processor.summarize()

        # Should have results for both GPUs
        gpu0_results = [r for r in results if "gpu0" in r.tag]
        gpu1_results = [r for r in results if "gpu1" in r.tag]

        assert len(gpu0_results) > 0
        assert len(gpu1_results) > 0


class TestComputeEfficiencyMetrics:
    """Test GPUTelemetryAccumulator.compute_efficiency_metrics."""

    @pytest.fixture
    def accumulator(
        self,
        mock_run,
        mock_pub_client,
    ) -> GPUTelemetryAccumulator:
        return GPUTelemetryAccumulator(
            run=mock_run,
            pub_client=mock_pub_client,
        )

    @pytest.fixture
    def time_filter(self) -> TimeRangeFilter:
        return TimeRangeFilter(start_ns=2_000_000_000, end_ns=5_000_000_000)

    def _make_gpu_mock(
        self,
        power_avg: float | None = None,
        energy_delta_mj: float | None = None,
    ) -> Mock:
        """Return a GpuTelemetryData mock with controllable metric results."""
        metadata = GpuMetadata(
            gpu_index=0,
            gpu_uuid="GPU-test-uuid-0000",
            gpu_model_name="Test GPU",
        )
        gpu = Mock(spec=GpuTelemetryData)
        gpu.metadata = metadata

        def get_metric_result(
            metric_name: str,
            tag: str,
            header: str,
            unit: str,
            time_filter: TimeRangeFilter | None = None,
            is_counter: bool = False,
        ) -> MetricResult:
            if metric_name == "gpu_power_usage":
                if power_avg is None:
                    raise NoMetricValue("No power data")
                return MetricResult(
                    tag=tag, header=header, unit=unit, avg=power_avg, count=3
                )
            if metric_name == "energy_consumption":
                if energy_delta_mj is None:
                    raise NoMetricValue("No energy data")
                return MetricResult(
                    tag=tag, header=header, unit=unit, avg=energy_delta_mj
                )
            raise NoMetricValue(f"No data for {metric_name}")

        gpu.get_metric_result.side_effect = get_metric_result
        return gpu

    def test_happy_path_all_metrics_present(
        self, accumulator: GPUTelemetryAccumulator, time_filter: TimeRangeFilter
    ) -> None:
        """power + energy + tokens + concurrency=1 (default) → four MetricResults."""
        gpu = self._make_gpu_mock(
            power_avg=200.0, energy_delta_mj=0.001
        )  # 0.001 MJ = 1000 J
        accumulator._hierarchy.dcgm_endpoints = {
            "http://node1:9401/metrics": {"GPU-test": gpu}
        }
        metric_results = [
            MetricResult(
                tag="total_output_tokens",
                header="Total Output Tokens",
                unit="tokens",
                avg=2000.0,
            )
        ]

        results = accumulator.compute_efficiency_metrics(metric_results, time_filter)

        tags = {r.tag for r in results}
        assert tags == {
            "total_gpu_power",
            "total_gpu_energy",
            "output_tokens_per_joule",
            "energy_per_user",
        }

        power = next(r for r in results if r.tag == "total_gpu_power")
        assert power.avg == pytest.approx(200.0)
        assert power.unit == str(PowerMetricUnit.WATT)
        assert power.header == "Total GPU Power (1 GPU)"

        energy = next(r for r in results if r.tag == "total_gpu_energy")
        assert energy.avg == pytest.approx(1000.0)  # 0.001 MJ → J
        assert energy.unit == str(EnergyMetricUnit.JOULE)
        assert energy.header == "Total GPU Energy (1 GPU)"

        tpj = next(r for r in results if r.tag == "output_tokens_per_joule")
        assert tpj.avg == pytest.approx(2.0)  # 2000 tokens / 1000 J
        assert tpj.unit == str(GenericMetricUnit.TOKENS_PER_JOULE)
        assert tpj.header == "Output Tokens per Joule (1 GPU)"

        epu = next(r for r in results if r.tag == "energy_per_user")
        assert epu.avg == pytest.approx(1000.0)  # 1000 J / 1 user (default)
        assert epu.unit == str(GenericMetricUnit.JOULES_PER_USER)
        assert epu.header == "Energy per User (1 GPU)"

    def test_energy_per_user_scales_with_concurrency(
        self, accumulator: GPUTelemetryAccumulator, time_filter: TimeRangeFilter
    ) -> None:
        """concurrency=N (positive int) → emit total_energy / N."""
        accumulator.run.cfg.get_profiling_phases()[0].concurrency = 8
        gpu = self._make_gpu_mock(power_avg=200.0, energy_delta_mj=0.001)  # 1000 J
        accumulator._hierarchy.dcgm_endpoints = {
            "http://node1:9401/metrics": {"GPU-test": gpu}
        }

        results = accumulator.compute_efficiency_metrics([], time_filter)

        epu = next(r for r in results if r.tag == "energy_per_user")
        assert epu.avg == pytest.approx(125.0)  # 1000 J / 8 users
        assert epu.unit == str(GenericMetricUnit.JOULES_PER_USER)
        assert epu.header == "Energy per User (1 GPU)"

    def test_energy_per_user_omitted_when_concurrency_none(
        self, accumulator: GPUTelemetryAccumulator, time_filter: TimeRangeFilter
    ) -> None:
        """concurrency=None (e.g. pure request-rate run) → energy_per_user omitted."""
        accumulator.run.cfg.get_profiling_phases()[0].concurrency = None
        gpu = self._make_gpu_mock(power_avg=200.0, energy_delta_mj=0.001)
        accumulator._hierarchy.dcgm_endpoints = {
            "http://node1:9401/metrics": {"GPU-test": gpu}
        }

        results = accumulator.compute_efficiency_metrics([], time_filter)

        tags = {r.tag for r in results}
        assert "energy_per_user" not in tags
        assert "total_gpu_energy" in tags  # sibling still emits

    def test_energy_per_user_omitted_when_no_energy_data(
        self, accumulator: GPUTelemetryAccumulator, time_filter: TimeRangeFilter
    ) -> None:
        """concurrency set but no GPU energy → energy_per_user omitted (no numerator)."""
        accumulator.run.cfg.get_profiling_phases()[0].concurrency = 8
        gpu = self._make_gpu_mock(power_avg=150.0, energy_delta_mj=None)
        accumulator._hierarchy.dcgm_endpoints = {
            "http://node1:9401/metrics": {"GPU-test": gpu}
        }

        results = accumulator.compute_efficiency_metrics([], time_filter)

        tags = {r.tag for r in results}
        assert "energy_per_user" not in tags
        assert "total_gpu_energy" not in tags

    def test_emitted_units_match_metric_class_units(
        self, accumulator: GPUTelemetryAccumulator, time_filter: TimeRangeFilter
    ) -> None:
        """Accumulator-emitted unit strings must equal str(MetricClass.unit) per tag.

        Locks the relationship rather than the literal — a rename of any of
        the unit enums would break this without needing per-test updates.
        """
        from aiperf.metrics.types.power_efficiency_metrics import (
            EnergyPerUserMetric,
            OutputTokensPerJouleMetric,
            TotalGpuEnergyMetric,
            TotalGpuPowerMetric,
        )

        gpu = self._make_gpu_mock(power_avg=200.0, energy_delta_mj=0.001)
        accumulator._hierarchy.dcgm_endpoints = {
            "http://node1:9401/metrics": {"GPU-test": gpu}
        }
        metric_results = [
            MetricResult(
                tag="total_output_tokens", header="h", unit="tokens", avg=2000.0
            )
        ]

        results = accumulator.compute_efficiency_metrics(metric_results, time_filter)
        by_tag = {r.tag: r for r in results}

        expected = {
            TotalGpuPowerMetric.tag: str(TotalGpuPowerMetric.unit),
            TotalGpuEnergyMetric.tag: str(TotalGpuEnergyMetric.unit),
            OutputTokensPerJouleMetric.tag: str(OutputTokensPerJouleMetric.unit),
            EnergyPerUserMetric.tag: str(EnergyPerUserMetric.unit),
        }
        for tag, expected_unit in expected.items():
            assert by_tag[tag].unit == expected_unit, (
                f"unit drift for {tag}: emitted={by_tag[tag].unit!r}, "
                f"metric class={expected_unit!r}"
            )

    def test_no_energy_data_omits_energy_and_tokens_per_joule(
        self, accumulator: GPUTelemetryAccumulator, time_filter: TimeRangeFilter
    ) -> None:
        """No energy data → only power metric returned; tokens/J absent."""
        gpu = self._make_gpu_mock(power_avg=150.0, energy_delta_mj=None)
        accumulator._hierarchy.dcgm_endpoints = {
            "http://node1:9401/metrics": {"GPU-test": gpu}
        }
        metric_results = [
            MetricResult(tag="total_output_tokens", header="h", unit="t", avg=1000.0)
        ]

        results = accumulator.compute_efficiency_metrics(metric_results, time_filter)

        tags = {r.tag for r in results}
        assert "total_gpu_power" in tags
        assert "total_gpu_energy" not in tags
        assert "output_tokens_per_joule" not in tags

    def test_no_gpu_data_returns_empty_list(
        self, accumulator: GPUTelemetryAccumulator, time_filter: TimeRangeFilter
    ) -> None:
        """No GPU data → empty list returned without error."""
        results = accumulator.compute_efficiency_metrics([], time_filter)
        assert results == []

    def test_missing_total_output_tokens_omits_tokens_per_joule(
        self, accumulator: GPUTelemetryAccumulator, time_filter: TimeRangeFilter
    ) -> None:
        """total_output_tokens absent from metric_results → tokens/J absent, no error."""
        gpu = self._make_gpu_mock(power_avg=200.0, energy_delta_mj=0.001)
        accumulator._hierarchy.dcgm_endpoints = {
            "http://node1:9401/metrics": {"GPU-test": gpu}
        }

        results = accumulator.compute_efficiency_metrics([], time_filter)

        tags = {r.tag for r in results}
        assert "total_gpu_power" in tags
        assert "total_gpu_energy" in tags
        assert "output_tokens_per_joule" not in tags

    def test_multiple_gpus_sums_power_and_energy(
        self, accumulator: GPUTelemetryAccumulator, time_filter: TimeRangeFilter
    ) -> None:
        """Multiple GPUs across endpoints → power and energy summed."""
        gpu0 = self._make_gpu_mock(power_avg=100.0, energy_delta_mj=0.0005)  # 500 J
        gpu1 = self._make_gpu_mock(power_avg=150.0, energy_delta_mj=0.0005)  # 500 J
        accumulator._hierarchy.dcgm_endpoints = {
            "http://node1:9401/metrics": {"GPU-0": gpu0, "GPU-1": gpu1}
        }
        metric_results = [
            MetricResult(tag="total_output_tokens", header="h", unit="t", avg=1000.0)
        ]

        results = accumulator.compute_efficiency_metrics(metric_results, time_filter)

        power = next(r for r in results if r.tag == "total_gpu_power")
        assert power.avg == pytest.approx(250.0)  # 100 + 150
        assert power.count is None
        assert power.header == "Total GPU Power (2 GPUs)"

        energy = next(r for r in results if r.tag == "total_gpu_energy")
        assert energy.avg == pytest.approx(1000.0)  # 500 + 500
        assert energy.count is None
        assert energy.header == "Total GPU Energy (2 GPUs)"

        tpj = next(r for r in results if r.tag == "output_tokens_per_joule")
        assert tpj.avg == pytest.approx(1.0)  # 1000 tokens / 1000 J
        assert tpj.header == "Output Tokens per Joule (2 GPUs)"

        epu = next(r for r in results if r.tag == "energy_per_user")
        assert epu.avg == pytest.approx(1000.0)  # 1000 J / 1 user (default)
        assert epu.header == "Energy per User (2 GPUs)"

    def test_header_reflects_partial_cohort_count(
        self, accumulator: GPUTelemetryAccumulator, time_filter: TimeRangeFilter
    ) -> None:
        """Header surfaces the *valid-data* GPU count, not the cohort total.

        Two GPUs are configured but only one reports energy; the total_gpu_energy
        header must read "(1 GPU)" so a "1 of 2" partial run is distinguishable
        from a "2 of 2" full run. MetricResult.count cannot carry this because
        to_json_result strips count to None for DERIVED metrics.
        """
        gpu_full = self._make_gpu_mock(power_avg=100.0, energy_delta_mj=0.001)
        gpu_power_only = self._make_gpu_mock(power_avg=150.0, energy_delta_mj=None)
        accumulator._hierarchy.dcgm_endpoints = {
            "http://node1:9401/metrics": {"GPU-0": gpu_full, "GPU-1": gpu_power_only}
        }
        metric_results = [
            MetricResult(tag="total_output_tokens", header="h", unit="t", avg=2000.0)
        ]

        results = accumulator.compute_efficiency_metrics(metric_results, time_filter)
        by_tag = {r.tag: r for r in results}

        assert by_tag["total_gpu_power"].header == "Total GPU Power (2 GPUs)"
        assert by_tag["total_gpu_energy"].header == "Total GPU Energy (1 GPU)"
        assert by_tag["output_tokens_per_joule"].header == (
            "Output Tokens per Joule (1 GPU)"
        )
        # energy_per_user inherits the energy-side count (its denominator).
        assert by_tag["energy_per_user"].header == "Energy per User (1 GPU)"

    def test_energy_filter_widens_end_ns_by_grace_while_power_filter_stays_bounded(
        self, accumulator: GPUTelemetryAccumulator, time_filter: TimeRangeFilter
    ) -> None:
        # Counter-based energy widens end_ns by FINAL_SCRAPE_GRACE_NS so the
        # trailing scrape (which lands after requests_end_ns on the
        # COLLECTION_INTERVAL cadence) is captured, but the window stays
        # bounded so cooldown, idle, or subsequent-phase samples don't leak
        # into the delta. Gauge-based power stays at the unwidened end_ns so
        # post-bench idle samples don't drag the average down.
        gpu = self._make_gpu_mock(power_avg=200.0, energy_delta_mj=0.001)
        accumulator._hierarchy.dcgm_endpoints = {
            "http://node1:9401/metrics": {"GPU-test": gpu}
        }
        metric_results = [
            MetricResult(
                tag="total_output_tokens",
                header="Total Output Tokens",
                unit="tokens",
                avg=2000.0,
            )
        ]

        accumulator.compute_efficiency_metrics(metric_results, time_filter)

        filters_by_metric: dict[str, TimeRangeFilter] = {
            call.args[0]: call.kwargs["time_filter"]
            for call in gpu.get_metric_result.call_args_list
        }

        power_filter = filters_by_metric["gpu_power_usage"]
        assert power_filter.start_ns == time_filter.start_ns
        assert power_filter.end_ns == time_filter.end_ns

        energy_filter = filters_by_metric["energy_consumption"]
        assert energy_filter.start_ns == time_filter.start_ns
        assert energy_filter.end_ns == (
            time_filter.end_ns + Environment.GPU.FINAL_SCRAPE_GRACE_NS
        )
        assert energy_filter.end_ns is not None, (
            "energy filter must remain bounded so a multi-phase run cannot leak "
            "cooldown or subsequent-phase samples into phase N's energy delta"
        )

    def test_repeated_calls_use_bounded_energy_window_per_phase(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        """Each `compute_efficiency_metrics` call must bound the energy filter at
        `phase.end_ns + grace` — never reach into a later phase's samples.

        Multi-phase regression guard: simulate WARMUP -> PROFILING by calling
        the method twice with non-overlapping windows. Both calls must produce
        bounded energy filters whose `end_ns` reflects the phase being closed,
        not "now" or the union of all stored samples.
        """
        gpu = self._make_gpu_mock(power_avg=100.0, energy_delta_mj=0.0005)
        accumulator._hierarchy.dcgm_endpoints = {
            "http://node1:9401/metrics": {"GPU-test": gpu}
        }
        metric_results = [
            MetricResult(tag="total_output_tokens", header="h", unit="t", avg=500.0)
        ]
        phase1 = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)
        phase2 = TimeRangeFilter(start_ns=3_000_000_000, end_ns=4_000_000_000)

        accumulator.compute_efficiency_metrics(metric_results, phase1)
        accumulator.compute_efficiency_metrics(metric_results, phase2)

        energy_filters = [
            call.kwargs["time_filter"]
            for call in gpu.get_metric_result.call_args_list
            if call.args[0] == "energy_consumption"
        ]
        assert len(energy_filters) == 2

        grace = Environment.GPU.FINAL_SCRAPE_GRACE_NS
        assert energy_filters[0].start_ns == phase1.start_ns
        assert energy_filters[0].end_ns == phase1.end_ns + grace
        assert energy_filters[1].start_ns == phase2.start_ns
        assert energy_filters[1].end_ns == phase2.end_ns + grace

        # Phase 1's bounded end must not extend into phase 2 — that's the leak
        # the bound prevents. Confirms grace is intentionally small.
        assert energy_filters[0].end_ns < phase2.start_ns, (
            f"phase 1 energy window ({energy_filters[0].end_ns}) extends past "
            f"phase 2 start ({phase2.start_ns}); the grace window is too large "
            f"for safe multi-phase use"
        )
