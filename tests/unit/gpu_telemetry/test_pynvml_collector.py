# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PyNVMLTelemetryCollector.

Tests use mocked pynvml module to verify collector behavior without requiring
actual GPU hardware.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.models import TelemetryRecord
from aiperf.gpu_telemetry.constants import PYNVML_SOURCE_IDENTIFIER
from aiperf.gpu_telemetry.pynvml_collector import ScalingFactors

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pynvml():
    """Create a mock pynvml module with typical NVML responses."""
    mock_module = MagicMock()

    # NVML error class and constants
    mock_module.NVMLError = Exception
    mock_module.NVML_TEMPERATURE_GPU = 0
    mock_module.NVML_PERF_POLICY_POWER = 0

    # Mock device count
    mock_module.nvmlDeviceGetCount.return_value = 2

    # Create mock device handles
    mock_handles = [MagicMock(), MagicMock()]

    def get_handle_by_index(idx):
        return mock_handles[idx]

    mock_module.nvmlDeviceGetHandleByIndex.side_effect = get_handle_by_index

    # GPU data with device-dependent return values
    mock_module.nvmlDeviceGetUUID.side_effect = lambda h: (
        "GPU-abc123" if h == mock_handles[0] else "GPU-def456"
    )
    mock_module.nvmlDeviceGetName.side_effect = lambda h: (
        "NVIDIA GeForce RTX 4090" if h == mock_handles[0] else "NVIDIA GeForce RTX 4080"
    )

    # PCI info
    pci_info_0 = SimpleNamespace(busId="00000000:01:00.0")
    pci_info_1 = SimpleNamespace(busId="00000000:02:00.0")
    mock_module.nvmlDeviceGetPciInfo.side_effect = lambda h: (
        pci_info_0 if h == mock_handles[0] else pci_info_1
    )

    # Power usage (milliwatts): 350W, 280W
    mock_module.nvmlDeviceGetPowerUsage.side_effect = lambda h: (
        350000 if h == mock_handles[0] else 280000
    )

    # Energy consumption (millijoules): 1000J, 800J
    mock_module.nvmlDeviceGetTotalEnergyConsumption.side_effect = lambda h: (
        1000000000 if h == mock_handles[0] else 800000000
    )

    # Utilization rates (GPU and memory bandwidth)
    util_0 = SimpleNamespace(gpu=95, memory=45)
    util_1 = SimpleNamespace(gpu=75, memory=35)
    mock_module.nvmlDeviceGetUtilizationRates.side_effect = lambda h: (
        util_0 if h == mock_handles[0] else util_1
    )

    # Memory info (bytes): 20 GB, 16 GB
    mem_0 = SimpleNamespace(used=20 * 1024 * 1024 * 1024)
    mem_1 = SimpleNamespace(used=16 * 1024 * 1024 * 1024)
    mock_module.nvmlDeviceGetMemoryInfo.side_effect = lambda h: (
        mem_0 if h == mock_handles[0] else mem_1
    )

    # Temperature (Celsius)
    mock_module.nvmlDeviceGetTemperature.side_effect = lambda h, t: (
        72 if h == mock_handles[0] else 68
    )

    # Video decoder/encoder/JPEG utilization (percent, sampling_period)
    mock_module.nvmlDeviceGetDecoderUtilization.side_effect = lambda h: (
        (25, 1000) if h == mock_handles[0] else (15, 1000)
    )
    mock_module.nvmlDeviceGetEncoderUtilization.side_effect = lambda h: (
        (30, 1000) if h == mock_handles[0] else (20, 1000)
    )
    mock_module.nvmlDeviceGetJpgUtilization.side_effect = lambda h: (
        (10, 1000) if h == mock_handles[0] else (5, 1000)
    )

    # Process utilization info
    proc_util_0 = SimpleNamespace(smUtil=85, encUtil=28, decUtil=22, jpgUtil=8)
    proc_util_1 = SimpleNamespace(smUtil=65, encUtil=18, decUtil=12, jpgUtil=3)
    mock_module.nvmlDeviceGetProcessesUtilizationInfo.side_effect = lambda h, t: (
        [proc_util_0] if h == mock_handles[0] else [proc_util_1]
    )

    # Power violation status (nanoseconds): 5ms, 2ms
    violation_0 = SimpleNamespace(violationTime=5000000)
    violation_1 = SimpleNamespace(violationTime=2000000)
    mock_module.nvmlDeviceGetViolationStatus.side_effect = lambda h, p: (
        violation_0 if h == mock_handles[0] else violation_1
    )

    # GPM (GPU Performance Metrics) support - disabled by default
    gpm_support = SimpleNamespace(isSupportedDevice=False)
    mock_module.nvmlGpmQueryDeviceSupport.return_value = gpm_support
    mock_module.nvmlGpmSampleAlloc.return_value = MagicMock()
    mock_module.nvmlGpmSampleFree.return_value = None
    mock_module.nvmlGpmSampleGet.return_value = MagicMock()

    # GPM metrics get - for computing SM utilization
    mock_module.NVML_GPM_METRICS_GET_VERSION = 1
    mock_module.NVML_GPM_METRIC_SM_UTIL = 2
    mock_module.c_nvmlGpmMetricsGet_t = MagicMock

    return mock_module


@pytest.fixture
def patch_pynvml(mock_pynvml):
    """Patch pynvml module reference in the collector module for testing."""
    from aiperf.gpu_telemetry import pynvml_collector
    from aiperf.gpu_telemetry.pynvml_collector import PyNVMLTelemetryCollector

    # Patch the pynvml reference in the collector module's namespace
    with patch.object(pynvml_collector, "pynvml", mock_pynvml):
        yield mock_pynvml, PyNVMLTelemetryCollector


@pytest.fixture
def collector(patch_pynvml):
    """Create an uninitialized collector."""
    _, PyNVMLTelemetryCollector = patch_pynvml
    return PyNVMLTelemetryCollector()


@pytest.fixture
async def initialized_collector(patch_pynvml):
    """Create and initialize a collector, yielding it for tests, then stopping."""
    _, PyNVMLTelemetryCollector = patch_pynvml
    collector = PyNVMLTelemetryCollector()
    await collector.initialize()
    yield collector
    await collector.stop()


# ---------------------------------------------------------------------------
# Test Initialization
# ---------------------------------------------------------------------------


class TestPyNVMLTelemetryCollectorInitialization:
    """Test PyNVMLTelemetryCollector initialization."""

    def test_initialization_with_custom_values(self, patch_pynvml):
        """Test collector initializes with custom values."""
        _, PyNVMLTelemetryCollector = patch_pynvml

        collector = PyNVMLTelemetryCollector(
            collection_interval=0.5,
            collector_id="test_collector",
        )

        assert collector.id == "test_collector"
        assert collector.collection_interval == 0.5
        assert collector.endpoint_url == PYNVML_SOURCE_IDENTIFIER
        assert not collector.was_initialized
        assert not collector.was_started

    def test_initialization_default_values(self, patch_pynvml):
        """Test collector uses default values when not specified."""
        _, PyNVMLTelemetryCollector = patch_pynvml

        collector = PyNVMLTelemetryCollector()

        assert collector.id == "pynvml_collector"
        assert collector.collection_interval == 0.333
        assert collector._record_callback is None
        assert collector._error_callback is None


# ---------------------------------------------------------------------------
# Test Reachability
# ---------------------------------------------------------------------------


class TestPyNVMLReachability:
    """Test NVML reachability checks."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "device_count,init_error,expected",
        [
            param(2, None, True, id="success"),
            param(0, None, False, id="no_gpus"),
            param(2, Exception("fail"), False, id="nvml_error"),
        ],
    )
    async def test_is_url_reachable(
        self, patch_pynvml, device_count, init_error, expected
    ):
        """Test reachability under various conditions."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        mock_pynvml.nvmlDeviceGetCount.return_value = device_count
        if init_error:
            mock_pynvml.nvmlInit.side_effect = init_error

        collector = PyNVMLTelemetryCollector()
        result = await collector.is_url_reachable()

        assert result is expected


# ---------------------------------------------------------------------------
# Test Lifecycle
# ---------------------------------------------------------------------------


class TestPyNVMLLifecycle:
    """Test collector lifecycle management."""

    @pytest.mark.asyncio
    async def test_initialize_discovers_gpus(self, initialized_collector):
        """Test initialization discovers and catalogs GPUs."""
        assert initialized_collector._nvml_initialized
        assert len(initialized_collector._gpus) == 2

        # Verify GPU metadata
        assert initialized_collector._gpus[0].metadata.gpu_uuid == "GPU-abc123"
        assert (
            initialized_collector._gpus[0].metadata.gpu_model_name
            == "NVIDIA GeForce RTX 4090"
        )
        assert initialized_collector._gpus[1].metadata.gpu_uuid == "GPU-def456"
        assert (
            initialized_collector._gpus[1].metadata.gpu_model_name
            == "NVIDIA GeForce RTX 4080"
        )

    @pytest.mark.asyncio
    async def test_stop_shuts_down_nvml(self, patch_pynvml):
        """Test stop properly shuts down NVML."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()
        assert collector._nvml_initialized

        await collector.stop()

        assert not collector._nvml_initialized
        mock_pynvml.nvmlShutdown.assert_called()

    @pytest.mark.asyncio
    async def test_stop_before_init_safe(self, collector):
        """Test stopping before initialization doesn't cause issues."""
        await collector.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_stop_clears_device_handles(self, patch_pynvml):
        """Test stop clears device handles and metadata."""
        _, PyNVMLTelemetryCollector = patch_pynvml

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()
        assert len(collector._gpus) == 2

        await collector.stop()

        assert collector._gpus == []

    @pytest.mark.asyncio
    async def test_init_failure_nvml_init_raises(self, patch_pynvml):
        """Test initialization fails gracefully when nvmlInit raises."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        mock_pynvml.nvmlInit.side_effect = mock_pynvml.NVMLError("Driver not loaded")

        collector = PyNVMLTelemetryCollector()

        with pytest.raises(asyncio.CancelledError, match="Failed to initialize NVML"):
            await collector.initialize()

        assert not collector._nvml_initialized

    @pytest.mark.asyncio
    async def test_init_failure_device_count_raises(self, patch_pynvml):
        """Test initialization cleans up when nvmlDeviceGetCount fails."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        mock_pynvml.nvmlDeviceGetCount.side_effect = mock_pynvml.NVMLError(
            "Device enumeration failed"
        )

        collector = PyNVMLTelemetryCollector()

        with pytest.raises(
            asyncio.CancelledError, match="Failed to get GPU device count"
        ):
            await collector.initialize()

        # Should have cleaned up NVML
        assert not collector._nvml_initialized
        mock_pynvml.nvmlShutdown.assert_called()

    @pytest.mark.asyncio
    async def test_init_skips_failed_device_handles(self, patch_pynvml):
        """Test initialization continues when individual GPU handle fails."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        # First GPU fails, second succeeds
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [
            mock_pynvml.NVMLError("GPU 0 failed"),
            MagicMock(),
        ]

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        # Should have initialized with only the second GPU
        assert collector._nvml_initialized
        assert len(collector._gpus) == 1

        await collector.stop()


# ---------------------------------------------------------------------------
# Test Metrics Collection
# ---------------------------------------------------------------------------


class TestPyNVMLMetricsCollection:
    """Test GPU metrics collection."""

    @pytest.mark.asyncio
    async def test_collect_gpu_metrics(self, initialized_collector):
        """Test metrics collection returns correct TelemetryRecord objects."""
        records = initialized_collector._collect_gpu_metrics()

        assert len(records) == 2
        assert all(isinstance(r, TelemetryRecord) for r in records)

        # Verify both GPUs have expected metadata and metrics
        gpu0 = next(r for r in records if r.gpu_index == 0)
        gpu1 = next(r for r in records if r.gpu_index == 1)

        # GPU 0 verification
        assert gpu0.dcgm_url == PYNVML_SOURCE_IDENTIFIER
        assert gpu0.gpu_uuid == "GPU-abc123"
        assert gpu0.gpu_model_name == "NVIDIA GeForce RTX 4090"
        assert gpu0.telemetry_data.gpu_power_usage == pytest.approx(350.0, rel=0.01)
        assert gpu0.telemetry_data.gpu_utilization == 95.0
        assert gpu0.telemetry_data.mem_utilization == 45.0
        assert gpu0.telemetry_data.gpu_temperature == 72.0
        assert gpu0.telemetry_data.gpu_memory_used == pytest.approx(20.0, rel=0.1)
        assert gpu0.telemetry_data.encoder_utilization == 30.0
        assert gpu0.telemetry_data.decoder_utilization == 25.0
        assert gpu0.telemetry_data.jpg_utilization == 10.0
        assert gpu0.telemetry_data.sm_utilization == 85.0
        assert gpu0.telemetry_data.power_violation == 5000.0

        # GPU 1 verification
        assert gpu1.gpu_uuid == "GPU-def456"
        assert gpu1.telemetry_data.gpu_power_usage == pytest.approx(280.0, rel=0.01)
        assert gpu1.telemetry_data.gpu_utilization == 75.0
        assert gpu1.telemetry_data.mem_utilization == 35.0
        assert gpu1.telemetry_data.encoder_utilization == 20.0
        assert gpu1.telemetry_data.decoder_utilization == 15.0
        assert gpu1.telemetry_data.jpg_utilization == 5.0
        assert gpu1.telemetry_data.sm_utilization == 65.0
        assert gpu1.telemetry_data.power_violation == 2000.0

    @pytest.mark.asyncio
    async def test_collect_handles_nvml_errors(self, patch_pynvml):
        """Test collection continues when individual metrics fail."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = mock_pynvml.NVMLError(
            "Power not supported"
        )

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        records = collector._collect_gpu_metrics()

        # Should still get records with other metrics
        assert len(records) == 2
        for r in records:
            assert r.telemetry_data.gpu_power_usage is None
            assert r.telemetry_data.gpu_utilization is not None
            assert r.telemetry_data.gpu_temperature is not None

        await collector.stop()

    @pytest.mark.asyncio
    async def test_collect_returns_empty_when_not_initialized(self, collector):
        """Test collection returns empty list when NVML not initialized."""
        records = collector._collect_gpu_metrics()
        assert records == []


# ---------------------------------------------------------------------------
# Test Callbacks
# ---------------------------------------------------------------------------


class TestPyNVMLCallbacks:
    """Test callback functionality."""

    @pytest.mark.asyncio
    async def test_record_callback_called(self, patch_pynvml):
        """Test record callback is called with collected records."""
        _, PyNVMLTelemetryCollector = patch_pynvml

        mock_callback = AsyncMock()
        collector = PyNVMLTelemetryCollector(
            record_callback=mock_callback,
            collector_id="test_collector",
        )

        await collector.initialize()
        await collector._collect_and_process_metrics()
        await collector.stop()

        mock_callback.assert_called_once()
        records, collector_id = mock_callback.call_args[0]

        assert len(records) == 2
        assert collector_id == "test_collector"

    @pytest.mark.asyncio
    async def test_error_callback_on_exception(self, patch_pynvml):
        """Test error callback is called when collection fails."""
        _, PyNVMLTelemetryCollector = patch_pynvml

        mock_error_callback = AsyncMock()
        collector = PyNVMLTelemetryCollector(
            error_callback=mock_error_callback,
            collector_id="test_collector",
        )

        await collector.initialize()

        # Force an error by making the collect method raise
        collector._collect_gpu_metrics = MagicMock(
            side_effect=Exception("Collection failed")
        )

        await collector._collect_and_process_metrics()
        await collector.stop()

        mock_error_callback.assert_called_once()
        error = mock_error_callback.call_args[0][0]
        assert hasattr(error, "message")


# ---------------------------------------------------------------------------
# Test Scaling Factors
# ---------------------------------------------------------------------------


class TestPyNVMLScalingFactors:
    """Test unit scaling factors."""

    @pytest.mark.parametrize(
        "field,factor,raw_value,expected",
        [
            param("gpu_power_usage", 1e-3, 350000, 350.0, id="power_mW_to_W"),
            param("energy_consumption", 1e-9, 1e9, 1.0, id="energy_mJ_to_MJ"),
            param("gpu_memory_used", 1e-9, 20e9, 20.0, id="memory_bytes_to_GB"),
        ],
    )
    def test_scaling_factor(self, field, factor, raw_value, expected):
        """Test scaling factors convert units correctly."""
        assert getattr(ScalingFactors, field) == factor
        assert raw_value * getattr(ScalingFactors, field) == expected


# ---------------------------------------------------------------------------
# Test Edge Cases
# ---------------------------------------------------------------------------


class TestPyNVMLEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "api_method,return_value,metadata_field,expected",
        [
            param(
                "nvmlDeviceGetName",
                b"NVIDIA RTX 4090",
                "gpu_model_name",
                "NVIDIA RTX 4090",
                id="name",
            ),
            param(
                "nvmlDeviceGetPciInfo",
                SimpleNamespace(busId=b"00000000:01:00.0"),
                "pci_bus_id",
                "00000000:01:00.0",
                id="pci_bus_id",
            ),
        ],
    )
    async def test_handles_bytes_values(
        self, patch_pynvml, api_method, return_value, metadata_field, expected
    ):
        """Test handles API values returned as bytes."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        if api_method == "nvmlDeviceGetName":
            mock_pynvml.nvmlDeviceGetName.side_effect = lambda h: return_value
        else:
            mock_pynvml.nvmlDeviceGetPciInfo.return_value = return_value

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        assert getattr(collector._gpus[0].metadata, metadata_field) == expected

        await collector.stop()

    @pytest.mark.asyncio
    async def test_energy_consumption_collected(self, initialized_collector):
        """Test energy consumption metric is collected and scaled correctly."""
        records = initialized_collector._collect_gpu_metrics()

        gpu0 = next(r for r in records if r.gpu_index == 0)
        # 1000000000 mJ * 1e-9 = 1.0 MJ
        assert gpu0.telemetry_data.energy_consumption == pytest.approx(1.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_sm_utilization_sums_multiple_processes(self, patch_pynvml):
        """Test SM utilization sums across multiple processes on same GPU."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        # Multiple processes on GPU 0
        proc1 = SimpleNamespace(smUtil=40, encUtil=10, decUtil=5, jpgUtil=2)
        proc2 = SimpleNamespace(smUtil=35, encUtil=8, decUtil=3, jpgUtil=1)
        mock_pynvml.nvmlDeviceGetProcessesUtilizationInfo.side_effect = lambda h, t: (
            [proc1, proc2] if h == mock_pynvml.nvmlDeviceGetHandleByIndex(0) else []
        )

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        records = collector._collect_gpu_metrics()
        gpu0 = next(r for r in records if r.gpu_index == 0)

        # Should sum: 40 + 35 = 75
        assert gpu0.telemetry_data.sm_utilization == 75.0

        await collector.stop()

    @pytest.mark.asyncio
    async def test_empty_process_list_zero_sm_utilization(self, patch_pynvml):
        """Test SM utilization is 0.0 when no processes are running."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        # Clear side_effect and set return_value (side_effect takes precedence)
        mock_pynvml.nvmlDeviceGetProcessesUtilizationInfo.side_effect = None
        mock_pynvml.nvmlDeviceGetProcessesUtilizationInfo.return_value = []

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        records = collector._collect_gpu_metrics()

        for r in records:
            assert r.telemetry_data.sm_utilization == 0.0

        await collector.stop()

    @pytest.mark.asyncio
    async def test_sm_utilization_capped_at_100(self, patch_pynvml):
        """Test SM utilization is capped at 100% when sum exceeds it."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        # Multiple processes with high utilization that sum > 100%
        proc1 = SimpleNamespace(smUtil=60, encUtil=10, decUtil=5, jpgUtil=2)
        proc2 = SimpleNamespace(smUtil=55, encUtil=8, decUtil=3, jpgUtil=1)
        mock_pynvml.nvmlDeviceGetProcessesUtilizationInfo.side_effect = lambda h, t: (
            [proc1, proc2] if h == mock_pynvml.nvmlDeviceGetHandleByIndex(0) else []
        )

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        records = collector._collect_gpu_metrics()
        gpu0 = next(r for r in records if r.gpu_index == 0)

        # Sum would be 60 + 55 = 115, but should be capped at 100.0
        assert gpu0.telemetry_data.sm_utilization == 100.0

        await collector.stop()

    @pytest.mark.asyncio
    async def test_is_url_reachable_when_already_initialized(
        self, initialized_collector
    ):
        """Test reachability returns True when already initialized with GPUs."""
        result = await initialized_collector.is_url_reachable()
        assert result is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "api_method,fallback_field,fallback_pattern",
        [
            param("nvmlDeviceGetUUID", "gpu_uuid", "GPU-unknown-0", id="uuid"),
            param("nvmlDeviceGetName", "gpu_model_name", "Unknown GPU", id="name"),
            param("nvmlDeviceGetPciInfo", "pci_bus_id", None, id="pci"),
        ],
    )
    async def test_metadata_fallback_on_error(
        self, patch_pynvml, api_method, fallback_field, fallback_pattern
    ):
        """Test fallback values when metadata APIs fail during initialization."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        getattr(mock_pynvml, api_method).side_effect = mock_pynvml.NVMLError(
            "Not supported"
        )

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        assert getattr(collector._gpus[0].metadata, fallback_field) == fallback_pattern

        await collector.stop()


# ---------------------------------------------------------------------------
# Test GPM (GPU Performance Metrics)
# ---------------------------------------------------------------------------


class TestPyNVMLGPM:
    """Test GPM (GPU Performance Metrics) functionality for efficient SM utilization."""

    @pytest.mark.asyncio
    async def test_gpm_not_supported_uses_process_api(self, patch_pynvml):
        """Test fallback to process API when GPM is not supported."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        # GPM not supported (default in fixture)
        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        # Should not have GPM enabled
        assert all(gpu.gpm_samples is None for gpu in collector._gpus)

        # Should still collect SM utilization via process API
        records = collector._collect_gpu_metrics()
        assert all(r.telemetry_data.sm_utilization is not None for r in records)

        await collector.stop()

    @pytest.mark.asyncio
    async def test_gpm_supported_allocates_samples(self, patch_pynvml):
        """Test GPM sample allocation when device supports GPM."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        # Enable GPM support
        gpm_support = SimpleNamespace(isSupportedDevice=True)
        mock_pynvml.nvmlGpmQueryDeviceSupport.return_value = gpm_support

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        # Should have GPM enabled for both GPUs (gpm_samples not None means supported)
        assert all(gpu.gpm_samples is not None for gpu in collector._gpus)

        # Each GPU should have two sample buffers allocated + initial sample taken
        assert mock_pynvml.nvmlGpmSampleAlloc.call_count == 4  # 2 GPUs * 2 samples each
        assert mock_pynvml.nvmlGpmSampleGet.call_count == 2  # Initial sample per GPU

        await collector.stop()

    @pytest.mark.asyncio
    async def test_gpm_samples_freed_on_shutdown(self, patch_pynvml):
        """Test GPM samples are freed during shutdown."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        # Enable GPM support
        gpm_support = SimpleNamespace(isSupportedDevice=True)
        mock_pynvml.nvmlGpmQueryDeviceSupport.return_value = gpm_support

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()
        assert all(gpu.gpm_samples is not None for gpu in collector._gpus)

        await collector.stop()

        # GPU list should be cleared on shutdown
        assert collector._gpus == []
        # 4 samples freed (2 GPUs * 2 samples each)
        assert mock_pynvml.nvmlGpmSampleFree.call_count == 4

    @pytest.mark.asyncio
    async def test_gpm_first_collection_uses_gpm(self, patch_pynvml):
        """Test first collection uses GPM (initial sample taken during init)."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        # Enable GPM support
        gpm_support = SimpleNamespace(isSupportedDevice=True)
        mock_pynvml.nvmlGpmQueryDeviceSupport.return_value = gpm_support

        # Mock GPM metrics result
        def mock_gpm_metrics_get(metrics_get):
            metrics_get.metrics[0].value = 88.5  # SM utilization from GPM
            return metrics_get

        mock_pynvml.nvmlGpmMetricsGet.side_effect = mock_gpm_metrics_get

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        # Initial sample taken during init (one per GPU)
        assert mock_pynvml.nvmlGpmSampleGet.call_count == 2

        # First collection should use GPM directly (no fallback needed)
        records = collector._collect_gpu_metrics()

        # GPM metrics should have been queried
        assert mock_pynvml.nvmlGpmMetricsGet.called

        # SM utilization should come from GPM
        gpu0 = next(r for r in records if r.gpu_index == 0)
        assert gpu0.telemetry_data.sm_utilization == 88.5

        await collector.stop()

    @pytest.mark.asyncio
    async def test_gpm_query_support_failure_disables_gpm(self, patch_pynvml):
        """Test GPM is disabled when nvmlGpmQueryDeviceSupport fails."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        # GPM query fails
        mock_pynvml.nvmlGpmQueryDeviceSupport.side_effect = mock_pynvml.NVMLError(
            "GPM not available"
        )

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        # GPM should be disabled (no samples allocated)
        assert all(gpu.gpm_samples is None for gpu in collector._gpus)

        # Should still work via process API
        records = collector._collect_gpu_metrics()
        assert all(r.telemetry_data.sm_utilization is not None for r in records)

        await collector.stop()

    @pytest.mark.asyncio
    async def test_gpm_sample_alloc_failure_disables_gpm(self, patch_pynvml):
        """Test GPM is disabled when sample allocation fails."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        # GPM supported but allocation fails
        gpm_support = SimpleNamespace(isSupportedDevice=True)
        mock_pynvml.nvmlGpmQueryDeviceSupport.return_value = gpm_support
        mock_pynvml.nvmlGpmSampleAlloc.side_effect = mock_pynvml.NVMLError(
            "Allocation failed"
        )

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        # GPM should be disabled due to allocation failure (no samples)
        assert all(gpu.gpm_samples is None for gpu in collector._gpus)

        await collector.stop()

    @pytest.mark.asyncio
    async def test_gpm_metrics_get_failure_falls_back_to_process_api(
        self, patch_pynvml
    ):
        """Test fallback to process API when nvmlGpmMetricsGet fails."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        # Enable GPM support
        gpm_support = SimpleNamespace(isSupportedDevice=True)
        mock_pynvml.nvmlGpmQueryDeviceSupport.return_value = gpm_support

        # GPM metrics get fails
        mock_pynvml.nvmlGpmMetricsGet.side_effect = mock_pynvml.NVMLError(
            "Metrics query failed"
        )

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        # First collection - primes the sample buffer
        collector._collect_gpu_metrics()

        # Second collection - GPM fails, should fall back to process API
        records = collector._collect_gpu_metrics()

        # Should still get SM utilization from process API fallback
        assert all(r.telemetry_data.sm_utilization is not None for r in records)

        # Process API should have been called
        mock_pynvml.nvmlDeviceGetProcessesUtilizationInfo.assert_called()

        await collector.stop()

    @pytest.mark.asyncio
    async def test_gpm_mixed_support(self, patch_pynvml):
        """Test handling when only some GPUs support GPM."""
        mock_pynvml, PyNVMLTelemetryCollector = patch_pynvml

        mock_handles = [MagicMock(), MagicMock()]
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = lambda i: mock_handles[i]

        # GPU 0 supports GPM, GPU 1 does not
        def gpm_support_check(handle):
            if handle == mock_handles[0]:
                return SimpleNamespace(isSupportedDevice=True)
            raise mock_pynvml.NVMLError("Not supported")

        mock_pynvml.nvmlGpmQueryDeviceSupport.side_effect = gpm_support_check

        collector = PyNVMLTelemetryCollector()
        await collector.initialize()

        # GPU 0 should have GPM (samples allocated), GPU 1 should not
        assert collector._gpus[0].gpm_samples is not None
        assert collector._gpus[1].gpm_samples is None

        await collector.stop()
