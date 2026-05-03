# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from aiperf.common.models.telemetry_models import TelemetryRecord
from aiperf.gpu_telemetry.dcgm_collector import DCGMTelemetryCollector


class TestDCGMTelemetryCollectorCore:
    """Test core DCGMTelemetryCollector functionality.

    This test class focuses exclusively on the data collection, parsing,
    and lifecycle management of the DCGMTelemetryCollector using the new async architecture.

    Key areas tested:
    - Initialization and configuration
    - Async lifecycle management
    - Prometheus metric parsing
    - Background collection tasks
    - Error handling and resilience
    """

    def test_collector_initialization_complete(self):
        """Test DCGMTelemetryCollector initialization with custom parameters.

        Verifies that the collector properly stores configuration parameters
        including DCGM URL, collection interval, and collector ID.
        Also checks that the initial lifecycle state is correct.
        """
        collector = DCGMTelemetryCollector(
            dcgm_url="http://localhost:9401/metrics",
            collection_interval=0.1,
            collector_id="test_collector",
        )

        assert collector.endpoint_url == "http://localhost:9401/metrics"
        assert collector.collection_interval == 0.1
        assert collector.id == "test_collector"
        assert collector._session is None  # Not initialized yet
        assert not collector.was_initialized
        assert not collector.was_started

    def test_collector_initialization_minimal(self):
        """Test DCGMTelemetryCollector initialization with minimal parameters.

        Verifies that the collector applies correct default values when only
        the required DCGM URL is provided. Tests default collection interval
        and default collector ID generation.
        """
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        assert collector.endpoint_url == "http://localhost:9401/metrics"
        assert collector.collection_interval == 0.333  # Default collection interval
        assert collector.id == "telemetry_collector"  # Default ID
        assert collector._record_callback is None
        assert collector._error_callback is None


class TestPrometheusMetricParsing:
    """Test DCGM Prometheus metric parsing functionality.

    This test class focuses on the parsing of DCGM Prometheus format responses
    using the new prometheus_client parser.
    """

    def test_complete_parsing_single_gpu(self, sample_dcgm_data):
        """Test parsing complete DCGM response into TelemetryRecord for one GPU.

        Verifies that the collector can parse a multi-line DCGM response containing
        various metrics for a single GPU and consolidate them into one TelemetryRecord.
        Tests proper unit scaling (MiB→GB for memory, mJ→MJ for energy) and
        that all metadata and metric values are correctly assigned.
        """
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        records = collector._parse_metrics_to_records(sample_dcgm_data)
        assert len(records) == 1

        record = records[0]
        assert record.dcgm_url == "http://localhost:9401/metrics"
        assert record.gpu_index == 0
        assert record.gpu_model_name == "NVIDIA RTX 6000 Ada Generation"
        assert record.gpu_uuid.startswith("GPU-")
        assert record.hostname == "ed7e7a5e585f"

        # Verify telemetry data has reasonable values from DCGMFaker
        assert record.telemetry_data.gpu_power_usage is not None
        assert 0 < record.telemetry_data.gpu_power_usage < 400
        assert record.telemetry_data.energy_consumption is not None
        assert record.telemetry_data.gpu_memory_used is not None

    def test_complete_parsing_multi_gpu(self, multi_gpu_dcgm_data):
        """Test parsing complete DCGM response for multiple GPUs.

        Verifies that the collector can parse a multi-line DCGM response containing
        metrics for multiple GPUs and create separate TelemetryRecord objects for each.
        Tests that GPU-specific metadata is correctly associated with the right GPU.
        """
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        records = collector._parse_metrics_to_records(multi_gpu_dcgm_data)
        assert len(records) == 3

        records.sort(key=lambda r: r.gpu_index)

        # Verify each GPU has correct metadata
        assert records[0].gpu_index == 0
        assert records[0].gpu_model_name == "NVIDIA RTX 6000 Ada Generation"
        assert records[0].gpu_uuid.startswith("GPU-")
        assert records[1].gpu_index == 1
        assert records[1].gpu_model_name == "NVIDIA RTX 6000 Ada Generation"
        assert records[2].gpu_index == 2
        assert records[2].gpu_model_name == "NVIDIA RTX 6000 Ada Generation"

        # Verify all GPUs have unique UUIDs
        uuids = {r.gpu_uuid for r in records}
        assert len(uuids) == 3

    def test_empty_response_handling(self):
        """Test parsing logic with empty or comment-only DCGM responses.

        UNIT TEST SCOPE: Tests the _parse_metrics_to_records() method directly
        with various empty input cases to ensure robust parsing logic.

        Verifies that the collector gracefully handles edge cases:
        - Completely empty responses
        - Responses containing only comments (# HELP, # TYPE)
        - Whitespace-only responses

        Should return empty record list without crashing.

        Note: For full pipeline testing with empty responses, see
        test_telemetry_integration.py::test_empty_dcgm_response_handling()
        """
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        empty_cases = [
            "",  # Empty
            "# HELP comment\n# TYPE comment",  # Only comments
            "   \n\n   ",  # Only whitespace
        ]

        for empty_data in empty_cases:
            records = collector._parse_metrics_to_records(empty_data)
            assert len(records) == 0


class TestHttpCommunication:
    """Test HTTP communication with DCGM endpoints using aiohttp."""

    @pytest.mark.asyncio
    async def test_endpoint_reachability_success(self):
        """Test DCGM endpoint reachability check with successful HTTP response."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        with patch("aiohttp.ClientSession.head") as mock_head:
            # Mock successful HEAD response with Prometheus content-type
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {
                "Content-Type": "text/plain; version=0.0.4; charset=utf-8"
            }
            mock_response.raise_for_status = AsyncMock()
            mock_head.return_value.__aenter__.return_value = mock_response

            # Initialize the collector to set up the session
            await collector.initialize()

            # Test reachability
            result = await collector.is_url_reachable()
            assert result is True

            await collector.stop()

    @pytest.mark.asyncio
    async def test_endpoint_reachability_failures(self, time_traveler):
        """Test DCGM endpoint reachability check with various failure scenarios."""
        collector = DCGMTelemetryCollector("http://nonexistent:9401/metrics")

        with (
            patch("aiohttp.ClientSession.head") as mock_head,
            patch("aiohttp.ClientSession.get") as mock_get,
        ):
            # Mock different failure scenarios
            failure_scenarios = [
                aiohttp.ClientError("Connection failed"),
                aiohttp.ServerTimeoutError("Timeout"),
            ]

            await collector.initialize()

            for exception in failure_scenarios:
                mock_head.side_effect = exception
                mock_get.side_effect = exception
                result = await collector.is_url_reachable()
                assert result is False

            await collector.stop()

    @pytest.mark.asyncio
    async def test_endpoint_reachability_head_fallback(self):
        """Test that HEAD request falls back to GET when HEAD returns non-200."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        with (
            patch("aiohttp.ClientSession.head") as mock_head,
            patch("aiohttp.ClientSession.get") as mock_get,
        ):
            # Mock HEAD returning 405 (Method Not Allowed)
            mock_head_response = AsyncMock()
            mock_head_response.status = 405
            mock_head.return_value.__aenter__.return_value = mock_head_response

            # Mock GET returning 200
            mock_get_response = AsyncMock()
            mock_get_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_get_response

            await collector.initialize()
            result = await collector.is_url_reachable()

            assert result is True
            # Both HEAD and GET should have been called
            mock_head.assert_called_once()
            mock_get.assert_called_once()

            await collector.stop()

    @pytest.mark.asyncio
    async def test_endpoint_reachability_without_session(self):
        """Test reachability check creates temporary session when collector not initialized."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        # Don't initialize - should create temporary session
        with patch("aiohttp.ClientSession") as mock_session_class:
            # Create mock response
            mock_response = MagicMock()
            mock_response.status = 200

            # Create mock for the response context manager
            mock_response_cm = MagicMock()
            mock_response_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response_cm.__aexit__ = AsyncMock(return_value=None)

            # Create mock session
            mock_session = MagicMock()
            mock_session.head = MagicMock(return_value=mock_response_cm)

            # Make ClientSession() return an async context manager that yields the mock_session
            mock_context_manager = MagicMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_context_manager

            result = await collector.is_url_reachable()

            assert result is True
            # Temporary session should be created
            mock_session_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_fetching(self, sample_dcgm_data):
        """Test successful HTTP fetching of DCGM metrics."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        with patch("aiohttp.ClientSession.get") as mock_get:
            # Mock successful response with sample data
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.return_value = sample_dcgm_data
            mock_response.raise_for_status = Mock(return_value=None)
            mock_get.return_value.__aenter__.return_value = mock_response

            await collector.initialize()

            result = await collector._fetch_metrics_text()
            # _fetch_metrics_text returns FetchResult with text and trace_timing
            assert result.text == sample_dcgm_data

            await collector.stop()

    @pytest.mark.asyncio
    async def test_fetch_metrics_session_closed(self):
        """Test fetch_metrics raises error when session is closed."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        await collector.initialize()

        # Close the session
        await collector._session.close()

        # Should raise CancelledError due to closed session
        with pytest.raises(asyncio.CancelledError):
            await collector._fetch_metrics_text()

    @pytest.mark.asyncio
    async def test_fetch_metrics_when_stop_requested(self):
        """Test fetch_metrics raises CancelledError when stop is requested."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        await collector.initialize()

        # Set stop_requested flag
        collector.stop_requested = True

        # Should raise CancelledError
        with pytest.raises(asyncio.CancelledError):
            await collector._fetch_metrics_text()

        # Clean up
        collector.stop_requested = False
        await collector.stop()

    @pytest.mark.asyncio
    async def test_fetch_metrics_no_session(self):
        """Test fetch_metrics raises error when session not initialized."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        # Don't initialize - session is None
        with pytest.raises(RuntimeError, match="HTTP session not initialized"):
            await collector._fetch_metrics_text()


class TestCollectionLifecycle:
    """Test async lifecycle and background collection functionality."""

    @pytest.mark.asyncio
    async def test_successful_collection_loop(self, faker):
        """Test successful telemetry collection with proper lifecycle management."""
        mock_callback = AsyncMock()

        collector = DCGMTelemetryCollector(
            dcgm_url="http://localhost:9401/metrics",
            collection_interval=0.1,
            record_callback=mock_callback,
        )

        with patch("aiohttp.ClientSession.get") as mock_get:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.side_effect = faker.generate
            mock_response.raise_for_status = Mock(return_value=None)
            mock_get.return_value.__aenter__.return_value = mock_response

            await collector.initialize()

            await collector._collect_and_process_metrics()

            await collector.stop()

            mock_callback.assert_called_once()
            call_args = mock_callback.call_args[0]
            records = call_args[0]
            assert len(records) == 2  # 2 GPUs
            assert all(isinstance(r, TelemetryRecord) for r in records)

    @pytest.mark.asyncio
    async def test_error_handling_in_collection_loop(self):
        """Test error handling by directly calling collection task wrapper.

        Tests that errors during collection are caught and sent via error callback.
        """
        mock_error_callback = AsyncMock()

        collector = DCGMTelemetryCollector(
            dcgm_url="http://localhost:9401/metrics",
            collection_interval=0.05,
            error_callback=mock_error_callback,
        )

        with patch("aiohttp.ClientSession.get") as mock_get:
            # Mock HTTP error
            mock_get.side_effect = aiohttp.ClientError("Connection failed")

            await collector.initialize()

            await collector.collect_and_process_metrics()
            await collector.collect_and_process_metrics()

            await collector.stop()

            assert mock_error_callback.call_count == 2
            for call in mock_error_callback.call_args_list:
                error = call[0][0]
                assert hasattr(error, "message")

    @pytest.mark.asyncio
    async def test_callback_exception_resilience(self, faker):
        """Test that collection continues even if callback raises exceptions.

        Verifies that exceptions in callbacks don't crash the collection process.
        """
        mock_callback = AsyncMock(side_effect=ValueError("Callback failed"))

        collector = DCGMTelemetryCollector(
            dcgm_url="http://localhost:9401/metrics",
            collection_interval=0.1,
            record_callback=mock_callback,
        )

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            # Use side_effect to generate different data each time to avoid deduplication
            mock_response.text.side_effect = faker.generate
            mock_response.raise_for_status = Mock(return_value=None)
            mock_get.return_value.__aenter__.return_value = mock_response

            await collector.initialize()

            await collector._collect_and_process_metrics()
            await collector._collect_and_process_metrics()

            await collector.stop()

            assert mock_callback.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_start_calls_safety(self):
        """Test that multiple start calls on same instance are handled safely.

        UNIT TEST SCOPE: Tests lifecycle safety when calling start() multiple times
        on the same collector instance, which should be idempotent.

        Note: For testing multiple start/stop cycles with separate instances
        (real-world usage), see integration test test_telemetry_collector_multiple_start_stop()
        """
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        await collector.initialize()

        # Multiple start calls should be safe
        await collector.start()
        await collector.start()  # Should be ignored

        assert collector.is_running

        await collector.stop()

    @pytest.mark.asyncio
    async def test_stop_before_start_safety(self):
        """Test that stopping before starting doesn't cause issues."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        # Should handle stop before start gracefully
        await collector.stop()  # Should not raise exceptions


class TestDataProcessingEdgeCases:
    """Test edge cases in data processing and scaling."""

    def test_unit_scaling_accuracy(self):
        """Test accuracy of unit scaling factors for different metrics."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        test_metrics = {
            "gpu_power_usage": 100.0,  # Should remain unchanged (W)
            "energy_consumption": 1000.0,  # mJ -> MJ (divide by 1e9)
            "gpu_memory_used": 1024.0,  # MiB -> GB (divide by 953.674...)
        }

        scaled = collector._apply_scaling_factors(test_metrics)

        assert scaled["gpu_power_usage"] == 100.0
        assert abs(scaled["energy_consumption"] - 1e-6) < 1e-10  # 1000mJ = 1e-6 MJ
        assert (
            abs(scaled["gpu_memory_used"] - 1.073741824) < 1e-6
        )  # 1024 MiB ≈ 1.073 GB

    def test_temporal_consistency_in_batches(self, sample_dcgm_data):
        """Test that all records in a batch have consistent timestamps."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        records = collector._parse_metrics_to_records(sample_dcgm_data)

        if len(records) > 1:
            timestamps = [r.timestamp_ns for r in records]
            # All records in same batch should have same timestamp
            assert all(ts == timestamps[0] for ts in timestamps)

    def test_mixed_quality_response_resilience(self):
        """Test resilience when DCGM response contains mix of valid/invalid data."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        mixed_response = """
        # Valid metric
        DCGM_FI_DEV_POWER_USAGE{gpu="0",modelName="RTX",UUID="valid-uuid"} 100.0
        # Invalid metric (missing GPU index)
        DCGM_FI_DEV_POWER_USAGE{modelName="RTX",UUID="invalid"} 200.0
        # Another valid metric
        DCGM_FI_DEV_GPU_UTIL{gpu="0",modelName="RTX",UUID="valid-uuid"} 85.0
        """

        records = collector._parse_metrics_to_records(mixed_response)

        # Should successfully parse valid entries, skip invalid ones
        assert len(records) == 1  # Only one valid GPU
        assert records[0].gpu_index == 0

    @pytest.mark.asyncio
    async def test_empty_url_reachability(self):
        """Test URL reachability check with empty URL."""
        collector = DCGMTelemetryCollector("")

        result = await collector.is_url_reachable()
        assert result is False

    def test_invalid_prometheus_format_handling(self):
        """Test handling of completely invalid Prometheus format."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        # Invalid format that cannot be parsed
        invalid_data = "invalid prometheus {{{{{ data"

        records = collector._parse_metrics_to_records(invalid_data)
        # Should return empty list without crashing
        assert records == []

    def test_nan_inf_values_filtering(self):
        """Test that NaN and inf values are filtered out during parsing."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        metrics_with_invalid_values = """
        # NaN value
        DCGM_FI_DEV_POWER_USAGE{gpu="0",modelName="RTX",UUID="uuid-1"} NaN
        # Inf value
        DCGM_FI_DEV_GPU_UTIL{gpu="0",modelName="RTX",UUID="uuid-1"} Inf
        # -Inf value
        DCGM_FI_DEV_GPU_TEMP{gpu="0",modelName="RTX",UUID="uuid-1"} -Inf
        # Valid value
        DCGM_FI_DEV_FB_USED{gpu="0",modelName="RTX",UUID="uuid-1"} 1024.0
        """

        records = collector._parse_metrics_to_records(metrics_with_invalid_values)

        # Should only include the valid metric
        assert len(records) == 1
        # NaN, Inf, -Inf should be filtered out
        assert records[0].telemetry_data.gpu_power_usage is None
        assert records[0].telemetry_data.gpu_utilization is None
        assert records[0].telemetry_data.gpu_temperature is None
        # Valid value should be present
        assert records[0].telemetry_data.gpu_memory_used is not None

    def test_invalid_gpu_index_handling(self):
        """Test handling of non-numeric GPU index values."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        invalid_gpu_index_data = """
        # Invalid GPU index (not a number)
        DCGM_FI_DEV_POWER_USAGE{gpu="invalid",modelName="RTX",UUID="uuid-1"} 100.0
        # Valid GPU index
        DCGM_FI_DEV_POWER_USAGE{gpu="1",modelName="RTX2",UUID="uuid-2"} 150.0
        """

        records = collector._parse_metrics_to_records(invalid_gpu_index_data)

        # Should only parse the record with valid GPU index
        assert len(records) == 1
        assert records[0].gpu_index == 1

    @pytest.mark.asyncio
    async def test_error_callback_exception_handling(self):
        mock_error_callback = AsyncMock(
            side_effect=RuntimeError("Error callback failed")
        )

        collector = DCGMTelemetryCollector(
            dcgm_url="http://localhost:9401/metrics",
            collection_interval=0.05,
            error_callback=mock_error_callback,
        )

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.side_effect = aiohttp.ClientError("Connection failed")

            await collector.initialize()
            await collector.collect_and_process_metrics()
            await collector.stop()

            mock_error_callback.assert_called_once()
            error = mock_error_callback.call_args[0][0]
            assert hasattr(error, "message")

    @pytest.mark.asyncio
    async def test_collection_without_callbacks(self, faker):
        collector = DCGMTelemetryCollector(
            dcgm_url="http://localhost:9401/metrics",
            collection_interval=0.1,
        )

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.side_effect = faker.generate
            mock_response.raise_for_status = Mock(return_value=None)
            mock_get.return_value.__aenter__.return_value = mock_response

            await collector.initialize()
            await collector._collect_and_process_metrics()
            await collector.stop()

    @pytest.mark.asyncio
    async def test_collection_with_empty_records(self):
        mock_callback = AsyncMock()

        collector = DCGMTelemetryCollector(
            dcgm_url="http://localhost:9401/metrics",
            collection_interval=0.1,
            record_callback=mock_callback,
        )

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.return_value = "# HELP comment\n# TYPE comment"
            mock_response.raise_for_status = Mock(return_value=None)
            mock_get.return_value.__aenter__.return_value = mock_response

            await collector.initialize()
            await collector._collect_and_process_metrics()
            await collector.stop()

            mock_callback.assert_not_called()

    def test_scaling_factors_with_none_values(self):
        """Test that scaling factors handle None values correctly."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        metrics_with_none = {
            "gpu_power_usage": None,
            "energy_consumption": 1000.0,
            "gpu_memory_used": None,
        }

        scaled = collector._apply_scaling_factors(metrics_with_none)

        # None values should remain None
        assert scaled["gpu_power_usage"] is None
        assert scaled["gpu_memory_used"] is None
        # Non-None values should be scaled
        assert abs(scaled["energy_consumption"] - 1e-6) < 1e-10

    def test_scaling_factors_preserves_unscaled_metrics(self):
        """Test that metrics without scaling factors are preserved as-is."""
        collector = DCGMTelemetryCollector("http://localhost:9401/metrics")

        metrics = {
            "gpu_power_usage": 100.0,
            "unscaled_metric": 999.0,
        }

        scaled = collector._apply_scaling_factors(metrics)

        # Unscaled metric should remain unchanged
        assert scaled["unscaled_metric"] == 999.0
        # Scaled metric should remain unchanged (power has factor 1.0)
        assert scaled["gpu_power_usage"] == 100.0
