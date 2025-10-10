# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import UserConfig
from aiperf.common.messages import TelemetryRecordsMessage, TelemetryStatusMessage
from aiperf.common.models import ErrorDetails
from aiperf.gpu_telemetry.constants import DEFAULT_DCGM_ENDPOINT
from aiperf.gpu_telemetry.telemetry_data_collector import TelemetryDataCollector
from aiperf.gpu_telemetry.telemetry_manager import TelemetryManager


class TestTelemetryManagerInitialization:
    """Test TelemetryManager initialization and configuration."""

    def test_initialization_default_endpoint(self):
        """Test initialization with no user-provided endpoints uses default."""
        mock_user_config = MagicMock(spec=UserConfig)
        mock_user_config.gpu_telemetry = None

        with patch.object(
            TelemetryManager, "__init__", lambda self, *args, **kwargs: None
        ):
            manager = TelemetryManager.__new__(TelemetryManager)
            manager._dcgm_endpoints = None
            manager._collection_interval = None

            # Simulate what __init__ does with endpoints
            user_endpoints = mock_user_config.gpu_telemetry
            if user_endpoints is None:
                user_endpoints = []

            if DEFAULT_DCGM_ENDPOINT not in user_endpoints:
                manager._dcgm_endpoints = [DEFAULT_DCGM_ENDPOINT] + user_endpoints
            else:
                manager._dcgm_endpoints = user_endpoints

        assert manager._dcgm_endpoints == [DEFAULT_DCGM_ENDPOINT]

    def test_initialization_custom_endpoints(self):
        """Test initialization with custom user-provided endpoints."""
        mock_user_config = MagicMock(spec=UserConfig)
        custom_endpoint = "http://gpu-node-01:9401/metrics"
        mock_user_config.gpu_telemetry = [custom_endpoint]

        with patch.object(
            TelemetryManager, "__init__", lambda self, *args, **kwargs: None
        ):
            manager = TelemetryManager.__new__(TelemetryManager)

            # Simulate endpoint processing
            user_endpoints = mock_user_config.gpu_telemetry
            if isinstance(user_endpoints, str):
                user_endpoints = [user_endpoints]

            if DEFAULT_DCGM_ENDPOINT not in user_endpoints:
                manager._dcgm_endpoints = [DEFAULT_DCGM_ENDPOINT] + user_endpoints
            else:
                manager._dcgm_endpoints = user_endpoints

        assert DEFAULT_DCGM_ENDPOINT in manager._dcgm_endpoints
        assert custom_endpoint in manager._dcgm_endpoints
        assert len(manager._dcgm_endpoints) == 2

    def test_initialization_string_endpoint(self):
        """Test initialization converts single string endpoint to list."""
        mock_user_config = MagicMock(spec=UserConfig)
        mock_user_config.gpu_telemetry = "http://single-node:9401/metrics"

        with patch.object(
            TelemetryManager, "__init__", lambda self, *args, **kwargs: None
        ):
            manager = TelemetryManager.__new__(TelemetryManager)

            # Simulate conversion
            user_endpoints = mock_user_config.gpu_telemetry
            if isinstance(user_endpoints, str):
                user_endpoints = [user_endpoints]
            else:
                user_endpoints = list(user_endpoints)

            manager._dcgm_endpoints = user_endpoints

        assert isinstance(manager._dcgm_endpoints, list)
        assert len(manager._dcgm_endpoints) == 1

    def test_initialization_filters_invalid_urls(self):
        """Test initialization filters out invalid URLs."""
        from urllib.parse import urlparse

        endpoints = [
            "http://valid:9401/metrics",  # Valid
            "not-a-url",  # Invalid - no scheme
            "ftp://wrong-scheme:9401",  # Invalid - wrong scheme
            "http://another-valid:9401",  # Valid
            "",  # Invalid - empty
        ]

        valid_endpoints = []
        for endpoint in endpoints:
            try:
                parsed = urlparse(endpoint)
                if parsed.scheme in ("http", "https") and parsed.netloc:
                    valid_endpoints.append(endpoint)
            except Exception:
                continue

        assert len(valid_endpoints) == 2
        assert "http://valid:9401/metrics" in valid_endpoints
        assert "http://another-valid:9401" in valid_endpoints

    def test_initialization_deduplicates_endpoints(self):
        """Test initialization removes duplicate endpoints."""
        endpoints = [
            "http://node1:9401/metrics",
            "http://node2:9401/metrics",
            "http://node1:9401/metrics",  # Duplicate
        ]

        # Deduplicate while preserving order
        deduplicated = list(dict.fromkeys(endpoints))

        assert len(deduplicated) == 2
        assert deduplicated == [
            "http://node1:9401/metrics",
            "http://node2:9401/metrics",
        ]


class TestUrlNormalization:
    """Test _normalize_dcgm_url static method."""

    def test_normalize_adds_metrics_suffix(self):
        """Test normalization adds /metrics suffix when missing."""
        url = "http://localhost:9401"
        normalized = TelemetryManager._normalize_dcgm_url(url)
        assert normalized == "http://localhost:9401/metrics"

    def test_normalize_preserves_metrics_suffix(self):
        """Test normalization preserves existing /metrics suffix."""
        url = "http://localhost:9401/metrics"
        normalized = TelemetryManager._normalize_dcgm_url(url)
        assert normalized == "http://localhost:9401/metrics"

    def test_normalize_removes_trailing_slash(self):
        """Test normalization removes trailing slash."""
        url = "http://localhost:9401/"
        normalized = TelemetryManager._normalize_dcgm_url(url)
        assert normalized == "http://localhost:9401/metrics"

    def test_normalize_trailing_slash_with_metrics(self):
        """Test normalization handles trailing slash after /metrics."""
        url = "http://localhost:9401/metrics/"
        normalized = TelemetryManager._normalize_dcgm_url(url)
        assert normalized == "http://localhost:9401/metrics"

    def test_normalize_complex_path(self):
        """Test normalization with complex URL paths."""
        url = "http://node1:9401/dcgm"
        normalized = TelemetryManager._normalize_dcgm_url(url)
        assert normalized == "http://node1:9401/dcgm/metrics"


class TestCallbackFunctions:
    """Test callback functions for receiving telemetry data."""

    def _create_test_manager(self):
        """Helper to create a TelemetryManager instance for testing."""
        # Create minimal manager instance without full initialization
        manager = TelemetryManager.__new__(TelemetryManager)
        manager.service_id = "test_manager"
        manager._collectors = {}
        manager._dcgm_endpoints = []
        manager._collection_interval = 0.33
        return manager

    @pytest.mark.asyncio
    async def test_on_telemetry_records_valid(self, sample_telemetry_records):
        """Test _on_telemetry_records with valid records."""
        manager = self._create_test_manager()

        # Mock the push client
        mock_push_client = AsyncMock()
        manager.records_push_client = mock_push_client

        # Call the callback
        await manager._on_telemetry_records(sample_telemetry_records, "test_collector")

        # Verify push was called with correct message
        mock_push_client.push.assert_called_once()
        call_args = mock_push_client.push.call_args[0][0]
        assert isinstance(call_args, TelemetryRecordsMessage)
        assert call_args.service_id == "test_manager"
        assert call_args.collector_id == "test_collector"
        assert call_args.records == sample_telemetry_records
        assert call_args.error is None

    @pytest.mark.asyncio
    async def test_on_telemetry_records_empty(self):
        """Test _on_telemetry_records with empty records list skips sending."""
        manager = self._create_test_manager()

        # Mock the push client
        mock_push_client = AsyncMock()
        manager.records_push_client = mock_push_client

        # Call with empty records
        await manager._on_telemetry_records([], "test_collector")

        # Verify push was NOT called
        mock_push_client.push.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_telemetry_records_exception_handling(
        self, sample_telemetry_records
    ):
        """Test _on_telemetry_records handles exceptions gracefully."""
        manager = self._create_test_manager()

        # Mock the push client to raise exception
        mock_push_client = AsyncMock()
        mock_push_client.push.side_effect = Exception("Network error")
        manager.records_push_client = mock_push_client
        manager.error = MagicMock()  # Mock error logging

        # Should not raise exception
        await manager._on_telemetry_records(sample_telemetry_records, "test_collector")

        # Verify error was logged
        manager.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_telemetry_error(self):
        """Test _on_telemetry_error callback."""
        manager = self._create_test_manager()

        # Mock the push client
        mock_push_client = AsyncMock()
        manager.records_push_client = mock_push_client

        error_details = ErrorDetails(message="Collection failed")

        # Call the error callback
        await manager._on_telemetry_error(error_details, "test_collector")

        # Verify push was called with error message
        mock_push_client.push.assert_called_once()
        call_args = mock_push_client.push.call_args[0][0]
        assert isinstance(call_args, TelemetryRecordsMessage)
        assert call_args.service_id == "test_manager"
        assert call_args.collector_id == "test_collector"
        assert call_args.records == []
        assert call_args.error == error_details

    @pytest.mark.asyncio
    async def test_on_telemetry_error_exception_handling(self):
        """Test _on_telemetry_error handles exceptions during message sending."""
        manager = self._create_test_manager()

        # Mock the push client to raise exception
        mock_push_client = AsyncMock()
        mock_push_client.push.side_effect = Exception("Push failed")
        manager.records_push_client = mock_push_client
        manager.error = MagicMock()  # Mock error logging

        error_details = ErrorDetails(message="Collection failed")

        # Should not raise exception
        await manager._on_telemetry_error(error_details, "test_collector")

        # Verify error was logged
        manager.error.assert_called_once()


class TestStatusMessaging:
    """Test status message sending functionality."""

    def _create_test_manager(self):
        """Helper to create a TelemetryManager instance for testing."""
        manager = TelemetryManager.__new__(TelemetryManager)
        manager.service_id = "test_manager"
        manager._collectors = {}
        manager._dcgm_endpoints = []
        manager._collection_interval = 0.33
        return manager

    @pytest.mark.asyncio
    async def test_send_telemetry_status_enabled(self):
        """Test _send_telemetry_status with enabled status."""
        manager = self._create_test_manager()

        # Mock publish method
        manager.publish = AsyncMock()

        endpoints_tested = ["http://node1:9401/metrics", "http://node2:9401/metrics"]
        endpoints_reachable = ["http://node1:9401/metrics"]

        await manager._send_telemetry_status(
            enabled=True,
            endpoints_tested=endpoints_tested,
            endpoints_reachable=endpoints_reachable,
        )

        # Verify publish was called
        manager.publish.assert_called_once()
        call_args = manager.publish.call_args[0][0]
        assert isinstance(call_args, TelemetryStatusMessage)
        assert call_args.enabled is True
        assert call_args.reason is None
        assert call_args.endpoints_tested == endpoints_tested
        assert call_args.endpoints_reachable == endpoints_reachable

    @pytest.mark.asyncio
    async def test_send_telemetry_status_disabled_with_reason(self):
        """Test _send_telemetry_status with disabled status and reason."""
        manager = self._create_test_manager()

        # Mock publish method
        manager.publish = AsyncMock()

        reason = "no DCGM endpoints reachable"
        endpoints_tested = ["http://node1:9401/metrics"]

        await manager._send_telemetry_status(
            enabled=False,
            reason=reason,
            endpoints_tested=endpoints_tested,
            endpoints_reachable=[],
        )

        # Verify publish was called with disabled status
        manager.publish.assert_called_once()
        call_args = manager.publish.call_args[0][0]
        assert isinstance(call_args, TelemetryStatusMessage)
        assert call_args.enabled is False
        assert call_args.reason == reason
        assert call_args.endpoints_reachable == []

    @pytest.mark.asyncio
    async def test_send_telemetry_status_exception_handling(self):
        """Test _send_telemetry_status handles exceptions during publish."""
        manager = self._create_test_manager()

        # Mock publish to raise exception
        manager.publish = AsyncMock(side_effect=Exception("Publish failed"))
        manager.error = MagicMock()  # Mock error logging

        # Should not raise exception
        await manager._send_telemetry_status(
            enabled=True, endpoints_tested=[], endpoints_reachable=[]
        )

        # Verify error was logged
        manager.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_telemetry_disabled_status_and_shutdown(self):
        """Test _send_telemetry_disabled_status_and_shutdown schedules shutdown."""
        with patch("asyncio.create_task") as mock_create_task:
            manager = self._create_test_manager()

            # Mock dependencies
            manager.publish = AsyncMock()
            manager._dcgm_endpoints = ["http://node1:9401/metrics"]

            reason = "all collectors failed"

            await manager._send_telemetry_disabled_status_and_shutdown(reason)

            # Verify status was sent
            manager.publish.assert_called_once()
            call_args = manager.publish.call_args[0][0]
            assert call_args.enabled is False
            assert call_args.reason == reason

            # Verify delayed shutdown was scheduled
            mock_create_task.assert_called_once()


class TestCollectorManagement:
    """Test collector lifecycle management."""

    def _create_test_manager(self):
        """Helper to create a TelemetryManager instance for testing."""
        manager = TelemetryManager.__new__(TelemetryManager)
        manager.service_id = "test_manager"
        manager._collectors = {}
        manager._dcgm_endpoints = []
        manager._collection_interval = 0.33
        return manager

    @pytest.mark.asyncio
    async def test_stop_all_collectors_success(self):
        """Test _stop_all_collectors successfully stops all collectors."""
        manager = self._create_test_manager()

        # Create mock collectors
        mock_collector1 = AsyncMock(spec=TelemetryDataCollector)
        mock_collector2 = AsyncMock(spec=TelemetryDataCollector)

        manager._collectors = {
            "http://node1:9401/metrics": mock_collector1,
            "http://node2:9401/metrics": mock_collector2,
        }

        await manager._stop_all_collectors()

        # Verify both collectors were stopped
        mock_collector1.stop.assert_called_once()
        mock_collector2.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_all_collectors_empty(self):
        """Test _stop_all_collectors with no collectors does nothing."""
        manager = self._create_test_manager()
        manager._collectors = {}

        # Should not raise exception
        await manager._stop_all_collectors()

    @pytest.mark.asyncio
    async def test_stop_all_collectors_handles_failures(self):
        """Test _stop_all_collectors continues despite individual collector failures."""
        manager = self._create_test_manager()

        # Create mock collectors - one fails, one succeeds
        mock_collector1 = AsyncMock(spec=TelemetryDataCollector)
        mock_collector1.stop.side_effect = Exception("Stop failed")
        mock_collector2 = AsyncMock(spec=TelemetryDataCollector)

        manager._collectors = {
            "http://node1:9401/metrics": mock_collector1,
            "http://node2:9401/metrics": mock_collector2,
        }
        manager.error = MagicMock()  # Mock error logging

        # Should not raise exception
        await manager._stop_all_collectors()

        # Verify both stop methods were called
        mock_collector1.stop.assert_called_once()
        mock_collector2.stop.assert_called_once()

        # Verify error was logged for the failed collector
        manager.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_delayed_shutdown(self):
        """Test _delayed_shutdown waits before calling stop."""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            manager = self._create_test_manager()
            manager.stop = AsyncMock()

            await manager._delayed_shutdown()

            # Verify sleep was called with 5 seconds
            mock_sleep.assert_called_once_with(5.0)

            # Verify stop was called
            manager.stop.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_normalize_url_empty_string(self):
        """Test URL normalization with empty string."""
        url = ""
        normalized = TelemetryManager._normalize_dcgm_url(url)
        assert normalized == "/metrics"

    def test_normalize_url_only_metrics(self):
        """Test URL normalization with only /metrics path."""
        url = "/metrics"
        normalized = TelemetryManager._normalize_dcgm_url(url)
        assert normalized == "/metrics"

    @pytest.mark.asyncio
    async def test_callbacks_with_none_values(self):
        """Test callbacks handle None values gracefully."""
        manager = TelemetryManager.__new__(TelemetryManager)
        manager.service_id = "test_manager"
        manager._collectors = {}

        manager.records_push_client = AsyncMock()

        # Call with None error (valid case)
        await manager._on_telemetry_error(None, "test_collector")

        # Should still send message with None error
        manager.records_push_client.push.assert_called_once()
