# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.record_models import RequestRecord
from aiperf.workers.worker import Worker


class MockWorker(Worker):
    """Mock implementation of Worker for testing."""

    def __init__(self):
        self.inference_client = Mock()
        self.inference_client.send_request = AsyncMock()

    @property
    def service_id(self):
        return "mock-service-id"


class TestWorker:
    @pytest.fixture
    def worker(self):
        """Create a mock Worker for testing."""
        return MockWorker()

    async def test_send_with_optional_cancel_should_cancel_false(self, worker):
        """Test _send_with_optional_cancel when should_cancel=False."""
        mock_record = RequestRecord(timestamp_ns=time.time_ns())

        async def mock_coroutine():
            return mock_record

        result = await worker._send_with_optional_cancel(
            send_coroutine=mock_coroutine(),
            should_cancel=False,
            cancel_after_ns=0,
        )

        assert result == mock_record

    @patch("asyncio.wait_for", side_effect=asyncio.TimeoutError())
    async def test_send_with_optional_cancel_zero_timeout(self, mock_wait_for, worker):
        """Test _send_with_optional_cancel when should_cancel=True with cancel_after_ns=0."""
        mock_record = RequestRecord(timestamp_ns=time.time_ns())

        async def mock_coroutine():
            return mock_record

        result = await worker._send_with_optional_cancel(
            send_coroutine=mock_coroutine(),
            should_cancel=True,
            cancel_after_ns=0,
        )

        assert result is None
        assert mock_wait_for.call_args[1]["timeout"] == 0

    @patch("asyncio.wait_for")
    async def test_send_with_optional_cancel_success(self, mock_wait_for, worker):
        """Test successful request with timeout."""
        mock_record = RequestRecord(timestamp_ns=time.time_ns())

        async def simple_coroutine():
            return mock_record

        # Mock wait_for to consume the coroutine and return the successful result
        async def mock_wait_for_impl(coro, timeout):
            # Properly consume the coroutine to avoid warnings
            await coro
            return mock_record

        mock_wait_for.side_effect = mock_wait_for_impl

        result = await worker._send_with_optional_cancel(
            send_coroutine=simple_coroutine(),
            should_cancel=True,
            cancel_after_ns=int(2.0 * NANOS_PER_SECOND),
        )

        assert result == mock_record
        # Verify wait_for was called with correct timeout (2.0 seconds)
        mock_wait_for.assert_called_once()
        call_args = mock_wait_for.call_args
        assert call_args[1]["timeout"] == 2.0

    @patch("asyncio.wait_for")
    async def test_send_with_optional_cancel_timeout(self, mock_wait_for, worker):
        """Test request that times out."""

        async def simple_coroutine():
            return RequestRecord(timestamp_ns=time.time_ns())

        # Mock wait_for to properly consume the coroutine before raising TimeoutError
        async def mock_wait_for_timeout(coro, timeout):
            # Consume the coroutine to avoid warnings, then raise timeout
            with contextlib.suppress(GeneratorExit):
                coro.close()  # Close the coroutine to prevent warnings
            raise asyncio.TimeoutError

        mock_wait_for.side_effect = mock_wait_for_timeout

        result = await worker._send_with_optional_cancel(
            send_coroutine=simple_coroutine(),
            should_cancel=True,
            cancel_after_ns=int(1.0 * NANOS_PER_SECOND),
        )

        assert result is None
        mock_wait_for.assert_called_once()
        call_args = mock_wait_for.call_args
        assert call_args[1]["timeout"] == 1.0

    @patch("asyncio.wait_for")
    async def test_timeout_conversion_precision(self, mock_wait_for, worker):
        """Test that nanoseconds are correctly converted to seconds with proper precision."""
        # Test various nanosecond values
        test_cases = [
            (int(0.5 * NANOS_PER_SECOND), 0.5),
            (int(1.0 * NANOS_PER_SECOND), 1.0),
            (int(2.5 * NANOS_PER_SECOND), 2.5),
            (int(10.123456789 * NANOS_PER_SECOND), 10.123456789),
        ]

        for cancel_after_ns, expected_timeout in test_cases:
            mock_wait_for.reset_mock()

            async def simple_coroutine():
                return RequestRecord(timestamp_ns=time.time_ns())

            # Mock wait_for to properly consume the coroutine and return result
            async def mock_wait_for_impl(coro, timeout):
                # Properly consume the coroutine to avoid warnings
                await coro
                return RequestRecord(timestamp_ns=time.time_ns())

            mock_wait_for.side_effect = mock_wait_for_impl

            await worker._send_with_optional_cancel(
                send_coroutine=simple_coroutine(),
                should_cancel=True,
                cancel_after_ns=cancel_after_ns,
            )

            # Verify the timeout was converted correctly
            call_args = mock_wait_for.call_args
            actual_timeout = call_args[1]["timeout"]
            assert abs(actual_timeout - expected_timeout) < 1e-9, (
                f"Expected timeout {expected_timeout}, got {actual_timeout}"
            )
