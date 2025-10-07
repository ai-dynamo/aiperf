# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.messages import CreditDropMessage
from aiperf.common.models import ParsedResponse, TextResponseData
from aiperf.common.models.record_models import RequestRecord
from aiperf.workers.worker import Worker


class MockWorker(Worker):
    """Mock implementation of Worker for testing."""

    def __init__(self):
        with (
            patch(
                "aiperf.clients.http.aiohttp_client.create_tcp_connector"
            ) as mock_tcp_connector,
            patch(
                "aiperf.common.factories.ResponseExtractorFactory.create_instance"
            ) as mock_extractor_factory,
            patch(
                "aiperf.common.factories.InferenceClientFactory.create_instance"
            ) as mock_client_factory,
        ):
            mock_tcp_connector.return_value = Mock()

            mock_extractor = Mock()
            mock_extractor.extract_response_data = AsyncMock(return_value=[])
            mock_extractor_factory.return_value = mock_extractor

            mock_client = Mock()
            mock_client.send_request = AsyncMock()
            mock_client_factory.return_value = mock_client

            super().__init__(
                service_config=ServiceConfig(),
                user_config=UserConfig(
                    endpoint=EndpointConfig(model_names=["test-model"]),
                ),
                service_id="mock-service-id",
            )


@pytest.mark.asyncio
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

    async def test_process_response(self, monkeypatch, worker):
        """Ensure process_response extracts text correctly from RequestRecord."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text="Hello, world!"),
        )
        mock_extractor = Mock()
        mock_extractor.extract_response_data = AsyncMock(
            return_value=[mock_parsed_response]
        )
        monkeypatch.setattr(worker, "extractor", mock_extractor)
        turn = await worker._process_response(RequestRecord())
        assert turn.texts[0].contents == ["Hello, world!"]

    async def test_process_response_empty(self, monkeypatch, worker):
        """Ensure process_response handles empty responses correctly."""
        # mock_parsed_response = RequestRecord(responses=[])
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text=""),
        )
        mock_extractor = Mock()
        mock_extractor.extract_response_data = AsyncMock(
            return_value=[mock_parsed_response]
        )
        monkeypatch.setattr(worker, "extractor", mock_extractor)
        turn = await worker._process_response(RequestRecord())
        assert turn is None

    @pytest.mark.asyncio
    async def test_build_response_record(
        self, worker, monkeypatch, sample_conversations
    ):
        """Test that _build_response_record sets all fields correctly."""
        # Use the first conversation from the fixture
        conversation = sample_conversations["session_1"]
        first_turn = conversation.turns[0]

        message = CreditDropMessage(
            service_id="test-service",
            conversation_id=conversation.session_id,
            phase=CreditPhase.PROFILING,
            credit_drop_ns=None,
            should_cancel=False,
            cancel_after_ns=123456789,
        )

        dummy_record = RequestRecord()
        dummy_record.start_perf_ns = 1000

        # Patch _call_inference_api_internal to return dummy_record
        monkeypatch.setattr(
            worker,
            "_call_inference_api_internal",
            AsyncMock(return_value=dummy_record),
        )
        # Patch model_endpoint
        worker.model_endpoint = Mock()
        worker.model_endpoint.primary_model_name = "primary-model"

        turn_index = 0
        drop_perf_ns = 900

        result = await worker._build_response_record(
            conversation_id=conversation.session_id,
            message=message,
            turn_list=[first_turn],
            turn_index=turn_index,
            drop_perf_ns=drop_perf_ns,
        )

        assert result.model_name == "test-model"  # From the fixture turn
        assert result.conversation_id == "session_1"  # From the fixture conversation
        assert result.turn_index == 0
        assert result.credit_phase == CreditPhase.PROFILING
        assert result.cancel_after_ns == 123456789
        assert result.credit_drop_latency == 100  # 1000 - 900

    @pytest.mark.asyncio
    async def test_build_response_record_credit_drop_latency_only_first_turn(
        self, worker, monkeypatch, sample_conversations
    ):
        """Test that credit_drop_latency is only set for the first turn."""

        conversation = sample_conversations["session_1"]
        first_turn = conversation.turns[0]

        message = CreditDropMessage(
            service_id="test-service",
            conversation_id=conversation.session_id,
            phase=CreditPhase.PROFILING,
            credit_drop_ns=None,
            should_cancel=False,
            cancel_after_ns=123456789,
        )

        dummy_record = RequestRecord()
        dummy_record.start_perf_ns = 1000

        # Patch _call_inference_api_internal to return dummy_record
        monkeypatch.setattr(
            worker,
            "_call_inference_api_internal",
            AsyncMock(return_value=dummy_record),
        )
        # Patch model_endpoint
        worker.model_endpoint = Mock()
        worker.model_endpoint.primary_model_name = "primary-model"

        turn_index = 1
        drop_perf_ns = 900

        result = await worker._build_response_record(
            conversation_id=conversation.session_id,
            message=message,
            turn_list=[first_turn],
            turn_index=turn_index,
            drop_perf_ns=drop_perf_ns,
        )

        assert result.model_name == "test-model"  # From the fixture turn
        assert result.conversation_id == "session_1"  # From the fixture conversation
        assert result.turn_index == 1
        assert result.credit_phase == CreditPhase.PROFILING
        assert result.cancel_after_ns == 123456789
        assert (
            not hasattr(result, "credit_drop_latency")
            or result.credit_drop_latency is None
        )

    @pytest.mark.asyncio
    async def test_x_request_id_and_x_correlation_id_passed_to_client(self, worker):
        """Test that x_request_id and x_correlation_id are passed to the inference client."""
        import uuid

        from aiperf.common.models import Text, Turn

        message = CreditDropMessage(
            service_id="test-service",
            phase=CreditPhase.PROFILING,
        )
        turn = Turn(texts=[Text(contents=["test"])], model="test-model")
        x_request_id = str(uuid.uuid4())

        captured_args = {}

        async def mock_send_request(*args, **kwargs):
            captured_args.update(kwargs)
            return RequestRecord(start_perf_ns=1000)

        worker.inference_client.send_request = mock_send_request
        worker.request_converter = Mock()
        worker.request_converter.format_payload = AsyncMock(
            return_value={"test": "payload"}
        )

        await worker._call_inference_api_internal(message, [turn], x_request_id)

        assert "x_request_id" in captured_args
        assert captured_args["x_request_id"] == x_request_id
        assert "x_correlation_id" in captured_args
        assert captured_args["x_correlation_id"] == message.request_id
