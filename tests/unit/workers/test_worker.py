# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

import aiperf.endpoints  # noqa: F401  # Import to register endpoints
from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.messages import CreditDropMessage
from aiperf.common.models import ParsedResponse, ReasoningResponseData, TextResponseData
from aiperf.common.models.record_models import RequestInfo, RequestRecord
from aiperf.workers.worker import Worker


class MockWorker(Worker):
    """Mock implementation of Worker for testing."""

    def __init__(self):
        with (
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector"
            ) as mock_tcp_connector,
            patch(
                "aiperf.common.factories.EndpointFactory.create_instance"
            ) as mock_client_factory,
        ):
            mock_tcp_connector.return_value = Mock()

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
        mock_wait_for.assert_called_once()
        call_args = mock_wait_for.call_args
        assert call_args[1]["timeout"] == 2.0

    @patch("asyncio.wait_for")
    async def test_send_with_optional_cancel_timeout(self, mock_wait_for, worker):
        """Test request that times out."""

        async def simple_coroutine():
            return RequestRecord(timestamp_ns=time.time_ns())

        async def mock_wait_for_timeout(coro, timeout):
            with contextlib.suppress(GeneratorExit):
                coro.close()
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
        """Test that nanoseconds are correctly converted to seconds."""
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

            async def mock_wait_for_impl(coro, timeout):
                await coro
                return RequestRecord(timestamp_ns=time.time_ns())

            mock_wait_for.side_effect = mock_wait_for_impl

            await worker._send_with_optional_cancel(
                send_coroutine=simple_coroutine(),
                should_cancel=True,
                cancel_after_ns=cancel_after_ns,
            )

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
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(worker.inference_client, "endpoint", mock_endpoint)
        turn = await worker._process_response(RequestRecord())
        assert turn.texts[0].contents == ["Hello, world!"]

    async def test_process_response_empty(self, monkeypatch, worker):
        """Ensure process_response handles empty responses correctly."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text=""),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(worker.inference_client, "endpoint", mock_endpoint)
        turn = await worker._process_response(RequestRecord())
        assert turn is None

    async def test_process_response_reasoning_extracts_content(
        self, monkeypatch, worker
    ):
        """Ensure process_response extracts content from reasoning responses."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=ReasoningResponseData(
                reasoning="Let me think...",
                content="The answer is 42.",
            ),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(worker.inference_client, "endpoint", mock_endpoint)
        turn = await worker._process_response(RequestRecord())
        assert turn.texts[0].contents == ["The answer is 42."]

    async def test_process_response_reasoning_only_returns_none(
        self, monkeypatch, worker
    ):
        """Ensure process_response returns None for reasoning-only responses (no content)."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=ReasoningResponseData(
                reasoning="Let me think about this...",
                content=None,
            ),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(worker.inference_client, "endpoint", mock_endpoint)
        turn = await worker._process_response(RequestRecord())
        assert turn is None

    async def test_process_response_mixed_reasoning_and_text_combines_content(
        self, monkeypatch, worker
    ):
        """Ensure process_response combines text and reasoning content."""
        mock_parsed_responses = [
            ParsedResponse(
                perf_ns=0,
                data=TextResponseData(text="Hello"),
            ),
            ParsedResponse(
                perf_ns=1,
                data=ReasoningResponseData(
                    reasoning="Thinking...",
                    content="World",
                ),
            ),
        ]
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=mock_parsed_responses)
        monkeypatch.setattr(worker.inference_client, "endpoint", mock_endpoint)
        turn = await worker._process_response(RequestRecord())
        assert turn.texts[0].contents == ["HelloWorld"]

    async def test_build_response_record(
        self, worker, monkeypatch, sample_conversations
    ):
        """Test that _build_response_record sets all fields correctly."""
        conversation = sample_conversations["session_1"]
        first_turn = conversation.turns[0]

        message = CreditDropMessage(
            service_id="test-service",
            conversation_id=conversation.session_id,
            phase=CreditPhase.PROFILING,
            credit_drop_ns=None,
            should_cancel=False,
            cancel_after_ns=123456789,
            credit_num=1,
        )

        dummy_record = RequestRecord()
        dummy_record.start_perf_ns = 1000

        # Patch _call_inference_api_internal to return dummy_record
        monkeypatch.setattr(
            worker,
            "_call_inference_api_internal",
            AsyncMock(return_value=dummy_record),
        )

        turn_index = 0
        drop_perf_ns = 900

        result = await worker._build_response_record(
            request_info=RequestInfo(
                model_endpoint=worker.model_endpoint,
                conversation_id=conversation.session_id,
                turn_index=turn_index,
                credit_phase=message.phase,
                cancel_after_ns=message.cancel_after_ns,
                credit_num=message.credit_num,
                turns=[first_turn],
            ),
            drop_perf_ns=drop_perf_ns,
        )

        assert result.model_name == "test-model"
        assert result.conversation_id == "session_1"
        assert result.turn_index == 0
        assert result.credit_phase == CreditPhase.PROFILING
        assert result.cancel_after_ns == 123456789
        assert result.credit_drop_latency == 100

    async def test_build_response_record_credit_drop_latency_only_first_turn(
        self, worker, monkeypatch, sample_conversations
    ):
        """Test that credit_drop_latency is only set for the first turn."""

        conversation = sample_conversations["session_1"]
        all_turns = conversation.turns

        message = CreditDropMessage(
            service_id="test-service",
            conversation_id=conversation.session_id,
            phase=CreditPhase.PROFILING,
            credit_drop_ns=None,
            should_cancel=False,
            cancel_after_ns=123456789,
            credit_num=1,
        )

        dummy_record = RequestRecord()
        dummy_record.start_perf_ns = 1000

        # Patch _call_inference_api_internal to return dummy_record
        monkeypatch.setattr(
            worker,
            "_call_inference_api_internal",
            AsyncMock(return_value=dummy_record),
        )

        turn_index = 1
        drop_perf_ns = 900

        result = await worker._build_response_record(
            request_info=RequestInfo(
                model_endpoint=worker.model_endpoint,
                conversation_id=conversation.session_id,
                turn_index=turn_index,
                credit_phase=message.phase,
                cancel_after_ns=message.cancel_after_ns,
                credit_num=message.credit_num,
                turns=all_turns,
            ),
            drop_perf_ns=drop_perf_ns,
        )

        assert result.model_name == "test-model"
        assert result.conversation_id == "session_1"
        assert result.turn_index == 1
        assert result.credit_phase == CreditPhase.PROFILING
        assert result.cancel_after_ns == 123456789
        assert (
            not hasattr(result, "credit_drop_latency")
            or result.credit_drop_latency is None
        )

    @pytest.mark.asyncio
    async def test_x_request_id_and_x_correlation_id_passed_to_client(self, worker):
        """Test that x_request_id and x_correlation_id are passed to the inference client via RequestInfo."""
        import uuid

        from aiperf.common.models import Text, Turn

        message = CreditDropMessage(
            service_id="test-service",
            phase=CreditPhase.PROFILING,
            credit_num=1,
        )
        turn = Turn(texts=[Text(contents=["test"])], model="test-model")
        x_request_id = str(uuid.uuid4())

        captured_request_info = None

        async def mock_transport_send(request_info, payload):
            nonlocal captured_request_info
            captured_request_info = request_info
            return RequestRecord(start_perf_ns=1000)

        worker.inference_client.transport.send_request = AsyncMock(
            side_effect=mock_transport_send
        )
        worker.inference_client.endpoint.format_payload = Mock(
            return_value={"test": "payload"}
        )

        request_info = RequestInfo(
            model_endpoint=worker.model_endpoint,
            x_request_id=x_request_id,
            x_correlation_id=message.request_id,
            turn_index=0,
            turns=[turn],
        )
        await worker._call_inference_api_internal(request_info, 0)

        assert captured_request_info is not None
        assert captured_request_info.x_request_id == x_request_id
        assert captured_request_info.x_correlation_id == message.request_id
