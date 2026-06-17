# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    Conversation,
    ParsedResponse,
    ReasoningResponseData,
    RequestRecord,
    SSEMessage,
    TextResponseData,
    Turn,
)
from aiperf.credit.structs import Credit, CreditContext
from aiperf.workers.worker import Worker
from tests.harness.fake_communication import FakeCommunication as FakeCommunication
from tests.harness.fake_service_manager import FakeServiceManager as FakeServiceManager
from tests.harness.fake_tokenizer import FakeTokenizer
from tests.harness.fake_transport import FakeTransport as FakeTransport


@pytest.fixture
async def mock_worker(
    user_config: UserConfig,
    service_config: ServiceConfig,
    fake_tokenizer: FakeTokenizer,
    skip_service_registration,
):
    """Create a fully initialized and started MockWorker (no SystemController needed)."""
    worker = Worker(
        service_config=service_config,
        user_config=user_config,
        service_id="mock-service-id",
    )
    await worker.initialize()
    await worker.start()
    yield worker
    await worker.stop()


@pytest.mark.asyncio
class TestWorker:
    async def test_create_request_info_overrides_only_outgoing_turn(self, mock_worker):
        original = Turn(max_tokens=4096)
        turns = [original]
        credit_context = CreditContext(
            credit=Credit(
                id=1,
                phase=CreditPhase.WARMUP,
                conversation_id="test-conv",
                x_correlation_id="test-correlation",
                turn_index=0,
                num_turns=1,
                issued_at_ns=0,
                max_tokens_override=1,
            ),
            drop_perf_ns=0,
        )

        request_info = mock_worker._create_request_info(
            x_request_id="request-id",
            credit_context=credit_context,
            turns=turns,
        )

        assert request_info.turns[-1].max_tokens == 1
        assert original.max_tokens == 4096

    async def test_process_response(
        self, monkeypatch, mock_worker, sample_request_record
    ):
        """Ensure process_response extracts text correctly from RequestRecord."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text="Hello, world!"),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(sample_request_record)
        assert turn.texts[0].contents == ["Hello, world!"]

    async def test_process_response_empty(
        self, monkeypatch, mock_worker, sample_request_record
    ):
        """Ensure process_response handles empty responses correctly."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text=""),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(sample_request_record)
        assert turn is None

    async def test_process_response_reasoning_extracts_content(
        self, monkeypatch, mock_worker
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
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(RequestRecord())
        assert turn.texts[0].contents == ["The answer is 42."]

    async def test_process_response_reasoning_only_returns_none(
        self, monkeypatch, mock_worker
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
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(RequestRecord())
        assert turn is None

    async def test_process_response_mixed_reasoning_and_text_combines_content(
        self, monkeypatch, mock_worker
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
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(RequestRecord())
        assert turn.texts[0].contents == ["HelloWorld"]


# --- FirstToken Callback Test Helpers ---


def create_first_token_callback(worker: Worker):
    """Create a first token callback that mirrors Worker implementation.

    This callback uses endpoint.parse_response to check if an SSE message
    contains meaningful content.

    Returns:
        Async callback function (ttft_ns, message) -> bool
    """

    async def first_token_callback(ttft_ns: int, message: SSEMessage) -> bool:
        parsed = worker.inference_client.endpoint.parse_response(message)
        return parsed is not None and parsed.data is not None

    return first_token_callback


def setup_mock_endpoint(worker: Worker, monkeypatch, parse_response_return):
    """Setup mock endpoint with specified parse_response return value.

    Args:
        worker: MockWorker instance
        monkeypatch: pytest monkeypatch fixture
        parse_response_return: Return value or side_effect for parse_response
    """
    mock_endpoint = Mock()
    if isinstance(parse_response_return, list):
        mock_endpoint.parse_response = Mock(side_effect=parse_response_return)
    else:
        mock_endpoint.parse_response = Mock(return_value=parse_response_return)
    mock_endpoint.extract_response_data = Mock()  # Should NOT be called
    monkeypatch.setattr(worker.inference_client, "endpoint", mock_endpoint)
    return mock_endpoint


@pytest.mark.asyncio
class TestWorkerFirstTokenCallback:
    """Test suite for Worker's first_token_callback logic."""

    @pytest.mark.parametrize(
        "parse_return,expected_result,description",
        [
            # Meaningful content - should return True
            pytest.param(
                ParsedResponse(
                    perf_ns=100_000_000, data=TextResponseData(text="Hello")
                ),
                True,
                "meaningful text content",
                id="meaningful_content",
            ),
            # None response - should return False
            pytest.param(
                None,
                False,
                "parse_response returns None",
                id="none_response",
            ),
            # ParsedResponse with data=None (usage only) - should return False
            pytest.param(
                ParsedResponse(
                    perf_ns=100_000_000,
                    data=None,
                    usage={"prompt_tokens": 10, "completion_tokens": 0},
                ),
                False,
                "usage-only response with data=None",
                id="none_data",
            ),
        ],
    )
    async def test_callback_return_value(
        self, monkeypatch, mock_worker, parse_return, expected_result, description
    ):
        """Test callback returns correct bool based on parse_response result."""
        setup_mock_endpoint(mock_worker, monkeypatch, parse_return)
        callback = create_first_token_callback(mock_worker)

        test_message = SSEMessage(perf_ns=100_000_000)
        result = await callback(50_000_000, test_message)

        assert result is expected_result, f"Failed for: {description}"

    async def test_callback_finds_first_meaningful_content_after_junk(
        self, monkeypatch, mock_worker
    ):
        """Test callback correctly identifies first meaningful content after junk messages."""
        parse_returns = [
            None,  # First: junk
            ParsedResponse(perf_ns=200_000_000, data=None),  # Second: usage only
            ParsedResponse(  # Third: actual content
                perf_ns=300_000_000,
                data=TextResponseData(text="Finally some content!"),
            ),
        ]

        setup_mock_endpoint(mock_worker, monkeypatch, parse_returns)
        callback = create_first_token_callback(mock_worker)

        messages = [SSEMessage(perf_ns=i * 100_000_000) for i in range(1, 4)]
        results = [await callback(msg.perf_ns, msg) for msg in messages]

        assert results == [False, False, True]


# --- Fixture for CreditContext ---


@pytest.fixture
def sample_credit_context() -> CreditContext:
    """Create a sample CreditContext for testing."""
    return CreditContext(
        credit=Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="test-conv-123",
            x_correlation_id="test-correlation-id",
            turn_index=0,
            num_turns=1,
            issued_at_ns=1000000,
        ),
        drop_perf_ns=2000000,
    )


# --- RetrieveConversation Tests ---


@pytest.mark.asyncio
class TestRetrieveConversation:
    """Test suite for Worker's _retrieve_conversation method."""

    async def test_returns_from_dataset_client_when_available(
        self, mock_worker, sample_credit_context
    ):
        """When _dataset_client is set, should return conversation from it."""
        expected_conversation = Conversation(session_id="test-conv-123", turns=[])
        mock_client = AsyncMock()
        mock_client.get_conversation = AsyncMock(return_value=expected_conversation)
        mock_worker._dataset_client = mock_client

        result = await mock_worker._retrieve_conversation(
            conversation_id="test-conv-123",
            credit_context=sample_credit_context,
        )

        assert result == expected_conversation
        mock_client.get_conversation.assert_called_once_with("test-conv-123")

    async def test_raises_cancelled_error_when_stop_requested_and_no_client(
        self, mock_worker, sample_credit_context
    ):
        """When _dataset_client is None and stop_requested, should raise CancelledError."""
        mock_worker._dataset_client = None
        mock_worker.stop_requested = True

        with pytest.raises(asyncio.CancelledError, match="Stop requested"):
            await mock_worker._retrieve_conversation(
                conversation_id="test-conv-123",
                credit_context=sample_credit_context,
            )

    async def test_falls_back_to_dataset_manager_when_no_client_and_not_stopping(
        self, monkeypatch, mock_worker, sample_credit_context
    ):
        """When _dataset_client is None and not stopping, should request from DatasetManager."""
        mock_worker._dataset_client = None
        expected_conversation = Conversation(session_id="test-conv-123", turns=[])
        mock_fallback = AsyncMock(return_value=expected_conversation)
        monkeypatch.setattr(
            mock_worker, "_request_conversation_from_dataset_manager", mock_fallback
        )

        result = await mock_worker._retrieve_conversation(
            conversation_id="test-conv-123",
            credit_context=sample_credit_context,
        )

        assert result == expected_conversation
        mock_fallback.assert_called_once_with("test-conv-123", sample_credit_context)


@pytest.mark.asyncio
class TestProcessCreditFastPathRouting:
    """Worker's payload-bytes fast path routing.

    The fast path (read ``payload_bytes`` directly from the dataset
    client, bypass session/conversation deserialisation) is gated on
    two conditions:
    1. ``self._is_payload_bytes`` is True (mmap format is PAYLOAD_BYTES)
    2. ``credit_context.credit.agent_depth == 0`` (not a DAG descendant)

    DAG descendants (``agent_depth > 0``) must go through the session
    path even under PAYLOAD_BYTES mmap so FORK children can seed their
    ``UserSession.turn_list`` from the parent session's local state.
    """

    def _make_credit_context(
        self, agent_depth: int, conversation_id: str = "conv-xyz"
    ) -> CreditContext:
        return CreditContext(
            credit=Credit(
                id=1,
                phase=CreditPhase.PROFILING,
                conversation_id=conversation_id,
                x_correlation_id="xcorr",
                turn_index=0,
                num_turns=1,
                issued_at_ns=0,
                agent_depth=agent_depth,
            ),
            drop_perf_ns=0,
        )

    async def test_root_credit_uses_fast_path_when_payload_bytes_mode(
        self, monkeypatch, mock_worker
    ):
        """agent_depth == 0 under PAYLOAD_BYTES mmap → fast path fires."""
        mock_client = AsyncMock()
        mock_client.get_payload_bytes = AsyncMock(
            return_value=b'{"model":"x","messages":[]}'
        )
        mock_worker._dataset_client = mock_client
        mock_worker._is_payload_bytes = True

        execute = AsyncMock()
        session_path = AsyncMock()
        monkeypatch.setattr(mock_worker, "_execute_request", execute)
        monkeypatch.setattr(mock_worker, "_process_credit_with_session", session_path)

        await mock_worker._process_credit(self._make_credit_context(agent_depth=0))

        mock_client.get_payload_bytes.assert_called_once()
        execute.assert_called_once()
        session_path.assert_not_called()

    async def test_child_credit_forced_to_session_path(self, monkeypatch, mock_worker):
        """agent_depth > 0 must bypass the fast path even when
        PAYLOAD_BYTES mmap is active. FORK children need the parent's
        session-local turn_list, which is inaccessible from the fast path.
        """
        mock_client = AsyncMock()
        mock_worker._dataset_client = mock_client
        mock_worker._is_payload_bytes = True

        execute = AsyncMock()
        session_path = AsyncMock()
        monkeypatch.setattr(mock_worker, "_execute_request", execute)
        monkeypatch.setattr(mock_worker, "_process_credit_with_session", session_path)

        await mock_worker._process_credit(self._make_credit_context(agent_depth=1))

        # Fast path never consulted the dataset client for bytes.
        mock_client.get_payload_bytes.assert_not_called()
        execute.assert_not_called()
        session_path.assert_called_once()

    async def test_non_payload_bytes_mode_always_session_path(
        self, monkeypatch, mock_worker
    ):
        """Without PAYLOAD_BYTES mmap, every credit (root or child) goes
        through the session path — the fast path is opt-in via mmap
        format."""
        mock_client = AsyncMock()
        mock_worker._dataset_client = mock_client
        mock_worker._is_payload_bytes = False

        execute = AsyncMock()
        session_path = AsyncMock()
        monkeypatch.setattr(mock_worker, "_execute_request", execute)
        monkeypatch.setattr(mock_worker, "_process_credit_with_session", session_path)

        await mock_worker._process_credit(self._make_credit_context(agent_depth=0))

        mock_client.get_payload_bytes.assert_not_called()
        execute.assert_not_called()
        session_path.assert_called_once()

    async def test_fast_path_falls_back_when_bytes_missing(
        self, monkeypatch, mock_worker
    ):
        """If ``get_payload_bytes`` returns None (stale index, missing
        turn), the worker falls back to the session path rather than
        dispatching an empty request."""
        mock_client = AsyncMock()
        mock_client.get_payload_bytes = AsyncMock(return_value=None)
        mock_worker._dataset_client = mock_client
        mock_worker._is_payload_bytes = True

        execute = AsyncMock()
        session_path = AsyncMock()
        monkeypatch.setattr(mock_worker, "_execute_request", execute)
        monkeypatch.setattr(mock_worker, "_process_credit_with_session", session_path)

        await mock_worker._process_credit(self._make_credit_context(agent_depth=0))

        mock_client.get_payload_bytes.assert_called_once()
        execute.assert_not_called()
        session_path.assert_called_once()


@pytest.mark.asyncio
class TestWorkerCreditRecordLockstep:
    """A credit returned as completed (not cancelled) MUST be accompanied by a
    record. The RecordsManager completion barrier waits for one record per
    completed credit with no timeout, so a completed-without-record credit
    leaves the count permanently short and hangs the run at end-of-phase.
    """

    async def test_completed_credit_without_record_emits_error_record(
        self, monkeypatch, mock_worker, sample_credit_context
    ):
        """When processing fails before any record is emitted, the worker still
        emits an error record for the completed credit (lockstep)."""
        mock_worker._is_payload_bytes = False

        async def boom(*args, **kwargs):
            raise ValueError("conversation retrieval failed before request sent")

        monkeypatch.setattr(mock_worker, "_process_credit_with_session", boom)

        send_record = AsyncMock()
        monkeypatch.setattr(mock_worker, "_send_inference_result_message", send_record)
        credit_send = AsyncMock()
        monkeypatch.setattr(mock_worker.credit_dealer_client, "send", credit_send)

        await mock_worker._on_credit_drop_message_task(sample_credit_context)

        # Credit is returned as completed, not cancelled...
        assert sample_credit_context.returned is True
        assert sample_credit_context.cancelled is False
        credit_send.assert_awaited_once()
        # ...so a record MUST be emitted to keep the records-side count in lockstep.
        send_record.assert_awaited_once()
        emitted_record = send_record.await_args.args[0]
        assert emitted_record.error is not None
        # The synthetic error must also be surfaced on the CreditReturn so
        # increment_returned(..., errored=...) counts it; otherwise the forwarded
        # error record is invisible to the phase-complete request_errors log line.
        assert sample_credit_context.error is not None
        credit_return = credit_send.await_args.args[0]
        assert credit_return.error is not None

    async def test_cancelled_credit_does_not_emit_record(
        self, monkeypatch, mock_worker, sample_credit_context
    ):
        """A cancelled credit is excluded from the barrier target, so the worker
        must NOT fabricate a record for it (which would over-count)."""
        mock_worker._is_payload_bytes = False

        async def cancel(*args, **kwargs):
            raise asyncio.CancelledError()

        monkeypatch.setattr(mock_worker, "_process_credit_with_session", cancel)

        send_record = AsyncMock()
        monkeypatch.setattr(mock_worker, "_send_inference_result_message", send_record)
        credit_send = AsyncMock()
        monkeypatch.setattr(mock_worker.credit_dealer_client, "send", credit_send)

        await mock_worker._on_credit_drop_message_task(sample_credit_context)

        assert sample_credit_context.cancelled is True
        credit_send.assert_awaited_once()
        send_record.assert_not_awaited()

    async def test_failure_record_emit_raising_still_returns_credit(
        self, monkeypatch, mock_worker, sample_credit_context
    ):
        """The lockstep emit must never abort the credit return. If
        _emit_credit_failure_record raises, the finally block must still send
        the CreditReturn and set returned=True -- otherwise the done callback
        returns the credit as completed-without-record and hangs the barrier
        (the exact break the lockstep guard exists to prevent)."""
        mock_worker._is_payload_bytes = False

        async def boom(*args, **kwargs):
            raise ValueError("processing failed before any record was emitted")

        monkeypatch.setattr(mock_worker, "_process_credit_with_session", boom)

        async def emit_boom(*args, **kwargs):
            raise RuntimeError("inference results push socket closed")

        monkeypatch.setattr(mock_worker, "_emit_credit_failure_record", emit_boom)
        credit_send = AsyncMock()
        monkeypatch.setattr(mock_worker.credit_dealer_client, "send", credit_send)

        # Must not propagate out of the task handler.
        await mock_worker._on_credit_drop_message_task(sample_credit_context)

        # Credit is still returned (not cancelled) so concurrency accounting and
        # the done-callback fallback stay consistent.
        assert sample_credit_context.returned is True
        assert sample_credit_context.cancelled is False
        credit_send.assert_awaited_once()
