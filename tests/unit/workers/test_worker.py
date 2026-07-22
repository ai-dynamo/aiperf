# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    Conversation,
    ErrorDetails,
    ParsedResponse,
    RequestRecord,
    SSEMessage,
    TextResponseData,
    Turn,
)
from aiperf.config.phases import ConcurrencyPhase
from aiperf.credit.structs import Credit, CreditContext
from aiperf.workers.worker import (
    Worker,
    _is_terminal_context_overflow,
    _phase_needs_first_token_callback,
)
from tests.harness.fake_communication import FakeCommunication as FakeCommunication
from tests.harness.fake_service_manager import FakeServiceManager as FakeServiceManager
from tests.harness.fake_tokenizer import FakeTokenizer
from tests.harness.fake_transport import FakeTransport as FakeTransport


@pytest.mark.parametrize(
    ("phase_data", "expected"),
    [
        (
            {
                "name": "profiling",
                "type": "concurrency",
                "requests": 1,
                "concurrency": 4,
                "prefill_concurrency": 2,
            },
            True,
        ),
        (
            {
                "name": "profiling",
                "type": "concurrency",
                "duration": 600,
                "concurrency": 200,
                "adaptive_scale": True,
                "adaptive_sustain_duration": 120,
                "sla": [
                    {
                        "metric_tag": "time_to_first_token",
                        "stat": "p95",
                        "op": "le",
                        "threshold": 30000,
                    }
                ],
            },
            True,
        ),
        (
            {
                "name": "profiling",
                "type": "concurrency",
                "duration": 600,
                "concurrency": 200,
                "adaptive_scale": True,
                "adaptive_sustain_duration": 120,
                "sla": [
                    {
                        "metric_tag": "request_latency",
                        "stat": "p95",
                        "op": "le",
                        "threshold": 30000,
                    }
                ],
            },
            False,
        ),
    ],
)
def test_phase_needs_first_token_callback(phase_data, expected):
    phase = ConcurrencyPhase.model_validate(phase_data)

    assert _phase_needs_first_token_callback(phase) is expected


@pytest.fixture
async def mock_worker(
    benchmark_run,
    fake_tokenizer: FakeTokenizer,
    skip_service_registration,
):
    """Create a fully initialized and started MockWorker (no SystemController needed)."""
    worker = Worker(
        run=benchmark_run,
        service_id="mock-service-id",
    )
    await worker.initialize()
    await worker.start()
    yield worker
    await worker.stop()


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


@pytest.mark.asyncio
class TestWorkerRequestLatency:
    async def test_request_latency_uses_last_parsed_content_response(
        self: "TestWorkerRequestLatency",
        monkeypatch: pytest.MonkeyPatch,
        mock_worker: Worker,
    ) -> None:
        parse_returns = [
            ParsedResponse(perf_ns=150, data=TextResponseData(text="first")),
            ParsedResponse(perf_ns=200, data=None),
            ParsedResponse(perf_ns=250, data=TextResponseData(text="last")),
        ]
        setup_mock_endpoint(mock_worker, monkeypatch, parse_returns)
        record = RequestRecord(
            start_perf_ns=100,
            responses=[
                SSEMessage(perf_ns=150),
                SSEMessage(perf_ns=200),
                SSEMessage(perf_ns=250),
            ],
        )

        assert mock_worker._request_latency_ns_for_record(record) == 150

    async def test_request_latency_is_none_without_content_response(
        self: "TestWorkerRequestLatency",
        monkeypatch: pytest.MonkeyPatch,
        mock_worker: Worker,
    ) -> None:
        setup_mock_endpoint(
            mock_worker,
            monkeypatch,
            ParsedResponse(perf_ns=200, data=None),
        )
        record = RequestRecord(
            start_perf_ns=100,
            responses=[SSEMessage(perf_ns=200)],
        )

        assert mock_worker._request_latency_ns_for_record(record) is None

    async def test_inter_token_latency_uses_output_sequence_length(
        self: "TestWorkerRequestLatency",
        monkeypatch: pytest.MonkeyPatch,
        mock_worker: Worker,
    ) -> None:
        parse_returns = [
            ParsedResponse(perf_ns=150, data=TextResponseData(text="first")),
            ParsedResponse(perf_ns=200, data=None),
            ParsedResponse(perf_ns=250, data=TextResponseData(text="middle")),
            ParsedResponse(
                perf_ns=350,
                data=TextResponseData(text="last"),
                usage={"completion_tokens": 6},
            ),
        ]
        setup_mock_endpoint(mock_worker, monkeypatch, parse_returns)
        record = RequestRecord(
            start_perf_ns=100,
            responses=[
                SSEMessage(perf_ns=150),
                SSEMessage(perf_ns=200),
                SSEMessage(perf_ns=250),
                SSEMessage(perf_ns=350),
            ],
        )

        assert mock_worker._inter_token_latency_ns_for_record(record) == 40

    async def test_output_sequence_length_uses_final_usage(
        self: "TestWorkerRequestLatency",
        mock_worker: Worker,
    ) -> None:
        parsed_responses = [
            ParsedResponse(
                perf_ns=150,
                data=TextResponseData(text="first"),
                usage={"completion_tokens": 3},
            ),
            ParsedResponse(
                perf_ns=250,
                data=TextResponseData(text="last"),
                usage={"completion_tokens": 7},
            ),
        ]

        assert mock_worker._output_sequence_length_for_responses(parsed_responses) == 7

    async def test_inter_token_latency_is_none_without_two_content_chunks(
        self: "TestWorkerRequestLatency",
        monkeypatch: pytest.MonkeyPatch,
        mock_worker: Worker,
    ) -> None:
        setup_mock_endpoint(
            mock_worker,
            monkeypatch,
            ParsedResponse(perf_ns=150, data=TextResponseData(text="only")),
        )
        record = RequestRecord(
            start_perf_ns=100,
            responses=[SSEMessage(perf_ns=150)],
        )

        assert mock_worker._inter_token_latency_ns_for_record(record) is None

    async def test_inter_token_latency_is_none_without_usage(
        self: "TestWorkerRequestLatency",
        mock_worker: Worker,
    ) -> None:
        record = RequestRecord(start_perf_ns=100, responses=[])
        parsed_responses = [
            ParsedResponse(perf_ns=150, data=TextResponseData(text="first")),
            ParsedResponse(perf_ns=250, data=TextResponseData(text="last")),
        ]
        content_perf_ns = [150, 250]

        assert (
            mock_worker._inter_token_latency_ns_for_record(
                record, content_perf_ns, parsed_responses
            )
            is None
        )

    async def test_inter_token_latency_is_none_for_short_output_sequence(
        self: "TestWorkerRequestLatency",
        mock_worker: Worker,
    ) -> None:
        record = RequestRecord(start_perf_ns=100, responses=[])
        parsed_responses = [
            ParsedResponse(perf_ns=150, data=TextResponseData(text="first")),
            ParsedResponse(
                perf_ns=250,
                data=TextResponseData(text="last"),
                usage={"completion_tokens": 1},
            ),
        ]
        content_perf_ns = [150, 250]

        assert (
            mock_worker._inter_token_latency_ns_for_record(
                record, content_perf_ns, parsed_responses
            )
            is None
        )

    async def test_inter_token_latency_is_none_for_negative_timing(
        self: "TestWorkerRequestLatency",
        mock_worker: Worker,
    ) -> None:
        record = RequestRecord(start_perf_ns=300, responses=[])
        parsed_responses = [
            ParsedResponse(perf_ns=150, data=TextResponseData(text="first")),
            ParsedResponse(
                perf_ns=250,
                data=TextResponseData(text="last"),
                usage={"completion_tokens": 6},
            ),
        ]
        content_perf_ns = [150, 250]

        assert (
            mock_worker._inter_token_latency_ns_for_record(
                record, content_perf_ns, parsed_responses
            )
            is None
        )


class TestWarmupSystemMessage:
    def test_profiling_preserves_system_message(self):
        assert (
            Worker._system_message_for_phase(
                system_message="existing system",
                phase=CreditPhase.PROFILING,
            )
            == "existing system"
        )

    def test_warmup_sets_system_message_when_missing(self):
        assert (
            Worker._system_message_for_phase(
                system_message=None,
                phase=CreditPhase.WARMUP,
            )
            == "warmup"
        )

    def test_warmup_prefixes_existing_system_message(self):
        assert (
            Worker._system_message_for_phase(
                system_message="existing system",
                phase=CreditPhase.WARMUP,
            )
            == "warmup\nexisting system"
        )


@pytest.mark.asyncio
class TestCreateRequestInfo:
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

    async def test_create_request_info_plumbs_finality_from_credit(self, mock_worker):
        # Real Credit struct (not a MagicMock, which would auto-create the
        # attributes and mask a missed plumb) carrying both finality facts.
        credit_context = CreditContext(
            credit=Credit(
                id=1,
                phase=CreditPhase.PROFILING,
                conversation_id="test-conv",
                x_correlation_id="test-correlation",
                turn_index=0,
                num_turns=1,
                issued_at_ns=0,
                is_parent_final=True,
                is_tree_final=True,
            ),
            drop_perf_ns=0,
        )

        request_info = mock_worker._create_request_info(
            x_request_id="request-id",
            credit_context=credit_context,
            turns=[Turn()],
        )

        assert request_info.is_parent_final is True
        assert request_info.is_tree_final is True

    async def test_create_request_info_plumbs_phase_fields_from_credit(
        self, mock_worker
    ):
        credit_context = CreditContext(
            credit=Credit(
                id=1,
                phase=CreditPhase.PROFILING,
                phase_index=2,
                profiling_index=1,
                phase_name="second-profiling",
                phase_kind="profiling",
                conversation_id="test-conv",
                x_correlation_id="test-correlation",
                turn_index=0,
                num_turns=1,
                issued_at_ns=0,
            ),
            drop_perf_ns=0,
        )

        request_info = mock_worker._create_request_info(
            x_request_id="request-id",
            credit_context=credit_context,
            turns=[Turn()],
        )

        assert request_info.phase_index == 2
        assert request_info.profiling_index == 1
        assert request_info.phase_name == "second-profiling"
        assert request_info.phase_kind == "profiling"


@pytest.mark.asyncio
class TestEmitCreditFailureRecord:
    async def test_emit_credit_failure_record_plumbs_phase_fields(
        self,
        mock_worker,
    ):
        credit_context = CreditContext(
            credit=Credit(
                id=1,
                phase=CreditPhase.PROFILING,
                phase_index=2,
                profiling_index=1,
                phase_name="second-profiling",
                phase_kind="profiling",
                conversation_id="test-conv",
                x_correlation_id="test-correlation",
                turn_index=0,
                num_turns=1,
                issued_at_ns=0,
            ),
            drop_perf_ns=0,
            error=ErrorDetails(message="boom", type="CreditProcessingError", code=500),
        )
        mock_worker._send_inference_result_message = AsyncMock()

        await mock_worker._emit_credit_failure_record(credit_context)

        record = mock_worker._send_inference_result_message.call_args.args[0]
        assert record.request_info.phase_index == 2
        assert record.request_info.profiling_index == 1
        assert record.request_info.phase_name == "second-profiling"
        assert record.request_info.phase_kind == "profiling"


# --- Fixture for CreditContext ---


@pytest.fixture
def sample_credit_context() -> CreditContext:
    """Create a sample CreditContext for testing."""
    return CreditContext(
        credit=Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            phase_index=2,
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


# --- Terminal Eviction / Session-Routing Hook Tests ---


@pytest.mark.asyncio
class TestReleaseAndEvictForTerminal:
    """Test suite for Worker's _release_and_evict_for_terminal method."""

    async def test_release_and_evict_for_terminal_notifies_session_end(
        self, mock_worker, sample_credit_context
    ):
        """A terminal eviction fires the routing plugin's post-session hook
        (``InferenceClient.notify_session_end``) so stateful plugins release the
        session even when abandoned before its final turn (e.g. cancellation)."""
        credit = sample_credit_context.credit
        mock_worker.inference_client.notify_session_end = Mock()

        mock_worker._release_and_evict_for_terminal(credit, credit.x_correlation_id)

        mock_worker.inference_client.notify_session_end.assert_called_once_with(
            credit.x_correlation_id
        )


_OVERFLOW_BODY = "This model's maximum context length is 8192 tokens"


class TestTerminalContextOverflowClassifier:
    """``_is_terminal_context_overflow``: a context-overflow error on a
    non-final, non-cancelled turn is terminal (agentic_replay recycles the lane
    and sends no final/cancel credit). Final-turn / cancelled returns go through
    the normal eviction path and must NOT be classified as overflow-terminal."""

    @pytest.mark.parametrize(
        "is_final, cancelled, error, expected",
        [
            param(False, False, _OVERFLOW_BODY, True, id="nonfinal-overflow"),
            param(False, False, "connection reset by peer", False, id="nonfinal-plain-error"),
            param(False, False, None, False, id="nonfinal-no-error"),
            param(True, False, _OVERFLOW_BODY, False, id="final-turn"),
            param(False, True, _OVERFLOW_BODY, False, id="cancelled"),
        ],
    )  # fmt: skip
    def test_classifier(self, is_final, cancelled, error, expected) -> None:
        credit = MagicMock(is_final_turn=is_final)
        ctx = MagicMock(cancelled=cancelled, error=error)
        assert _is_terminal_context_overflow(credit, ctx) is expected


class TestSessionRoutingTerminalHooks:
    """Every terminal disposition reaches ``InferenceClient.notify_session_end``
    so routing plugins get their idempotent post-session hook on ALL four
    terminal paths: final turn, cancellation, terminal context overflow, and the
    done-callback cancel-before-start branch (which the finally block never sees).
    """

    def _dispatch_terminal(
        self, *, is_final: bool, cancelled: bool, error
    ) -> MagicMock:
        """Drive the real terminal-disposition gate on a mock worker.

        Both ``_handle_terminal_disposition`` and ``_release_and_evict_for_terminal``
        are bound as real methods so the final/cancelled path actually reaches
        ``notify_session_end``; everything else (session_manager) stays mocked.
        """
        worker = MagicMock()
        worker.inference_client = MagicMock()
        worker._release_and_evict_for_terminal = (
            Worker._release_and_evict_for_terminal.__get__(worker)
        )
        credit = MagicMock(is_final_turn=is_final)
        ctx = MagicMock(cancelled=cancelled, error=error)
        Worker._handle_terminal_disposition.__get__(worker)(credit, ctx, "conv-X")
        return worker

    def test_final_turn_notifies_session_end(self) -> None:
        worker = self._dispatch_terminal(is_final=True, cancelled=False, error=None)
        worker.inference_client.notify_session_end.assert_called_once_with("conv-X")

    def test_cancelled_notifies_session_end(self) -> None:
        worker = self._dispatch_terminal(is_final=False, cancelled=True, error=None)
        worker.inference_client.notify_session_end.assert_called_once_with("conv-X")

    def test_terminal_context_overflow_notifies_session_end(self) -> None:
        worker = self._dispatch_terminal(
            is_final=False, cancelled=False, error=_OVERFLOW_BODY
        )
        worker.inference_client.notify_session_end.assert_called_once_with("conv-X")

    def test_nonfinal_plain_error_does_not_notify(self) -> None:
        worker = self._dispatch_terminal(
            is_final=False, cancelled=False, error="connection reset by peer"
        )
        worker.inference_client.notify_session_end.assert_not_called()

    def _done_callback(self, *, returned: bool) -> MagicMock:
        """Drive the real done-callback on a mock worker with a not-yet-returned
        (or already-returned) credit context."""
        worker = MagicMock()
        worker.inference_client = MagicMock()
        worker.service_id = "worker-1"
        credit = MagicMock()
        credit.id = 7
        credit.x_correlation_id = "conv-done"
        ctx = MagicMock()
        ctx.returned = returned
        ctx.credit = credit
        task = MagicMock()
        task.cancelled.return_value = True
        Worker._on_credit_drop_message_task_done.__get__(worker)(task, ctx)
        return worker

    def test_done_callback_not_returned_branch_notifies_session_end(self) -> None:
        """The cancel-before-start path (task done, credit never returned) is a
        terminal disposition the finally block never sees, so the done-callback
        must fire the hook too."""
        worker = self._done_callback(returned=False)
        worker.inference_client.notify_session_end.assert_called_once_with("conv-done")

    def test_done_callback_already_returned_does_not_double_notify(self) -> None:
        """When the credit already returned, the finally block already notified;
        the done-callback short-circuits without a second hook call."""
        worker = self._done_callback(returned=True)
        worker.inference_client.notify_session_end.assert_not_called()


# --- Payload Bytes Fast Path Tests ---


@pytest.mark.asyncio
class TestPayloadBytesFastPath:
    """Branch coverage for Worker._try_payload_bytes_fast_path."""

    async def test_returns_false_when_not_payload_bytes(
        self, mock_worker, sample_credit_context
    ):
        mock_worker._is_payload_bytes = False
        mock_worker._dataset_client = AsyncMock()

        handled = await mock_worker._try_payload_bytes_fast_path(
            sample_credit_context, "x-req-1", None
        )

        assert handled is False
        mock_worker._dataset_client.get_payload_bytes.assert_not_called()

    async def test_returns_false_for_dag_descendants(self, mock_worker):
        mock_worker._is_payload_bytes = True
        mock_worker._dataset_client = AsyncMock()
        credit_context = CreditContext(
            credit=Credit(
                id=2,
                phase=CreditPhase.PROFILING,
                conversation_id="conv-dag",
                x_correlation_id="corr-dag",
                turn_index=0,
                num_turns=1,
                issued_at_ns=1000000,
                agent_depth=1,
                parent_correlation_id="parent-corr",
            ),
            drop_perf_ns=2000000,
        )

        handled = await mock_worker._try_payload_bytes_fast_path(
            credit_context, "x-req-2", None
        )

        assert handled is False
        mock_worker._dataset_client.get_payload_bytes.assert_not_called()

    async def test_returns_false_when_no_payload_for_turn(
        self, mock_worker, sample_credit_context
    ):
        """A missing per-turn payload defers to the normal session path."""
        mock_worker._is_payload_bytes = True
        mock_worker._dataset_client = AsyncMock()
        mock_worker._dataset_client.get_payload_bytes.return_value = None

        handled = await mock_worker._try_payload_bytes_fast_path(
            sample_credit_context, "x-req-3", None
        )

        assert handled is False
        mock_worker._dataset_client.get_payload_bytes.assert_awaited_once_with(
            "test-conv-123", 0
        )

    async def test_request_error_is_recorded_on_credit_context(
        self, mock_worker, sample_credit_context
    ):
        mock_worker._is_payload_bytes = True
        mock_worker._dataset_client = AsyncMock()
        mock_worker._dataset_client.get_payload_bytes.return_value = b'{"p": 1}'

        error_record = RequestRecord(
            timestamp_ns=1,
            start_perf_ns=1,
            end_perf_ns=2,
            error=ErrorDetails(message="boom", type="TestError"),
        )
        mock_worker.inference_client.send_request = AsyncMock(return_value=error_record)
        mock_worker._send_inference_result_message = AsyncMock()
        mock_worker._process_credit_with_session = AsyncMock()

        handled = await mock_worker._try_payload_bytes_fast_path(
            sample_credit_context, "x-req-4", None
        )

        assert handled is True
        assert sample_credit_context.error == error_record.error
        mock_worker._send_inference_result_message.assert_awaited_once_with(
            error_record
        )
        sent_request_info = mock_worker.inference_client.send_request.call_args.args[0]
        assert sent_request_info.payload_bytes == b'{"p": 1}'
        # The fast path handled the credit; the session path never ran.
        mock_worker._process_credit_with_session.assert_not_awaited()


# --- First Token Callback Factory Tests ---


@pytest.mark.asyncio
class TestMakeFirstTokenCallback:
    """Coverage for the REAL Worker._make_first_token_callback factory."""

    async def test_returns_none_when_prefill_concurrency_disabled(
        self, mock_worker, sample_credit_context
    ):
        mock_worker._prefill_concurrency_enabled = False
        assert mock_worker._make_first_token_callback(sample_credit_context) is None

    async def test_callback_skips_meaningless_content(
        self, monkeypatch, mock_worker, sample_credit_context
    ):
        mock_worker._prefill_concurrency_enabled = True
        mock_worker.credit_return_push_client = AsyncMock()
        setup_mock_endpoint(mock_worker, monkeypatch, None)

        callback = mock_worker._make_first_token_callback(sample_credit_context)
        assert callback is not None

        result = await callback(50_000_000, SSEMessage(perf_ns=100_000_000))

        assert result is False
        assert sample_credit_context.first_token_sent is False
        mock_worker.credit_return_push_client.send.assert_not_called()

    async def test_callback_sends_first_token_on_meaningful_content(
        self, monkeypatch, mock_worker, sample_credit_context
    ):
        mock_worker._prefill_concurrency_enabled = True
        mock_worker.credit_return_push_client = AsyncMock()
        setup_mock_endpoint(
            mock_worker,
            monkeypatch,
            ParsedResponse(perf_ns=100_000_000, data=TextResponseData(text="hi")),
        )

        callback = mock_worker._make_first_token_callback(sample_credit_context)
        assert callback is not None

        result = await callback(50_000_000, SSEMessage(perf_ns=100_000_000))

        assert result is True
        assert sample_credit_context.first_token_sent is True
        sent = mock_worker.credit_return_push_client.send.call_args.args[0]
        assert sent.credit_id == sample_credit_context.credit.id
        assert sent.phase_index == sample_credit_context.credit.phase_index
        assert sent.ttft_ns == 50_000_000
