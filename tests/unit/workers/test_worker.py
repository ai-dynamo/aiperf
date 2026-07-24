# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pytest import param

from aiperf.common.enums import (
    ConversationContextMode,
    CreditPhase,
    MemoryMapFormat,
)
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.models import (
    Conversation,
    DatasetMetadata,
    ErrorDetails,
    MemoryMapClientMetadata,
    ParsedResponse,
    RequestRecord,
    SSEMessage,
    TextResponseData,
    Turn,
)
from aiperf.config.phases import ConcurrencyPhase
from aiperf.credit.structs import Credit, CreditContext
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.workers.worker import Worker, _phase_needs_first_token_callback
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
        parsed_responses = [
            response for response in parse_response_return if response is not None
        ]
    else:
        mock_endpoint.parse_response = Mock(return_value=parse_response_return)
        parsed_responses = (
            [parse_response_return] if parse_response_return is not None else []
        )
    mock_endpoint.extract_response_data = Mock(return_value=parsed_responses)
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
class TestWorkerResponseProcessing:
    """Verify optimized and protocol-compatible response processing."""

    async def test_process_responses_for_record_optimized_endpoint_uses_fast_path(
        self: "TestWorkerResponseProcessing",
        monkeypatch: pytest.MonkeyPatch,
        mock_worker: Worker,
    ) -> None:
        parsed_responses = [
            ParsedResponse(perf_ns=150, data=TextResponseData(text="response"))
        ]
        assistant_turn = Turn(role="assistant")
        process_responses = Mock(return_value=(parsed_responses, assistant_turn))
        extract_response_data = Mock()
        endpoint = SimpleNamespace(
            process_responses=process_responses,
            extract_response_data=extract_response_data,
        )
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", endpoint)
        record = RequestRecord(start_perf_ns=100)

        result = mock_worker._process_responses_for_record(
            record, capture_assistant_turn=True
        )

        assert result == (parsed_responses, assistant_turn)
        process_responses.assert_called_once_with(record, capture_assistant_turn=True)
        extract_response_data.assert_not_called()

    @pytest.mark.parametrize(
        ("capture_assistant_turn", "include_builder", "expected_builder_calls"),
        [
            param(False, True, 0, id="capture-disabled"),
            param(True, False, 0, id="builder-unavailable"),
            param(True, True, 1, id="builder-available"),
        ],
    )  # fmt: skip
    async def test_process_responses_for_record_protocol_endpoint_uses_extraction(
        self: "TestWorkerResponseProcessing",
        monkeypatch: pytest.MonkeyPatch,
        mock_worker: Worker,
        capture_assistant_turn: bool,
        include_builder: bool,
        expected_builder_calls: int,
    ) -> None:
        parsed_responses = [
            ParsedResponse(perf_ns=150, data=TextResponseData(text="response"))
        ]
        assistant_turn = Turn(role="assistant")
        extract_response_data = Mock(return_value=parsed_responses)
        build_assistant_turn = Mock(return_value=assistant_turn)
        endpoint_attributes = {"extract_response_data": extract_response_data}
        if include_builder:
            endpoint_attributes["build_assistant_turn"] = build_assistant_turn
        endpoint = SimpleNamespace(**endpoint_attributes)
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", endpoint)
        record = RequestRecord(start_perf_ns=100)

        result = mock_worker._process_responses_for_record(
            record, capture_assistant_turn=capture_assistant_turn
        )

        expected_turn = (
            assistant_turn if capture_assistant_turn and include_builder else None
        )
        assert result == (parsed_responses, expected_turn)
        extract_response_data.assert_called_once_with(record)
        assert build_assistant_turn.call_count == expected_builder_calls


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
        endpoint = setup_mock_endpoint(mock_worker, monkeypatch, parse_returns)
        record = RequestRecord(
            start_perf_ns=100,
            responses=[
                SSEMessage(perf_ns=150),
                SSEMessage(perf_ns=200),
                SSEMessage(perf_ns=250),
            ],
        )

        assert mock_worker._request_latency_ns_for_record(record) == 150
        endpoint.extract_response_data.assert_called_once_with(record)

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


# --- Payload Bytes Fast Path Tests ---


@pytest.mark.asyncio
class TestUniqueSessionPrefixWorker:
    async def test_full_context_requests_reuse_prefix_without_history_hydration(
        self, mock_worker, sample_credit_context
    ) -> None:
        mock_worker._unique_session_prefix_length = 8
        conversation = Conversation(
            session_id="full-context",
            context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
            turns=[
                Turn(raw_messages=[{"role": "user", "content": "complete-0"}]),
                Turn(raw_messages=[{"role": "user", "content": "complete-1"}]),
            ],
        )
        session = mock_worker.session_manager.create_and_store(
            "resampled-instance-1",
            conversation,
            num_turns=2,
        )

        session.hydrate_seed_history(1)
        assert session.turn_list == []
        session.advance_turn(0)
        first = mock_worker._create_request_info(
            x_request_id="request-0",
            credit_context=sample_credit_context,
            session=session,
        )
        session.advance_turn(1)
        second = mock_worker._create_request_info(
            x_request_id="request-1",
            credit_context=sample_credit_context,
            session=session,
        )

        assert first.user_context_message == second.user_context_message
        assert first.user_context_message
        assert second.turns == [conversation.turns[1]]

    async def test_payload_bytes_rejects_unique_prefix(
        self, monkeypatch, mock_worker, tmp_path: Path
    ) -> None:
        class FakeDatasetClient:
            def __init__(self, *, client_metadata) -> None:
                self.client_metadata = client_metadata

            async def initialize(self) -> None:
                return None

            async def stop(self) -> None:
                return None

        monkeypatch.setattr(
            "aiperf.workers.worker.plugins.get_class",
            lambda *_args, **_kwargs: FakeDatasetClient,
        )
        mock_worker._unique_session_prefix_length = 8
        notification = DatasetConfiguredNotification(
            service_id="dataset-manager",
            metadata=DatasetMetadata(
                sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            ),
            client_metadata=MemoryMapClientMetadata(
                format=MemoryMapFormat.PAYLOAD_BYTES,
                data_file_path=tmp_path / "data.mmap",
                index_file_path=tmp_path / "index.mmap",
            ),
        )

        with pytest.raises(ValueError, match="payload_bytes dataset"):
            await mock_worker._on_dataset_configured(notification)


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
