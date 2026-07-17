# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    Conversation,
    ParsedResponse,
    RequestRecord,
    SSEMessage,
    TextResponseData,
)
from aiperf.config.phases import ConcurrencyPhase
from aiperf.credit.structs import Credit, CreditContext
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


# --- Session-Routing Terminal Hook Tests ---


@pytest.mark.asyncio
class TestReleaseAndEvictForTerminal:
    """The terminal-eviction path fires the routing plugin's post-session hook
    (``InferenceClient.notify_session_end``) so stateful plugins release the
    session on ANY terminal outcome, even one abandoned before its final turn
    (e.g. cancellation)."""

    async def test_release_and_evict_for_terminal_notifies_session_end(
        self, mock_worker, sample_credit_context
    ):
        credit = sample_credit_context.credit
        mock_worker.inference_client.notify_session_end = Mock()

        mock_worker._release_and_evict_for_terminal(credit, credit.x_correlation_id)

        mock_worker.inference_client.notify_session_end.assert_called_once_with(
            credit.x_correlation_id
        )

    async def test_raising_on_session_end_does_not_block_eviction(
        self, mock_worker, sample_credit_context
    ):
        """A routing plugin whose on_session_end raises must not abort the
        worker's eviction: the InferenceClient hook swallows + warns, so the
        session still gets evicted."""
        from aiperf.workers.inference_client import InferenceClient

        credit = sample_credit_context.credit
        fake_client = MagicMock()
        fake_client._routing.on_session_end.side_effect = RuntimeError("plugin boom")
        fake_client._routing_mode = "dynamo_headers"
        # Drive the REAL hook logic (try/except-log) bound to the fake client.
        fake_client.notify_session_end = InferenceClient.notify_session_end.__get__(
            fake_client
        )
        mock_worker.inference_client = fake_client
        mock_worker.session_manager.get = Mock(return_value=None)
        mock_worker.session_manager.evict = Mock()

        # Must not raise despite the plugin fault.
        mock_worker._release_and_evict_for_terminal(credit, credit.x_correlation_id)

        mock_worker.session_manager.evict.assert_called_once_with(
            credit.x_correlation_id
        )
        fake_client._routing.on_session_end.assert_called_once_with(
            credit.x_correlation_id
        )
        fake_client.warning.assert_called_once()


class TestSessionRoutingTerminalHooks:
    """Every terminal disposition this codebase has reaches
    ``InferenceClient.notify_session_end`` so routing plugins get their
    idempotent post-session hook: the final-turn / cancellation eviction path
    and the done-callback cancel-before-start branch (which the finally block
    never sees)."""

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
        ctx.error = None
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


@pytest.mark.asyncio
class TestCreateRequestInfoFinality:
    """Touch #3: worker -> RequestInfo lineage-finality plumb.

    The Credit is a REAL struct (not a MagicMock, which would auto-create the
    attributes and mask a missed plumb) carrying both finality facts plus a
    tree root id.
    """

    async def test_create_request_info_plumbs_finality_from_credit(self, mock_worker):
        from aiperf.common.models import Turn

        credit_context = CreditContext(
            credit=Credit(
                id=1,
                phase=CreditPhase.PROFILING,
                conversation_id="test-conv",
                x_correlation_id="child-x",
                turn_index=0,
                num_turns=1,
                issued_at_ns=0,
                agent_depth=1,
                parent_correlation_id="root-x",
                root_correlation_id="root-x",
                is_parent_final=True,
                is_tree_final=True,
            ),
            drop_perf_ns=0,
        )

        # Session fields the builder reads, set to real values (the guard is on
        # the Credit, so a lightweight session stand-in is fine here).
        session = MagicMock()
        session.x_correlation_id = "child-x"
        session.conversation.session_id = "test-conv"
        session.turn_index = 0
        session.turn_list = [Turn()]
        session.url_index = None

        request_info = mock_worker._create_request_info(
            x_request_id="request-id",
            session=session,
            credit_context=credit_context,
        )

        assert request_info.is_parent_final is True
        assert request_info.is_tree_final is True
        # effective_root_correlation_id resolves to the stamped root id.
        assert request_info.root_correlation_id == "root-x"

    async def test_create_request_info_root_defaults_to_own_correlation(
        self, mock_worker
    ):
        """A root credit (no root_correlation_id) surfaces its own
        x_correlation_id as the record's tree root, with conservative finality.
        """
        from aiperf.common.models import Turn

        credit_context = CreditContext(
            credit=Credit(
                id=2,
                phase=CreditPhase.PROFILING,
                conversation_id="test-conv",
                x_correlation_id="root-x",
                turn_index=0,
                num_turns=1,
                issued_at_ns=0,
            ),
            drop_perf_ns=0,
        )
        session = MagicMock()
        session.x_correlation_id = "root-x"
        session.conversation.session_id = "test-conv"
        session.turn_index = 0
        session.turn_list = [Turn()]
        session.url_index = None

        request_info = mock_worker._create_request_info(
            x_request_id="request-id",
            session=session,
            credit_context=credit_context,
        )

        assert request_info.root_correlation_id == "root-x"
        assert request_info.is_parent_final is None
        assert request_info.is_tree_final is False
