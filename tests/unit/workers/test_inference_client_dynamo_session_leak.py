# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the legacy Dynamo ``_dynamo_opened_sessions`` leak.

Legacy session_control sends a non-idempotent ``open`` exactly once per
session, tracked in the per-worker ``_dynamo_opened_sessions`` set. The inline
discard only fires on a successful final turn, so a session abandoned BEFORE
its final turn (agentic_replay terminal-overflow recycle, cancellation) would
leak its entry forever on GC-disabled workers. ``discard_dynamo_session`` is
the worker terminal-eviction hook that bounds the set on ANY terminal outcome.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models.dataset_models import Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo
from aiperf.plugin.enums import EndpointType, TransportType
from aiperf.workers.inference_client import InferenceClient
from aiperf.workers.worker import Worker, _is_terminal_context_overflow


@pytest.fixture
def mock_http_transport_entry():
    entry = MagicMock()
    entry.name = TransportType.HTTP.value
    entry.metadata = {"url_schemes": ["http", "https"]}
    return entry


@pytest.fixture
def legacy_dynamo_endpoint() -> ModelEndpointInfo:
    """Endpoint with legacy Dynamo conversation-aware routing enabled."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_url="http://localhost:8000/v1/test",
            use_dynamo_conv_aware_routing=True,
            use_legacy_dynamo_session_control=True,
            dynamo_session_timeout_seconds=300,
        ),
    )


@pytest.fixture
def legacy_dynamo_client(legacy_dynamo_endpoint, mock_http_transport_entry):
    """An InferenceClient wired for the legacy Dynamo session_control path."""
    mock_transport = MagicMock()
    mock_endpoint = MagicMock()
    mock_endpoint.get_endpoint_headers.return_value = {}
    mock_endpoint.get_endpoint_params.return_value = {}
    mock_endpoint.format_payload.return_value = {"messages": []}

    def mock_get_class(protocol, name):
        if protocol == "endpoint":
            return lambda **kwargs: mock_endpoint
        if protocol == "transport":
            return lambda **kwargs: mock_transport
        raise ValueError(f"Unknown protocol: {protocol}")

    with (
        patch(
            "aiperf.workers.inference_client.plugins.get_class",
            side_effect=mock_get_class,
        ),
        patch(
            "aiperf.workers.inference_client.plugins.list_entries",
            return_value=[mock_http_transport_entry],
        ),
    ):
        return InferenceClient(
            model_endpoint=legacy_dynamo_endpoint, service_id="test-service-id"
        )


def _request_info(
    endpoint: ModelEndpointInfo, *, session_id: str, is_final_turn: bool
) -> RequestInfo:
    return RequestInfo(
        model_endpoint=endpoint,
        credit_phase=CreditPhase.PROFILING,
        credit_num=0,
        x_request_id="req-0",
        x_correlation_id=session_id,
        conversation_id=session_id,
        turn_index=0,
        turns=[Turn(texts=[Text(contents=["hi"])])],
        is_final_turn=is_final_turn,
    )


class TestDynamoOpenedSessionsLeak:
    @pytest.mark.asyncio
    async def test_open_then_discard_removes_abandoned_session(
        self, legacy_dynamo_client, legacy_dynamo_endpoint
    ):
        """A session opened but abandoned before its final turn is discarded.

        The non-final open populates ``_dynamo_opened_sessions``; the inline
        final-turn discard never runs because the session is recycled early.
        The worker terminal-eviction hook (``discard_dynamo_session``) must
        drop the entry so the set stays bounded.
        """
        legacy_dynamo_client.transport.send_request = AsyncMock(
            return_value=MagicMock()
        )

        # First (non-final) request emits 'open' and tracks the session.
        await legacy_dynamo_client._send_request_to_transport(
            _request_info(
                legacy_dynamo_endpoint, session_id="conv-1", is_final_turn=False
            )
        )
        assert "conv-1" in legacy_dynamo_client._dynamo_opened_sessions

        # Session abandoned before its final turn -> worker evicts it.
        legacy_dynamo_client.discard_dynamo_session("conv-1")
        assert "conv-1" not in legacy_dynamo_client._dynamo_opened_sessions

    def test_discard_is_idempotent_and_safe_for_untracked_session(
        self, legacy_dynamo_client
    ):
        """Discarding an unknown session is a no-op (terminal eviction may fire
        for sessions that never emitted a legacy open)."""
        legacy_dynamo_client.discard_dynamo_session("never-opened")
        legacy_dynamo_client.discard_dynamo_session("never-opened")
        assert legacy_dynamo_client._dynamo_opened_sessions == set()

    @pytest.mark.asyncio
    async def test_final_turn_still_discards_inline(
        self, legacy_dynamo_client, legacy_dynamo_endpoint
    ):
        """The existing 'open exactly once / close on final turn' behavior is
        intact: a final turn still discards inline without the worker hook."""
        legacy_dynamo_client.transport.send_request = AsyncMock(
            return_value=MagicMock()
        )

        await legacy_dynamo_client._send_request_to_transport(
            _request_info(
                legacy_dynamo_endpoint, session_id="conv-2", is_final_turn=False
            )
        )
        assert "conv-2" in legacy_dynamo_client._dynamo_opened_sessions

        await legacy_dynamo_client._send_request_to_transport(
            _request_info(
                legacy_dynamo_endpoint, session_id="conv-2", is_final_turn=True
            )
        )
        assert "conv-2" not in legacy_dynamo_client._dynamo_opened_sessions


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


class TestWorkerHandlesOverflowRecycleDiscard:
    """#14 real trigger: the agentic_replay terminal-overflow recycle returns a
    non-final turn carrying a context-overflow error and sends NO final/cancel
    credit, so the worker's terminal-disposition gate must still discard the
    legacy Dynamo open-session entry. Exercises the real worker gate
    (``Worker._handle_terminal_disposition``), not the discard bypass."""

    def _dispatch(self, *, is_final: bool, cancelled: bool, error) -> MagicMock:
        worker = MagicMock()
        worker.inference_client = MagicMock()
        credit = MagicMock(is_final_turn=is_final)
        ctx = MagicMock(cancelled=cancelled, error=error)
        Worker._handle_terminal_disposition.__get__(worker)(credit, ctx, "conv-X")
        return worker

    def test_nonfinal_overflow_discards_dynamo_session(self) -> None:
        worker = self._dispatch(is_final=False, cancelled=False, error=_OVERFLOW_BODY)
        worker.inference_client.discard_dynamo_session.assert_called_once_with("conv-X")

    def test_nonfinal_plain_error_does_not_discard(self) -> None:
        worker = self._dispatch(
            is_final=False, cancelled=False, error="connection reset by peer"
        )
        worker.inference_client.discard_dynamo_session.assert_not_called()
