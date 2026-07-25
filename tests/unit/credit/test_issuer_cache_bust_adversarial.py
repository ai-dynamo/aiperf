# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial coverage for cache_bust field propagation through CreditIssuer."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import Credit, TurnToSend


@pytest.fixture
def mock_stop_checker():
    mock = MagicMock()
    mock.can_send_any_turn = MagicMock(return_value=True)
    mock.can_start_new_session = MagicMock(return_value=True)
    mock.can_send_child_turn = MagicMock(return_value=True)
    return mock


@pytest.fixture
def mock_progress():
    mock = MagicMock()
    mock.increment_sent = MagicMock(return_value=(1, False))
    mock.freeze_sent_counts = MagicMock()
    mock.all_credits_sent_event = asyncio.Event()
    return mock


@pytest.fixture
def mock_concurrency():
    mock = MagicMock()
    mock.acquire_session_slot = AsyncMock(return_value=True)
    mock.acquire_prefill_slot = AsyncMock(return_value=True)
    mock.release_session_slot = MagicMock()
    return mock


@pytest.fixture
def mock_router():
    mock = MagicMock()
    mock.send_credit = AsyncMock()
    return mock


@pytest.fixture
def mock_cancellation():
    mock = MagicMock()
    mock.next_cancellation_delay_ns = MagicMock(return_value=None)
    return mock


@pytest.fixture
def mock_lifecycle():
    mock = MagicMock()
    mock.time_left_in_seconds = MagicMock(return_value=None)
    mock.phase_start_ns = 0
    mock.started_at_ns = time.time_ns()
    mock.started_at_perf_ns = time.perf_counter_ns()
    return mock


@pytest.fixture
def credit_issuer(
    mock_stop_checker,
    mock_progress,
    mock_concurrency,
    mock_router,
    mock_cancellation,
    mock_lifecycle,
):
    return CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=mock_stop_checker,
        progress=mock_progress,
        concurrency_manager=mock_concurrency,
        credit_router=mock_router,
        cancellation_policy=mock_cancellation,
        lifecycle=mock_lifecycle,
    )


async def test_issue_credit_propagates_cache_bust_marker_and_target(
    credit_issuer, mock_router
):
    """A TurnToSend's cache_bust_marker and cache_bust_target surface verbatim on the issued Credit."""
    turn = TurnToSend(
        conversation_id="conv-x",
        x_correlation_id="corr-x",
        turn_index=0,
        num_turns=2,
        cache_bust_marker="[rid:abc]\n\n",
        cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
    )

    await credit_issuer.issue_credit(turn)

    sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
    assert sent_credit.cache_bust_marker == "[rid:abc]\n\n"
    assert sent_credit.cache_bust_target == CacheBustTarget.SYSTEM_PREFIX


async def test_issue_credit_default_cache_bust_when_turn_unset(
    credit_issuer, mock_router
):
    """A TurnToSend with unset cache_bust_* fields yields a Credit with marker=None and target=NONE."""
    turn = TurnToSend(
        conversation_id="conv-y",
        x_correlation_id="corr-y",
        turn_index=0,
        num_turns=1,
    )

    await credit_issuer.issue_credit(turn)

    sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
    assert sent_credit.cache_bust_marker is None
    assert sent_credit.cache_bust_target == CacheBustTarget.NONE


async def test_issue_credit_msgpack_roundtrip_preserves_cache_bust_through_zmq_seam(
    credit_issuer, mock_router
):
    """A Credit's cache_bust fields survive msgpack encode + decode unchanged across the ZMQ seam."""
    turn = TurnToSend(
        conversation_id="conv-rt",
        x_correlation_id="corr-rt",
        turn_index=0,
        num_turns=2,
        cache_bust_marker="[rid:roundtrip01]\n\n",
        cache_bust_target=CacheBustTarget.SYSTEM_SUFFIX,
    )

    await credit_issuer.issue_credit(turn)
    sent_credit: Credit = mock_router.send_credit.call_args.kwargs["credit"]

    encoded = msgspec.msgpack.encode(sent_credit)
    decoded = msgspec.msgpack.decode(encoded, type=Credit)

    assert decoded.cache_bust_marker == "[rid:roundtrip01]\n\n"
    assert decoded.cache_bust_target == CacheBustTarget.SYSTEM_SUFFIX
    assert decoded.conversation_id == "conv-rt"
    assert decoded.x_correlation_id == "corr-rt"
