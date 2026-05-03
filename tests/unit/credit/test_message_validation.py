# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for credit struct validation."""

import time

import msgspec
import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import (
    CreditReturn,
    FirstToken,
    WorkerToRouterMessage,
)
from aiperf.credit.structs import Credit, CreditContext

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def credit_factory():
    """Factory fixture for creating test credits with customizable parameters."""

    def _create(
        credit_id: int = 1,
        phase: CreditPhase = CreditPhase.PROFILING,
        turn_index: int = 0,
        num_turns: int = 1,
        conversation_id: str = "conv-1",
        x_correlation_id: str = "corr-1",
    ) -> Credit:
        return Credit(
            id=credit_id,
            phase=phase,
            turn_index=turn_index,
            num_turns=num_turns,
            conversation_id=conversation_id,
            x_correlation_id=x_correlation_id,
            issued_at_ns=time.time_ns(),
        )

    return _create


@pytest.fixture
def sample_credit(credit_factory) -> Credit:
    """Simple single-turn credit for basic tests."""
    return credit_factory()


# =============================================================================
# Credit Validation Tests
# =============================================================================


class TestCreditValidation:
    """Test validation logic for Credit struct."""

    @pytest.mark.parametrize(
        "turn_index,num_turns,expected_final",
        [(1, 3, False), (2, 3, True)],  # Sample: middle and final
    )
    def test_credit_is_final_turn(
        self, credit_factory, turn_index, num_turns, expected_final
    ):
        """Credit.is_final_turn correctly identifies final turns."""
        credit = credit_factory(turn_index=turn_index, num_turns=num_turns)
        assert credit.is_final_turn is expected_final


# =============================================================================
# FirstToken Validation Tests
# =============================================================================


class TestFirstTokenValidation:
    """Test validation logic for FirstToken struct."""

    def test_first_token_serialization_roundtrip(self):
        """FirstToken serializes/deserializes correctly via msgspec."""
        original = FirstToken(
            credit_id=99, phase=CreditPhase.WARMUP, ttft_ns=250_000_000
        )
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(original), type=FirstToken
        )

        assert decoded.credit_id == original.credit_id
        assert decoded.phase == original.phase
        assert decoded.ttft_ns == original.ttft_ns

    def test_first_token_in_union_type(self):
        """FirstToken can be decoded as part of WorkerToRouterMessage union."""
        first_token = FirstToken(
            credit_id=42, phase=CreditPhase.PROFILING, ttft_ns=150_000_000
        )
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(first_token), type=WorkerToRouterMessage
        )

        assert isinstance(decoded, FirstToken)
        assert decoded.credit_id == first_token.credit_id


# =============================================================================
# CreditReturn Validation Tests (Deadlock Prevention)
# =============================================================================


class TestCreditReturnValidation:
    """Test CreditReturn struct, including first_token_sent for deadlock prevention."""

    @pytest.mark.parametrize(
        "first_token_sent,cancelled,error",
        [(True, False, None), (False, True, None)],  # Sample: normal and cancelled
    )  # fmt: skip
    def test_credit_return_scenarios(
        self, sample_credit, first_token_sent, cancelled, error
    ):
        """CreditReturn handles various completion scenarios."""
        credit_return = CreditReturn(
            credit=sample_credit,
            first_token_sent=first_token_sent,
            cancelled=cancelled,
            error=error,
        )

        assert credit_return.first_token_sent is first_token_sent
        assert credit_return.cancelled is cancelled
        assert credit_return.error == error

    def test_credit_return_serialization_roundtrip(self, sample_credit):
        """CreditReturn preserves all fields through msgpack serialization."""
        original = CreditReturn(
            credit=sample_credit, first_token_sent=True, cancelled=False
        )
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(original), type=CreditReturn
        )

        assert decoded.first_token_sent == original.first_token_sent
        assert decoded.cancelled == original.cancelled


# =============================================================================
# CreditContext Validation Tests (Worker-side Tracking)
# =============================================================================


class TestCreditContextValidation:
    """Test CreditContext struct (mutable worker-side tracking)."""

    def test_credit_context_mutation(self, sample_credit):
        """CreditContext allows mutation for state tracking."""
        credit_context = CreditContext(
            credit=sample_credit,
            drop_perf_ns=time.perf_counter_ns(),
        )

        assert credit_context.first_token_sent is False
        credit_context.first_token_sent = True
        assert credit_context.first_token_sent is True

        credit_context.cancelled = True
        credit_context.returned = True
        assert credit_context.cancelled is True
        assert credit_context.returned is True
