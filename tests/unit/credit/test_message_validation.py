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

    @pytest.fixture
    def sample_first_token(self) -> FirstToken:
        return FirstToken(
            credit_id=42,
            phase=CreditPhase.PROFILING,
            ttft_ns=150_000_000,
        )

    def test_first_token_creation(self, sample_first_token):
        """FirstToken can be created with required fields."""
        assert sample_first_token.credit_id == 42
        assert sample_first_token.phase == CreditPhase.PROFILING
        assert sample_first_token.ttft_ns == 150_000_000

    def test_first_token_immutable(self, sample_first_token):
        """FirstToken struct is frozen/immutable."""
        with pytest.raises(AttributeError):
            sample_first_token.credit_id = 2  # type: ignore[misc]

    def test_first_token_serialization_roundtrip(self):
        """FirstToken can be serialized and deserialized via msgspec."""
        original = FirstToken(
            credit_id=99, phase=CreditPhase.WARMUP, ttft_ns=250_000_000
        )
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(original), type=FirstToken
        )

        assert decoded.credit_id == original.credit_id
        assert decoded.phase == original.phase
        assert decoded.ttft_ns == original.ttft_ns

    def test_first_token_in_union_type(self, sample_first_token):
        """FirstToken can be decoded as part of WorkerToRouterMessage union."""
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(sample_first_token), type=WorkerToRouterMessage
        )

        assert isinstance(decoded, FirstToken)
        assert decoded.credit_id == sample_first_token.credit_id

    @pytest.mark.parametrize("phase", [CreditPhase.WARMUP, CreditPhase.PROFILING])
    def test_first_token_supports_phases(self, phase):
        """FirstToken works with all credit phases."""
        first_token = FirstToken(credit_id=1, phase=phase, ttft_ns=50_000_000)
        assert first_token.phase == phase

    def test_first_token_zero_ttft(self):
        """FirstToken accepts zero ttft_ns (edge case)."""
        first_token = FirstToken(credit_id=1, phase=CreditPhase.PROFILING, ttft_ns=0)
        assert first_token.ttft_ns == 0


# =============================================================================
# CreditReturn Validation Tests (Deadlock Prevention)
# =============================================================================


class TestCreditReturnValidation:
    """Test CreditReturn struct, including first_token_sent for deadlock prevention."""

    def test_credit_return_defaults(self, sample_credit):
        """CreditReturn has expected default values."""
        credit_return = CreditReturn(credit=sample_credit)

        assert credit_return.first_token_sent is False
        assert credit_return.cancelled is False
        assert credit_return.error is None

    @pytest.mark.parametrize(
        "first_token_sent,cancelled,error,description",
        [
            (True, False, None, "streaming_success"),       # Normal streaming completion
            (False, False, None, "non_streaming"),          # Non-streaming request (deadlock case)
            (False, True, None, "cancelled_before_ttft"),   # Cancelled before first token (deadlock case)
            (False, False, "Connection timeout", "error"),  # Error before first token (deadlock case)
            (True, True, None, "cancelled_after_ttft"),     # Cancelled after first token
        ],
    )  # fmt: skip
    def test_credit_return_scenarios(
        self, sample_credit, first_token_sent, cancelled, error, description
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

    def test_credit_return_in_union_type(self, sample_credit):
        """CreditReturn can be decoded as part of WorkerToRouterMessage union."""
        credit_return = CreditReturn(credit=sample_credit, first_token_sent=True)
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(credit_return), type=WorkerToRouterMessage
        )

        assert isinstance(decoded, CreditReturn)
        assert decoded.first_token_sent is True

    def test_credit_return_immutable(self, sample_credit):
        """CreditReturn struct is frozen/immutable."""
        credit_return = CreditReturn(credit=sample_credit)

        with pytest.raises(AttributeError):
            credit_return.first_token_sent = True  # type: ignore[misc]


# =============================================================================
# CreditContext Validation Tests (Worker-side Tracking)
# =============================================================================


class TestCreditContextValidation:
    """Test CreditContext struct (mutable worker-side tracking)."""

    @pytest.fixture
    def credit_context(self, sample_credit) -> CreditContext:
        return CreditContext(
            credit=sample_credit,
            drop_perf_ns=time.perf_counter_ns(),
        )

    def test_credit_context_defaults(self, credit_context):
        """CreditContext has expected default values."""
        assert credit_context.first_token_sent is False
        assert credit_context.cancelled is False
        assert credit_context.returned is False
        assert credit_context.error is None

    def test_credit_context_mutable(self, credit_context):
        """CreditContext allows mutation (worker tracks state changes)."""
        assert credit_context.first_token_sent is False
        credit_context.first_token_sent = True
        assert credit_context.first_token_sent is True

    @pytest.mark.parametrize(
        "first_token_sent,cancelled,error,description",
        [
            (True, False, None, "streaming_success"),       # Normal streaming completion
            (False, True, None, "cancelled_before_ttft"),   # Cancelled before first token (deadlock case)
            (False, False, "HTTP 500", "error_before_ttft"),  # Error before first token (deadlock case)
        ],
    )  # fmt: skip
    def test_credit_context_state_transitions(
        self, credit_context, first_token_sent, cancelled, error, description
    ):
        """CreditContext tracks various state transitions."""
        if first_token_sent:
            credit_context.first_token_sent = True
        if cancelled:
            credit_context.cancelled = True
        if error:
            credit_context.error = error
        credit_context.returned = True

        assert credit_context.first_token_sent is first_token_sent
        assert credit_context.cancelled is cancelled
        assert credit_context.error == error
        assert credit_context.returned is True
