# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SampledSession.build_seeded_turn (mid-conversation start turns)."""

from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.models import ConversationMetadata, TurnMetadata
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.conversation_source import SampledSession
from aiperf.timing.phase.credit_counter import CreditCounter
from aiperf.timing.strategies.request_rate import RequestRateStrategy


def _session(n_turns: int) -> SampledSession:
    return SampledSession(
        conversation_id="c",
        metadata=ConversationMetadata(
            conversation_id="c",
            turns=[TurnMetadata() for _ in range(n_turns)],
        ),
        x_correlation_id="x",
    )


class TestBuildSeededTurn:
    @pytest.mark.parametrize(
        "k",
        [
            param(1, id="k=1"),
            param(3, id="k=3"),
            param(4, id="k=last"),
        ],
    )  # fmt: skip
    def test_starts_at_k(self, k: int) -> None:
        turn = _session(5).build_seeded_turn(k)
        assert turn.turn_index == k
        assert turn.start_turn_index == k
        assert turn.num_turns == 5
        assert turn.is_session_start is True
        assert turn.is_final_turn is (k == 4)

    def test_clamps_to_last_turn(self) -> None:
        # A fraction-derived index can never consume the whole conversation.
        turn = _session(3).build_seeded_turn(10)
        assert turn.turn_index == 2
        assert turn.start_turn_index == 2
        assert turn.is_final_turn is True

    @pytest.mark.parametrize(
        "k",
        [
            param(0, id="zero"),
            param(-1, id="negative"),
        ],
    )  # fmt: skip
    def test_non_positive_falls_back_to_first_turn(self, k: int) -> None:
        turn = _session(4).build_seeded_turn(k)
        assert turn.turn_index == 0
        assert turn.start_turn_index == 0
        assert turn.is_session_start is True

    def test_empty_conversation_raises(self) -> None:
        with pytest.raises(ValueError, match="empty conversation"):
            _session(0).build_seeded_turn(1)

    def test_seeded_start_counts_only_remaining_wire_turns(self) -> None:
        counter = CreditCounter(
            CreditPhaseConfig(
                phase=CreditPhase.WARMUP,
                timing_mode=TimingMode.REQUEST_RATE,
            )
        )

        counter.increment_sent(_session(5).build_seeded_turn(3))

        assert counter.sent_sessions == 1
        assert counter.total_session_turns == 2


class TestTrajectoryStartSelection:
    @staticmethod
    def _strategy(min_ratio: float, max_ratio: float) -> RequestRateStrategy:
        strategy = object.__new__(RequestRateStrategy)
        strategy._config = CreditPhaseConfig(
            phase=CreditPhase.WARMUP,
            timing_mode=TimingMode.REQUEST_RATE,
            trajectory_start_min_ratio=min_ratio,
            trajectory_start_max_ratio=max_ratio,
        )
        strategy._trajectory_rng = MagicMock()
        return strategy

    def test_samples_ratio_within_configured_range(self) -> None:
        strategy = self._strategy(0.3, 0.7)
        strategy._trajectory_rng.uniform.return_value = 0.6

        turn = strategy._build_start_turn(_session(10))

        strategy._trajectory_rng.uniform.assert_called_once_with(0.3, 0.7)
        assert turn.turn_index == 6
        assert turn.start_turn_index == 6

    def test_disabled_range_starts_at_turn_zero_without_sampling(self) -> None:
        strategy = self._strategy(0.0, 0.0)

        turn = strategy._build_start_turn(_session(10))

        strategy._trajectory_rng.uniform.assert_not_called()
        assert turn.turn_index == 0
        assert turn.start_turn_index == 0
