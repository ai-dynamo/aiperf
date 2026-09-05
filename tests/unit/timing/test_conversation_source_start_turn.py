# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SampledSession.start_turn_index and build_turn_at_index."""

from unittest.mock import MagicMock

import pytest

from aiperf.timing.conversation_source import SampledSession


def _make_metadata_with_n_turns(n: int) -> MagicMock:
    md = MagicMock()
    md.turns = [MagicMock(has_forks=False) for _ in range(n)]
    return md


def test_sampled_session_default_start_turn_index_is_zero():
    sess = SampledSession(
        conversation_id="c1",
        metadata=_make_metadata_with_n_turns(5),
        x_correlation_id="cor1",
    )
    assert sess.start_turn_index == 0


def test_build_turn_at_index_returns_turn_with_requested_index():
    sess = SampledSession(
        conversation_id="c1",
        metadata=_make_metadata_with_n_turns(10),
        x_correlation_id="cor1",
    )
    turn = sess.build_turn_at_index(3)
    assert turn.turn_index == 3
    assert turn.conversation_id == "c1"
    assert turn.x_correlation_id == "cor1"


def test_build_turn_at_index_out_of_range_raises():
    sess = SampledSession(
        conversation_id="c1",
        metadata=_make_metadata_with_n_turns(3),
        x_correlation_id="cor1",
    )
    with pytest.raises(IndexError):
        sess.build_turn_at_index(3)


def test_build_first_turn_unchanged_for_existing_callers():
    sess = SampledSession(
        conversation_id="c1",
        metadata=_make_metadata_with_n_turns(5),
        x_correlation_id="cor1",
    )
    turn = sess.build_first_turn()
    assert turn.turn_index == 0
    assert turn.num_turns == 5
