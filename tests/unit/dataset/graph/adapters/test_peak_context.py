# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-trace peak-context sizing for dynamo captures: replay preference, peaking, fallbacks."""

from __future__ import annotations

import math

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace_reader import (
    AgentReplayMetrics,
    AgentRequestMetrics,
    AgentTraceRecord,
)
from aiperf.dataset.graph.adapters.shared.peak_context import (
    dynamo_tree_peak_context,
    dynamo_tree_peak_input,
)


def _replay_record(input_length: int, output_tokens: int) -> AgentTraceRecord:
    """A ``request_end`` record whose replay block carries an explicit input length."""
    return AgentTraceRecord(
        schema="dynamo.request.trace.v1",
        event_type="request_end",
        event_time_unix_ms=0,
        request=AgentRequestMetrics(
            request_id="req-1",
            output_tokens=output_tokens,
            replay=AgentReplayMetrics(
                trace_block_size=16,
                input_length=input_length,
                # A representable recording: dynamo hashes every whole block plus
                # one partial tail, and ``_assert_block_aligned`` rejects any other
                # count. peak_context never reads them, but an impossible fixture
                # invites conclusions the real lowering could never produce.
                input_sequence_hashes=[1] * math.ceil(input_length / 16),
            ),
        ),
    )


def _tokens_only_record(input_tokens: int, output_tokens: int) -> AgentTraceRecord:
    """A ``request_end`` record with no replay block, so sizing must use ``input_tokens``."""
    return AgentTraceRecord(
        schema="dynamo.request.trace.v1",
        event_type="request_end",
        event_time_unix_ms=0,
        request=AgentRequestMetrics(
            request_id="req-2", input_tokens=input_tokens, output_tokens=output_tokens
        ),
    )


def _request_free_record() -> AgentTraceRecord:
    """A ``tool_end`` record with no ``request`` at all, exercising the last-resort floor."""
    return AgentTraceRecord(
        schema="dynamo.request.trace.v1",
        event_type="tool_end",
        event_time_unix_ms=0,
    )


@pytest.mark.parametrize(
    "records,expected",
    [
        param([_replay_record(28832, 174)], 29006, id="replay_input_length_plus_output"),
        param(
            [_replay_record(28832, 174), _replay_record(100, 10)],
            29006,
            id="peaks_over_all_records",
        ),
        # 42 input_tokens lowers to 2 virtual blocks of 16 (32 tokens sent).
        param([_tokens_only_record(42, 8)], 40, id="falls_back_to_input_tokens"),
        param([_request_free_record()], 1, id="request_free_falls_back_to_one"),
    ],
)  # fmt: skip
def test_dynamo_peak_context_sizes_records(
    records: list[AgentTraceRecord], expected: int
) -> None:
    """Peak context is the max over records of input length plus output tokens."""
    # Fallback ladder for input length: replay.input_length -> request.input_tokens -> 1;
    # output_tokens falls back to 0 when there is no request at all.
    assert dynamo_tree_peak_context(records) == expected


class TestBlockExactScreening:
    """Screening sizes the prompt that is SENT, not the raw recorded length."""

    def test_partial_tail_is_excluded_from_peak_context(self) -> None:
        """The dropped ``in % trace_block_size`` tail is not screened against."""
        # 100 tokens at block size 16 = 6 whole blocks (96) + a 4-token tail the
        # trie never emits.
        assert dynamo_tree_peak_context([_replay_record(100, 10)]) == 106

    def test_partial_tail_is_excluded_from_peak_input(self) -> None:
        """The ISL screen agrees with the context screen."""
        assert dynamo_tree_peak_input([_replay_record(100, 10)]) == 96

    def test_sub_block_prompt_keeps_its_recorded_length(self) -> None:
        """A prompt shorter than one block is sent whole by the small-prompt fallback."""
        assert dynamo_tree_peak_input([_replay_record(9, 0)]) == 9
