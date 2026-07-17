# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-trace peak-context helpers (weka + dynamo)."""

from __future__ import annotations

from aiperf.dataset.graph.adapters.dynamo.trace_reader import (
    AgentReplayMetrics,
    AgentRequestMetrics,
    AgentTraceRecord,
)
from aiperf.dataset.graph.adapters.shared.peak_context import (
    dynamo_tree_peak_context,
    weka_trace_peak_context,
)
from aiperf.dataset.graph.adapters.weka.trace_models import (
    WekaNormalRequest,
    WekaSubagentEntry,
    WekaTrace,
)


def _weka_trace_parent_and_subagent() -> WekaTrace:
    """Top-level parent (in=100/out=50) plus a subagent child (in=200/out=400)."""
    return WekaTrace(
        id="trace-1",
        models=["m"],
        block_size=16,
        hash_id_scope="local",
        requests=[
            WekaNormalRequest(
                t=0.0, type="n", model="m", input_length=100, output_length=50
            ),
            WekaSubagentEntry(
                t=1.0,
                type="subagent",
                agent_id="agent_001",
                subagent_type="Explore",
                status="completed",
                models=["m"],
                requests=[
                    WekaNormalRequest(
                        t=1.5,
                        type="n",
                        model="m",
                        input_length=200,
                        output_length=400,
                    ),
                ],
            ),
        ],
    )


def test_weka_peak_context_uncapped_uses_raw_output_lengths() -> None:
    trace = _weka_trace_parent_and_subagent()
    # max(100+50, 200+400) == 600.
    assert weka_trace_peak_context(trace, max_osl=None) == 600


def test_weka_peak_context_max_osl_caps_top_level_only() -> None:
    trace = _weka_trace_parent_and_subagent()
    # Parent leg capped to 100+min(50,10)=110, subagent child stays 200+400=600.
    # If the child were also capped it would drop to 200+10=210, so a peak of
    # 600 proves the subagent body is left uncapped.
    assert weka_trace_peak_context(trace, max_osl=10) == 600


def test_weka_peak_context_max_osl_reduces_dominant_top_level_leg() -> None:
    trace = WekaTrace(
        id="trace-2",
        models=["m"],
        block_size=16,
        hash_id_scope="local",
        requests=[
            WekaNormalRequest(
                t=0.0, type="n", model="m", input_length=100, output_length=50
            ),
        ],
    )
    assert weka_trace_peak_context(trace, max_osl=None) == 150
    # Only leg is top-level, so max_osl=10 caps it to 100+10.
    assert weka_trace_peak_context(trace, max_osl=10) == 110


def _dynamo_record(input_length: int, output_tokens: int) -> AgentTraceRecord:
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
                input_sequence_hashes=[1, 2, 3],
            ),
        ),
    )


def test_dynamo_peak_context_uses_replay_input_length_plus_output_tokens() -> None:
    records = [_dynamo_record(input_length=28832, output_tokens=174)]
    # 28832 + 174 == 29006.
    assert dynamo_tree_peak_context(records) == 29006


def test_dynamo_peak_context_peaks_over_all_records() -> None:
    records = [
        _dynamo_record(input_length=28832, output_tokens=174),
        _dynamo_record(input_length=100, output_tokens=10),
    ]
    assert dynamo_tree_peak_context(records) == 29006


def test_dynamo_peak_context_falls_back_to_input_tokens_then_one() -> None:
    with_input_tokens = AgentTraceRecord(
        schema="dynamo.request.trace.v1",
        event_type="request_end",
        event_time_unix_ms=0,
        request=AgentRequestMetrics(
            request_id="req-2", input_tokens=42, output_tokens=8
        ),
    )
    assert dynamo_tree_peak_context([with_input_tokens]) == 50

    request_free = AgentTraceRecord(
        schema="dynamo.request.trace.v1",
        event_type="tool_end",
        event_time_unix_ms=0,
    )
    # No request: input_length falls back to 1, output_tokens to 0.
    assert dynamo_tree_peak_context([request_free]) == 1
