# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``aiperf.cli_commands._chat_stats`` -- the pure
response-parsing, token-counting, and per-turn stats logic.

These verify parity with ``aiperf profile`` (reasoning vs output token
bucketing, metric reuse, stats formatting) without a live server.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.cli_commands._chat_stats import (
    build_record,
    compute_record_metrics,
    count_tokens,
    format_stats,
    input_tokens_from_usage,
    make_response_data,
    split_delta,
)
from aiperf.common.models.record_models import (
    ParsedResponse,
    ParsedResponseRecord,
    ReasoningResponseData,
    TextResponseData,
)
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric


@pytest.mark.parametrize(
    "delta,expected",
    [
        param({"content": "hi"}, ("hi", None), id="content_only"),
        param(
            {"reasoning_content": "think"},
            (None, "think"),
            id="reasoning_content_field",
        ),
        param({"reasoning": "think"}, (None, "think"), id="reasoning_field"),
        param(
            {"content": "hi", "reasoning_content": "think"},
            ("hi", "think"),
            id="both",
        ),
        param({}, (None, None), id="empty"),
    ],
)  # fmt: skip
def test_split_delta_mirrors_profile_extraction(
    delta: dict, expected: tuple[str | None, str | None]
) -> None:
    assert split_delta(delta) == expected


def test_make_response_data_reasoning_takes_precedence() -> None:
    data = make_response_data("answer", "thinking")
    assert isinstance(data, ReasoningResponseData)
    assert data.content == "answer"
    assert data.reasoning == "thinking"


def test_make_response_data_content_only_is_text() -> None:
    assert make_response_data("answer", None) == TextResponseData(text="answer")


def test_make_response_data_empty_is_none() -> None:
    assert make_response_data(None, None) is None


def _record(
    start_ns: int,
    perf_nss: list[int],
    *,
    output: int | None,
    reasoning: int | None = None,
    input_tokens: int | None = None,
    usage: dict | None = None,
) -> ParsedResponseRecord:
    """Build a record with content responses at the given timestamps.

    Pass ``usage`` (a raw OpenAI-style usage dict) to attach a server usage
    chunk, which feeds the cache-hit metric.
    """
    responses = [
        ParsedResponse(perf_ns=ns, data=TextResponseData(text="x")) for ns in perf_nss
    ]
    if usage is not None:
        responses.append(ParsedResponse(perf_ns=perf_nss[-1], usage=usage))
    return build_record(
        model="test-model",
        start_ns=start_ns,
        end_ns=perf_nss[-1],
        timestamp_ns=start_ns,
        responses=responses,
        input_tokens=input_tokens,
        output_tokens=output,
        reasoning_tokens=reasoning,
    )


def test_compute_record_metrics_matches_profile_formulas() -> None:
    # start=0, first token at 20ms, last at 1.2s -> TTFT=20ms, latency=1.2s
    record = _record(0, [20 * 1_000_000, 1_200 * 1_000_000], output=186)
    metrics = compute_record_metrics(record)

    assert metrics[TTFTMetric.tag] == 20 * 1_000_000
    assert metrics[RequestLatencyMetric.tag] == 1_200 * 1_000_000
    assert metrics[OutputSequenceLengthMetric.tag] == 186


def test_osl_includes_reasoning_tokens_like_profile() -> None:
    # OSL formula is output + reasoning (output_sequence_length_metric.py).
    record = _record(0, [10, 100], output=44, reasoning=142)
    metrics = compute_record_metrics(record)
    assert metrics[OutputSequenceLengthMetric.tag] == 186


def test_format_stats_renders_full_block() -> None:
    # With all data present the block is TTFT, TPS, ITL, and Cache -- one
    # ordered set of first-class per-turn metrics.
    # ttft=20ms, latency=520ms, osl=101 -> TPS=101/0.52, ITL=(520-20)/100=5ms.
    usage = {"prompt_tokens": 480, "prompt_tokens_details": {"cached_tokens": 412}}
    record = _record(
        0, [20 * 1_000_000, 520 * 1_000_000], output=101, input_tokens=480, usage=usage
    )
    stats = format_stats(compute_record_metrics(record), reasoning_tokens=None)
    assert stats == (
        "TTFT: 20.00 ms\n"
        "TPS:  194.23 tokens/s (101 tokens in 0.52s)\n"
        "ITL:  5.00 ms/token (decode 200.00 tokens/s)\n"
        "Cache: 412/480 prompt tokens cached (85.8%)"
    )


def test_format_stats_annotates_reasoning_tokens() -> None:
    # OSL is output + reasoning (44 + 142 = 186 total generated tokens).
    record = _record(0, [20 * 1_000_000, 1_200 * 1_000_000], output=44, reasoning=142)
    stats = format_stats(compute_record_metrics(record), reasoning_tokens=142)
    assert "186 tokens, 142 reasoning in 1.20s" in stats


def test_format_stats_omits_decode_line_for_single_token() -> None:
    # ITL needs >=2 output tokens; a single-token reply has no inter-token gap.
    record = _record(0, [20 * 1_000_000], output=1)
    stats = format_stats(compute_record_metrics(record), reasoning_tokens=None)
    assert "ITL:" not in stats


def test_format_stats_cache_line_omitted_without_usage() -> None:
    # No usage chunk -> no cache metric -> no Cache line.
    record = _record(0, [20 * 1_000_000, 520 * 1_000_000], output=101)
    stats = format_stats(compute_record_metrics(record), reasoning_tokens=None)
    assert "Cache:" not in stats


def test_input_tokens_from_usage_reads_prompt_tokens() -> None:
    assert input_tokens_from_usage({"prompt_tokens": 480}) == 480
    assert input_tokens_from_usage(None) is None


def test_count_tokens_encodes_nonempty_and_returns_none_for_empty() -> None:
    assert count_tokens(lambda s: list(s), "abc") == 3
    assert count_tokens(lambda s: list(s), "") is None


def test_format_stats_no_tokens_received() -> None:
    record = build_record(
        model="test-model",
        start_ns=0,
        end_ns=1,
        timestamp_ns=0,
        responses=[],
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
    )
    assert format_stats(compute_record_metrics(record), None) == "(no tokens received)"
