# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure response-parsing, token-counting, and per-turn stats helpers for the
``aiperf chat`` command.

Separated from ``chat.py`` (which owns the HTTP/REPL I/O) so this layer stays
free of transport concerns and easy to unit test. Everything here is built on
the same metric classes ``aiperf profile`` uses, keeping the numbers
definitionally consistent.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from aiperf.common.constants import NANOS_PER_MILLIS, NANOS_PER_SECOND
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models.record_models import (
    ParsedResponse,
    ParsedResponseRecord,
    ReasoningResponseData,
    RequestRecord,
    TextResponseData,
    TokenCounts,
)
from aiperf.common.models.usage_models import Usage
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.input_sequence_length_metric import InputSequenceLengthMetric
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric
from aiperf.metrics.types.usage_cache_metrics import UsagePromptCacheReadTokensMetric

# Per-record metrics surfaced after each turn. Reusing the real metric classes
# (instead of recomputing TTFT/latency/OSL/ITL/cache inline) keeps the numbers
# definitionally identical to ``aiperf profile``.
_METRIC_TAGS = (
    TTFTMetric.tag,
    RequestLatencyMetric.tag,
    OutputSequenceLengthMetric.tag,
    InterTokenLatencyMetric.tag,
    InputSequenceLengthMetric.tag,
    UsagePromptCacheReadTokensMetric.tag,
)


def split_delta(delta: dict) -> tuple[str | None, str | None]:
    """Split one streaming ``delta`` into ``(content, reasoning)`` text pieces.

    Mirrors ``ChatEndpoint.extract_chat_response_data`` (``openai_chat.py``):
    a server emits reasoning either in a dedicated ``reasoning_content`` /
    ``reasoning`` field (when it runs a reasoning parser) or inline inside
    ``content`` as ``<think>...</think>``. Keep this in sync with that method
    so reasoning-token accounting matches ``aiperf profile``.
    """
    content = delta.get("content")
    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
    return content, reasoning


def make_response_data(
    content: str | None, reasoning: str | None
) -> ReasoningResponseData | TextResponseData | None:
    """Build the parsed data object for one chunk, typed the way ``profile``
    types it so client-side output/reasoning token buckets match."""
    if reasoning:
        return ReasoningResponseData(content=content or None, reasoning=reasoning)
    if content:
        return TextResponseData(text=content)
    return None


def count_tokens(encode: Callable[[str], list], text: str) -> int | None:
    """Token count for ``text`` (``None`` when empty), matching profile's
    empty-separator join + single encode in ``_compute_token_count``."""
    if not text:
        return None
    return len(encode(text))


def input_tokens_from_usage(usage: dict | None) -> int | None:
    """Server-reported prompt token count, used as ISL.

    Unlike output tokens (we have the full generated text), client-side input
    tokenization can't reproduce the server's chat template, so we take the
    server's exact ``prompt_tokens`` when usage is available.
    """
    if not usage:
        return None
    return Usage(usage).prompt_tokens


def build_record(
    *,
    model: str,
    start_ns: int,
    end_ns: int,
    timestamp_ns: int,
    responses: list[ParsedResponse],
    input_tokens: int | None,
    output_tokens: int | None,
    reasoning_tokens: int | None,
) -> ParsedResponseRecord:
    """Assemble the same ``ParsedResponseRecord`` shape the profile pipeline
    feeds to the metric classes."""
    request = RequestRecord(
        model_name=model,
        start_perf_ns=start_ns,
        timestamp_ns=timestamp_ns,
        end_perf_ns=end_ns,
    )
    return ParsedResponseRecord(
        request=request,
        responses=responses,
        token_counts=TokenCounts(
            input=input_tokens, output=output_tokens, reasoning=reasoning_tokens
        ),
    )


def compute_record_metrics(record: ParsedResponseRecord) -> MetricRecordDict:
    """Run the per-record metric classes in dependency order, mirroring the
    record stage of the profile metric pipeline.

    Metrics that do not apply to a turn (e.g. OSL with no tokens received)
    raise ``NoMetricValue`` and are skipped, exactly as in the pipeline.
    """
    metrics: MetricRecordDict = MetricRecordDict()
    for tag in MetricRegistry.create_dependency_order_for(_METRIC_TAGS):
        with contextlib.suppress(NoMetricValue):
            metrics[tag] = MetricRegistry.get_class(tag)().parse_record(record, metrics)
    return metrics


def format_stats(
    metrics: MetricRecordDict,
    reasoning_tokens: int | None,
    *,
    interactive: bool = False,
) -> str:
    """Render the per-turn stats block from computed metric values.

    TPS is the vLLM-familiar end-to-end rate (generated tokens / e2e latency,
    which includes TTFT); OSL and latency themselves come straight from the
    reused metric classes.

    In ``interactive`` mode (the REPL loop, with or without history) two extra
    lines are appended where they apply:
    - ITL / decode-TPS, which isolates decode speed from prefill and so stays
      comparable across turns even as resent history inflates TTFT.
    - Prompt-cache hit rate, which climbs as the shared conversation prefix
      gets served from the server's cache (low when ``--no-history`` is set,
      since no prefix is resent). Both are omitted when the server does not
      provide the underlying data.
    """
    ttft_ns = metrics.get(TTFTMetric.tag)
    latency_ns = metrics.get(RequestLatencyMetric.tag)
    osl = metrics.get(OutputSequenceLengthMetric.tag)
    if ttft_ns is None or latency_ns is None or not osl:
        return "(no tokens received)"

    latency_s = latency_ns / NANOS_PER_SECOND
    tps = osl / latency_s if latency_s > 0 else 0.0
    tokens_desc = f"{osl} tokens"
    if reasoning_tokens:
        tokens_desc += f", {reasoning_tokens} reasoning"
    lines = [
        f"TTFT: {ttft_ns / NANOS_PER_MILLIS:.2f} ms",
        f"TPS:  {tps:.2f} tokens/s ({tokens_desc} in {latency_s:.2f}s)",
    ]
    if interactive:
        lines.extend(_interactive_stat_lines(metrics))
    return "\n".join(lines)


def _interactive_stat_lines(metrics: MetricRecordDict) -> list[str]:
    """Build the interactive-only ITL and cache lines from computed metrics."""
    lines: list[str] = []

    # ITL requires >=2 output tokens; the metric raised NoMetricValue otherwise
    # and the tag is simply absent.
    itl_ns = metrics.get(InterTokenLatencyMetric.tag)
    if itl_ns:
        decode_tps = NANOS_PER_SECOND / itl_ns
        lines.append(
            f"ITL:  {itl_ns / NANOS_PER_MILLIS:.2f} ms/token "
            f"(decode {decode_tps:.2f} tokens/s)"
        )

    # Cache hit rate needs both the prompt-token count and the server-reported
    # cached-read count. Absent when the server doesn't report prompt-cache
    # usage (no prefix caching, or the field isn't surfaced).
    isl = metrics.get(InputSequenceLengthMetric.tag)
    cache_read = metrics.get(UsagePromptCacheReadTokensMetric.tag)
    if isl and cache_read is not None:
        rate = 100 * cache_read / isl
        lines.append(f"Cache: {cache_read}/{isl} prompt tokens cached ({rate:.1f}%)")
    return lines
