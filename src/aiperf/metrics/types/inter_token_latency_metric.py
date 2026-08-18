# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric


class InterTokenLatencyMetric(BaseRecordMetric[float]):
    """
    Post Processor for calculating Inter Token Latency (ITL) metric.

    Formula:
        Inter Token Latency = (Request Latency - Time to First Token) / (Output Sequence Length - First Content Chunk Tokens)

    The decode window ``Request Latency - Time to First Token`` covers the tokens
    that arrive AFTER the first content chunk, so the divisor subtracts that chunk's
    token count rather than assuming exactly one token arrived first. When a server
    bundles several tokens into the first streamed chunk (e.g. TRT-LLM
    ``stream-interval``), assuming one token over-counts the decode tokens and
    inflates TPS/user (``1 / ITL``). The first-chunk count comes from the server's
    per-chunk usage (``--per-chunk-usage``); when it is unavailable the divisor falls
    back to subtracting one, which is exact for servers that stream one token per chunk.
    """

    tag = "inter_token_latency"
    header = "Inter Token Latency"
    short_header = "ITL"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 400
    flags = (
        MetricFlags.STREAMING_TOKENS_ONLY
        | MetricFlags.PERCENTILE_INCLUDES_FAILED_REQUESTS
    )
    required_metrics = {
        RequestLatencyMetric.tag,
        TTFTMetric.tag,
        OutputSequenceLengthMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """
        Calculates the Inter Token Latency (ITL) metric.
        """
        osl = record_metrics.get_or_raise(OutputSequenceLengthMetric)

        # Subtract the first content chunk's real token count (the chunk that set
        # TTFT) instead of a hard-coded 1, so a server that bundles the first chunk
        # cannot inflate the decode-token count. Falls back to 1 when the per-chunk
        # count is absent (server did not report per-chunk usage).
        first_chunk_tokens = (
            record.token_counts.first_content_chunk_tokens
            if record.token_counts is not None
            and record.token_counts.first_content_chunk_tokens
            else 1
        )
        decode_tokens = osl - first_chunk_tokens  # type: ignore
        if decode_tokens < 1:
            raise NoMetricValue(
                f"No decode-phase tokens after the first content chunk "
                f"(OSL={osl}, first chunk={first_chunk_tokens}); ITL is undefined."
            )

        ttft = record_metrics.get_or_raise(TTFTMetric)
        request_latency = record_metrics.get_or_raise(RequestLatencyMetric)

        return (request_latency - ttft) / decode_tokens  # type: ignore
