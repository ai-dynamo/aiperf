# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.aiperf_logger import AIPerfLogger
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

_logger = AIPerfLogger(__name__)


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

    # Warn only once per process when the first-chunk count is inconsistent with OSL.
    _mismatch_warned: bool = False

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """
        Calculates the Inter Token Latency (ITL) metric.
        """
        osl = record_metrics.get_or_raise(OutputSequenceLengthMetric)
        if osl < 2:  # type: ignore
            raise NoMetricValue(f"Output sequence length must be at least 2, got {osl}")

        # Subtract the first content chunk's real output-token count (the chunk that
        # set TTFT) instead of a hard-coded 1, so a server that bundles the first
        # chunk cannot inflate the decode-token count. The count is populated only
        # under --per-chunk-usage from server per-chunk usage; None (or a
        # non-positive lagging value) means "not reported", so fall back to
        # assuming one token in the first chunk -- the legacy `osl - 1`.
        first_chunk_tokens = (
            record.token_counts.first_content_chunk_tokens
            if record.token_counts is not None
            else None
        )
        if first_chunk_tokens is None:
            # Not reported (no --per-chunk-usage, or server sent no per-chunk usage):
            # legacy divisor, no warning -- this is the expected default path.
            decode_tokens = osl - 1  # type: ignore
        else:
            decode_tokens = osl - first_chunk_tokens  # type: ignore
            # A server-reported count that is non-positive or >= OSL is inconsistent
            # (e.g. per-chunk usage lagging a chunk, or the whole response in one
            # chunk). Degrade to `osl - 1` and warn once rather than silently
            # degrading a value the server actually reported.
            if first_chunk_tokens <= 0 or decode_tokens < 1:
                if not type(self)._mismatch_warned:
                    type(self)._mismatch_warned = True
                    _logger.warning(
                        lambda: (
                            f"Inter-token latency: server-reported first content chunk "
                            f"token count ({first_chunk_tokens}) is inconsistent with output "
                            f"sequence length ({osl}); falling back to (OSL - 1). Check "
                            f"--per-chunk-usage server support."
                        )
                    )
                decode_tokens = osl - 1  # type: ignore

        ttft = record_metrics.get_or_raise(TTFTMetric)
        request_latency = record_metrics.get_or_raise(RequestLatencyMetric)

        return (request_latency - ttft) / decode_tokens  # type: ignore
