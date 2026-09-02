# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect when a server reports token usage but not prompt-cache reads.

A server that supports prefix/KV caching but hasn't been told to report
``usage.prompt_tokens_details.cached_tokens`` (vLLM ``--enable-prompt-tokens-details``,
SGLang ``--enable-cache-report``; TRT-LLM reports it by default) produces empty
prompt-cache metrics that look like a bug. These helpers detect that case so
AIPerf can hint the user both mid-run (per streamed record) and in the
end-of-run summary (aggregated results), sharing one message and one definition.
"""

from collections.abc import Mapping
from typing import Any

from aiperf.common.models import MetricResult
from aiperf.metrics.types.usage_cache_metrics import UsagePromptCacheReadTokensMetric
from aiperf.metrics.types.usage_metrics import UsagePromptTokensMetric
from aiperf.metrics.types.usage_total_metrics import (
    TotalUsagePromptCacheReadTokensMetric,
    TotalUsagePromptTokensMetric,
)

CACHE_REPORTING_HINT = (
    "Token usage is reported but no prompt-cache read tokens were seen "
    "(usage.prompt_tokens_details.cached_tokens absent). If your server uses "
    "prefix/KV caching, enable cache reporting so AIPerf can populate the "
    "prompt-cache metrics: vLLM --enable-prompt-tokens-details, SGLang "
    "--enable-cache-report (TRT-LLM reports it by default)."
)

_RESULT_VALUE_FIELDS = ("avg", "sum", "min", "max", "p50", "current")


def usage_without_cache_in_record(record: Mapping[str, Any]) -> bool:
    """Return True if a per-record metric map reports prompt tokens but no cache reads.

    Used mid-run on streamed ``MetricRecordsData.results`` dicts. A cache-read
    value of 0 (caching on, no hits) counts as reported; only an absent value
    (cache reporting off) triggers the hint.
    """
    return (
        record.get(UsagePromptTokensMetric.tag) is not None
        and record.get(UsagePromptCacheReadTokensMetric.tag) is None
    )


def usage_without_cache_in_results(results: list[MetricResult]) -> bool:
    """Return True if aggregated results report prompt tokens but no cache-read total.

    Used at end-of-run. No-value metrics flow through with ``count=0`` and all
    stats ``None``, so this checks the value fields rather than mere tag presence
    (a reported total of 0 counts as reported and does not trigger the hint).
    """
    by_tag = {result.tag: result for result in results}

    def _has_value(result: MetricResult | None) -> bool:
        return result is not None and any(
            getattr(result, field) is not None for field in _RESULT_VALUE_FIELDS
        )

    return _has_value(by_tag.get(TotalUsagePromptTokensMetric.tag)) and not _has_value(
        by_tag.get(TotalUsagePromptCacheReadTokensMetric.tag)
    )
