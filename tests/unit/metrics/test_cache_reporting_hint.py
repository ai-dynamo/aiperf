# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the prompt-cache-reporting hint detectors.

Both detectors flag the same condition — token usage reported but no prompt-cache
read tokens — from two inputs: per-record metric maps (mid-run) and aggregated
results (end-of-run). A cache-read value of 0 means caching is on but had no
hits, and must NOT trigger the hint; only an absent value does.
"""

from aiperf.common.models import MetricResult
from aiperf.metrics.cache_reporting_hint import (
    usage_without_cache_in_record,
    usage_without_cache_in_results,
)

USAGE_TAG = "usage_prompt_tokens"
CACHE_TAG = "usage_prompt_cache_read_tokens"
TOTAL_USAGE_TAG = "total_usage_prompt_tokens"
TOTAL_CACHE_TAG = "total_usage_prompt_cache_read_tokens"


class TestUsageWithoutCacheInRecord:
    """Mid-run, per-record metric maps (``MetricRecordsData.results`` dicts)."""

    def test_usage_present_cache_absent_returns_true(self):
        assert usage_without_cache_in_record({USAGE_TAG: 1024}) is True

    def test_usage_present_cache_zero_returns_false(self):
        # Cache reporting on, no hits this request → reported, not missing.
        assert usage_without_cache_in_record({USAGE_TAG: 1024, CACHE_TAG: 0}) is False

    def test_usage_present_cache_nonzero_returns_false(self):
        assert usage_without_cache_in_record({USAGE_TAG: 1024, CACHE_TAG: 512}) is False

    def test_usage_absent_returns_false(self):
        # No usage reported at all → different problem; stay quiet.
        assert usage_without_cache_in_record({"output_sequence_length": 32}) is False

    def test_empty_record_returns_false(self):
        assert usage_without_cache_in_record({}) is False


class TestUsageWithoutCacheInResults:
    """End-of-run, aggregated ``MetricResult`` totals."""

    def _usage(self, **stats) -> MetricResult:
        return MetricResult(
            tag=TOTAL_USAGE_TAG,
            header="Total Usage Prompt Tokens",
            unit="tokens",
            **stats,
        )

    def _cache(self, **stats) -> MetricResult:
        return MetricResult(
            tag=TOTAL_CACHE_TAG,
            header="Total Usage Prompt Cache Read Tokens",
            unit="tokens",
            **stats,
        )

    def test_usage_present_cache_no_value_returns_true(self):
        # No-value metrics come through with count=0 and None stats.
        results = [self._usage(sum=58250, avg=1165.0, count=50), self._cache(count=0)]
        assert usage_without_cache_in_results(results) is True

    def test_usage_present_cache_tag_missing_returns_true(self):
        results = [self._usage(sum=58250, avg=1165.0, count=50)]
        assert usage_without_cache_in_results(results) is True

    def test_cache_reported_zero_returns_false(self):
        results = [
            self._usage(sum=58250, avg=1165.0, count=50),
            self._cache(sum=0, avg=0.0, count=50),
        ]
        assert usage_without_cache_in_results(results) is False

    def test_cache_reported_nonzero_returns_false(self):
        results = [
            self._usage(sum=58250, avg=1165.0, count=50),
            self._cache(sum=50176, avg=1003.5, count=50),
        ]
        assert usage_without_cache_in_results(results) is False

    def test_usage_absent_returns_false(self):
        assert usage_without_cache_in_results([self._cache(count=0)]) is False
