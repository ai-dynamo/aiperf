# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SLA evaluation helpers for adaptive scale timing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from aiperf.timing.strategies.adaptive_scale_types import (
    WindowRequestSample,
    WindowStats,
)

if TYPE_CHECKING:
    from aiperf.config.sweep.adaptive import SLAFilter


LATENCY_STATS = {
    "avg",
    "min",
    "max",
    "p1",
    "p5",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
}
THROUGHPUT_STATS = {"avg", "min", "max"}
TTFT_METRICS = {"time_to_first_token", "ttft"}
ITL_METRICS = {"inter_token_latency", "itl", "tpot"}
QUALITY_METRICS = {"request_latency", *TTFT_METRICS, *ITL_METRICS}
GOODPUT_METRICS = {"goodput"}
OUTPUT_TOKEN_THROUGHPUT_METRICS = {"output_token_throughput"}
GOODPUT_RATIO_METRICS = {"goodput_ratio"}
SUCCESS_RATE_METRICS = {"success_rate", "request_success_rate"}
RATE_METRICS = {
    "throughput",
    "request_throughput",
    "completed_request_throughput",
    *GOODPUT_METRICS,
    *OUTPUT_TOKEN_THROUGHPUT_METRICS,
    *GOODPUT_RATIO_METRICS,
    *SUCCESS_RATE_METRICS,
    "error_rate",
    "request_error_rate",
    "cancellation_rate",
    "request_cancellation_rate",
}
SUPPORTED_METRICS_MESSAGE = (
    "adaptive_scale supports request_latency, time_to_first_token, "
    "inter_token_latency, request throughput, output_token_throughput, "
    "goodput, goodput_ratio, "
    "success_rate, error_rate, and cancellation_rate SLA metrics in this release"
)


def _rate_metric_name(metric_tag: str) -> str:
    if metric_tag in GOODPUT_METRICS:
        return "goodput"
    if metric_tag in OUTPUT_TOKEN_THROUGHPUT_METRICS:
        return "output_token_throughput"
    if metric_tag in GOODPUT_RATIO_METRICS:
        return "goodput_ratio"
    if metric_tag in SUCCESS_RATE_METRICS:
        return "success_rate"
    if metric_tag in {"error_rate", "request_error_rate"}:
        return "error_rate"
    if metric_tag in {"cancellation_rate", "request_cancellation_rate"}:
        return "cancellation_rate"
    return "throughput"


class AdaptiveScaleSLAEvaluator:
    """Evaluate adaptive-scale SLA filters against assessment windows."""

    @staticmethod
    def request_latency_value(samples: list[int], stat: str) -> float:
        if not samples:
            raise ValueError("request_latency SLA requires completed request samples")
        values_ms = [sample / 1_000_000 for sample in samples]
        match stat:
            case "avg":
                return sum(values_ms) / len(values_ms)
            case "min":
                return min(values_ms)
            case "max":
                return max(values_ms)
            case "p1" | "p5" | "p10" | "p25" | "p50" | "p75" | "p90" | "p95" | "p99":
                percentile = float(stat[1:])
                return percentile_value(samples, percentile) / 1_000_000
        raise ValueError(f"Unsupported request_latency SLA stat: {stat}")

    @staticmethod
    def time_to_first_token_value(samples: list[int], stat: str) -> float:
        if not samples:
            return math.inf
        return AdaptiveScaleSLAEvaluator.request_latency_value(samples, stat)

    @staticmethod
    def inter_token_latency_value(samples: list[float], stat: str) -> float:
        if not samples:
            return math.inf
        match stat:
            case "avg":
                return sum(samples) / len(samples) / 1_000_000
            case "min":
                return min(samples) / 1_000_000
            case "max":
                return max(samples) / 1_000_000
            case "p1" | "p5" | "p10" | "p25" | "p50" | "p75" | "p90" | "p95" | "p99":
                percentile = float(stat[1:])
                return percentile_value(samples, percentile) / 1_000_000
        raise ValueError(f"Unsupported inter_token_latency SLA stat: {stat}")

    @staticmethod
    def throughput_value(stats: WindowStats, stat: str) -> float:
        match stat:
            case "avg" | "min" | "max":
                return stats.throughput
        raise ValueError(f"Unsupported throughput SLA stat: {stat}")

    @staticmethod
    def output_token_throughput_value(stats: WindowStats, stat: str) -> float:
        match stat:
            case "avg" | "min" | "max":
                return stats.output_token_throughput
        raise ValueError(f"Unsupported output_token_throughput SLA stat: {stat}")

    def throughput_family_value(self, sla: SLAFilter, stats: WindowStats) -> float:
        if sla.metric_tag in OUTPUT_TOKEN_THROUGHPUT_METRICS:
            return self.output_token_throughput_value(stats, sla.stat)
        return self.throughput_value(stats, sla.stat)

    @staticmethod
    def success_rate_value(stats: WindowStats, stat: str) -> float:
        match stat:
            case "avg" | "min" | "max":
                if stats.total == 0:
                    return 0.0
                return len(stats.samples) / stats.total
        raise ValueError(f"Unsupported success_rate SLA stat: {stat}")

    @staticmethod
    def _per_request_quality_value(
        request: WindowRequestSample, metric_tag: str
    ) -> float | None:
        if metric_tag == "request_latency":
            return request.request_latency_ns / 1_000_000
        if metric_tag in TTFT_METRICS:
            if request.ttft_ns is None:
                return None
            return request.ttft_ns / 1_000_000
        if metric_tag in ITL_METRICS:
            if request.inter_token_latency_ns is None:
                return None
            return request.inter_token_latency_ns / 1_000_000
        return None

    @classmethod
    def _good_request_count(
        cls, stats: WindowStats, sla_filters: list[SLAFilter]
    ) -> int:
        quality_filters = [
            sla for sla in sla_filters if sla.metric_tag in QUALITY_METRICS
        ]
        if not quality_filters:
            raise ValueError(
                "quality goodput SLA requires at least one request_latency, "
                "time_to_first_token, or inter_token_latency quality filter"
            )
        good_count = 0
        for request in stats.requests:
            request_passed = True
            for sla in quality_filters:
                observed = cls._per_request_quality_value(request, sla.metric_tag)
                if observed is None or not cls.passes_single(sla, observed):
                    request_passed = False
                    break
            if request_passed:
                good_count += 1
        return good_count

    @classmethod
    def goodput_value(
        cls, stats: WindowStats, stat: str, sla_filters: list[SLAFilter]
    ) -> float:
        match stat:
            case "avg" | "min" | "max":
                pass
            case _:
                raise ValueError(f"Unsupported goodput SLA stat: {stat}")
        if stats.elapsed_sec <= 0:
            return 0.0
        good_count = cls._good_request_count(stats, sla_filters)
        return good_count / stats.elapsed_sec

    @classmethod
    def goodput_ratio_value(
        cls, stats: WindowStats, stat: str, sla_filters: list[SLAFilter]
    ) -> float:
        match stat:
            case "avg" | "min" | "max":
                pass
            case _:
                raise ValueError(f"Unsupported goodput_ratio SLA stat: {stat}")
        if stats.total == 0:
            return 0.0
        good_count = cls._good_request_count(stats, sla_filters)
        return good_count / stats.total

    @staticmethod
    def error_rate_value(stats: WindowStats, stat: str) -> float:
        match stat:
            case "avg" | "min" | "max":
                if stats.total == 0:
                    return 0.0
                return stats.errors / stats.total
        raise ValueError(f"Unsupported error_rate SLA stat: {stat}")

    @staticmethod
    def cancellation_rate_value(stats: WindowStats, stat: str) -> float:
        match stat:
            case "avg" | "min" | "max":
                if stats.total == 0:
                    return 0.0
                return stats.cancelled / stats.total
        raise ValueError(f"Unsupported cancellation_rate SLA stat: {stat}")

    def value(
        self,
        sla: SLAFilter,
        stats: WindowStats,
        sla_filters: list[SLAFilter] | None = None,
    ) -> float:
        match sla.metric_tag:
            case "request_latency":
                return self.request_latency_value(stats.samples, sla.stat)
            case metric if metric in TTFT_METRICS:
                return self.time_to_first_token_value(stats.ttfts, sla.stat)
            case metric if metric in ITL_METRICS:
                return self.inter_token_latency_value(stats.itls, sla.stat)
            case (
                "throughput"
                | "request_throughput"
                | "completed_request_throughput"
                | "output_token_throughput"
            ):
                return self.throughput_family_value(sla, stats)
            case metric if metric in GOODPUT_METRICS:
                return self.goodput_value(stats, sla.stat, sla_filters or [])
            case metric if metric in GOODPUT_RATIO_METRICS:
                return self.goodput_ratio_value(stats, sla.stat, sla_filters or [])
            case metric if metric in SUCCESS_RATE_METRICS:
                return self.success_rate_value(stats, sla.stat)
            case "error_rate" | "request_error_rate":
                return self.error_rate_value(stats, sla.stat)
            case "cancellation_rate" | "request_cancellation_rate":
                return self.cancellation_rate_value(stats, sla.stat)
        raise ValueError(f"{SUPPORTED_METRICS_MESSAGE}, got {sla.metric_tag!r}")

    def values(
        self, sla_filters: list[SLAFilter], stats: WindowStats
    ) -> dict[str, float]:
        return {
            self.key(sla): self.value(sla, stats, sla_filters) for sla in sla_filters
        }

    def validate_filters(self, sla_filters: list[SLAFilter]) -> None:
        has_quality_goodput = any(
            sla.metric_tag in GOODPUT_METRICS | GOODPUT_RATIO_METRICS
            for sla in sla_filters
        )
        has_quality_filter = any(
            sla.metric_tag in QUALITY_METRICS for sla in sla_filters
        )
        if has_quality_goodput and not has_quality_filter:
            raise ValueError(
                "quality goodput SLA requires at least one request_latency, "
                "time_to_first_token, or inter_token_latency quality filter"
            )
        for sla in sla_filters:
            self.validate_single_filter(sla)

    @staticmethod
    def validate_single_filter(sla: SLAFilter) -> None:
        if sla.op not in {"lt", "le", "gt", "ge"}:
            raise ValueError(f"Unsupported SLA operator: {sla.op}")
        allowed_stats, metric_name = AdaptiveScaleSLAEvaluator._stat_family_for_metric(
            sla.metric_tag
        )
        if sla.stat not in allowed_stats:
            raise ValueError(f"Unsupported {metric_name} SLA stat: {sla.stat}")

    @staticmethod
    def _stat_family_for_metric(metric_tag: str) -> tuple[set[str], str]:
        if metric_tag == "request_latency":
            return LATENCY_STATS, "request_latency"
        if metric_tag in TTFT_METRICS:
            return LATENCY_STATS, "time_to_first_token"
        if metric_tag in ITL_METRICS:
            return LATENCY_STATS, "inter_token_latency"
        if metric_tag in RATE_METRICS:
            return THROUGHPUT_STATS, _rate_metric_name(metric_tag)
        raise ValueError(f"{SUPPORTED_METRICS_MESSAGE}, got {metric_tag!r}")

    @staticmethod
    def key(sla: SLAFilter) -> str:
        return f"{sla.metric_tag}:{sla.stat}:{sla.op}:{sla.threshold:g}"

    def passes(self, sla_filters: list[SLAFilter], observed: dict[str, float]) -> bool:
        return all(
            self.passes_single(sla, observed[self.key(sla)]) for sla in sla_filters
        )

    @staticmethod
    def passes_single(sla: SLAFilter, observed: float) -> bool:
        match sla.op:
            case "lt":
                return observed < sla.threshold
            case "le":
                return observed <= sla.threshold
            case "gt":
                return observed > sla.threshold
            case "ge":
                return observed >= sla.threshold
        raise ValueError(f"Unsupported SLA operator: {sla.op}")


def percentile_value(samples: list[int | float], percentile: float) -> float:
    """Return the linearly interpolated percentile for nanosecond samples."""
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
    if not samples:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percentile / 100) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


# Backward-compatible alias for existing unit tests and internal imports.
_percentile = percentile_value
