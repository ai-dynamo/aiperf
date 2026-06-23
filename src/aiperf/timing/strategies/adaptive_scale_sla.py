# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SLA evaluation helpers for adaptive scale timing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from aiperf.timing.strategies.adaptive_scale_types import WindowStats

if TYPE_CHECKING:
    from aiperf.config.sweep.adaptive import SLAFilter


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
    def throughput_value(stats: WindowStats, stat: str) -> float:
        match stat:
            case "avg" | "min" | "max":
                return stats.throughput
        raise ValueError(f"Unsupported throughput SLA stat: {stat}")

    @staticmethod
    def goodput_ratio_value(stats: WindowStats, stat: str) -> float:
        match stat:
            case "avg" | "min" | "max":
                if stats.total == 0:
                    return 0.0
                return len(stats.samples) / stats.total
        raise ValueError(f"Unsupported goodput_ratio SLA stat: {stat}")

    def value(self, sla: SLAFilter, stats: WindowStats) -> float:
        match sla.metric_tag:
            case "request_latency":
                return self.request_latency_value(stats.samples, sla.stat)
            case "throughput" | "request_throughput" | "completed_request_throughput":
                return self.throughput_value(stats, sla.stat)
            case "goodput_ratio" | "success_rate" | "request_success_rate":
                return self.goodput_ratio_value(stats, sla.stat)
        raise ValueError(
            "adaptive_scale supports request_latency, request throughput, "
            "and goodput_ratio SLA metrics in this release, got "
            f"{sla.metric_tag!r}"
        )

    def values(
        self, sla_filters: list[SLAFilter], stats: WindowStats
    ) -> dict[str, float]:
        return {self.key(sla): self.value(sla, stats) for sla in sla_filters}

    def validate_filters(self, sla_filters: list[SLAFilter]) -> None:
        for sla in sla_filters:
            self.validate_single_filter(sla)

    @staticmethod
    def validate_single_filter(sla: SLAFilter) -> None:
        if sla.op not in {"lt", "le", "gt", "ge"}:
            raise ValueError(f"Unsupported SLA operator: {sla.op}")
        match sla.metric_tag:
            case "request_latency":
                if sla.stat not in {
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
                }:
                    raise ValueError(
                        f"Unsupported request_latency SLA stat: {sla.stat}"
                    )
            case "throughput" | "request_throughput" | "completed_request_throughput":
                if sla.stat not in {"avg", "min", "max"}:
                    raise ValueError(f"Unsupported throughput SLA stat: {sla.stat}")
            case "goodput_ratio" | "success_rate" | "request_success_rate":
                if sla.stat not in {"avg", "min", "max"}:
                    raise ValueError(f"Unsupported goodput_ratio SLA stat: {sla.stat}")
            case _:
                raise ValueError(
                    "adaptive_scale supports request_latency, request throughput, "
                    "and goodput_ratio SLA metrics in this release, got "
                    f"{sla.metric_tag!r}"
                )

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


def percentile_value(samples: list[int], percentile: float) -> float:
    """Return the linearly interpolated percentile for nanosecond samples."""
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
