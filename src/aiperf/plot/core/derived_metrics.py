# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derived metric calculators for AIPerf plotting."""

from typing import Any

from aiperf.common.constants import STAT_KEYS


class DerivedMetricCalculator:
    """
    Registry for derived metric calculations.

    Provides a centralized registry of functions that compute derived metrics
    from base metrics when GPU telemetry data is available. New derived metrics
    can be added by registering additional calculator functions.
    """

    @staticmethod
    def per_gpu_throughput(
        aggregated: dict[str, Any], gpu_count: int
    ) -> dict[str, Any] | None:
        """
        Calculate per-GPU throughput by dividing total throughput by GPU count.

        Args:
            aggregated: Aggregated metrics dictionary
            gpu_count: Total number of GPUs

        Returns:
            Dictionary with per-GPU throughput stats and unit, or None if base metric not found
        """
        throughput_data = None

        if (
            "metrics" in aggregated
            and "output_token_throughput" in aggregated["metrics"]
        ):
            throughput_data = aggregated["metrics"]["output_token_throughput"]
        elif "output_token_throughput" in aggregated:
            throughput_data = aggregated["output_token_throughput"]

        if throughput_data is None:
            return None

        per_gpu_data = {"unit": "tokens/sec/gpu"}

        if isinstance(throughput_data, dict):
            for key, value in throughput_data.items():
                if key == "unit":
                    continue
                if isinstance(value, int | float):
                    per_gpu_data[key] = value / gpu_count
        else:
            for stat_name in STAT_KEYS:
                stat_value = getattr(throughput_data, stat_name, None)
                if stat_value is not None and isinstance(stat_value, int | float):
                    per_gpu_data[stat_name] = stat_value / gpu_count

        return per_gpu_data


DERIVED_METRICS_REGISTRY: dict[str, callable] = {
    "output_token_throughput_per_gpu": DerivedMetricCalculator.per_gpu_throughput,
}
