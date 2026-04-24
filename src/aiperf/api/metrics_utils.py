# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics utilities for the AIPerf API.

Provides helper functions for building labels and formatting metrics as JSON.
"""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import msgspec

from aiperf.api.prometheus_formatter import InfoLabels
from aiperf.common.models import MetricResult
from aiperf.common.models.base_models import _msgspec_enc_hook
from aiperf.config import AIPerfConfig
from aiperf.config.parsing import coerce_value


def build_info_labels(config: AIPerfConfig) -> InfoLabels:
    """Build info labels for metrics from AIPerfConfig.

    These labels identify the benchmark and are included in Prometheus metrics.

    Args:
        config: The benchmark configuration.

    Returns:
        Dictionary of label names to values for the info metric.
    """
    labels: InfoLabels = {}

    if config.benchmark_id:
        labels["benchmark_id"] = config.benchmark_id

    labels["model"] = ",".join(config.get_model_names())
    labels["endpoint_type"] = str(config.endpoint.type)
    labels["streaming"] = str(config.endpoint.streaming).lower()

    labels["config"] = config.model_dump(
        mode="json", exclude_none=True, exclude_unset=True
    )

    return labels


def format_metrics_json(
    metrics: list[MetricResult],
    info_labels: InfoLabels | None = None,
    benchmark_id: str | None = None,
) -> dict[str, Any]:
    """Format metrics as JSON.

    Args:
        metrics: List of MetricResult objects from realtime metrics.
        info_labels: Optional dict of labels for additional metadata.
        benchmark_id: Optional benchmark ID to include.

    Returns:
        Formatted metrics as a dictionary.
    """
    result: dict[str, Any] = {
        "aiperf_version": version("aiperf"),
    }

    if benchmark_id:
        result["benchmark_id"] = benchmark_id

    if info_labels:
        result.update(
            {
                key: coerce_value(value)
                for key, value in info_labels.items()
                if key not in ("config", "version")
            }
        )

    metrics_dict = {}
    for metric in metrics:
        payload = msgspec.to_builtins(metric, enc_hook=_msgspec_enc_hook)
        # Mirror the previous Pydantic `exclude_none=True, exclude={"tag"}`
        # semantics: drop the tag (used as the dict key) and any None values.
        metrics_dict[metric.tag] = {
            k: v for k, v in payload.items() if v is not None and k != "tag"
        }

    result["metrics"] = metrics_dict
    return result
