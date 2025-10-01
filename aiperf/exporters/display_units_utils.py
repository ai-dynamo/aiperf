# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from urllib.parse import urlparse

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.exceptions import MetricUnitError
from aiperf.common.models import MetricResult
from aiperf.common.types import MetricTagT
from aiperf.metrics.metric_registry import MetricRegistry

_logger = AIPerfLogger(__name__)
STAT_KEYS = [
    "avg",
    "min",
    "max",
    "p1",
    "p5",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "std",
]


def normalize_endpoint_display(url: str) -> str:
    """Normalize endpoint URL for display by removing scheme and trimming /metrics suffix.

    Args:
        url: The full URL to normalize (e.g., "https://host:9400/api/metrics")

    Returns:
        Normalized display string with netloc and trimmed path (e.g., "host:9400/api")
    """
    parsed = urlparse(url)
    path = parsed.path

    if path.endswith("/metrics"):
        path = path[: -len("/metrics")]

    display = parsed.netloc
    if path:
        display += path

    return display


def to_display_unit(result: MetricResult, registry: MetricRegistry) -> MetricResult:
    """
    Return a new MetricResult converted to its display unit (if different).
    """
    metric_cls = registry.get_class(result.tag)
    if result.unit and result.unit != metric_cls.unit.value:
        _logger.error(
            f"Metric {result.tag} has a unit ({result.unit}) that does not match the expected unit ({metric_cls.unit.value}). "
            f"({metric_cls.unit.value}) will be used for conversion."
        )

    display_unit = metric_cls.display_unit or metric_cls.unit

    if display_unit == metric_cls.unit:
        return result

    record = result.model_copy(deep=True)
    record.unit = display_unit.value

    for stat in STAT_KEYS:
        val = getattr(record, stat, None)
        if val is None:
            continue
        # Only convert numeric values
        if isinstance(val, int | float):
            try:
                new_value = metric_cls.unit.convert_to(display_unit, val)
            except MetricUnitError as e:
                _logger.warning(
                    f"Error converting {stat} for {result.tag} from {metric_cls.unit.value} to {display_unit.value}: {e}"
                )
                continue
            setattr(record, stat, new_value)
    return record


def convert_all_metrics_to_display_units(
    records: Iterable[MetricResult], registry: MetricRegistry
) -> dict[MetricTagT, MetricResult]:
    """Helper for exporters that want a tag->result mapping in display units."""
    out: dict[MetricTagT, MetricResult] = {}
    for r in records:
        out[r.tag] = to_display_unit(r, registry)
    return out
